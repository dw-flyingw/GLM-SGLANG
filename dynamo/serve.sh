#!/usr/bin/env bash
#
# Serve GLM-5.2-FP8 via NVIDIA Dynamo (SGLang backend) on 8x H200, aggregated TP=8.
# Brings up etcd + NATS + Dynamo frontend + one SGLang worker (full model, all GPUs).
#
# Serves on host port 8000 (OpenAI-compatible). Set PORT= to change.
#
# Tunables (env): PORT, MAX_MODEL_LEN, MEM_FRACTION, TP_SIZE, PAGE_SIZE,
#   HF_CACHE, MODEL, SERVED_NAME, DYNAMO_IMAGE, PROFILE, KV_SCRATCH_ROOT,
#   KV_SCRATCH_DIR, HICACHE_GB.  See docker-compose.yml.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${DYNAMO_IMAGE:-glm52-dynamo-sglang:0.5.13post1}"

PROFILE="${PROFILE:-cache}"
case "${PROFILE}" in
  cache|longctx) ;;
  *) echo "Unknown PROFILE '${PROFILE}' (expected: cache, longctx)" >&2; exit 1 ;;
esac

if [ "${PROFILE}" = "cache" ]; then
  KV_SCRATCH_ROOT="${KV_SCRATCH_ROOT:-/scratch/kvcache}"
  KV_SCRATCH_DIR="${KV_SCRATCH_DIR:-/scratch/kvcache/glm52}"

  # Normalize both paths before the prefix check below. realpath -m collapses
  # '..' traversal and does not require the path to exist, so
  # KV_SCRATCH_DIR=/scratch/kvcache/../../tmp/evil can no longer pass the
  # check by textually starting with KV_SCRATCH_ROOT while actually
  # resolving outside it.
  KV_SCRATCH_ROOT="$(realpath -m "${KV_SCRATCH_ROOT}")"
  KV_SCRATCH_DIR="$(realpath -m "${KV_SCRATCH_DIR}")"

  # Only KV_SCRATCH_ROOT is bind-mounted into the container (see docker-compose.yml).
  # If KV_SCRATCH_DIR is overridden to somewhere outside it, the directory is absent
  # inside the container and SGLang's os.makedirs() silently writes an unbounded
  # cache into the container's ephemeral overlay on the 438 GB root filesystem --
  # exactly what the /scratch bind mount exists to prevent.
  case "${KV_SCRATCH_DIR}" in
    "${KV_SCRATCH_ROOT}"/*|"${KV_SCRATCH_ROOT}")
      ;;
    *)
      echo "KV_SCRATCH_DIR (${KV_SCRATCH_DIR}) must be under KV_SCRATCH_ROOT (${KV_SCRATCH_ROOT})," \
           "or it won't be visible inside the container (only KV_SCRATCH_ROOT is bind-mounted)." >&2
      exit 1
      ;;
  esac

  # Cheap existence check first so a missing directory gives a clear message
  # without spawning a container.
  if [ ! -d "${KV_SCRATCH_DIR}" ]; then
    cat >&2 <<EOF
PROFILE=cache needs a KV cache directory at ${KV_SCRATCH_DIR}

  sudo mkdir -p ${KV_SCRATCH_DIR}
  sudo chown -R 1000:\$(id -g) ${KV_SCRATCH_ROOT}
  sudo chmod -R 2775 ${KV_SCRATCH_ROOT}

(chown/chmod target KV_SCRATCH_ROOT, not KV_SCRATCH_DIR: it is the whole
bind-mounted tree, and KV_SCRATCH_DIR is guaranteed to be inside it -- see
the prefix check above -- so fixing ROOT's ownership covers DIR too.)

Or set KV_SCRATCH_DIR/KV_SCRATCH_ROOT to a writable location.
EOF
    exit 1
  fi

  export KV_SCRATCH_ROOT KV_SCRATCH_DIR
fi

# Build the custom SGLang-0.5.13.post1 image if it's not present. --network=host is
# required: Docker's default bridge network can't reach pypi.org behind a proxy.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} not found; building (needs host networking + proxy for pip) ..."
  docker build --network=host \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}" \
    -t "${IMAGE}" -f Dockerfile .
fi

# Profile A writes its L3 KV tier to /scratch, from INSIDE THE CONTAINER, which
# runs as uid=1000(dynamo) gid=0(root) (the image's Config.User=dynamo -- verify
# with `docker image inspect "${IMAGE}" --format '{{.Config.User}}'` plus
# `docker run --rm "${IMAGE}" id dynamo` if the image ever changes). That is NOT
# the host user invoking this script. A host-side `test -w` check on
# KV_SCRATCH_DIR therefore tells you nothing about whether the container can
# write there -- and SGLang's HiCacheFile SWALLOWS write failures (logs and
# continues), so a permissions mismatch here means the worker starts clean and
# the entire L3 cache tier is silently dead. Probe as the container actually
# will: run a throwaway container as the same uid:gid and try a real write.
# This runs after the image build above (it needs the image present) but
# before `docker compose up`, and a probe failure must NOT trigger a build.
if [ "${PROFILE}" = "cache" ]; then
  if ! docker run --rm -u 1000:0 \
       -v "${KV_SCRATCH_ROOT}:${KV_SCRATCH_ROOT}" \
       "${IMAGE}" bash -c "touch '${KV_SCRATCH_DIR}/.probe' && rm -f '${KV_SCRATCH_DIR}/.probe'" \
       >/dev/null 2>&1; then
    cat >&2 <<EOF
PROFILE=cache needs ${KV_SCRATCH_DIR} writable by the CONTAINER's user
(uid=1000 gid=0, the image's dynamo user), not just the host user running
this script. A container write probe to that directory failed.

  sudo chown -R 1000:\$(id -g) ${KV_SCRATCH_ROOT}
  sudo chmod -R 2775 ${KV_SCRATCH_ROOT}

Why: owner 1000 so the container can write; group kept as your host group
with g+w so the host-side reaper (dynamo/kv_reaper.py, run from cron as the
host user) can still delete what the container creates; setgid (2775) so
new files/dirs the container creates inherit that group. (Targets
KV_SCRATCH_ROOT, the whole bind-mounted tree, not just KV_SCRATCH_DIR --
DIR is guaranteed to be inside ROOT by the prefix check above.)

Or set KV_SCRATCH_DIR/KV_SCRATCH_ROOT to a location writable by uid 1000.
EOF
    exit 1
  fi
fi

echo "Starting Dynamo (SGLang) stack for GLM-5.2-FP8 [profile: ${PROFILE}] ..."
docker compose --profile "${PROFILE}" up -d

if [ "${PROFILE}" = "longctx" ]; then
  WORKER_SERVICE="worker-longctx"
else
  WORKER_SERVICE="worker"
fi

cat <<EOF

Up. The model load + NSA/MTP warmup takes several minutes (756 GB of weights).

  Follow worker startup:   docker compose -f $(pwd)/docker-compose.yml logs -f ${WORKER_SERVICE}
  Frontend logs:           docker compose -f $(pwd)/docker-compose.yml logs -f frontend
  Stop everything:         ./stop.sh

Once the worker registers, test:
  curl http://localhost:${PORT:-8000}/v1/models
EOF
