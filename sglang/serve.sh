#!/usr/bin/env bash
#
# Serve a GLM model via SGLang on 8x H200, aggregated TP=8.
#   PROFILE=flash    GLM-5.3-Flash  (primary; upstream image, pulled not built)
#   PROFILE=cache    GLM-5.2-FP8    (retained for rollback)
#   PROFILE=longctx  GLM-5.2-FP8    (1M attempt; starts but cannot serve)
# Brings up one SGLang worker (full model, all GPUs), serving the OpenAI API itself.
#
# Serves on host port 8000 (OpenAI-compatible). Set PORT= to change.
#
# Tunables (env): PORT, MAX_MODEL_LEN, MEM_FRACTION, TP_SIZE, PAGE_SIZE,
#   HF_CACHE, MODEL, SERVED_NAME, SGLANG_IMAGE, PROFILE, KV_SCRATCH_ROOT,
#   KV_SCRATCH_DIR, HICACHE_GB.  PROFILE=flash also reads SGLANG_FLASH_IMAGE,
#   FLASH_MODEL, FLASH_SERVED_NAME, MAMBA_RATIO, MAMBA_SSM_DTYPE,
#   HICACHE_WRITE_POLICY.  See docker-compose.yml.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${SGLANG_IMAGE:-glm52-sglang:0.5.13post1}"

PROFILE="${PROFILE:-cache}"
case "${PROFILE}" in
  cache|longctx|flash) ;;
  *) echo "Unknown PROFILE '${PROFILE}' (expected: cache, longctx, flash)" >&2; exit 1 ;;
esac

# PROFILE=flash serves a DIFFERENT MODEL (GLM-5.3-Flash) on a DIFFERENT IMAGE.
# glm5_next is in no public SGLang release, so the image is PULLED from
# upstream, never built from our Dockerfile -- see the build guard below.
if [ "${PROFILE}" = "flash" ]; then
  IMAGE="${SGLANG_FLASH_IMAGE:-lmsysorg/sglang:glm-5.3-flash}"
fi

# PROFILE=flash also runs a file-backed L3 tier, in its OWN directory: the
# storage keys carry no model identity, so sharing one directory between two
# different models would let 5.2 and 5.3 collide on the same cache keys.
if [ "${PROFILE}" = "cache" ] || [ "${PROFILE}" = "flash" ]; then
  KV_SCRATCH_ROOT="${KV_SCRATCH_ROOT:-/scratch/kvcache}"
  if [ "${PROFILE}" = "flash" ]; then
    KV_SCRATCH_DIR="${KV_SCRATCH_DIR:-/scratch/kvcache/glm53}"
  else
    KV_SCRATCH_DIR="${KV_SCRATCH_DIR:-/scratch/kvcache/glm52}"
  fi

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

# Crash-dump directory (both profiles). This is a WARNING, not a hard failure:
# a broken diagnostics path must never stop the model from serving.
#
# It is still worth warning loudly, because the failure is silent in exactly
# the way that matters. Docker auto-creates a missing bind-mount source as
# root-owned, the container runs as uid=1000, so the mount SUCCEEDS and the
# dump write fails later -- at the only moment you needed it to work.
DIAG_DIR="$(realpath -m "${DIAG_DIR:-/scratch/diag}")"
if [ ! -d "${DIAG_DIR}/coredumps" ]; then
  cat >&2 <<EOF
WARNING: crash-dump directory ${DIAG_DIR}/coredumps is missing.

The worker will serve normally, but CUDA coredumps will have nowhere to land,
and the archived worker logs that carry SGLang's py-spy watchdog dump cannot
be written either -- the same blind spot that made the 2026-09-03 exit-137
hang un-diagnosable.

  sudo mkdir -p ${DIAG_DIR}/coredumps ${DIAG_DIR}/logs
  sudo chown -R 1000:\$(id -g) ${DIAG_DIR}
  sudo chmod -R 2775 ${DIAG_DIR}

(Ownership matches KV_SCRATCH_ROOT's, and for the same reasons: owner 1000 so
the container can write the coredumps, group kept as your host group with g+w
so you can read and delete them AND so serve.sh/stop.sh -- which run as you,
not as the container -- can write the log archives, setgid so new files
inherit that group.)
EOF
fi
export DIAG_DIR

# Build the custom SGLang-0.5.13.post1 image if it's not present. --network=host is
# required: Docker's default bridge network can't reach pypi.org behind a proxy.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1 && [ "${PROFILE}" = "flash" ]; then
  cat >&2 <<EOF
Image ${IMAGE} not found, and PROFILE=flash must NOT build it.

GLM-5.3-Flash needs an SGLang build carrying glm5_next, which landed on main
2026-09-06 -- after the v0.5.19 release. There is no released sglang that can
serve it, so our Dockerfile (pip install sglang==0.5.13.post1) cannot produce
a working image. Pull upstream's purpose-built image instead:

  docker pull ${IMAGE}

EOF
  exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} not found; building (needs host networking + proxy for pip) ..."
  docker build --network=host \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}" \
    -t "${IMAGE}" -f Dockerfile .
fi

# The build above fires only when the image is ABSENT, which is not enough for
# the py-spy file capability the crash handler depends on. An image built
# before that Dockerfile stanza existed is present, passes every other check,
# and then silently returns "Permission Denied" on each watchdog dump -- which
# is the 2026-09-03 failure reproduced exactly. Worse, the cap is easy to
# "have" transiently: applying the xattr to a RUNNING container works and
# survives until the next `docker compose up`, at which point the fix vanishes
# with the container. So probe the IMAGE, never a live container.
#
# A warning, not a hard failure, for the usual reason: a broken diagnostics
# path must never stop the model from serving. But the model DOES serve
# blind until this is rebuilt.
#
# Only meaningful for an image that runs as a NON-root user. The file
# capability exists because the GLM-5.2 image runs as uid 1000 (dynamo), where
# cap_add: SYS_PTRACE lands only in the bounding set. An image running as root
# gets CAP_SYS_PTRACE in its EFFECTIVE set from cap_add directly, so py-spy
# needs no xattr. That is the GLM-5.3-Flash image (Config.User empty):
# verified 2026-09-12 that its schedulers run as uid 0 with CAP_SYS_PTRACE
# effective and that `py-spy dump` attaches and prints a stack. Its py-spy also
# lives at /opt/sglang/bin, not the /usr/local/bin probed below -- so without
# this guard the check fired a FALSE warning on every flash start, and its
# suggested fix (rebuild from our Dockerfile) yields an image that cannot load
# glm5_next at all.
IMAGE_USER="$(docker image inspect "${IMAGE}" --format '{{.Config.User}}' 2>/dev/null || true)"
case "${IMAGE_USER%%:*}" in
  ""|root|0) IMAGE_RUNS_AS_ROOT=1 ;;
  *)         IMAGE_RUNS_AS_ROOT=0 ;;
esac
if [ "${IMAGE_RUNS_AS_ROOT}" = 0 ] && ! docker run --rm "${IMAGE}" \
     python3 -c "import os; os.getxattr('/usr/local/bin/py-spy', b'security.capability')" \
     >/dev/null 2>&1; then
  cat >&2 <<EOF
WARNING: ${IMAGE} has no cap_sys_ptrace on /usr/local/bin/py-spy.

The worker will serve normally, but if the scheduler hangs, SGLang's watchdog
will fail every py-spy dump with "Permission Denied" and the stall will be as
un-diagnosable as it was on 2026-09-03. \`cap_add: SYS_PTRACE\` alone does NOT
cover this: it only puts CAP_SYS_PTRACE in the container's BOUNDING set, and
the worker runs as uid 1000, so CapPrm/CapEff stay 0 and the cap grants
nothing until the binary itself carries it.

Rebuild the image (needs host networking + proxy for pip):

  docker build --network=host -t ${IMAGE} -f $(pwd)/Dockerfile $(pwd)

Then recreate the worker so it picks the new image up:

  ./stop.sh && ./serve.sh
EOF
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
if [ "${PROFILE}" = "cache" ] || [ "${PROFILE}" = "flash" ]; then
  if ! docker run --rm -u 1000:0 \
       -v "${KV_SCRATCH_ROOT}:${KV_SCRATCH_ROOT}" \
       "${IMAGE}" bash -c "touch '${KV_SCRATCH_DIR}/.probe' && rm -f '${KV_SCRATCH_DIR}/.probe'" \
       >/dev/null 2>&1; then
    cat >&2 <<EOF
PROFILE=${PROFILE} needs ${KV_SCRATCH_DIR} writable by the CONTAINER's user
(uid=1000 gid=0, the image's dynamo user), not just the host user running
this script. A container write probe to that directory failed.

  sudo chown -R 1000:\$(id -g) ${KV_SCRATCH_ROOT}
  sudo chmod -R 2775 ${KV_SCRATCH_ROOT}

Why: owner 1000 so the container can write; group kept as your host group
with g+w so the host-side reaper (sglang/kv_reaper.py, run from cron as the
host user) can still delete what the container creates; setgid (2775) so
new files/dirs the container creates inherit that group. (Targets
KV_SCRATCH_ROOT, the whole bind-mounted tree, not just KV_SCRATCH_DIR --
DIR is guaranteed to be inside ROOT by the prefix check above.)

Or set KV_SCRATCH_DIR/KV_SCRATCH_ROOT to a location writable by uid 1000.
EOF
    exit 1
  fi
fi

if [ "${PROFILE}" = "longctx" ]; then
  WORKER_SERVICE="worker-longctx"
elif [ "${PROFILE}" = "flash" ]; then
  WORKER_SERVICE="worker-flash"
else
  WORKER_SERVICE="worker"
fi

# LAST CHANCE TO SAVE THE PREVIOUS WORKER'S LOG. `up -d` recreates the worker
# whenever its config changed, and docker deletes a container's log along with
# the container. That log is the ONLY place SGLang's watchdog writes its
# per-rank py-spy dump, so recreating over a worker that died in the night
# destroys the sole evidence of why -- which is precisely how the 2026-09-03
# exit-137 hang ended up un-diagnosable. See archive_worker_log.sh.
PREV_WORKER="$(docker compose --profile "${PROFILE}" ps -aq "${WORKER_SERVICE}" \
                 2>/dev/null | head -n1 || true)"
if [ -n "${PREV_WORKER}" ]; then
  ./archive_worker_log.sh "${PREV_WORKER}" "${DIAG_DIR}/logs" || true
fi

# These used to be hardcoded to GLM-5.2, so PROFILE=flash announced the wrong
# model and the wrong weight size.
if [ "${PROFILE}" = "flash" ]; then
  MODEL_LABEL="GLM-5.3-Flash"
  LOAD_NOTE="~306 GB of weights; ~6 min measured on this node, warm or cold JIT cache"
else
  MODEL_LABEL="GLM-5.2-FP8"
  LOAD_NOTE="756 GB of weights"
fi
echo "Starting SGLang stack for ${MODEL_LABEL} [profile: ${PROFILE}] ..."
docker compose --profile "${PROFILE}" up -d

cat <<EOF

Up. The model load + MTP warmup takes several minutes (${LOAD_NOTE}).

  Follow worker startup:   docker compose -f $(pwd)/docker-compose.yml logs -f ${WORKER_SERVICE}
  Stop everything:         ./stop.sh

Once the worker finishes loading, test:
(:${PORT:-8000} does not accept connections until the engine is up, so an
answered request is now a true readiness signal -- the old Dynamo frontend
answered immediately with an empty model list.)
  curl http://localhost:${PORT:-8000}/v1/models
EOF
