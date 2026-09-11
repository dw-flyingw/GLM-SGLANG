#!/usr/bin/env bash
# Stop and remove the SGLang stack.
# Pass --volumes to also drop the JIT kernel cache volume (see the warning below).
set -euo pipefail
cd "$(dirname "$0")"

# --volumes used to drop only the etcd data volume, which cost nothing to
# rebuild. That volume is gone; the JIT kernel cache is now the ONLY volume,
# so the same flag now throws away every DeepGEMM/FlashInfer/Triton kernel the
# worker has compiled and buys a ~10-20 min precompile on the next start.
# Warn, but do not prompt -- this script is called from other scripts.
for arg in "$@"; do
  case "${arg}" in
    --volumes|-v)
      echo "WARNING: --volumes will delete the JIT kernel cache volume." >&2
      echo "         The next ./serve.sh will pay a ~10-20 min DeepGEMM precompile." >&2
      ;;
  esac
done
# `--profile "*"` (verified working: `docker compose --profile "*" config --services`
# lists all profile-gated services) reaches ANY worker profile, present or future,
# so a new profile added later can't escape teardown and orphan a worker holding
# all 8 GPUs. Falls back to enumerating known profiles if the wildcard is ever
# unsupported by the installed Compose version.
if docker compose --profile "*" config --services >/dev/null 2>&1; then
  PROFILE_ARGS=(--profile "*")
else
  PROFILE_ARGS=(--profile cache --profile longctx)
fi

# LAST CHANCE TO SAVE THE WORKER'S LOG -- and the likelier of the two, since
# `down` removes the containers unconditionally, where serve.sh only recreates
# on a config change. Docker deletes a container's log along with the
# container, and that log is the ONLY place SGLang's watchdog writes its
# per-rank py-spy dump. Tearing down a worker that died in a hang therefore
# destroys the sole evidence of why, which is exactly how the 2026-09-03
# exit-137 hang ended up un-diagnosable. See archive_worker_log.sh.
#
# Both worker services are tried because stop.sh tears down every profile; only
# one of them normally exists. DIAG_DIR matches serve.sh's default.
DIAG_DIR="$(realpath -m "${DIAG_DIR:-/scratch/diag}")"
for svc in worker worker-longctx; do
  cid="$(docker compose "${PROFILE_ARGS[@]}" ps -aq "${svc}" 2>/dev/null | head -n1 || true)"
  if [ -n "${cid}" ]; then
    ./archive_worker_log.sh "${cid}" "${DIAG_DIR}/logs" || true
  fi
done

docker compose "${PROFILE_ARGS[@]}" down "$@"
echo "SGLang stack stopped."
