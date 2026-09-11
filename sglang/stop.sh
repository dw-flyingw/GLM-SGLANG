#!/usr/bin/env bash
# Stop and remove the Dynamo (SGLang) stack. Keeps the etcd data volume.
# Pass --volumes to also drop the etcd volume.
set -euo pipefail
cd "$(dirname "$0")"
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
echo "Dynamo stack stopped."
