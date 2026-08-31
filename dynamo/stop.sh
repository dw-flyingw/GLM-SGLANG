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
  docker compose --profile "*" down "$@"
else
  docker compose --profile cache --profile longctx down "$@"
fi
echo "Dynamo stack stopped."
