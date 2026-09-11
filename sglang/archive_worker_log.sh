#!/usr/bin/env bash
#
# Copy a container's docker log somewhere durable, BEFORE the container is
# removed. Usage:
#
#   archive_worker_log.sh <container-name-or-id> <dest-dir>
#
# Why this exists
# ---------------
# When the SGLang worker hangs, its 300 s watchdog fires and dumps a py-spy
# stack for every rank. That dump goes to the worker's stdout/stderr and
# NOWHERE else -- there is no log file inside the container. Docker keeps it in
# the container's json-file log, and deletes that log along with the container.
#
# Both teardown paths in this repo do exactly that:
#
#   stop.sh   -> `docker compose down`  removes the containers
#   serve.sh  -> `docker compose up -d` recreates a worker whose config changed
#
# So the sequence that actually happened on 2026-09-03 was: worker hung at
# ~15:04, watchdog dumped every rank's stack at 15:09, worker died at 15:10 --
# and the 18:05 restart deleted the only copy of that output. The incident was
# left with no evidence of WHERE the scheduler was stuck, which is the whole
# question. (The dumps were empty that day anyway, because Yama ptrace_scope=1
# denied py-spy; `cap_add: SYS_PTRACE` in docker-compose.yml fixes that half.
# Both halves have to work, or the next hang is just as opaque.)
#
# Deliberately NOT `set -e`: a diagnostics helper must never abort the caller.
# Every failure below warns and exits 0, so serve.sh/stop.sh keep going. The
# only non-zero exit is usage error, which is a bug in the caller, not a
# runtime condition. Callers still append `|| true` as belt and braces.
set -uo pipefail

CONTAINER="${1:-}"
DEST="${2:-}"

if [ -z "${CONTAINER}" ] || [ -z "${DEST}" ]; then
  echo "usage: $(basename "$0") <container-name-or-id> <dest-dir>" >&2
  exit 2
fi

# `docker inspect` rather than `docker ps`, because the container we most want
# is a DEAD one -- by the time anyone runs stop.sh the worker has usually
# already exited 137.
CID="$(docker inspect --format '{{.Id}}' "${CONTAINER}" 2>/dev/null)" || CID=""
if [ -z "${CID}" ]; then
  # Not an error: there is no previous worker on a first-ever start.
  exit 0
fi

# Label the archive with the container's own name, not whatever the caller
# happened to pass. stop.sh resolves services to raw ids, and a 64-hex prefix
# would bury the one thing you scan the directory for -- which worker it was.
# Docker returns it slash-prefixed ("/dynamo-worker-1"); fall back to the id if
# it is ever empty.
CNAME="$(docker inspect --format '{{.Name}}' "${CID}" 2>/dev/null | sed 's#^/##')" || CNAME=""
[ -n "${CNAME}" ] || CNAME="${CID}"

if ! mkdir -p "${DEST}" 2>/dev/null; then
  echo "WARNING: cannot create ${DEST}; ${CONTAINER}'s log will be lost when" \
       "the container is removed." >&2
  exit 0
fi

RAW="$(mktemp "${DEST}/.archive.XXXXXX" 2>/dev/null)" || {
  echo "WARNING: ${DEST} is not writable; ${CONTAINER}'s log will be lost when" \
       "the container is removed." >&2
  exit 0
}
trap 'rm -f "${RAW}"' EXIT

# --timestamps prefixes each line with the host's RFC3339 clock. SGLang stamps
# its own lines, but docker's stamps are what let you line the worker up
# against /var/log/kern.log and the frontend's log -- which is how the
# 2026-09-03 timeline was reconstructed at all. 2>&1 because SGLang logs to
# stderr; an archive that kept only stdout would miss the watchdog dump.
if ! docker logs --timestamps "${CID}" >"${RAW}" 2>&1; then
  echo "WARNING: 'docker logs ${CONTAINER}' failed; no archive written." >&2
  exit 0
fi

# Check emptiness on the RAW capture: gzip of empty input is still ~20 bytes,
# so testing the compressed file would never look empty.
if [ ! -s "${RAW}" ]; then
  exit 0
fi

SAFE="$(printf '%s' "${CNAME}" | tr -c 'A-Za-z0-9._-' '_')"
OUT="${DEST}/${SAFE}-$(date -u +%Y%m%dT%H%M%SZ)-${CID:0:12}.log.gz"

# Compress to a .part in the same directory, then rename: an interrupted run
# leaves no half-written file that looks like a complete archive. The short
# container id in the name keeps two incidents in the same second apart.
if ! gzip -c "${RAW}" >"${OUT}.part" 2>/dev/null; then
  echo "WARNING: could not write ${OUT}; ${CONTAINER}'s log will be lost." >&2
  rm -f "${OUT}.part"
  exit 0
fi
mv -f "${OUT}.part" "${OUT}"

echo "Archived ${CONTAINER} log -> ${OUT}"
