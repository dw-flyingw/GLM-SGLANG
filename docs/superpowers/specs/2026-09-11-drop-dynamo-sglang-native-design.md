# Drop NVIDIA Dynamo: serve GLM-5.2-FP8 from SGLang directly

**Date:** 2026-09-11
**Status:** Approved, not yet implemented
**Scope:** `dynamo/` → `sglang/` (rename), `docker-compose.yml`, `serve.sh`, `stop.sh`,
`bench.sh`, `bench_stream.py`, `tests/conftest.py`, and the live docs. The Dockerfile
and every engine tunable are unchanged.

## Problem

The stack runs four containers — etcd, NATS, the Dynamo frontend, and one SGLang
worker — to serve one model on one node. Dynamo contributes nothing this deployment
uses, and the repo's own documentation already says so:

| Dynamo provides | Status in this deployment |
|---|---|
| OpenAI-compatible HTTP frontend | Duplicates `sglang.launch_server`, which serves the same API |
| etcd + NATS service discovery | Exists to connect multiple workers; there is one worker |
| KV-aware routing (`--router-mode kv`) | `dynamo/README.md`: *"only a win with ≥ 2 workers/replicas"* — unused |
| Disaggregated prefill/decode | `dynamo/README.md`: needs ≥ 2 nodes for this model — unused |

The cost of keeping it is three extra containers, two extra network services on the
critical path, a second volume, and a layer of indirection in every doc and comment
in the repo. The engine underneath was always SGLang; `Dockerfile` builds an SGLang
image, and every flag in the worker's `command:` is already a raw SGLang flag.

## Findings (verified against the running image, not assumed)

All checks below were run against `glm52-dynamo-sglang:0.5.13post1` — the image
currently serving — via `sglang.srt.server_args.ServerArgs.add_cli_args`.

### The two parsers map 1:1, and they are the only Dynamo-side feature in use

The worker passes `--dyn-tool-call-parser=glm47` and `--dyn-reasoning-parser=glm45`.
These are **frontend** flags: Dynamo, not SGLang, does that parsing today. They are
the single piece of request-path behavior that leaves with Dynamo.

Native SGLang 0.5.13.post1 offers exact equivalents:

- `--reasoning-parser` choices include `glm45`
- `--tool-call-parser` choices include `glm47`

Both verified present. `--dyn-tool-call-parser` is confirmed **absent** from SGLang's
parser, so the rename is required, not cosmetic — the old spelling would fail at startup.

### `--enable-cache-report` is mandatory, and its absence is silent

`bench_stream.py:134` reads `usage.prompt_tokens_details.cached_tokens`. That field is
the direct measurement behind the headline KV-tiering result in
`RESULTS-kv-tiering.md` (135,232 / 135,271 cached, 100%) — the comment at
`bench_stream.py:129` calls it *"worth far more than the timing numbers for verifying
the tiering actually works."*

The Dynamo frontend populates it unconditionally. **SGLang's native server populates it
only when `--enable-cache-report` is set.** The flag is verified present in the image and
is **not** in the current worker `command:` — it never needed to be.

Without it there is no error and no warning: the key is simply absent from `usage`,
`bench_stream.py` records `None`, and the cache-hit column goes blank. Cache
effectiveness would have to be inferred from TTFT instead of read directly. This is the
same silent-failure shape as `HiCacheFile` swallowing write errors, which `serve.sh`
already guards against with a container-side write probe.

**`--enable-cache-report` goes in both profiles' `command:`, with a comment stating that
it is load-bearing for the benchmark, not a debug convenience.**

### The rename orphans the running stack and the JIT cache

Compose derives its project name from the containing directory. Confirmed on the host:

```
containers:  dynamo-worker-1, dynamo-frontend-1, dynamo-nats-1, dynamo-etcd-1
volumes:     dynamo_dynamo-jit-cache, dynamo_etcd-data
```

After `git mv dynamo sglang`, the project name becomes `sglang`. Two consequences, both
bad and both silent:

1. `./stop.sh` run from `sglang/` matches no containers. The old worker keeps running,
   holding all 8 GPUs, and the new worker cannot start.
2. `dynamo_dynamo-jit-cache` is not found. The new stack mounts a fresh empty volume and
   pays the full DeepGEMM JIT precompile + CUDA-graph capture — the ~10–20 min first-boot
   cost that volume exists to avoid.

Fix: stop inheriting the project identity from the path.

```yaml
name: glm52-sglang    # top-level key; survives any future directory rename
```

This makes the identity explicit and stable, and the one-time migration below carries
the existing JIT cache across so nothing is recompiled.

### Readiness semantics improve

Today the frontend binds `:8000` immediately and `/v1/models` returns an empty list
until the worker registers via etcd — so a successful connection means nothing.
`sglang.launch_server` does not listen until the engine is loaded, so "port open" becomes
a true readiness signal.

### Newly reachable endpoints (loopback only)

The native server exposes operational endpoints the Dynamo frontend did not proxy:
`/health`, `/health_generate`, `/get_server_info`, `/flush_cache`. These bind to
`${HTTP_HOST:-127.0.0.1}` along with everything else, so the network exposure posture is
unchanged. `/flush_cache` is a genuine convenience for future eviction benchmarking,
which currently requires a worker restart.

## Architecture

```
BEFORE                                      AFTER
  etcd :2379       ─┐                         sglang.launch_server
  nats :4222       ─┼─ dynamo.frontend :8000    └─ :8000  (host net, 127.0.0.1)
  dynamo.sglang    ─┘
  4 containers, 2 volumes                     1 container, 1 volume
```

Unchanged: host networking, loopback-only bind, port 8000, the OpenAI API surface,
TP=8, both profiles (`cache` / `longctx`), and every memory, attention, hicache,
hisparse, and speculative-decoding flag.

## Changes

### `docker-compose.yml`

| Action | Detail |
|---|---|
| Add | top-level `name: glm52-sglang` |
| Delete | `etcd`, `nats`, `frontend` services |
| Delete | `x-dynamo-env` anchor (`ETCD_ENDPOINTS`, `NATS_SERVER`) and its three references |
| Delete | `depends_on: [etcd, nats]` from `x-worker-base` |
| Delete | `etcd-data` volume |
| Change | volume key `dynamo-jit-cache` → `jit-cache`, and the `volumes:` mount in both worker services that references it. With the pinned project name this resolves to `glm52-sglang_jit-cache` (see the runbook — the migration command must target this exact name) |
| Change | `entrypoint: ["python3", "-m", "dynamo.sglang"]` → `["python3", "-m", "sglang.launch_server"]` |
| Change | `--dyn-tool-call-parser=glm47` → `--tool-call-parser=glm47` (both profiles) |
| Change | `--dyn-reasoning-parser=glm45` → `--reasoning-parser=glm45` (both profiles) |
| Add | `--host=${HTTP_HOST:-127.0.0.1}` and `--port=${PORT:-8000}` (both profiles) |
| Add | `--enable-cache-report` (both profiles) |
| Keep | `cap_add: SYS_PTRACE`, `init: true`, the coredump env block, all ulimits, every bind mount (HF cache, KV scratch, diag), `restart: "no"`, both profile definitions |

`HTTP_HOST` and `PORT` keep their exact current names, defaults, and meaning, so the
"do not expose this on the network — front it with a reverse proxy" guidance moves over
verbatim. It simply now applies to the worker rather than the frontend.

### Rename

`git mv dynamo sglang` (preserves history), plus:

| From | To |
|---|---|
| `DYNAMO_IMAGE` | `SGLANG_IMAGE` |
| `glm52-dynamo-sglang:0.5.13post1` | `glm52-sglang:0.5.13post1` |
| `tests/conftest.py` `DYNAMO` constant | `SGLANG` |

### `serve.sh`

Unchanged in substance — the KV-scratch container write probe, the py-spy capability
probe, and the pre-teardown log archiving are all engine-level concerns that outlive
Dynamo. Edits:

- `IMAGE` default and env var renamed per the table above.
- Closing message: drop the frontend-logs line; "Once the worker registers" →
  "Once the worker finishes loading", since there is no registry to register with.

### `stop.sh`

Profile-wildcard teardown logic and log archiving unchanged. One correction: the
`--volumes` comment documents it as "keeps the etcd data volume". That volume no longer
exists, and `--volumes` now destroys the **JIT kernel cache** instead — turning a
near-free flag into a 10–20 min penalty on next start. The comment and the header
usage line must say so.

### `bench.sh` / `bench_stream.py`

Env var rename. The `bench_stream.py:129` comment is currently conditional — *"If the
Dynamo frontend populates it"* — and becomes a statement of fact with a pointer to the
flag that guarantees it:

> the server populates this because `--enable-cache-report` is set in
> `docker-compose.yml`; without that flag the field is silently absent

### Documentation

**Rewritten** (describe the live stack, must be accurate):
`README.md`, `sglang/README.md`, `CONTEXT_WINDOW.md`, and the `docker-compose.yml`
header comments.

Sections that survive unchanged because they remain true and remain the reason for the
current design: "Why SGLang, not vLLM", "Why aggregated only (no disaggregation on one
node)", the EP+DP-attention regression table, and the entire "Tiered KV cache" section.

One rewrite of substance: under *Performance levers*, "KV-aware routing — add
`--router-mode kv` to the frontend" describes a frontend that no longer exists. It
becomes a note that KV-aware routing would require reintroducing a router, and needed
≥ 2 workers to pay off regardless.

**Preserved as-is** (dated records of measurements taken on the Dynamo stack; rewriting
them would falsify the record): `RESULTS-kv-tiering.md`,
`docs/superpowers/plans/2026-08-31-kv-cache-tiering.md`,
`docs/superpowers/specs/2026-08-31-scratch-kv-cache-tiering-design.md`.

`RESULTS-kv-tiering.md` gets exactly one added header line noting the numbers were
measured under Dynamo + SGLang 0.5.13.post1, and that the engine and every engine flag
are unchanged by this migration — so the measurements carry over.

## Cutover runbook

Not executed as part of this work. The live stack keeps serving until the operator runs
this. Written into `sglang/README.md`.

```bash
# 1. Tear down the old stack (archives the worker log first).
cd dynamo && ./stop.sh

# 2. One-time: carry the JIT kernel cache across the project rename,
#    avoiding a ~10-20 min DeepGEMM recompile on first start.
docker volume create glm52-sglang_jit-cache
docker run --rm \
  -v dynamo_dynamo-jit-cache:/from \
  -v glm52-sglang_jit-cache:/to \
  alpine sh -c 'cp -a /from/. /to/'

# 3. One-time: retag the existing image so serve.sh does not rebuild it
#    (the rebuild needs host networking + a pip proxy).
docker tag glm52-dynamo-sglang:0.5.13post1 glm52-sglang:0.5.13post1

# 4. Start.
cd ../sglang && ./serve.sh
docker compose logs -f worker
```

Step 3 matters: `serve.sh` builds the image only when it is absent, so a renamed default
tag would trigger a full rebuild through the proxy on an otherwise offline host.

## Verification

### Automated

Both existing suites must pass unchanged in behavior:
`pytest tests/` — `test_bench_stream.py`, `test_kv_reaper.py`, `test_archive_worker_log.py`.
The only test edit is the `conftest.py` path constant.

Static: `docker compose --profile cache config` and `--profile longctx config` must both
render without error, and `docker compose --profile "*" config --services` must list both
workers and nothing else.

### Post-cutover (operator, per the runbook)

| # | Check | Passes when |
|---|---|---|
| 1 | `curl localhost:8000/v1/models` | returns `glm-5.2-fp8` |
| 2 | Chat completion, streamed | `reasoning_content` deltas arrive — proves `--reasoning-parser glm45` survived the move off the frontend |
| 3 | `./bench_stream.py --concurrency 1 --num 16` run **twice** | second run reports **non-zero `cached_tokens`** |
| 4 | `docker compose ps` | exactly one container |

Check 3 is the specific regression test for the `--enable-cache-report` finding and is
the one that must not be skipped — it is the only check that fails silently if the flag
is dropped.

## Risks and rollback

| Risk | Mitigation |
|---|---|
| `--enable-cache-report` omitted or later removed | Verification check 3; a load-bearing comment at the flag |
| Old stack left running, holding 8 GPUs | Runbook step 1 runs `./stop.sh` from `dynamo/` **before** the rename takes effect in the operator's shell |
| Forced DeepGEMM recompile | Runbook step 2 migrates the volume; worst case is a slow first boot, not a failure |
| Accidental image rebuild on an offline host | Runbook step 3 retags |
| Something unforeseen in the native server | Rollback is `git revert` + `./serve.sh` from the restored `dynamo/`. The old image is untouched on disk under its original tag, and no engine flag changed, so the old stack comes back bit-identical. |

Rollback cost is one worker restart. No data migration is involved: the `/scratch` L3 KV
tier is engine-managed and format-identical across the change, and the JIT cache is
copied rather than moved, so the original volume survives intact.

## Explicitly out of scope

- **The GLM-5.3-Flash cookbook link** added to `README.md` in the working tree. It
  references a different model and a different config recipe (`hicache=l3`,
  `strategy=low-latency`, `dcp=off`). Evaluating that model is separate work with its own
  measurements; the line is removed from this change rather than committed half-considered.
- **Any engine tunable.** No change to `--mem-fraction-static`, `--context-length`,
  hicache policy, HiSparse config, or the speculative-decoding flags. This migration must
  be measurable as a no-op on serving behavior; changing an engine flag in the same commit
  would make the two indistinguishable.
- **The base image.** Stays `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13`
  with the SGLang 0.5.13.post1 bump. It is already an SGLang image and runs
  `launch_server` directly; switching to `lmsysorg/sglang` would mean a fresh torch /
  sgl-kernel / flashinfer pin set, a re-verified py-spy capability stanza, and a ~20 GB
  proxied pull — a second variable, for no gain here.
- **Profile B's two SGLang HiSparse bugs.** Unaffected by this change and still upstream.
- **`./chat.py`**, advertised in `README.md` but absent from the repo. Pre-existing
  discrepancy, unrelated to this change; noted so it is not mistaken for collateral damage.
