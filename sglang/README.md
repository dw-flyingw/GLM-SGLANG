# GLM-5.2-FP8 on SGLang (8× H200)

Serves `zai-org/GLM-5.2-FP8` on a single node with **SGLang**, which serves the
OpenAI-compatible API itself on host port `:8000` from `sglang.launch_server`.
One container, no orchestration layer.

This used to run under NVIDIA Dynamo (etcd + NATS + a frontend process in front of
the same worker). Dynamo was removed on 2026-09-11: its frontend duplicated
`sglang.launch_server`, and the three things it added over that — multi-worker
discovery, KV-aware routing, and disaggregated prefill/decode — are all unreachable
on a single node holding one ~756 GB copy of the model. The engine, the image, and
every engine flag are unchanged by the removal, so every measurement in
[`RESULTS-kv-tiering.md`](RESULTS-kv-tiering.md) carries over. See
[Cutover from the Dynamo stack](#cutover-from-the-dynamo-stack) for the one-time
migration.

## Why SGLang, not vLLM

GLM-5.2 is `GlmMoeDsaForCausalLM`: MoE + MLA + **DeepSeek-style Sparse Attention (DSA)**
with `head_size=704`. The vLLM path needs **vLLM ≥ 0.23.0** for that.

Verified (2026-06-27): **no published Dynamo `vllm-runtime` image, nor the
`ai-dynamo` PyPI wheel, ships vLLM ≥ 0.23.0.**
`1.2.1`→0.20.1, `1.3.0-dev.1`→0.22.0 (its sparse-MLA backends cap at `head_size=576`),
`kimi-k2.6-dev`→0.21.0. So we use **SGLang** — NVIDIA's own GLM-5 recipe
(`ai-dynamo/dynamo/recipes/glm-5-nvfp4`) is SGLang too. Dynamo itself is no longer
used; the image it gave us is — see the `Dockerfile`.

Our custom image (see `Dockerfile`) bundles **SGLang 0.5.13.post1**, which registers
`GlmMoeDsaForCausalLM` and **auto-selects the DSA attention backend** for this arch: on
Hopper + fp8 KV it picks `dsa`/`flashmla_kv` (prefill+decode), with the DSA indexer
(`sgl-kernel` topk) + MTP support. We pass **no** `--attention-backend` and let SGLang
choose — that is the SGLang-recommended path and beats the legacy
`nsa` backend we used to force (~+8% system tok/s at conc 32).

## Why aggregated only (no disaggregation on one node)

The weights are **~756 GB**. A full copy needs ~6 H200s (143 GB each). Disaggregated
serving runs **separate** prefill and decode workers, each holding a **full** model copy —
that's ~12 GPUs, impossible on a single 8-GPU node (TP4 = 189 GB/GPU > 143 GB). NVIDIA's
own GLM-5 recipe is therefore 5 nodes / 20 GPUs. **Disaggregation here needs ≥ 2 nodes.**

On one node the only fit is **aggregated, tensor-parallel over all 8 GPUs** — what this
stack does. This is also why Dynamo was removed: everything it added over plain SGLang
(KV-aware routing, request migration, a path to multi-node disaggregation) only starts
paying at ≥ 2 workers or ≥ 2 nodes, which one 756 GB model copy on 8 GPUs cannot reach.
If this ever grows to a second node, reintroducing an orchestration layer is the right
move — and none of the engine configuration below would need to change to do it.

## Layout

- `docker-compose.yml` — one SGLang worker (TP=8), serving the OpenAI API itself. Host
  networking; tunables via `${...}` env (see below).
- `serve.sh` / `stop.sh` — bring the stack up / down.
- `archive_worker_log.sh` — copy a worker's docker log somewhere durable before
  the container is removed. Called automatically by both of the above.
- `bench.sh` — load-test any OpenAI endpoint with aiperf/genai-perf.

## Usage

```bash
cd sglang
./serve.sh                       # start (detached)
docker compose logs -f worker    # watch model load (several minutes, 756 GB)
curl http://localhost:8000/v1/models
./stop.sh
```

`:8000` does not accept connections until the engine has finished loading, so an
answered request is a true readiness signal. (Under Dynamo the frontend answered
immediately and `/v1/models` returned an empty list until the worker registered.)

### Operational endpoints

The engine serves the API itself, which exposes endpoints the Dynamo frontend did not
proxy — all on `${HTTP_HOST:-127.0.0.1}` like everything else, so the exposure posture
is unchanged:

| Endpoint | Use |
|---|---|
| `/health`, `/health_generate` | liveness / can-it-actually-decode |
| `/get_server_info` | the engine's effective config, without grepping the boot log |
| `/flush_cache` | drop the prefix cache between benchmark runs — this replaces a full worker restart, which is how eviction tests were staged before |

Tunables (env): `PORT`, `MAX_MODEL_LEN` (→ sglang `--context-length`, default 524288),
`MEM_FRACTION` (default 0.85), `TP_SIZE` (default 8), `PAGE_SIZE` (default 64, DSA),
`MAX_RUNNING` (→ `--max-running-requests`, default 128 — the concurrency ceiling;
spec decoding would otherwise auto-cap it to 48),
`HF_CACHE` (default `/root/.cache/huggingface`), `MODEL`, `SERVED_NAME`, `SGLANG_IMAGE`,
`HTTP_HOST` (default `127.0.0.1` — the worker binds the host interface directly under
`network_mode: host`, so this is loopback-only unless you change it).
MTP speculative decoding is on by default; tune via `SPEC_ALGO` (default `EAGLE`),
`SPEC_NUM_STEPS` (2), `SPEC_EAGLE_TOPK` (1), `SPEC_NUM_DRAFT` (3), or disable by
editing the worker `command:` in `docker-compose.yml`.

## Cutover from the Dynamo stack

One-time, run by hand. The Dynamo stack keeps serving until you run this.

By this point `dynamo/` no longer exists — the rename has landed — so the old stack
cannot be torn down by its own `stop.sh`. Reach it by project name instead:
`docker compose -p dynamo ps` resolves a running project from container labels with
no compose file present (`config` does not, but `ps` and `down` do).

```bash
# 1. Archive the old worker's log BEFORE teardown -- docker deletes a
#    container's log with the container, and that log is the only place
#    SGLang's watchdog writes its per-rank py-spy dump.
./sglang/archive_worker_log.sh dynamo-worker-1 /scratch/diag/logs

# 2. Tear down the old stack by project name.
docker compose -p dynamo down

# 3. One-time: carry the JIT kernel cache across the project rename,
#    avoiding a ~10-20 min DeepGEMM recompile on first start.
docker volume create glm52-sglang_jit-cache
docker run --rm \
  -v dynamo_dynamo-jit-cache:/from \
  -v glm52-sglang_jit-cache:/to \
  alpine sh -c 'cp -a /from/. /to/'

# 4. One-time: retag the existing image so serve.sh does not rebuild it
#    (the rebuild needs host networking + a pip proxy).
docker tag glm52-dynamo-sglang:0.5.13post1 glm52-sglang:0.5.13post1

# 5. Start.
cd sglang && ./serve.sh
docker compose logs -f worker
```

Step 4 matters: `serve.sh` builds the image only when it is absent, so the renamed
default tag would otherwise trigger a full rebuild through the proxy on an
otherwise-offline host.

Then verify — the third check is the one that must not be skipped, because it is the
only one that fails silently:

```bash
curl localhost:8000/v1/models              # returns glm-5.2-fp8
./bench_stream.py --concurrency 1 --num 4  # reasoning_content streams
./bench_stream.py --concurrency 1 --num 4  # 2nd run: cached_tokens > 0
```

Rollback is `git revert` plus one restart. The old image survives under its original
tag and `dynamo_dynamo-jit-cache` is copied rather than moved, so both are intact.

## Serving config (mirrors the model card / NVIDIA recipe)

- `--tp-size 8`, `--kv-cache-dtype fp8_e4m3`, `--page-size 64`
- **DSA attention backend auto-selected** — no `--attention-backend`; SGLang picks
  `dsa`/`flashmla_kv` (Hopper + fp8 KV). Verified ~+8% system tok/s at conc 32 vs the
  legacy `nsa` backend we used to force.
- `--tool-call-parser glm47`, `--reasoning-parser glm45` (native SGLang parsers; these
  were `--dyn-*` under Dynamo, where the frontend did the parsing)
- **`--enable-cache-report`** — makes the server populate
  `usage.prompt_tokens_details.cached_tokens`. Not optional: `bench_stream.py` reads it
  as a direct count of prefix-cache hits, and without the flag the field is *silently*
  absent. The Dynamo frontend used to populate it for free.
- **`--context-length 524288` (512K), served under `PROFILE=cache` with a tiered
  prefix cache.** 1M is the model's max, but a single node can't hold a 1M-token
  KV cache next to the ~94 GB/GPU weights: measured 2026-08-31, the decode pool
  under Profile A is `max_total_num_tokens=540928`, so 512K is the largest length
  a full request can actually be served at with full prefix-caching fidelity.
  `PROFILE=longctx` is the single-node attempt at the full 1M context via
  SGLang HiSparse instead of a bigger GPU pool — see "Tiered KV cache" below for
  why it doesn't currently start on this hardware, and why true 1M still needs
  ≥ 2 nodes (same reason as disaggregation).
- **MTP / EAGLE speculative decoding (enabled).** GLM-5.2 ships 1 MTP layer; EAGLE
  drives it from the main checkpoint (no separate draft model):
  `--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`.
  Verified to compose with the `nsa` backend on H200 (Hopper) — worker logs show
  `accept len ≈ 2.2–2.8` and ~2× single-stream decode (see Benchmark).

### Performance levers

- **MoE EP + DP-attention** (`--ep-size 8 --dp-size 8 --enable-dp-attention`) — SGLang's
  documented 8×H200 "throughput" config. **Benchmarked (2026-06-28) and it
  regressed on every axis**, so it is *not* used:
  | conc | metric | TP=8 (this stack) | EP+DP-attention |
  |---|---|---|---|
  | 1  | decode tok/s/req | **111.6** | 57.8 |
  | 32 | system tok/s     | **1580.4** | 910.1 |
  | 32 | TTFT p99         | 959 ms | (64-conc) 8447 ms |

  DP-attention *replicates* the MLA/dense weights on every DP rank (94→102 GB/GPU),
  halving the KV pool (540k→292k tokens, capping single-request context at ~292K), and
  only pays off at hundreds of concurrent requests — but one node caps at 48 max-running
  (6/rank). It's the right config only on ≥ 2 nodes / very high concurrency.
- **KV-aware routing** — would require reintroducing a router process in front of the
  worker (this is one of the things Dynamo provided). Moot either way at this scale: it
  only pays off with ≥ 2 workers/replicas, and one node holds exactly one copy of a
  756 GB model.

### Tiered KV cache

`PROFILE=cache` (the default) adds a hierarchical prefix cache below the GPU
radix cache: GPU pool → host RAM (L2) → `/scratch` NVMe (L3). All figures
below are measured 2026-08-31 against this deployment (image tag
`glm52-dynamo-sglang:0.5.13post1`), recorded in full in
[`RESULTS-kv-tiering.md`](RESULTS-kv-tiering.md).

**Tier sizes:**

| Tier | Size | Notes |
|---|---|---|
| GPU pool | **540,928 tokens** (`max_total_num_tokens`), measured | shared radix cache, evicted under pressure |
| L2 host RAM | **96 GB/rank × 8 TP ranks** (~972 GB RAM in use, `free -g` measured) | MLA replicates the pool identically across ranks — 8 real allocations of the same content, not 8 distinct shards |
| L3 `/scratch` | deduplicated across TP ranks | the `file` storage backend's keys carry **no `tp_rank` suffix** for MLA models — verified by direct inspection of the files under `/scratch/kvcache/glm52` (`<hash>_glm-5.2-fp8.bin`, `<hash>.indexer_glm-5.2-fp8.bin`, `<hash>.draft_glm-5.2-fp8.bin`, 3 files per page-key) — so disk holds **one** copy per unique page, unlike L2's 8x replication |

**Headline result — eviction survival.** A 131,072-token shared prefix was
primed, confirmed warm, then evicted by flooding 655,360 tokens (5 distinct
131K prefixes) through the 540,928-token GPU pool. Re-requesting the evicted
prefix:

| State | TTFT | Cached tokens |
|---|---:|---:|
| Before tiering (plain GPU radix cache only) | **16,769 ms** | 0 / 135,271 (0%) |
| After tiering (`write_through`, same eviction test) | **1,230 ms** | 135,232 / 135,271 (100%) |
| Restart-persistence test (**separate run**, fresh 131K-token prefix, seed 4242, no flood): after tiering **and a worker restart** (L2 host RAM freshly reallocated and confirmed empty at boot) | **5,121 ms** | 135,232 / 135,257 (100%) |

The third row is the proof `/scratch` — not host RAM — served the prefix:
L2 is wiped on every worker restart (8 fresh `Allocating 96.00 GB host
memory` lines, all timestamped at the new boot), so the only place those
135,232 cached tokens could have come from after a restart is L3 on disk.
TTFT after restart (5,121 ms) sits between a cold refill (~16.7-20.5 s) and
an in-RAM hit (~650-750 ms) — the expected shape of a disk read followed by
a copy back into GPU/L2, and categorically faster than recomputing the
prefix from scratch.

**The honest cost.** Tiering is not free under concurrent load:

| Metric | No hicache (baseline) | `write_through` (default) | Δ |
|---|---:|---:|---:|
| conc-32 system throughput | 2087.1 tok/s | **1905.2 tok/s** | **-8.7%** |
| conc-1 decode (per-req) | 150.6 tok/s | **149.8 tok/s** | -0.5% (unaffected) |

`write_through_selective` (write only hotter pages, instead of every page)
was tested as the prescribed mitigation for the conc-32 regression. **It did
not recover it** — throughput came back *lower* (1853.1 tok/s), and
`/scratch` write volume was essentially unchanged (47 GB / 38,070 files vs.
48 GB / 38,106 files for the identical benchmark sequence), meaning
selectivity did not reduce what got written for this traffic pattern.
`write_through` therefore remains the default. The eviction-recovery result
is identical under either policy (100% cached, TTFT statistically
indistinguishable), so the regression is a real, accepted trade for the
eviction-survival benefit — it is not something write-policy tuning alone
closes.

**Tunables (env):**

| Var | Default | Purpose |
|---|---|---|
| `PROFILE` | `cache` | `cache` = Profile A (tiered KV cache, 512K); `longctx` = Profile B (HiSparse, 1M target — starts but crashes on the first request, see below) |
| `HICACHE_GB` | `96` | L2 host RAM pool size, **per TP rank** (see tier table above for why this multiplies by 8) |
| `HICACHE_WRITE_POLICY` | `write_through` | `write_through` (every page, default) vs `write_through_selective` (hotter pages only — measured not to help, see above) |
| `KV_SCRATCH_ROOT` | `/scratch/kvcache` | host directory bind-mounted into the container (the whole tree) |
| `KV_SCRATCH_DIR` | `/scratch/kvcache/glm52` | this model's subdirectory under `KV_SCRATCH_ROOT`; must stay under it or it isn't visible in the container |
| `HISPARSE_DEVICE_BUFFER` | `4096` | Profile B only — HiSparse device-side buffer size |
| `HISPARSE_RATIO` | `2` | Profile B only — HiSparse host-to-device ratio |
| `MAX_MODEL_LEN_LONG` | `1048576` | Profile B only — target context length (the full 1M); lowering it does **not** free GPU memory (attempt 4, see below) |
| `MEM_FRACTION_LONG` | `0.82` | Profile B only — the only value measured to start; `0.88`/`0.86` both OOM in CUDA-graph capture (see below). The only knob that actually resizes the KV pool |
| `CUDA_GRAPH_MAX_BS_LONG` | `128` | Profile B only — caps CUDA-graph capture at the concurrency ceiling instead of SGLang's default 512; reduced but did not close the OOM (see below) |

**Profile B (HiSparse, 1M context) — starts at 1M, but cannot serve a
request.** Six attempts. Attempts 1–4 were CUDA OOM during CUDA-graph capture;
`mem-fraction 0.82` cleared that outright, and what remains are two SGLang
defects:

1. `--mem-fraction-static=0.88` (default): OOM in generic graph capture,
   488.19 MiB free at failure (GPU 6).
2. `MEM_FRACTION_LONG=0.86`: same failure mode, but *more* free memory at
   failure (1.24 GiB, GPU 1) — freeing more static memory did not translate
   into proportionally more graph-capture headroom.
3. `0.88` with `CUDA_GRAPH_MAX_BS_LONG=128`: got past the generic
   graph-capture stage (host pools allocated, DeepGEMM warmup complete), then
   OOM'd one call deeper, inside the **HiSparse indexer's own graph capture**
   (`dsa_indexer.py::_get_topk_paged`), needing 1.50 GiB with only 488 MiB
   free and the GPU fully pinned (139.20/139.80 GiB in use).
4. `MAX_MODEL_LEN_LONG=786432` (2026-09-01) — attempt 3's config with **only**
   the context length changed, 768K instead of 1M: failed *earlier* in the
   sequence, back in generic graph capture (`cuda_graph_runner.py:628`), with
   *less* memory free (48.06 MiB, GPU 1) — `Tried to allocate 60.00 MiB`,
   135.16 GiB allocated by PyTorch, 1.18 GiB in CUDA-graph private pools.

5. `MEM_FRACTION_LONG=0.82` at the full 1M (2026-09-01): **OOM gone** —
   capture entered with 9.16–9.63 GB free instead of 488 MiB and cleared.
   Died in the MTP draft model on all 8 ranks: `AttributeError:
   'HiSparseDSATokenToKVPool' object has no attribute
   'full_to_hisparse_device_index_mapping'`. HiSparse + speculative decoding
   is unwired in SGLang 0.5.13.post1.
6. Attempt 5 minus the four speculative flags (2026-09-01): **started and
   registered at `context_window: 1048576`**, `max_total_num_tokens=460352`,
   `available_gpu_mem=10.52 GB`, 47.11 GB host KV per rank. Then **crashed on
   the first request** — `TypeError: 'NoneType' object is not subscriptable`
   at `hisparse_coordinator.py:535`, because `req_pool_indices_cpu` is only
   assigned on the extend path (`schedule_batch.py:2094`) and the decode path
   passes `None`. One request returned a single token before the worker died;
   the other 15 got HTTP 503.

Diagnosis (revised 2026-09-01): the memory story is settled —
`--mem-fraction-static` fixes the static budget (weights + KV pool) and
SGLang sizes the KV pool to fill whatever remains of it after the weights
load. The pool is **not** derived from `--context-length`, which only caps
per-request length; attempt 4 proved lowering the target frees no GPU memory,
and 0.82 proved mem-fraction is the knob that does. What blocks Profile B now
is **not capacity** but two SGLang HiSparse bugs — one in the MTP integration,
one on the plain decode path — neither fixable from this repo. Re-test on an
engine newer than 0.5.13.post1. **No throughput or latency numbers exist for
Profile B and none are claimed.** Full logs and exact commands:
`RESULTS-kv-tiering.md`, section "Profile B (hisparse, long context)".

## Benchmark

The runtime image does **not** ship `aiperf`/`genai-perf` (so `bench.sh` won't run
here), and `sglang.bench_serving`'s random dataset needs to fetch a corpus from
HF Hub — both blocked in an offline environment. Use the bundled stdlib streamer instead
(no tokenizer / no docker), which streams `/v1/chat/completions` and reports
TTFT / ITL / decode throughput:

```bash
./bench_stream.py --concurrency 1  --num 16  --max-tokens 256   # latency
./bench_stream.py --concurrency 32 --num 128 --max-tokens 256   # throughput
```

### Results — MTP speculative decoding OFF vs ON (8× H200, TP=8, 2026-06-27)

Two MTP-on cells below (conc-1 decode tok/s and conc-32 system tok/s) were
corrected 2026-08-31 against the no-hicache baseline re-measured in
`RESULTS-kv-tiering.md` (same `bench_stream.py --concurrency 1 --num 16` /
`--concurrency 32 --num 128` scenarios); the earlier figures here were stale.
The remaining cells are unchanged from the original 2026-06-27 measurement
and were not re-run for this correction.

| Profile | Metric | MTP off | MTP on | Δ |
|---|---|---|---|---|
| Latency (conc 1)   | decode tok/s (per req) | 75.4   | **150.6** | **2.00×** |
| Latency (conc 1)   | system tok/s           | 74.7   | 136.9     | 1.83× |
| Latency (conc 1)   | TTFT mean              | 45 ms  | 169 ms    | +draft overhead |
| Throughput (conc 32) | system tok/s         | 1636.9 | **2087.1** | 1.28× |
| Throughput (conc 32) | decode tok/s (per req) | 52.5 | 73.9      | 1.41× |

Worker decode stats with MTP on show `accept len ≈ 2.2–2.8` (accept rate
0.54–0.90). Spec decoding ~doubles single-stream decode and adds ~20% aggregate
throughput; TTFT rises (the draft pass) and the win shrinks at high concurrency —
the expected speculative-decoding profile.

## Status / verification checklist

- [x] vLLM path blocked (no Dynamo image ≥ vLLM 0.23.0) — SGLang chosen
- [x] Model requires **SGLang ≥ 0.5.13.post1** (model card). Stock images too old
      (1.2.1→0.5.11, 1.3.0-dev.1→0.5.12.post1) → custom image (see `Dockerfile`):
      Dynamo dev base + `sglang==0.5.13.post1` + `sglang-kernel==0.4.3`.
- [x] Model fits aggregated TP=8 (~94 GB/GPU weights); disagg needs ≥ 2 nodes
- [x] Aggregated stack boots, worker loads weights, DSA backend auto-selected on H200
      (`dsa`/`flashmla_kv`, no `--attention-backend` override), registers
- [x] OpenAI smoke tests pass: `/v1/models` lists `glm-5.2-fp8`; chat completion
      returns a clean answer with `reasoning_content` (glm45 reasoning parser working)
- [x] Tool-call (`glm47`) exercised with a real tool schema — `get_weather` via the
      LiteLLM gateway returns `finish_reason: tool_calls` with structured args
- [x] **MTP / EAGLE speculative decoding enabled and verified** — composes with the
      auto-selected DSA backend on H200; `accept len ≈ 2.2–2.8`, ~2× single-stream decode
- [x] **EP + DP-attention "throughput" config benchmarked and rejected** (2026-06-28):
      regressed to 910 vs 1580 system tok/s at conc 32 + halved KV pool — TP=8 stays
      (it's a ≥ 2-node lever; see Performance levers)
- [x] **Context raised to 512K** (`--context-length 524288`); full-fidelity 1M is
      not servable on one node (KV pool `max_total_num_tokens=540928` measured
      under Profile A, mem-fraction 0.85)
- [x] Benchmark recorded (MTP off vs on, `bench_stream.py`) — see Benchmark section
- [ ] Benchmark vs the vLLM path (still blocked: no Dynamo vLLM ≥ 0.23.0 image)
- [x] **Profile A (tiered KV cache) benchmarked** (2026-08-31): eviction survival
      is a clean PASS (evicted 131K-token prefix: 0% cached / 16,769 ms TTFT
      before tiering → 100% cached / 1,230 ms after); conc-32 throughput
      regresses -8.7% (1905.2 vs 2087.1 tok/s), conc-1 unaffected (-0.5%) — see
      "Tiered KV cache" above
- [x] **Restart persistence (L3 survival) verified** (2026-08-31): a fresh
      131K-token prefix, cached, then read back at 100% cached / 5,121 ms TTFT
      after a full worker restart that wiped L2 host RAM — proves `/scratch`
      (not host RAM) served it
- [x] **Profile B (HiSparse, 1M context) attempted and does not start on this
      hardware** (2026-08-31): 3/3 attempts CUDA OOM during CUDA-graph
      capture (default mem-fraction, reduced mem-fraction, and reduced
      cuda-graph-max-bs) — see "Tiered KV cache" above and `CONTEXT_WINDOW.md`

### Known issues

- **`./stop.sh` intermittently fails to kill the worker container** with
  `PID ... is zombie and can not be killed. Use the --init option when
  creating containers to run an init inside the container that forwards
  signals and reaps processes.` This hit every restart performed during the
  KV-cache-tiering measurement pass (2026-08-31). Workaround: run
  `docker kill glm52-sglang-worker-1` first (exits cleanly within a few seconds),
  then `./stop.sh` completes its teardown normally. Not investigated further
  (root cause is presumably the container missing an init/reaper process);
  cosmetic in that teardown still succeeds once you clear the zombie.

  **Fixed 2026-08-31 by `init: true`** on `x-worker-base` (docker-init as pid 1
  reaps, so the container is no longer unkillable; stop and `compose up -d`
  recreate now complete unaided). **The exit code is still 137** and that part
  is NOT fixed: measured on a loaded worker 2026-09-01, SIGTERM at 01:41:51,
  engine shutdown complete at 01:41:57, container SIGKILLed at 01:42:35.
  Root cause is upstream, not reaping -- the shutdown handler calls
  `kill_process_tree(..., include_parent=False)` and never terminates the engine
  process itself, so pid 1 waits on a child that never exits. Treat 137 as
  expected: the graceful path (unregister, engine shutdown, KV flush) has
  already completed by +6s.

- **Scheduler watchdog kill (also exit 137, but a real fault — not the cosmetic
  teardown one above).** On 2026-09-03 the worker died after ~2.5 days up. The
  signature is distinct from the teardown 137 and worth learning to tell apart:

  | | Teardown 137 (cosmetic) | Watchdog 137 (real) |
  |---|---|---|
  | Trigger | you ran `./stop.sh` | nothing — it died on its own |
  | Log marker | `SIGTERM` → engine shutdown | `Scheduler watchdog timeout` |
  | Preceded by | a clean unregister | ~5 min of *zero* forward passes |

  What happened: last forward pass at 15:01:37, then the scheduler stalled with
  4 requests in flight. Requests kept arriving (15:05–15:09) with no prefill or
  decode progress. At 15:09:02 — exactly `watchdog_timeout=300` s later — the
  watchdog fired **on all 8 ranks within 70 ms of each other**, which points at
  a hang in a collective rather than a single-rank fault. Then SIGQUIT →
  `kill_process_tree` → SIGKILL.

  Not an OOM: `docker inspect` reported `OOMKilled=false`, and the host had
  2.2 TB with 48 GB used. Root cause **not** identified, because both crash
  diagnostics were unavailable at the time (fixed below).

  **Ruled out afterwards (2026-09-03, host-level evidence that outlived the
  container).** None of these explain the stall, so don't re-tread them:

  - *GPU hardware fault* — no `Xid` anywhere in `/var/log/kern.log`; 0
    uncorrected volatile ECC errors on all 8 GPUs; every NVLink up at
    26.562 GB/s; `nvidia-fabricmanager` active since 2026-09-02 06:50 with
    `Fabric State: Completed / Status: Success` on all 8.
  - *Host OOM killer* — the kernel log records no OOM kill of any process, at
    any time. (Distinct from the `OOMKilled=false` check, which only covers the
    container's own cgroup.)
  - *The KV reaper deleting cache files under a live worker* — plausible on
    paper, but `/scratch/kvcache/reaper.log` shows `removed 0 files` on **every**
    run: the tree is ~1.8 TB against a 10 TB budget, so it has never evicted
    anything.
  - *A recurring pattern* — the frontend's continuous log covers 2 days and
    contains exactly one unexplained worker death (the two on 09-01 were the
    deliberate Profile B experiments). This fired once.

  **Leading hypothesis: the hierarchical-cache collectives.** Established
  2026-09-03 by reading the engine source and sampling the live scheduler
  (which the py-spy fix below made possible). Profile A runs
  `--enable-hierarchical-cache` with `--hicache-storage-backend=file` on
  `/scratch`, and that path performs **unconditional CPU all-reduces on every
  prefill batch**, from `check_hicache_events` ←
  `scheduler.py:2577 _get_new_batch_prefill_raw`:

  | Collective | Reduces | Source |
  |---|---|---|
  | `loading_check` | `finish_count` of completed async KV loads, `ReduceOp.MIN` | `hiradix_cache.py:950` |
  | `drain_storage_control_queues` | `[prefetch_revoke, ack_backup, host_mem_release]` queue depths, `ReduceOp.MIN` | `hiradix_cache.py:1307` |

  Both go through `_all_reduce_attn_groups` (`hiradix_cache.py:196`) over the
  TP group, and neither has a timeout. The `MIN` semantics are deliberate — the
  docstring says it exists "to minimize TP synchronization" — and they keep all
  8 ranks in lockstep on work whose pace is set by **disk I/O to `/scratch`**.

  Why this fits: if one rank blocks in the file backend (a stalled NVMe read, a
  slow fsync, an FS hiccup) it never *arrives* at the next collective, and the
  other seven block in `all_reduce` forever. That produces precisely the
  observed signature — every rank stops making forward progress at the same
  instant, so all 8 trip the 300 s watchdog within 70 ms of each other, with
  zero forward passes in between — and it is reachable **only in Profile A**,
  which is what was running.

  **Read the caveat before chasing it.** Sampling the live, *healthy* worker
  shows all ranks sitting in exactly these frames, because an idle scheduler
  polls `check_hicache_events` in a loop. Ranks being here is the NORMAL state
  and is **not** evidence of a fault. This is a mechanism that fits the
  evidence, not a diagnosis; nothing from the actual incident survived.

  **The discriminating observation**, for whoever reads the next watchdog dump:
  check whether *one* rank is somewhere else — inside the hicache file backend
  — while the other seven sit in `_all_reduce_attn_groups`. That asymmetry
  would confirm it. All 8 in the all-reduce with none in the backend would
  point elsewhere. Cheap falsification if it recurs: run with
  `--hicache-storage-backend` disabled (costs the L3 tier) and see whether the
  hang follows.

  **Also open, lower priority:** every worker start logs
  `NV_ERR_FABRIC_STATE_OUT_OF_SYNC` at `mem_multicast_fabric.c` in the kernel
  log, paired with `CUDASymmetricMemory.cu` warning `init_multicast_for_block`
  failed. NVLink *multicast* allocation is failing and torch is falling back.
  This is **not** an explanation — it happens on successful starts too,
  including the currently-serving one — but the hang was in a collective, so
  correlate it against the next dump rather than dismissing it.

- **Crash diagnostics (added 2026-09-03).** The watchdog kill above produced no
  usable evidence — every py-spy dump returned `Permission Denied` and the CUDA
  path bailed with `CUDA user-triggered coredump is not enabled`. Three changes
  make the next one diagnosable:

  1. `cap_add: SYS_PTRACE` on `x-worker-base`, **and** a `cap_sys_ptrace+ep`
     file capability on `/usr/local/bin/py-spy` in the `Dockerfile`. Both are
     required and neither works alone: the host runs Yama `ptrace_scope=1` (a
     process may only ptrace its own *descendants*, and the handler dumps
     *sibling* ranks), while the container runs as uid 1000 with no ambient
     capabilities — so `cap_add` alone only reaches `CapBnd`, leaving
     `CapPrm`/`CapEff` at `0`. The file capability is what actually delivers the
     cap to py-spy at exec.
  2. `CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1`, with `CUDA_COREDUMP_FILE` pointed
     at `${DIAG_DIR:-/scratch/diag}/coredumps` — its own bind mount, deliberately
     **outside** the KV tree (the reaper deletes the oldest regular file of *any*
     name under its root, so a dump parked there would be both reaper-bait and a
     cause of real KV eviction). `CUDA_COREDUMP_PIPE` is left at the driver
     default on purpose: SGLang looks for the trigger pipe at the cwd-relative
     path, and overriding it would move the pipe away from the code that fires it.
  3. `CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory`, because dumping device
     global memory writes ~122 GB *per rank* (~1 TB per incident across 8) and
     answers nothing about where a rank is stuck. Backtraces plus local/shared
     memory are kept.
  4. **The log has to outlive the container.** 1–3 make the watchdog produce
     evidence; this makes it survive long enough to read. The watchdog writes
     its py-spy dump to the worker's stdout and nowhere else — there is no log
     file inside the container — and docker deletes a container's log *along
     with the container*. Both teardown paths here do exactly that: `stop.sh`
     runs `compose down` (always removes), `serve.sh` runs `compose up -d`
     (recreates on any config change). That is how the 2026-09-03 evidence was
     actually lost: the worker died at 15:10 and the 18:05 restart destroyed
     the only copy. Both scripts now call `archive_worker_log.sh` *before*
     tearing down, writing `${DIAG_DIR:-/scratch/diag}/logs/<service>-<UTC
     timestamp>-<short id>.log.gz` (~25 KB gzipped for a full startup;
     `docker logs --timestamps`, so the host clock lines up with
     `/var/log/kern.log` and the frontend log). A failure only warns — a broken
     diagnostics path must never block serving or a teardown that frees 8 GPUs.
     Archives are never pruned; they are tiny, and auto-deleting crash evidence
     is the bug this fixes. Regression tests: `tests/test_archive_worker_log.py`.

  To pull stacks from a wedged worker by hand, without waiting for the watchdog:

  ```bash
  docker exec glm52-sglang-worker-1 bash -c \
    'for p in $(pgrep -f sglang::scheduler); do echo "== $p"; py-spy dump --pid $p; done'
  ```

  If that prints `Permission Denied`, the running image predates the `Dockerfile`
  change — rebuild it, or patch the live container (does not survive recreation):

  ```bash
  docker exec -u 0:0 glm52-sglang-worker-1 python3 -c \
    "import os; os.setxattr('/usr/local/bin/py-spy', b'security.capability', \
     bytes.fromhex('0100000200000800000000000000000000000000'))"
  ```

### Build gotchas (offline / behind-a-proxy environments)

- **Behind a proxy:** the Docker bridge can't reach PyPI; the build needs
  `--network=host` + `--build-arg HTTP(S)_PROXY` (handled by `serve.sh`).
- **Worker needs the HF cache:** it loads the model config/tokenizer and weights and
  will otherwise try huggingface.co and fail — the compose mounts the HF cache and sets
  `HF_HUB_OFFLINE=1`.
- **First start is slow:** SGLang runs a DeepGEMM JIT pre-compile (~10–20 min) +
  CUDA-graph capture. During this phase GPU0 drives compilation (oscillates 0↔100%)
  while GPUs 1–7 spin-wait at a barrier (100% util but ~126 W). The JIT kernels are
  persisted to the `jit-cache` volume (`/home/dynamo/.cache` — `dynamo` is the image's
  uid-1000 user, unrelated to the removed orchestration layer), so **only the
  first start pays this cost** — later restarts on the same image/GPU reuse the cache
  and come up fast. (Removing the volume or changing the SGLang/GPU arch invalidates
  it and triggers one more recompile.)

### Maintenance: the KV cache reaper

The worker's tiered KV cache (`--hicache-storage-backend=file`, `KV_SCRATCH_DIR`,
writing to `/scratch/kvcache/glm52`) has **no eviction and no size cap** — SGLang's
`file` backend just keeps writing one `.bin` per page component forever. Left alone
it grows without bound until the 28 TB `/scratch` volume fills and the worker starts
failing writes. `sglang/kv_reaper.py` (20 tests) enforces a byte budget on
that directory, deleting the oldest files first until the tree fits.

**Schedule** — a user crontab entry, no sudo required (the invoking user owns
`/scratch/kvcache`). Crontab entries need an absolute path, so substitute
your actual checkout location for `/path/to/GLM-SGLANG` below:

```cron
*/15 * * * * /path/to/GLM-SGLANG/sglang/kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB >> /scratch/kvcache/reaper.log 2>&1
```

Install with `crontab -e` (interactive; not automatable). The log is written to
`/scratch/kvcache/reaper.log` — one level **above** the reaped root
(`/scratch/kvcache/glm52`) — so the reaper never counts or deletes its own log.

- **Budget: 10 TB of the 28 TB `/scratch` volume.** Leaves headroom for other
  consumers of the volume and for bursts between 15-minute runs.
- **Eviction is by mtime, not atime.** `/scratch` is mounted `relatime`, which only
  advances atime when it is already older than mtime or older than 24h — far too
  coarse to drive an LRU. mtime (set once, at file creation; these cache files are
  never rewritten) approximates insertion order well enough for oldest-first
  eviction.
- **Deleting under a live worker is safe.** Every file in the tree is a regenerable
  cache entry — a miss just falls back to recompute (or a lower cache tier). This
  was validated directly: a file owned by the container's uid (1000) was
  deleted by the host user while the worker was live and serving, and the deletion
  succeeded and stuck. The containing directory is `0777` with no sticky bit, so
  POSIX only requires write+execute on the directory (which the host user has), not
  ownership of the file.
- **Observed growth rate:** roughly **8 GB per 131K-token prefix primed** — one
  prime-then-warm cycle took the cache from ~47 GB to ~55 GB. At that rate the
  10 TB budget has substantial headroom, but it's the number to use when reasoning
  about how fast the budget fills under heavier prefix-caching workloads.
