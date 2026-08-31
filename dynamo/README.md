# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang backend, 8× H200)

Serves `zai-org/GLM-5.2-FP8` on a single node using **NVIDIA Dynamo** as the
serving/orchestration layer, OpenAI-compatible on host port `:8000`. This is the
project's serving path (the earlier plain-vLLM container was removed).

## Why SGLang, not vLLM

GLM-5.2 is `GlmMoeDsaForCausalLM`: MoE + MLA + **DeepSeek-style Sparse Attention (DSA)**
with `head_size=704`. The vLLM path needs **vLLM ≥ 0.23.0** for that.

Verified (2026-06-27): **no published Dynamo `vllm-runtime` image, nor the
`ai-dynamo` PyPI wheel, ships vLLM ≥ 0.23.0.**
`1.2.1`→0.20.1, `1.3.0-dev.1`→0.22.0 (its sparse-MLA backends cap at `head_size=576`),
`kimi-k2.6-dev`→0.21.0. So we use Dynamo's **SGLang** backend — NVIDIA's own GLM-5
Dynamo recipe (`ai-dynamo/dynamo/recipes/glm-5-nvfp4`) is SGLang too.

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
stack does. Dynamo still adds: an OpenAI frontend + runtime/observability, KV-aware
routing (kicks in once you scale to multiple replicas/nodes), request migration, and a
clean path to multi-node disaggregation later.

## Layout

- `docker-compose.yml` — etcd + NATS + Dynamo frontend + one SGLang worker (TP=8). Host
  networking; tunables via `${...}` env (see below).
- `serve.sh` / `stop.sh` — bring the stack up / down.
- `bench.sh` — load-test any OpenAI endpoint (Dynamo or vLLM) with aiperf/genai-perf.

## Usage

```bash
cd dynamo
./serve.sh                       # start (detached)
docker compose logs -f worker    # watch model load (several minutes, 756 GB)
curl http://localhost:8000/v1/models
./stop.sh
```

Tunables (env): `PORT`, `MAX_MODEL_LEN` (→ sglang `--context-length`, default 524288),
`MEM_FRACTION` (default 0.85), `TP_SIZE` (default 8), `PAGE_SIZE` (default 64, DSA),
`MAX_RUNNING` (→ `--max-running-requests`, default 128 — the concurrency ceiling;
spec decoding would otherwise auto-cap it to 48),
`HF_CACHE` (default `/root/.cache/huggingface`), `MODEL`, `SERVED_NAME`, `DYNAMO_IMAGE`.
MTP speculative decoding is on by default; tune via `SPEC_ALGO` (default `EAGLE`),
`SPEC_NUM_STEPS` (2), `SPEC_EAGLE_TOPK` (1), `SPEC_NUM_DRAFT` (3), or disable by
editing the worker `command:` in `docker-compose.yml`.

## Serving config (mirrors the model card / NVIDIA recipe)

- `--tp-size 8`, `--kv-cache-dtype fp8_e4m3`, `--page-size 64`
- **DSA attention backend auto-selected** — no `--attention-backend`; SGLang picks
  `dsa`/`flashmla_kv` (Hopper + fp8 KV). Verified ~+8% system tok/s at conc 32 vs the
  legacy `nsa` backend we used to force.
- `--dyn-tool-call-parser glm47`, `--dyn-reasoning-parser glm45` (Dynamo frontend parsers)
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
- **KV-aware routing** — add `--router-mode kv` to the frontend and a `--kv-events-config`
  to the worker; only a win with ≥ 2 workers/replicas.

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
| `PROFILE` | `cache` | `cache` = Profile A (tiered KV cache, 512K); `longctx` = Profile B (HiSparse, 1M target — does not currently start, see below) |
| `HICACHE_GB` | `96` | L2 host RAM pool size, **per TP rank** (see tier table above for why this multiplies by 8) |
| `HICACHE_WRITE_POLICY` | `write_through` | `write_through` (every page, default) vs `write_through_selective` (hotter pages only — measured not to help, see above) |
| `KV_SCRATCH_ROOT` | `/scratch/kvcache` | host directory bind-mounted into the container (the whole tree) |
| `KV_SCRATCH_DIR` | `/scratch/kvcache/glm52` | this model's subdirectory under `KV_SCRATCH_ROOT`; must stay under it or it isn't visible in the container |
| `HISPARSE_DEVICE_BUFFER` | `4096` | Profile B only — HiSparse device-side buffer size |
| `HISPARSE_RATIO` | `2` | Profile B only — HiSparse host-to-device ratio |
| `MAX_MODEL_LEN_LONG` | `1048576` | Profile B only — target context length (the full 1M) |
| `MEM_FRACTION_LONG` | `0.88` | Profile B only — tried at `0.88` and `0.86`, both OOM (see below) |
| `CUDA_GRAPH_MAX_BS_LONG` | `128` | Profile B only — caps CUDA-graph capture at the concurrency ceiling instead of SGLang's default 512; reduced but did not close the OOM (see below) |

**Profile B (HiSparse, 1M context) — does not start on this hardware.** Three
attempts, all CUDA OOM during CUDA-graph capture, none reaching a registered
state:

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

Diagnosis: the indexer's capture working set scales with HiSparse's
1,048,576-token addressable range, not with request concurrency, so
graph-count and mem-fraction tuning cannot close the gap — the untried lever
is a smaller `--context-length` for this profile. Full logs and exact
commands: `RESULTS-kv-tiering.md`, section "Profile B (hisparse, long
context)".

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

- **`./stop.sh` intermittently fails to kill `dynamo-worker-1`** with
  `PID ... is zombie and can not be killed. Use the --init option when
  creating containers to run an init inside the container that forwards
  signals and reaps processes.` This hit every restart performed during the
  KV-cache-tiering measurement pass (2026-08-31). Workaround: run
  `docker kill dynamo-worker-1` first (exits cleanly within a few seconds),
  then `./stop.sh` completes its teardown normally. Not investigated further
  (root cause is presumably the container missing an init/reaper process);
  cosmetic in that teardown still succeeds once you clear the zombie.

### Build gotchas (offline / behind-a-proxy environments)

- **Behind a proxy:** the Docker bridge can't reach PyPI; the build needs
  `--network=host` + `--build-arg HTTP(S)_PROXY` (handled by `serve.sh`).
- **Frontend needs the HF cache:** on discovery the frontend loads the model
  config/tokenizer (not weights) and will otherwise try huggingface.co and fail —
  the compose mounts the HF cache + sets `HF_HUB_OFFLINE=1` for the frontend too.
- **First start is slow:** SGLang runs a DeepGEMM JIT pre-compile (~10–20 min) +
  CUDA-graph capture. During this phase GPU0 drives compilation (oscillates 0↔100%)
  while GPUs 1–7 spin-wait at a barrier (100% util but ~126 W). The JIT kernels are
  persisted to the `dynamo-jit-cache` volume (`/home/dynamo/.cache`), so **only the
  first start pays this cost** — later restarts on the same image/GPU reuse the cache
  and come up fast. (Removing the volume or changing the SGLang/GPU arch invalidates
  it and triggers one more recompile.)

### Maintenance: the KV cache reaper

The worker's tiered KV cache (`--hicache-storage-backend=file`, `KV_SCRATCH_DIR`,
writing to `/scratch/kvcache/glm52`) has **no eviction and no size cap** — SGLang's
`file` backend just keeps writing one `.bin` per page component forever. Left alone
it grows without bound until the 28 TB `/scratch` volume fills and the worker starts
failing writes. `dynamo/kv_reaper.py` (20 tests) enforces a byte budget on
that directory, deleting the oldest files first until the tree fits.

**Schedule** — a user crontab entry, no sudo required (the invoking user owns
`/scratch/kvcache`). Crontab entries need an absolute path, so substitute
your actual checkout location for `/path/to/GLM-5.2-FP8` below:

```cron
*/15 * * * * /path/to/GLM-5.2-FP8/dynamo/kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB >> /scratch/kvcache/reaper.log 2>&1
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
