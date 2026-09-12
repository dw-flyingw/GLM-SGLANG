# Context Window

| Model | Profile | Served context | Status |
|---|---|---|---|
| **GLM-5.3-Flash** | `flash` | **1,048,576 (1M)** | **serving, verified 2026-09-12** |
| GLM-5.2-FP8 | `cache` | 524,288 (512K) | retired 2026-09-12, rollback only |
| GLM-5.2-FP8 | `longctx` | 1,048,576 (configured) | never served a request |

GLM-5.2 was retired on 2026-09-12 once GLM-5.3-Flash was verified. Its weights are
kept on disk as a rollback path; its 8.7 TB `/scratch/kvcache/glm52` KV tier was
deleted, and the reaper cron for it removed. Rolling back therefore starts with a
cold cache.

## GLM-5.3-Flash serves the full 1M (2026-09-12)

The single-node 1M barrier that the rest of this document is about **does not
apply to GLM-5.3-Flash**. Nothing was fixed; the architecture is different.
GLM-5.2 is 78 layers of DSA attention, all of which hold a paged KV cache.
GLM-5.3-Flash is 45 layers of which only **11** are DSA -- the other 34 are KDA
linear-attention layers carrying fixed-size recurrent state -- and the
checkpoint is ~306 GB rather than ~756 GB. On the same 8 H200s:

| | GLM-5.2-FP8 | GLM-5.3-Flash |
|---|---|---|
| KV pool (`max_total_num_tokens`) | 540,928 | **3,687,104** |
| A single 1M-token request | **does not fit** (pool < request) | fits, ~3.5x headroom |
| Served context | 524,288 | **1,048,576** |

Verified end-to-end, not merely configured: **999,256 tokens** prefilled with a
needle planted at 10% depth and recalled correctly (~49 s), reproduced at
999,915 and 999,483. See `sglang/RESULTS-glm53-longctx.md` for the runs, the two
harness bugs that nearly produced a false pass, and the scope limits (it is a
retrieval check, not a reasoning-at-depth check).

`PROFILE=flash` pins no `--context-length`, so the engine takes the checkpoint
maximum (`max_position_embeddings: 1048576`, `rope_scaling: None` -- native 1M,
not RoPE-extended).

**Concurrency trades directly against context.** ~1M per request against a
3,687,104-token pool is roughly **3 concurrent full-length requests**;
`--max-running-requests=128` does not change that. ~18 at 200K, ~36 at 100K.

---

# Everything below is about GLM-5.2-FP8

Retained as the record of why 1M was unreachable for **that** model on this
hardware. It remains accurate for `PROFILE=cache` and `PROFILE=longctx`.

The GLM-5.2-FP8 model supports a **1M token** max context natively, but that
deployment serves **512K** (`--context-length 524288`) under `PROFILE=cache`,
which was the repo's default profile until 2026-09-12.

## Why not 1M on a single node (GLM-5.2)

- Hardware: 8× H200 on a single node, TP=8.
- The KV cache pool tops out at **~541K tokens** alongside the ~756 GB of FP8
  weights (≈94 GB/GPU). There isn't enough GPU memory left to hold a full 1M
  KV pool at that concurrency ceiling.
- Disaggregated prefill/decode — the path that would let us exceed single-node
  KV limits — **is not possible on a single node** for this model, since a full
  weights copy per worker exceeds 8 GPUs.

## The single-node attempt at 1M (Profile B / HiSparse), and why it can't serve

`docker-compose.yml` defines a second profile, `PROFILE=longctx`, that tries to
reach the model's full 1,048,576-token context on this same single node via
SGLang's HiSparse attention path (`--enable-hisparse --disable-radix-cache`,
`--context-length=1048576`). As of 2026-09-01 it **does start at 1M** — but it
**crashes on the first request it is given**, so it is still unusable. Six
attempts, `sglang/RESULTS-kv-tiering.md`:

- **Attempt 1** (`--mem-fraction-static=0.88`, the documented default for this
  profile): worker died during CUDA-graph capture —
  `CUDA out of memory. Tried to allocate 1.50 GiB. ... 488.19 MiB is free.`
- **Attempt 2** (`MEM_FRACTION_LONG=0.86`, freeing more headroom): same
  failure mode; more memory was free at the moment of failure (1.24 GiB vs.
  attempt 1's 488 MiB) but not proportionally more, and still not enough —
  `CUDA out of memory. Tried to allocate 1.31 GiB. ... 1.24 GiB is free.`
- **Attempt 3** (`0.88` plus `--cuda-graph-max-bs=128`, capping wasted CUDA
  graphs at the real concurrency ceiling): got further — the generic
  graph-capture OOM was gone, host memory pools were allocated, DeepGEMM
  warmup completed — then died one call deeper, inside the **HiSparse
  indexer's own graph capture** (`dsa_indexer.py::_get_topk_paged` →
  `deep_gemm.fp8_paged_mqa_logits`): `Tried to allocate 1.50 GiB ... 488 MiB
  is free`, with the GPU fully pinned (139.20/139.80 GiB in use).
- **Attempt 4** (2026-09-01; attempt 3's exact config — `0.88` plus
  `--cuda-graph-max-bs=128` — with **only** `MAX_MODEL_LEN_LONG=786432`
  changed, i.e. a 768K target instead of 1M): failed *earlier* in the
  sequence than attempt 3, back in the **generic** graph capture
  (`cuda_graph_runner.py:628` via `init_device_graphs`), and with *less*
  memory free — `Tried to allocate 60.00 MiB. GPU 1 ... 48.06 MiB is free`,
  135.16 GiB allocated by PyTorch, 1.18 GiB in CUDA-graph private pools.
- **Attempt 5** (2026-09-01; `MEM_FRACTION_LONG=0.82` at the full 1M target,
  otherwise attempt 3's config): **the OOM is gone.** Capture was entered with
  9.16–9.63 GB free rather than 488 MiB, and cleared completely. The worker
  then died in the MTP draft model on all 8 ranks —
  `AttributeError: 'HiSparseDSATokenToKVPool' object has no attribute
  'full_to_hisparse_device_index_mapping'` (`deepseek_nextn.py:227/350` →
  `deepseek_v2.py:1868 forward_absorb_core`). That attribute is installed by
  `HiSparseDSATokenToKVPool.register_mapping()`, which `hisparse_coordinator.py`
  calls only for the main pool; the NextN/MTP draft model gets its own pool
  instance that never has it registered. **HiSparse + speculative decoding is
  unwired in SGLang 0.5.13.post1.**
- **Attempt 6** (2026-09-01; attempt 5 with the four speculative flags
  removed): **the worker started and registered at the full 1M.**
  `context_len=1048576`, `max_total_num_tokens=460352`,
  `available_gpu_mem=10.52 GB`, 47.11 GB of host hierarchical KV per rank
  (~377 GB total), CUDA-graph capture 132.32 s / 1.32 GB. `/v1/models`
  advertised `context_window: 1048576`. It then **crashed on the first
  benchmark request**: one request returned a single token (TTFT 846 ms) and
  the worker died (exit 137), the remaining 15 getting HTTP 503 —
  `TypeError: 'NoneType' object is not subscriptable` at
  `hisparse_coordinator.py:535` (`req_idx = int(req_pool_indices_cpu[i])`) via
  `map_last_loc_to_buffer` ← `schedule_batch.py:2568 prepare_for_decode`, on
  all 8 ranks. `req_pool_indices_cpu` defaults to `None`
  (`schedule_batch.py:1609`) and is assigned only on the extend/alloc path
  (line 2094), so the **decode** path passes `None` into HiSparse's backup
  hook. **HiSparse's decode path is broken in SGLang 0.5.13.post1**,
  independent of hardware, memory, or configuration.

Attempts 1–4 are CUDA OOM during CUDA-graph capture, not a config mistake or
a transient fault — reproduced across three different mitigations. Attempts 5
and 6 clear that OOM entirely and expose two separate SGLang defects behind
it.

**Reducing `--context-length` is not a lever.** Measured 2026-09-01 by
attempt 4; this supersedes the diagnosis previously recorded here (and in
`sglang/README.md` and `sglang/docker-compose.yml`), which claimed a smaller
`--context-length` for Profile B was the promising untried lever. It is not.
`--mem-fraction-static` fixes the static budget (weights + KV pool), and
SGLang sizes the KV pool to fill whatever remains of that budget once the
weights are loaded; the pool is **not** derived from `--context-length`,
which only caps how long any single request may be. Dropping the target from
1,048,576 to 786,432 therefore freed no GPU memory whatsoever — it only
changed which rank lost the race first (48 MiB free on GPU 1 rather than
attempt 3's 488 MiB on GPU 6). At `0.88` the static budget is
0.88 × 139.80 = 123.0 GiB/GPU, and the process still ended with 139.63 GiB
in use.

`--mem-fraction-static` was the knob that actually mattered, and `0.82`
closed the memory gap outright (attempt 5). Note the conversion is not 1:1 —
attempt 2 showed ~2.8 GiB of freed static budget buying only ~0.75 GiB of
extra headroom — which is why the 0.88→0.86 step looked discouraging while
0.88→0.82 succeeded.

**No throughput or latency numbers exist for Profile B and none are claimed
here.** Attempt 6 is the only one that ever served a token, and it crashed
mid-benchmark; its single 846 ms TTFT sample is a crash artifact, not a
measurement.

## What it would take for a full-fidelity 1M context

- **An SGLang version that fixes the HiSparse decode path** (the
  `req_pool_indices_cpu is None` crash in attempt 6). Until then Profile B
  cannot serve a request at any context length on any hardware — this is no
  longer a capacity question. Re-test if the engine is upgraded past
  0.5.13.post1; if the decode fix lands, the MTP/HiSparse gap from attempt 5
  is worth re-testing too.
- ≥ **2 nodes** so the stack can disaggregate prefill and decode (separate
  workers hold weights/KV on different nodes), unblocking a full 1M context
  without the HiSparse prefix-caching trade-off above. This remains the route
  to 1M *with* prefix caching and MTP, which Profile B gives up even when it
  works.

## Current settings

(GLM-5.2 profiles only; for GLM-5.3-Flash see the top of this document.)

| Setting | Profile A (`cache`; default until 2026-09-12) | Profile B (`longctx`) |
|---|---|---|
| Model max context | 1,048,576 (1M) | 1,048,576 (1M) |
| Served context (`--context-length`) | 524,288 (512K) | 1,048,576 (1M, configured) |
| KV pool (measured, `max_total_num_tokens`) | **540,928** tokens | **460,352** tokens at `mem-fraction 0.82` (attempt 6) — *smaller* than Profile A; the rest pages to ~377 GB of host memory (47.11 GB/rank) |
| Attention | DSA, `flashmla_kv`, hierarchical radix cache (GPU→host→`/scratch`) | DSA, `flashmla_kv`, HiSparse (`--disable-radix-cache`, no prefix caching) |
| Backend | SGLang 0.5.13.post1 | SGLang 0.5.13.post1 |
| Speculative decoding | MTP/EAGLE on | **removed** — crashes with HiSparse (attempt 5); Profile B forgoes MTP's ~2× single-stream decode |
| Status | serving, measured (see `sglang/RESULTS-kv-tiering.md`) | **starts at 1M, cannot serve** — crashes on first request (SGLang HiSparse decode bug). Attempts 1–4 OOM; 5 MTP crash; 6 decode crash |

References: `README.md`, `sglang/README.md`, `sglang/RESULTS-kv-tiering.md`,
`sglang/docker-compose.yml`.
