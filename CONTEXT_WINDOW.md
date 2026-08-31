# Context Window: Why 1M Isn't Enabled

The GLM-5.2-FP8 model supports a **1M token** max context natively, but this
deployment serves **512K** (`--context-length 524288`) as its default profile
(`PROFILE=cache`).

## Why not 1M on a single node

- Hardware: 8× H200 on a single node, TP=8.
- The KV cache pool tops out at **~541K tokens** alongside the ~756 GB of FP8
  weights (≈94 GB/GPU). There isn't enough GPU memory left to hold a full 1M
  KV pool at that concurrency ceiling.
- Disaggregated prefill/decode — the path that would let us exceed single-node
  KV limits — **is not possible on a single node** for this model, since a full
  weights copy per worker exceeds 8 GPUs.

## The single-node attempt at 1M (Profile B / HiSparse), and why it doesn't start here

`docker-compose.yml` defines a second profile, `PROFILE=longctx`, that tries to
reach the model's full 1,048,576-token context on this same single node via
SGLang's HiSparse attention path (`--enable-hisparse --disable-radix-cache`,
`--context-length=1048576`). It exists in the repo but **does not currently
start on this hardware** — measured 2026-08-31, `dynamo/RESULTS-kv-tiering.md`:

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

All three attempts are CUDA OOM during CUDA-graph capture, not a config
mistake or a transient fault — reproduced across two different mitigations.
Diagnosis: the indexer's capture working set scales with HiSparse's
1,048,576-token addressable range, not with concurrency, so tuning
graph-count or mem-fraction further does not close the gap — a smaller
`--context-length` for Profile B is the untried lever. Because Profile B
never reached a registered state, no throughput/latency numbers exist for it,
and none are claimed here.

## What it would take for a full-fidelity 1M context

- ≥ **2 nodes** so the stack can disaggregate prefill and decode (separate
  workers hold weights/KV on different nodes), unblocking a full 1M context
  without the HiSparse prefix-caching trade-off above.

## Current settings

| Setting | Profile A (`cache`, default) | Profile B (`longctx`) |
|---|---|---|
| Model max context | 1,048,576 (1M) | 1,048,576 (1M) |
| Served context (`--context-length`) | 524,288 (512K) | 1,048,576 (1M, configured) |
| KV pool (measured, `max_total_num_tokens`) | **540,928** tokens | not reached — worker OOMs during CUDA-graph capture before this is logged |
| Attention | DSA, `flashmla_kv`, hierarchical radix cache (GPU→host→`/scratch`) | DSA, `flashmla_kv`, HiSparse (`--disable-radix-cache`, no prefix caching) |
| Backend | Dynamo + SGLang 0.5.13.post1 | Dynamo + SGLang 0.5.13.post1 |
| Speculative decoding | MTP/EAGLE on | MTP/EAGLE configured but unverified (worker never starts) |
| Status | serving, measured (see `dynamo/RESULTS-kv-tiering.md`) | does not start on this node (3/3 attempts OOM) |

References: `README.md`, `dynamo/README.md`, `dynamo/RESULTS-kv-tiering.md`,
`dynamo/docker-compose.yml`.
