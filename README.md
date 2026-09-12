# GLM on SGLang (8× H200)

An SGLang serving stack for GLM-family models on a single 8× H200 node, exposing an
OpenAI-compatible API on `:8000`.

**Currently serving [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
(`PROFILE=flash`), at the full 1M context.** GLM-5.2-FP8 is retained as a rollback path
but is no longer run.

```bash
cd sglang && PROFILE=flash ./serve.sh
```

| Profile | Model | Context | Status |
|---|---|---|---|
| **`flash`** | **GLM-5.3-Flash** | **1,048,576 (1M)** | **primary** — verified 2026-09-12 |
| `cache` | GLM-5.2-FP8 | 524,288 (512K) | retained for rollback, not running |
| `longctx` | GLM-5.2-FP8 | 1M configured | never served a request |

The two GLM-5.2 profiles still work and their measurements
([`sglang/RESULTS-kv-tiering.md`](sglang/RESULTS-kv-tiering.md)) remain valid, but they
describe a model this node no longer serves. GLM-5.2's weights are kept on disk; its
`/scratch` KV cache was deleted on 2026-09-12 when it was retired.

**The two models are not interchangeable.** GLM-5.3-Flash is `Glm5Next` — 45 layers of
hybrid attention (34 KDA linear + 11 DSA), 288 routed experts, natively multimodal,
~306 GB — against GLM-5.2's 78 all-DSA layers and ~756 GB. Almost no tuning carries
over, and the wrong `--kv-cache-dtype` is invalid rather than merely slow on Hopper.
It also needs a **different engine image**: `glm5_next` is in no public SGLang release.
See [`sglang/README.md`](sglang/README.md) and
[`CONTEXT_WINDOW.md`](CONTEXT_WINDOW.md).

> This repo has served the model two ways before: a plain vLLM container, and then
> NVIDIA Dynamo with the SGLang backend. vLLM was dropped because no Dynamo runtime
> ships vLLM ≥ 0.23.0, which GLM-5.2's sparse MLA needs. Dynamo itself was dropped on
> 2026-09-11: its frontend duplicated `sglang.launch_server`, and multi-worker
> discovery, KV-aware routing, and disaggregated prefill/decode are all unreachable on
> a single node holding one ~756 GB model. The engine has been SGLang throughout, and
> no engine flag changed when Dynamo was removed. See `sglang/README.md`.

## Model (GLM-5.2-FP8, retained for rollback)

- Architecture: `GlmMoeDsaForCausalLM` — MoE (256 routed + 1 shared experts, 8/tok),
  MLA attention, **DeepSeek-style Sparse Attention (DSA)** with an indexer
  (`index_topk=2048`), FP8 block-quant (128×128, e4m3), 78 layers, 1 MTP layer, 1M
  max context. ~756 GB of weights (≈94 GB/GPU at TP=8).
- Weights live in a shared Hugging Face cache (point `HF_CACHE` at it); they are not re-downloaded.

## Serve

`PROFILE=flash` (GLM-5.3-Flash) is the default; `PROFILE=cache` (GLM-5.2-FP8) is
kept for rollback. Both use a file-backed L3 KV tier on `/scratch`, each in its own
directory (`glm53` and `glm52` -- storage keys carry no model identity, so they must
not share one). It must exist and be writable **by uid 1000, not just the host
user** before the first start (`serve.sh` probes as uid 1000; the GLM-5.3-Flash
image itself runs as root):

```bash
sudo mkdir -p /scratch/kvcache/glm53
sudo chown -R 1000:$(id -g) /scratch/kvcache
sudo chmod -R 2775 /scratch/kvcache
```

Owner `1000` so the container can write; your host group kept with `g+w` so the
reaper (`sglang/kv_reaper.py`, run from cron as the host user) can still delete
what the container creates; setgid so new files inherit that group. `serve.sh`
prints these same two commands if its probe fails.

Without root, `chmod -R 0777 /scratch/kvcache` also works — the owner may chmod
without being root — but it is world-writable, so prefer the `chown` above where
you can.

A fresh clone following a naive `chown $(id -u)` (host user, not uid 1000) will
hit `serve.sh`'s guard: it fails fast with a **containerized** write probe (a
throwaway container writes as uid 1000 before `docker compose up` runs), because
SGLang's file-backed hicache tier silently swallows write failures instead of
erroring — a permissions mismatch here would otherwise start the worker clean
with a dead L3 cache tier and no warning.

```bash
cd sglang
./serve.sh                           # default PROFILE=flash: GLM-5.3-Flash, 1M context, tiered KV cache
# PROFILE=cache ./serve.sh           # GLM-5.2-FP8 rollback: 512K context + tiered KV cache
# PROFILE=longctx ./serve.sh         # GLM-5.2-FP8 1M attempt via HiSparse: starts, but crashes
#                                     # on its first request -- see CONTEXT_WINDOW.md
docker compose logs -f worker-flash  # watch startup (`worker` under PROFILE=cache)
./stop.sh                            # stop + remove; on GLM-5.2 the first `down` can fail, re-run it
```

Under `PROFILE=cache`, the GLM-5.2 stack = one SGLang worker (aggregated, TP=8,
auto-selected DSA attention backend (`flashmla_kv` on Hopper+fp8), fp8 KV cache,
**MTP/EAGLE speculative decoding on** — ~2× single-stream decode), with context
served at **512K** (`--context-length
524288`) with a tiered prefix cache (GPU radix → host RAM → `/scratch`
NVMe) so that evicted prefixes can still be served from cache instead of
recomputed. That protection is not free: measured conc-32 system throughput
drops to **1905.2 tok/s from a 2087.1 tok/s baseline (~9%)** under the
default write policy, while conc-1 decode is unaffected (149.8 vs 150.6
tok/s) — see [`sglang/README.md`](sglang/README.md#tiered-kv-cache) for the
full breakdown. The model's 1M max is not servable on one node with full
fidelity: the measured KV pool tops out at ~541K tokens alongside the
weights, and `PROFILE=longctx` — the single-node attempt at the full 1M
context via SGLang HiSparse — fails to initialize on this hardware (see
`CONTEXT_WINDOW.md` for the three measured OOM attempts). Disaggregated
prefill/decode is **not possible on a single node** for this model (a full
copy per worker exceeds 8 GPUs) — it needs ≥ 2 nodes, and so does a
full-fidelity 1M context.
Details, tunables, and benchmarks: [`sglang/README.md`](sglang/README.md).

> First start runs a DeepGEMM JIT pre-compile + CUDA-graph capture (~10–20 min). It's
> cacheable with `python3 -m sglang.compile_deep_gemm` (same args).

## Test

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.2-fp8",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

Or use the bundled streaming CLI: `./chat.py` (stdlib only; reads/streams the
`reasoning_content` from the glm45 reasoning parser).
