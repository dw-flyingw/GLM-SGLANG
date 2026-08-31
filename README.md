# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang, 8× H200)

Serves [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) on a single node
across all 8 H200 GPUs via **NVIDIA Dynamo** with the **SGLang** backend, exposing an
OpenAI-compatible API on `:8000`.

> Previously this repo served the model with a plain vLLM container (`serve.sh`). That
> path was removed in favor of Dynamo/SGLang — see `dynamo/` for the full setup and the
> `dynamo/README.md` for the why (vLLM ≥ 0.23.0 isn't available in any Dynamo runtime
> yet; SGLang 0.5.13.post1 is the supported engine for GLM-5.2's sparse attention).

## Model

- Architecture: `GlmMoeDsaForCausalLM` — MoE (256 routed + 1 shared experts, 8/tok),
  MLA attention, **DeepSeek-style Sparse Attention (DSA)** with an indexer
  (`index_topk=2048`), FP8 block-quant (128×128, e4m3), 78 layers, 1 MTP layer, 1M
  max context. ~756 GB of weights (≈94 GB/GPU at TP=8).
- Weights live in a shared Hugging Face cache (point `HF_CACHE` at it); they are not re-downloaded.

## Serve

`PROFILE=cache` is the default and only serving profile that has been measured
working on this hardware. `/scratch` must exist and be writable **by the
container's uid 1000, not just the host user** before the first start:

```bash
sudo mkdir -p /scratch/kvcache/glm52
sudo chown -R 1000:$(id -g) /scratch/kvcache
sudo chmod -R 2775 /scratch/kvcache        # current live state: 0777 on /scratch/kvcache/glm52
```

A fresh clone following a naive `chown $(id -u)` (host user, not uid 1000) will
hit `serve.sh`'s guard: it fails fast with a **containerized** write probe (a
throwaway container writes as uid 1000 before `docker compose up` runs), because
SGLang's file-backed hicache tier silently swallows write failures instead of
erroring — a permissions mismatch here would otherwise start the worker clean
with a dead L3 cache tier and no warning.

```bash
cd dynamo
PROFILE=cache ./serve.sh         # default: 512K context + tiered KV cache (GPU->host RAM->/scratch)
# PROFILE=longctx ./serve.sh     # attempts the model's full 1M context via HiSparse instead;
#                                 # does NOT currently start on this hardware (CUDA OOM during
#                                 # CUDA-graph capture in every attempt) -- see CONTEXT_WINDOW.md
docker compose logs -f worker    # watch startup (first boot is slow; see below)
./stop.sh                        # stop + remove the stack
```

The stack = etcd + NATS + Dynamo frontend + one SGLang worker (aggregated, TP=8,
auto-selected DSA attention backend (`flashmla_kv` on Hopper+fp8), fp8 KV cache,
**MTP/EAGLE speculative decoding on** — ~2× single-stream decode). Under the
default `PROFILE=cache`, context is served at **512K** (`--context-length
524288`) with a tiered prefix cache (GPU radix → host RAM → `/scratch`
NVMe) so that evicted prefixes can still be served from cache instead of
recomputed. The model's 1M max is not servable on one node with full
fidelity: the measured KV pool tops out at ~541K tokens alongside the
weights, and `PROFILE=longctx` — the single-node attempt at the full 1M
context via SGLang HiSparse — fails to initialize on this hardware (see
`CONTEXT_WINDOW.md` for the three measured OOM attempts). Disaggregated
prefill/decode is **not possible on a single node** for this model (a full
copy per worker exceeds 8 GPUs) — it needs ≥ 2 nodes, and so does a
full-fidelity 1M context.
Details, tunables, and benchmarks: [`dynamo/README.md`](dynamo/README.md).

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
