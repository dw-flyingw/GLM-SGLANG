# Tiered KV cache for GLM-5.2-FP8: `/scratch` as an L3 prefix-cache tier

**Date:** 2026-08-31
**Status:** Approved, not yet implemented
**Scope:** `dynamo/docker-compose.yml`, `dynamo/bench_stream.py`, a new `/scratch` reaper, and doc corrections.

## Problem

The stack serves GLM-5.2-FP8 on 8× H200 with a GPU-only radix prefix cache. When a prefix is
evicted it is recomputed from scratch — at 512K context that is seconds of H200 prefill per
turn. Three workloads all pay this cost:

1. Long-context multi-turn / agentic (same huge prefix hit every turn)
2. Many users sharing system prompts / RAG boilerplate
3. Batch passes over a fixed corpus

Meanwhile the host has **2.26 TB of RAM sitting idle** and **28 TB of empty NVMe at `/scratch`**.
Neither is used by the serving stack today.

## Findings (verified against the running image, not assumed)

All paths below are inside `dynamo-worker-1`, under
`~/.local/lib/python3.12/site-packages/sglang/` (SGLang 0.5.13.post1).

### Hardware

| | |
|---|---|
| `/scratch` | 28 TB XFS on `vgdata-lvdata` (4× 7.68 TB NVMe: `nvme4/6/7/8n1`), empty, root-owned |
| `/data` | 28 TB XFS on `vgdata1-lvdata1` (4× NVMe), holds the HF weight cache |
| Host RAM | 2267 GB total, 2168 GB available |
| CPU | 2× AMD EPYC 9535, 256 threads, 2 NUMA nodes |

`/scratch` and `/data` are on **disjoint** NVMe devices, so KV-cache I/O will not contend with
weight loading.

### HiCache is present and nothing in our config disqualifies it

`srt/server_args.py` exposes `--enable-hierarchical-cache`, `--hicache-ratio` (default 2.0),
`--hicache-size` (default 0), `--hicache-write-policy` (default `write_through`),
`--hicache-io-backend` (default `kernel`), `--hicache-mem-layout` (default `layer_first`),
`--hicache-storage-backend` (`file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic`, `eic`,
`simm`), `--hicache-storage-prefetch-policy` (default `timeout`).

The disqualifying guards are diffusion-LLM inference, `--disable-radix-cache`
(`server_args.py:4180`), and optimistic prefill — none apply. `_resolve_io_decode_attention_compatibility`
downgrades `kernel` I/O to `direct` **only when the effective decode backend is `fa3`**; we run
the auto-selected `dsa`/`flashmla_kv`, so `kernel` I/O survives.

**MTP/EAGLE speculative decoding is not on any incompatibility list.**

### HiCache has first-class DSA support

`srt/mem_cache/memory_pool_host.py` defines `MLATokenToKVPoolHost` (line 899) and
`DSAIndexerPoolHost` (line 2720, *"Host-side DSA index buffers only. Slot layout matches the
anchor MLA host pool"*). `hiradix_cache.py:92` selects the MLA host pool for `MLATokenToKVPool`,
which `DSATokenToKVPool` subclasses. The DSA indexer cache is mirrored to host as a separate,
additional allocation.

### The L2 host pool is replicated 8× under TP=8; the L3 storage tier is not

`HostKVCache.__init__` (`memory_pool_host.py:229`) runs **per TP rank** and reads `host_size` as
that rank's own pool: `self.size = int(host_size * 1e9 // self.size_per_token)`. With TP=8 the
host pool is allocated eight times, even though MLA makes all eight copies bit-identical. It
hard-fails at startup if it over-requests (`"Not enough host memory available"`), so
over-sizing is fail-safe.

The storage tier behaves differently. `hicache_storage.py:335`:

```python
if not is_mla_model:
    self.config_suffix += f"_{tp_rank}_{tp_size}"
```

For an MLA model the storage key carries **no rank suffix**, so all 8 ranks address one key and
`/scratch` holds **one copy, not eight**.

This is the crux of the design: `/scratch` stores ~85× more *unique* cached tokens than the
entire 2.26 TB of host RAM can, purely because of that dedup.

### HiSparse exists, targets this exact model, and excludes HiCache

`srt/arg_groups/hisparse_hook.py`:

- `validate_hisparse` asserts the model is DSA, with the error text naming
  *"DeepSeek V3.2, GLM-5"* as supported.
- With `--kv-cache-dtype fp8_e4m3` it selects `flashmla_kv` — the backend the stack already
  auto-selects.
- It asserts `--disable-radix-cache`.

Since `server_args.py:4180` raises if `--enable-hierarchical-cache` is combined with
`--disable-radix-cache`, **HiCache and HiSparse are mutually exclusive**.

`hisparse_memory_pool.py:38` (`HiSparseDSATokenToKVPool`) passes
`index_buf_size = size * host_to_device_ratio`: the DSA **indexer** is GPU-resident at the full
logical size while only a `size`-token working set of the heavy MLA latent KV stays on GPU, the
remainder paging from host RAM. HiSparse contains **no storage backend** — no `file`, `mmap`, or
`storage` references — so it tiers GPU↔host RAM only and `/scratch` plays no part in it.

## KV sizing

From `config.json`: `kv_lora_rank 512`, `qk_rope_head_dim 64`, `index_head_dim 128`,
`num_hidden_layers 78`, `index_topk 2048`, `max_position_embeddings 1048576`.

Per token, per rank (MLA latent is replicated across TP, not sharded), at fp8:

| Component | B/token/layer | × 78 layers |
|---|---|---|
| MLA latent KV (512 + 64) | 576 | 44,928 B (43.9 KiB) |
| DSA indexer (128 + 4 B scale) | 132 | 10,296 B (10.1 KiB) |
| **Total** | **708** | **55,224 B (53.9 KiB)** |

Cross-check: 540,800 tokens × 55,224 B = **29.9 GB/GPU**, consistent with the observed
`max_total_num_tokens=540800` at `--mem-fraction-static 0.85`.

Add one layer (79) when the MTP layer's KV is allocated; this shifts every figure by ~1.3% and
changes no conclusion.

### Resulting tiers

| Tier | Unique tokens | Cost | vs GPU pool |
|---|---|---|---|
| GPU device pool | 540.8K | 29.9 GB × 8 | 1× |
| L2 host RAM (`--hicache-size 96`/rank) | 2.14M | ~944 GB (8 copies) | 3.9× |
| L3 `/scratch`, 10 TB budget | 181M | 10 TB of 28 TB | 335× |

L2 detail: 96e9 / 44,928 = **2,136,752 tokens/rank**; the `DSAIndexerPoolHost` adds
2,136,752 × 10,296 = 22.0 GB/rank; per-rank total ~118 GB; **×8 = ~944 GB**, leaving ~1.22 TB
headroom of the 2168 GB available.

L3 detail: 10e12 / 55,224 = **181M tokens**. The full 28 TB would hold 507M.

## Architecture

Two selectable profiles in one `docker-compose.yml`. They cannot coexist (HiCache XOR HiSparse),
so this is a switch, not a merge.

**Switching mechanism:** Compose cannot vary a service's `command:` by env var, so the two
profiles are two *service definitions* — `worker` and `worker-longctx` — sharing a YAML anchor
for the common args and gated by native Compose profiles:

```yaml
worker:          { profiles: ["cache"],   ... }
worker-longctx:  { profiles: ["longctx"], ... }
```

`serve.sh` gains `PROFILE=${PROFILE:-cache}` and invokes
`docker compose --profile "$PROFILE" up -d`. `etcd`, `nats`, and `frontend` carry no `profiles:`
key, so they start under either. Exactly one worker ever runs; the GPUs cannot host both.

```
PROFILE=cache  (default)              PROFILE=longctx
┌──────────────────────────┐          ┌──────────────────────────┐
│ GPU: radix + KV 540K     │          │ GPU: indexer @ full size │
│      ~29.9 GB/GPU        │          │    + MLA KV working set  │
├──────────────────────────┤          ├──────────────────────────┤
│ L2 host RAM ~2.1M tok    │          │ host RAM ~41 GB          │
│    (8× replicated)       │          │    (paged per decode)    │
├──────────────────────────┤          └──────────────────────────┘
│ L3 /scratch ~181M tok    │           context ~911K, NO prefix
│    deduped, persistent   │           cache, /scratch unused
└──────────────────────────┘
 context 512K, full reuse
```

## Profile A (`cache`) — the default

Worker args added to `docker-compose.yml`:

```yaml
- --enable-hierarchical-cache
- --hicache-size=${HICACHE_GB:-96}          # per TP rank, decimal GB
- --hicache-write-policy=write_through
- --hicache-io-backend=kernel               # survives: decode backend is dsa/flashmla_kv, not fa3
- --hicache-mem-layout=page_first           # page locality for the storage tier
- --hicache-storage-backend=file
- --hicache-storage-prefetch-policy=timeout
```

Worker environment and mount:

```yaml
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR: "${KV_SCRATCH_DIR:-/scratch/kvcache/glm52}"
volumes:
  - "${KV_SCRATCH_ROOT:-/scratch/kvcache}:${KV_SCRATCH_ROOT:-/scratch/kvcache}"
```

(`HiCacheFile.__init__`, `hicache_storage.py:322`, reads that env var and otherwise defaults to
`/tmp/hicache` — which would land on the 446 GB root filesystem. Setting it is mandatory.)

`page_first` + `kernel` is a stable combination: the layout resolver rewrites `page_first` →
`page_first_direct` only for `direct` I/O (`server_args.py:3798`), and `page_first_direct` +
`kernel` → `direct` (`server_args.py:3789`). Neither fires here. Confirm from the startup log
rather than trusting this.

Everything else is untouched: TP=8, MTP/EAGLE, `--page-size 64`, `--context-length 524288`,
`--kv-cache-dtype fp8_e4m3`, DSA auto-selected backend.

## `/scratch` provisioning

`/scratch` is root-owned and `sudo` on this host requires a password, so this step is run by the
user, not by the implementation:

```bash
sudo mkdir -p /scratch/kvcache/glm52
sudo chown -R $(id -u):$(id -g) /scratch/kvcache
```

## The reaper (in scope)

The `file` backend writes one `.bin` per page component and has **no eviction and no size cap**.
Left alone it grows until `/scratch` fills. A reaper is therefore part of this work, not a
follow-up.

Design: a standalone script (`dynamo/kv_reaper.sh`) enforcing a **byte budget**, invoked by a
systemd timer or cron every 15 minutes.

- Budget `KV_CACHE_MAX_BYTES`, default 10 TB (of 28 TB, leaving room for other `/scratch` use).
- When the tree exceeds budget, delete oldest-first by **mtime** until under it.
- Use mtime, not atime: `/scratch` is mounted `relatime`, so atime only advances if it is older
  than mtime or older than 24h — too coarse for LRU. mtime approximates insertion order, which
  is an acceptable proxy given the cache is regenerable by definition.
- Never follow symlinks, never delete outside `KV_SCRATCH_ROOT`, and no-op if that path is unset
  or not a directory.
- Log bytes reclaimed and files removed so cache pressure is observable.

Deleting a page file under a live worker is safe: the storage tier is a cache, and a miss falls
back to L2/recompute.

## Profile B (`longctx`) — the ~1M attempt

```yaml
- --enable-hisparse
- --disable-radix-cache
- --hisparse-config={"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}
- --context-length=1048576
```

`top_k: 2048` mirrors the model's own `index_topk`. `device_buffer_size` starts at 4096 (the
value in SGLang's own `--hisparse-config` help text) and is tuned from there: it is the
per-request GPU staging buffer for selected pages, so it must be at least `top_k` and larger
values trade GPU memory for fewer host round-trips. Tuning it is part of the Profile B
benchmark, not a prerequisite for starting.

Projected sizing, from `index_buf_size = size × host_to_device_ratio` against the current
~29.9 GB device budget:

```
size × 44,928  +  2 × size × 10,296  =  29.9e9
size × 65,520 = 29.9e9   →   size ≈ 456K,  size_full ≈ 911K
```

So ~911K tokens at the present `--mem-fraction-static 0.85`. Reaching the model's full
1,048,576 needs `size = 524,288`, i.e. 524,288 × 65,520 = **34.4 GB/GPU**, about 4.5 GB more —
roughly `--mem-fraction-static 0.88`. That is inside the safe band (`dynamo/README.md` puts the
OOM-at-graph-capture risk near 0.93). Host side needs only ~41 GB of the 2.1 TB free.

**These are projections from source arithmetic, not measurements.** Two things the source does
not settle and the benchmark must:

1. Whether MTP/EAGLE composes with HiSparse. Nothing forbids it; nothing exercises it either.
   If it conflicts, Profile B loses the ~2× single-stream decode that MTP provides.
2. The per-decode-step cost of paging MLA KV from host over PCIe.

Profile B's cost is unambiguous and large: `--disable-radix-cache` means **no prefix reuse at
all**, so every request re-prefills from zero. It is the wrong default for all three workloads
above, which is exactly why it is a switch.

## Verification

`bench_stream.py` (stdlib-only, no tokenizer, no network) is the harness — `aiperf`/`genai-perf`
are absent from the image and `sglang.bench_serving` needs HF Hub.

### Required addition to `bench_stream.py`

It currently cannot measure the benefit Profile A exists to deliver. Add a shared-prefix mode:

- `--shared-prefix-tokens N` — prepend an identical synthetic N-token prefix to every request.
- `--passes K` — run the request set K times against the same endpoint.
- Report TTFT per pass separately, so pass 1 (cold) vs pass 2 (warm) is legible.

Generate the prefix deterministically from a seed so a re-run after a worker restart reproduces
the *same* prefix — that is what makes the persistence claim testable.

### Measurement sequence

1. Baseline on the current config: `--concurrency 1 --num 16` and `--concurrency 32 --num 128`,
   plus the new shared-prefix run.
2. Profile A, same three runs.
3. Restart the worker, re-run the shared-prefix run **without** repopulating.
4. Profile B: a single >524,288-token request, plus the two standard runs.

### Success criteria

**Profile A ships if all three hold:**

- **No cold-path regression.** Concurrency-32 system tok/s within 5% of the 1975.0 tok/s
  baseline, and concurrency-1 decode within 5% of 150.0 tok/s. Tiering must be free when it
  misses.
- **Warm-prefix TTFT drops materially.** Pass-2 TTFT at `--shared-prefix-tokens 131072` at least
  5× better than pass 1. (A 128K prefill is seconds; an L2/L3 fetch should not be.)
- **The cache survives a restart.** After step 3, pass-1 TTFT is close to warm, not cold. This is
  the claim host RAM cannot make and is `/scratch`'s entire justification — if it fails, the L3
  tier is not earning its place and the design collapses back to L2-only.

**Profile B is recorded as viable if:**

- A request longer than 524,288 tokens completes and returns coherent output. That alone
  falsifies the "≥2 nodes" claim in `CONTEXT_WINDOW.md`.
- Its decode throughput is logged honestly beside Profile A's, including whether MTP survived.

Profile B is not required to beat Profile A on throughput — it trades throughput for reach, and
the write-up should say so.

## Risks and rollback

| Risk | Mitigation |
|---|---|
| Host RAM exhaustion from the 8× L2 replication | 96 GB/rank leaves ~1.22 TB headroom; SGLang hard-fails at startup rather than OOMing mid-serve |
| `/scratch` fills | The reaper, with a 10 TB budget against 28 TB |
| `page_first`/`kernel` silently rewritten | Assert the effective layout and I/O backend from the startup log |
| Storage-tier writes stall decode | `--hicache-storage-prefetch-policy timeout`; the no-cold-path-regression criterion catches it |
| 8 ranks writing one deduped key | Watch for write amplification or contention on the shared key in the concurrency-32 run |
| Profile B changes behavior | Off by default, selected explicitly by `PROFILE=longctx` |

Profile A is additive: remove the flags and `./serve.sh` returns to today's configuration
exactly. The JIT cache volume is untouched, so rollback restarts are fast.

## Documentation to correct

- `CONTEXT_WINDOW.md` — states 1M requires ≥2 nodes. If Profile B verifies, that is wrong; it
  requires HiSparse and the sacrifice of prefix caching. Rewrite as a trade, not a hardware limit.
- `README.md` and `dynamo/README.md` — same claim, plus a new section on the tiering and the
  `PROFILE` switch.

Correct these **only after** the corresponding benchmark passes.

## Explicitly out of scope

- `nixl` (GPUDirect Storage, DMA NVMe→GPU bypassing the host bounce) and `hf3fs` as L3 backends.
  Both are plausible upgrades over the naive `file` backend, and `SGLANG_HICACHE_NIXL_USE_DIRECT_IO`
  defaults to true, but they are optimizations of a path that must first be shown to work.
- Multi-node disaggregation and KV-aware routing — unchanged ≥2-node levers.
- Moving the `dynamo-jit-cache` volume off the root filesystem. Unrelated to KV caching.
- Precomputing a corpus KV cache offline for workload 3. It becomes attractive once Profile A's
  persistence criterion passes, but it is a separate piece of work.
