# Mooncake, and why this stack does not use it

**Decision: we use `--hicache-storage-backend=file`, not `mooncake`.**
Recorded 2026-09-11, when GLM-5.3-Flash was brought up with a tiered KV cache.

## What Mooncake is

Mooncake is the open-source (Apache-2.0) serving platform behind **Kimi**
(Moonshot AI): <https://github.com/kvcache-ai/Mooncake>. The design is
published as *"Mooncake: A KVCache-centric Disaggregated Architecture for LLM
Serving"* (FAST '25, <https://www.usenix.org/system/files/fast25-qin.pdf>).
Moonshot report it lets Kimi serve **75% more requests** under SLO on real
workloads. It joined the **PyTorch Ecosystem** in February 2026.

The core idea: conventional serving ties the KV cache to whichever worker
computed it, so when that worker is busy or restarts the cache is stranded.
Mooncake makes the KV cache a first-class *disaggregated* resource — pool DRAM
and SSD across many machines into one logical store, move pages over **RDMA**
(zero-copy), and let any worker reuse a prefix that any other worker computed.

Two components recur in integrations:

- **Transfer Engine** — the RDMA data path; the piece most projects adopt first.
- **Mooncake Store** — the distributed KV cache pool built on top of it.

It is not SGLang-specific, and describing it as merely "SGLang's L3 option"
undersells it. Current adoption:

| Project | Use |
|---|---|
| vLLM | Mooncake Store officially featured (2026-05); Transfer Engine as a v1 KV Connector |
| SGLang | HiCache storage backend (2025-09); EPD disaggregation transfer backend; RDMA P2P weight transfer for RL |
| TensorRT-LLM | KVCache transfer for PD-disaggregated inference |
| PyTorch | Ecosystem project |

For scale: SGLang used the Transfer Engine for P2P weight transfer on
**Kimi-K2 (1T parameters)** across thousands of GPUs, cutting weight updates
from **53s to 7.2s**.

In SGLang it is one choice for `--hicache-storage-backend`, i.e. the **L3**
tier underneath the GPU radix cache (L1) and the host-RAM tier (L2).

**The library is already present in our image** —
`/opt/sglang/lib/python3.12/site-packages/mooncake/` ships `engine.so`,
`async_store.py`, `http_metadata_server.py` and friends. What adopting it
would actually cost is not a pip install: it is a **running Mooncake service
and a config file on every serving node**, pointed at by
`SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`.

## Why upstream lists it for GLM-5.3-Flash

The SGLang cookbook's GLM-5.3-Flash page offers three HiCache settings:

| Setting | Flags |
|---|---|
| Off (default, recommended) | — |
| L1 + L2 | `--enable-hierarchical-cache --hicache-size 32` |
| + L3 | the above plus `--hicache-storage-backend mooncake` |

Upstream's own note on the L3 option is *"Start Mooncake and place the
configuration file on every serving node."* Their reference deployments are
multi-node (the GB300 recipe is 4 GPUs/node across several nodes; NVIDIA's
GLM-5 recipe is 5 nodes / 20 GPUs). **Cross-node prefix sharing is the entire
point of Mooncake**, and it is what a local directory cannot do.

## Why it buys us nothing here

This stack is **one node, 8x H200, one aggregated worker (TP=8)**. There is no
second worker and no second node to share a cache with. Mooncake's advantage
over a local directory is precisely the part we cannot use.

This is the same reasoning that removed NVIDIA Dynamo from this repo on
2026-09-11: multi-worker discovery, KV-aware routing and disaggregated
prefill/decode are all unreachable on a single node holding one model copy.
Adding Mooncake would repeat that mistake one layer down — a new service to
run, monitor and fail, in exchange for a capability the topology cannot reach.

The `file` backend, by contrast:

- is a plain directory, no service, no config file, nothing extra to supervise;
- **deduplicates across TP ranks** for MLA models — storage keys carry no
  `tp_rank` suffix, so `/scratch` holds ONE copy, not eight. That dedup is why
  the disk tier held ~85x more unique tokens than all of host RAM on GLM-5.2;
- is already proven on this hardware by the GLM-5.2 measurements in
  `sglang/RESULTS-kv-tiering.md`.

## Our configuration

`worker-flash` in `sglang/docker-compose.yml`:

```yaml
- --enable-hierarchical-cache
- --hicache-size=32                       # PER TP RANK -> ~256 GB host at TP=8
- --hicache-write-policy=write_through
- --hicache-io-backend=kernel
- --hicache-mem-layout=page_first
- --hicache-storage-backend=file
- --hicache-storage-prefetch-policy=timeout
```

with `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/scratch/kvcache/glm53`.

**The L3 directory is per-model.** GLM-5.2 uses `/scratch/kvcache/glm52` and
GLM-5.3-Flash uses `/scratch/kvcache/glm53`. Storage keys carry no model
identity, so pointing two different models at one directory would let them
collide on the same cache keys. The `file` backend has **no eviction** — see
`sglang/kv_reaper.py`, run from cron against each directory with its own byte
budget.

## Caveat: this is a local choice, not the upstream recommendation

Upstream's **verified** GLM-5.3-Flash cell has HiCache **off**, with the note
*"Keep HiCache off when GPU memory is sufficient"*, and every HiCache option is
marked **Not Verified** for this model (tracking issue:
[sgl-project/sglang#38474](https://github.com/sgl-project/sglang/issues/38474)).

On this box that guidance has real force. Measured at boot on 2026-09-11:

| | `max_total_num_tokens` |
|---|---|
| GLM-5.2-FP8 | 540,928 |
| GLM-5.3-Flash | **3,500,608** |

A **6.5x larger** GPU-resident KV pool, because only 11 of GLM-5.3-Flash's 45
layers hold a paged KV cache — the other 34 are KDA linear-attention layers
with fixed-size state — and the model is ~306 GB rather than ~756 GB.

So the tiering that clearly paid for itself on GLM-5.2 has much less to do
here: the GPU can already hold 3.5M tokens. **Benchmark the tiered
configuration against the no-HiCache baseline before assuming it helps.**
`sglang/bench_stream.py` reports `cached_tokens` directly (that is what
`--enable-cache-report` is for), which is the measurement that settles it.

## When to revisit Mooncake

Every capability above concerns moving KV **between machines or between
roles** — prefill<->decode, encoder<->language model, inference<->training.
This node has one aggregated worker at TP=8, so there is no second party to
transfer to. That is the whole argument, and it is topological, not a
judgement about Mooncake's quality.

Three concrete triggers would change it:

1. **A second node.** Cross-node prefix sharing becomes reachable. This is the
   main one, and it is the same threshold that would justify reintroducing an
   orchestration layer for KV-aware routing.

2. **Disaggregated prefill/decode.** `sglang/README.md` already records that
   this needs >= 2 nodes, since each worker holds a full model copy.

3. **Heavy multimodal traffic — relevant to THIS model specifically.**
   GLM-5.3-Flash is natively multimodal with a 24-layer vision encoder, and
   SGLang's Encode-Prefill-Decode (EPD) disaggregation uses Mooncake as the
   transfer backend to move ViT embeddings off the language-model node
   (merged 2025-12; a Mooncake-powered global multimodal embedding cache for
   cross-instance ViT embedding reuse followed in 2026-02). Upstream's
   GLM-5.3-Flash cookbook separately suggests encoder disaggregation for very
   long video on 4x GB300. We serve no image or video traffic today, so this
   is dormant — but unlike (1) and (2) it is specific to the model we are
   actually running, and would arrive with the workload rather than with new
   hardware.
