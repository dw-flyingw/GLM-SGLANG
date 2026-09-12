# HiCache on GLM-5.3-Flash: read this before trusting the benchmark

**Short version: `sglang/RESULTS-glm53-hicache-ab.md` shows HiCache costing ~38%
of sustained throughput. That number does NOT apply to this deployment's actual
workload, and HiCache should stay ON. Do not turn it off on the strength of that
file alone.**

Recorded 2026-09-12, after the A/B was run and its scope understood.

## What the benchmark actually measured

`ab_hicache.py` drives `bench_stream.py --shared-prefix-tokens 2000
--prefix-seed 42`. Every request in every arm shares **one identical 2000-token
prefix**. Reported hit rate:

```
cached tokens mean: 2099 (97.8% of prompt)
```

That ~98% is served by the **GPU radix cache**, which on this model holds
**3,687,104 tokens**. A single 2000-token prefix fits into that roughly 1800
times over, so nothing is ever evicted, and the host (L2) and disk (L3) tiers
never serve a hit the GPU tier would not have served anyway.

The tiers therefore contribute **write overhead with zero recall benefit**. The
measured penalty is HiCache's cost in the one scenario where it has no upside.
It is a valid measurement of a scenario that is not ours.

## Why this deployment is the opposite case

The workload is long agentic loops: multi-turn tool-using agents, LLM-as-judge
evaluation loops, and LangChain deep agents (which fork sub-agents from a shared
parent context). Measured on the pre-cutover stack, 35,551 real requests:

```
mean TTFT     : 16.6 s
mean duration : 39.7 s
mean output   : 1368 tokens
```

For comparison, the synthetic benchmark's 2000-token prompts had a TTFT of
**130-180 ms**. Production TTFT is ~100x that. Some of it is queueing at 64
concurrent, but prompts taking seconds to prefill are in the **100K+ token**
range -- an agent re-sending accumulated tool output, sub-agent results and
scratchpad on every iteration.

At that context size the GPU pool stops being generous:

| context per conversation | concurrent conversations before the GPU pool fills |
|---|---|
| 2K (the benchmark) | ~1,800 -- never evicts, L3 is dead weight |
| 100K | **~36** |
| 200K | **~18** |

This box was observed serving **64 inflight requests**. At agentic context sizes
that overflows the GPU cache, so entries ARE evicted -- and eviction followed by
recall is exactly what the L2/L3 tiers exist for.

Two further properties of these loops that the benchmark cannot reproduce:

- **Gaps.** Agentic loops pause between turns waiting on tool calls. Other
  traffic evicts their KV during the gap. The next turn then needs a 100K-token
  prefix that is no longer resident: recompute it, or fetch it from L3. Every
  request in the benchmark was back-to-back, so this never happened.
- **Fan-out.** Judging loops re-evaluate the same long content with different
  instructions, and deep-agent sub-agents fork from one parent context. Many
  consumers, one enormous shared prefix -- high reuse, but only if the prefix
  survives long enough to be reused.

## What the benchmark IS good for

Two findings from it do transfer, because they are not workload-dependent:

- **`--hicache-write-policy write_through_selective`, never plain
  `write_through`.** Plain write_through collapsed concurrency-1 throughput to
  109 tok/s vs 207-247 for every other config, with ITL p99 98.7 ms vs ~13 ms.
  That is synchronous per-page writes stalling decode, and it is a property of
  the write path, not of the prefix distribution.
- **The L3 tier survives a worker restart.** The `on-warm` arm recovered
  cold-start throughput markedly over `on-cold` (rep1: 653.7 vs 475.3 tok/s at
  concurrency 32, pass 1). Restarts are frequent enough here to matter.

It also established that the measurement noise in the FIRST sweep
(`RESULTS-glm53-sweep.md`, ~24% between identical configs) was caused by the L3
directory accumulating across runs, not by inherent variance. With `/scratch`
cleared between arms, repeated runs of the same config land within ~2%.

## How to settle this properly

Do not write another synthetic harness. The flash profile already carries
`--enable-metrics` and `--enable-cache-report`, so real traffic answers it
directly:

- `cached_tokens` vs `prompt_tokens` per request -- the true prefix hit rate on
  real agent loops (this is what `--enable-cache-report` is for; without it the
  field silently disappears).
- Growth and hit behaviour of `/scratch/kvcache/glm53` -- whether the disk tier
  is serving genuine recalls rather than just absorbing writes.
- The TTFT distribution -- the metric agentic loops actually feel, since a
  recalled 100K-token prefix is the difference between a fast turn and a
  full re-prefill.

Measuring the real workload beats any synthetic approximation of it.

## Upstream's position, and why we diverge knowingly

Upstream keeps HiCache **off** for GLM-5.3-Flash and marks every HiCache option
Not Verified (sgl-project/sglang#38474), on the reasoning "keep HiCache off when
GPU memory is sufficient". With a 3.5M-token pool that is sound **for short
prompts**. It stops being sound once per-conversation context reaches six
figures and concurrency is high, which is this deployment.

We diverge deliberately, with the reasoning recorded here rather than
rediscovered later.
