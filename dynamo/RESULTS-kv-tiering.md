# KV cache tiering: measured results

Date: 2026-08-31
Image tag: `glm52-dynamo-sglang:0.5.13post1`

## Baseline (no hicache)

Stack state at time of measurement: all four containers (`etcd`, `nats`, `frontend`, `worker`) up and healthy (running since 2026-08-10T16:23 UTC, no restart performed for this task). `curl -s http://localhost:8000/v1/models` returned `glm-5.2-fp8` with `context_window: 524288`.

### Worker startup configuration

Command:

```bash
docker compose -f dynamo/docker-compose.yml logs worker 2>&1 | grep -iE "max_total_num_tokens|attention backend|hicache|hierarchical" | head -20
```

Output (verbatim, timestamps/log-level ANSI codes as emitted; two near-duplicate `ServerArgs` dumps are both present in the logs because sglang logs it once at `engine.__init__` and once at `engine._launch_subprocesses`):

```
2026-08-10T16:23:48.787477Z  INFO server_args._handle_model_specific_adjustments: Use dsa attention backend for DeepSeek with DSA.
2026-08-10T16:23:49.995247Z  INFO engine.__init__: server_args=ServerArgs(... attention_backend='dsa', decode_attention_backend=None, prefill_attention_backend=None, ... dsa_prefill_backend='flashmla_kv', dsa_decode_backend='flashmla_kv', dsa_topk_backend='sgl-kernel', ... enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, ...)
2026-08-10T16:23:50.007726Z  INFO engine._launch_subprocesses: server_args=ServerArgs(... attention_backend='dsa', decode_attention_backend=None, prefill_attention_backend=None, ... dsa_prefill_backend='flashmla_kv', dsa_decode_backend='flashmla_kv', dsa_topk_backend='sgl-kernel', ... enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, ...)
2026-08-10T16:29:42.674904Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.715294Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.716243Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.731129Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.731228Z  INFO scheduler.init_model_worker: max_total_num_tokens=540800, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=524288, available_gpu_mem=8.69 GB
2026-08-10T16:29:42.731446Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.768645Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.794894Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
2026-08-10T16:29:42.834393Z  INFO registry.create_tree_cache: Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
```

Full unabridged `ServerArgs(...)` lines (each ~8KB, truncated above with `...` for readability in this table) confirm:
- `max_total_num_tokens=540800`
- `attention_backend='dsa'`, `dsa_prefill_backend='flashmla_kv'`, `dsa_decode_backend='flashmla_kv'`
- `enable_hierarchical_cache=False` and `hicache_size=0` — hicache is off, matching expectations for the pre-change baseline
- Tree cache: `impl=RadixCache ... hierarchical=False` — plain GPU radix cache, no hierarchical tiering

This is the "before" state: no hicache, GPU radix cache only.

### Benchmark 1: baseline-latency (concurrency 1, num 16, max-tokens 256)

Command:

```bash
./bench_stream.py --concurrency 1 --num 16 --max-tokens 256 --tag baseline-latency
```

Output (verbatim):

```
=== bench baseline-latency pass 1/1  conc=1 num=16 max_tokens=256 ===
requests ok/err     : 16/0
wall time           : 29.47 s
TTFT  mean/p50/p99  : 148 / 144 / 205 ms
ITL   mean/p50/p99  : 16.1 / 16.1 / 16.9 ms
per-req decode tok/s: mean 150.6  (min 150.5, max 150.7)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 4096
system output tok/s : 139.0
```

Note: `cached tokens mean` of 95.5% across 16 sequential single-flight requests against the fixed default prompt is the existing GPU radix cache working as designed — later requests in the run are hitting the cached prefix laid down by earlier ones. This is expected behavior, not an anomaly.

### Benchmark 2: baseline-throughput (concurrency 32, num 128, max-tokens 256)

Command:

```bash
./bench_stream.py --concurrency 32 --num 128 --max-tokens 256 --tag baseline-throughput
```

Output (verbatim):

```
=== bench baseline-throughput pass 1/1  conc=32 num=128 max_tokens=256 ===
requests ok/err     : 128/0
wall time           : 15.70 s
TTFT  mean/p50/p99  : 605 / 291 / 1577 ms
ITL   mean/p50/p99  : 30.6 / 30.3 / 34.4 ms
per-req decode tok/s: mean 77.4  (min 69.7, max 84.1)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 32768
system output tok/s : 2087.1
```

### Benchmark 3: baseline-prefix (concurrency 1, num 1, passes 2, shared-prefix-tokens 131072, max-tokens 32)

This is the most important baseline run: it measures prefix reuse with the **GPU radix cache only**, no hicache. Later profiles must beat pass 2 of this run, not merely beat pass 1's cold prefill.

Command:

```bash
./bench_stream.py --concurrency 1 --num 1 --passes 2 --shared-prefix-tokens 131072 --max-tokens 32 --tag baseline-prefix
```

Output (verbatim):

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench baseline-prefix pass 1/2  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.90 s
TTFT  mean/p50/p99  : 16677 / 16677 / 16677 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 21.8 ms
per-req decode tok/s: mean 145.3  (min 145.3, max 145.3)
prompt tokens mean  : 135271
output tokens total : 32
system output tok/s : 1.9

=== bench baseline-prefix pass 2/2  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.85 s
TTFT  mean/p50/p99  : 644 / 644 / 644 ms
ITL   mean/p50/p99  : 17.9 / 17.6 / 22.1 ms
per-req decode tok/s: mean 157.7  (min 157.7, max 157.7)
prompt tokens mean  : 135271
cached tokens mean  : 135168 (99.9% of prompt)
output tokens total : 32
system output tok/s : 37.9

=== TTFT by pass (mean ms) ===
pass 1: 16677
pass 2: 644
pass1/pass2 speedup : 25.91x
```

Notable: pass 1 (cold, first time this exact prefix is seen) prints **no** `cached tokens mean` line at all — `bench_stream.py` only emits that line when the server reports a nonzero `cached_tokens` value, and pass 1's `usage.prompt_tokens_details.cached_tokens` was 0/absent. Pass 2, reusing the identical 131072-token prefix immediately afterward, shows 135168/135271 tokens cached (99.9%) purely from the GPU radix cache, with TTFT dropping from 16677 ms to 644 ms (25.91x speedup). This is the correct, expected result of the existing radix cache and is recorded here as the bar any hicache profile must clear — it is not a problem to fix.

## Baseline eviction-pressure test (radix cache only)

Date: 2026-08-31. This task exists because the baseline-prefix result above only proves the GPU radix cache is fast when a prefix is still resident (*warm* reuse) — it says nothing about what happens once the working set exceeds `max_total_num_tokens=540800` and a prefix is pushed out. Since the tiered cache's whole value proposition is surviving eviction from the GPU pool, we need a measured "cold again after eviction" number from the plain radix cache to compare a later hicache-enabled run against. Stack was left running (no restart) for this test, only `docker compose ps` / `docker compose logs` were used.

### Commands

```bash
cd /home/users/wrightda/src/GLM-5.2-FP8/dynamo

# 1. Prime prefix A (seed 1234). Expect COLD (~16-17 s).
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-baseline-prime

# 2. Confirm A is warm before we evict it. Expect FAST (<1 s), high cached %.
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-baseline-warm

# 3. Flood with 5 DISTINCT prefixes to push A out of the 540,800-token pool.
#    5 x 131072 = 655,360 tokens of new content, forcing A fully out.
for s in 2 3 4 5 6; do
  ./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed $s \
      --num 1 --concurrency 1 --max-tokens 32 --tag evict-baseline-flood-seed$s
done

# 4. Re-request A. THIS IS THE MEASUREMENT.
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-baseline-recheck
```

### IMPORTANT deviation from expectations: step 1 ("prime") was NOT cold

The task instructions expected step 1 to be a cold prime (~16-17 s TTFT, no cache hit), because seed 1234 was assumed to be a fresh prefix. It is not fresh: seed 1234 is the **same seed used in Benchmark 3 (`baseline-prefix`) above**, run earlier in this file. That benchmark's pass 2 already left the seed-1234 prefix resident in the GPU radix cache, and it evidently survived until this test began (no worker restart occurred between the two tests). As a result, step 1 came back **warm** (723 ms TTFT, 100% cached) instead of cold. This does not invalidate the test — step 2 independently reconfirms A is warm immediately before the flood, and step 4 is the actual measurement — but it means step 1 cannot be read as "the cold-prefill baseline for prefix A"; that number instead comes from Benchmark 3 pass 1 (16677 ms) above, and separately from the flood requests in step 3 (all of which were genuinely cold, ~16.6-17.0 s, confirming cold prefill costs ~16-17 s consistently on this server).

### Step 1: evict-baseline-prime (seed 1234) — expected cold, actually WARM

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-baseline-prime pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.94 s
TTFT  mean/p50/p99  : 723 / 723 / 723 ms
ITL   mean/p50/p99  : 17.7 / 17.6 / 21.9 ms
per-req decode tok/s: mean 145.6  (min 145.6, max 145.6)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 34.0
```

### Step 2: evict-baseline-warm (seed 1234) — confirms A resident before flood

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-baseline-warm pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.87 s
TTFT  mean/p50/p99  : 665 / 665 / 665 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 21.9 ms
per-req decode tok/s: mean 158.0  (min 158.0, max 158.0)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 36.9
```

### Step 3: flood with 5 distinct prefixes (seeds 2-6), 655,360 tokens total

```
===== SEED 2 =====
shared prefix: ~131072 tokens, seed 2, 905409 chars

=== bench evict-baseline-flood-seed2 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.86 s
TTFT  mean/p50/p99  : 16641 / 16641 / 16641 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 21.7 ms
per-req decode tok/s: mean 145.3  (min 145.3, max 145.3)
prompt tokens mean  : 135150
output tokens total : 32
system output tok/s : 1.9

===== SEED 3 =====
shared prefix: ~131072 tokens, seed 3, 905248 chars

=== bench evict-baseline-flood-seed3 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.89 s
TTFT  mean/p50/p99  : 16669 / 16669 / 16669 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 22.3 ms
per-req decode tok/s: mean 144.8  (min 144.8, max 144.8)
prompt tokens mean  : 135151
output tokens total : 32
system output tok/s : 1.9

===== SEED 4 =====
shared prefix: ~131072 tokens, seed 4, 904689 chars

=== bench evict-baseline-flood-seed4 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.93 s
TTFT  mean/p50/p99  : 16730 / 16730 / 16730 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 22.3 ms
per-req decode tok/s: mean 157.8  (min 157.8, max 157.8)
prompt tokens mean  : 135164
output tokens total : 32
system output tok/s : 1.9

===== SEED 5 =====
shared prefix: ~131072 tokens, seed 5, 905176 chars

=== bench evict-baseline-flood-seed5 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.04 s
TTFT  mean/p50/p99  : 16841 / 16841 / 16841 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 22.1 ms
per-req decode tok/s: mean 158.3  (min 158.3, max 158.3)
prompt tokens mean  : 135278
output tokens total : 32
system output tok/s : 1.9

===== SEED 6 =====
shared prefix: ~131072 tokens, seed 6, 905114 chars

=== bench evict-baseline-flood-seed6 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.05 s
TTFT  mean/p50/p99  : 16829 / 16829 / 16829 ms
ITL   mean/p50/p99  : 17.6 / 17.7 / 19.5 ms
per-req decode tok/s: mean 144.7  (min 144.7, max 144.7)
prompt tokens mean  : 135255
output tokens total : 32
system output tok/s : 1.9
```

All five flood requests were genuinely cold (no `cached tokens mean` line, TTFT ~16.6-17.0 s), confirming each of the five distinct seeds landed on fresh, never-before-cached content.

### Step 4: evict-baseline-recheck (seed 1234) — THE MEASUREMENT

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-baseline-recheck pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.97 s
TTFT  mean/p50/p99  : 16769 / 16769 / 16769 ms
ITL   mean/p50/p99  : 17.9 / 17.7 / 22.4 ms
per-req decode tok/s: mean 157.7  (min 157.7, max 157.7)
prompt tokens mean  : 135271
output tokens total : 32
system output tok/s : 1.9
```

Prefix A came back **cold**: TTFT 16769 ms, and no `cached tokens mean` line at all (server reported 0/absent cached tokens), matching the cold-prefill behavior seen in every flood request and in Benchmark 3 pass 1. Prefix A was fully evicted from the 540,800-token GPU radix-cache pool by the 655,360-token flood.

### Worker log check

```bash
docker compose logs --tail=100 worker
```

The tail=100 window covers roughly the seed-4 flood request through the step 4 recheck. Every `report_prefill_stats` line in that window shows `#cached-token: 0` for every prefill chunk, including all of step 4's prefill batches — consistent with prefix A having zero overlap with anything resident in the radix cache at recheck time. `grep -iE "evict|hicache|hierarchical|cache.*full|drop"` against the same 100-line window returned **no matches** — this sglang/dynamo build does not appear to emit an explicit "evicted node" or "cache full" log line at INFO level; the only cache-related signal available is the per-batch `#cached-token` count in `report_prefill_stats`/`report_decode_stats`, which is what was used above as evidence of eviction. No errors, warnings, or anomalies were present in the log window.

### Summary table

| Step | Tag | TTFT (ms) | Cached tokens | Cached % |
|---|---|---:|---:|---:|
| 1 (prime, expected cold — actually warm, see deviation note) | evict-baseline-prime | 723 | 135232 / 135271 | 100.0% |
| 2 (confirm warm) | evict-baseline-warm | 665 | 135232 / 135271 | 100.0% |
| 4 (recheck after 655,360-token flood — THE MEASUREMENT) | evict-baseline-recheck | 16769 | 0 / 135271 (no cache-hit line reported) | 0% |

For reference, the flood requests (seeds 2-6, all genuinely cold, not resident before the flood) ranged 16641-16841 ms TTFT — essentially identical to step 4's 16769 ms, reinforcing that step 4 was a full cold prefill, not a partial cache hit.

**Conclusion:** Prefix A (seed 1234) was successfully and fully evicted from the GPU radix cache by flooding with 655,360 tokens (5 x 131,072) of distinct content against the 540,800-token pool. Re-requesting A after the flood cost the same ~16.8 s cold-prefill penalty as a never-before-seen prefix, with 0% cache reuse. This is the baseline the KV-cache tiering feature must improve on: a later task will re-run this identical prime/warm/flood/recheck sequence with tiering enabled and should show step 4 landing much closer to the ~650 ms warm numbers in steps 1/2 rather than the ~16.8 s cold number measured here.

