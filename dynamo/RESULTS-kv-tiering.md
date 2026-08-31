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

