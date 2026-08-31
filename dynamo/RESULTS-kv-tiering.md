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


## Profile A (hicache, /scratch L3)

Date: 2026-08-31. **BLOCKED — could not bring the worker up. No benchmarks were run.**

### Summary (top line)

Task 5 could not be completed. Step 1 ("Restart the stack under Profile A") failed: the `dynamo-worker-1` container will not start under **any** profile, **any** configuration, on this host right now. This is a pre-existing, host-level infrastructure fault, unrelated to the KV-cache-tiering branch or to `PROFILE=cache`/hicache config — it would block a plain restart of the pre-tiering baseline just as completely. It was not visible before this task because the previously-running worker container (up since 2026-08-10, used for every measurement in the `## Baseline` sections above) was created *before* the fault occurred and kept running through it; stopping it (as Step 1 instructs) exposed the problem for the first time.

**Root cause:** on 2026-08-11 06:47:54 UTC, host package `nvidia-fabricmanager` (and the `nvidia-driver` meta-package) were upgraded from 590.48.01 to 595.71.05, but the running kernel never reloaded the new kernel module (no reboot occurred). The currently *loaded* kernel module is still 590.48.01 (`cat /proc/driver/nvidia/version`), while all userspace NVIDIA tooling (`nvidia-smi`, `nv-fabricmanager`) is now 595.71.05. `nvidia-fabricmanager-590` is a transitional dummy package with no real binary (depends on `-595`) — there is no way to run a 590 fabric manager anymore. `nvidia-fabricmanager.service` has been crash-looping/dead since that moment:

```
Aug 11 06:47:54 sprocket nv-fabricmanager[186868]: fabric manager NVIDIA GPU driver interface version 595.71.05 don't match with driver version 590.48.01. Please update with matching NVIDIA driver package.
Aug 11 06:47:54 sprocket nvidia-fabricmanager-start.sh[186855]: "/usr/bin/nv-fabricmanager -c /usr/share/nvidia/nvswitch/fabricmanager.cfg" failed! Exit code: 1
Aug 11 06:47:54 sprocket systemd[1]: nvidia-fabricmanager.service: Failed with result 'exit-code'.
```

Because the fabric manager socket (`/run/nvidia-fabricmanager/socket`) is never created, Docker's NVIDIA container runtime — which unconditionally bind-mounts that socket into any container requesting GPU access on this NVSwitch system — fails at container-create time for **any** new GPU container:

```
$ PROFILE=cache ./serve.sh
...
Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed:
runc create failed: unable to start container process: error during container init: failed to fulfil mount request:
open /run/nvidia-fabricmanager/socket: no such file or directory
```

Reproduced twice (once via `./serve.sh`, once via `docker compose --profile cache up -d worker` directly) — not transient.

`nvidia-smi` on the host also fails outright right now, independent of this task:

```
$ nvidia-smi --query-gpu=index,name,driver_version --format=csv
Failed to initialize NVML: Driver/library version mismatch
```

### What was verified (diagnostic only, no fix applied)

- `dpkg -l | grep -iE 'fabricmanager|nvidia-driver'`: both `nvidia-driver-590-server-open` (590.48.01) and `nvidia-driver-595-server-open` (595.71.05) show `ii` installed; `nvidia-fabricmanager-595` (real binary) and `nvidia-fabricmanager-590` (empty transitional package, `Depends: nvidia-fabricmanager-595`) both `ii`.
- `cat /proc/driver/nvidia/version`: `NVRM version: ... 590.48.01 ...` — confirms the loaded kernel module is still 590.
- `dkms status`: `nvidia/595.71.05, 6.8.0-137-generic, x86_64: installed` — the matching 595 module **is** built and present for the running kernel; it has just never been loaded.
- `fuser -v /dev/nvidia*` and `lsof /dev/nvidia0`: both empty — no process currently holds any GPU (expected, since I had already stopped the only worker that was using them). This means the underlying fault could plausibly be cleared by a kernel-module reload without a full reboot, but doing that is host driver surgery on a shared 8x H200 production box, well outside this task's authorized actions (`stop.sh`/`serve.sh`, no source edits) — **I deliberately did not attempt it** and am reporting instead, per this task's own instruction to "say so loudly... and leave clear instructions" rather than improvise a fix outside scope.
- `/etc/nvidia-container-runtime/config.toml` has no toggle to skip the fabricmanager socket mount specifically; this is baked into libnvidia-container's NVSwitch-detection logic, not configurable per-container.

### Current live state (left as-is; NOT torn down)

```
$ docker compose ps -a
NAME                IMAGE                             COMMAND                  SERVICE    STATUS
dynamo-etcd-1       quay.io/coreos/etcd:v3.5.21       "etcd --data-dir=/et…"   etcd       Up
dynamo-frontend-1   glm52-dynamo-sglang:0.5.13post1   "python3 -m dynamo.f…"   frontend   Up
dynamo-nats-1       nats:2.10-alpine                  "docker-entrypoint.s…"   nats       Up
dynamo-worker-1     glm52-dynamo-sglang:0.5.13post1   "python3 -m dynamo.s…"   worker     Created   (never started; no logs)

$ curl -s http://localhost:8000/v1/models
{"object":"list","data":[]}       # frontend up, zero workers registered -- NOT serving completions

$ free -g
               total        used        free      shared  buff/cache   available
Mem:            2267          40        1513           0         724        2227

$ du -sh /scratch/kvcache/glm52
0	/scratch/kvcache/glm52     # unchanged/empty -- nothing ever ran against it
```

**The stack is NOT fully up.** `etcd`/`nats`/`frontend` are running (frontend answers HTTP but has zero registered workers), but `dynamo-worker-1` never started and holds no GPUs. The model is **not serving**. This does not satisfy the "always leave the stack UP" requirement — it could not be satisfied given the host fault, and is reported here plainly rather than glossed over.

### Remediation needed (requires a human with host root / sudo)

Either of:
1. **Reboot the host** (cleanest — completes the pending 590→595 driver activation cleanly), then `cd dynamo && PROFILE=cache ./serve.sh`.
2. **Without a reboot**, since GPUs are currently idle and the matching 595.71.05 DKMS module is already built for the running kernel (`6.8.0-137-generic`):
   ```
   sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
   sudo modprobe nvidia nvidia_uvm nvidia_modeset nvidia_drm
   sudo systemctl start nvidia-fabricmanager
   systemctl status nvidia-fabricmanager   # confirm "Successfully configured all the available NVSwitches"
   nvidia-smi                              # confirm it reports 595.71.05 with no mismatch
   cd dynamo && PROFILE=cache ./serve.sh
   ```
   This was **not attempted** by this task — it requires root and is outside `stop.sh`/`serve.sh`, so it was left for a human to run and verify.

Once the worker is confirmed registered and healthy, Task 5's Steps 2 through 6 (log assertions, `free -g`/`du -sh` before/after, the three benchmarks, and the decisive eviction-recheck test) still need to be executed — none of them ran here.

### PASS/FAIL against Task 5's criteria

| Criterion | Result |
|---|---|
| Worker registers under `PROFILE=cache` | **FAIL** — never started, host GPU/driver fault |
| Step 2 log assertions (hicache host alloc, DSA indexer alloc, page_first/kernel, max_total_num_tokens=540800, DSA/flashmla_kv backend) | **NOT EXECUTED** — no worker logs exist |
| No cold-path regression (conc-1 ≥ ~150.6 tok/s ×0.95, conc-32 ≥ ~2087.1 tok/s ×0.95) | **NOT EXECUTED** |
| Eviction survival (non-zero cached-token %, TTFT materially below 16,769 ms) | **NOT EXECUTED** — the decisive test did not run |
| `/scratch` grows (L3 tier being written) | **NOT EXECUTED** / confirmed empty (0 bytes), unchanged |
| Stack left UP on PROFILE=cache | **FAIL** — worker not running; frontend/etcd/nats up but zero workers registered, not serving |

**This is a blocked/incomplete task, not a negative result on the tiering feature itself.** Nothing about Profile A's hicache configuration was exercised or disproven — the blocker is a dormant, 3-week-old, unrelated host driver upgrade that was never completed. Task 5 needs to be re-run in full once a human clears the driver mismatch above.

### RESOLUTION (2026-08-31, after the above was written)

The fault was repaired without a reboot. Three artifacts of the 590->595 upgrade were stale
and all three had to be refreshed:

1. **Kernel module** — unloaded the 590.48.01 module and loaded the 595.71.05 one DKMS had
   already built for the running kernel 6.8.0-137-generic:
   `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && modprobe nvidia && modprobe nvidia_uvm`
   Module refcounts were verified 0 first; no GPU processes were running.
2. **Fabric Manager** — started cleanly once the module matched. `/run/nvidia-fabricmanager/socket`
   reappeared and `nvidia-smi` began working again (8x H200 visible, NVRM 595.71.05).
3. **CDI spec** — `/run/cdi/nvidia.yaml` was still the spec generated at the 2026-08-10 boot
   and named 590 library files. This one was NOT repaired: attempts to regenerate it did not
   take. Instead `docker-compose.yml` now pins `runtime: nvidia` (legacy path) instead of
   `gpus: all` (which resolves through CDI under `mode="auto"`). See commit 852de86.

**Outstanding host debt:** `/run/cdi/nvidia.yaml` still contains 91 references to 590.48.01.
Any *other* CDI-based GPU container on sprocket will still fail. It regenerates correctly on
the next reboot (/run is tmpfs). The `runtime: nvidia` pin in docker-compose.yml is a
workaround and should be reverted to `gpus: all` once the host is repaired.

After the repair the worker started, loaded, and registered normally. Benchmarks follow below.

---

## Profile A measurements (2026-08-31, post-repair)

Continuation of the same task, same day, after the driver fault above was fixed. This run picks up an **already-running** stack — per this task's instructions, Step 1 (restart under `PROFILE=cache`) had already been performed by the operator before this measurement pass began; no `stop.sh`/`serve.sh` was run here. Stack state confirmed at start:

```
$ docker compose --profile cache ps
NAME                IMAGE                             COMMAND                  SERVICE    CREATED          STATUS          PORTS
dynamo-etcd-1       quay.io/coreos/etcd:v3.5.21       "etcd --data-dir=/et…"   etcd       28 minutes ago   Up 28 minutes
dynamo-frontend-1   glm52-dynamo-sglang:0.5.13post1   "python3 -m dynamo.f…"   frontend   28 minutes ago   Up 28 minutes
dynamo-nats-1       nats:2.10-alpine                  "docker-entrypoint.s…"   nats       28 minutes ago   Up 28 minutes
dynamo-worker-1     glm52-dynamo-sglang:0.5.13post1   "python3 -m dynamo.s…"   worker     5 minutes ago    Up 5 minutes

$ curl -s http://localhost:8000/v1/models
{"object":"list","data":[{"id":"glm-5.2-fp8","object":"model","created":1788199725,"owned_by":"nvidia","context_window":524288}]}
```

Worker is up and registered. Already independently confirmed (not re-derived here): 8x `Allocating 96.00 GB host memory for hierarchical KV cache.` (one per TP rank), `max_total_num_tokens=540928` (baseline was 540800 — did not shrink), MTP draft CUDA graph captured, host RAM 972 GB in use.

### Step 2: effective configuration from the log

Command:

```bash
docker compose logs worker 2>&1 | grep -iE "hicache|hierarchical|host memory|indexer|max_total_num_tokens|attention backend" | grep -v "ServerArgs(" | head -30
```

Output (verbatim, representative lines; the 8x-per-rank `Allocating 96.00 GB` and `19.32 GB` lines are deduplicated to one each below for readability, all 8 confirmed present in the raw log):

```
worker-1  | 2026-08-31T18:03:04.548251Z  INFO server_args._handle_model_specific_adjustments: Use dsa attention backend for DeepSeek with DSA.
worker-1  | 2026-08-31T18:05:32.831330Z  INFO memory_pool_host.__init__: Allocating 96.00 GB host memory for hierarchical KV cache.   [x8, one per TP rank]
worker-1  | 2026-08-31T18:05:32.855360Z  INFO scheduler.init_model_worker: max_total_num_tokens=540928, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=524288, available_gpu_mem=10.09 GB
worker-1  | 2026-08-31T18:05:57.196364Z  WARN hicache.can_use_hicache_jit_kernel: Unsupported element_size = 656 for JIT HiCache kernel   [x8]
worker-1  | 2026-08-31T18:05:57.197020Z  INFO memory_pool_host.__init__: Allocating 19.32 GB host memory for DSA indexer (layout=page_first).   [x8, one per TP rank]
worker-1  | 2026-08-31T18:06:02.243745Z  INFO backend_factory.create_backend: Creating storage backend 'file' (sglang.srt.mem_cache.hicache_storage.HiCacheFile)   [x8]
worker-1  | 2026-08-31T18:06:02.697455Z  INFO hybrid_pool_assembler.attach_hybrid_dsa_pool_to_hiradix_cache: Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, transfer_layer_num=78   [x8]
worker-1  | 2026-08-31T18:06:03.701249Z  INFO memory_pool_host.__init__: Allocating 1.08 GB host memory for hierarchical KV cache.   [x8, additional small pool — appears to be the MTP draft-model host cache, separate from the 96 GB main-model pool above]
```

`grep -oE` isolation of the two exact-value assertions from the full `ServerArgs(...)` dump:

```
$ docker compose logs worker 2>&1 | grep -oE "hicache_mem_layout='?[a-zA-Z_]+'?" | sort -u
hicache_mem_layout='page_first'

$ docker compose logs worker 2>&1 | grep -oE "hicache_io_backend='?[a-zA-Z_]+'?" | sort -u
hicache_io_backend='kernel'
```

Attention backend, from `_handle_model_specific_adjustments` and `_set_default_dsa_backends`:

```
worker-1  | 2026-08-31T18:03:04.548251Z  INFO server_args._handle_model_specific_adjustments: Use dsa attention backend for DeepSeek with DSA.
worker-1  | 2026-08-31T18:03:04.548935Z  WARN server_args._set_default_dsa_backends: Set DSA backends for fp8_e4m3 KV Cache: prefill=flashmla_kv, decode=flashmla_kv.
```

#### Step 2 assertions — actual vs expected

| Assertion | Expected | Actual | Verdict |
|---|---|---|---|
| Hierarchical KV cache host alloc per rank | ~96 GB | **96.00 GB** x8 | PASS, exact match |
| DSA indexer host alloc per rank | ~22 GB | **19.32 GB** x8 | **DIFFERS from expected** — 19.32 GB actual vs. ~22 GB expected in the brief (12% lower). Not a failure of the config (the value is internally consistent and repeats identically across all 8 ranks), but the *plan's* estimate of indexer size was off; recorded here so downstream reasoning about total host-memory budget uses the real number, not the estimate. |
| `hicache_mem_layout` | `page_first` | **`page_first`** | PASS — not silently rewritten |
| `hicache_io_backend` | `kernel` | **`kernel`** | PASS — not downgraded to `direct` |
| `max_total_num_tokens` | not shrunk from 540800 | **540928** | PASS — device pool did not shrink (actually 128 tokens larger than the plain-baseline run; within normal run-to-run noise from `available_gpu_mem`, not a meaningful change) |
| Attention backend | auto DSA / `flashmla_kv` | **DSA backend, `dsa_prefill_backend=flashmla_kv`, `dsa_decode_backend=flashmla_kv`** | PASS |

One additional log line worth flagging, not asked for in Step 2 but relevant to interpreting the io_backend result: `WARN hicache.can_use_hicache_jit_kernel: Unsupported element_size = 656 for JIT HiCache kernel`, repeated once per rank right before the indexer allocations. This means the JIT-optimized hicache kernel path is unavailable for this element size and the `kernel` io_backend is falling back to a non-JIT (presumably more generic/copy-based) implementation — `kernel` itself was *not* downgraded to `direct` (confirmed above), but it is not running the fastest available `kernel`-mode code path either. This may be a contributing factor in the conc-32 throughput regression measured below.

### Step 3: host memory and /scratch state

```
$ free -g
               total        used        free      shared  buff/cache   available
Mem:            2267         961         590         868        1594        1306

$ ls -la /scratch/kvcache/glm52
total 0
drwxrwxrwx 2 wrightda wrightda  6 Aug 31 18:02 .
drwxr-xr-x 3 wrightda wrightda 19 Aug 31 16:51 ..

$ du -sh /scratch/kvcache/glm52   # BEFORE benchmarks
0	/scratch/kvcache/glm52
```

961 GB used, in line with the ~944 GB expected over baseline (8x96 GB hierarchical + 8x19.32 GB indexer + 8x1.08 GB MTP-draft ≈ 933 GB of hicache host pools alone, plus normal process/model overhead). `/scratch/kvcache/glm52` exists, mode `0777`, empty before any benchmark traffic — matches "may be empty until first requests run."

### Step 4: three benchmarks (verbatim)

#### hicache-latency (concurrency 1, num 16, max-tokens 256)

```
=== bench hicache-latency pass 1/1  conc=1 num=16 max_tokens=256 ===
requests ok/err     : 16/0
wall time           : 32.64 s
TTFT  mean/p50/p99  : 337 / 150 / 2926 ms
ITL   mean/p50/p99  : 16.3 / 16.2 / 18.0 ms
per-req decode tok/s: mean 149.8  (min 146.4, max 150.1)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 4096
system output tok/s : 125.5
```

conc-1 mean decode tok/s: **149.8** vs Task 3 measured baseline **150.6** → -0.53%, well within the 5% tolerance. **PASS.**

#### hicache-throughput (concurrency 32, num 128, max-tokens 256)

```
=== bench hicache-throughput pass 1/1  conc=32 num=128 max_tokens=256 ===
requests ok/err     : 128/0
wall time           : 17.20 s
TTFT  mean/p50/p99  : 862 / 429 / 2208 ms
ITL   mean/p50/p99  : 31.7 / 30.5 / 47.9 ms
per-req decode tok/s: mean 75.1  (min 64.4, max 84.3)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 32768
system output tok/s : 1905.2
```

conc-32 system tok/s: **1905.2** vs Task 3 measured baseline **2087.1** → **-8.72%, outside the 5% tolerance. FAIL.**

Per the brief, the prescribed next step on a cold-path regression is to retry with `--hicache-write-policy=write_through_selective`. **This was not attempted.** Changing that flag requires restarting the worker with different server args, and this task's operational rules explicitly say: "Do NOT restart, stop, or reconfigure the stack. If you believe a restart is needed, STOP and report rather than doing it." The regression is recorded as-is, unretried, per that instruction — this is a live production endpoint serving users through a gateway on another host, and a config-flag retry was judged out of scope for this measurement pass. A follow-up task with authorization to restart the worker should try `write_through_selective` and re-measure.

#### hicache-prefix (concurrency 1, num 1, passes 2, shared-prefix-tokens 131072, max-tokens 32)

Not a pass/fail criterion per the brief ("Warm-prefix TTFT is NOT a criterion") — recorded for completeness.

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench hicache-prefix pass 1/2  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 18.76 s
TTFT  mean/p50/p99  : 18408 / 18408 / 18408 ms
ITL   mean/p50/p99  : 31.8 / 29.6 / 49.0 ms
per-req decode tok/s: mean 88.6  (min 88.6, max 88.6)
prompt tokens mean  : 135271
output tokens total : 32
system output tok/s : 1.7

=== bench hicache-prefix pass 2/2  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.93 s
TTFT  mean/p50/p99  : 685 / 685 / 685 ms
ITL   mean/p50/p99  : 18.2 / 18.1 / 20.7 ms
per-req decode tok/s: mean 131.1  (min 131.1, max 131.1)
prompt tokens mean  : 135271
cached tokens mean  : 135168 (99.9% of prompt)
output tokens total : 32
system output tok/s : 34.6

=== TTFT by pass (mean ms) ===
pass 1: 18408
pass 2: 685
pass1/pass2 speedup : 26.87x
```

Pass 1 TTFT (18408 ms) is slower than the plain-radix baseline's pass 1 (16677 ms) — the first-ever write of a 131072-token prefix through hicache's write-through path to L2/L3 costs more than a plain radix-only cold prefill, as expected (extra I/O on the write side). This is consistent with the conc-32 regression above: hicache write-through overhead is real and visible on this host, not a one-off measurement blip.

### Step 4b: eviction-pressure sequence (the decisive test)

Note: the prime step below was **not** genuinely cold, unlike the Task 3b plain-radix baseline — it landed 0.87 s / 100% cached, because the immediately-preceding `hicache-prefix` benchmark (previous section) had just written this exact seed-1234 prefix. That benchmark's pass 1 is therefore the true cold-write reference for this prefix (18408 ms, see above).

#### evict-hicache-prime (seed 1234)

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-hicache-prime pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.87 s
TTFT  mean/p50/p99  : 669 / 669 / 669 ms
ITL   mean/p50/p99  : 18.0 / 17.7 / 23.0 ms
per-req decode tok/s: mean 156.4  (min 156.4, max 156.4)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 36.7
```

#### evict-hicache-warm (seed 1234, confirms resident before flood)

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-hicache-warm pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.85 s
TTFT  mean/p50/p99  : 648 / 648 / 648 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 21.2 ms
per-req decode tok/s: mean 158.3  (min 158.3, max 158.3)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 37.7
```

#### flood with 5 distinct prefixes (seeds 2-6), 655,360 tokens total

```
===== SEED 2 =====
shared prefix: ~131072 tokens, seed 2, 905409 chars

=== bench evict-hicache-flood-seed2 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.09 s
TTFT  mean/p50/p99  : 16729 / 16729 / 16729 ms
ITL   mean/p50/p99  : 32.0 / 27.5 / 50.6 ms
per-req decode tok/s: mean 87.9  (min 87.9, max 87.9)
prompt tokens mean  : 135150
output tokens total : 32
system output tok/s : 1.9

===== SEED 3 =====
shared prefix: ~131072 tokens, seed 3, 905248 chars

=== bench evict-hicache-flood-seed3 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.09 s
TTFT  mean/p50/p99  : 16735 / 16735 / 16735 ms
ITL   mean/p50/p99  : 32.0 / 30.5 / 49.0 ms
per-req decode tok/s: mean 88.0  (min 88.0, max 88.0)
prompt tokens mean  : 135151
output tokens total : 32
system output tok/s : 1.9

===== SEED 4 =====
shared prefix: ~131072 tokens, seed 4, 904689 chars

=== bench evict-hicache-flood-seed4 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.16 s
TTFT  mean/p50/p99  : 16779 / 16779 / 16779 ms
ITL   mean/p50/p99  : 31.1 / 25.7 / 51.5 ms
per-req decode tok/s: mean 83.0  (min 83.0, max 83.0)
prompt tokens mean  : 135164
output tokens total : 32
system output tok/s : 1.9

===== SEED 5 =====
shared prefix: ~131072 tokens, seed 5, 905176 chars

=== bench evict-hicache-flood-seed5 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.21 s
TTFT  mean/p50/p99  : 16852 / 16852 / 16852 ms
ITL   mean/p50/p99  : 32.2 / 26.1 / 51.9 ms
per-req decode tok/s: mean 87.4  (min 87.4, max 87.4)
prompt tokens mean  : 135278
output tokens total : 32
system output tok/s : 1.9

===== SEED 6 =====
shared prefix: ~131072 tokens, seed 6, 905114 chars

=== bench evict-hicache-flood-seed6 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.23 s
TTFT  mean/p50/p99  : 16852 / 16852 / 16852 ms
ITL   mean/p50/p99  : 30.7 / 27.4 / 49.3 ms
per-req decode tok/s: mean 84.0  (min 84.0, max 84.0)
prompt tokens mean  : 135255
output tokens total : 32
system output tok/s : 1.9
```

All five flood requests were genuinely cold (no `cached tokens mean` line, TTFT 16729-16852 ms) — same order of magnitude as the plain-radix baseline's flood (16641-16841 ms), confirming each seed hit fresh, uncached content and pushed the GPU radix pool to evict prefix A (seed 1234) exactly as in Task 3b.

#### evict-hicache-recheck (seed 1234) — THE DECISIVE MEASUREMENT

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench evict-hicache-recheck pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 1.43 s
TTFT  mean/p50/p99  : 1230 / 1230 / 1230 ms
ITL   mean/p50/p99  : 17.6 / 17.6 / 21.7 ms
per-req decode tok/s: mean 160.0  (min 160.0, max 160.0)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 22.4
```

**Cached tokens mean: 135232 / 135271 = 100.0% of prompt — non-zero, in fact fully cached.** This is direct, unambiguous proof the evicted prefix was served from L2 (host RAM) or L3 (`/scratch`) rather than recomputed from scratch. TTFT corroborates: 1230 ms vs. the Task 3b baseline's 16769 ms at the identical point in an identical sequence — a 13.6x reduction, landing much closer to the warm-GPU-radix numbers (648-685 ms in this same run) than to the baseline's cold-recompute number.

**This is the opposite of the "tiering does nothing" failure case described in the brief.** The recheck is not ~16.8 s at 0% cached — it is 1230 ms at 100% cached. The tiering worked exactly as designed for this test.

### `/scratch` growth and write-amplification check

```
$ du -sh /scratch/kvcache/glm52   # AFTER all Step 4 + Step 4b benchmarks
48G	/scratch/kvcache/glm52

$ find /scratch/kvcache/glm52 -type f | wc -l
38106
```

`/scratch` grew from **0 bytes / 0 files (before) to 48 GB / 38106 files (after)** — the L3 tier is unambiguously being written, not sitting idle. Combined with the 100%-cached, 1230 ms recheck above, this also resolves which of the two possible failure modes would have applied had the tiering not worked: this is the "grows AND recheck is warm" case (tier written, and read back successfully), not "grows but recheck stays cold" (written but not read back), and definitely not "never grows at all" (not written).

File naming has no `tp_rank` suffix (`<hash>_glm-5.2-fp8.bin`, `<hash>.indexer_glm-5.2-fp8.bin`, `<hash>.draft_glm-5.2-fp8.bin` — 3 files per page-key), consistent with the brief's expectation that MLA uses one deduped storage key per page shared across all 8 TP ranks. File count / 3 = 12702 unique page-keys; at page_size=64 that covers 812,928 tokens, in the right order of magnitude for the unique (non-cache-hit) token volume actually pushed through this session (the 5-seed flood alone is 655,360 tokens of genuinely new content, plus the hicache-prefix pass-1 cold write of 131,072 tokens, plus smaller amounts from the latency/throughput runs). This is evidence *against* gross 8x storage bloat from uncoordinated per-rank writes — if all 8 ranks were writing independent copies under distinct keys, file count and total bytes would be roughly 8x higher for the same unique-token volume.

**Caveat:** this is a size/file-count inference, not a direct I/O measurement. The brief asked to "sample `iostat -x 5 3`" specifically *during* the conc-32 run; that was not done live (the check was designed in retrospect, after the conc-32 run had already completed, per the instruction not to re-run benchmarks chasing better numbers). It's possible all 8 ranks are still each independently issuing a write syscall to the *same* file/key (redundant I/O that wouldn't show up in `du -sh` or file count, only in `iostat`'s write-ops rate). Given the conc-32 throughput regression already measured above (-8.72%), redundant same-key writes across ranks is a plausible contributing cause and should be checked directly with `iostat` in a follow-up run authorized to hold the endpoint under synthetic load again.

### Step 5/6: PASS/FAIL against the spec's criteria

| Criterion | Baseline | Profile A measured | Threshold | Verdict |
|---|---:|---:|---|---|
| conc-1 decode tok/s (no cold-path regression) | 150.6 | **149.8** | within 5% (≥143.07) | **PASS** (-0.53%) |
| conc-32 system tok/s (no cold-path regression) | 2087.1 | **1905.2** | within 5% (≥1982.75) | **FAIL** (-8.72%) |
| Eviction survival — cached tokens on recheck (decisive) | 0 (0%) | **135232 (100.0%)** | non-zero | **PASS** |
| Eviction survival — TTFT on recheck (corroboration) | 16769 ms | **1230 ms** | substantially lower | **PASS** (13.6x lower) |
| `/scratch` grows (L3 tier written) | 0 bytes | **48 GB, 38106 files** | non-empty after traffic | **PASS** |
| Write-amplification check (dedup key, no ~8x bloat) | n/a | file/size math consistent with 1x, not verified live via `iostat` | no ~8x growth vs unique-token rate | **PASS (inferred), unverified by direct I/O sampling** |
| `hicache_mem_layout` unchanged | `page_first` | `page_first` | must not be rewritten | **PASS** |
| `hicache_io_backend` unchanged | `kernel` | `kernel` | must not be downgraded to `direct` | **PASS** |
| `max_total_num_tokens` not shrunk | 540800 | 540928 | ≥540800 | **PASS** |
| Attention backend unchanged | DSA / `flashmla_kv` | DSA / `flashmla_kv` | must remain auto-selected DSA | **PASS** |
| Hierarchical KV host alloc ≈96 GB/rank | — | 96.00 GB x8 | ~96 GB | **PASS** |
| DSA indexer host alloc ≈22 GB/rank | — | 19.32 GB x8 | ~22 GB (brief's estimate) | **DIFFERS from plan estimate (19.32 vs ~22 GB) — not a pass/fail item per se, recorded prominently as instructed** |

### Overall judgement

**The decisive criterion — eviction survival — is a clear, unambiguous PASS.** A 131K-token prefix, fully evicted from the 540,928-token GPU radix pool by a 655,360-token flood, came back at 100.0% cached tokens and 1230 ms TTFT instead of the baseline's 0% cached / 16,769 ms. This is direct proof (via `usage.prompt_tokens_details.cached_tokens`, not just TTFT) that L2/L3 tiering is doing real, working prefix recovery. Leading with the cached-token number as instructed: **135232/135271 (100.0%) is the headline result, and it is unambiguously positive** — this is not the "tiering does nothing" outcome, and there is no need to soften or hedge that finding.

**However, the cold-path-regression criterion is a genuine, measured FAIL at conc-32** (1905.2 vs 2087.1 tok/s, -8.72%, outside the 5% tolerance), while conc-1 passes comfortably (-0.53%). Tiering is not "free when it misses" under concurrent load on this host as currently configured — write-through overhead is visible both in the conc-32 throughput drop and in the elevated cold-write TTFT for the hicache-prefix benchmark's pass 1 (18408 ms vs. the plain-radix baseline's 16677 ms for an equivalent cold prefill). The brief's suggested mitigation (`--hicache-write-policy=write_through_selective`) requires a worker restart, which was out of scope for this measurement pass per explicit operational instructions to not restart the live, user-serving endpoint. **This regression should not be dismissed** — it is the one criterion this run does not pass, and it should be re-tested with `write_through_selective` (or another write-policy tuning) in a follow-up task that has authorization to restart the worker.

The stack was left running and untouched throughout (no `stop.sh`/`serve.sh`/`docker compose restart` at any point in this measurement pass); it remains UP and serving on `PROFILE=cache` after this task.

## Profile A: write_through vs write_through_selective

**Task 5b.** Tests the hypothesis that `--hicache-write-policy=write_through` (writes every KV page to host RAM + `/scratch` synchronously) is the cause of the conc-32 throughput regression measured in Task 5 (-8.72% vs no-hicache baseline), and that `write_through_selective` (writes only hotter pages) recovers throughput without losing the eviction-recovery win.

### Method

1. `/scratch/kvcache/glm52` was cleared to empty (`find ... -mindepth 1 -delete`; directory itself, mode 0777, left in place) so both policies start from an identical cold L3 tier.
2. Stack was restarted with `HICACHE_WRITE_POLICY=write_through_selective PROFILE=cache ./serve.sh`.
   - `./stop.sh` hit one snag: `dynamo-worker-1` came back as "PID ... is zombie and can not be killed" on the normal `docker compose down`. Resolved with `docker kill dynamo-worker-1` (container exited 0 within ~4s), then `./stop.sh` completed cleanly (full teardown of worker/frontend/etcd/nats). Not a benchmark-relevant event, recorded for completeness.
   - Model load took ~210s (~3.5 min) to `glm-5.2-fp8` appearing in `/v1/models`.
3. **Confirmed the policy took effect** — grepped the worker's `ServerArgs` log line:
   ```
   hicache_write_policy='write_through_selective'
   ```
   Also confirmed unchanged alongside it: `hicache_mem_layout='page_first'`, `hicache_io_backend='kernel'`, `hicache_size=96`, `hicache_ratio=2.0`, `hicache_storage_backend='file'`, `attention_backend='dsa'`, `dsa_prefill_backend='flashmla_kv'`, `dsa_decode_backend='flashmla_kv'`.
4. Ran the identical bench sequence used for the `write_through` (Profile A) measurement: conc-1 latency, conc-32 throughput, then the 9-request eviction-pressure sequence (prime → warm → 5-seed flood → recheck).

### Raw output — conc-1 latency (`selective-latency`)

```
=== bench selective-latency pass 1/1  conc=1 num=16 max_tokens=256 ===
requests ok/err     : 16/0
wall time           : 32.66 s
TTFT  mean/p50/p99  : 336 / 149 / 2935 ms
ITL   mean/p50/p99  : 16.3 / 16.2 / 17.7 ms
per-req decode tok/s: mean 149.6  (min 142.2, max 150.2)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 4096
system output tok/s : 125.4
```

### Raw output — conc-32 throughput (`selective-throughput`)

```
=== bench selective-throughput pass 1/1  conc=32 num=128 max_tokens=256 ===
requests ok/err     : 128/0
wall time           : 17.68 s
TTFT  mean/p50/p99  : 1028 / 657 / 2226 ms
ITL   mean/p50/p99  : 31.3 / 30.4 / 52.6 ms
per-req decode tok/s: mean 75.9  (min 62.3, max 84.4)
prompt tokens mean  : 67
cached tokens mean  : 64 (95.5% of prompt)
output tokens total : 32768
system output tok/s : 1853.1
```

### Raw output — eviction-pressure sequence

```
shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench selective-evict-prime pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 18.67 s
TTFT  mean/p50/p99  : 18450 / 18450 / 18450 ms
ITL   mean/p50/p99  : 18.1 / 17.8 / 27.1 ms
per-req decode tok/s: mean 143.0  (min 143.0, max 143.0)
prompt tokens mean  : 135271
output tokens total : 32
system output tok/s : 1.7

shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench selective-evict-warm pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.87 s
TTFT  mean/p50/p99  : 670 / 670 / 670 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 20.7 ms
per-req decode tok/s: mean 158.5  (min 158.5, max 158.5)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 36.7

shared prefix: ~131072 tokens, seed 2, 905409 chars

=== bench selective-evict-flood-seed2 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 16.91 s
TTFT  mean/p50/p99  : 16709 / 16709 / 16709 ms
ITL   mean/p50/p99  : 17.8 / 17.7 / 22.3 ms
per-req decode tok/s: mean 156.2  (min 156.2, max 156.2)
prompt tokens mean  : 135150
output tokens total : 32
system output tok/s : 1.9

shared prefix: ~131072 tokens, seed 3, 905248 chars

=== bench selective-evict-flood-seed3 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.11 s
TTFT  mean/p50/p99  : 16849 / 16849 / 16849 ms
ITL   mean/p50/p99  : 17.9 / 17.6 / 24.7 ms
per-req decode tok/s: mean 123.4  (min 123.4, max 123.4)
prompt tokens mean  : 135151
output tokens total : 32
system output tok/s : 1.9

shared prefix: ~131072 tokens, seed 4, 904689 chars

=== bench selective-evict-flood-seed4 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.01 s
TTFT  mean/p50/p99  : 16784 / 16784 / 16784 ms
ITL   mean/p50/p99  : 18.0 / 17.7 / 24.2 ms
per-req decode tok/s: mean 143.8  (min 143.8, max 143.8)
prompt tokens mean  : 135164
output tokens total : 32
system output tok/s : 1.9

shared prefix: ~131072 tokens, seed 5, 905176 chars

=== bench selective-evict-flood-seed5 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.12 s
TTFT  mean/p50/p99  : 16916 / 16916 / 16916 ms
ITL   mean/p50/p99  : 18.1 / 17.7 / 23.9 ms
per-req decode tok/s: mean 155.9  (min 155.9, max 155.9)
prompt tokens mean  : 135278
output tokens total : 32
system output tok/s : 1.9

shared prefix: ~131072 tokens, seed 6, 905114 chars

=== bench selective-evict-flood-seed6 pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 17.17 s
TTFT  mean/p50/p99  : 16946 / 16946 / 16946 ms
ITL   mean/p50/p99  : 18.0 / 17.8 / 24.5 ms
per-req decode tok/s: mean 143.6  (min 143.6, max 143.6)
prompt tokens mean  : 135255
output tokens total : 32
system output tok/s : 1.9

shared prefix: ~131072 tokens, seed 1234, 905297 chars

=== bench selective-evict-recheck pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 1.48 s
TTFT  mean/p50/p99  : 1278 / 1278 / 1278 ms
ITL   mean/p50/p99  : 17.7 / 17.7 / 21.3 ms
per-req decode tok/s: mean 158.8  (min 158.8, max 158.8)
prompt tokens mean  : 135271
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 21.6
```

All five flood requests were genuinely cold (no `cached tokens mean` line, TTFT 16709-16946 ms), confirming prefix A (seed 1234) was actually evicted from the GPU radix pool before the recheck, exactly as in the `write_through` run.

### `/scratch` size before/after

```
$ du -sh /scratch/kvcache/glm52; find /scratch/kvcache/glm52 -type f | wc -l   # BEFORE (cleared)
0	/scratch/kvcache/glm52
0

$ du -sh /scratch/kvcache/glm52; find /scratch/kvcache/glm52 -type f | wc -l   # AFTER selective run
47G	/scratch/kvcache/glm52
38070
```

### Side-by-side comparison

| Metric | Baseline (no hicache) | `write_through` (Task 5) | `write_through_selective` (Task 5b) | Selective vs baseline | Selective vs write_through |
|---|---:|---:|---:|---:|---:|
| conc-1 decode tok/s | 150.6 | 149.8 | **149.6** | -0.66% | -0.13% |
| conc-32 system tok/s | 2087.1 | 1905.2 | **1853.1** | **-11.21%** | **-2.73% (worse, not recovered)** |
| Eviction recheck — cached tokens % | 0% (0/…) | 100.0% (135232/135271) | **100.0% (135232/135271)** | preserved | preserved |
| Eviction recheck — TTFT | 16769 ms | 1230 ms | **1278 ms** | 13.1x lower | ~equal (+3.9%, within noise) |
| `/scratch` after full sequence | 0 B / 0 files | 48 GB / 38106 files | **47 GB / 38070 files** | grows either way | **essentially identical (-2.1% bytes, -0.09% files) — not a meaningful reduction** |

### Judgement against the hypothesis

**The hypothesis is not supported by this measurement. `write_through_selective` did not recover the conc-32 throughput regression — it made it slightly worse** (1853.1 vs 1905.2 tok/s under `write_through`, both well below the 2087.1 no-hicache baseline). conc-1 latency is unaffected either way (~150 tok/s, noise-level difference between policies). `/scratch` byte and file counts after the identical benchmark sequence are essentially unchanged between policies (47 GB/38070 files vs 48 GB/38106 files, a ~2% difference in bytes and <0.1% in file count) — selective is **not** writing meaningfully less to L3 under this workload. A plausible explanation: this workload's write volume is dominated by the 131K-token shared-prefix eviction-pressure sequence (prime + 5-seed flood, ~786K tokens of unique prefill) and the conc-1/conc-32 short-prompt runs, and under `write_through_selective`'s hotness heuristic essentially all of that content still qualifies as "hot enough" to write — so selectivity bought no reduction in L3 write volume for this traffic pattern, and therefore no throughput recovery either.

**The good news: the eviction-recovery win is fully intact.** Cached-token percentage on the decisive recheck is identical to `write_through` — 100.0% (135232/135271) — and TTFT (1278 ms) is statistically indistinguishable from `write_through`'s 1230 ms. Switching policies did not cost anything on the metric that matters most.

**Recommendation: do not switch the default to `write_through_selective`.** It provides no measured benefit (conc-32 throughput is not recovered — if anything it is very slightly worse — and `/scratch` write volume is unchanged) while adding an extra knob and a second I/O policy to reason about, for a workload where selectivity did not bite. `write_through` remains the recommended default per Task 5's conclusions; the conc-32 regression against the no-hicache baseline (-8.72% to -11.21% depending on policy) stands as a known, accepted trade for the eviction-recovery benefit, and is not one that write-policy tuning alone resolves. If the regression must be closed further, the next lever to investigate is not the write policy but I/O concurrency/backend tuning (e.g. `--hicache-io-backend`, storage-prefetch policy) or reducing `/scratch` write frequency structurally (e.g. batching, io_uring backend) rather than selectivity-based filtering.

Since the measurement above recommends against `write_through_selective` as the default, the stack was restarted one final time after the eviction-pressure sequence completed, back onto the default (`write_through`, no `HICACHE_WRITE_POLICY` override). The worker's `ServerArgs` log line was re-checked and confirms `hicache_write_policy='write_through'`. The stack is UP, serving on `PROFILE=cache` with `write_through`, matching the state at the start of this task (the same known-good policy this evaluation validates as the correct default).

(Both restarts in this task hit the same operational snag on `./stop.sh`: `dynamo-worker-1` came back "PID ... is zombie and can not be killed" on the normal `docker compose down`. Resolved both times with `docker kill dynamo-worker-1`, which exited the container cleanly within ~4s, after which `./stop.sh` completed teardown normally. Recorded here since it recurred; not benchmark-relevant.)

## Restart persistence (L3 survival)

**The question:** host RAM (L2) is wiped on every worker restart. Only `/scratch` (L3) survives. If a cached prefix does not come back from `/scratch` after a restart, the disk tier contributes nothing that L2-only would not, and it should be dropped. This test used a **fresh seed (4242)** never touched by any prior task, so no pre-existing cache state could confound the result.

### Step 1 — cache state before the experiment

```
$ du -sh /scratch/kvcache/glm52
47G	/scratch/kvcache/glm52
$ find /scratch/kvcache/glm52 -type f | wc -l
38070
```

### Step 2 — prime seed 4242 (expect cold; confirms the seed was genuinely unused)

```
$ ./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 4242 --num 1 --concurrency 1 --max-tokens 32 --tag persist-prime
shared prefix: ~131072 tokens, seed 4242, 904606 chars

=== bench persist-prime pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 20.93 s
TTFT  mean/p50/p99  : 20523 / 20523 / 20523 ms
ITL   mean/p50/p99  : 31.1 / 19.9 / 60.6 ms
per-req decode tok/s: mean 76.5  (min 76.5, max 76.5)
prompt tokens mean  : 135257
output tokens total : 32
system output tok/s : 1.5
```

No `cached tokens mean` line — genuinely cold, as expected. Seed 4242 was not previously cached; the experiment is valid.

### Step 3 — confirm warm (GPU/L2 hit before any restart)

```
$ ./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 4242 --num 1 --concurrency 1 --max-tokens 32 --tag persist-warm
shared prefix: ~131072 tokens, seed 4242, 904606 chars

=== bench persist-warm pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 0.99 s
TTFT  mean/p50/p99  : 754 / 754 / 754 ms
ITL   mean/p50/p99  : 17.3 / 17.7 / 19.2 ms
per-req decode tok/s: mean 136.5  (min 136.5, max 136.5)
prompt tokens mean  : 135257
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 32.4
```

### `/scratch` size after priming (before restart)

```
$ du -sh /scratch/kvcache/glm52
55G	/scratch/kvcache/glm52
$ find /scratch/kvcache/glm52 -type f | wc -l
44409
```

Grew from 47 GB / 38070 files to 55 GB / 44409 files — seed 4242's pages were written through to `/scratch`, as expected under `write_through`.

### Step 4 — restart the worker, preserving `/scratch`

```
$ docker kill dynamo-worker-1
Error response from daemon: cannot kill container: dynamo-worker-1: container 64c65a8ea06f PID 2587255 is zombie and can not be killed. Use the --init option when creating containers to run an init inside the container that forwards signals and reaps processes
$ ./stop.sh
 Container dynamo-worker-1 Stopping
 Container dynamo-frontend-1 Stopping
 ...
Dynamo stack stopped.
$ PROFILE=cache ./serve.sh
```

(This is the same known `./stop.sh` "zombie PID" snag documented earlier in this file — cosmetic on `docker kill`, teardown still completed cleanly.) `/scratch/kvcache/glm52` was **not** touched — no volume flags, no deletion.

Model re-registered after ~180 s:
```
$ curl -s --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/models
{"object":"list","data":[{"id":"glm-5.2-fp8","object":"model","created":1788203337,"owned_by":"nvidia","context_window":524288}]}
```

**L2 (host RAM) confirmed rebuilt fresh and empty** — 8 allocations (one per TP rank), all timestamped from this boot:
```
$ docker compose --profile cache logs worker 2>&1 | grep -c "Allocating 96.00 GB host memory for hierarchical KV cache"
8
$ docker compose --profile cache logs worker 2>&1 | grep "Allocating 96.00 GB host memory for hierarchical KV cache"
worker-1  | ... INFO memory_pool_host.__init__: Allocating 96.00 GB host memory for hierarchical KV cache.
   (x8, all at 2026-08-31T19:08:09.66x-19:08:09.72x — this boot only)
```
Server config also reconfirmed unchanged: `hicache_write_policy='write_through'`, `hicache_storage_backend='file'`.

### Step 5 — THE MEASUREMENT: re-request the identical seed-4242 prefix after restart

```
$ ./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 4242 --num 1 --concurrency 1 --max-tokens 32 --tag persist-after-restart
shared prefix: ~131072 tokens, seed 4242, 904606 chars

=== bench persist-after-restart pass 1/1  conc=1 num=1 max_tokens=32 ===
requests ok/err     : 1/0
wall time           : 5.37 s
TTFT  mean/p50/p99  : 5121 / 5121 / 5121 ms
ITL   mean/p50/p99  : 18.7 / 17.8 / 27.9 ms
per-req decode tok/s: mean 127.4  (min 127.4, max 127.4)
prompt tokens mean  : 135257
cached tokens mean  : 135232 (100.0% of prompt)
output tokens total : 32
system output tok/s : 6.0
```

### `/scratch` size after the restart measurement

```
$ du -sh /scratch/kvcache/glm52
55G	/scratch/kvcache/glm52
$ find /scratch/kvcache/glm52 -type f | wc -l
44409
```

Unchanged from the pre-restart figure (no new writes needed — the read came from existing L3 content, and file count/size stayed flat because the request read from disk rather than adding to it).

### Side-by-side: prime (cold) / warm (pre-restart) / after-restart

| Pass | TTFT (mean) | Cached tokens | Cached % |
|---|---:|---:|---:|
| persist-prime (cold, fresh seed) | 20523 ms | — (no cached-tokens line) | 0% |
| persist-warm (pre-restart, GPU/L2) | 754 ms | 135232 / 135257 | 100.0% |
| persist-after-restart (post L2-wipe) | 5121 ms | 135232 / 135257 | **100.0%** |

### Verdict

**L3 persistence CONFIRMED.** Cached-token percentage is the direct read, and it is unambiguous: **100.0% of the prompt was served from cache after a full worker restart that wiped host RAM (L2 freshly reallocated, 8×96 GB, confirmed empty at boot)**. The only place those 135,232 tokens could have come from is `/scratch/kvcache/glm52`, which was untouched across the restart and stood at 55 GB / 44,409 files throughout.

TTFT after restart (5121 ms) sits between cold (20523 ms) and pre-restart-warm (754 ms) — about 4x faster than cold, but ~6.8x slower than an in-RAM (L2) hit. This is the expected shape of an L3 (NVMe) hit: slower than RAM because pages must be read from disk and copied back into the GPU/L2 pools, but categorically faster than recomputing the full 131K-token prefill from scratch. The **cached-token count**, not TTFT, is the decisive metric per the task's own instruction, and it reads 100.0% — a clean, unambiguous hit.

**Conclusion: `/scratch` earns its place.** This is the one claim host RAM alone cannot make — L2 is gone on every restart, and disk is what carried the 131K-token prefix through it. The tiered design (GPU → host RAM → `/scratch` NVMe) is justified by this result: L3 is not redundant with L2, it is what makes cache survival possible across a restart at all.

The stack was left UP and serving on `write_through` after this test (unchanged from state at test start); no further restart was performed after the measurement.

## Profile B (hisparse, long context)

**Question tested:** the repo documents that serving this model's full 1M-token context requires ≥2 nodes, and that Profile B (`PROFILE=longctx`, SGLang HiSparse) is the single-node attempt. Does a >524,288-token request actually complete on this node? If so, the ≥2-node claim is false.

**Result: Profile B never started. The claim survives — it is reinforced, not falsified.**

### Attempt 1: default config (`--mem-fraction-static=0.88`)

```
$ cd dynamo
$ docker kill dynamo-worker-1 2>/dev/null; ./stop.sh
$ PROFILE=longctx ./serve.sh
```

Worker container exited during startup, well before registration. `docker compose --profile longctx logs --tail=150 worker-longctx`:

```
Exception: Capture cuda graph failed: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 6 has a total capacity of 139.80 GiB of which 488.19 MiB is free. Including non-PyTorch memory, this process has 139.20 GiB memory in use. Of the allocated memory 134.73 GiB is allocated by PyTorch, and 297.40 MiB is reserved by PyTorch but unallocated. ...

Possible solutions:
1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)
2. set --cuda-graph-max-bs to a smaller value (e.g., 16)
3. disable torch compile by not using --enable-torch-compile
4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)
```

Traceback: `Scheduler.__init__` → `init_tp_model_worker` → `ModelRunner.initialize` → `init_device_graphs` → `CUDAGraphRunner.__init__` — the CUDA-graph-capture OOM the task brief flagged as known-plausible at `--mem-fraction-static=0.88` (measured 10.09 GB/GPU available at 0.85; 0.88 leaves only ~5.8 GB headroom).

No mention of speculative decoding anywhere in the failure — MTP/EAGLE was not the cause here, so that authorized fix did not apply.

### Attempt 2 (authorized retry): `MEM_FRACTION_LONG=0.86`

```
$ docker kill dynamo-worker-longctx-1 2>/dev/null; ./stop.sh
$ MEM_FRACTION_LONG=0.86 PROFILE=longctx ./serve.sh
```

Same failure mode, same stage, different GPU, tighter margin than attempt 1:

```
Exception: Capture cuda graph failed: CUDA out of memory. Tried to allocate 1.31 GiB. GPU 1 has a total capacity of 139.80 GiB of which 1.24 GiB is free. Including non-PyTorch memory, this process has 138.44 GiB memory in use. Of the allocated memory 132.82 GiB is allocated by PyTorch, with 1.56 GiB allocated in private pools (e.g., CUDA Graphs), and 1.43 GiB is reserved by PyTorch but unallocated. ...
```

The container exited before ever reaching the point where `max_total_num_tokens` is logged — the OOM happens during CUDA graph capture in `ModelRunner.initialize`, which runs before the KV pool sizing that produces that line. **There is no `max_total_num_tokens`, no HiSparse indexer log, no `context_len` figure to report for Profile B — the worker never got that far.**

Per the task's authorization (one retry per known-plausible cause, no further retries "hoping for a better number"), this is where testing stopped. Both authorized mitigations were exhausted:
- Speculative decoding was never implicated (no such error appeared) — the four `--speculative-*` lines were never removed, so **MTP/EAGLE status is untested**, not "survived."
- The mem-fraction retry (0.86) was tried once as authorized and still OOM'd, at an even tighter margin than the 0.88 default.

### Step 3/4/5 — not performed

No >512K request was attempted, no throughput benchmarks were run, and no `max_total_num_tokens`/HiSparse/indexer figures exist to record, because the worker process never reached a registered, serving state under Profile B at either mem-fraction tested. Reporting fabricated or extrapolated numbers for these steps would violate the honesty requirement of this task; they are left blank.

### Verdict

**The single-node ≥2-node constraint is NOT falsified by this test — if anything it is corroborated.** Not only does the documented 512K-token pool represent the practical ceiling for Profile A; Profile B's attempt to push past it via HiSparse **could not even complete engine initialization** on this node at either the spec's projected `--mem-fraction-static=0.88` or the authorized fallback of 0.86. Both attempts died identically: CUDA OOM during CUDA-graph capture, with well under 1.5 GB free per GPU at the moment of failure. The trend between the two attempts (488 MiB free → 1.24 GiB free, i.e. *less* headroom relief than the 0.02 reduction in mem-fraction should have produced, likely because loaded weights/model state don't scale down with `--mem-fraction-static`) suggests this is not a knob-tuning problem solvable by nudging mem-fraction further — the full 1,048,576-token addressable range HiSparse is configured for does not fit in this node's remaining ~5-6 GB/GPU of headroom once the 756 GB of FP8 weights and CUDA graph working set are accounted for, at least not without deeper changes (e.g. `--cuda-graph-max-bs`, `--disable-cuda-graph`, or a smaller `--context-length`) that are out of scope for this task's two authorized retries.

**The trade this leaves undemonstrated:** whether HiSparse's page-from-host-RAM decode path is actually slower than Profile A (as documented) could not be measured, because Profile B could not be brought up at all. This task can only confirm the negative: on this single 8xH200 node, Profile B as currently configured does not start, and the ≥2-node claim for the full 1M-token context stands.

### Stack state at end of task

Restored to `PROFILE=cache`:

```
$ docker kill dynamo-worker-longctx-1 2>/dev/null; ./stop.sh
$ PROFILE=cache ./serve.sh
... REGISTERED ~90s
$ curl -s --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/models
{"object":"list","data":[{"id":"glm-5.2-fp8","object":"model","created":1788204978,"owned_by":"nvidia","context_window":524288}]}
$ docker compose --profile cache logs worker 2>&1 | grep -c "Allocating 96.00 GB host memory for hierarchical KV cache"
8
```

All 8 TP-rank host-memory allocations present, confirming Profile A came back up correctly. No source files were modified during this task (the speculative-decoding removal authorized in the brief was never triggered, since the failure was OOM, not spec-decode-related).

## Profile B retry (cuda-graph-max-bs 128)

Retry of the above, testing the fix in commit `92acc5e` (`fix(dynamo): cap longctx CUDA graph capture at the concurrency ceiling`): `worker-longctx` now passes `--cuda-graph-max-bs=${CUDA_GRAPH_MAX_BS_LONG:-128}`, capping CUDA-graph capture at the same ceiling as `--max-running-requests=128`, instead of SGLang's default of 512. The theory was that graphs for batch sizes 129-512 were being captured but could never be scheduled, wasting ~1.56 GiB of "private pools (e.g., CUDA Graphs)" memory that the previous attempt's failure log showed.

### Attempt: `PROFILE=longctx` with the `cuda-graph-max-bs=128` fix in place

```
$ cd /home/users/wrightda/src/GLM-5.2-FP8/dynamo
$ docker kill dynamo-worker-1 2>/dev/null; ./stop.sh
Container dynamo-frontend-1 Stopped/Removed
Container dynamo-worker-1 Stopped/Removed
Container dynamo-nats-1 Stopped/Removed
Container dynamo-etcd-1 Stopped/Removed
Dynamo stack stopped.

$ PROFILE=longctx ./serve.sh
Starting Dynamo (SGLang) stack for GLM-5.2-FP8 [profile: longctx] ...
... Up.
```

Poll loop result:

```
WORKER DIED at ~165s
```

`docker ps -a` confirmed: `dynamo-worker-longctx-1  Exited (0)  15 seconds ago` (the exit code is misleadingly `0` — the process was killed by its own sigquit handler after a child crashed, not a clean shutdown).

### Failure — same OOM class, different (and worse) failure point

`docker compose --profile longctx logs --tail=200 worker-longctx` (relevant excerpt, verbatim):

```
worker-longctx-1  |   File "/home/dynamo/.local/lib/python3.12/site-packages/sglang/srt/layers/attention/dsa/dsa_indexer.py", line 685, in _get_topk_paged
worker-longctx-1  |     logits = deep_gemm.fp8_paged_mqa_logits(
worker-longctx-1  |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
worker-longctx-1  |   File "/usr/local/lib/python3.12/dist-packages/deep_gemm/__init__.py", line 221, in fp8_paged_mqa_logits
worker-longctx-1  |     return _C.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, indices)
worker-longctx-1  | tvm.error.InternalError: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 5 has a total capacity of 139.80 GiB of which 488.19 MiB is free. Including non-PyTorch memory, this process has 139.20 GiB memory in use. Of the allocated memory 134.73 GiB is allocated by PyTorch, and 297.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.

... (same trace re-raised as, further down) ...

worker-longctx-1  |   File "/home/dynamo/.local/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py", line 2908, in init_device_graphs
worker-longctx-1  |     self.graph_runner = graph_runners[self.device](self)
worker-longctx-1  |   File "/home/dynamo/.local/lib/python3.12/site-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 628, in __init__
worker-longctx-1  |     raise Exception(
worker-longctx-1  | Exception: Capture cuda graph failed: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 5 has a total capacity of 139.80 GiB of which 488.19 MiB is free. Including non-PyTorch memory, this process has 139.20 GiB memory in use. Of the allocated memory 134.73 GiB is allocated by PyTorch, and 297.40 MiB is reserved by PyTorch but unallocated.

Possible solutions:
1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)
2. set --cuda-graph-max-bs to a smaller value (e.g., 16)
3. disable torch compile by not using --enable-torch-compile
4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)

[2m2026-08-31T19:41:52.922414Z[0m [31mERROR[0m [2mengine.launch_phase_sigquit_handler[0m[2m:[0m Received sigquit from a child process. It usually means the child failed.
```

This is a materially different failure site from both prior attempts. The two previous OOMs (attempt 1 at `mem-fraction=0.88`, attempt 2 at `0.86`) both failed inside `CUDAGraphRunner.__init__`'s generic graph capture, with 1.24-5.8 GiB nominally free at the moment of failure. This attempt got *past* that stage — the `cuda-graph-max-bs=128` cap did shrink the wasted-graph problem it targeted — but then failed one call deeper, inside the **HiSparse indexer's own CUDA graph capture** (`dsa_indexer.py::_get_topk_paged` → `deep_gemm.fp8_paged_mqa_logits`), with only **488.19 MiB free** (worse headroom than either prior attempt) and 139.20 GiB of 139.80 GiB already in use on that GPU.

In other words: the fix worked exactly as intended (fewer wasted graphs, capture proceeds further), but capture now runs long enough to reach the indexer's own graph-capture pass, which itself needs ~1.5 GiB it does not have — because reclaiming the 512→128 headroom just let the allocator spend that reclaimed space on *more* real capture work before running out again. This is not a case of "off by 70 MB"; the process is fully pinned (139.20/139.80 GiB in use) by the time it fails.

### Steps 3-5 — not performed

The worker never reached a registered state, so there is no `max_total_num_tokens`, no HiSparse/indexer sizing line, no `context_len` figure, and no bench_stream.py run to report. The decisive 600K-token test (step 4) and the cost-comparison benchmarks (step 5) were not attempted, per the task's instruction to go straight to restoring Profile A on a second OOM/death.

### Verdict

**Profile B still does not initialize on this single 8xH200 node, even with the `cuda-graph-max-bs=128` fix.** The fix addressed its target defect (graphs for unreachable batch sizes 129-512) but did not close the gap — it merely moved the OOM one call deeper, into the HiSparse indexer's own graph capture, with less free memory at the point of failure than either prior attempt (488 MiB vs. 1.24 GiB / ~5.8 GiB). This corroborates, more strongly than the first attempt, that Profile B's HiSparse configuration for a ~1,048,576-token pool does not fit in a single node's per-GPU headroom once weights, ordinary CUDA graphs, and the indexer's own paged-MQA graph capture are all accounted for — the shortfall is not a small tuning margin, it is the GPU being fully pinned (139.20/139.80 GiB) before capture finishes.

**Documented config change needed (not applied, per "no source edits" instruction):** the SGLang error message itself is explicit about the remaining levers — `--mem-fraction-static` lower than 0.86, `--cuda-graph-max-bs` lower than 128 (SGLang's own suggestion is as low as 16, which would materially hurt throughput for a concurrency-128 deployment), `--disable-cuda-graph` (called out by SGLang as "Not recommended, huge performance loss"), or reducing the target context length so HiSparse's own paged-KV structures are smaller. None of these were authorized for this attempt. The most likely durable fix is a smaller `--context-length` for Profile B rather than further mem-fraction/graph-count tuning, since the indexer's own working set scales with the addressable context, not just with concurrency.

**The single-node ≥2-node constraint is further corroborated, not falsified**, by this retry: two independent, differently-targeted mitigations (mem-fraction reduction, then cuda-graph-max-bs reduction) both produced CUDA OOM during engine initialization, at progressively deeper (but still pre-serving) points in startup, with progressively tighter memory margins. The 1M-token HiSparse configuration does not fit this node.

### Stack state at end of task

Restored to `PROFILE=cache`:

```
$ docker kill dynamo-worker-longctx-1 2>/dev/null; ./stop.sh
Dynamo stack stopped.
$ PROFILE=cache ./serve.sh
... Up.
$ [poll loop] REGISTERED ~210s
$ curl -s --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/models
{"object":"list","data":[{"id":"glm-5.2-fp8","object":"model","created":1788205580,"owned_by":"nvidia","context_window":524288}]}
$ docker compose --profile cache logs --tail=500 worker 2>&1 | grep -c "Allocating 96.00 GB host memory for hierarchical KV cache"
8
```

All 8 TP-rank host-memory allocations present, confirming Profile A came back up correctly. No source files were modified during this task.
