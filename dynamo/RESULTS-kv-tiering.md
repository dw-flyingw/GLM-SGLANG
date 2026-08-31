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
