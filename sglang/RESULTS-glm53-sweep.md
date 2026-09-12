# GLM-5.3-Flash sweep

Started 2026-09-12T00:00:48

Each row is a full worker restart. `--enable-mfu-metrics` added to every run.


## hicache-off

boot time: 353s

pools: `{'max_mamba_cache_size': 798, 'ssm_state_GB': 13.65, 'conv_state_GB': 0.48, 'max_total_num_tokens': 3500608, 'io_backend_downgraded': False}`

### concurrency 1

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 81.57 s
TTFT  mean/p50/p99  : 9037 / 132 / 71123 ms
ITL   mean/p50/p99  : 13.1 / 11.2 / 19.1 ms
per-req decode tok/s: mean 230.0  (min 134.5, max 279.6)
prompt tokens mean  : 2147
cached tokens mean  : 2048 (95.4% of prompt)
output tokens total : 2048
system output tok/s : 25.1

=== bench pass 2/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 8.30 s
TTFT  mean/p50/p99  : 134 / 134 / 139 ms
ITL   mean/p50/p99  : 11.8 / 11.4 / 14.2 ms
per-req decode tok/s: mean 284.7  (min 228.1, max 307.9)
prompt tokens mean  : 2147
cached tokens mean  : 2048 (95.4% of prompt)
output tokens total : 2048
system output tok/s : 246.8

=== TTFT by pass (mean ms) ===
pass 1: 9037
pass 2: 134
pass1/pass2 speedup : 67.44x

```
### concurrency 32

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 77.13 s
TTFT  mean/p50/p99  : 7355 / 380 / 18444 ms
ITL   mean/p50/p99  : 58.5 / 17.8 / 231.5 ms
per-req decode tok/s: mean 56.0  (min 12.5, max 109.2)
prompt tokens mean  : 2147
cached tokens mean  : 2051 (95.5% of prompt)
output tokens total : 40960
system output tok/s : 531.1

=== bench pass 2/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 35.39 s
TTFT  mean/p50/p99  : 2542 / 320 / 8396 ms
ITL   mean/p50/p99  : 33.0 / 17.9 / 230.0 ms
per-req decode tok/s: mean 69.1  (min 21.3, max 110.8)
prompt tokens mean  : 2147
cached tokens mean  : 2051 (95.5% of prompt)
output tokens total : 40960
system output tok/s : 1157.3

=== TTFT by pass (mean ms) ===
pass 1: 7355
pass 2: 2542
pass1/pass2 speedup : 2.89x

```

## wt (current)

boot time: 373s

pools: `{'max_mamba_cache_size': 798, 'ssm_state_GB': 13.65, 'conv_state_GB': 0.48, 'max_total_num_tokens': 3500608, 'io_backend_downgraded': False}`

### concurrency 1

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 99.57 s
TTFT  mean/p50/p99  : 11332 / 148 / 81128 ms
ITL   mean/p50/p99  : 12.9 / 11.3 / 19.0 ms
per-req decode tok/s: mean 238.0  (min 152.7, max 319.9)
prompt tokens mean  : 2147
cached tokens mean  : 2048 (95.4% of prompt)
output tokens total : 2048
system output tok/s : 20.6

=== bench pass 2/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 18.78 s
TTFT  mean/p50/p99  : 148 / 146 / 161 ms
ITL   mean/p50/p99  : 26.0 / 11.4 / 98.7 ms
per-req decode tok/s: mean 219.3  (min 25.5, max 309.7)
prompt tokens mean  : 2147
cached tokens mean  : 2048 (95.4% of prompt)
output tokens total : 2048
system output tok/s : 109.0

=== TTFT by pass (mean ms) ===
pass 1: 11332
pass 2: 148
pass1/pass2 speedup : 76.66x

```
### concurrency 32

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 69.93 s
TTFT  mean/p50/p99  : 5879 / 359 / 18259 ms
ITL   mean/p50/p99  : 58.3 / 17.7 / 263.0 ms
per-req decode tok/s: mean 54.0  (min 8.9, max 110.6)
prompt tokens mean  : 2147
cached tokens mean  : 2051 (95.5% of prompt)
output tokens total : 40960
system output tok/s : 585.7

=== bench pass 2/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 43.83 s
TTFT  mean/p50/p99  : 4027 / 341 / 16492 ms
ITL   mean/p50/p99  : 34.2 / 17.8 / 250.3 ms
per-req decode tok/s: mean 66.1  (min 21.6, max 112.3)
prompt tokens mean  : 2147
cached tokens mean  : 2051 (95.5% of prompt)
output tokens total : 40960
system output tok/s : 934.5

=== TTFT by pass (mean ms) ===
pass 1: 5879
pass 2: 4027
pass1/pass2 speedup : 1.46x

```

## wt-selective

boot time: 373s

pools: `{'max_mamba_cache_size': 798, 'ssm_state_GB': 13.65, 'conv_state_GB': 0.48, 'max_total_num_tokens': 3500608, 'io_backend_downgraded': False}`

### concurrency 1

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 83.86 s
TTFT  mean/p50/p99  : 8918 / 185 / 69939 ms
ITL   mean/p50/p99  : 17.8 / 11.3 / 18.6 ms
per-req decode tok/s: mean 232.5  (min 46.4, max 342.8)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 2048
system output tok/s : 24.4

=== bench pass 2/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 9.85 s
TTFT  mean/p50/p99  : 179 / 181 / 188 ms
ITL   mean/p50/p99  : 11.9 / 11.3 / 13.2 ms
per-req decode tok/s: mean 243.3  (min 221.3, max 262.7)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 2048
system output tok/s : 207.9

=== TTFT by pass (mean ms) ===
pass 1: 8918
pass 2: 179
pass1/pass2 speedup : 49.71x

```
### concurrency 32

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 33.71 s
TTFT  mean/p50/p99  : 2125 / 532 / 10943 ms
ITL   mean/p50/p99  : 35.3 / 18.7 / 358.1 ms
per-req decode tok/s: mean 62.6  (min 18.5, max 106.3)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 40960
system output tok/s : 1215.0

=== bench pass 2/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 37.96 s
TTFT  mean/p50/p99  : 2140 / 551 / 8991 ms
ITL   mean/p50/p99  : 40.3 / 18.5 / 240.7 ms
per-req decode tok/s: mean 63.1  (min 19.5, max 107.3)
prompt tokens mean  : 2147
cached tokens mean  : 2099 (97.8% of prompt)
output tokens total : 40960
system output tok/s : 1079.0

=== TTFT by pass (mean ms) ===
pass 1: 2125
pass 2: 2140
pass1/pass2 speedup : 0.99x

```

## wts+ssm-float32

boot time: 373s

pools: `{'max_mamba_cache_size': 798, 'ssm_state_GB': 13.65, 'conv_state_GB': 0.48, 'max_total_num_tokens': 3500608, 'io_backend_downgraded': False}`

### concurrency 1

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 86.66 s
TTFT  mean/p50/p99  : 9256 / 186 / 72676 ms
ITL   mean/p50/p99  : 18.4 / 11.3 / 22.2 ms
per-req decode tok/s: mean 225.6  (min 46.7, max 276.2)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 2048
system output tok/s : 23.6

=== bench pass 2/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 9.57 s
TTFT  mean/p50/p99  : 180 / 179 / 192 ms
ITL   mean/p50/p99  : 11.6 / 11.3 / 13.2 ms
per-req decode tok/s: mean 256.1  (min 224.8, max 360.9)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 2048
system output tok/s : 214.0

=== TTFT by pass (mean ms) ===
pass 1: 9256
pass 2: 180
pass1/pass2 speedup : 51.30x

```
### concurrency 32

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 32.66 s
TTFT  mean/p50/p99  : 2205 / 577 / 11117 ms
ITL   mean/p50/p99  : 32.1 / 18.5 / 355.0 ms
per-req decode tok/s: mean 66.0  (min 19.6, max 102.5)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 40960
system output tok/s : 1254.1

=== bench pass 2/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 30.61 s
TTFT  mean/p50/p99  : 1739 / 509 / 8885 ms
ITL   mean/p50/p99  : 32.7 / 18.6 / 355.0 ms
per-req decode tok/s: mean 65.4  (min 21.3, max 115.8)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 40960
system output tok/s : 1338.2

=== TTFT by pass (mean ms) ===
pass 1: 2205
pass 2: 1739
pass1/pass2 speedup : 1.27x

```

## wts+ssm-bfloat16

boot time: 373s

pools: `{'max_mamba_cache_size': 1548, 'ssm_state_GB': 13.24, 'conv_state_GB': 0.93, 'max_total_num_tokens': 3687104, 'io_backend_downgraded': False}`

### concurrency 1

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 85.91 s
TTFT  mean/p50/p99  : 9223 / 183 / 72447 ms
ITL   mean/p50/p99  : 18.0 / 11.1 / 16.7 ms
per-req decode tok/s: mean 244.0  (min 46.7, max 320.9)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 2048
system output tok/s : 23.8

=== bench pass 2/2  conc=1 num=8 max_tokens=256 ===
requests ok/err     : 8/0
wall time           : 7.85 s
TTFT  mean/p50/p99  : 178 / 177 / 186 ms
ITL   mean/p50/p99  : 11.7 / 11.3 / 13.5 ms
per-req decode tok/s: mean 280.4  (min 236.4, max 334.9)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 1820
system output tok/s : 231.9

=== TTFT by pass (mean ms) ===
pass 1: 9223
pass 2: 178
pass1/pass2 speedup : 51.95x

```
### concurrency 32

```
shared prefix: ~2000 tokens, seed 42, 13865 chars

=== bench pass 1/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 33.57 s
TTFT  mean/p50/p99  : 2228 / 534 / 11041 ms
ITL   mean/p50/p99  : 33.4 / 18.1 / 276.6 ms
per-req decode tok/s: mean 64.7  (min 20.2, max 108.4)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 40960
system output tok/s : 1220.2

=== bench pass 2/2  conc=32 num=160 max_tokens=256 ===
requests ok/err     : 160/0
wall time           : 29.60 s
TTFT  mean/p50/p99  : 1730 / 493 / 8968 ms
ITL   mean/p50/p99  : 33.7 / 18.3 / 264.7 ms
per-req decode tok/s: mean 69.5  (min 2.8, max 136.6)
prompt tokens mean  : 2147
cached tokens mean  : 2112 (98.4% of prompt)
output tokens total : 40730
system output tok/s : 1376.0

=== TTFT by pass (mean ms) ===
pass 1: 2228
pass 2: 1730
pass1/pass2 speedup : 1.29x

```
