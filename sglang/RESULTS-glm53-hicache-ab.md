# GLM-5.3-Flash: HiCache A/B

Started 2026-09-12T14:13:22

3 reps/arm. L3 cleared before `off` and `on-cold`; kept for `on-warm`.

- rep1 **off** (l3 cleared: 26200) conc1 p1/p2 25.9/243.4 | conc32 p1/p2 523.6/1506.8 tok/s
- rep1 **on-cold** (l3 cleared: 0) conc1 p1/p2 24.5/237.7 | conc32 p1/p2 475.3/1493.4 tok/s
- rep1 **on-warm** (l3 cleared: no) conc1 p1/p2 24.1/214.9 | conc32 p1/p2 653.7/936.4 tok/s
- rep2 **off** (l3 cleared: 687) conc1 p1/p2 26.3/224.5 | conc32 p1/p2 586.5/1537.1 tok/s
- rep2 **on-cold** (l3 cleared: 0) conc1 p1/p2 24.9/226.8 | conc32 p1/p2 658.0/956.9 tok/s
- rep2 **on-warm** (l3 cleared: no) conc1 p1/p2 22.6/228.0 | conc32 p1/p2 739.8/923.7 tok/s
- rep3 **off** (l3 cleared: 682) conc1 p1/p2 22.7/232.7 | conc32 p1/p2 752.8/911.8 tok/s
- rep3 **on-cold** (l3 cleared: 0) conc1 p1/p2 20.3/227.9 | conc32 p1/p2 642.9/1156.0 tok/s
- rep3 **on-warm** (l3 cleared: no) conc1 p1/p2 22.5/214.7 | conc32 p1/p2 648.8/942.6 tok/s

## Summary (system tok/s, mean +/- spread)

| arm | conc1 p1 | conc1 p2 | conc32 p1 | conc32 p2 |
|---|---|---|---|---|
| off | 25 ± 2 | 234 ± 9 | 621 ± 115 | 1319 ± 313 |
| on-cold | 23 ± 2 | 231 ± 5 | 592 ± 91 | 1202 ± 268 |
| on-warm | 23 ± 1 | 219 ± 7 | 681 ± 46 | 934 ± 9 |
