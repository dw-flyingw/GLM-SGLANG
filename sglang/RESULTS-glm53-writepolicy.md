# GLM-5.3-Flash: write_through vs write_through_selective (concurrency 1)

Started 2026-09-12T23:23:23. Method and rationale: docstring of `ab_writepolicy.py`.

Prior evidence being tested: one sweep run (RESULTS-glm53-sweep.md), conc-1 pass-2 `write_through` 109.0 vs `write_through_selective` 207.9 tok/s; and RESULTS-kv-tiering.md Task 5b (GLM-5.2), which found selective no better.

Order: rep1 write_through, rep1 write_through_selective, rep2 write_through_selective, rep2 write_through, rep3 write_through, rep3 write_through_selective

| rep | policy | teardown of previous worker | L3 cleared (ok/failed) | boot | p1 sys | p2 sys | p2 decode | p2 ITL mean | p2 ITL p99 | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | write_through | clean; log archived | 57316/0 | 372s | 23.8 | 241.0 | 279.5 | 11.6 | 14.2 | 0 |
| 1 | write_through_selective | clean; log archived | 128/0 | 362s | 22.3 | 229.2 | 264.3 | 11.5 | 13.1 | 0 |
| 2 | write_through_selective | clean; log archived | 63/0 | 362s | 22.3 | 233.3 | 270.3 | 11.9 | 13.5 | 0 |
| 2 | write_through | clean; log archived | 63/0 | 372s | 24.2 | 228.5 | 265.0 | 11.6 | 14.4 | 0 |
| 3 | write_through | clean; log archived | 128/0 | 372s | 24.1 | 236.6 | 275.2 | 11.6 | 13.7 | 0 |
| 3 | write_through_selective | clean; log archived | 128/0 | 372s | 24.4 | 236.5 | 273.4 | 11.5 | 14.1 | 0 |

## Summary: pass 2, concurrency 1 (mean, then min-max)

| policy | n | sys tok/s | decode tok/s | ITL p99 ms |
|---|---|---|---|---|
| `write_through` | 3 | 235.4 (228.5-241.0) | 273.2 (265.0-279.5) | 14.1 (13.7-14.4) |
| `write_through_selective` | 3 | 233.0 (229.2-236.5) | 269.3 (264.3-273.4) | 13.6 (13.1-14.1) |

sys tok/s ranges **OVERLAP**: `write_through` 228.5-241.0, `write_through_selective` 229.2-236.5. Non-overlapping ranges at n=3 each is the bar for calling a difference real; overlapping ranges mean this harness cannot tell the policies apart.

The worker is left running the last arm (`write_through_selective`), which matches the committed flash profile.

## Outcome (recorded after the run)

**No measurable difference between the policies, so the flash profile's default
is switched back to `write_through`**, the engine default. The sentence directly
above ("matches the committed flash profile") was true when the script wrote it
and is superseded by that change.

- The prior single sample (`write_through` 109.0 tok/s, ITL p99 98.7 ms) did not
  reproduce in 3 interleaved runs (228.5-241.0 tok/s, ITL p99 13.7-14.4 ms). It
  was a slow-mode sample from a harness already shown to be bimodal.
- This agrees with `RESULTS-kv-tiering.md` Task 5b on GLM-5.2, which also found
  `write_through_selective` no better and recommended against it.
- **Teardown, observed for the first time with HiCache on GLM-5.3-Flash:** 6/6
  clean -- first `docker compose down` succeeded every time -- in contrast to
  GLM-5.2, where it failed 3/3 (see the `init:` comment in `docker-compose.yml`).
- **stop.sh log archiving:** 6/6 archived under real teardowns, validating the
  fix to its previously hardcoded worker list.
- **L3 clearing:** 0 failed deletions across 6 arms (57,316 files in the first),
  confirming in practice that the host user can remove what the root-run
  container writes.
