# GLM-5.3-Flash: long-context verification

Measured 2026-09-12 on 8x H200, `PROFILE=flash`, SGLang
`0.0.0.dev1+gfe236ea6c3.mmfix1`, bf16 KV + tilelang DSA, MTP 5/1/6 adaptive.

**Result: the full ~1M context works.** 999,256-999,915 tokens prefilled and a
needle planted at 10% depth recalled correctly, reproduced across three
independent runs, ~50 s prefill.

This is the constraint that `CONTEXT_WINDOW.md` documents GLM-5.2 failing to
clear: its KV pool (540,928 tokens) was smaller than a single 1M-token request,
and six attempts at a 1M profile OOMed or crashed. GLM-5.3-Flash has a
3,687,104-token pool because only 11 of its 45 layers hold paged KV, so a
full-length request fits with ~3.5x headroom.

## Headline runs

| target | actual prompt_tokens | elapsed | needle | verdict |
|---|---|---|---|---|
| 128K | 131,063 | 4.9 s | YES | PASS |
| 512K | 523,807 | 28.2 s | NO (see below) | content-specific miss |
| ~1M | 999,256 | 49.1 s | YES | **PASS** |

~1M independently reproduced at 999,915 (55.2 s) and 999,483 (51.3 s), the
latter answering in 87 completion tokens -- i.e. directly, not after a struggle.

## The 512K miss is a needle artefact, not a length limit

One specific combination -- content seed 7, needle at 10% depth -- misses
reproducibly. Everything around it passes:

| variation | result |
|---|---|
| 512K, seeds 1 / 2 / 3, depth 10% | PASS / PASS / PASS |
| 512K, seed 7, depths 2% / 50% / 90% | PASS / PASS / PASS |
| 512K, seed 7, depth 10% | **MISS** (reproduced twice) |

6 of 6 neighbouring configurations pass. Treat this as normal
needle-in-haystack variance at extreme length, not a deployment limit.

## Two harness bugs this found (both mine, both worth not repeating)

**1. Assumed chars-per-token silently halved every test.** The first version
hardcoded 3.6 chars/token. Every length landed at ~52% of target, so a run
labelled "1024K" was really **548,006 tokens** -- and it PASSED, which would
have gone into the docs as "1M verified". Only logging the server-reported
`prompt_tokens` alongside the target caught it. The script now calibrates
against the live server at startup and prints the ratio (numeric-dense filler
measures ~2.97 chars/token; repetitive prose ~6.89).

**2. Homogeneous filler is adversarial to sparse attention.** The first filler
repeated one identical sentence. At 523,829 tokens the model then failed to
find the needle twice, burning a full 4096-token budget, while succeeding at
999,483. That is not a length effect: DSA's indexer selects top-k blocks by
relevance and has nothing to discriminate on when every block is byte-identical.
With heterogeneous records both 512K and ~1M answer in under 90 completion
tokens. Any long-context test on a DSA model needs varied filler.

## Scope

This is a **retrieval** check. It shows the model can locate a specific fact in
a ~1M-token context. It does NOT establish that reasoning quality is uniform
across the window. For agentic loops that synthesise across a long context
rather than extracting one fact, that remains open and is best measured on real
traces.

## Concurrency trade-off

At ~1M tokens per request against a 3,687,104-token pool, roughly **3
concurrent full-length requests** fit. `--max-running-requests=128` does not
change that; the pool is the binding constraint.

| context per request | concurrent requests |
|---|---|
| ~1M | ~3 |
| 200K | ~18 |
| 100K | ~36 |

Reproduce with `./test_longctx.py` (defaults 128K / 512K / 1M).
