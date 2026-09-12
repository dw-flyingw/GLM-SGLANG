#!/usr/bin/env python3
"""Verify GLM-5.3-Flash actually serves long context, not just advertises it.

CONTEXT_WINDOW.md records that GLM-5.2 could not serve 1M on this node: the
KV pool (540,928 tokens) was smaller than a single 1M-token request, and six
attempts at a 1M profile either OOMed or crashed. GLM-5.3-Flash reports
max_model_len 1,048,576 with a 3,687,104-token pool, so it should fit -- but
"should fit" is arithmetic, not evidence.

This ramps 128K -> 512K -> 1M. At each length it runs a NEEDLE test: a unique
token is planted early in the filler and the model is asked to recall it at
the end. A request that merely returns without erroring proves the engine did
not crash; only the needle proves it actually read the context.

Writes RESULTS-glm53-longctx.md.
"""
import json, os, sys, time, urllib.request, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "RESULTS-glm53-longctx.md")
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "glm-5.3-flash"
NEEDLE = "MAGENTA-7731"
# Filler must be HETEROGENEOUS. A first version repeated one identical
# sentence; at 523,829 tokens the model then FAILED to find the needle (twice,
# burning a full 4096-token budget), while succeeding at 999,483. That is not a
# context-length limit -- it is what a degenerate input does to DSA sparse
# attention, whose indexer selects top-k blocks by relevance and has nothing to
# discriminate on when every block is byte-identical. With varied records below,
# both 512K and ~1M answer directly in <90 completion tokens.
#
# Chars-per-token is CALIBRATED at runtime, never assumed: the first version
# hardcoded 3.6 and every length silently landed at ~52% of target, so a "1M"
# run was really 548,006 tokens. Numeric-dense text here measures ~2.97.
NOUNS = ["depot", "warehouse", "terminal", "hub", "yard", "dock", "station", "annex"]
VERBS = ["recorded", "logged", "reported", "flagged", "noted", "registered", "filed", "posted"]

def _body(nchars, rng):
    parts, tot = [], 0
    while tot < nchars:
        s = (f"Record {rng.randint(10000,99999)}: the {rng.choice(NOUNS)} "
             f"{rng.choice(VERBS)} {rng.randint(100,9999)} units on shift "
             f"{rng.randint(1,4)} with variance {rng.random():.3f}. ")
        parts.append(s); tot += len(s)
    return "".join(parts)

def calibrate():
    """Measure chars/token against the live server rather than assuming it."""
    import random
    probe = _body(200_000, random.Random(7))
    usage, _, _ = ask(probe, max_tokens=8)
    return len(probe) / usage["prompt_tokens"]

def build(target_tokens, cpt):
    import random
    b = _body(int(target_tokens * cpt), random.Random(7))
    cut = len(b) // 10
    return (b[:cut]
            + f"\n\nIMPORTANT RECORD: The secret code is {NEEDLE}. Remember it.\n\n"
            + b[cut:])

def ask(prompt, max_tokens=64, timeout=1800):
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content":
                          prompt + "\n\nQuestion: What is the secret code stated in the record above? "
                                   "Reply with only the code."}],
            "max_tokens": max_tokens, "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    el = time.time() - t0
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    return d["usage"], text, el

def main():
    targets = [int(x) for x in (sys.argv[1:] or [131072, 524288, 1000000])]
    cpt = calibrate()
    print(f"calibrated: {cpt:.3f} chars/token", flush=True)
    with open(OUT, "w") as f:
        f.write(f"# GLM-5.3-Flash long-context verification\n\n")
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("Needle test: a unique code is planted at ~10% depth and recalled at the end.\n")
        f.write("A non-erroring response only proves no crash; the needle proves the context was read.\n\n")
        f.write(f"Calibrated at {cpt:.3f} chars/token against the live server.\n\n")
        f.write("| target | actual prompt_tokens | elapsed | needle found | verdict |\n|---|---|---|---|---|\n")
    for t in targets:
        label = f"{t//1024}K"
        print(f"=== {label} ===", flush=True)
        try:
            usage, text, el = ask(build(t, cpt), max_tokens=2048)
            found = NEEDLE in text
            row = (f"| {label} | {usage['prompt_tokens']:,} | {el:.1f}s | "
                   f"{'YES' if found else 'NO'} | {'PASS' if found else 'RESPONDED BUT NEEDLE MISSED'} |")
            print(f"{label}: prompt_tokens={usage['prompt_tokens']} needle={found} {el:.1f}s", flush=True)
        except Exception as e:
            row = f"| {label} | - | - | - | **FAILED**: {type(e).__name__} {str(e)[:120]} |"
            print(f"{label}: FAILED {e}", flush=True)
        with open(OUT, "a") as f:
            f.write(row + "\n")
    print(f"LONGCTX COMPLETE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
