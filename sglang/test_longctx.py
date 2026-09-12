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
# ~3.6 chars/token for this filler; corrected against reported prompt_tokens.
FILLER = ("The quarterly logistics report notes routine depot activity and "
          "no exceptions worth escalating to the regional coordinator. ")

def build(target_tokens):
    approx_chars = int(target_tokens * 3.6)
    reps = max(1, approx_chars // len(FILLER))
    body = FILLER * reps
    cut = len(body) // 10
    return (body[:cut]
            + f"\n\nIMPORTANT RECORD: The secret code is {NEEDLE}. Remember it.\n\n"
            + body[cut:])

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
    targets = [int(x) for x in (sys.argv[1:] or [131072, 524288, 1048576])]
    with open(OUT, "w") as f:
        f.write(f"# GLM-5.3-Flash long-context verification\n\n")
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("Needle test: a unique code is planted at ~10% depth and recalled at the end.\n")
        f.write("A non-erroring response only proves no crash; the needle proves the context was read.\n\n")
        f.write("| target | actual prompt_tokens | elapsed | needle found | verdict |\n|---|---|---|---|---|\n")
    for t in targets:
        label = f"{t//1024}K"
        print(f"=== {label} ===", flush=True)
        try:
            usage, text, el = ask(build(t))
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
