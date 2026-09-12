#!/usr/bin/env python3
"""Capability smoke tests for the served model. Run against a live worker.

These exist because the flash profile runs a fast-moving unreleased engine
build (lmsysorg/sglang:glm-5.3-flash, a dev build off sglang main) against a
days-old architecture with an active upstream bug tracker. An image bump can
break any of the below silently -- a tool-call parser mismatch in particular
produces plausible prose instead of a tool_calls field, with no error.

Covers the paths an agent framework actually uses: tool calling (single, multi,
and the result round-trip), STREAMED tool-call deltas, reasoning/content
separation, JSON mode, vision, and that the model's preferred sampling params
are advertised.

    ./test_inference.py            # all
    ./test_inference.py --quiet    # one line per test

Exit code is the number of failures, so it works as a CI gate.
"""
import argparse, base64, json, struct, sys, time, urllib.request, zlib

BASE = "http://127.0.0.1:8000"
MODEL = "glm-5.3-flash"
RESULTS = []

def post(path, payload, timeout=300, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    out = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: "):
            body = line[6:]
            if body == "[DONE]":
                break
            out.append(json.loads(body))
    return out

def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())

def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return ok

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
POP = {"type": "function", "function": {
    "name": "get_population", "description": "Get city population",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}

def quad_png():
    """4-colour PNG built inline -- no image fixtures, no pillow dependency."""
    W = H = 64
    def px(x, y):
        if x < W//2 and y < H//2: return (220, 30, 30)
        if x >= W//2 and y < H//2: return (30, 120, 220)
        if x < W//2 and y >= H//2: return (240, 200, 40)
        return (40, 170, 70)
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(W)) for y in range(H))
    def ck(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            + ck(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + ck(b"IDAT", zlib.compress(raw)) + ck(b"IEND", b""))

def t_model_info():
    d = get("/get_model_info")
    psp = d.get("preferred_sampling_params")
    # Must be non-null: without it, clients omitting temperature/top_p get
    # engine defaults instead of the checkpoint's generation_config values.
    check("model_info: preferred_sampling_params set", psp not in (None, "", {}), str(psp))
    check("model_info: tool_call_parser", d.get("tool_call_parser") == "glm47", str(d.get("tool_call_parser")))
    check("model_info: reasoning_parser", d.get("reasoning_parser") == "glm45", str(d.get("reasoning_parser")))

def t_tool_single():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "user", "content": "What's the weather in Tokyo in celsius?"}],
        "tools": [WEATHER]})
    m = d["choices"][0]["message"]; tc = m.get("tool_calls")
    if not check("tool call: returned", bool(tc), (m.get("content") or "")[:80]):
        return None
    fn = tc[0]["function"]
    check("tool call: right function", fn["name"] == "get_weather", fn["name"])
    try:
        args = json.loads(fn["arguments"])
        check("tool call: args are valid JSON", "Tokyo" in str(args.get("city")), str(args))
    except Exception as e:
        check("tool call: args are valid JSON", False, f"{e}: {fn['arguments'][:80]}")
    return m

def t_tool_roundtrip(first):
    """The core agent loop: send a tool result back and get a grounded answer."""
    if not first or not first.get("tool_calls"):
        return check("tool round-trip", False, "no first-turn tool call")
    tc = first["tool_calls"][0]
    msgs = [{"role": "user", "content": "What's the weather in Tokyo in celsius?"},
            {"role": "assistant", "tool_calls": first["tool_calls"], "content": first.get("content") or ""},
            {"role": "tool", "tool_call_id": tc["id"],
             "content": json.dumps({"temp_c": 18, "conditions": "light rain"})}]
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 512,
                                      "messages": msgs, "tools": [WEATHER]})
    out = (d["choices"][0]["message"].get("content") or "")
    check("tool round-trip: result used", "18" in out and "rain" in out.lower(), out.strip()[:90])

def t_tool_multi():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "user", "content": "Compare the weather AND population of Paris. Use the tools."}],
        "tools": [WEATHER, POP]})
    tcs = d["choices"][0]["message"].get("tool_calls") or []
    names = sorted(t["function"]["name"] for t in tcs)
    check("tool call: multi-tool selection", names == ["get_population", "get_weather"], str(names))

def t_reasoning():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 1024,
        "messages": [{"role": "user", "content": "A train leaves at 14:05 and takes 97 minutes. Arrival time?"}]})
    m = d["choices"][0]["message"]
    rc, c = m.get("reasoning_content") or "", m.get("content") or ""
    check("reasoning: separated from content", bool(rc) and bool(c), f"{len(rc)}/{len(c)} chars")
    check("reasoning: answer correct", "15:42" in c, c.strip()[:60])

def t_json_mode():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "Return JSON with keys name, year for Apollo 11."}]})
    c = (d["choices"][0]["message"].get("content") or "").strip()
    try:
        check("json mode: valid JSON", json.loads(c).get("year") == 1969, c[:80])
    except Exception as e:
        check("json mode: valid JSON", False, f"{e}: {c[:80]}")

def t_vision():
    b64 = base64.b64encode(quad_png()).decode()
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 512,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Four coloured quadrants. Name the colour in each: "
                                     "top-left, top-right, bottom-left, bottom-right."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
    c = (d["choices"][0]["message"].get("content") or "").lower()
    hits = [k for k in ("red", "blue", "yellow", "green") if k in c]
    check("vision: all four colours", len(hits) == 4, f"found {hits}")

def t_stream():
    t0 = time.time()
    ch = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 256, "stream": True,
        "messages": [{"role": "user", "content": "Count 1 to 5, then say done."}]}, stream=True)
    dc = sum(1 for c in ch if (c["choices"][0].get("delta") or {}).get("content"))
    check("streaming: content deltas", dc > 1, f"{len(ch)} chunks, {dc} content, {time.time()-t0:.1f}s")

def t_stream_tools():
    """Agent frameworks consume tool calls as incremental deltas; they must
    reassemble into parseable JSON."""
    tools = [{"type": "function", "function": {"name": "search_docs",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}]
    ch = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 256, "stream": True,
        "messages": [{"role": "user", "content": "Search the docs for 'retry policy'."}],
        "tools": tools}, stream=True)
    name = args = ""
    for c in ch:
        for td in ((c["choices"][0].get("delta") or {}).get("tool_calls") or []):
            fn = td.get("function") or {}
            name += fn.get("name") or ""
            args += fn.get("arguments") or ""
    fin = [c["choices"][0].get("finish_reason") for c in ch if c["choices"][0].get("finish_reason")]
    check("streamed tools: finish_reason", "tool_calls" in fin, str(fin))
    try:
        check("streamed tools: deltas reassemble", bool(json.loads(args).get("query")), f"{name}({args})")
    except Exception as e:
        check("streamed tools: deltas reassemble", False, f"{e}: {name}({args[:60]})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        get("/v1/models", timeout=10)
    except Exception as e:
        print(f"server not reachable at {BASE}: {e}", file=sys.stderr)
        return 1
    t_model_info()
    first = t_tool_single()
    t_tool_roundtrip(first)
    t_tool_multi()
    t_reasoning()
    t_json_mode()
    t_vision()
    t_stream()
    t_stream_tools()
    fails = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, detail in RESULTS:
        if a.quiet and ok:
            continue
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    print(f"\n{len(RESULTS)-fails}/{len(RESULTS)} passed")
    return fails

if __name__ == "__main__":
    sys.exit(main())
