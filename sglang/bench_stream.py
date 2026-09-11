#!/usr/bin/env python3
"""Tiny streaming benchmark for the GLM-5.2-FP8 SGLang OpenAI endpoint.

Stdlib only (no aiperf/genai-perf/tokenizer needed — those aren't in the runtime
image and an offline environment can't fetch a corpus for sglang.bench_serving). Measures
TTFT, inter-token latency (ITL), end-to-end latency and decode throughput by
streaming /v1/chat/completions with usage accounting. Designed to compare MTP
speculative decoding OFF vs ON: run it before and after enabling the EAGLE flags.

Usage:
    ./bench_stream.py --concurrency 1  --num 16 --max-tokens 256
    ./bench_stream.py --concurrency 32 --num 128 --max-tokens 256
    BASE_URL=http://localhost:8000 ./bench_stream.py --tag mtp-off
    ./bench_stream.py --concurrency 1 --num 1 --passes 2 \
        --shared-prefix-tokens 131072 --max-tokens 32   # prefix-cache reuse
"""
import argparse
import json
import os
import random
import statistics as stats
import sys
import threading
import time
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "glm-5.2-fp8")

# Generous on purpose: a cold 131K-token prefill legitimately takes ~20 s
# (see --shared-prefix-tokens runs), and a large --max-tokens decode pass at
# high concurrency can run for minutes. Without a timeout at all, one hung
# request wedges urlopen()'s thread forever and the whole benchmark pass
# (run_pass joins every thread) never returns. Override via env for unusually
# slow scenarios; do not set this below ~300s.
REQUEST_TIMEOUT_S = float(os.environ.get("BENCH_STREAM_TIMEOUT_S", "300"))

# A fixed, deterministic prompt so OFF vs ON see identical work. Asks for a
# sized output so decode (where MTP helps) dominates over prefill.
PROMPT = (
    "You are benchmarking a language model server. Write a detailed, continuous "
    "technical explanation of how tensor parallelism, MoE expert routing, and "
    "speculative decoding interact on an 8-GPU inference server. Keep writing "
    "until you are asked to stop; do not use bullet lists."
)

# Deterministic filler for --shared-prefix-tokens. Short, common words so most
# map to a single token; the true size is whatever the server reports as
# prompt_tokens, which we print. The flag is a target, not a guarantee -- there
# is no tokenizer in this environment to make it exact.
_FILLER_VOCAB = (
    "system model server memory cache token layer batch request tensor kernel "
    "expert router weight buffer stream decode prefill context window latency "
    "throughput device pool page index sparse dense attention query value state"
).split()


def make_shared_prefix(approx_tokens, seed):
    """Build a deterministic filler paragraph of roughly approx_tokens tokens.

    Same seed => byte-identical text, so a run after a worker restart requests
    the exact prefix the previous run cached. That is what makes L3 persistence
    testable at all.
    """
    if approx_tokens <= 0:
        return ""
    rng = random.Random(seed)
    return " ".join(rng.choice(_FILLER_VOCAB) for _ in range(approx_tokens))


def build_prompt(shared_prefix, pass_idx, req_idx):
    """Shared prefix + a suffix unique to this (pass, request).

    Unique suffixes keep the radix cache hitting on the PREFIX only. Without
    them a second pass would hit a whole-request cache entry and overstate
    reuse, which is the easiest way to fool this benchmark.
    """
    if not shared_prefix:
        return PROMPT
    return (
        f"{shared_prefix}\n\n"
        f"Given the reference text above, answer question {req_idx} "
        f"in series {pass_idx}: {PROMPT}"
    )


def build_body(prompt, max_tokens, no_think):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if no_think:
        # GLM honours an explicit no-think hint; keeps output to plain content.
        body["messages"].insert(0, {"role": "system", "content": "/nothink Reply directly."})
    return body


def one_request(prompt, max_tokens, no_think):
    data = json.dumps(build_body(prompt, max_tokens, no_think)).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    last = t0
    itls = []
    completion_tokens = None
    prompt_tokens = None
    cached_tokens = None
    chunk_count = 0
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                # OpenAI-shaped cache accounting -- a DIRECT read of prefix-cache
                # hits rather than a TTFT inference, worth far more than the
                # timing numbers for verifying the tiering actually works.
                # The server populates this because --enable-cache-report is set
                # in docker-compose.yml; WITHOUT that flag the field is silently
                # absent and this reads None.
                cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                chunk_count += 1
    total = time.perf_counter() - t0
    out_tok = completion_tokens if completion_tokens else chunk_count
    return {
        "ttft": ttft if ttft is not None else total,
        "total": total,
        "out_tok": out_tok,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "itls": itls,
        "decode_tps": (out_tok - 1) / (total - (ttft or 0)) if total > (ttft or 0) and out_tok > 1 else 0.0,
    }


def run_pass(args, shared_prefix, pass_idx):
    """Fire args.num requests at args.concurrency. Returns (results, errors, wall)."""
    results = []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)
    errors = [0]

    def worker(req_idx):
        with sem:
            try:
                prompt = build_prompt(shared_prefix, pass_idx, req_idx)
                r = one_request(prompt, args.max_tokens, args.no_think)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                sys.stderr.write(f"req error: {e}\n")

    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.num)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors[0], time.perf_counter() - wall0


def pct(xs, p):
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return sorted(xs)[i]


def summarize(results, errors, wall, label, args):
    ttfts = sorted(r["ttft"] for r in results)
    decode_tps = [r["decode_tps"] for r in results if r["decode_tps"] > 0]
    all_itls = [x for r in results for x in r["itls"]]
    tot_out = sum(r["out_tok"] for r in results)

    print(f"\n=== bench {label}  conc={args.concurrency} num={args.num} "
          f"max_tokens={args.max_tokens} ===")
    print(f"requests ok/err     : {len(results)}/{errors}")
    print(f"wall time           : {wall:.2f} s")
    print(f"TTFT  mean/p50/p99  : {stats.mean(ttfts)*1000:.0f} / {pct(ttfts,50)*1000:.0f} "
          f"/ {pct(ttfts,99)*1000:.0f} ms")
    if all_itls:
        print(f"ITL   mean/p50/p99  : {stats.mean(all_itls)*1000:.1f} / {pct(all_itls,50)*1000:.1f} "
              f"/ {pct(all_itls,99)*1000:.1f} ms")
    if decode_tps:
        print(f"per-req decode tok/s: mean {stats.mean(decode_tps):.1f}  "
              f"(min {min(decode_tps):.1f}, max {max(decode_tps):.1f})")
    prompt_toks = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    if prompt_toks:
        mean_prompt = stats.mean(prompt_toks)
        print(f"prompt tokens mean  : {mean_prompt:.0f}")
        cached = [r["cached_tokens"] for r in results if r["cached_tokens"] is not None]
        if cached:
            print(f"cached tokens mean  : {stats.mean(cached):.0f} "
                  f"({stats.mean(cached)/mean_prompt*100:.1f}% of prompt)")
    print(f"output tokens total : {tot_out}")
    print(f"system output tok/s : {tot_out / wall:.1f}")
    return stats.mean(ttfts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help="Prepend an identical ~N-token filler prefix to every request "
                         "(0 = off, use the fixed PROMPT). Exercises prefix-cache reuse.")
    ap.add_argument("--prefix-seed", type=int, default=1234,
                    help="Seed for the shared prefix. The same seed reproduces a "
                         "byte-identical prefix, so a run after a worker restart tests "
                         "whether the L3 cache survived.")
    ap.add_argument("--passes", type=int, default=1,
                    help="Run the request set this many times. Pass 1 is cold; later "
                         "passes should hit the cache. Reported separately.")
    args = ap.parse_args()

    shared_prefix = make_shared_prefix(args.shared_prefix_tokens, args.prefix_seed)
    if shared_prefix:
        print(f"shared prefix: ~{args.shared_prefix_tokens} tokens, seed {args.prefix_seed}, "
              f"{len(shared_prefix)} chars")

    mean_ttfts = []
    for p in range(1, args.passes + 1):
        results, errors, wall = run_pass(args, shared_prefix, p)
        if not results:
            print(f"pass {p}: no successful requests")
            sys.exit(1)
        label = f"{args.tag or ''} pass {p}/{args.passes}".strip()
        mean_ttfts.append(summarize(results, errors, wall, label, args))

    if len(mean_ttfts) > 1:
        print("\n=== TTFT by pass (mean ms) ===")
        for i, t in enumerate(mean_ttfts, 1):
            print(f"pass {i}: {t*1000:.0f}")
        if mean_ttfts[1] > 0:
            print(f"pass1/pass2 speedup : {mean_ttfts[0]/mean_ttfts[1]:.2f}x")


if __name__ == "__main__":
    main()
