#!/usr/bin/env python3
"""Concurrency-1 write-policy A/B for GLM-5.3-Flash: write_through vs
write_through_selective.

Why. The flash profile adopted write_through_selective on ONE sweep run
(RESULTS-glm53-sweep.md) where plain write_through gave 109 tok/s at
concurrency 1 against 207-247 for every other config. That sample came from a
benchmark later shown to be bimodal, and it contradicts RESULTS-kv-tiering.md
Task 5b, which tested both policies on GLM-5.2 and recommended against
selective. This settles it for GLM-5.3-Flash.

Design, from what earlier runs got wrong:
  - Concurrency 1 only. Across the nine HiCache A/B runs, conc-1 pass 2 held to
    214-243 tok/s while conc-32 swung 911-1537, so conc-1 is the one metric this
    harness can resolve. It is also how a single agent loop behaves.
  - L3 cleared AFTER the old worker is down and BEFORE the new one starts.
    ab_hicache.py cleared it while the old worker was still running and able
    to write entries in behind the clear.
  - Policies interleaved A-B, B-A, A-B so drift across the run cannot pose as a
    policy effect.
  - Flags identical to the committed flash profile except the write policy.

Also records every teardown: GLM-5.3-Flash WITH HiCache teardown is
unobserved (see the init: comment in docker-compose.yml), and each stop
exercises stop.sh's worker-flash log archiving under a real teardown.

Writes RESULTS-glm53-writepolicy.md.
"""
import datetime, json, os, re, shutil, statistics, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = os.path.join(HERE, "docker-compose.yml")
OVERRIDE = os.path.join(HERE, ".ab-override.yml")
OUT = os.path.join(HERE, "RESULTS-glm53-writepolicy.md")
L3 = "/scratch/kvcache/glm53"
URL = "http://127.0.0.1:8000"
CONTAINER = "glm52-sglang-worker-flash-1"
WT, WTS = "write_through", "write_through_selective"
ORDER = [(1, WT), (1, WTS), (2, WTS), (2, WT), (3, WT), (3, WTS)]  # ends on the committed default
PASS_RE = re.compile(r"=== bench pass (\d)/2.*?===(.*?)(?====|\Z)", re.S)


def sh(cmd, timeout=900):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def base_command():
    r = sh(f"docker compose -f {COMPOSE} --profile flash config --format json")
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout)["services"]["worker-flash"]["command"]


def with_policy(cmd, policy):
    out = [f"--hicache-write-policy={policy}" if c.startswith("--hicache-write-policy") else c
           for c in cmd]
    assert out.count(f"--hicache-write-policy={policy}") == 1, "write-policy flag not found once"
    return out


def running():
    r = sh("docker inspect " + CONTAINER + " --format '{{.State.Running}}'")
    return r.stdout.strip() == "true"


def exists():
    return CONTAINER in sh("docker ps -a --format '{{.Names}}'").stdout.split()


def clear_l3():
    removed = failed = 0
    for e in os.listdir(L3):
        p = os.path.join(L3, e)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            removed += 1
        except OSError:
            failed += 1
    return removed, failed


def restart(cmd):
    """Stop the previous worker, clear L3, start this arm. Returns what the
    teardown of the PREVIOUS worker did."""
    with open(OVERRIDE, "w") as f:
        json.dump({"services": {"worker-flash": {"command": cmd}}}, f)
    stop = sh(f"cd {HERE} && ./stop.sh", timeout=600)
    text = stop.stdout + stop.stderr
    info = {"first_down_failed": stop.returncode != 0 or "Error while Stopping" in text,
            "log_archived": "Archived" in text}
    for _ in range(60):
        if not running():
            break
        time.sleep(3)
    info["second_down_needed"] = exists()
    sh(f"docker compose -f {COMPOSE} --profile flash down", timeout=600)
    info["l3_removed"], info["l3_failed"] = clear_l3()
    up = sh(f"docker compose -f {COMPOSE} -f {OVERRIDE} --profile flash up -d", timeout=600)
    if up.returncode != 0:
        raise RuntimeError(f"up failed: {up.stderr[-400:]}")
    return info


def wait_ready(limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        if sh(f"curl -sf -m 3 {URL}/v1/models").returncode == 0:
            return True, time.time() - t0
        if not running():
            return False, time.time() - t0
        time.sleep(10)
    return False, limit


def bench():
    r = sh(f"cd {HERE} && ./bench_stream.py --concurrency 1 --num 8 "
           f"--shared-prefix-tokens 2000 --prefix-seed 42 --passes 2", timeout=1800)
    res = {}
    for pn, body in PASS_RE.findall(r.stdout):
        def num(pat):
            m = re.search(pat, body)
            return float(m.group(1)) if m else None
        res[f"p{pn}_sys"] = num(r"system output tok/s\s*:\s*([\d.]+)")
        res[f"p{pn}_decode"] = num(r"decode tok/s: mean\s*([\d.]+)")
        res[f"p{pn}_itl_mean"] = num(r"ITL\s+mean/p50/p99\s*:\s*([\d.]+)")
        res[f"p{pn}_itl_p99"] = num(r"ITL\s+mean/p50/p99\s*:\s*[\d.]+\s*/\s*[\d.]+\s*/\s*([\d.]+)")
        m = re.search(r"requests ok/err\s*:\s*(\d+)/(\d+)", body)
        res[f"p{pn}_err"] = int(m.group(2)) if m else None
    return res


def fmt(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.1f}"
    return f"{statistics.mean(v):.1f} ({min(v):.1f}-{max(v):.1f})"


def main():
    base = base_command()
    rows = []
    with open(OUT, "w") as f:
        f.write("# GLM-5.3-Flash: write_through vs write_through_selective (concurrency 1)\n\n")
        f.write(f"Started {datetime.datetime.now().isoformat(timespec='seconds')}. "
                "Method and rationale: docstring of `ab_writepolicy.py`.\n\n")
        f.write("Prior evidence being tested: one sweep run (RESULTS-glm53-sweep.md), "
                "conc-1 pass-2 `write_through` 109.0 vs `write_through_selective` 207.9 tok/s; "
                "and RESULTS-kv-tiering.md Task 5b (GLM-5.2), which found selective no better.\n\n")
        f.write("Order: " + ", ".join(f"rep{r} {p}" for r, p in ORDER) + "\n\n")
        f.write("| rep | policy | teardown of previous worker | L3 cleared (ok/failed) | boot | "
                "p1 sys | p2 sys | p2 decode | p2 ITL mean | p2 ITL p99 | errors |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for rep, policy in ORDER:
        label = f"rep{rep} {policy}"
        print(f"=== {label} ===", flush=True)
        try:
            td = restart(with_policy(base, policy))
            ok, secs = wait_ready()
        except Exception as e:
            print(f"{label}: RESTART FAILED {e}", flush=True)
            rows.append((rep, policy, None))
            continue
        teardown = "clean" if not td["first_down_failed"] else (
            "first down FAILED" + ("; second down needed" if td["second_down_needed"] else ""))
        teardown += "; log archived" if td["log_archived"] else "; log NOT archived"
        cleared = f"{td['l3_removed']}/{td['l3_failed']}"
        if not ok:
            print(f"{label}: BOOT FAILED after {secs:.0f}s", flush=True)
            with open(OUT, "a") as f:
                f.write(f"| {rep} | {policy} | {teardown} | {cleared} | FAILED {secs:.0f}s "
                        "| - | - | - | - | - | - |\n")
            rows.append((rep, policy, None))
            continue
        b = bench()
        rows.append((rep, policy, b))
        cell = lambda k: "-" if b.get(k) is None else f"{b[k]:.1f}"
        errs = (b.get("p1_err") or 0) + (b.get("p2_err") or 0)
        with open(OUT, "a") as f:
            f.write(f"| {rep} | {policy} | {teardown} | {cleared} | {secs:.0f}s | {cell('p1_sys')} | "
                    f"{cell('p2_sys')} | {cell('p2_decode')} | {cell('p2_itl_mean')} | "
                    f"{cell('p2_itl_p99')} | {errs} |\n")
        print(f"{label}: done p2_sys={b.get('p2_sys')} p2_itl_p99={b.get('p2_itl_p99')}", flush=True)
    with open(OUT, "a") as f:
        f.write("\n## Summary: pass 2, concurrency 1 (mean, then min-max)\n\n")
        f.write("| policy | n | sys tok/s | decode tok/s | ITL p99 ms |\n|---|---|---|---|---|\n")
        sysv = {}
        for pol in (WT, WTS):
            bs = [b for _, p, b in rows if p == pol and b]
            sysv[pol] = [b["p2_sys"] for b in bs if b.get("p2_sys") is not None]
            f.write(f"| `{pol}` | {len(bs)} | {fmt([b.get('p2_sys') for b in bs])} | "
                    f"{fmt([b.get('p2_decode') for b in bs])} | {fmt([b.get('p2_itl_p99') for b in bs])} |\n")
        a, s = sysv[WT], sysv[WTS]
        if a and s:
            overlap = not (max(a) < min(s) or max(s) < min(a))
            f.write(f"\nsys tok/s ranges **{'OVERLAP' if overlap else 'DO NOT OVERLAP'}**: "
                    f"`{WT}` {min(a):.1f}-{max(a):.1f}, `{WTS}` {min(s):.1f}-{max(s):.1f}. "
                    "Non-overlapping ranges at n=3 each is the bar for calling a difference real; "
                    "overlapping ranges mean this harness cannot tell the policies apart.\n")
        f.write("\nThe worker is left running the last arm (`write_through_selective`), which "
                "matches the committed flash profile.\n")
    if os.path.exists(OVERRIDE):
        os.remove(OVERRIDE)
    print(f"WRITEPOLICY COMPLETE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
