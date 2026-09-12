#!/usr/bin/env python3
"""Clean HiCache A/B for GLM-5.3-Flash.

The first sweep could not resolve HiCache on-vs-off: two runs of an identical
config differed by 24%, which swamped the ~7% difference being measured. Two
fixes here:

  1. REPS runs per arm, so the spread is visible instead of assumed.
  2. /scratch/kvcache/glm53 is cleared before each arm, so a later arm cannot
     inherit L3 entries written by an earlier one (the confound that made the
     first sweep's pass-1 numbers meaningless).

Three arms per rep:
  off        - hicache disabled, L3 cleared
  on-cold    - hicache enabled, L3 cleared   (no restart benefit)
  on-warm    - hicache enabled, L3 KEPT from on-cold, worker restarted
               (this is the one thing HiCache genuinely buys: prefix reuse
                that survives a restart)

Writes RESULTS-glm53-hicache-ab.md.
"""
import json, os, re, shutil, subprocess, time, datetime, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = os.path.join(HERE, "docker-compose.yml")
OVERRIDE = os.path.join(HERE, ".ab-override.yml")
OUT = os.path.join(HERE, "RESULTS-glm53-hicache-ab.md")
L3 = "/scratch/kvcache/glm53"
URL = "http://127.0.0.1:8000"
HICACHE_PREFIXES = ("--enable-hierarchical-cache", "--hicache-")
REPS = 3

def sh(cmd, timeout=900):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def base_command():
    r = sh(f"docker compose -f {COMPOSE} --profile flash config --format json")
    return json.loads(r.stdout)["services"]["worker-flash"]["command"]

def clear_l3():
    n = 0
    if os.path.isdir(L3):
        for e in os.listdir(L3):
            p = os.path.join(L3, e)
            try:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p); n += 1
            except OSError:
                pass
    return n

def restart(cmd):
    with open(OVERRIDE, "w") as f:
        json.dump({"services": {"worker-flash": {"command": cmd}}}, f)
    sh(f"cd {HERE} && ./stop.sh", timeout=600)
    for _ in range(60):
        if sh("docker inspect glm52-sglang-worker-flash-1 --format '{{.State.Status}}'").stdout.strip() != "running":
            break
        time.sleep(3)
    sh(f"docker compose -f {COMPOSE} --profile flash down", timeout=600)
    sh(f"docker compose -f {COMPOSE} -f {OVERRIDE} --profile flash up -d", timeout=600)

def wait_ready(limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        if sh(f"curl -sf -m 3 {URL}/v1/models").returncode == 0:
            return True
        if "worker-flash" not in sh("docker ps --format '{{.Names}}'").stdout:
            return False
        time.sleep(10)
    return False

def bench(conc, num):
    out = sh(f"cd {HERE} && ./bench_stream.py --concurrency {conc} --num {num} "
             f"--shared-prefix-tokens 2000 --prefix-seed 42 --passes 2", timeout=2400).stdout
    res = {}
    for pn, body in re.findall(r"=== bench pass (\d)/2.*?===(.*?)(?====|\Z)", out, re.S):
        m = re.search(r"system output tok/s\s*:\s*([\d.]+)", body)
        t = re.search(r"TTFT  mean/p50/p99  : (\d+)", body)
        if m: res[f"p{pn}_sys"] = float(m.group(1))
        if t: res[f"p{pn}_ttft"] = int(t.group(1))
    return res

def main():
    base = base_command()
    on = [c for c in base]
    off = [c for c in base if not c.startswith(HICACHE_PREFIXES)]
    rows = []
    with open(OUT, "w") as f:
        f.write(f"# GLM-5.3-Flash: HiCache A/B\n\nStarted {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"\n{REPS} reps/arm. L3 cleared before `off` and `on-cold`; kept for `on-warm`.\n\n")
    for rep in range(1, REPS + 1):
        for arm, cmd, clear in (("off", off, True), ("on-cold", on, True), ("on-warm", on, False)):
            n = clear_l3() if clear else -1
            restart(cmd)
            if not wait_ready():
                print(f"rep{rep} {arm}: BOOT FAILED", flush=True)
                rows.append((rep, arm, None, None)); continue
            b1, b32 = bench(1, 8), bench(32, 160)
            rows.append((rep, arm, b1, b32))
            with open(OUT, "a") as f:
                f.write(f"- rep{rep} **{arm}** (l3 cleared: {n if n>=0 else 'no'}) "
                        f"conc1 p1/p2 {b1.get('p1_sys')}/{b1.get('p2_sys')} | "
                        f"conc32 p1/p2 {b32.get('p1_sys')}/{b32.get('p2_sys')} tok/s\n")
            print(f"rep{rep} {arm}: done", flush=True)
    # summary
    with open(OUT, "a") as f:
        f.write("\n## Summary (system tok/s, mean +/- spread)\n\n")
        f.write("| arm | conc1 p1 | conc1 p2 | conc32 p1 | conc32 p2 |\n|---|---|---|---|---|\n")
        for arm in ("off", "on-cold", "on-warm"):
            vals = {k: [r[2 if "conc1" in k else 3].get(k.split("|")[1])
                        for r in rows if r[1] == arm and r[2]]
                    for k in ("conc1|p1_sys", "conc1|p2_sys", "conc32|p1_sys", "conc32|p2_sys")}
            cells = []
            for k in ("conc1|p1_sys", "conc1|p2_sys", "conc32|p1_sys", "conc32|p2_sys"):
                v = [x for x in vals[k] if x]
                cells.append(f"{statistics.mean(v):.0f} ± {(max(v)-min(v))/2:.0f}" if len(v) > 1
                             else (f"{v[0]:.0f}" if v else "-"))
            f.write(f"| {arm} | " + " | ".join(cells) + " |\n")
    if os.path.exists(OVERRIDE):
        os.remove(OVERRIDE)
    print(f"AB COMPLETE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
