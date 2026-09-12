#!/usr/bin/env python3
"""GLM-5.3-Flash config sweep: HiCache write policy and mamba SSM dtype.

Each config is a full worker restart. For each one we record the two pool
sizes the engine prints at boot (they are what actually gate concurrency)
and then benchmark at two concurrency levels.

Writes RESULTS-glm53-sweep.md. Run from sglang/.
"""
import json, os, re, subprocess, sys, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = os.path.join(HERE, "docker-compose.yml")
OVERRIDE = os.path.join(HERE, ".sweep-override.yml")
OUT = os.path.join(HERE, "RESULTS-glm53-sweep.md")
URL = "http://127.0.0.1:8000"
HICACHE_PREFIXES = ("--enable-hierarchical-cache", "--hicache-")

def sh(cmd, timeout=900, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r

def base_command():
    r = sh(f"docker compose -f {COMPOSE} --profile flash config --format json", check=True)
    return json.loads(r.stdout)["services"]["worker-flash"]["command"]

def build(cmd, *, hicache=True, write_policy=None, ssm_dtype=None, extra=()):
    out = []
    for c in cmd:
        if not hicache and c.startswith(HICACHE_PREFIXES):
            continue
        if write_policy and c.startswith("--hicache-write-policy"):
            c = f"--hicache-write-policy={write_policy}"
        out.append(c)
    if ssm_dtype:
        out.append(f"--mamba-ssm-dtype={ssm_dtype}")
    out.extend(extra)
    return out

def restart(cmd):
    with open(OVERRIDE, "w") as f:
        json.dump({"services": {"worker-flash": {"command": cmd}}}, f)
    sh(f"cd {HERE} && ./stop.sh", timeout=600)
    for _ in range(60):
        s = sh("docker inspect glm52-sglang-worker-flash-1 --format '{{.State.Status}}'").stdout.strip()
        if s != "running":
            break
        time.sleep(3)
    sh(f"docker compose -f {COMPOSE} --profile flash down", timeout=600)
    sh(f"docker compose -f {COMPOSE} -f {OVERRIDE} --profile flash up -d", timeout=600, check=True)

def wait_ready(limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        if sh(f"curl -sf -m 3 {URL}/v1/models").returncode == 0:
            return True, time.time() - t0
        if sh("docker ps --format '{{.Names}}'").stdout.count("worker-flash") == 0:
            return False, time.time() - t0
        time.sleep(10)
    return False, limit

def pools():
    log = sh(f"docker compose -f {COMPOSE} logs worker-flash", timeout=300).stdout
    d = {}
    m = re.search(r"max_mamba_cache_size:\s*(\d+)", log)
    if m: d["max_mamba_cache_size"] = int(m.group(1))
    m = re.search(r"ssm_state size:\s*([\d.]+)GB", log)
    if m: d["ssm_state_GB"] = float(m.group(1))
    m = re.search(r"conv_state size:\s*([\d.]+)GB", log)
    if m: d["conv_state_GB"] = float(m.group(1))
    mt = sh(f"curl -s -m 5 {URL}/metrics").stdout
    m = re.search(r"^sglang:max_total_num_tokens\{[^}]*\}\s+([\d.e+]+)", mt, re.M)
    if m: d["max_total_num_tokens"] = int(float(m.group(1)))
    d["io_backend_downgraded"] = "downgrad" in log.lower()
    return d

def bench(conc, num):
    r = sh(f"cd {HERE} && ./bench_stream.py --concurrency {conc} --num {num} "
           f"--shared-prefix-tokens 2000 --prefix-seed 42 --passes 2", timeout=2400)
    return r.stdout + r.stderr

CONFIGS = [
    ("hicache-off",       dict(hicache=False)),
    ("wt (current)",      dict(write_policy="write_through")),
    ("wt-selective",      dict(write_policy="write_through_selective")),
    ("wts+ssm-float32",   dict(write_policy="write_through_selective", ssm_dtype="float32")),
    ("wts+ssm-bfloat16",  dict(write_policy="write_through_selective", ssm_dtype="bfloat16")),
]

def main():
    base = base_command()
    extra = ["--enable-mfu-metrics"]
    with open(OUT, "w") as f:
        f.write(f"# GLM-5.3-Flash sweep\n\nStarted {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("Each row is a full worker restart. `--enable-mfu-metrics` added to every run.\n\n")
    for name, kw in CONFIGS:
        print(f"=== {name} ===", flush=True)
        cmd = build(base, extra=extra, **kw)
        try:
            restart(cmd)
            ok, secs = wait_ready()
        except Exception as e:
            ok, secs = False, 0
            print(f"{name}: restart failed {e}", flush=True)
        with open(OUT, "a") as f:
            f.write(f"\n## {name}\n\n")
            if not ok:
                f.write(f"**FAILED TO BOOT** after {secs:.0f}s\n")
                log = sh(f"docker compose -f {COMPOSE} logs worker-flash", timeout=300).stdout
                errs = [l for l in log.splitlines()
                        if re.search(r"error|traceback|assert|not supported", l, re.I)][-8:]
                f.write("```\n" + "\n".join(errs) + "\n```\n")
                print(f"{name}: FAILED", flush=True)
                continue
            p = pools()
            f.write(f"boot time: {secs:.0f}s\n\npools: `{p}`\n\n")
            for conc, num in ((1, 8), (32, 160)):
                f.write(f"### concurrency {conc}\n\n```\n{bench(conc, num)}\n```\n")
        print(f"{name}: done ({secs:.0f}s)", flush=True)
    if os.path.exists(OVERRIDE):
        os.remove(OVERRIDE)
    print(f"SWEEP COMPLETE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
