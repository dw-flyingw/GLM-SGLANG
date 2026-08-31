# KV Cache Tiering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the GLM-5.2-FP8 Dynamo/SGLang stack a tiered KV cache — GPU radix → host RAM (L2) → `/scratch` NVMe (L3) — plus a second, switchable profile that trades prefix caching for ~1M context.

**Architecture:** Two Docker Compose profiles over one worker image. `PROFILE=cache` (default) enables SGLang HiCache with the `file` storage backend pointed at `/scratch`; `PROFILE=longctx` enables SGLang HiSparse instead. They are mutually exclusive in SGLang, so exactly one worker service runs at a time, selected by native Compose profiles. A byte-budget reaper keeps the unbounded `file` backend from filling `/scratch`.

**Tech Stack:** Docker Compose, NVIDIA Dynamo + SGLang 0.5.13.post1, Python 3.11+ stdlib, pytest (dev only), bash.

**Spec:** `docs/superpowers/specs/2026-08-31-scratch-kv-cache-tiering-design.md` — read it before starting. It carries the source-level evidence (file:line) for every claim the config depends on.

## Global Constraints

- **Python is stdlib-only.** `pyproject.toml` declares `dependencies = []` and `bench_stream.py` is deliberately stdlib-only (the runtime image has no aiperf/genai-perf and the environment is offline). pytest is a dev tool; do not add it, or anything else, to `pyproject.toml` dependencies.
- **`requires-python = ">=3.11"`.** The host runs 3.12.3.
- **Do not change these worker args:** `--tp-size 8`, `--page-size 64`, `--kv-cache-dtype fp8_e4m3`, and the absence of `--attention-backend` (SGLang must keep auto-selecting `dsa`/`flashmla_kv`).
- **Profile A must be removable by deleting flags.** No change may make the current configuration unreachable.
- **Never commit `dynamo/.env`** — it is gitignored and holds the host-specific `HF_CACHE=/data/huggingface`.
- **Docs are corrected only after the corresponding benchmark passes** (Task 10). Do not pre-emptively edit README claims.
- **`/scratch` is root-owned and `sudo` needs a password**, so directory creation is a human step (Task 4), never automated.
- Exact values that must appear verbatim: `--hicache-size` default `96`, reaper budget default `10TB`, `top_k` `2048`, `host_to_device_ratio` `2`, `device_buffer_size` `4096`, longctx `--mem-fraction-static` `0.88`.

## Amendments during execution

The plan changed while it was being executed. Each entry says what changed and why, so the
committed artifact matches what was actually built.

1. **Task 3b inserted** (complete, `592d743`). The baseline showed Task 5's original
   success criterion was already satisfied before any change. See the Task 3b section for
   the full reasoning and the measured control. Task 5 Steps 4b and 5 were rewritten
   around it. Approved by the human partner.
2. **Baseline numbers corrected.** Measured conc-32 throughput is **2087.1 tok/s** and
   conc-1 decode **150.6 tok/s** (Task 3, `2a01582`). The 1975.0 / 150.0 figures quoted in
   `dynamo/README.md` are stale; Task 10 corrects them. Compare against the measured
   values, not the documented ones.
3. **`cached_tokens` is available.** Task 2's smoke test confirmed the Dynamo frontend
   populates `usage.prompt_tokens_details.cached_tokens`. Every cache-related criterion
   should lead with that direct count and treat TTFT as corroboration, since timing is
   noisy and load-dependent. This was not known when the plan was written.
4. **YAML anchor for the two worker services** (human ruling). Tasks 4 and 8 use
   `x-worker-base: &worker-base` carrying the eight identical service keys (`image`,
   `network_mode`, `gpus`, `ipc`, `ulimits`, `depends_on`, `entrypoint`, `restart`), with
   each service spelling out only its own `profiles`, `environment`, `volumes`, and
   `command`. Task 4 introduces the anchor since it already edits `worker`; Task 8
   consumes it. This overrides the fully-duplicated service block written out in Task 8.
5. **Task 1's exception branches diverge from the code printed in Task 1 Step 4.** Review
   found the printed `reap()` fails to decrement `total` on `FileNotFoundError`, which
   over-evicts under overlapping cron runs. Shipped code decrements `total` on
   `FileNotFoundError` (space genuinely freed) without crediting `removed`/`reclaimed`,
   and leaves the general `OSError` path undecremented (file still present). The shipped
   behaviour is correct; the code block in Step 4 is not. Commits `8eec6da`, `cc01e9f`.

## Deviation from the spec

The spec names the reaper `dynamo/kv_reaper.sh` (bash). This plan implements `dynamo/kv_reaper.py` (stdlib Python) instead. Reason: the reaper recursively deletes files on a 28 TB volume and its safety guards (never escape the root, never follow symlinks, refuse shallow paths) are the part most worth testing — and this host has neither `shellcheck` nor `bats`, while pytest 9.0.3 is present. Python also matches `bench_stream.py`, the repo's existing CLI style.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `dynamo/kv_reaper.py` | create | Enforce a byte budget on the L3 cache dir; oldest-first eviction. Nothing else. |
| `tests/conftest.py` | create | Load the two `dynamo/*.py` scripts as importable modules (they are not a package). |
| `tests/test_kv_reaper.py` | create | Reaper behavior + safety guards. |
| `dynamo/bench_stream.py` | modify | Add shared-prefix mode, multi-pass runs, and prompt/cached-token reporting. |
| `tests/test_bench_stream.py` | create | Prefix determinism and request-body construction (no server needed). |
| `dynamo/docker-compose.yml` | modify | Add Compose profiles; `worker` gains HiCache args + `/scratch` mount; new `worker-longctx`. |
| `dynamo/serve.sh` | modify | `PROFILE` selection + fail-fast guard on the `/scratch` dir. |
| `dynamo/stop.sh` | modify | Tear down both profiles. |
| `dynamo/RESULTS-kv-tiering.md` | create | Accumulated benchmark evidence; the input to the doc corrections. |
| `README.md`, `dynamo/README.md`, `CONTEXT_WINDOW.md` | modify (Task 10) | Correct the "1M needs ≥2 nodes" claim; document the profiles. |

Tasks 1–7 deliver complete, working software (Profile A shipped and verified). Tasks 8–10 add the second profile and correct the docs. Stopping after Task 7 leaves the repo in a good state.

---

### Task 1: The `/scratch` reaper

**Files:**
- Create: `dynamo/kv_reaper.py`
- Create: `tests/conftest.py`
- Test: `tests/test_kv_reaper.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_size(text: str) -> int` — decimal-suffixed size strings to bytes.
  - `validate_root(root: pathlib.Path) -> pathlib.Path` — resolved dir, or `SystemExit`.
  - `collect_files(root) -> list[tuple[float, int, pathlib.Path]]` — `(mtime, size, path)`, symlinks excluded.
  - `reap(root, max_bytes: int, dry_run: bool = False) -> tuple[int, int]` — `(files_removed, bytes_reclaimed)`.
  - Task 7 invokes this script from cron.

- [ ] **Step 1: Write the module loader that both test files need**

`dynamo/*.py` are standalone scripts, not a package, so tests load them by path.

Create `tests/conftest.py`:

```python
"""Load dynamo/ scripts as importable modules.

They are standalone CLIs, not a package, so there is nothing to `import`.
Session-scoped: loading is cheap but neither module has import side effects
worth repeating.
"""
import importlib.util
import pathlib
import sys

import pytest

DYNAMO = pathlib.Path(__file__).resolve().parents[1] / "dynamo"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DYNAMO / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def kv_reaper():
    return _load("kv_reaper")


@pytest.fixture(scope="session")
def bench_stream():
    return _load("bench_stream")
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_kv_reaper.py`:

```python
import os
import pathlib

import pytest


def _write(path, size, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    os.utime(path, (mtime, mtime))


def test_under_budget_deletes_nothing(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "a.bin", 100, 1000)
    assert kv_reaper.reap(root, max_bytes=1000) == (0, 0)
    assert (root / "a.bin").exists()


def test_deletes_oldest_first_until_under_budget(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "old.bin", 100, 1000)
    _write(root / "mid.bin", 100, 2000)
    _write(root / "new.bin", 100, 3000)
    assert kv_reaper.reap(root, max_bytes=150) == (2, 200)
    assert not (root / "old.bin").exists()
    assert not (root / "mid.bin").exists()
    assert (root / "new.bin").exists()


def test_stops_as_soon_as_it_is_under_budget(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    for i in range(5):
        _write(root / f"f{i}.bin", 100, 1000 + i)
    # 500 bytes total, budget 300 -> must drop exactly 2, not more.
    assert kv_reaper.reap(root, max_bytes=300) == (2, 200)


def test_recurses_into_subdirectories(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "sub" / "deep" / "old.bin", 100, 1000)
    _write(root / "new.bin", 100, 2000)
    assert kv_reaper.reap(root, max_bytes=100) == (1, 100)
    assert not (root / "sub" / "deep" / "old.bin").exists()


def test_dry_run_reports_but_deletes_nothing(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "old.bin", 100, 1000)
    _write(root / "new.bin", 100, 2000)
    assert kv_reaper.reap(root, max_bytes=100, dry_run=True) == (1, 100)
    assert (root / "old.bin").exists()


def test_symlinks_are_skipped_and_their_targets_survive(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    outside = tmp_path / "precious.bin"
    _write(outside, 500, 1)
    _write(root / "real.bin", 100, 3000)
    (root / "link.bin").symlink_to(outside)
    removed, reclaimed = kv_reaper.reap(root, max_bytes=50)
    assert outside.exists(), "reaper must never delete through a symlink"
    assert (removed, reclaimed) == (1, 100), "symlink must not count toward the budget"


def test_validate_root_refuses_the_filesystem_root(kv_reaper):
    with pytest.raises(SystemExit):
        kv_reaper.validate_root(pathlib.Path("/"))


def test_validate_root_refuses_a_shallow_path(kv_reaper):
    with pytest.raises(SystemExit):
        kv_reaper.validate_root(pathlib.Path("/scratch"))


def test_validate_root_refuses_a_non_directory(kv_reaper, tmp_path):
    f = tmp_path / "a" / "file.txt"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    with pytest.raises(SystemExit):
        kv_reaper.validate_root(f)


def test_validate_root_accepts_a_deep_directory(kv_reaper, tmp_path):
    root = tmp_path / "scratch" / "kvcache"
    root.mkdir(parents=True)
    assert kv_reaper.validate_root(root) == root.resolve()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("10e12", 10**13),
        ("10T", 10**13),
        ("10TB", 10**13),
        ("500GB", 5 * 10**11),
        ("1M", 10**6),
        ("2K", 2000),
    ],
)
def test_parse_size(kv_reaper, text, expected):
    assert kv_reaper.parse_size(text) == expected
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_kv_reaper.py -v`
Expected: collection error — `FileNotFoundError` / `spec_from_file_location` returning `None` for the missing `dynamo/kv_reaper.py`.

- [ ] **Step 4: Write the implementation**

Create `dynamo/kv_reaper.py` (mode 755):

```python
#!/usr/bin/env python3
"""Enforce a byte budget on the SGLang HiCache file-backend cache directory.

The `file` storage backend (--hicache-storage-backend=file) writes one .bin per
page component and has NO eviction and NO size cap, so left alone it grows until
/scratch fills. This trims it back to a budget, oldest-first.

mtime, not atime: /scratch is mounted `relatime`, so atime only advances if it is
already older than mtime or older than 24h -- too coarse to drive an LRU. mtime
approximates insertion order. Evicting a still-hot page is safe by construction:
every file here is a cache entry, and a miss falls back to L2 or recompute.

Deleting under a live worker is likewise safe for the same reason.

Usage:
    ./kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB
    ./kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB --dry-run
"""
import argparse
import os
import pathlib
import sys

# Decimal suffixes, matching how NVMe capacity and --hicache-size are quoted
# (SGLang's HostKVCache also uses host_size * 1e9, not 2**30).
_SUFFIXES = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}

# Refuse to walk anything shallower than this many path components. "/" is 1,
# "/scratch" is 2; the intended target /scratch/kvcache/glm52 is 4. This is the
# guard that stops a mistyped --root from eating a mount point.
_MIN_ROOT_DEPTH = 3


def parse_size(text):
    """Parse '1024', '10e12', '500GB', '10T' into bytes (decimal units)."""
    s = text.strip().upper().rstrip("B")
    if s and s[-1] in _SUFFIXES:
        return int(float(s[:-1]) * _SUFFIXES[s[-1]])
    return int(float(s))


def validate_root(root):
    """Resolve root and refuse anything that is not a deep-enough directory."""
    resolved = pathlib.Path(root).resolve()
    if len(resolved.parts) < _MIN_ROOT_DEPTH:
        raise SystemExit(
            f"kv_reaper: refusing to operate on {resolved}: too close to the "
            f"filesystem root (need at least {_MIN_ROOT_DEPTH} path components)"
        )
    if not resolved.is_dir():
        raise SystemExit(f"kv_reaper: not a directory: {resolved}")
    return resolved


def collect_files(root):
    """Return [(mtime, size, path)] for regular files under root.

    Symlinks are skipped entirely -- not followed, not counted, not deleted --
    so a link planted in the cache dir can never redirect a delete outside it.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = pathlib.Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                st = path.stat()
            except OSError:
                continue  # vanished under us; the worker owns this tree too
            found.append((st.st_mtime, st.st_size, path))
    return found


def reap(root, max_bytes, dry_run=False):
    """Delete oldest-first until the tree fits in max_bytes.

    Returns (files_removed, bytes_reclaimed).
    """
    entries = collect_files(root)
    total = sum(size for _mtime, size, _path in entries)
    if total <= max_bytes:
        return 0, 0

    entries.sort(key=lambda e: e[0])  # oldest mtime first
    removed = 0
    reclaimed = 0
    for _mtime, size, path in entries:
        if total <= max_bytes:
            break
        if not dry_run:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as e:
                sys.stderr.write(f"kv_reaper: cannot remove {path}: {e}\n")
                continue
        total -= size
        removed += 1
        reclaimed += size
    return removed, reclaimed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="KV cache directory to trim (the HiCache file-backend storage dir)")
    ap.add_argument("--max-bytes", required=True, type=parse_size,
                    help="Byte budget, e.g. 10TB, 500GB, 10e12")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted without deleting it")
    args = ap.parse_args()

    root = validate_root(args.root)
    removed, reclaimed = reap(root, args.max_bytes, args.dry_run)
    prefix = "would remove" if args.dry_run else "removed"
    print(f"kv_reaper: {root} {prefix} {removed} files, "
          f"{reclaimed / 1e9:.1f} GB (budget {args.max_bytes / 1e12:.1f} TB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_kv_reaper.py -v`
Expected: PASS, 17 tests (10 behavioral + 7 parametrized `parse_size` cases).

- [ ] **Step 6: Verify the CLI works end to end**

```bash
chmod +x dynamo/kv_reaper.py
mkdir -p /tmp/reaper-demo/cache/glm52 && head -c 1000000 /dev/zero > /tmp/reaper-demo/cache/glm52/a.bin
dynamo/kv_reaper.py --root /tmp/reaper-demo/cache/glm52 --max-bytes 1M --dry-run
dynamo/kv_reaper.py --root /scratch --max-bytes 10TB   # must refuse: too shallow
rm -rf /tmp/reaper-demo
```

Expected: the dry run prints a `kv_reaper: ... would remove` line; the `/scratch` invocation exits non-zero with the "too close to the filesystem root" message.

- [ ] **Step 7: Commit**

```bash
git add dynamo/kv_reaper.py tests/conftest.py tests/test_kv_reaper.py
git commit -m "feat(kv): add byte-budget reaper for the /scratch KV cache tier

The HiCache file storage backend has no eviction and no size cap, so the
L3 tier grows until /scratch fills. Trims oldest-first by mtime (atime is
unusable under relatime). Guards: refuses paths shallower than 3
components, never follows or counts symlinks."
```

---

### Task 2: Shared-prefix mode for `bench_stream.py`

Without this, Profile A's entire benefit is unmeasurable — the current harness sends one fixed prompt and reports no cache signal at all.

**Files:**
- Modify: `dynamo/bench_stream.py` (rewrite `one_request` signature at :37, `main` at :96)
- Test: `tests/test_bench_stream.py`

**Interfaces:**
- Consumes: `tests/conftest.py::bench_stream` fixture from Task 1.
- Produces:
  - `make_shared_prefix(approx_tokens: int, seed: int) -> str`
  - `build_prompt(shared_prefix: str, pass_idx: int, req_idx: int) -> str`
  - `build_body(prompt: str, max_tokens: int, no_think: bool) -> dict`
  - `one_request(prompt, max_tokens, no_think) -> dict` — gains `prompt_tokens`, `cached_tokens` keys
  - CLI flags `--shared-prefix-tokens`, `--prefix-seed`, `--passes`, used by Tasks 3, 5, 6, 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench_stream.py`:

```python
def test_shared_prefix_is_deterministic_for_a_seed(bench_stream):
    a = bench_stream.make_shared_prefix(64, seed=1234)
    b = bench_stream.make_shared_prefix(64, seed=1234)
    assert a == b
    assert a != ""


def test_shared_prefix_changes_with_the_seed(bench_stream):
    assert bench_stream.make_shared_prefix(64, 1) != bench_stream.make_shared_prefix(64, 2)


def test_shared_prefix_grows_with_the_token_target(bench_stream):
    short = bench_stream.make_shared_prefix(16, 1234)
    long = bench_stream.make_shared_prefix(256, 1234)
    assert len(long) > len(short)


def test_zero_tokens_yields_no_prefix(bench_stream):
    assert bench_stream.make_shared_prefix(0, 1234) == ""


def test_prompts_share_the_prefix_but_differ_in_suffix(bench_stream):
    prefix = bench_stream.make_shared_prefix(64, 1234)
    p0 = bench_stream.build_prompt(prefix, pass_idx=1, req_idx=0)
    p1 = bench_stream.build_prompt(prefix, pass_idx=1, req_idx=1)
    assert p0.startswith(prefix)
    assert p1.startswith(prefix)
    assert p0 != p1


def test_a_later_pass_reuses_the_prefix_with_a_new_suffix(bench_stream):
    prefix = bench_stream.make_shared_prefix(64, 1234)
    first = bench_stream.build_prompt(prefix, 1, 0)
    second = bench_stream.build_prompt(prefix, 2, 0)
    assert second.startswith(prefix)
    assert first != second


def test_no_shared_prefix_falls_back_to_the_fixed_prompt(bench_stream):
    assert bench_stream.build_prompt("", 1, 0) == bench_stream.PROMPT
    assert bench_stream.build_prompt("", 3, 7) == bench_stream.PROMPT


def test_build_body_requests_streaming_with_usage(bench_stream):
    body = bench_stream.build_body("hello", max_tokens=8, no_think=False)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 8
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_no_think_prepends_a_system_message(bench_stream):
    body = bench_stream.build_body("hello", 8, no_think=True)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1] == {"role": "user", "content": "hello"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_bench_stream.py -v`
Expected: FAIL — `AttributeError: module 'bench_stream' has no attribute 'make_shared_prefix'`.

- [ ] **Step 3: Add the prefix and body helpers**

In `dynamo/bench_stream.py`, add `import random` to the import block (after `import os`), then insert this immediately after the `PROMPT` definition (currently ending at line 34):

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_bench_stream.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Rewire `one_request` to take a prompt and report token usage**

Replace the body of `one_request` (lines 37–93) with:

```python
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
    with urllib.request.urlopen(req) as resp:
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
                # OpenAI-shaped cache accounting. If the Dynamo frontend
                # populates it, this is a DIRECT read of prefix-cache hits
                # rather than a TTFT inference -- worth far more than the
                # timing numbers for verifying the tiering actually works.
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
```

- [ ] **Step 6: Replace `main` with a multi-pass driver**

Replace `main` (lines 96–157) with `run_pass`, `summarize`, and a new `main`:

```python
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
```

Also update the module docstring usage block (lines 10–13) to add:

```
    ./bench_stream.py --concurrency 1 --num 1 --passes 2 \
        --shared-prefix-tokens 131072 --max-tokens 32   # prefix-cache reuse
```

- [ ] **Step 7: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 26 tests. No test contacts a server.

- [ ] **Step 8: Verify backward compatibility against the live endpoint**

The stack is currently running, so the default path must still work and still produce numbers comparable to the recorded baseline:

```bash
dynamo/bench_stream.py --concurrency 1 --num 4 --max-tokens 64 --tag smoke
```

Expected: a single `=== bench smoke pass 1/1 ... ===` block. Confirm `prompt tokens mean` now appears. If `cached tokens mean` appears too, note it — the Dynamo frontend is reporting cache accounting, which makes Task 5 far easier to judge.

- [ ] **Step 9: Commit**

```bash
git add dynamo/bench_stream.py tests/test_bench_stream.py
git commit -m "feat(bench): add shared-prefix, multi-pass, and token-usage reporting

The harness could not measure prefix-cache reuse, which is the entire
benefit of the tiering work. Adds --shared-prefix-tokens (deterministic
from --prefix-seed, so a post-restart run re-requests a byte-identical
prefix) and --passes, plus prompt_tokens/cached_tokens reporting.
Defaults are unchanged, so existing baselines stay comparable."
```

---

### Task 3: Record the pre-change baseline

Must happen before any compose change. The spec's success criteria compare against these numbers, and the currently-recorded figures (1975.0 system tok/s at conc 32; 150.0 decode tok/s at conc 1) come from a different date and possibly a different image.

**Files:**
- Create: `dynamo/RESULTS-kv-tiering.md`

**Interfaces:**
- Consumes: `bench_stream.py` flags from Task 2.
- Produces: the `## Baseline` section that Tasks 5, 6, and 9 compare against.

- [ ] **Step 1: Confirm the stack is up and serving**

```bash
docker compose -f dynamo/docker-compose.yml ps
curl -s http://localhost:8000/v1/models | head -c 400
```

Expected: `worker`, `frontend`, `etcd`, `nats` all Up; `/v1/models` lists `glm-5.2-fp8`. If not, `cd dynamo && ./serve.sh` and wait for the worker to register before continuing.

- [ ] **Step 2: Record the worker's effective startup configuration**

```bash
docker compose -f dynamo/docker-compose.yml logs worker 2>&1 | grep -iE "max_total_num_tokens|attention backend|hicache|hierarchical" | head -20
```

Expected: `max_total_num_tokens=540800`, DSA/`flashmla_kv` backend selected, and **no** hicache lines. Save the output — it is the "before" side of Task 5's assertions.

- [ ] **Step 3: Run the three baseline benchmarks**

```bash
cd dynamo
./bench_stream.py --concurrency 1  --num 16  --max-tokens 256 --tag baseline-latency
./bench_stream.py --concurrency 32 --num 128 --max-tokens 256 --tag baseline-throughput
./bench_stream.py --concurrency 1 --num 1 --passes 2 \
    --shared-prefix-tokens 131072 --max-tokens 32 --tag baseline-prefix
```

The third run establishes what prefix reuse looks like with the **GPU radix cache only** — pass 2 will already be faster than pass 1 today. Profile A must beat this, not merely beat a cold prefill; recording it prevents claiming credit for the radix cache that already exists.

- [ ] **Step 4: Write the results file**

Create `dynamo/RESULTS-kv-tiering.md` with a `# KV cache tiering: measured results` heading, a note of the date, image tag (`glm52-dynamo-sglang:0.5.13post1`), and a `## Baseline (no hicache)` section containing the Step 2 log lines and all three benchmark outputs verbatim in fenced blocks.

- [ ] **Step 5: Commit**

```bash
git add dynamo/RESULTS-kv-tiering.md
git commit -m "bench: record pre-change baseline for KV cache tiering

Includes a radix-cache-only shared-prefix run, so Profile A has to beat
the prefix reuse that already exists rather than just a cold prefill."
```

---

### Task 3b: Baseline eviction-pressure control — ADDED MID-EXECUTION

**Status: COMPLETE (commit `592d743`).** Recorded here so the plan matches what was actually done.

**Why it was added.** Task 3's baseline showed the GPU radix cache alone already delivers 25.9× warm-prefix TTFT (16,677 ms → 644 ms) at 99.9% cached tokens. Task 5's original criterion — "pass-2 TTFT at least 5× better than pass 1" — was therefore *already satisfied before any change*, and would have passed trivially while proving nothing. Profile A's actual value is capacity and persistence, not warm TTFT, so the test had to become one the tiers can genuinely fail. Human ruling approved the change.

**Why it had to run before Task 4/5.** Task 5 restarts the worker. Once that happens the pre-change comparison is unrecoverable, so the baseline half of the eviction test had to be captured while the stack was still in its original configuration.

**Method.** Prime a 131,072-token prefix (seed 1234), confirm it is warm, then flood with 5 distinct 131K prefixes (seeds 2–6 = 655,360 tokens) to push it out of the 540,800-token GPU pool, then re-request it. Five seeds rather than four: four would force only ~115K of eviction against a 131K prefix, leaving a partial hit and an ambiguous result.

**Measured baseline result:**

| Step | TTFT | Cached tokens |
|---|---|---|
| Warm, pre-flood | 665 ms | 100.0% |
| Recheck, post-flood | **16,769 ms** | **0%** |

Prefix A fully evicted. This is the number Task 5 Step 4b must beat.

One benign anomaly, recorded in `dynamo/RESULTS-kv-tiering.md`: the prime step returned warm (723 ms) because seed 1234 was still cached from Task 3's own prefix run — no restart had intervened. The warm-confirmation step independently establishes the pre-flood state, and all five flood requests were genuinely cold (16,641–16,841 ms), so the measurement stands.

---

### Task 4: Profile A configuration

**Files:**
- Modify: `dynamo/docker-compose.yml` (the `worker` service, currently lines 76–133)
- Modify: `dynamo/serve.sh` (lines 1–30)
- Modify: `dynamo/stop.sh` (line 6)

**Interfaces:**
- Consumes: `dynamo/kv_reaper.py` (Task 7 schedules it against the dir configured here).
- Produces: `PROFILE` env contract (`cache` | `longctx`); `KV_SCRATCH_ROOT` (default `/scratch/kvcache`) and `KV_SCRATCH_DIR` (default `/scratch/kvcache/glm52`); `HICACHE_GB` (default `96`). Task 8 adds the `longctx` service alongside.

- [ ] **Step 1: Human prerequisite — create the `/scratch` directory**

`/scratch` is root-owned and `sudo` needs a password here, so this is run by the user, not the agent:

```bash
sudo mkdir -p /scratch/kvcache/glm52
sudo chown -R $(id -u):$(id -g) /scratch/kvcache
```

Verify: `test -w /scratch/kvcache/glm52 && echo writable`

- [ ] **Step 2: Add Compose profiles and the HiCache args to the `worker` service**

In `dynamo/docker-compose.yml`, add to the `worker` service, directly under `worker:`:

```yaml
    # Profile A: tiered KV cache (GPU radix -> host RAM L2 -> /scratch L3).
    # Mutually exclusive with worker-longctx: SGLang forbids hierarchical cache
    # together with --disable-radix-cache, which HiSparse requires.
    profiles: ["cache"]
```

Extend its `environment:` block with:

```yaml
      # HiCacheFile defaults to /tmp/hicache -- the 446 GB ROOT filesystem --
      # if this is unset, which would fill the boot disk. Setting it is not
      # optional. serve.sh fails fast if the directory is missing.
      SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR: "${KV_SCRATCH_DIR:-/scratch/kvcache/glm52}"
```

Extend its `volumes:` block with:

```yaml
      # L3 KV tier. Separate NVMe devices from /data (the weight cache), so
      # cache I/O does not contend with weight loading.
      - "${KV_SCRATCH_ROOT:-/scratch/kvcache}:${KV_SCRATCH_ROOT:-/scratch/kvcache}"
```

Append to its `command:` list, after the existing speculative-decoding args:

```yaml
      # --- Tiered KV cache -------------------------------------------------
      # See docs/superpowers/specs/2026-08-31-scratch-kv-cache-tiering-design.md
      #
      # --hicache-size is PER TP RANK: HostKVCache.__init__ reads it as this
      # rank's own pool (host_size * 1e9 // size_per_token), so TP=8 allocates
      # it EIGHT times -- MLA makes all eight copies identical, but they are
      # still eight allocations. 96 GB/rank => ~770 GB of MLA host pool plus
      # ~22 GB/rank of DSA indexer host pool = ~944 GB of the ~2168 GB free.
      # SGLang hard-fails at startup if it over-requests, so raising this is
      # fail-safe, not fail-dangerous.
      - --enable-hierarchical-cache
      - --hicache-size=${HICACHE_GB:-96}
      - --hicache-write-policy=write_through
      # `kernel` I/O survives here: it is only downgraded to `direct` when the
      # effective DECODE backend is fa3, and ours is the auto-selected
      # dsa/flashmla_kv. Verify from the startup log rather than trusting this.
      - --hicache-io-backend=kernel
      - --hicache-mem-layout=page_first
      # The `file` backend deduplicates across TP ranks for MLA models (storage
      # keys carry no tp_rank suffix), so /scratch holds ONE copy, not eight.
      # That dedup is why the disk tier holds ~85x more unique tokens than all
      # of host RAM. It has no eviction -- see dynamo/kv_reaper.py.
      - --hicache-storage-backend=file
      - --hicache-storage-prefetch-policy=timeout
```

- [ ] **Step 3: Teach `serve.sh` about profiles and guard the `/scratch` dir**

In `dynamo/serve.sh`, after the `IMAGE=` line, insert:

```bash
PROFILE="${PROFILE:-cache}"
case "${PROFILE}" in
  cache|longctx) ;;
  *) echo "Unknown PROFILE '${PROFILE}' (expected: cache, longctx)" >&2; exit 1 ;;
esac

# Profile A writes its L3 KV tier to /scratch. If the directory is missing,
# SGLang's HiCacheFile silently falls back to /tmp/hicache on the 446 GB root
# filesystem and fills the boot disk. Fail fast instead.
if [ "${PROFILE}" = "cache" ]; then
  KV_SCRATCH_ROOT="${KV_SCRATCH_ROOT:-/scratch/kvcache}"
  KV_SCRATCH_DIR="${KV_SCRATCH_DIR:-/scratch/kvcache/glm52}"
  if [ ! -d "${KV_SCRATCH_DIR}" ] || [ ! -w "${KV_SCRATCH_DIR}" ]; then
    cat >&2 <<EOF
PROFILE=cache needs a writable KV cache directory at ${KV_SCRATCH_DIR}

  sudo mkdir -p ${KV_SCRATCH_DIR}
  sudo chown -R \$(id -u):\$(id -g) ${KV_SCRATCH_ROOT}

Or set KV_SCRATCH_DIR/KV_SCRATCH_ROOT to a writable location.
EOF
    exit 1
  fi
  export KV_SCRATCH_ROOT KV_SCRATCH_DIR
fi
```

Change the startup line from `docker compose up -d` to:

```bash
echo "Starting Dynamo (SGLang) stack for GLM-5.2-FP8 [profile: ${PROFILE}] ..."
docker compose --profile "${PROFILE}" up -d
```

(Delete the now-duplicated `echo "Starting Dynamo (SGLang) stack for GLM-5.2-FP8 ..."` line above it.)

- [ ] **Step 4: Teach `stop.sh` to tear down either profile**

In `dynamo/stop.sh`, replace `docker compose down "$@"` with:

```bash
# Both profiles named explicitly so `down` reaches whichever worker is running.
docker compose --profile cache --profile longctx down "$@"
```

- [ ] **Step 5: Validate the compose file without starting anything**

```bash
cd dynamo
docker compose --profile cache config >/dev/null && echo "cache profile parses"
docker compose --profile cache config | grep -A2 'hicache-size'
docker compose config --services            # profile-gated services are hidden
docker compose --profile cache config --services
```

Expected: the file parses; `--hicache-size=96` appears with interpolation resolved; `worker` is absent from the unprofiled service list and present in the `cache` list.

- [ ] **Step 6: Verify the guard fires**

```bash
cd dynamo
PROFILE=cache KV_SCRATCH_DIR=/nonexistent ./serve.sh; echo "exit=$?"
PROFILE=bogus ./serve.sh; echo "exit=$?"
```

Expected: both exit 1 with their respective messages, and **neither starts a container** (`docker compose ps` unchanged).

- [ ] **Step 7: Commit**

```bash
git add dynamo/docker-compose.yml dynamo/serve.sh dynamo/stop.sh
git commit -m "feat(dynamo): add PROFILE=cache with tiered KV cache on /scratch

Compose profiles select the worker; serve.sh defaults to PROFILE=cache and
fails fast if the /scratch KV dir is missing, because HiCacheFile would
otherwise silently fall back to /tmp/hicache on the root filesystem."
```

---

### Task 5: Bring up Profile A and verify it

**Files:**
- Modify: `dynamo/RESULTS-kv-tiering.md`

**Interfaces:**
- Consumes: Task 3's baseline, Task 4's config.
- Produces: the `## Profile A` results section; the go/no-go on the spec's first two success criteria.

- [ ] **Step 1: Restart the stack under Profile A**

```bash
cd dynamo
./stop.sh
PROFILE=cache ./serve.sh
docker compose logs -f worker
```

The JIT cache volume is untouched, so this should come up in minutes, not the ~10–20 min first-boot compile. Wait for the worker to register.

- [ ] **Step 2: Assert the effective configuration from the log**

```bash
docker compose logs worker 2>&1 | grep -iE "hicache|hierarchical|host memory|indexer|max_total_num_tokens|attention backend" | head -30
```

Assert all of these, and stop if any fails:
- An "Allocating N GB host memory for hierarchical KV cache" line at roughly **96 GB** per rank.
- A separate DSA indexer host allocation of roughly **22 GB** per rank.
- The mem layout is still **`page_first`** and the io backend still **`kernel`** — if either was rewritten, the spec's reasoning about the fa3 guard was wrong and the numbers below need re-interpreting.
- `max_total_num_tokens` is still **540800** — the device pool must not have shrunk.
- The attention backend is still the auto-selected DSA/`flashmla_kv`.

- [ ] **Step 3: Confirm host memory landed where expected**

```bash
free -g          # expect ~944 GB more used than the baseline's ~98 GB
ls -la /scratch/kvcache/glm52 | head
```

Expected: used memory around 1.0–1.1 TB with >1 TB still available; the `/scratch` dir exists and is being populated (it may be empty until the first requests run).

- [ ] **Step 4: Run the same three benchmarks**

```bash
./bench_stream.py --concurrency 1  --num 16  --max-tokens 256 --tag hicache-latency
./bench_stream.py --concurrency 32 --num 128 --max-tokens 256 --tag hicache-throughput
./bench_stream.py --concurrency 1 --num 1 --passes 2 \
    --shared-prefix-tokens 131072 --max-tokens 32 --tag hicache-prefix
```

- [ ] **Step 4b: Re-run the eviction-pressure sequence — the decisive test**

Identical to Task 3b, so the two are directly comparable:

```bash
cd /home/users/wrightda/src/GLM-5.2-FP8/dynamo
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-hicache-prime
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-hicache-warm
for s in 2 3 4 5 6; do
  ./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed $s \
      --num 1 --concurrency 1 --max-tokens 32 --tag evict-hicache-flood-seed$s
done
./bench_stream.py --shared-prefix-tokens 131072 --prefix-seed 1234 \
    --num 1 --concurrency 1 --max-tokens 32 --tag evict-hicache-recheck
```

Note the worker was restarted in Step 1, so unlike Task 3b the prime here starts genuinely cold.

- [ ] **Step 5: Judge against the spec's criteria**

- **No cold-path regression:** conc-32 system tok/s within 5% of **2087.1** and conc-1 decode tok/s within 5% of **150.6** — the Task 3 measured baseline. (The 1975.0 / 150.0 figures in `dynamo/README.md` are stale; do not compare against them.) Tiering must be free when it misses.
- **Eviction survival — THE decisive criterion.** Compare Step 4b's final recheck against the Task 3b baseline, where an evicted 131K prefix cost **16,769 ms at 0% cached tokens**. Profile A must do materially better: substantially lower TTFT and, above all, a **non-zero `cached tokens mean`**, which is direct proof the prefix was served from L2/L3 rather than recomputed. Lead with the cached-token count; TTFT is corroboration, since it is noisy and load-dependent.
  - If the recheck is also ~16.8 s at 0% cached, **the tiering is not doing anything** and neither L2 nor L3 has earned its place. Say so plainly rather than defending the design.
- **Warm-prefix TTFT is NOT a criterion.** The Task 3 baseline already achieves 25.9× (16,677 ms → 644 ms) and 99.9% cached tokens on `--passes 2` using the GPU radix cache alone. Any bar based on it passes trivially and proves nothing. Record the number for completeness; do not treat it as evidence for or against Profile A.
- Check `/scratch` actually grew: `du -sh /scratch/kvcache/glm52`. If it is still empty, the L3 tier is not being written and Task 6 cannot pass — investigate before proceeding.
- **Watch for write amplification on the deduped key.** All 8 TP ranks address one storage key per page (no `tp_rank` suffix for MLA), so they may all write it. During the conc-32 run, sample `iostat -x 5 3` or compare `du -sh` before and after: if `/scratch` grows at roughly 8× the unique-token rate, the ranks are not coordinating and the write path needs `write_through_selective`. Record either way.

If the cold path regressed, try `--hicache-write-policy=write_through_selective` (writes fewer, hotter pages) and re-run before concluding.

- [ ] **Step 6: Record and commit**

Append a `## Profile A (hicache, /scratch L3)` section to `dynamo/RESULTS-kv-tiering.md` with the Step 2 log assertions, `du -sh` output, all three benchmark outputs, and an explicit PASS/FAIL against each criterion.

```bash
git add dynamo/RESULTS-kv-tiering.md
git commit -m "bench: record Profile A results vs baseline"
```

---

### Task 6: Prove the cache survives a restart

This is the criterion that justifies `/scratch` at all. Host RAM cannot make this claim — if this fails, the L3 tier is not earning its place and the honest outcome is an L2-only design.

**Files:**
- Modify: `dynamo/RESULTS-kv-tiering.md`

**Interfaces:**
- Consumes: Task 5's warm-pass TTFT.
- Produces: the persistence verdict feeding Task 10's doc claims.

- [ ] **Step 1: Confirm the cache is populated and note its size**

```bash
du -sh /scratch/kvcache/glm52
```

- [ ] **Step 2: Restart the worker only, without clearing `/scratch`**

```bash
cd dynamo
docker compose --profile cache restart worker
docker compose logs -f worker    # wait for registration
```

Do **not** run `stop.sh --volumes`, and do not delete anything under `/scratch`.

- [ ] **Step 3: Re-request the identical prefix**

```bash
./bench_stream.py --concurrency 1 --num 1 --passes 1 \
    --shared-prefix-tokens 131072 --max-tokens 32 --prefix-seed 1234 \
    --tag hicache-post-restart
```

The default `--prefix-seed 1234` reproduces a byte-identical prefix to Task 5, which is the whole point.

- [ ] **Step 4: Judge**

Compare this TTFT to Task 5's pass-1 (cold) and pass-2 (warm) numbers:
- **Close to warm** → the L3 tier survived the restart. `/scratch` is justified.
- **Close to cold** → nothing was reloaded from disk. Before concluding failure, check that the host L2 pool was rebuilt empty (expected) and that the worker logged a storage-backend hit attempt; also confirm `/scratch/kvcache/glm52` is non-empty.
- If `cached tokens mean` is reported, use it as the direct read — it beats inferring from timing.

- [ ] **Step 5: Record and commit**

Append a `## Restart persistence` section with the command, the three TTFT numbers side by side, and the verdict.

```bash
git add dynamo/RESULTS-kv-tiering.md
git commit -m "bench: record L3 cache persistence across a worker restart"
```

---

### Task 7: Schedule the reaper

**Files:**
- Modify: `dynamo/README.md` (add a maintenance subsection)

**Interfaces:**
- Consumes: `dynamo/kv_reaper.py` from Task 1; `KV_SCRATCH_DIR` from Task 4.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Dry-run the reaper against the real, now-populated cache**

```bash
dynamo/kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB --dry-run
dynamo/kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 1M --dry-run
```

Expected: the first reports 0 files (well under 10 TB); the second reports a non-zero count, proving it finds and orders the real files. Neither deletes anything.

- [ ] **Step 2: Install a user crontab entry**

No sudo required — this runs as the invoking user, who owns `/scratch/kvcache`. Add via `crontab -e`:

```cron
*/15 * * * * /home/users/wrightda/src/GLM-5.2-FP8/dynamo/kv_reaper.py --root /scratch/kvcache/glm52 --max-bytes 10TB >> /scratch/kvcache/reaper.log 2>&1
```

The log lives in `/scratch/kvcache/`, one level **above** the reaped root, so the reaper never counts or deletes its own log.

- [ ] **Step 3: Verify it is installed and fires**

```bash
crontab -l | grep kv_reaper
sleep 900 && tail -5 /scratch/kvcache/reaper.log
```

Expected: the entry is listed, and within 15 minutes a `kv_reaper: ... removed 0 files` line appears.

- [ ] **Step 4: Document it**

Add a `### Maintenance: the KV cache reaper` subsection to `dynamo/README.md` covering: why it exists (the `file` backend has no eviction or size cap), the crontab line verbatim, the 10 TB-of-28 TB budget, that mtime rather than atime drives eviction and why (`relatime`), and that deleting under a live worker is safe because every file is a regenerable cache entry.

- [ ] **Step 5: Commit**

```bash
git add dynamo/README.md
git commit -m "docs(dynamo): document the KV cache reaper and its cron schedule"
```

---

### Task 8: Profile B configuration

Profile A is complete and shipped at this point. Everything below adds the second profile.

**Files:**
- Modify: `dynamo/docker-compose.yml` (add a `worker-longctx` service after `worker`)

**Interfaces:**
- Consumes: the `PROFILE` contract from Task 4.
- Produces: `PROFILE=longctx`; env knobs `HISPARSE_DEVICE_BUFFER` (default `4096`), `HISPARSE_RATIO` (default `2`), `MAX_MODEL_LEN_LONG` (default `1048576`), `MEM_FRACTION_LONG` (default `0.88`).

- [ ] **Step 1: Add the `worker-longctx` service**

Append to `dynamo/docker-compose.yml`, after the `worker` service and before the `volumes:` block:

```yaml
  # Profile B: HiSparse long-context worker. GPU holds the DSA indexer at the
  # FULL logical size (index_buf_size = size * host_to_device_ratio) while only
  # a working set of the heavy MLA latent KV stays resident, the rest paging
  # from host RAM. That is what raises the servable context past 512K.
  #
  # THE COST IS PREFIX CACHING, ENTIRELY. HiSparse asserts --disable-radix-cache,
  # so every request re-prefills from zero. This is the wrong default for
  # multi-turn, shared-prompt, and batch-corpus workloads alike -- hence a
  # separate profile rather than a flag.
  #
  # /scratch is unused here: HiSparse has no storage backend, only GPU<->host.
  worker-longctx:
    profiles: ["longctx"]
    image: ${DYNAMO_IMAGE:-glm52-dynamo-sglang:0.5.13post1}
    network_mode: host
    gpus: all
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    environment:
      <<: *dynamo-env
      HF_HOME: "${HF_CACHE:-/root/.cache/huggingface}"
      HF_HUB_OFFLINE: "1"
    volumes:
      - "${HF_CACHE:-/root/.cache/huggingface}:${HF_CACHE:-/root/.cache/huggingface}"
      - "dynamo-jit-cache:/home/dynamo/.cache"
    depends_on: [etcd, nats]
    entrypoint: ["python3", "-m", "dynamo.sglang"]
    command:
      - --model-path=${MODEL:-zai-org/GLM-5.2-FP8}
      - --served-model-name=${SERVED_NAME:-glm-5.2-fp8}
      - --tp-size=${TP_SIZE:-8}
      - --kv-cache-dtype=fp8_e4m3
      - --max-running-requests=${MAX_RUNNING:-128}
      - --page-size=${PAGE_SIZE:-64}
      - --trust-remote-code
      - --dyn-tool-call-parser=glm47
      - --dyn-reasoning-parser=glm45
      # HiSparse selects flashmla_kv itself for fp8_e4m3 KV -- the same backend
      # the aggregated worker auto-selects. Do NOT add --attention-backend.
      - --enable-hisparse
      - --disable-radix-cache
      - '--hisparse-config={"top_k":2048,"device_buffer_size":${HISPARSE_DEVICE_BUFFER:-4096},"host_to_device_ratio":${HISPARSE_RATIO:-2}}'
      # ~911K tokens fit at mem-fraction 0.85; the full 1,048,576 needs
      # ~34.4 GB/GPU, i.e. ~0.88. The OOM-at-graph-capture risk starts near 0.93.
      - --context-length=${MAX_MODEL_LEN_LONG:-1048576}
      - --mem-fraction-static=${MEM_FRACTION_LONG:-0.88}
      # MTP/EAGLE is carried over UNVERIFIED: nothing in SGLang forbids it with
      # HiSparse and nothing exercises it either. If startup fails citing
      # speculative decoding, delete these four lines and re-run -- Profile B
      # then loses the ~2x single-stream decode MTP provides, which must be
      # recorded in RESULTS-kv-tiering.md.
      - --speculative-algorithm=${SPEC_ALGO:-EAGLE}
      - --speculative-num-steps=${SPEC_NUM_STEPS:-2}
      - --speculative-eagle-topk=${SPEC_EAGLE_TOPK:-1}
      - --speculative-num-draft-tokens=${SPEC_NUM_DRAFT:-3}
    restart: "no"
```

- [ ] **Step 2: Validate both profiles still parse**

```bash
cd dynamo
docker compose --profile longctx config >/dev/null && echo "longctx parses"
docker compose --profile longctx config | grep -E 'hisparse|context-length|mem-fraction'
docker compose --profile cache config --services
docker compose --profile longctx config --services
```

Expected: `--hisparse-config={"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}` with interpolation resolved and the JSON intact (the single quotes keep YAML from parsing the braces as a flow mapping); `context-length=1048576`; `mem-fraction-static=0.88`. The `cache` list contains `worker` and not `worker-longctx`; the `longctx` list, the reverse.

- [ ] **Step 3: Commit**

```bash
git add dynamo/docker-compose.yml
git commit -m "feat(dynamo): add PROFILE=longctx HiSparse worker

Targets the model's full 1M context on one node at mem-fraction 0.88, at
the cost of prefix caching entirely (HiSparse requires --disable-radix-cache).
Off by default. MTP is carried over unverified -- see the inline note."
```

---

### Task 9: Verify Profile B

**Files:**
- Modify: `dynamo/RESULTS-kv-tiering.md`

**Interfaces:**
- Consumes: Task 8's config, Task 3's baseline.
- Produces: the evidence Task 10 needs before rewriting the 1M claim.

- [ ] **Step 1: Switch profiles**

```bash
cd dynamo
./stop.sh
PROFILE=longctx ./serve.sh
docker compose logs -f worker-longctx
```

If startup fails citing speculative decoding, remove the four `--speculative-*` lines from `worker-longctx` and retry, then record that MTP is incompatible.

- [ ] **Step 2: Record the achieved pool size**

```bash
docker compose logs worker-longctx 2>&1 | grep -iE "max_total_num_tokens|hisparse|host|indexer|context" | head -30
```

The number to capture is `max_total_num_tokens`. The spec projects ~1,048,576 at mem-fraction 0.88. If it lands materially short, lower `--context-length` to the achieved figure rather than leaving a length the server cannot actually serve — that is exactly the trap the current 512K setting exists to avoid.

- [ ] **Step 3: Prove a >512K request works**

This is the criterion that falsifies the "≥2 nodes" claim. `--shared-prefix-tokens` is the easiest way to build a genuinely long prompt:

```bash
./bench_stream.py --concurrency 1 --num 1 --passes 1 \
    --shared-prefix-tokens 600000 --max-tokens 64 --tag longctx-600k
```

Expected: the request completes and returns coherent output, with `prompt tokens mean` above 524,288. Confirm from the reported figure, not from the requested one — `--shared-prefix-tokens` is a target, not a guarantee.

The filler runs about 7 bytes per token, so 600K tokens is a **~4 MB JSON request body**. If this fails with a 413, a connection reset, or a frontend parse error rather than a model error, that is an HTTP body-size limit in the Dynamo frontend, not a context-length failure — raise the frontend's limit or step down to 550K and retry before concluding anything about Profile B.

- [ ] **Step 4: Benchmark the throughput cost**

```bash
./bench_stream.py --concurrency 1  --num 16  --max-tokens 256 --tag longctx-latency
./bench_stream.py --concurrency 32 --num 128 --max-tokens 256 --tag longctx-throughput
```

These are expected to be **worse** than Profile A. That is the trade, not a bug. Record honestly.

- [ ] **Step 5: Return to the default profile**

```bash
./stop.sh
PROFILE=cache ./serve.sh
docker compose logs -f worker    # confirm registration before finishing
```

- [ ] **Step 6: Record and commit**

Append a `## Profile B (hisparse, long context)` section: achieved `max_total_num_tokens`, whether MTP survived, the >512K request's reported `prompt_tokens`, both throughput runs beside Profile A's, and a plain statement of the trade.

```bash
git add dynamo/RESULTS-kv-tiering.md
git commit -m "bench: record Profile B long-context results and their cost"
```

---

### Task 10: Correct the documentation

Only the claims the measurements actually support. If Profile B fell short of 1M, say what it reached.

**Files:**
- Modify: `CONTEXT_WINDOW.md`
- Modify: `README.md`
- Modify: `dynamo/README.md`

**Interfaces:**
- Consumes: every result in `dynamo/RESULTS-kv-tiering.md`.

- [ ] **Step 1: Rewrite `CONTEXT_WINDOW.md`**

It currently frames 512K as a hardware ceiling requiring ≥2 nodes. Recast it as a **trade**: the default profile serves 512K *with* a tiered prefix cache; `PROFILE=longctx` serves the measured long-context figure *without* prefix caching. Keep the KV-pool arithmetic — it is correct and now better sourced. Replace "What it would take: ≥2 nodes" with the HiSparse path and its actual cost. Update the settings table with both profiles.

- [ ] **Step 2: Update `README.md`**

In the "Serve" section, document `PROFILE=cache` (default) and `PROFILE=longctx`. Correct the sentence stating the model's 1M "isn't servable on one node". Add the `/scratch` prerequisite (the `sudo mkdir`/`chown`) to the serve instructions, since a fresh clone will hit the `serve.sh` guard.

- [ ] **Step 3: Update `dynamo/README.md`**

Add a "Tiered KV cache" section under "Performance levers" with the tier table (unique tokens per tier and the TP-replication-vs-storage-dedup asymmetry that drives it), the full tunable list (`HICACHE_GB`, `KV_SCRATCH_DIR`, `KV_SCRATCH_ROOT`, `HISPARSE_DEVICE_BUFFER`, `HISPARSE_RATIO`, `MAX_MODEL_LEN_LONG`, `MEM_FRACTION_LONG`), and the measured before/after numbers. Add entries to the status checklist for Profile A, restart persistence, and Profile B. Correct the two existing statements that 1M and disaggregation both need ≥2 nodes — disaggregation still does; 1M no longer does.

- [ ] **Step 4: Verify no stale claims remain**

```bash
grep -rn "2 nodes\|two nodes\|isn't servable\|not servable" README.md dynamo/README.md CONTEXT_WINDOW.md
```

Expected: every surviving hit is about **disaggregated prefill/decode**, which genuinely still needs ≥2 nodes. No surviving hit should be about context length.

- [ ] **Step 5: Commit**

```bash
git add README.md dynamo/README.md CONTEXT_WINDOW.md
git commit -m "docs: 1M context is a profile trade, not a node-count limit

Documents both serving profiles and the measured tiering results. The
>=2-node requirement remains true for disaggregated prefill/decode and is
left in place; it is no longer true for context length."
```

---

## Notes for the implementer

- **Two tasks depend on a human.** Task 4 Step 1 (`sudo mkdir`) and Task 7 Step 2 (`crontab -e`) cannot be automated here. Stop and ask.
- **Tasks 3, 5, 6, and 9 are empirical.** Their deliverable is recorded evidence, not code. Do not paraphrase benchmark output — paste it. Do not claim a criterion passed without the numbers beside it.
- **The most likely surprise** is Task 5 Step 2: if the startup log shows the mem layout or io backend rewritten, the spec's reasoning about the fa3 guard was wrong. Record what actually happened before interpreting any timing.
- **The most likely failure** is Task 6. If the cache does not survive a restart, `/scratch` has not earned its place; say so plainly and propose dropping to L2-only rather than defending the design.
- **`README.md` references `./chat.py`, which is not in the repo** (`git ls-files` has no such file). Pre-existing, unrelated to this work — do not fix it here, but mention it when reporting.
