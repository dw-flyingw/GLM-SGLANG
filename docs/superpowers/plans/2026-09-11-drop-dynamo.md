# Drop Dynamo — SGLang-Native Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-container Dynamo stack (etcd + NATS + `dynamo.frontend` + `dynamo.sglang`) with a single container running `sglang.launch_server` directly, with no change to the serving behavior, the image, or any engine flag.

**Architecture:** The worker image is already an SGLang image and every flag in its `command:` is already a raw SGLang flag, so this is a wrapper removal, not an engine change. Three services are deleted, the entrypoint changes, two Dynamo-frontend flags are renamed to their verified SGLang equivalents, and three flags are added (`--host`, `--port`, `--enable-cache-report`). The Compose project identity is pinned with a top-level `name:` so the directory rename cannot orphan containers or the JIT cache volume.

**Tech Stack:** Docker Compose, SGLang 0.5.13.post1 (unchanged image `glm52-dynamo-sglang:0.5.13post1`, retagged `glm52-sglang:0.5.13post1`), Python 3.11+ stdlib, pytest (dev only), bash.

**Spec:** `docs/superpowers/specs/2026-09-11-drop-dynamo-sglang-native-design.md` — read it before starting. It carries the verified evidence (which flags exist in the image, which volumes and containers exist on the host) behind every change here.

## Global Constraints

- **No engine tunable may change.** `--mem-fraction-static`, `--context-length`, `--tp-size 8`, `--page-size 64`, `--kv-cache-dtype fp8_e4m3`, every `--hicache-*`, every `--hisparse*`, every `--speculative-*`, and the *absence* of `--attention-backend` all stay exactly as they are. This migration must be measurable as a no-op on serving behavior; changing an engine flag in the same commit would make the two indistinguishable.
- **Python is stdlib-only.** `pyproject.toml` declares `dependencies = []`. pytest is a dev tool; do not add it, or anything else, to `pyproject.toml` dependencies.
- **`requires-python = ">=3.11"`.** The host runs 3.12.3.
- **Do not run `./serve.sh`, `./stop.sh`, `docker compose up`, or `docker compose down` against the live stack.** The Dynamo stack is serving (up 7+ days, holding all 8 GPUs). Cutover is the operator's, per the spec's runbook. `docker compose config` and `docker compose -p dynamo ps` are read-only and safe.
- **Never commit `sglang/.env`** — it is gitignored and holds the host-specific `HF_CACHE=/data/huggingface`. `git mv` moves it in the working tree; it must stay untracked.
- **Historical records are frozen.** `sglang/RESULTS-kv-tiering.md`, `docs/superpowers/plans/2026-08-31-kv-cache-tiering.md`, and `docs/superpowers/specs/2026-08-31-scratch-kv-cache-tiering-design.md` keep their Dynamo references. `RESULTS-kv-tiering.md` gets exactly one added header line and nothing else.
- Exact values that must appear verbatim in the new compose file: project name `glm52-sglang`, image tag default `glm52-sglang:0.5.13post1`, env var `SGLANG_IMAGE`, volume key `jit-cache`, `--tool-call-parser=glm47`, `--reasoning-parser=glm45`, `--enable-cache-report`, `--host=${HTTP_HOST:-127.0.0.1}`, `--port=${PORT:-8000}`.

## File Structure

| File | Responsibility after this change |
|---|---|
| `sglang/docker-compose.yml` | One worker service per profile, nothing else. The only file that knows the engine's command line. |
| `sglang/serve.sh` | Preflight (KV-scratch write probe, py-spy cap probe, log archive) + `compose up`. Unchanged in substance. |
| `sglang/stop.sh` | Log archive + `compose down`. Gains a runtime warning on `--volumes`. |
| `sglang/bench.sh`, `sglang/bench_stream.py` | Load generation. Env var rename + one comment correction. |
| `tests/conftest.py` | Module loader; one path constant. |
| `tests/test_compose_config.py` | **New.** Static regression guard on the rendered Compose config — the commit-time net for the silent `--enable-cache-report` failure. |
| `README.md`, `sglang/README.md`, `CONTEXT_WINDOW.md` | Live docs describing the SGLang-native stack. |

Task order is deliberate: the rename lands **first** so every later task edits final paths, and the compose conversion lands **before** the scripts that reference its image tag.

---

### Task 1: Rename `dynamo/` → `sglang/` and repoint the test harness

Pure rename. No behavior change, no Dynamo removal yet — that keeps this task independently reviewable and independently revertible.

**Files:**
- Rename: `dynamo/` → `sglang/` (via `git mv`, preserves history)
- Modify: `tests/conftest.py:1`, `tests/conftest.py:13`, `tests/conftest.py:17`, `tests/conftest.py:37`
- Modify: `tests/test_archive_worker_log.py:1` (docstring path)
- Test: `tests/test_bench_stream.py`, `tests/test_kv_reaper.py`, `tests/test_archive_worker_log.py` (all must pass unchanged)

**Interfaces:**
- Consumes: nothing.
- Produces: the `sglang/` directory path that every later task edits. The `conftest.py` fixtures keep their exact names — `kv_reaper`, `bench_stream`, `archive_script` — so no test file signature changes.

- [ ] **Step 1: Confirm the test suite passes before touching anything**

Run: `python3 -m pytest tests/ -q`
Expected: all tests pass. If they do not, stop and report — this plan assumes a green baseline.

- [ ] **Step 2: Do the rename**

```bash
cd "$(git rev-parse --show-toplevel)"
git mv dynamo sglang
ls sglang/.env    # must still exist (untracked, moved by the filesystem)
```

`git mv` on a directory moves tracked files; `.env` is untracked and gitignored, so verify it came along. If it did not, move it by hand: `mv dynamo/.env sglang/.env`.

- [ ] **Step 3: Run the tests to verify they now fail**

Run (from the repo root): `python3 -m pytest tests/ -q`
Expected: FAIL — `conftest.py` still points at `parents[1] / "dynamo"`, so `_load` raises on a missing file and the `archive_script` fixture's `assert path.is_file()` trips.

- [ ] **Step 4: Repoint `tests/conftest.py`**

Replace the module docstring's first line and the path constant:

```python
"""Load sglang/ scripts as importable modules.

They are standalone CLIs, not a package, so there is nothing to `import`.
Session-scoped: loading is cheap but neither module has import side effects
worth repeating.
"""
import importlib.util
import pathlib
import sys

import pytest

SGLANG = pathlib.Path(__file__).resolve().parents[1] / "sglang"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SGLANG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
```

And in the `archive_script` fixture, `DYNAMO / "archive_worker_log.sh"` becomes `SGLANG / "archive_worker_log.sh"`. The three fixture functions are otherwise untouched.

- [ ] **Step 5: Fix the docstring in `tests/test_archive_worker_log.py`**

Line 1: `"""Regression tests for dynamo/archive_worker_log.sh.` → `"""Regression tests for sglang/archive_worker_log.sh.`

- [ ] **Step 6: Run the tests to verify they pass**

Run (from the repo root): `python3 -m pytest tests/ -q`
Expected: PASS, with the same test count as Step 1.

- [ ] **Step 7: Verify no stale `dynamo/` path references remain in test or tooling code**

Run: `grep -rn 'dynamo/' tests/ pyproject.toml`
Expected: no output. (Docs and `sglang/` contents still mention Dynamo — those are Tasks 2–4.)

- [ ] **Step 8: Commit**

```bash
git add -A tests/ sglang/ pyproject.toml
git commit -m "refactor: rename dynamo/ -> sglang/

Pure rename, no behavior change. git mv preserves history. The only code
edits are the path constant in tests/conftest.py and two docstrings.

The directory has never held anything but an SGLang deployment; the name
described the wrapper being removed in the commits that follow."
```

---

### Task 2: Convert `docker-compose.yml` to the SGLang-native stack

The heart of the change, and the task whose failure mode is silent — hence a new static test that runs against the *rendered* config rather than the YAML source, so it tests what actually reaches the engine after anchors, profiles, and `${VAR:-default}` substitution are resolved.

**Files:**
- Create: `tests/test_compose_config.py`
- Modify: `sglang/docker-compose.yml`
- Test: `tests/test_compose_config.py`

**Interfaces:**
- Consumes: the `sglang/` path from Task 1.
- Produces: a compose file whose rendered config contains `--enable-cache-report`, `--tool-call-parser=glm47`, `--reasoning-parser=glm45`, `--host`, `--port`; project name `glm52-sglang`; volume key `jit-cache`; and exactly one service per profile. Task 3's `serve.sh` depends on the image tag default `glm52-sglang:0.5.13post1` and on the service names `worker` / `worker-longctx`, both unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compose_config.py`:

```python
"""Static checks on the RENDERED Compose config.

`docker compose config` resolves YAML anchors, profile filtering, and
${VAR:-default} substitution, so asserting against its output tests what
actually reaches the engine -- not what the source YAML looks like.

The cache-report test is the important one. SGLang's native server
populates usage.prompt_tokens_details.cached_tokens ONLY when
--enable-cache-report is set; the Dynamo frontend used to populate it for
free. Dropping the flag produces no error and no warning -- the field just
vanishes and bench_stream.py's cache column goes blank, which is the
measurement behind the entire KV-tiering result. This test is the
commit-time net for that.
"""
import pathlib
import shutil
import subprocess

import pytest

COMPOSE = pathlib.Path(__file__).resolve().parents[1] / "sglang" / "docker-compose.yml"

WORKER_PROFILES = [("cache", "worker"), ("longctx", "worker-longctx")]


@pytest.fixture(scope="session")
def _docker_available():
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")


def render(profile):
    """Rendered `docker compose config` for one profile, as text."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "--profile", profile, "config"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"compose config failed:\n{result.stderr}"
    return result.stdout


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_cache_report_is_enabled_in_every_worker_profile(_docker_available, profile, service):
    """Without this flag the cached_tokens measurement silently disappears."""
    assert "--enable-cache-report" in render(profile)


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_entrypoint_is_native_sglang(_docker_available, profile, service):
    rendered = render(profile)
    assert "sglang.launch_server" in rendered
    assert "dynamo.sglang" not in rendered
    assert "dynamo.frontend" not in rendered


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_parsers_use_the_sglang_spellings(_docker_available, profile, service):
    rendered = render(profile)
    assert "--tool-call-parser=glm47" in rendered
    assert "--reasoning-parser=glm45" in rendered
    # The Dynamo spellings are not accepted by sglang.launch_server and would
    # fail at startup, after the old stack is already down.
    assert "--dyn-tool-call-parser" not in rendered
    assert "--dyn-reasoning-parser" not in rendered


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_binds_loopback_on_8000_by_default(_docker_available, profile, service):
    rendered = render(profile)
    assert "--host=127.0.0.1" in rendered
    assert "--port=8000" in rendered


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_orchestration_services_are_gone(_docker_available, profile, service):
    rendered = render(profile)
    for gone in ("etcd", "nats", "ETCD_ENDPOINTS", "NATS_SERVER"):
        assert gone not in rendered, f"{gone!r} still present in profile {profile}"


@pytest.mark.parametrize("profile,service", WORKER_PROFILES)
def test_exactly_one_service_per_profile(_docker_available, profile, service):
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "--profile", profile,
         "config", "--services"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.split() == [service]


def test_project_name_is_pinned(_docker_available):
    """Unpinned, the project name follows the directory -- so a rename
    orphans the running containers and the JIT cache volume."""
    assert "name: glm52-sglang" in render("cache")


def test_wildcard_profile_reaches_every_worker(_docker_available):
    """stop.sh tears down with --profile "*" so a worker can never be
    orphaned holding 8 GPUs. That must reach both workers and nothing else."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "--profile", "*",
         "config", "--services"],
        capture_output=True, text=True, check=True,
    )
    assert sorted(result.stdout.split()) == ["worker", "worker-longctx"]


def test_engine_tunables_are_unchanged(_docker_available):
    """This migration must be a no-op on serving behavior."""
    rendered = render("cache")
    for flag in (
        "--tp-size=8",
        "--page-size=64",
        "--kv-cache-dtype=fp8_e4m3",
        "--mem-fraction-static=0.85",
        "--context-length=524288",
        "--enable-hierarchical-cache",
        "--hicache-size=96",
        "--hicache-write-policy=write_through",
        "--hicache-storage-backend=file",
        "--speculative-algorithm=EAGLE",
    ):
        assert flag in rendered, f"engine tunable {flag!r} changed or lost"
    # SGLang must keep auto-selecting dsa/flashmla_kv.
    assert "--attention-backend" not in rendered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_compose_config.py -q`
Expected: FAIL. Against the current Dynamo compose file, at minimum
`test_cache_report_is_enabled_in_every_worker_profile`,
`test_entrypoint_is_native_sglang`, `test_parsers_use_the_sglang_spellings`,
`test_binds_loopback_on_8000_by_default`, `test_orchestration_services_are_gone`,
`test_exactly_one_service_per_profile`, and `test_project_name_is_pinned` all fail.
`test_engine_tunables_are_unchanged` should **pass** already — it is the guard that
those values survive the edit.

- [ ] **Step 3: Add the project name and delete the Dynamo services**

At the very top of `sglang/docker-compose.yml`, above the existing header comment block, add:

```yaml
# Pin the Compose project name instead of inheriting it from the directory.
# Inherited, it changes whenever the directory is renamed -- which silently
# orphans the running containers (teardown matches nothing while the worker
# holds all 8 GPUs) and the JIT cache volume (forcing a ~10-20 min DeepGEMM
# recompile). This survived the dynamo/ -> sglang/ rename precisely because
# it is pinned here.
name: glm52-sglang
```

Then delete, in full:
- the `x-dynamo-env: &dynamo-env` anchor block
- the `etcd:` service
- the `nats:` service
- the `frontend:` service
- the `etcd-data:` entry under top-level `volumes:`

- [ ] **Step 4: Update the shared worker base**

In `x-worker-base`, change the image default and entrypoint and drop `depends_on`:

```yaml
x-worker-base: &worker-base
  image: ${SGLANG_IMAGE:-glm52-sglang:0.5.13post1}
  network_mode: host
  gpus: all
  ipc: host
```

Delete the line `  depends_on: [etcd, nats]` and replace

```yaml
  entrypoint: ["python3", "-m", "dynamo.sglang"]
```

with

```yaml
  entrypoint: ["python3", "-m", "sglang.launch_server"]
```

Leave `init: true`, `cap_add: [SYS_PTRACE]`, `ulimits`, `restart: "no"` and every comment above them exactly as they are — they document engine-level behavior that this change does not touch.

- [ ] **Step 5: Update both worker services' `environment:` blocks**

In `worker` and `worker-longctx`, the merge key currently reads:

```yaml
      <<: [*dynamo-env, *coredump-env]
```

The `*dynamo-env` anchor no longer exists. Change both to:

```yaml
      <<: *coredump-env
```

- [ ] **Step 6: Rename the JIT cache volume key**

Under top-level `volumes:`, the block becomes exactly:

```yaml
volumes:
  jit-cache:   # persisted JIT kernel cache (fast worker restarts)
```

and in **both** worker services the mount line

```yaml
      - "dynamo-jit-cache:/home/dynamo/.cache"
```

becomes

```yaml
      - "jit-cache:/home/dynamo/.cache"
```

(The container-side path `/home/dynamo/.cache` is the image's user's home and does **not** change — it has nothing to do with Dynamo the product.)

- [ ] **Step 7: Update the `command:` of both worker services**

In **both** `worker` and `worker-longctx`, immediately after the `--served-model-name=` line, insert:

```yaml
      # The engine now serves the OpenAI API itself -- there is no separate
      # frontend process. network_mode: host means this binds the host's
      # interface directly, so the default is loopback ONLY. The model should
      # not be reachable over the network: front it with your own reverse proxy
      # or an SSH tunnel. Override with HTTP_HOST=0.0.0.0 (not recommended).
      - --host=${HTTP_HOST:-127.0.0.1}
      - --port=${PORT:-8000}
```

In **both**, replace

```yaml
      - --dyn-tool-call-parser=glm47
      - --dyn-reasoning-parser=glm45
```

with

```yaml
      # Native SGLang parsers. These were --dyn-* under Dynamo because the
      # FRONTEND did the parsing; sglang.launch_server rejects that spelling.
      - --tool-call-parser=glm47
      - --reasoning-parser=glm45
      # NOT OPTIONAL, and its absence is SILENT. This is what makes the server
      # populate usage.prompt_tokens_details.cached_tokens, which bench_stream.py
      # reads as a DIRECT count of prefix-cache hits -- the measurement behind
      # the whole tiered-KV result in RESULTS-kv-tiering.md. The Dynamo frontend
      # populated it for free. Drop this flag and there is no error: the field
      # just disappears and cache effectiveness has to be guessed from TTFT.
      - --enable-cache-report
```

- [ ] **Step 8: Rewrite the file's header comment**

The header block currently opens `# NVIDIA Dynamo serving stack for GLM-5.2-FP8 on 8x H200.` and explains the Dynamo/SGLang split. Replace the first paragraph with:

```yaml
# SGLang serving stack for GLM-5.2-FP8 on 8x H200.
#
# Engine: SGLang 0.5.13.post1 (custom image, see Dockerfile), serving the
# OpenAI-compatible API directly from sglang.launch_server. This used to run
# under NVIDIA Dynamo (etcd + NATS + a frontend process in front of the same
# worker); that was removed on 2026-09-11 because none of what Dynamo adds --
# multi-worker discovery, KV-aware routing, disaggregated prefill/decode -- is
# reachable on a single node with a single worker holding a ~756 GB model. The
# engine, the image, and every engine flag below are unchanged by that removal.
```

Leave the Topology paragraph (the TP=8-vs-EP+DP benchmark rationale) intact — it is still the reason for the current configuration. In it, change the phrase `All services use host networking, so the frontend binds :8000 on the host and workers reach etcd (:2379) / nats (:4222) over localhost.` to `The worker uses host networking and binds :8000 on the host directly.`

- [ ] **Step 9: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_compose_config.py -q`
Expected: PASS, all tests.

- [ ] **Step 10: Verify the rendered worker command by eye, once**

Run: `docker compose -f sglang/docker-compose.yml --profile cache config`
Expected: one service (`worker`); its `entrypoint` is `python3 -m sglang.launch_server`; the command list carries `--host=127.0.0.1`, `--port=8000`, `--enable-cache-report`, `--tool-call-parser=glm47`, `--reasoning-parser=glm45`; no `etcd`/`nats` anywhere; `name: glm52-sglang` at the top.

Repeat for `--profile longctx` and confirm the same, minus the hicache flags (that profile has never had them).

- [ ] **Step 11: Confirm the live stack is untouched**

Run: `docker compose -p dynamo ps`
Expected: the four `dynamo-*` containers still `Up`. Nothing in this task may stop them.

- [ ] **Step 12: Run the full suite and commit**

```bash
python3 -m pytest tests/ -q
git add sglang/docker-compose.yml tests/test_compose_config.py
git commit -m "feat(sglang): serve directly from sglang.launch_server, drop Dynamo

Deletes etcd, NATS and the Dynamo frontend. One container replaces four.
The engine, the image and every engine tunable are unchanged, so this is
measurable as a no-op on serving behavior.

Two changes are load-bearing rather than mechanical:

  --enable-cache-report is now required. The Dynamo frontend populated
  usage.prompt_tokens_details.cached_tokens for free; SGLang's native
  server does so only with this flag, and its absence is silent -- the
  field simply vanishes, taking the direct prefix-cache measurement
  behind RESULTS-kv-tiering.md with it. tests/test_compose_config.py
  asserts it at commit time for both profiles.

  name: glm52-sglang pins the Compose project. Inherited from the
  directory, it would have changed under the dynamo/ -> sglang/ rename,
  orphaning both the running containers and the JIT cache volume.

--dyn-tool-call-parser/--dyn-reasoning-parser were frontend flags and are
rejected by launch_server; they become --tool-call-parser=glm47 and
--reasoning-parser=glm45, verified present in the image."
```

---

### Task 3: Update the shell scripts

`serve.sh`, `stop.sh` and `bench.sh` share the `DYNAMO_IMAGE` → `SGLANG_IMAGE` rename, so they move together — a commit renaming only some call sites would leave the default tag inconsistent.

**Files:**
- Modify: `sglang/serve.sh`
- Modify: `sglang/stop.sh`
- Modify: `sglang/bench.sh`
- Test: `tests/test_compose_config.py` (extended with a stale-reference guard)

**Interfaces:**
- Consumes: the image tag default `glm52-sglang:0.5.13post1` and the service names `worker` / `worker-longctx` from Task 2.
- Produces: no new interfaces. `serve.sh` keeps its `PROFILE` contract (`cache` | `longctx`) and every existing env var except the renamed image one.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compose_config.py`:

```python
SCRIPTS = ["serve.sh", "stop.sh", "bench.sh"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_use_the_renamed_image_var(script):
    text = (COMPOSE.parent / script).read_text()
    assert "DYNAMO_IMAGE" not in text
    assert "glm52-dynamo-sglang" not in text


def test_stop_warns_before_destroying_the_jit_cache():
    """--volumes used to drop only the cheap etcd volume. The JIT cache is
    now the only volume, so the same flag costs a 10-20 min DeepGEMM
    recompile -- it has to say so at runtime, not just in a comment."""
    text = (COMPOSE.parent / "stop.sh").read_text()
    assert "etcd" not in text
    assert "jit" in text.lower()


@pytest.mark.parametrize("script", SCRIPTS + ["archive_worker_log.sh"])
def test_scripts_are_syntactically_valid(script):
    path = COMPOSE.parent / script
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_compose_config.py -q -k "scripts or stop_warns"`
Expected: FAIL — `DYNAMO_IMAGE` is still in all three scripts, and `stop.sh` still mentions `etcd`. `test_scripts_are_syntactically_valid` should already PASS.

- [ ] **Step 3: Update `sglang/serve.sh`**

Four edits, all mechanical except the last:

1. Header comment line 3: `# Serve GLM-5.2-FP8 via NVIDIA Dynamo (SGLang backend) on 8x H200, aggregated TP=8.` → `# Serve GLM-5.2-FP8 via SGLang on 8x H200, aggregated TP=8.`
2. Header comment line 4: `# Brings up etcd + NATS + Dynamo frontend + one SGLang worker (full model, all GPUs).` → `# Brings up one SGLang worker (full model, all GPUs), serving the OpenAI API itself.`
3. Tunables comment: `DYNAMO_IMAGE` → `SGLANG_IMAGE`.
4. The `IMAGE` assignment:

```bash
IMAGE="${SGLANG_IMAGE:-glm52-sglang:0.5.13post1}"
```

Then the closing here-doc. Replace:

```bash
  Follow worker startup:   docker compose -f $(pwd)/docker-compose.yml logs -f ${WORKER_SERVICE}
  Frontend logs:           docker compose -f $(pwd)/docker-compose.yml logs -f frontend
  Stop everything:         ./stop.sh
EOF
```

with:

```bash
  Follow worker startup:   docker compose -f $(pwd)/docker-compose.yml logs -f ${WORKER_SERVICE}
  Stop everything:         ./stop.sh
EOF
```

and change the line `Once the worker registers, test:` to `Once the worker finishes loading, test:` — there is no registry to register with any more. Note in passing that `:${PORT:-8000}` now does not accept connections at all until the engine is up, which makes it a true readiness signal; the old frontend answered immediately with an empty model list.

Everything else in `serve.sh` — the `PROFILE` case guard, the `realpath -m` normalization, the `KV_SCRATCH_DIR` prefix check, the containerized write probe, the py-spy capability probe, the pre-`up` log archive — stays exactly as it is. All of it is engine-level and none of it involved Dynamo.

- [ ] **Step 4: Update `sglang/stop.sh`**

Header comment: `# Stop and remove the Dynamo (SGLang) stack. Keeps the etcd data volume.` → `# Stop and remove the SGLang stack.`

Then replace the `--volumes` usage line and add the runtime warning. The second line currently reads `# Pass --volumes to also drop the etcd volume.`; it becomes:

```bash
# Pass --volumes to also drop the JIT kernel cache volume (see the warning below).
```

And immediately after the `cd "$(dirname "$0")"` line, insert:

```bash
# --volumes used to drop only the etcd data volume, which cost nothing to
# rebuild. That volume is gone; the JIT kernel cache is now the ONLY volume,
# so the same flag now throws away every DeepGEMM/FlashInfer/Triton kernel the
# worker has compiled and buys a ~10-20 min precompile on the next start.
# Warn, but do not prompt -- this script is called from other scripts.
for arg in "$@"; do
  case "${arg}" in
    --volumes|-v)
      echo "WARNING: --volumes will delete the JIT kernel cache volume." >&2
      echo "         The next ./serve.sh will pay a ~10-20 min DeepGEMM precompile." >&2
      ;;
  esac
done
```

Finally, the last line's message: `echo "Dynamo stack stopped."` → `echo "SGLang stack stopped."`

The profile-wildcard teardown block and the two `archive_worker_log.sh` calls are unchanged — both worker service names still exist.

- [ ] **Step 5: Update `sglang/bench.sh`**

Two edits:

1. Header: `# Benchmark the running OpenAI endpoint (Dynamo on :8000) with NVIDIA's load` → `# Benchmark the running OpenAI endpoint (SGLang on :8000) with NVIDIA's load`
2. `IMAGE="${DYNAMO_IMAGE:-glm52-dynamo-sglang:0.5.13post1}"` → `IMAGE="${SGLANG_IMAGE:-glm52-sglang:0.5.13post1}"`

- [ ] **Step 6: Run the tests to verify they pass**

Run (from the repo root): `python3 -m pytest tests/ -q`
Expected: PASS, all tests including the three new ones.

- [ ] **Step 7: Verify `stop.sh`'s warning fires without running docker**

Run: `bash -n sglang/stop.sh && grep -A2 'WARNING: --volumes' sglang/stop.sh`
Expected: syntax OK, and the warning block is present ahead of any `docker` invocation.

- [ ] **Step 8: Commit**

```bash
git add sglang/serve.sh sglang/stop.sh sglang/bench.sh tests/test_compose_config.py
git commit -m "refactor(sglang): repoint the scripts at the SGLang-native stack

DYNAMO_IMAGE -> SGLANG_IMAGE, default tag glm52-sglang:0.5.13post1, and
the frontend-logs line drops out of serve.sh's closing message.

stop.sh --volumes gains a runtime warning. It used to drop only the etcd
data volume, which was free to rebuild; the JIT kernel cache is now the
only volume, so the same flag costs a ~10-20 min DeepGEMM precompile. It
warns rather than prompts because other scripts call it.

serve.sh's preflight -- the containerized KV-scratch write probe, the
py-spy capability probe, and the pre-up log archive -- is unchanged. All
of it is engine-level and none of it involved Dynamo."
```

---

### Task 4: Correct the documentation and add the cutover runbook

Prose only. No executable behavior changes, which is why it lands last and why the historical records are explicitly excluded.

**Files:**
- Modify: `README.md`
- Modify: `sglang/README.md`
- Modify: `CONTEXT_WINDOW.md`
- Modify: `sglang/RESULTS-kv-tiering.md` (exactly one added line)
- Modify: `sglang/bench_stream.py:2`, `sglang/bench_stream.py:129-132` (comments only)
- Test: `tests/test_bench_stream.py` (must still pass — comments only, no code change)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: nothing consumed by later tasks. This is the last task.

- [ ] **Step 1: Correct `sglang/bench_stream.py` (comments only)**

Line 2: `"""Tiny streaming benchmark for the GLM-5.2-FP8 Dynamo (SGLang) OpenAI endpoint.` → `"""Tiny streaming benchmark for the GLM-5.2-FP8 SGLang OpenAI endpoint.`

Lines 129–132, the conditional comment becomes a statement of fact:

```python
                # OpenAI-shaped cache accounting -- a DIRECT read of prefix-cache
                # hits rather than a TTFT inference, worth far more than the
                # timing numbers for verifying the tiering actually works.
                # The server populates this because --enable-cache-report is set
                # in docker-compose.yml; WITHOUT that flag the field is silently
                # absent and this reads None.
```

Do not touch the `cached_tokens = ...` line itself or any other code.

- [ ] **Step 2: Verify no behavior changed**

Run: `python3 -m pytest tests/test_bench_stream.py -q`
Expected: PASS, same count as before.

- [ ] **Step 3: Add the one-line provenance header to `sglang/RESULTS-kv-tiering.md`**

Immediately below the document's title line, insert:

```markdown
> **Measured 2026-08-31/09-01 on the NVIDIA Dynamo + SGLang 0.5.13.post1 stack**, before
> Dynamo was removed (2026-09-11). The engine, the image, and every engine flag are
> unchanged by that removal, so these numbers carry over; only the wrapper around them
> is gone. Dynamo references below are left as written — this is a record of what was
> measured, not a description of the current stack.
```

Change nothing else in this file. Same for `docs/superpowers/plans/2026-08-31-kv-cache-tiering.md` and `docs/superpowers/specs/2026-08-31-scratch-kv-cache-tiering-design.md`: leave them entirely alone.

- [ ] **Step 4: Rewrite the top-level `README.md`**

- **Delete** the two leading lines added in the working tree: the blank line and the `https://docs.sglang.io/cookbook/.../GLM-5.3-Flash#...` URL. That link is about a different model and a different recipe; it is explicitly out of scope and should not ship inside this change.
- Title: `# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang, 8× H200)` → `# GLM-5.2-FP8 on SGLang (8× H200)`
- Opening paragraph: replace `across all 8 H200 GPUs via **NVIDIA Dynamo** with the **SGLang** backend, exposing an OpenAI-compatible API on `:8000`.` with `across all 8 H200 GPUs with **SGLang**, exposing an OpenAI-compatible API on `:8000`.`
- Replace the existing block-quote about vLLM removal with one that records both migrations:

```markdown
> This repo has served the model two ways before: a plain vLLM container, and then
> NVIDIA Dynamo with the SGLang backend. vLLM was dropped because no Dynamo runtime
> ships vLLM ≥ 0.23.0, which GLM-5.2's sparse MLA needs. Dynamo itself was dropped on
> 2026-09-11: its frontend duplicated `sglang.launch_server`, and multi-worker
> discovery, KV-aware routing, and disaggregated prefill/decode are all unreachable on
> a single node holding one ~756 GB model. The engine has been SGLang throughout, and
> no engine flag changed when Dynamo was removed. See `sglang/README.md`.
```

- Every `dynamo/` path reference becomes `sglang/`.
- In the Serve section, the stack description `The stack = etcd + NATS + Dynamo frontend + one SGLang worker (aggregated, TP=8, ...` becomes `The stack = one SGLang worker (aggregated, TP=8, ...`.
- `docker compose logs -f worker` and the `curl` examples are unchanged.
- Leave the `./chat.py` line alone. It is a pre-existing discrepancy (the file is not in the repo) and fixing it here would be unrelated scope creep — flag it in the final report instead.

- [ ] **Step 5: Rewrite `sglang/README.md`**

- Title: `# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang backend, 8× H200)` → `# GLM-5.2-FP8 on SGLang (8× H200)`
- Opening: drop "using **NVIDIA Dynamo** as the serving/orchestration layer"; state that `sglang.launch_server` serves the OpenAI API directly on `:8000`.
- **Keep in full, unchanged:** "Why SGLang, not vLLM" (still the reason for the image), "Why aggregated only (no disaggregation on one node)", the EP+DP-attention regression table, and the entire "Tiered KV cache" section including both Profile B write-ups. All of it is engine-level and still true.
- In "Why SGLang, not vLLM", the sentence about no Dynamo `vllm-runtime` image shipping vLLM ≥ 0.23.0 stays — it is the historical reason the image is what it is — but add: "Dynamo itself is no longer used; the image it gave us is, see the Dockerfile."
- Under **Performance levers**, the KV-aware routing bullet currently says "add `--router-mode kv` to the frontend". There is no frontend. Replace that bullet with:

```markdown
- **KV-aware routing** — would require reintroducing a router process in front of the
  worker (this is one of the things Dynamo provided). Moot either way at this scale: it
  only pays off with ≥ 2 workers/replicas, and one node holds exactly one copy of a
  756 GB model.
```

- In the file-inventory list, `bench.sh — load-test any OpenAI endpoint (Dynamo or vLLM)` → `bench.sh — load-test any OpenAI endpoint`.
- Tunables list: `DYNAMO_IMAGE` → `SGLANG_IMAGE`; add `HTTP_HOST` (default `127.0.0.1`) and note `PORT` now goes to the worker.
- **Add a `### Operational endpoints` note** under Usage. The native server exposes endpoints the Dynamo frontend did not proxy — `/health`, `/health_generate`, `/get_server_info`, `/flush_cache` — all on `${HTTP_HOST:-127.0.0.1}` like everything else, so the exposure posture is unchanged. Call out `/flush_cache` specifically: eviction benchmarking currently requires a full worker restart, and this replaces it.

- **Add a new `## Cutover from the Dynamo stack` section** with exactly this content:

````markdown
## Cutover from the Dynamo stack

One-time, run by hand. The Dynamo stack keeps serving until you run this.

By this point `dynamo/` no longer exists — the rename has landed — so the old stack
cannot be torn down by its own `stop.sh`. Reach it by project name instead:
`docker compose -p dynamo ps` resolves a running project from container labels with
no compose file present (`config` does not, but `ps` and `down` do).

```bash
# 1. Archive the old worker's log BEFORE teardown -- docker deletes a
#    container's log with the container, and that log is the only place
#    SGLang's watchdog writes its per-rank py-spy dump.
./sglang/archive_worker_log.sh dynamo-worker-1 /scratch/diag/logs

# 2. Tear down the old stack by project name.
docker compose -p dynamo down

# 3. One-time: carry the JIT kernel cache across the project rename,
#    avoiding a ~10-20 min DeepGEMM recompile on first start.
docker volume create glm52-sglang_jit-cache
docker run --rm \
  -v dynamo_dynamo-jit-cache:/from \
  -v glm52-sglang_jit-cache:/to \
  alpine sh -c 'cp -a /from/. /to/'

# 4. One-time: retag the existing image so serve.sh does not rebuild it
#    (the rebuild needs host networking + a pip proxy).
docker tag glm52-dynamo-sglang:0.5.13post1 glm52-sglang:0.5.13post1

# 5. Start.
cd sglang && ./serve.sh
docker compose logs -f worker
```

Step 4 matters: `serve.sh` builds the image only when it is absent, so the renamed
default tag would otherwise trigger a full rebuild through the proxy on an
otherwise-offline host.

Then verify — the third check is the one that must not be skipped, because it is the
only one that fails silently:

```bash
curl localhost:8000/v1/models                      # returns glm-5.2-fp8
./bench_stream.py --concurrency 1 --num 4          # reasoning_content streams
./bench_stream.py --concurrency 1 --num 4          # 2nd run: cached_tokens > 0
```

Rollback is `git revert` plus one restart. The old image survives under its original
tag and `dynamo_dynamo-jit-cache` is copied rather than moved, so both are intact.
````

- [ ] **Step 6: Update `CONTEXT_WINDOW.md`**

Six references. All six are either a `dynamo/` path (→ `sglang/`) or the phrase `Dynamo + SGLang 0.5.13.post1` in the comparison table's Backend row (→ `SGLang 0.5.13.post1`). The measured findings, the OOM attempt history, and every number in this file are unchanged.

- [ ] **Step 7: Verify no stale references survive outside the frozen records**

Run:

```bash
grep -rni dynamo --include='*.md' --include='*.sh' --include='*.py' --include='*.yml' --include='*.toml' . \
  | grep -v '^./.git' \
  | grep -v 'RESULTS-kv-tiering.md' \
  | grep -v 'docs/superpowers/plans/2026-08-31' \
  | grep -v 'docs/superpowers/specs/2026-08-31' \
  | grep -v 'docs/superpowers/specs/2026-09-11' \
  | grep -v 'docs/superpowers/plans/2026-09-11'
```

Expected: only *intentional historical* mentions remain — the migration note in `README.md`, the "Dynamo itself is no longer used" line and the cutover runbook in `sglang/README.md`, the compose header's removal note, and `stop.sh`/`serve.sh` comments that explain what was removed. Every one should be a deliberate past-tense reference. Any `dynamo/` **path**, any `DYNAMO_IMAGE`, or any present-tense claim that Dynamo is in use is a bug — fix it.

- [ ] **Step 8: Update `pyproject.toml`**

`description = "Serve GLM-5.2-FP8 on 8x H200 via NVIDIA Dynamo (SGLang backend)"` → `description = "Serve GLM-5.2-FP8 on 8x H200 with SGLang"`

- [ ] **Step 9: Run the full suite**

Run (from the repo root): `python3 -m pytest tests/ -q`
Expected: PASS, all tests.

- [ ] **Step 10: Confirm the live stack is still untouched**

Run: `docker compose -p dynamo ps`
Expected: the four `dynamo-*` containers still `Up`. Nothing in this plan stops them — cutover is the operator's call.

- [ ] **Step 11: Commit**

```bash
git add README.md sglang/README.md CONTEXT_WINDOW.md sglang/RESULTS-kv-tiering.md \
        sglang/bench_stream.py pyproject.toml
git commit -m "docs: describe the SGLang-native stack, add the cutover runbook

Rewrites the live docs -- README.md, sglang/README.md, CONTEXT_WINDOW.md,
compose comments -- around one SGLang container instead of four Dynamo
ones, and adds the operator cutover runbook to sglang/README.md.

Sections that survive unchanged because they remain true and remain the
reason for the current config: 'Why SGLang, not vLLM', 'Why aggregated
only', the EP+DP-attention regression table, and all of 'Tiered KV
cache'. The KV-aware-routing lever is rewritten -- it described adding a
flag to a frontend that no longer exists.

RESULTS-kv-tiering.md and the 2026-08-31 spec/plan keep their Dynamo
references and gain only a provenance header: they are dated records of
what was measured, and the engine and flags they measured are unchanged,
so the numbers carry over. Rewriting them would falsify the record.

Also drops the GLM-5.3-Flash cookbook link from the working tree --
different model, different recipe, its own evaluation."
```

---

## Post-implementation report

After Task 4, report to the human partner:

1. **What is committed and what is not.** All four commits are on branch `drop-dynamo`. The live Dynamo stack is still serving and untouched.
2. **The cutover is theirs to run** — quote the five-step runbook from `sglang/README.md`, and state plainly that steps 3 and 4 (volume migration, image retag) are one-time and skipping them costs a ~10–20 min recompile and a full proxied image rebuild respectively.
3. **The three post-cutover checks that cannot be run here**, from the spec's Verification section: `/v1/models` returns `glm-5.2-fp8`; a streamed completion yields `reasoning_content`; and `bench_stream.py` run twice reports non-zero `cached_tokens` on the second run. Flag the third as the one that must not be skipped — it is the only check that fails silently.
4. **The pre-existing `./chat.py` discrepancy** in `README.md`, untouched, so it is not mistaken for collateral damage from this change.
