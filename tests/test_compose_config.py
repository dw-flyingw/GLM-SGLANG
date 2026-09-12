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
import re
import shutil
import subprocess

import pytest

COMPOSE = pathlib.Path(__file__).resolve().parents[1] / "sglang" / "docker-compose.yml"
SERVE = COMPOSE.parent / "serve.sh"

# Every worker profile. worker-flash was missing from this list from the day it was
# added, so the parametrized checks below -- including the cache-report one this
# file calls the important one -- never covered the model actually being served.
WORKER_PROFILES = [
    ("cache", "worker"),
    ("longctx", "worker-longctx"),
    ("flash", "worker-flash"),
]


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
    orphaned holding 8 GPUs. That must reach EVERY worker and nothing else --
    add any new worker service here, or stop.sh will silently leave it
    running on all 8 GPUs."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "--profile", "*",
         "config", "--services"],
        capture_output=True, text=True, check=True,
    )
    assert sorted(result.stdout.split()) == [
        "worker", "worker-flash", "worker-longctx"]


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


SCRIPTS = ["serve.sh", "stop.sh", "bench.sh"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_use_the_renamed_image_var(script):
    text = (COMPOSE.parent / script).read_text()
    assert "DYNAMO_IMAGE" not in text
    assert "glm52-dynamo-sglang" not in text


def test_stop_warns_before_destroying_the_jit_cache():
    """--volumes used to drop only the cheap etcd volume. The JIT cache is
    now the only volume, so the same flag costs a 10-20 min DeepGEMM
    recompile -- it has to say so at RUNTIME, not just in a comment.

    Asserted on behavior, not vocabulary: stop.sh may still explain what
    etcd was, it just may not still claim to be keeping its volume.
    """
    text = (COMPOSE.parent / "stop.sh").read_text()
    assert "Keeps the etcd data volume" not in text
    assert "drop the etcd volume" not in text
    # The warning must be echoed, i.e. reachable at runtime.
    assert "WARNING: --volumes will delete the JIT kernel cache volume." in text
    assert "--volumes|-v)" in text


@pytest.mark.parametrize("script", SCRIPTS + ["archive_worker_log.sh"])
def test_scripts_are_syntactically_valid(script):
    path = COMPOSE.parent / script
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_serve_defaults_to_flash_profile():
    """A bare ./serve.sh -- or any restart script that forgets PROFILE -- boots
    whatever this default names onto all 8 GPUs. It was `cache` until
    2026-09-12, which after GLM-5.2 was retired meant silently starting the
    retired model. Static check: needs no docker."""
    m = re.search(r'^PROFILE="\$\{PROFILE:-(\w+)\}"$', SERVE.read_text(), re.M)
    assert m, "could not find the PROFILE default line in serve.sh"
    assert m.group(1) == "flash"
