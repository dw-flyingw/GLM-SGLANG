"""Regression tests for dynamo/archive_worker_log.sh.

The 2026-09-03 exit-137 hang was un-diagnosable for two independent reasons.
The first (py-spy denied by Yama ptrace_scope=1) is fixed by `cap_add:
SYS_PTRACE` in docker-compose.yml TOGETHER WITH the cap_sys_ptrace+ep file
capability on py-spy in the Dockerfile -- neither works alone, since the
worker runs as uid 1000 and cap_add only reaches the bounding set. These
tests pin the second reason.

SGLang's watchdog writes its py-spy stack dumps to the worker's stdout/stderr,
which lands in the container's docker json-file log and NOWHERE else. Both
teardown paths in this repo destroy that log:

  - stop.sh runs `docker compose down`, which REMOVES the containers;
  - serve.sh runs `docker compose up -d`, which RECREATES a worker whose
    config changed.

Docker deletes a container's log directory along with the container, so the
only copy of the watchdog's output is gone before anyone reads it. That is
exactly what happened: the worker died at 15:10 and the log was destroyed by
the 18:05 restart.

The archive must therefore outlive the container it came from.
"""
import gzip
import shutil
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(
        ["docker", "info"], capture_output=True, text=True
    ).returncode
    != 0,
    reason="needs a usable docker daemon",
)


def _run(script, *args):
    return subprocess.run(
        [str(script), *args], capture_output=True, text=True, timeout=120
    )


@pytest.fixture
def container():
    """A stopped throwaway container, removed even if the test fails."""
    made = []

    def _make(shell):
        name = f"archive-test-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            ["docker", "run", "--name", name, "alpine:3", "sh", "-c", shell],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        made.append(name)
        return name

    yield _make

    for name in made:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_archive_survives_container_removal(archive_script, container, tmp_path):
    """The regression: the log must still be readable after the container is gone."""
    marker = f"watchdog-dump-{uuid.uuid4().hex}"
    name = container(f"echo {marker}")

    result = _run(archive_script, name, str(tmp_path))
    assert result.returncode == 0, result.stderr

    # Destroy the container exactly as `docker compose down` would.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=True)
    assert (
        subprocess.run(
            ["docker", "logs", name], capture_output=True
        ).returncode
        != 0
    ), "container should be gone, so docker logs must fail"

    archives = list(tmp_path.glob("*.log.gz"))
    assert len(archives) == 1, f"expected exactly one archive, got {archives}"
    assert marker in gzip.decompress(archives[0].read_bytes()).decode()


def test_captures_stderr_as_well_as_stdout(archive_script, container, tmp_path):
    """SGLang logs through stderr; an archive that dropped it would be useless."""
    marker = f"stderr-{uuid.uuid4().hex}"
    name = container(f"echo {marker} >&2")

    assert _run(archive_script, name, str(tmp_path)).returncode == 0

    archives = list(tmp_path.glob("*.log.gz"))
    assert len(archives) == 1
    assert marker in gzip.decompress(archives[0].read_bytes()).decode()


def test_absent_container_is_not_an_error(archive_script, tmp_path):
    """A broken diagnostics path must never stop the model from serving.

    serve.sh calls this on a first-ever start, when no previous worker exists.
    """
    result = _run(archive_script, "definitely-not-a-container", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("*.log.gz")) == []


def test_unwritable_destination_is_not_an_error(archive_script, container, tmp_path):
    """Same rule: a missing or read-only DIAG_DIR must not abort the caller."""
    name = container("echo hello")
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        result = _run(archive_script, name, str(unwritable / "logs"))
        assert result.returncode == 0, result.stderr
    finally:
        unwritable.chmod(0o700)


def test_successive_archives_do_not_clobber_each_other(
    archive_script, container, tmp_path
):
    """Two incidents in one day must not collapse into a single file."""
    first = container(f"echo first-{uuid.uuid4().hex}")
    second = container(f"echo second-{uuid.uuid4().hex}")

    assert _run(archive_script, first, str(tmp_path)).returncode == 0
    assert _run(archive_script, second, str(tmp_path)).returncode == 0

    assert len(list(tmp_path.glob("*.log.gz"))) == 2


def test_archive_is_labelled_by_container_name_even_when_given_an_id(
    archive_script, container, tmp_path
):
    """stop.sh resolves services to raw ids; the filename must still say which
    worker it was, not just a 64-hex blob."""
    name = container("echo hello")
    cid = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert _run(archive_script, cid, str(tmp_path)).returncode == 0

    archives = list(tmp_path.glob("*.log.gz"))
    assert len(archives) == 1
    assert archives[0].name.startswith(name), archives[0].name
    assert not archives[0].name.startswith(cid), "raw id used instead of name"


def test_empty_log_is_skipped(archive_script, container, tmp_path):
    """Don't litter the diag directory with zero-content archives."""
    name = container("true")

    assert _run(archive_script, name, str(tmp_path)).returncode == 0
    assert list(tmp_path.glob("*.log.gz")) == []
