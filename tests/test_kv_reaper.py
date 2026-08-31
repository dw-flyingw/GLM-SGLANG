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
