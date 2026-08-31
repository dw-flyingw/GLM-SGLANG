import os
import pathlib
import sys

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


def test_filenotfounderror_does_not_over_evict(kv_reaper, tmp_path, monkeypatch):
    """FileNotFoundError: file already gone, don't over-delete valid entries.

    Four 100-byte files with increasing mtimes, budget 250 (so two must be
    deleted). Simulate the oldest vanishing before reap reaches it via
    monkeypatch. Without the fix (FileNotFoundError not decrementing total,
    i.e. treated like a plain OSError), reap makes three unlink attempts
    (oldest.bin, old.bin, new.bin) but only two files are actually deleted
    (old.bin and new.bin) -- oldest.bin was already gone, so it evicts one
    file further than necessary. With the fix, it deletes exactly one
    (old.bin) and the two newest survive.
    """
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "oldest.bin", 100, 1000)
    _write(root / "old.bin", 100, 2000)
    _write(root / "new.bin", 100, 3000)
    _write(root / "newest.bin", 100, 4000)

    # Monkeypatch unlink to raise FileNotFoundError for oldest.bin
    original_unlink = pathlib.Path.unlink

    def patched_unlink(self):
        if self.name == "oldest.bin":
            raise FileNotFoundError("already gone")
        return original_unlink(self)

    monkeypatch.setattr(pathlib.Path, "unlink", patched_unlink)

    removed, reclaimed = kv_reaper.reap(root, max_bytes=250)
    # Should delete exactly 1 (oldest.bin's FileNotFoundError doesn't count)
    # and old.bin's real delete
    assert removed == 1, "reap should only count its own deletions, not already-gone files"
    assert reclaimed == 100, "should only credit the one file we actually deleted"
    assert not (root / "old.bin").exists(), "old.bin should be deleted"
    assert (root / "new.bin").exists(), "new.bin must survive"
    assert (root / "newest.bin").exists(), "newest.bin must survive"


def test_undeletable_file_still_counts_against_the_budget(kv_reaper, tmp_path, monkeypatch):
    """OSError: undeletable file counts toward budget, forcing deeper deletion.

    Four 100-byte files (f0..f3) with budget 150: need to free 250 bytes (3 files).
    Make f0 (oldest) undeletable via PermissionError. Correct behaviour: f0 fails
    and stays in total=400, so loop must delete f1, f2, f3 to reach 100.
    Result: reap(...) == (3, 300), f0 survives, f3 is gone.

    Regression to catch: if OSError wrongly decremented total, loop would stop
    at 2 deletions (f1, f2 only), f3 would survive, and reap would return (2, 200).
    """
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "f0.bin", 100, 1000)  # oldest, undeletable
    _write(root / "f1.bin", 100, 2000)
    _write(root / "f2.bin", 100, 3000)
    _write(root / "f3.bin", 100, 4000)  # newest

    # Monkeypatch unlink to raise PermissionError for f0.bin
    original_unlink = pathlib.Path.unlink

    def patched_unlink(self):
        if self.name == "f0.bin":
            raise PermissionError("cannot delete")
        return original_unlink(self)

    monkeypatch.setattr(pathlib.Path, "unlink", patched_unlink)

    removed, reclaimed = kv_reaper.reap(root, max_bytes=150)
    # f0 is undeletable and still counts toward total=400.
    # Need 250 bytes freed: deletes f1, f2, f3 to reach 100.
    assert removed == 3, "must delete 3 files to offset the uncounted one"
    assert reclaimed == 300, "must reclaim 300 bytes (3 * 100)"
    assert (root / "f0.bin").exists(), "undeletable f0 must survive"
    assert not (root / "f3.bin").exists(), "even newest file f3 must be deleted to compensate"


def test_main_warns_and_exits_nonzero_when_every_delete_fails(
    kv_reaper, tmp_path, monkeypatch, capsys
):
    """If the tree is over budget and every unlink fails, main() must not
    report a quiet success (identical to the healthy under-budget case) --
    cron needs a real failure signal or /scratch silently fills.

    Two 100-byte files, budget 100 (one must go), but unlink always raises.
    reap() returns (0, 0) -- same value as a genuinely under-budget run --
    so main() must distinguish the two some other way (by checking whether
    the tree is still over budget after the run) and exit non-zero with a
    stderr warning in the over-budget case.
    """
    root = tmp_path / "scratch" / "kvcache"
    _write(root / "old.bin", 100, 1000)
    _write(root / "new.bin", 100, 2000)

    def patched_unlink(self):
        raise PermissionError("cannot delete")

    monkeypatch.setattr(pathlib.Path, "unlink", patched_unlink)
    monkeypatch.setattr(
        sys, "argv",
        ["kv_reaper", "--root", str(root), "--max-bytes", "100"],
    )

    with pytest.raises(SystemExit) as exc_info:
        kv_reaper.main()
    assert exc_info.value.code != 0, "must exit non-zero, unlike the healthy under-budget case"

    captured = capsys.readouterr()
    assert "removed 0 files" in captured.out
    assert "over budget" in captured.err.lower()
    assert (root / "old.bin").exists()
    assert (root / "new.bin").exists()
