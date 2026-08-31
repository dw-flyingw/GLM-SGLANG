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
