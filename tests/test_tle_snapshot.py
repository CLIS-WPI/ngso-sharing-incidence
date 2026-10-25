"""The pinned TLE snapshot: present, byte-exact, and never silently replaced."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ngsi.tle import check_tle_cache, fetch_group, recorded_tle_sha256

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # sha256(b"abc")


def test_snapshot_matches_readme():
    recorded = recorded_tle_sha256(RAW / "README.md")
    assert {"starlink.tle", "oneweb.tle"} <= set(recorded)
    for name, digest in recorded.items():
        path = RAW / "tle" / name
        assert path.is_file(), f"{path} missing: git checkout -- data/raw/tle"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def _pinned(tmp_path: Path, content: bytes) -> Path:
    (tmp_path / "tle").mkdir()
    (tmp_path / "README.md").write_text(f"| `tle/demo.tle` | x | `{SHA256_ABC}` |\n")
    (tmp_path / "tle" / "demo.tle").write_bytes(content)
    return tmp_path / "tle"


def test_fetch_group_reads_matching_snapshot(tmp_path):
    assert fetch_group("demo", _pinned(tmp_path, b"abc")) == "abc"


def test_fetch_group_fails_on_hash_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="sha256"):
        fetch_group("demo", _pinned(tmp_path, b"abd"))


def test_fetch_group_fails_when_missing_instead_of_downloading(tmp_path):
    cache = _pinned(tmp_path, b"abc")
    (cache / "demo.tle").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        fetch_group("demo", cache)


def test_check_tle_cache_fails_when_not_recorded(tmp_path):
    cache = _pinned(tmp_path, b"abc")
    (cache / "other.tle").write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="No sha256 recorded"):
        check_tle_cache(cache / "other.tle")
