"""Tests for trucktrack.valhalla.setup: building tiles from an OSM extract."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("valhalla")

from trucktrack.valhalla.setup import build_tiles, main  # noqa: E402


def test_missing_extract_fails_before_building(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PBF file not found"):
        build_tiles(tmp_path / "missing.osm.pbf", tmp_path / "tiles")
    assert not (tmp_path / "tiles").exists()


def test_failed_build_leaves_no_config(tmp_path: Path) -> None:
    # Valhalla's admin build rejects an empty extract within a second.
    empty = tmp_path / "empty.osm.pbf"
    empty.touch()
    with pytest.raises(subprocess.CalledProcessError):
        build_tiles(empty, tmp_path / "tiles")
    assert not (tmp_path / "tiles" / "valhalla.json").exists()


def test_command_line_reports_missing_extract(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--pbf", str(tmp_path / "missing.osm.pbf")])
    assert excinfo.value.code == 2
