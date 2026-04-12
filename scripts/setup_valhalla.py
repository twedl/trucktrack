#!/usr/bin/env python3
"""Build Valhalla routing tiles from an OSM PBF extract.

Usage:
    uv run python scripts/setup_valhalla.py planet.osm.pbf [--tile-dir valhalla_tiles]

This creates a valhalla.json config via valhalla_build_config and runs the
tile builder. The resulting tile directory can be passed to trucktrack via
--tile-extract.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

CONFIG_OPTIONS: dict[str, bool | int | float] = {
    "mjolnir-max-cache-size": 1_000_000_000,
    "mjolnir-id-table-size": 13_000_000_000,
    "mjolnir-use-lru-mem-cache": True,
    "mjolnir-lru-mem-cache-hard-control": True,
    "mjolnir-use-simple-mem-cache": False,
    "mjolnir-keep-all-osm-node-ids": False,
    "mjolnir-keep-osm-node-ids": True,
    "mjolnir-include-bicycle": False,
    "mjolnir-include-pedestrian": False,
    "mjolnir-include-driving": True,
    "mjolnir-data-processing-use-direction-on-ways": True,
    "service-limits-trace-max-distance": 2_000_000.0,
    "service-limits-trace-max-alternates": 10,
    "service-limits-trace-max-alternates-shape": 1000,
    "service-limits-trace-max-search-radius": 200.0,
    # Meili map-matching defaults tuned for sparse truck GPS (~60s intervals).
    "meili-default-breakage-distance": 3000,
    "meili-default-search-radius": 50,
    "meili-default-gps-accuracy": 15.0,
    "meili-default-max-search-radius": 200,
    "meili-default-max-route-distance-factor": 10,
    "meili-default-max-route-time-factor": 10,
    "meili-default-beta": 5,
    "meili-default-interpolation-distance": 20,
}


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess, printing a status line and exiting on failure."""
    print(f"{description}: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"{cmd[0]} exited with code {result.returncode}", file=sys.stderr)
        raise SystemExit(result.returncode)


def build_config(tile_dir: Path, tar_path: Path, config_path: Path) -> None:
    """Generate a Valhalla config file with truck-optimised settings."""
    cmd = [
        sys.executable,
        "-m",
        "valhalla.valhalla_build_config",
        "--mjolnir-tile-dir",
        str(tile_dir),
        "--mjolnir-tile-extract",
        str(tar_path),
    ]
    for key, value in CONFIG_OPTIONS.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["-o", str(config_path)])
    _run(cmd, f"Generating config: {config_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Valhalla routing tiles from an OSM PBF file.",
    )
    parser.add_argument("pbf", type=Path, help="Path to an OSM PBF extract.")
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=Path("valhalla_tiles"),
        help="Directory for tile output (default: valhalla_tiles).",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Where to write valhalla.json (default: <tile-dir>/valhalla.json).",
    )
    args = parser.parse_args(argv)

    if not args.pbf.exists():
        parser.error(f"PBF file not found: {args.pbf}")

    tile_dir: Path = args.tile_dir.resolve()
    tile_dir.mkdir(parents=True, exist_ok=True)

    tar_path = (tile_dir / "valhalla_tiles.tar").resolve()
    config_path = (args.config_out or tile_dir / "valhalla.json").resolve()
    build_config(tile_dir, tar_path, config_path)

    _run(
        ["valhalla_build_tiles", "-c", str(config_path), str(args.pbf.resolve())],
        "Building tiles",
    )

    # Tar the tile hierarchy and remove loose tile directories.
    tile_subdirs = sorted(p for p in tile_dir.iterdir() if p.is_dir())
    print(f"Packing {len(tile_subdirs)} tile directories into {tar_path}", file=sys.stderr)
    try:
        with tarfile.open(tar_path, "w") as tar:
            for subdir in tile_subdirs:
                tar.add(subdir, arcname=subdir.name)
    except Exception:
        tar_path.unlink(missing_ok=True)
        raise
    for subdir in tile_subdirs:
        shutil.rmtree(subdir)

    print(f"Done: {tar_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
