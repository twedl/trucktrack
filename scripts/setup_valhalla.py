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
import subprocess
import sys
from pathlib import Path


def build_config(tile_dir: Path, config_path: Path) -> None:
    """Generate a Valhalla config file with truck-optimised settings."""
    cmd = [
        sys.executable,
        "-m",
        "valhalla.valhalla_build_config",
        "--mjolnir-tile-dir",
        str(tile_dir),
        "--mjolnir-max-cache-size",
        "1000000000",
        "--mjolnir-id-table-size",
        "13000000000",
        "--mjolnir-use-lru-mem-cache",
        "True",
        "--mjolnir-lru-mem-cache-hard-control",
        "True",
        "--mjolnir-use-simple-mem-cache",
        "False",
        "--mjolnir-keep-all-osm-node-ids",
        "False",
        "--mjolnir-keep-osm-node-ids",
        "True",
        "--mjolnir-include-bicycle",
        "False",
        "--mjolnir-include-pedestrian",
        "False",
        "--mjolnir-include-driving",
        "True",
        "--mjolnir-data-processing-use-direction-on-ways",
        "True",
        "--service-limits-trace-max-distance",
        "2000000.0",
        "-o",
        str(config_path),
    ]
    print(f"Generating config: {config_path}", file=sys.stderr)
    subprocess.run(cmd, check=True)


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

    config_path = (args.config_out or tile_dir / "valhalla.json").resolve()
    build_config(tile_dir, config_path)

    cmd = [
        "valhalla_build_tiles",
        "-c",
        str(config_path),
        str(args.pbf.resolve()),
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"valhalla_build_tiles exited with code {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode

    print(f"Tiles built in {tile_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
