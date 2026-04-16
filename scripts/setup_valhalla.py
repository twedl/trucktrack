#!/usr/bin/env python3
"""Build local Valhalla routing tiles via the bundled pyvalhalla binaries.

Running with no arguments downloads the Ontario OSM extract from Geofabrik
(~250 MB) and runs the full build pipeline:

1. Download ``ontario-latest.osm.pbf`` into ``data/osm/`` (skipped if present).
2. Write ``valhalla_tiles/valhalla.json`` with truck-tuned Meili defaults
   via ``python -m valhalla.valhalla_build_config``.
3. Build ``valhalla_tiles/admin.sqlite`` via ``valhalla_build_admins``.
4. Build the tile hierarchy via ``valhalla_build_tiles`` and pack it into
   ``valhalla_tiles/valhalla_tiles.tar``.

All outputs live under ``valhalla_tiles/`` (gitignored).  trucktrack's
``trucktrack.valhalla.find_config()`` then discovers the json automatically.

Usage::

    uv run python scripts/setup_valhalla.py                      # Ontario default
    uv run python scripts/setup_valhalla.py --pbf my-region.osm.pbf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import valhalla

GEOFABRIK_URL = (
    "https://download.geofabrik.de/north-america/canada/ontario-latest.osm.pbf"
)
DEFAULT_PBF_PATH = Path("data/osm/ontario-latest.osm.pbf")

# Parameters that client requests are allowed to override at call time.
# valhalla_build_config has no CLI flag for meili.customizable, so we patch
# the generated JSON in build_config().
MEILI_CUSTOMIZABLE: list[str] = [
    "search_radius",
    "gps_accuracy",
    "breakage_distance",
    "turn_penalty_factor",
    "interpolation_distance",
    "max_route_distance_factor",
    "max_route_time_factor",
    "beta",
]

CONFIG_OPTIONS: dict[str, bool | int | float] = {
    "mjolnir-max-cache-size": 1_000_000_000,
    "mjolnir-id-table-size": 13_000_000_000,
    "mjolnir-use-lru-mem-cache": True,
    "mjolnir-lru-mem-cache-hard-control": True,
    "mjolnir-use-simple-mem-cache": False,
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


def _valhalla_bin(name: str) -> Path:
    """Locate a pyvalhalla-bundled binary (valhalla_build_tiles, ...)."""
    path = Path(valhalla.__file__).parent / "bin" / name
    if not path.is_file():
        raise SystemExit(
            f"{name} not found at {path}. Install pyvalhalla: uv sync --extra valhalla"
        )
    return path


def _download_pbf(dest: Path) -> None:
    """Download the Ontario OSM extract from Geofabrik with a progress bar."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GEOFABRIK_URL} -> {dest}", file=sys.stderr)
    with requests.get(GEOFABRIK_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with (
            dest.open("wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))


def build_config(
    tile_dir: Path, tar_path: Path, admin_path: Path, config_path: Path
) -> None:
    """Generate a Valhalla config file with truck-optimised settings.

    ``valhalla_build_config`` prints the JSON config to stdout; we capture
    it into *config_path*.
    """
    cmd = [
        sys.executable,
        "-m",
        "valhalla.valhalla_build_config",
        "--mjolnir-tile-dir",
        str(tile_dir),
        "--mjolnir-tile-extract",
        str(tar_path),
        "--mjolnir-admin",
        str(admin_path),
    ]
    for key, value in CONFIG_OPTIONS.items():
        cmd.extend([f"--{key}", str(value)])
    print(f"Generating config: {config_path}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    # valhalla_build_config has no flag for meili.customizable, so patch it
    # after generation.  Without this, client-side overrides for
    # breakage_distance, turn_penalty_factor, etc. are silently ignored.
    config = json.loads(result.stdout)
    config.setdefault("meili", {})["customizable"] = MEILI_CUSTOMIZABLE
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pbf",
        type=Path,
        default=None,
        help=(
            "Path to an OSM PBF extract.  If omitted, "
            f"{DEFAULT_PBF_PATH} is downloaded from Geofabrik."
        ),
    )
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

    pbf_path: Path = args.pbf if args.pbf is not None else DEFAULT_PBF_PATH
    if not pbf_path.exists():
        if args.pbf is not None:
            parser.error(f"PBF file not found: {pbf_path}")
        _download_pbf(pbf_path)

    tile_dir: Path = args.tile_dir.resolve()
    tile_dir.mkdir(parents=True, exist_ok=True)

    tar_path = (tile_dir / "valhalla_tiles.tar").resolve()
    admin_path = (tile_dir / "admin.sqlite").resolve()
    config_path = (args.config_out or tile_dir / "valhalla.json").resolve()
    build_config(tile_dir, tar_path, admin_path, config_path)

    _run(
        [
            str(_valhalla_bin("valhalla_build_admins")),
            "-c",
            str(config_path),
            str(pbf_path.resolve()),
        ],
        "Building admins",
    )

    _run(
        [
            str(_valhalla_bin("valhalla_build_tiles")),
            "-c",
            str(config_path),
            str(pbf_path.resolve()),
        ],
        "Building tiles",
    )

    tile_subdirs = sorted(
        p for p in tile_dir.iterdir() if p.is_dir() and p.name.isdigit()
    )
    print(
        f"Packing {len(tile_subdirs)} tile directories into {tar_path}",
        file=sys.stderr,
    )
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
