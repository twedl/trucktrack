"""Command-line interface for trucktrack."""

from __future__ import annotations

import argparse
import io
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

import trucktrack
from trucktrack.generate import TripConfig, generate_trace, traces_to_parquet
from trucktrack.partition import partition_existing_parquet

# ── Output helpers ───────────────────────────────────────────────────────


def _write_output(
    result: pl.DataFrame,
    output: str,
    fmt: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    to_stdout = output == "-"
    fmt = fmt or ("csv" if to_stdout else "parquet")

    if to_stdout and fmt == "parquet" and sys.stdout.isatty():
        parser.error(
            "Refusing to write binary parquet to a terminal. "
            "Redirect stdout or use --format csv."
        )

    if to_stdout:
        if fmt == "csv":
            sys.stdout.write(result.write_csv())
        else:
            buf = io.BytesIO()
            result.write_parquet(buf)
            sys.stdout.buffer.write(buf.getvalue())
    else:
        out = Path(output)
        if fmt == "parquet":
            result.write_parquet(out)
        else:
            out.write_text(result.write_csv(), encoding="utf-8")
        print(f"Wrote {len(result)} rows to {out}", file=sys.stderr)


# ── Subcommand handlers ─────────────────────────────────────────────────


def _cmd_process(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    to_stdout = args.output == "-"
    fmt = args.format or ("csv" if to_stdout else "parquet")

    if not to_stdout and fmt == "parquet":
        n = trucktrack.process_parquet_in_rust(args.input, args.output)
        print(f"Wrote {n} rows to {args.output}", file=sys.stderr)
    else:
        result = trucktrack.process_dataframe_in_rust(
            trucktrack.read_parquet(args.input)
        )
        _write_output(result, args.output, args.format, parser)
    return 0


def _cmd_split_gap(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    to_stdout = args.output == "-"
    fmt = args.format or ("csv" if to_stdout else "parquet")
    gap = timedelta(seconds=args.gap)

    if not to_stdout and fmt == "parquet":
        n = trucktrack.split_by_observation_gap_file(
            args.input, args.output, gap, min_length=args.min_length
        )
        print(f"Wrote {n} rows to {args.output}", file=sys.stderr)
    else:
        df = trucktrack.read_parquet(args.input)
        result = trucktrack.split_by_observation_gap(
            df, gap, min_length=args.min_length
        )
        _write_output(result, args.output, args.format, parser)
    return 0


def _cmd_split_stops(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    to_stdout = args.output == "-"
    fmt = args.format or ("csv" if to_stdout else "parquet")
    duration = timedelta(seconds=args.duration)

    if not to_stdout and fmt == "parquet":
        n = trucktrack.split_by_stops_file(
            args.input,
            args.output,
            max_diameter=args.diameter,
            min_duration=duration,
            min_length=args.min_length,
        )
        print(f"Wrote {n} rows to {args.output}", file=sys.stderr)
    else:
        df = trucktrack.read_parquet(args.input)
        result = trucktrack.split_by_stops(
            df,
            max_diameter=args.diameter,
            min_duration=duration,
            min_length=args.min_length,
        )
        _write_output(result, args.output, args.format, parser)
    return 0


def _parse_latlon(value: str) -> tuple[float, float]:
    try:
        lat_s, lon_s = value.split(",", 1)
        return float(lat_s), float(lon_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected LAT,LON, got {value!r}") from exc


def _cmd_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    departure = (
        datetime.fromisoformat(args.departure) if args.departure else datetime.now(UTC)
    )
    config = TripConfig(
        origin=args.origin,
        destination=args.destination,
        departure_time=departure,
        gps_noise_meters=args.noise,
        seed=args.seed,
        valhalla_url=args.valhalla_url,
        tile_extract=args.tile_extract,
    )
    points = generate_trace(config)
    if not points:
        parser.error("generate produced zero points")

    out = Path(args.output)
    traces_to_parquet([(points, config.trip_id)], str(out))
    print(
        f"Wrote {len(points)} points to {out} (trip_id={config.trip_id})",
        file=sys.stderr,
    )
    return 0


def _cmd_partition(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    summary = partition_existing_parquet(args.input, args.output_dir)
    if not summary:
        print("No partitions written (input was empty?)", file=sys.stderr)
        return 0
    parts = ", ".join(f"{tier}={n}" for tier, n in sorted(summary.items()))
    print(f"Wrote partitions: {parts}", file=sys.stderr)
    return 0


# ── Parser ───────────────────────────────────────────────────────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="Input parquet file.")
    p.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="PATH",
        help="Output file path, or '-' for stdout (default: -).",
    )
    p.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Output format. Defaults to csv (stdout) or parquet (file).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trucktrack",
        description="Process and split trajectory data using the Rust backend.",
    )
    sub = parser.add_subparsers(dest="command")

    # process (default)
    p_proc = sub.add_parser("process", help="Add derived columns (speed_mps).")
    _add_common_args(p_proc)

    # split-gap
    p_gap = sub.add_parser("split-gap", help="Split at observation gaps.")
    _add_common_args(p_gap)
    p_gap.add_argument(
        "--gap",
        type=float,
        required=True,
        metavar="SECS",
        help="Gap threshold in seconds.",
    )
    p_gap.add_argument(
        "--min-length", type=int, default=0, help="Min rows per segment."
    )

    # split-stops
    p_stop = sub.add_parser("split-stops", help="Split at detected stops.")
    _add_common_args(p_stop)
    p_stop.add_argument(
        "--diameter",
        type=float,
        required=True,
        metavar="METERS",
        help="Max spatial diameter for a stop (meters).",
    )
    p_stop.add_argument(
        "--duration",
        type=float,
        required=True,
        metavar="SECS",
        help="Min stop duration in seconds.",
    )
    p_stop.add_argument(
        "--min-length", type=int, default=0, help="Min rows per segment."
    )

    # generate
    p_gen = sub.add_parser(
        "generate", help="Synthesize a truck GPS trace and write to parquet."
    )
    p_gen.add_argument(
        "--origin",
        type=_parse_latlon,
        required=True,
        metavar="LAT,LON",
        help="Origin coordinates as LAT,LON.",
    )
    p_gen.add_argument(
        "--destination",
        type=_parse_latlon,
        required=True,
        metavar="LAT,LON",
        help="Destination coordinates as LAT,LON.",
    )
    p_gen.add_argument(
        "-o", "--output", type=Path, required=True, help="Output parquet path."
    )
    p_gen.add_argument(
        "--departure",
        default=None,
        help="ISO-8601 departure time (default: now, UTC).",
    )
    p_gen.add_argument(
        "--noise", type=float, default=3.0, help="GPS noise stddev in meters."
    )
    p_gen.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p_gen.add_argument(
        "--valhalla-url",
        default="http://localhost:8002",
        help="Valhalla base URL (falls back to straight-line if unreachable).",
    )
    p_gen.add_argument(
        "--tile-extract",
        default=None,
        metavar="PATH",
        help="Path to Valhalla tile extract (.tar). Uses local pyvalhalla.",
    )

    # partition
    p_part = sub.add_parser(
        "partition",
        help="Rewrite a flat parquet as a Valhalla-tile-aligned hive dataset.",
    )
    p_part.add_argument("input", type=Path, help="Input parquet file.")
    p_part.add_argument("output_dir", type=Path, help="Output directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to "process" when no subcommand given
    if args.command is None:
        # Re-parse as "process" subcommand
        sub_argv = ["process"] + (argv or sys.argv[1:])
        parser.parse_args(sub_argv, args)
        args.command = "process"

    handlers = {
        "process": _cmd_process,
        "split-gap": _cmd_split_gap,
        "split-stops": _cmd_split_stops,
        "generate": _cmd_generate,
        "partition": _cmd_partition,
    }
    return handlers[args.command](args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
