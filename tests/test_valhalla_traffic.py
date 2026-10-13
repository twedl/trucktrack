"""Tests for trucktrack.valhalla.traffic: predicted-traffic CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("valhalla")

from trucktrack.valhalla import write_tile_build_id  # noqa: E402
from trucktrack.valhalla._build_id import (  # noqa: E402
    base_extract_path,
    pristine_extract,
)
from trucktrack.valhalla.traffic import (  # noqa: E402
    encode_week,
    speed_buckets,
    traffic_rows,
    write_traffic_csvs,
)
from valhalla.utils import (  # noqa: E402
    GraphId,
    decode_compressed_speeds,
    decompress_speed_bucket,
)

# An edge in level 2 tile 770804; its GraphId value as observations carry it.
_EDGE = GraphId(770804, 2, 31604).value


def _week(hourly: list[float]) -> list[float]:
    coefficients = decode_compressed_speeds(encode_week(speed_buckets(hourly)))
    return [decompress_speed_bucket(coefficients, b) for b in range(2016)]


class TestSpeedBuckets:
    def test_hour_middles_keep_their_speed(self) -> None:
        hourly = np.arange(168, dtype=float)
        buckets = speed_buckets(hourly)
        assert buckets.shape == (2016,)
        assert buckets.dtype == np.float32
        # Bucket 6 is centred on 00:32:30, 2.5 min past hour 0's middle.
        assert buckets[6] == pytest.approx(2.5 / 60)
        assert buckets[12 * 100 + 6] == pytest.approx(100 + 2.5 / 60)

    def test_wraps_around_the_week(self) -> None:
        hourly = np.full(168, 50.0)
        hourly[0] = 20.0  # Sunday 00:00-01:00
        buckets = speed_buckets(hourly)
        # Saturday's last bucket, 23:57:30, is 27.5 of the 60 min from
        # Saturday 23:30 (50) to Sunday 00:30 (20).
        assert buckets[-1] == pytest.approx(50 - 30 * 27.5 / 60)

    def test_needs_a_whole_week(self) -> None:
        with pytest.raises(ValueError, match="168"):
            speed_buckets([50.0] * 24)

    def test_rows_of_weeks(self) -> None:
        weeks = np.vstack([np.full(168, 50.0), np.arange(168.0)])
        out = speed_buckets(weeks)
        assert out.shape == (2, 2016)
        assert out[1] == pytest.approx(speed_buckets(np.arange(168.0)))


class TestEncodeWeek:
    def test_round_trips_through_valhalla(self) -> None:
        assert _week([50.0] * 168) == pytest.approx([50.0] * 2016, abs=0.5)

    def test_keeps_a_rush_hour(self) -> None:
        hourly = [100.0] * 168
        tuesday_8 = 2 * 24 + 8
        hourly[tuesday_8] = 50.0
        speeds = _week(hourly)
        middle = tuesday_8 * 12 + 6
        assert speeds[middle] < 70.0
        assert speeds[middle - 36] == pytest.approx(100.0, abs=3.0)


def _historical(speed: float = 80.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tile_build_id": ["b"] * 168,
            "edge_id": [_EDGE] * 168,
            "hour_of_week": list(range(168)),
            "speed": [speed] * 168,
        },
        schema_overrides={"edge_id": pl.UInt64},
    )


def _flows(free: float = 80.4, constrained: float = 61.6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "edge_id": [_EDGE],
            "freeflow_speed": [free],
            "constrained_speed": [constrained],
        },
        schema_overrides={"edge_id": pl.UInt64},
    )


class TestTrafficRows:
    def test_line_per_edge_in_its_tiles_csv(self) -> None:
        rows = traffic_rows(_historical(), _flows())
        assert rows["path"].to_list() == ["2/000/770/804.csv"]
        graph_id, free, constrained, week = rows["line"].item().split(",")
        assert (graph_id, free, constrained) == ("2/770804/31604", "80", "62")
        assert decompress_speed_bucket(decode_compressed_speeds(week), 0) == (
            pytest.approx(80.0, abs=0.5)
        )

    def test_speeds_clipped_to_what_valhalla_takes(self) -> None:
        rows = traffic_rows(_historical(speed=300.0), _flows(free=0.2, constrained=300))
        _, free, constrained, _ = rows["line"].item().split(",")
        assert (free, constrained) == ("1", "140")

    def test_flows_default_to_the_weeks(self) -> None:
        _, free, constrained, _ = traffic_rows(_historical())["line"].item().split(",")
        assert (free, constrained) == ("80", "80")

    def test_every_edge_needs_a_whole_week(self) -> None:
        with pytest.raises(ValueError, match="168 hours"):
            traffic_rows(_historical().head(167), _flows())

    def test_written_one_csv_per_tile(self, tmp_path: Path) -> None:
        rows = traffic_rows(_historical(), _flows())
        assert write_traffic_csvs(rows, tmp_path) == 1
        text = (tmp_path / "2/000/770/804.csv").read_text()
        assert text.startswith("2/770804/31604,80,62,")


class TestPristineExtract:
    def _config(self, tmp_path: Path) -> tuple[Path, Path]:
        extract = tmp_path / "valhalla_tiles.tar"
        extract.write_bytes(b"tiles")
        config = tmp_path / "valhalla.json"
        config.write_text(json.dumps({"mjolnir": {"tile_extract": str(extract)}}))
        write_tile_build_id(extract)
        return config, extract

    def test_copied_once_while_the_extract_is_as_built(self, tmp_path: Path) -> None:
        config, extract = self._config(tmp_path)
        base = pristine_extract(config)
        assert base == base_extract_path(extract)
        assert base.name == "valhalla_tiles.base.tar"
        assert base.read_bytes() == b"tiles"
        extract.write_bytes(b"tiles with traffic")
        assert pristine_extract(config).read_bytes() == b"tiles"

    def test_refuses_a_changed_extract_without_a_copy(self, tmp_path: Path) -> None:
        config, extract = self._config(tmp_path)
        extract.write_bytes(b"tiles with traffic")
        with pytest.raises(RuntimeError, match="setup_valhalla"):
            pristine_extract(config)

    def test_new_build_drops_the_old_copy(self, tmp_path: Path) -> None:
        config, extract = self._config(tmp_path)
        pristine_extract(config)
        extract.write_bytes(b"rebuilt tiles")
        write_tile_build_id(extract)
        assert pristine_extract(config).read_bytes() == b"rebuilt tiles"
