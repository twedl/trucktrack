"""Edge speed observations from the map-matching pipeline.

The tile-build ID and argument checks run anywhere; the rest needs a
discoverable ``valhalla.json`` and runs on the committed ``data/trucks``
sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from trucktrack.congestion import OBSERVATION_COLUMNS, ObservationConfig
from trucktrack.valhalla import parse_fix_matches, tile_build_id, write_tile_build_id
from trucktrack.valhalla._bridge import (
    BridgeConfig,
    map_match_dataframe_with_bridges,
)
from trucktrack.valhalla.map_matching import FIX_MATCH_SCHEMA

_SAMPLE = Path(__file__).parent.parent / "data" / "trucks"


def _valhalla_works() -> bool:
    try:
        from trucktrack.valhalla import get_actor

        get_actor()
    except Exception:
        return False
    return True


requires_tiles = pytest.mark.skipif(
    not _valhalla_works(), reason="no working valhalla.json discoverable"
)


class TestTileBuildId:
    @pytest.fixture
    def config(self, tmp_path: Path) -> Path:
        (tmp_path / "tiles.tar").write_bytes(b"graph tiles v1")
        config = tmp_path / "valhalla.json"
        config.write_text(
            json.dumps({"mjolnir": {"tile_extract": str(tmp_path / "tiles.tar")}})
        )
        return config

    def test_written_beside_the_extract(self, config: Path) -> None:
        build_id = tile_build_id(config)
        assert len(build_id) == 16
        assert (config.parent / "tile_build_id").read_text().strip() == build_id

    def test_survives_rewriting_the_extract(self, config: Path) -> None:
        # Adding predicted traffic rewrites the tar but keeps GraphIds.
        before = tile_build_id(config)
        (config.parent / "tiles.tar").write_bytes(b"graph tiles v1 + traffic")
        assert tile_build_id(config) == before

    def test_rebuild_writes_a_new_id(self, config: Path) -> None:
        before = tile_build_id(config)
        (config.parent / "tiles.tar").write_bytes(b"graph tiles v2")
        assert write_tile_build_id(config.parent / "tiles.tar") != before
        assert tile_build_id(config) != before


# Trimmed from a real trace_attributes response on the local Ontario
# tiles (synthetic trip through Brampton), edges re-indexed.  The
# unmatched point and the sentinel edge_index are the two ways Valhalla
# reports a point with no edge.
_RESPONSE: dict[str, Any] = {
    "units": "kilometers",
    "matched_points": [
        {
            "type": "matched",
            "edge_index": 0,
            "distance_along_edge": 0.892249,
            "distance_from_trace_point": 1.612005,
        },
        {
            "type": "matched",
            "edge_index": 1,
            "distance_along_edge": 1.0,
            "distance_from_trace_point": 2.30917,
        },
        {"type": "unmatched"},
        {
            "type": "interpolated",
            "edge_index": 18446744073709551615,
            "distance_along_edge": 0.0,
            "distance_from_trace_point": 0.0,
        },
        {
            "type": "matched",
            "edge_index": 2,
            "distance_along_edge": 0.529403,
            "distance_from_trace_point": 32.865639,
        },
    ],
    "edges": [
        {
            "way_id": 370115604,
            "id": 1634000558945,
            "begin_heading": 119,
            "end_heading": 119,
            "forward": True,
            "road_class": "motorway",
            "length": 0.688,
        },
        {
            "way_id": 808977465,
            "id": 1638664624993,
            "begin_heading": 138,
            "end_heading": 138,
        },
        {
            "way_id": 920928491,
            "id": 1342686762890,
            "begin_heading": 334,
            "end_heading": 279,
        },
    ],
}


class TestParseFixMatches:
    def test_one_value_per_point(self) -> None:
        fm = parse_fix_matches(_RESPONSE, 5)
        assert list(fm) == list(FIX_MATCH_SCHEMA)
        # The sentinel point follows an unmatched one, so it stays edgeless.
        assert fm["edge_id"] == [
            1634000558945,
            1638664624993,
            None,
            None,
            1342686762890,
        ]
        assert fm["way_id"] == [370115604, 808977465, None, None, 920928491]
        assert fm["match_type"][2:4] == ["unmatched", "interpolated"]
        assert fm["begin_heading"][4] == 334
        assert fm["forward"][0] is True
        assert fm["road_class"][0] == "motorway"
        assert fm["edge_length"][0] == pytest.approx(688.0)
        assert fm["edge_length"][4] is None
        df = pl.DataFrame(fm, schema=FIX_MATCH_SCHEMA)
        assert df.height == 5

    def test_edgeless_interpolated_point_put_back_on_its_edge(self) -> None:
        # A truck standing in a queue: Valhalla places the interpolated
        # points further along edge 0 but drops their edge_index.
        response = {
            "matched_points": [
                {"type": "matched", "edge_index": 0, "distance_along_edge": 0.40},
                {"type": "interpolated", "distance_along_edge": 0.42},
                {
                    "type": "interpolated",
                    "edge_index": 18446744073709551615,
                    "distance_along_edge": 0.44,
                },
                # Behind the last point on edge 0: can't be edge 0.
                {"type": "interpolated", "distance_along_edge": 0.10},
                {"type": "matched", "edge_index": 1, "distance_along_edge": 0.07},
            ],
            "edges": _RESPONSE["edges"][:2],
        }
        e0, e1 = 1634000558945, 1638664624993
        assert parse_fix_matches(response, 5)["edge_id"] == [e0, e0, e0, None, e1]

    def test_point_count_must_match(self) -> None:
        with pytest.raises(ValueError, match="5 matched points for 4 points"):
            parse_fix_matches(_RESPONSE, 4)

    def test_empty_response(self) -> None:
        df = pl.DataFrame(parse_fix_matches({}, 0), schema=FIX_MATCH_SCHEMA)
        assert df.is_empty()


def test_bridges_reject_per_fix_with_shapes() -> None:
    df = pl.DataFrame({"time": [], "lat": [], "lon": []})
    with pytest.raises(ValueError, match="collect_shapes"):
        map_match_dataframe_with_bridges(df, per_fix=True, collect_shapes=True)


@pytest.fixture(scope="module")
def split(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from trucktrack.buckets import bucket_by_truck
    from trucktrack.pipeline import run_pipeline

    bucketed = tmp_path_factory.mktemp("bucketed")
    bucket_by_truck(_SAMPLE, bucketed, n_buckets=4)
    out = tmp_path_factory.mktemp("split")
    run_pipeline(bucketed, out)
    return out


def _moving_chunks(input_dir: Path) -> list[Path]:
    """Chunks with any non-stop row; all-stop chunks produce no output."""
    return sorted(
        p.relative_to(input_dir)
        for p in input_dir.rglob("*.parquet")
        if not pl.read_parquet(p)["is_stop"].all()
    )


def _run_buckets(
    input_dir: Path,
    output_dir: Path,
    observations: ObservationConfig | None,
) -> int:
    """Map-match every bucket in-process; return chunks skipped."""
    from trucktrack.buckets import bucket_dirs, bucket_files
    from trucktrack.pipeline import parquet_row_count
    from trucktrack.valhalla.pipeline import _process_bucket

    skipped = 0
    for bdir in bucket_dirs(input_dir):
        counts = {p: parquet_row_count(p) for p in bucket_files(bdir)}
        _, s, _, _, _ = _process_bucket(
            bdir,
            output_dir,
            None,
            counts,
            observations=observations,
            build_id="test-build",
        )
        skipped += s
    return skipped


@requires_tiles
class TestPipelineObservations:
    def test_observations_written_beside_ways(
        self, split: Path, tmp_path: Path
    ) -> None:
        assert _run_buckets(split, tmp_path, ObservationConfig()) == 0
        chunks = _moving_chunks(split)
        obs_files = sorted(
            p.relative_to(tmp_path / "_observations")
            for p in (tmp_path / "_observations").rglob("*.parquet")
        )
        assert obs_files == chunks

        obs = pl.read_parquet(tmp_path / "_observations" / "**" / "*.parquet")
        assert obs.columns == [*OBSERVATION_COLUMNS, "tile_build_id"]
        assert obs["tile_build_id"].unique().to_list() == ["test-build"]
        assert (obs["w"] > 0).all()
        assert obs["edge_id"].null_count() == 0

        moving = (
            pl.scan_parquet(split / "**" / "*.parquet")
            .filter(~pl.col("is_stop"))
            .select(pl.len())
            .collect()
            .item()
        )
        assert 0.8 * moving < len(obs) <= moving

    def test_skips_only_when_every_output_exists(
        self, split: Path, tmp_path: Path
    ) -> None:
        n_chunks = len(_moving_chunks(split))
        assert _run_buckets(split, tmp_path, None) == 0
        assert not (tmp_path / "_observations").exists()

        # Ways exist, observations don't: every chunk is redone.
        assert _run_buckets(split, tmp_path, ObservationConfig()) == 0
        assert _run_buckets(split, tmp_path, ObservationConfig()) == n_chunks

        lost = next((tmp_path / "_observations").rglob("*.parquet"))
        lost.unlink()
        assert _run_buckets(split, tmp_path, ObservationConfig()) == n_chunks - 1
        assert lost.exists()

    def test_bridged_fix_matches_line_up_with_rows(self, split: Path) -> None:
        trips = (
            pl.scan_parquet(split / "**" / "*.parquet")
            .filter(~pl.col("is_stop"))
            .collect()
        )
        trip_id = trips.group_by("id").len().sort("len", descending=True)["id"][0]
        trip = trips.filter(pl.col("id") == trip_id).sort("time")
        # Cut out the middle third: a gap far beyond max_dist_m.
        n = len(trip)
        gappy = pl.concat([trip.head(n // 3), trip.tail(n // 3)])

        result = map_match_dataframe_with_bridges(
            gappy, bridges=BridgeConfig(max_dist_m=5000), per_fix=True
        )
        assert not result.fallback_used
        assert len(result.fits) == 1
        fm = result.fix_matches
        assert fm is not None
        assert {len(values) for values in fm.values()} == {len(gappy)}
        on_edge = [e is not None for e in fm["edge_id"]]
        # Both segments matched: fixes on either side of the gap landed.
        assert any(on_edge[: n // 3]) and any(on_edge[n // 3 :])
