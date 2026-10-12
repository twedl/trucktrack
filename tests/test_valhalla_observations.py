"""Edge speed observations from the map-matching pipeline.

The tile-build ID and argument checks run anywhere; the rest needs a
discoverable ``valhalla.json`` and runs on the committed ``data/trucks``
sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from trucktrack.congestion import (
    OBSERVATION_ATTRIBUTES,
    OBSERVATION_COLUMNS,
    ObservationConfig,
)
from trucktrack.valhalla import tile_build_id, write_tile_build_id
from trucktrack.valhalla._bridge import (
    BridgeConfig,
    map_match_dataframe_with_bridges,
)

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


def test_bridges_reject_attributes_with_shapes() -> None:
    df = pl.DataFrame({"time": [], "lat": [], "lon": []})
    with pytest.raises(ValueError, match="collect_shapes"):
        map_match_dataframe_with_bridges(
            df, attributes=OBSERVATION_ATTRIBUTES, collect_shapes=True
        )


@pytest.fixture(scope="module")
def partitioned(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from trucktrack.pipeline import run_pipeline

    out = tmp_path_factory.mktemp("partitioned")
    run_pipeline(_SAMPLE, out)
    return out


def _moving_chunks(input_dir: Path) -> list[Path]:
    """Chunks with any non-stop row; all-stop chunks produce no output."""
    return sorted(
        p.relative_to(input_dir)
        for p in input_dir.rglob("*.parquet")
        if not pl.read_parquet(p)["is_stop"].all()
    )


def _run_partitions(
    input_dir: Path,
    output_dir: Path,
    observations: ObservationConfig | None,
) -> int:
    """Map-match every partition in-process; return chunks skipped."""
    from trucktrack.pipeline import parquet_row_count
    from trucktrack.valhalla.pipeline import _process_partition

    skipped = 0
    for pdir in sorted({p.parent for p in input_dir.rglob("*.parquet")}):
        counts = {p: parquet_row_count(p) for p in pdir.glob("*.parquet")}
        _, _, s, _, _, _ = _process_partition(
            pdir,
            input_dir,
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
        self, partitioned: Path, tmp_path: Path
    ) -> None:
        assert _run_partitions(partitioned, tmp_path, ObservationConfig()) == 0
        chunks = _moving_chunks(partitioned)
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
            pl.scan_parquet(partitioned / "**" / "*.parquet")
            .filter(~pl.col("is_stop"))
            .select(pl.len())
            .collect()
            .item()
        )
        assert 0.8 * moving < len(obs) <= moving

    def test_skips_only_when_every_output_exists(
        self, partitioned: Path, tmp_path: Path
    ) -> None:
        n_chunks = len(_moving_chunks(partitioned))
        assert _run_partitions(partitioned, tmp_path, None) == 0
        assert not (tmp_path / "_observations").exists()

        # Ways exist, observations don't: every chunk is redone.
        assert _run_partitions(partitioned, tmp_path, ObservationConfig()) == 0
        assert _run_partitions(partitioned, tmp_path, ObservationConfig()) == n_chunks

        lost = next((tmp_path / "_observations").rglob("*.parquet"))
        lost.unlink()
        assert (
            _run_partitions(partitioned, tmp_path, ObservationConfig()) == n_chunks - 1
        )
        assert lost.exists()

    def test_bridged_responses_cover_each_segment(self, partitioned: Path) -> None:
        trips = (
            pl.scan_parquet(partitioned / "**" / "*.parquet")
            .filter(~pl.col("is_stop"))
            .collect()
        )
        trip_id = trips.group_by("id").len().sort("len", descending=True)["id"][0]
        trip = trips.filter(pl.col("id") == trip_id).sort("time")
        # Cut out the middle third: a gap far beyond max_dist_m.
        n = len(trip)
        gappy = pl.concat([trip.head(n // 3), trip.tail(n // 3)])

        result = map_match_dataframe_with_bridges(
            gappy,
            bridges=BridgeConfig(max_dist_m=5000),
            attributes=OBSERVATION_ATTRIBUTES,
        )
        if result.fallback_used:
            assert [off for off, _ in result.responses] == [0]
        else:
            assert [off for off, _ in result.responses] == [0, n // 3]
        for offset, resp in result.responses:
            n_points = len(resp["matched_points"])
            assert offset + n_points <= len(gappy)
            assert all("id" in e for e in resp["edges"])
