"""Realistic trip segments that trigger Valhalla 444 via trace_attributes.

The 444 warning ("Map Match algorithm failed to find path: map_snap
algorithm failed to snap the shape points to the correct shape") comes
from src/thor/trace_attributes_action.cc when the inner Meili matcher
throws.  Empirically, the inner throw is reached only when points
correlate individually (Loki succeeds) but the road graph has no
drivable chain between them — either because two primaries live in
disconnected graph components, or because ConstructRoute / the Viterbi
path fails downstream of HasMinimumCandidates.

Meili's matcher is surprisingly robust to mild jitter, dwell drift, and
short multipath bursts — those cases just match to a plausible edge
sequence.  To get 444 on real truck-tracking data you generally need a
coordinate corruption that places the fix in a different connected
component of the road graph (device power cycle with bad GPS lock,
satellite ephemeris glitch, operator manually injecting a bad fix).

Each scenario below models a realistic mechanism; the harness reports
which actually reproduce the 444 wrapper given our local tile extract.

Usage::

    uv run python examples/trace_visualizations/quality_444_realistic.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

import random

from trucktrack.valhalla.quality import (
    MapMatchQuality,
    evaluate_map_match_attributes,
)


def _report(label: str, q: MapMatchQuality) -> None:
    hit = q.error and "failed to snap the shape" in q.error
    marker = "*** 444 ***" if hit else "    other  "
    print(f"{marker}  [{label}] n={q.n_points}  error={q.error!r}")


def scenario_power_cycle_wrong_fix() -> MapMatchQuality:
    # Driver pulls into a weigh station on Hwy 401 near Toronto.  The
    # device power-cycles and the first post-boot fixes lock onto a
    # spoofed / reflected satellite signal, reporting the truck in
    # Rochester NY across Lake Ontario for two samples before the real
    # fix returns.  Our Ontario tile extract has edges on both shores
    # but no drivable chain across the lake — Viterbi gives up.
    return evaluate_map_match_attributes(
        "power_cycle_wrong_fix",
        [
            (43.6532, -79.3832),  # Toronto, real
            (43.1566, -77.6088),  # Rochester, glitched
            (43.6532, -79.3832),  # Toronto, recovered
        ],
    )


def scenario_latlon_swap_burst() -> MapMatchQuality:
    # A known ELD firmware bug occasionally emits fixes with lat and
    # lon swapped for a few samples in a row.  Coordinates still fall
    # inside tile coverage when the absolute values happen to match
    # both a latitude and a longitude in the region (e.g. 43 / 79 -> a
    # real point ~79° north where no roads exist in our tiles, or more
    # commonly landing in an ocean — captured by Loki).  The realistic
    # survivable variant is a *partial* swap that puts the fix on a
    # distant but still road-mapped area.  We fake that with a jump to
    # Ottawa (a real tile-covered city) long enough that the Viterbi
    # transition distance factor blows up.
    return evaluate_map_match_attributes(
        "latlon_swap_burst",
        [
            (43.6532, -79.3832),  # Toronto
            (43.6540, -79.3825),
            (45.4236, -75.7009),  # glitch: fix jumps to Ottawa
            (45.4245, -75.7002),
            (43.6550, -79.3815),  # back to Toronto
        ],
    )


def scenario_multipath_urban_canyon() -> MapMatchQuality:
    # Urban canyon multipath: GPS alternately locks onto a reflection
    # placing the truck on a crossing street separated by rail tracks
    # (no at-grade connection).  Included for comparison — Meili
    # usually handles this as jitter, but tighter search_radius pushes
    # it toward the failure mode.
    real = (43.6532, -79.3832)      # King St W
    bounce = (43.6470, -79.3900)    # south of rail corridor
    pts = []
    for i in range(12):
        base = real if i % 2 == 0 else bounce
        pts.append((base[0] + i * 0.00005, base[1] + i * 0.00005))
    return evaluate_map_match_attributes(
        "multipath_urban_canyon",
        pts,
        trace_options={"search_radius": 10, "gps_accuracy": 5},
    )


def scenario_dwell_drift_crosslane() -> MapMatchQuality:
    # Long traffic-jam stop on Hwy 401.  Cold-start drift pulls the
    # fix across the median onto the opposing carriageway and back.
    # No legal U-turn exists, so the edge sequence alternates between
    # two one-way edges that don't connect.  Included for comparison.
    rng = random.Random(3)
    westbound = (43.8120, -79.3355)
    eastbound = (43.8135, -79.3352)
    pts: list[tuple[float, float]] = []
    for i in range(30):
        base = westbound if (i // 3) % 2 == 0 else eastbound
        pts.append(
            (base[0] + rng.gauss(0, 0.0005), base[1] + rng.gauss(0, 0.0006))
        )
    return evaluate_map_match_attributes("dwell_drift_crosslane", pts)


def main() -> None:
    for fn in [
        scenario_power_cycle_wrong_fix,
        scenario_latlon_swap_burst,
        scenario_multipath_urban_canyon,
        scenario_dwell_drift_crosslane,
    ]:
        try:
            _report(fn.__name__, fn())
        except Exception as exc:
            print(f"   ???       [{fn.__name__}]  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
