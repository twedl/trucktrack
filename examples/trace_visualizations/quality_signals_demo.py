"""Trigger the two new MapMatchQuality signals.

``path_length_ratio``
    Fires when the matcher strings together a long detour that is much
    farther than the straight-line distance the input points imply —
    the classic "GPS on the wrong side of a one-way pair" or "fix drifts
    across a barrier with no nearby crossing" pattern.

``heading_reversals``
    Fires when the matched polyline reverses direction many times, which
    usually means jitter alternated the input between nearby streets and
    the matcher threaded it with U-turns.  The tuned default
    ``turn_penalty_factor=500`` suppresses this, so the zigzag demo
    explicitly overrides it to 0 to show the signal firing; the
    "defaults" variant shows the same trace coming back clean with the
    tuned options.

Usage::

    uv run python examples/trace_visualizations/quality_signals_demo.py

Requires a ``valhalla.json`` in cwd.
"""

from __future__ import annotations

from trucktrack.valhalla.quality import MapMatchQuality, evaluate_map_match


def _report(label: str, q: MapMatchQuality) -> None:
    ratio = f"{q.path_length_ratio:.2f}" if q.path_length_ratio is not None else "None"
    print(f"\n[{label}]")
    print(f"  ok                = {q.ok}")
    print(f"  error             = {q.error}")
    print(f"  n_points          = {q.n_points}")
    print(f"  n_polylines       = {q.n_polylines}")
    print(f"  path_length_ratio = {ratio}")
    print(f"  heading_reversals = {q.heading_reversals}")
    print(f"  has_issues        = {q.has_issues}")


def scenario_zigzag_parallel_streets(
    n_points: int = 30,
    trace_options: dict[str, object] | None = None,
) -> MapMatchQuality:
    # Alternate fixes between King St W and Wellington St W (one block
    # apart) every sample.  With turn_penalty_factor low, the matcher
    # will happily weave between the two via cross streets, producing
    # a heading reversal at each hop.
    king = (43.6467, -79.3836)
    wellington = (43.6454, -79.3830)
    pts = []
    for i in range(n_points):
        base = king if i % 2 == 0 else wellington
        pts.append((base[0] + i * 0.00002, base[1] + i * 0.00010))
    return evaluate_map_match(
        "zigzag_parallel_streets",
        pts,
        trace_options=trace_options,
    )


def main() -> None:
    # Both signals fire: ratio well above 1.5 and >5 heading reversals.
    _report(
        "zigzag_turn_penalty_0",
        scenario_zigzag_parallel_streets(trace_options={"turn_penalty_factor": 0}),
    )
    # Same input, tuned defaults: path_length_ratio still fires (the
    # input alternation guarantees extra path length no matter what),
    # but turn_penalty_factor=500 suppresses the spurious U-turns so
    # heading_reversals drops below its threshold.
    _report(
        "zigzag_tuned_defaults",
        scenario_zigzag_parallel_streets(),
    )


if __name__ == "__main__":
    main()
