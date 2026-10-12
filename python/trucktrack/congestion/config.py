"""Thresholds for turning matched fixes into edge speed observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    """Thresholds for turning matched fixes into edge speed observations.

    *max_half_gap_s* caps each side of a fix's time weight (see
    :func:`~trucktrack.congestion.time_weights`).  *stationary_kmh*,
    *max_stationary_step_m* and *max_queue_stop_s* separate dwell zeros
    from queue zeros (see :func:`~trucktrack.congestion.flag_dwell`).

    A fix is ``wrong_way`` when its heading is more than
    *wrong_way_deg* off the matched edge's bearing and it is moving at
    *heading_min_speed_kmh* or more.  Below that, GPS course is noise.
    The edge's bearing at the fix is interpolated between its begin and
    end headings.  That only holds on edges that turn by at most
    *max_edge_turn_deg* (a cloverleaf loop doesn't), so other edges get
    no heading check.  Parallel roadways in the same direction (a
    collector beside an express lane) look alike and pass.
    """

    max_half_gap_s: float = 30.0
    stationary_kmh: float = 2.0
    max_stationary_step_m: float = 30.0
    max_queue_stop_s: float = 180.0
    heading_min_speed_kmh: float = 15.0
    wrong_way_deg: float = 90.0
    max_edge_turn_deg: float = 90.0


DEFAULT_CONFIG = ObservationConfig()
