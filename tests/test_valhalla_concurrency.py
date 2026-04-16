"""Concurrency probe for ``pyvalhalla.Actor``.

Documents that a single shared ``valhalla.Actor`` is **not** safe to
call from multiple threads concurrently in pyvalhalla 3.6.x.  This is
why ``trucktrack.valhalla._actor.get_actor`` uses ``threading.local()``
to isolate Actors per worker thread.

The stress probe is launched in a **subprocess** so the native crash
(SIGTRAP on macOS in our measurement, 2026-04-16) doesn't terminate the
pytest session.  When a future pyvalhalla version gains concurrent
``trace_attributes`` safety, this test will fail (the subprocess will
exit 0 instead of crashing) — that's the signal to revisit the
per-thread cache in ``_actor.py`` and the related measurement plan at
``/Users/tweedle/.claude/plans/crystalline-floating-micali.md``.

Guarded by the same skip pattern as ``test_valhalla.py`` — skipped if
pyvalhalla isn't installed or no ``valhalla.json`` is discoverable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_pyvalhalla_available = False
try:
    import valhalla as _valhalla  # noqa: F401

    _pyvalhalla_available = True
except ImportError:
    pass

requires_pyvalhalla = pytest.mark.skipif(
    not _pyvalhalla_available,
    reason="pyvalhalla not installed",
)


def _valhalla_works() -> bool:
    if not _pyvalhalla_available:
        return False
    try:
        from trucktrack.valhalla import get_actor

        get_actor()
    except Exception:
        return False
    return True


_valhalla_ok = _valhalla_works()

requires_tiles = pytest.mark.skipif(
    not _valhalla_ok,
    reason="no working valhalla.json discoverable",
)


# Probe hammers a single Actor from 4 threads with 50 concurrent
# ``trace_attributes`` calls.  Observed to crash reliably on pyvalhalla
# 3.6.3; smaller reproductions may exist but this one is known-good.
_PROBE = textwrap.dedent(
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from trucktrack.valhalla import get_actor

    actor = get_actor()
    body = json.dumps({
        "shape": [{"lat": 43.65, "lon": -79.38}, {"lat": 43.66, "lon": -79.39}],
        "costing": "auto",
        "shape_match": "map_snap",
    })
    # Sanity: one sequential call must succeed before we measure concurrency.
    assert actor.trace_attributes(body)

    def _call(_):
        return actor.trace_attributes(body)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_call, i) for i in range(50)]
        for f in as_completed(futs):
            f.result(timeout=60)
    """
).strip()


@requires_pyvalhalla
@requires_tiles
def test_shared_actor_concurrent_access_crashes(tmp_path: Path) -> None:
    """Shared Actor + concurrent ``trace_attributes`` must crash today.

    This test passes by asserting the subprocess *fails* — a non-zero
    exit confirms pyvalhalla isn't thread-safe for shared-Actor use,
    which is the behavioral contract ``_actor.py::get_actor`` relies on.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    result = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        "Shared-Actor concurrent probe exited cleanly (returncode=0).\n"
        "pyvalhalla may have gained thread-safe trace_attributes.\n"
        "Revisit _actor.py per-thread caching and the Phase 3a plan at "
        "/Users/tweedle/.claude/plans/crystalline-floating-micali.md.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr[-800:]!r}"
    )
