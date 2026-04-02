"""Tests for trucktrack._core (Rust extension) via the public trucktrack API."""

from __future__ import annotations

import trucktrack


def test_version_is_string() -> None:
    assert isinstance(trucktrack.__version__, str)
    assert len(trucktrack.__version__) > 0
