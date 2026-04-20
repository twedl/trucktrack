"""JSON marshaling wrappers that prefer ``orjson`` when available.

The stdlib ``json`` module holds the GIL for both serialization and
parsing.  At pipeline scale (thousands of Valhalla calls across
:data:`max_workers` threads), that turns JSON into a serialization
point even though the underlying Valhalla C++ work releases the GIL
cleanly.  ``orjson`` is a C-level JSON codec that releases the GIL
during the heavy work, restoring real parallelism.

The fallback to stdlib keeps the module importable without ``orjson``
installed — lint, tests, and single-threaded users don't pay, but
``pip install trucktrack[valhalla]`` pulls it in for pipeline use.
"""

from __future__ import annotations

from typing import Any

try:
    import orjson as _orjson

    def dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode("utf-8")

    def loads(s: str | bytes) -> Any:
        return _orjson.loads(s)

except ImportError:
    import json as _json

    def dumps(obj: Any) -> str:
        return _json.dumps(obj)

    def loads(s: str | bytes) -> Any:
        return _json.loads(s)
