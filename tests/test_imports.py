"""trucktrack imports without its optional extras."""

from __future__ import annotations

import subprocess
import sys


def test_imports_without_pyvalhalla() -> None:
    # None in sys.modules makes `import valhalla` raise ImportError.
    code = (
        "import sys; sys.modules['valhalla'] = None\n"
        "import trucktrack, trucktrack.congestion, trucktrack.valhalla, "
        "trucktrack.query"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
