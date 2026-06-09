from __future__ import annotations

import sys
from pathlib import Path

_SIMULSTREAMING_ROOT = Path(__file__).resolve().parent / "SimulStreaming"
if str(_SIMULSTREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMULSTREAMING_ROOT))
