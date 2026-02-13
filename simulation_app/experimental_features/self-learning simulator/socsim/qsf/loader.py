from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def load_qsf(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
