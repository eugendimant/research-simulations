from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json

@dataclass
class BenchmarkCase:
    name: str
    spec: str
    expected_moments: Dict[str, float]
    tolerance: float

def load_suite(path: Path) -> List[BenchmarkCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[BenchmarkCase] = []
    for c in data.get("cases", []):
        out.append(BenchmarkCase(
            name=str(c["name"]),
            spec=str(c["spec"]),
            expected_moments=dict(c.get("expected_moments", {})),
            tolerance=float(c.get("tolerance", 0.1)),
        ))
    return out
