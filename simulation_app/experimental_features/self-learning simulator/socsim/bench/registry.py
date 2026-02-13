from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

@dataclass
class Benchmark:
    id: str
    game: str
    source: Dict[str, Any]
    context_features: Dict[str, Any]
    notes: str

class BenchmarkRegistry:
    def __init__(self, version: str, benchmarks: List[Benchmark]) -> None:
        self.version = version
        self.benchmarks = benchmarks

    @staticmethod
    def load(path: Path) -> "BenchmarkRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        bms: List[Benchmark] = []
        for b in data.get("benchmarks", []):
            bms.append(Benchmark(
                id=str(b["id"]),
                game=str(b["game"]),
                source=dict(b["source"]),
                context_features=dict(b.get("context_features", {})),
                notes=str(b.get("notes", "")),
            ))
        return BenchmarkRegistry(version=str(data.get("version","")), benchmarks=bms)
