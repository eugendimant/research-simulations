from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime, timezone

@dataclass
class BenchPack:
    benchmark_id: str
    created_at_utc: str
    payload: Dict[str, Any]

def create_bench_pack(bench_dir: Path, out_path: Path) -> BenchPack:
    manifest = bench_dir / "manifest.json"
    adapter = bench_dir / "adapter.json"
    payload = {
        "benchmark_id": bench_dir.name,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None,
        "adapter": json.loads(adapter.read_text(encoding="utf-8")) if adapter.exists() else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BenchPack(benchmark_id=bench_dir.name, created_at_utc=payload["created_at_utc"], payload=payload)
