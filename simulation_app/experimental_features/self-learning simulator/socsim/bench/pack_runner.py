from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
import hashlib

from .download import download_with_cache, verify_sha256
from .moments import moments_from_dict, compute_targets

@dataclass
class PackRunResult:
    benchmark_id: str
    targets_path: Path
    provenance_path: Path
    targets: Dict[str, Any]
    provenance: Dict[str, Any]

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def run_benchmark_pack(bench_dir: Path, out_dir: Path, cache_dir: Path, timeout_s: int = 60) -> PackRunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))
    adapter = json.loads((bench_dir / "adapter.json").read_text(encoding="utf-8"))
    moments = json.loads((bench_dir / "moments.json").read_text(encoding="utf-8"))

    url = manifest["url"]
    expected = manifest.get("sha256")

    if isinstance(url, str) and url.startswith("LOCAL:"):
        rel = url.split("LOCAL:", 1)[1]
        dl_path = (bench_dir / rel).resolve()
        b = dl_path.read_bytes()
        dl_sha = _sha256_bytes(b)
        dl = type("DL", (), {"path": dl_path, "sha256": dl_sha, "from_cache": True})
    else:
        dl = download_with_cache(url, cache_dir=cache_dir, timeout_s=timeout_s)

    if expected:
        verify_sha256(dl.path, expected)

    fmt = adapter.get("format", "csv")
    if fmt == "csv":
        df = pd.read_csv(dl.path)
    elif fmt == "parquet":
        df = pd.read_parquet(dl.path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    dropna_cols = adapter.get("dropna_cols") or []
    if dropna_cols:
        df = df.dropna(subset=dropna_cols)

    add_cols = adapter.get("add_columns") or {}
    for k, v in add_cols.items():
        df[k] = v

    colmap: Dict[str, str] = adapter.get("columns", {})
    for src, dst in colmap.items():
        if src != dst and src in df.columns:
            df = df.rename(columns={src: dst})

    targets = compute_targets(df, moments_from_dict(moments))
    targets_path = out_dir / "targets.json"
    targets_path.write_text(json.dumps(targets, indent=2), encoding="utf-8")

    provenance = {
        "benchmark_id": bench_dir.name,
        "url": url,
        "download_sha256": getattr(dl, "sha256", None),
        "from_cache": bool(getattr(dl, "from_cache", False)),
        "expected_sha256": expected,
        "source_url": manifest.get("source_url"),
        "license": manifest.get("license"),
        "citation": manifest.get("citation"),
    }
    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    return PackRunResult(bench_dir.name, targets_path, prov_path, targets, provenance)
