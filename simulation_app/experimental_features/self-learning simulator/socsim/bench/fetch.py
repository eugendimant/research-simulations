from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timezone

from .registry import BenchmarkRegistry
from .providers import DryadClient, DataverseClient, OSFClient

@dataclass
class FetchResult:
    downloaded_files: List[str]
    manifest_path: str

def fetch_all(registry_path: Path, out_dir: Path, user_agent: str, timeout_s: int = 30, min_interval_s: float = 1.0) -> List[FetchResult]:
    reg = BenchmarkRegistry.load(registry_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    dryad = DryadClient(user_agent=user_agent, timeout_s=timeout_s, min_interval_s=min_interval_s)
    osf = OSFClient(user_agent=user_agent, timeout_s=timeout_s, min_interval_s=min_interval_s)

    results: List[FetchResult] = []
    for b in reg.benchmarks:
        bdir = out_dir / b.id
        raw = bdir / "raw"
        raw.mkdir(parents=True, exist_ok=True)

        src = b.source
        provider = src.get("provider")
        manifest: Dict[str, Any] = {
            "benchmark_id": b.id,
            "game": b.game,
            "provider": provider,
            "source": src,
            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": [],
        }
        downloaded: List[str] = []

        if provider == "dryad":
            doi = src.get("doi")
            if not doi:
                raise ValueError(f"{b.id}: dryad requires source.doi")
            files, man = dryad.download_files(doi=doi, out_dir=raw)
            manifest["provider_manifest"] = man
            for f in files:
                manifest["files"].append({"path": str(f.path), "url": f.url, "sha256": f.sha256})
                downloaded.append(str(f.path))

        elif provider == "dataverse":
            pid = src.get("persistentId")
            if not pid:
                raise ValueError(f"{b.id}: dataverse requires source.persistentId")
            server = src.get("server", "https://dataverse.harvard.edu")
            dv = DataverseClient(server=server, user_agent=user_agent, timeout_s=timeout_s, min_interval_s=min_interval_s)
            files, man = dv.download_dataset_files(persistent_id=pid, out_dir=raw)
            manifest["provider_manifest"] = man
            for f in files:
                manifest["files"].append({"path": str(f.path), "url": f.url, "sha256": f.sha256})
                downloaded.append(str(f.path))

        elif provider == "osf":
            osf_id = src.get("osf_id")
            if not osf_id:
                raise ValueError(f"{b.id}: osf requires source.osf_id")
            files, man = osf.download_shortlink(osf_id=osf_id, out_dir=raw, filename=src.get("filename"))
            manifest["provider_manifest"] = man
            for f in files:
                manifest["files"].append({"path": str(f.path), "url": f.url, "sha256": f.sha256})
                downloaded.append(str(f.path))

        else:
            raise ValueError(f"{b.id}: unsupported provider {provider}")

        mpath = bdir / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        results.append(FetchResult(downloaded_files=downloaded, manifest_path=str(mpath)))

    return results
