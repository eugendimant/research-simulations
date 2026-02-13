from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import zipfile
from datetime import datetime, timezone

def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class BundleManifest:
    created_at_utc: str
    files: Dict[str, Dict[str, str]]

def export_bundle(out_zip: Path, files: List[Path], extra: Optional[Dict] = None) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest: Dict[str, Dict[str, str]] = {}
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if not p.exists():
                continue
            arc = str(p)
            if p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file():
                        rel = str(fp)
                        z.write(fp, rel)
                        manifest[rel] = {"sha256": sha256_path(fp)}
            else:
                z.write(p, arc)
                manifest[arc] = {"sha256": sha256_path(p)}
        payload = {"created_at_utc": created, "files": manifest}
        if extra:
            payload["extra"] = extra
        z.writestr("MANIFEST.json", json.dumps(payload, indent=2))
    return out_zip
