from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import requests

@dataclass
class DownloadResult:
    path: Path
    sha256: str
    from_cache: bool

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_with_cache(url: str, cache_dir: Path, timeout_s: int = 60, min_interval_s: float = 0.5) -> DownloadResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = hashlib.sha256(url.encode("utf-8")).hexdigest()
    out = cache_dir / safe
    if out.exists():
        return DownloadResult(path=out, sha256=sha256_file(out), from_cache=True)
    time.sleep(float(min_interval_s))
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    out.write_bytes(r.content)
    return DownloadResult(path=out, sha256=sha256_file(out), from_cache=False)

def verify_sha256(path: Path, expected_sha256: str) -> None:
    got = sha256_file(path)
    if got.lower() != expected_sha256.lower():
        raise ValueError(f"SHA256 mismatch: expected {expected_sha256} got {got}")
