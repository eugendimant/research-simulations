from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import time
import requests

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class DownloadedFile:
    path: Path
    url: str
    sha256: str

class RateLimiter:
    def __init__(self, min_interval_s: float = 1.0) -> None:
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0

    def wait(self) -> None:
        now = time.time()
        dt = now - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.time()

class DryadClient:
    """Dryad API v2 public download helper."""
    BASE = "https://datadryad.org/api/v2"

    def __init__(self, user_agent: str, timeout_s: int = 30, min_interval_s: float = 1.0) -> None:
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": user_agent})
        self.timeout_s = int(timeout_s)
        self.rl = RateLimiter(min_interval_s=min_interval_s)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.rl.wait()
        r = self.sess.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def resolve_doi_to_dataset_id(self, doi: str) -> str:
        data = self._get_json(f"{self.BASE}/datasets/dois/{doi}")
        return str(data["id"])

    def list_latest_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        vers = self._get_json(f"{self.BASE}/datasets/{dataset_id}/versions")
        items = vers.get("_embedded", {}).get("stash:versions", [])
        if not items:
            raise ValueError("No versions found for dataset")
        items_sorted = sorted(items, key=lambda x: x.get("versionNumber", 0), reverse=True)
        v_id = str(items_sorted[0]["id"])
        files = self._get_json(f"{self.BASE}/versions/{v_id}/files")
        return files.get("_embedded", {}).get("stash:files", [])

    def download_files(self, doi: str, out_dir: Path) -> Tuple[List[DownloadedFile], Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = self.resolve_doi_to_dataset_id(doi)
        files = self.list_latest_files(dataset_id)
        downloaded: List[DownloadedFile] = []
        manifest = {"provider":"dryad","doi":doi,"dataset_id":dataset_id,"files":[]}

        for f in files:
            href = f.get("_links", {}).get("stash:download", {}).get("href")
            if not href:
                continue
            filename = f.get("path") or f.get("filename") or str(f.get("id"))
            target = out_dir / str(filename).split("/")[-1]
            self.rl.wait()
            r = self.sess.get(href, timeout=self.timeout_s, stream=True)
            r.raise_for_status()
            with target.open("wb") as w:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        w.write(chunk)
            digest = sha256_file(target)
            downloaded.append(DownloadedFile(path=target, url=href, sha256=digest))
            manifest["files"].append({"name":target.name,"url":href,"sha256":digest,"bytes":target.stat().st_size})
        return downloaded, manifest

class DataverseClient:
    def __init__(self, server: str, user_agent: str, timeout_s: int = 30, min_interval_s: float = 1.0) -> None:
        self.server = server.rstrip("/")
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": user_agent})
        self.timeout_s = int(timeout_s)
        self.rl = RateLimiter(min_interval_s=min_interval_s)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.rl.wait()
        r = self.sess.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def download_dataset_files(self, persistent_id: str, out_dir: Path) -> Tuple[List[DownloadedFile], Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = self._get_json(f"{self.server}/api/datasets/:persistentId", params={"persistentId": persistent_id})
        data = ds.get("data", {})
        files = data.get("latestVersion", {}).get("files", [])
        downloaded: List[DownloadedFile] = []
        manifest = {"provider":"dataverse","server":self.server,"persistentId":persistent_id,"files":[]}
        for f in files:
            df = f.get("dataFile", {})
            file_id = df.get("id")
            if file_id is None:
                continue
            url = f"{self.server}/api/access/datafile/{file_id}"
            target = out_dir / (df.get("filename") or str(file_id))
            self.rl.wait()
            r = self.sess.get(url, timeout=self.timeout_s, stream=True)
            r.raise_for_status()
            with target.open("wb") as w:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        w.write(chunk)
            digest = sha256_file(target)
            downloaded.append(DownloadedFile(path=target, url=url, sha256=digest))
            manifest["files"].append({"name":target.name,"url":url,"sha256":digest,"bytes":target.stat().st_size})
        return downloaded, manifest

class OSFClient:
    def __init__(self, user_agent: str, timeout_s: int = 30, min_interval_s: float = 1.0) -> None:
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": user_agent})
        self.timeout_s = int(timeout_s)
        self.rl = RateLimiter(min_interval_s=min_interval_s)

    def download_shortlink(self, osf_id: str, out_dir: Path, filename: Optional[str] = None) -> Tuple[List[DownloadedFile], Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://osf.io/{osf_id}/download"
        target = out_dir / (filename or f"osf_{osf_id}.bin")
        self.rl.wait()
        r = self.sess.get(url, timeout=self.timeout_s, stream=True)
        r.raise_for_status()
        with target.open("wb") as w:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    w.write(chunk)
        digest = sha256_file(target)
        manifest = {"provider":"osf","osf_id":osf_id,"files":[{"name":target.name,"url":url,"sha256":digest,"bytes":target.stat().st_size}]}
        return [DownloadedFile(path=target, url=url, sha256=digest)], manifest
