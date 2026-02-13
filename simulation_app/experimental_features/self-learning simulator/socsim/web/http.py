from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import hashlib
import json
import random
import time

import requests

@dataclass
class FetchResult:
    url: str
    status: int
    headers: Dict[str, str]
    body_text: str
    retrieved_at_utc: str
    body_sha256: str

class WebClient:
    """Conservative HTTP client with caching, rate limiting, and bounded retries."""

    def __init__(
        self,
        user_agent: str = "socsim/0.9.0 (metadata-only; contact: unspecified)",
        timeout_s: int = 30,
        min_interval_s: float = 1.0,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_s = int(timeout_s)
        self.min_interval_s = float(min_interval_s)
        self._last_call = 0.0
        self.cache_path = cache_path
        if cache_path is not None:
            self._init_cache(cache_path)

    def _init_cache(self, path: Path) -> None:
        import sqlite3
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS http_cache (
                url TEXT PRIMARY KEY,
                status INTEGER,
                headers_json TEXT,
                body_text TEXT,
                retrieved_at_utc TEXT,
                body_sha256 TEXT
            )
            """)
            con.commit()

    def _cache_get(self, url: str) -> Optional[FetchResult]:
        if self.cache_path is None:
            return None
        import sqlite3
        with sqlite3.connect(self.cache_path) as con:
            cur = con.execute(
                "SELECT status, headers_json, body_text, retrieved_at_utc, body_sha256 FROM http_cache WHERE url=?",
                (url,),
            )
            row = cur.fetchone()
            if not row:
                return None
            status, headers_json, body_text, retrieved_at_utc, body_sha256 = row
            return FetchResult(
                url=url,
                status=int(status),
                headers=json.loads(headers_json),
                body_text=str(body_text),
                retrieved_at_utc=str(retrieved_at_utc),
                body_sha256=str(body_sha256),
            )

    def _cache_put(self, res: FetchResult) -> None:
        if self.cache_path is None:
            return
        import sqlite3
        with sqlite3.connect(self.cache_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO http_cache(url,status,headers_json,body_text,retrieved_at_utc,body_sha256) VALUES(?,?,?,?,?,?)",
                (res.url, int(res.status), json.dumps(res.headers), res.body_text, res.retrieved_at_utc, res.body_sha256),
            )
            con.commit()

    def fetch_text(self, url: str, use_cache: bool = True, max_retries: int = 4) -> FetchResult:
        if use_cache:
            cached = self._cache_get(url)
            if cached is not None:
                return cached

        # rate limit
        now = time.time()
        dt = now - self._last_call
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)

        headers = {"User-Agent": self.user_agent, "Accept": "application/json,text/plain,*/*"}
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout_s)
                text = resp.text or ""
                sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                retrieved = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                res = FetchResult(
                    url=url,
                    status=int(resp.status_code),
                    headers={k: v for k, v in resp.headers.items()},
                    body_text=text,
                    retrieved_at_utc=retrieved,
                    body_sha256=sha,
                )
                self._last_call = time.time()
                self._cache_put(res)
                return res
            except Exception as e:
                last_exc = e
                sleep_s = (2 ** attempt) * 0.5 + random.random() * 0.25
                time.sleep(sleep_s)

        raise RuntimeError(f"Failed to fetch {url} after retries: {last_exc}")
