"""Fetch a benchmark study's human data and verify it, without redistributing it.

Usage: python -m benchmarks.fetch_human_data brader_valentino_suhay_2008 [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(study_id: str, out_dir: str) -> str:
    spec = json.load(open(os.path.join(HERE, "studies", study_id, "study.json")))
    url = spec["human_data"]["url"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)  # noqa: S310 - pinned https URL from study.json
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    pinned = os.path.join(HERE, "studies", study_id, "human_data.sha256")
    if os.path.exists(pinned):
        want = open(pinned).read().split()[0]
        if want != digest:
            raise SystemExit(f"SHA-256 mismatch for {path}: expected {want}, got {digest}")
    else:
        open(pinned, "w").write(f"{digest}  {os.path.basename(url)}\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--out", default=os.path.join(HERE, "_human_data"))
    a = ap.parse_args()
    print(fetch(a.study, a.out))
