from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import zipfile
import pandas as pd
from datetime import datetime, timezone

from .registry import BenchmarkRegistry
from .moments import Moment, mean, var
from .providers import sha256_file

def _extract_archives(raw_dir: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for p in raw_dir.glob("*"):
        if p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(extracted_dir / p.stem)

def _load_tables(root: Path) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        try:
            if suf in [".csv", ".tsv"]:
                sep = "," if suf == ".csv" else "\t"
                dfs.append(pd.read_csv(p, sep=sep))
            elif suf == ".xlsx":
                dfs.append(pd.read_excel(p))
            elif suf == ".dta":
                dfs.append(pd.read_stata(p))
        except Exception:
            continue
    return dfs

def _score(name: str, keys: List[str]) -> float:
    n = name.lower()
    return float(sum(1 for k in keys if k in n))

def _best_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    best = None
    best_s = 0.0
    for c in df.columns:
        s = _score(str(c), keys)
        if s > best_s:
            best_s = s
            best = str(c)
    return best if best_s > 0 else None

def extract_targets_for_benchmark(bench_id: str, game: str, bench_dir: Path) -> Dict[str, Any]:
    raw = bench_dir / "raw"
    extracted = bench_dir / "extracted"
    _extract_archives(raw, extracted)

    dfs = _load_tables(bench_dir)

    moments: List[Moment] = []
    used_inputs: List[str] = []

    if game in ("ultimatum","dictator"):
        for df in dfs:
            give = _best_col(df, ["offer","give","donat","amount"])
            endow = _best_col(df, ["endow","pie","total","budget"])
            if give and endow:
                g = pd.to_numeric(df[give], errors="coerce")
                e = pd.to_numeric(df[endow], errors="coerce")
                frac = (g / e).dropna().clip(0,1)
                if len(frac) >= 30:
                    xs = frac.astype(float).tolist()
                    moments.append(Moment("obs::give_frac:mean", mean(xs), 1.0, f"{give}/{endow}"))
                    moments.append(Moment("obs::give_frac:var", var(xs), 0.5, f"{give}/{endow}"))
                    used_inputs.append("heuristic_table")
                    break

    if game in ("repeated_public_goods","public_goods_punishment"):
        for df in dfs:
            contrib = _best_col(df, ["contrib","contribution","give"])
            rnd = _best_col(df, ["round"])
            if contrib and rnd:
                c = pd.to_numeric(df[contrib], errors="coerce")
                r = pd.to_numeric(df[rnd], errors="coerce")
                tmp = pd.DataFrame({"c":c,"r":r}).dropna()
                if len(tmp) >= 50:
                    rmin = tmp["r"].min()
                    rmax = tmp["r"].max()
                    c1 = tmp.loc[tmp["r"]==rmin, "c"].astype(float).tolist()
                    cl = tmp.loc[tmp["r"]==rmax, "c"].astype(float).tolist()
                    moments.append(Moment("obs::contrib_r1:mean", mean(c1), 1.0, f"{contrib} @ {rmin}"))
                    moments.append(Moment("obs::contrib_rlast:mean", mean(cl), 1.0, f"{contrib} @ {rmax}"))
                    used_inputs.append("heuristic_table")
                    break

    prov = {"created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": used_inputs, "checksums": {}}
    for p in raw.glob("*"):
        if p.is_file():
            prov["checksums"][p.name] = sha256_file(p)

    return {"benchmark_id": bench_id,
            "moments": [m.__dict__ for m in moments],
            "provenance": prov}

def build_all_targets(registry_path: Path, bench_root: Path, out_root: Path) -> List[Path]:
    reg = BenchmarkRegistry.load(registry_path)
    out_root.mkdir(parents=True, exist_ok=True)
    outs: List[Path] = []
    for b in reg.benchmarks:
        t = extract_targets_for_benchmark(b.id, b.game, bench_root / b.id)
        outp = out_root / f"{b.id}.targets.json"
        outp.write_text(json.dumps(t, indent=2), encoding="utf-8")
        outs.append(outp)
    return outs
