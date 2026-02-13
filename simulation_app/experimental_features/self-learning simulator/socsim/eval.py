from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import csv
import math
import json

@dataclass
class MomentError:
    key: str
    pred: float
    obs: float
    abs_err: float

def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def _mean_numeric(rows: List[Dict[str, Any]], col: str) -> float | None:
    xs = []
    for row in rows:
        v = row.get(col, "")
        try:
            x = float(v)
            if math.isfinite(x):
                xs.append(x)
        except Exception:
            continue
    if not xs:
        return None
    return sum(xs)/len(xs)

def compare_moments(pred_csv: Path, obs_csv: Path, moment_cols: List[str]) -> List[MomentError]:
    pred = _read_csv(pred_csv)
    obs = _read_csv(obs_csv)
    out: List[MomentError] = []
    for c in moment_cols:
        mp = _mean_numeric(pred, c)
        mo = _mean_numeric(obs, c)
        if mp is None or mo is None:
            continue
        out.append(MomentError(key=c, pred=float(mp), obs=float(mo), abs_err=float(abs(mp-mo))))
    return out

def write_report(errors: List[MomentError], out_path: Path) -> None:
    payload = {"moments":[e.__dict__ for e in errors],
               "mae": (sum(e.abs_err for e in errors)/len(errors)) if errors else None}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
