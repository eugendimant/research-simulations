from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import numpy as np

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def stable_hash(obj: Any) -> str:
    b = json_bytes(obj)
    return hashlib.sha256(b).hexdigest()[:16]

def json_bytes(obj: Any) -> bytes:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")

def sample_trunc_normal(rng: np.random.Generator, mean: float, sd: float, lo: float, hi: float) -> float:
    for _ in range(25_000):
        x = rng.normal(mean, sd)
        if lo <= x <= hi:
            return float(x)
    return float(clamp(mean, lo, hi))

@dataclass
class ParamPrior:
    mean: float
    sd: float
    lo: float
    hi: float

def parse_priors(priors: Dict[str, Dict]) -> Dict[str, ParamPrior]:
    out: Dict[str, ParamPrior] = {}
    for k, v in priors.items():
        out[k] = ParamPrior(mean=float(v["mean"]), sd=float(v["sd"]), lo=float(v["min"]), hi=float(v["max"]))
    return out
