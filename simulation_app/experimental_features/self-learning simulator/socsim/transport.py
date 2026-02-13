from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import math

@dataclass(frozen=True)
class TransportScore:
    score: float
    notes: str

def transportability(features: Dict[str, Any], expected: Dict[str, Any] | None) -> TransportScore:
    """Transport score in [0,1] based on explicit overlap with expected context_features.

    Missing keys reduce score.
    """
    if not expected:
        return TransportScore(score=0.6, notes="no expected context_features")

    matches = 0.0
    total = 0.0
    missing = 0
    for k, v in expected.items():
        total += 1.0
        if k not in features:
            missing += 1
            continue
        if str(features.get(k)) == str(v):
            matches += 1.0

    if total <= 0:
        return TransportScore(score=0.6, notes="empty expected context_features")

    raw = matches / total
    penalty = 1.0 - min(0.4, 0.1 * missing)
    score = max(0.05, min(1.0, raw * penalty))
    return TransportScore(score=score, notes=f"matches={matches}/{total}, missing={missing}")

def attenuate(delta: float, score: float) -> float:
    t = max(0.0, min(1.0, float(score)))
    return float(delta) * (t ** 1.5)
