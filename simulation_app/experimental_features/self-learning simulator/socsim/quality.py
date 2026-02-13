from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

@dataclass(frozen=True)
class QualityScore:
    weight: float
    notes: str

def score_quality(quality: Optional[Dict[str, Any]]) -> QualityScore:
    """Conservative weight in [0.05, 1.0] computed ONLY from explicit metadata."""
    if not quality:
        return QualityScore(weight=0.15, notes="no quality metadata")

    et = str(quality.get("evidence_type", "unknown"))
    rob = str(quality.get("risk_of_bias", "unknown"))
    n = quality.get("sample_size", None)

    base = {
        "meta_analysis": 0.9,
        "field_experiment": 0.75,
        "experiment": 0.65,
        "observational": 0.45,
        "measurement": 0.4,
        "theory": 0.35,
        "unknown": 0.2,
    }.get(et, 0.2)

    bias_mult = {
        "low": 1.0,
        "some_concerns": 0.75,
        "high": 0.45,
        "unknown": 0.6,
    }.get(rob, 0.6)

    n_mult = 1.0
    if isinstance(n, int) and n > 0:
        n_mult = min(1.0, 0.5 + 0.5 * math.tanh(math.log10(max(n, 10)) / 2.0))

    w = max(0.05, min(1.0, float(base * bias_mult * n_mult)))
    return QualityScore(weight=w, notes=f"type={et}, bias={rob}, n={n}")
