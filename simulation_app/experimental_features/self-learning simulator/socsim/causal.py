from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from .utils import ParamPrior, sample_trunc_normal

@dataclass
class DoIntervention:
    overrides: Dict[str, float]

@dataclass
class ParamDrawResult:
    params: Dict[str, float]
    mean_shifts: Dict[str, float]
    extra_sd: Dict[str, float]
    matched_evidence_ids: list[str]

def draw_params(
    rng: np.random.Generator,
    priors: Dict[str, ParamPrior],
    mean_shifts: Dict[str, float],
    extra_sd: Dict[str, float],
    class_shift: Dict[str, float],
    intervention: Optional[DoIntervention] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, prior in priors.items():
        if intervention and k in intervention.overrides:
            out[k] = float(intervention.overrides[k])
            continue
        mu = prior.mean + float(mean_shifts.get(k, 0.0)) + float(class_shift.get(k, 0.0))
        sd = float(prior.sd)
        if k in extra_sd:
            sd = float((sd**2 + float(extra_sd[k])**2) ** 0.5)
        out[k] = sample_trunc_normal(rng, mu, sd, prior.lo, prior.hi)
    return out
