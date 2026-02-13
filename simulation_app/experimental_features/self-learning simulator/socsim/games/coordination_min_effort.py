from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

@dataclass
class CoordinationMinEffort(Game):
    name: str = "coordination_min_effort"

    def simulate_group(self, rng: np.random.Generator, agents: List[Persona], params: Dict[str, Any]) -> Tuple[List[Dict[str, float]], Dict[str, float], Dict[str, Any]]:
        levels = int(params.get("levels", 7))
        base_risk = float(params.get("riskiness", 0.25))
        efforts = []
        rows = []
        for a in agents:
            trust = float(a.params.get("trust_propensity", 0.0))
            risk = base_risk + float(a.params.get("risk_aversion", 0.0))
            score = trust - risk
            p_hi = 1.0 / (1.0 + np.exp(-score))
            e = 1 + int(np.floor((levels - 1) * p_hi + rng.normal(0.0, 0.5)))
            e = int(np.clip(e, 1, levels))
            efforts.append(e)
            rows.append({"act::effort": float(e)})
        return rows, {"min_effort": float(min(efforts))}, {"efforts": efforts}

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona, params: Dict[str, Any]) -> GameOutcome:
        return GameOutcome(actions={}, payoffs={})
