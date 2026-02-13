from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class TullockContest(Game):
    name = "tullock_contest"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        endowment = float(spec.get("endowment", 10.0))
        comp = float(a.params.get("competitiveness", 0.0))
        risk = float(a.params.get("risk_aversion", 0.0))
        base = 0.35 + 0.25 * comp - 0.15 * risk
        base = float(np.clip(base, 0.0, 1.0))
        e = base * endowment + float(rng.normal(0.0, endowment * 0.07))
        e = float(np.clip(e, 0.0, endowment))
        return GameOutcome(actions={"effort": e}, payoffs={"A": 0.0, "B": 0.0}, trace={"base": base})
