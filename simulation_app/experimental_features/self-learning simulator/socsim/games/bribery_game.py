from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class BriberyGame(Game):
    name = "bribery_game"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        endowment = float(spec.get("endowment", 10.0))
        dishon = float(a.params.get("dishonesty", 0.0))
        risk = float(-a.params.get("risk_aversion", 0.0))
        norm = float(a.params.get("norm_sensitivity", 0.0))
        p = 0.3 + 0.25 * dishon + 0.10 * risk - 0.20 * norm
        p = float(np.clip(p, 0.0, 1.0))
        bribe = bool(rng.random() < p)
        amount = 0.0
        if bribe:
            amount = float(rng.normal(endowment * 0.4, endowment * 0.15))
            amount = float(np.clip(amount, 0.0, endowment))
        return GameOutcome(actions={"bribe": int(bribe), "bribe_amount": amount}, payoffs={"A": 0.0, "B": 0.0}, trace={"p": p})
