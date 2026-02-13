from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class CommonPoolResource(Game):
    name = "common_pool_resource"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        max_take = float(spec.get("max_take", 30.0))
        pros = float(a.params.get("prosociality", 0.0))
        norm = float(a.params.get("norm_sensitivity", 0.0))
        base = 0.6 - 0.25 * pros - 0.15 * norm
        base = float(np.clip(base, 0.05, 0.95))
        take = base * max_take + float(rng.normal(0.0, max_take * 0.08))
        take = float(np.clip(take, 0.0, max_take))
        return GameOutcome(actions={"take": take}, payoffs={"A": 0.0, "B": 0.0}, trace={"base": base})
