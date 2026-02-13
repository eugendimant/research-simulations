from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class BeautyContest(Game):
    name = "beauty_contest"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        p = float(spec.get("p", 2/3))
        max_number = float(spec.get("max_number", 100.0))
        k = int(round(np.clip(a.params.get("strategic_depth", 1.0), 0.0, 4.0)))
        if k == 0:
            x = float(rng.uniform(0.0, max_number))
        else:
            x = float((p ** k) * (max_number / 2.0))
            x += float(rng.normal(0.0, max_number * 0.05 * (1.0 + 0.5 / (k + 1))))
        x = float(np.clip(x, 0.0, max_number))
        return GameOutcome(
            actions={"guess": x, "k_level": k, "p": p},
            payoffs={"A": 0.0, "B": 0.0},
            trace={"spec": {"p": p, "max_number": max_number}}
        )
