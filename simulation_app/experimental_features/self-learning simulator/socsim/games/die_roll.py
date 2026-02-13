from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice

class DieRollTask(Game):
    name = "die_roll"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        lam = float(a.params.get("noise_lambda", 2.0))
        honesty_cost = float(a.params.get("honesty_cost", 1.0))

        true_roll = int(rng.integers(1, 7))
        reports = np.arange(1, 7, dtype=int)

        u = []
        for r in reports:
            payoff = float(r)
            lie = abs(r - true_roll)
            u.append(payoff - honesty_cost * float(lie))
        u = np.array(u, dtype=float)

        idx, probs = logit_choice(rng, u, lam=lam)
        report = int(reports[idx])

        return GameOutcome(
            actions={"true_roll": true_roll, "report": report, "lied": bool(report != true_roll)},
            payoffs={"a": float(report)},
            trace={"reports": reports.tolist(), "probs": probs.tolist(), "honesty_cost": honesty_cost},
        )
