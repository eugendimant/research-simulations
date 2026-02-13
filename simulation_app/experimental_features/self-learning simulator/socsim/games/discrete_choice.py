from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class DiscreteChoiceTask(Game):
    name = "discrete_choice"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        # spec: {"alternatives":[{"price":..., "feature_x":...}, ...], "beta": {"price":-1, "feature_x":0.5}, "gumbel_scale":1.0}
        alts: List[Dict[str, Any]] = list(spec.get("alternatives", []) or [])
        if not alts:
            raise ValueError("discrete_choice requires a non-empty list of alternatives")
        beta = dict(spec.get("beta", {}) or {})
        scale = float(spec.get("gumbel_scale", 1.0))

        # allow persona-level taste shifts (random coefficients)
        taste = dict(a.params.get("taste", {}) or {})
        util = []
        for alt in alts:
            u = 0.0
            for k, b0 in beta.items():
                x = float(alt.get(k, 0.0))
                b_eff = float(b0) + float(taste.get(k, 0.0))
                u += b_eff * x
            # Gumbel noise (logit)
            noise = -np.log(-np.log(max(1e-12, rng.random())))
            util.append(u + scale*noise)

        choice = int(np.argmax(util))
        return GameOutcome(
            actions={"choice": choice},
            payoffs={"A": 0.0, "B": 0.0},
            trace={"utilities": util, "alternatives": alts},
        )
