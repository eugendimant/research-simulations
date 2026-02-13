from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from .base import Game, GameOutcome
from ..surveys.likert_grm import LikertItem, simulate_likert_block

class SurveyLikert(Game):
    name = "survey_likert"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        items_spec = spec.get("items", [])
        if not items_spec:
            items_spec = [
                {"id": "q1", "a": 1.2, "thresholds": [-1.0, 0.0, 1.0, 2.0]},
                {"id": "q2", "a": 0.8, "thresholds": [-0.5, 0.5, 1.5, 2.5]},
            ]
        items: List[LikertItem] = [LikertItem(id=str(it["id"]), a=float(it["a"]), thresholds=list(map(float, it["thresholds"]))) for it in items_spec]
        theta = float(a.params.get("likert_theta", 0.0))
        out = simulate_likert_block(rng, theta, items)

        # payoffs not meaningful for surveys, return 0
        return GameOutcome(
            actions={f"resp::{k}": v for k, v in out["responses"].items()},
            payoffs={"A": 0.0},
            trace=out["trace"],
        )
