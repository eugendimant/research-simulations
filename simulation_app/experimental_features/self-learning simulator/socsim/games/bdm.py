from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class BDMTask(Game):
    name = "bdm"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        # spec: {"value":1.0, "price_low":0.0, "price_high":1.0, "noise":0.1}
        value = float(spec.get("value", 1.0))
        price_low = float(spec.get("price_low", 0.0))
        price_high = float(spec.get("price_high", 1.0))
        noise = float(spec.get("noise", 0.05))

        # willingness to pay based on value plus persona "valuation_bias"
        bias = float(a.params.get("valuation_bias", 0.0))
        wtp = float(np.clip(value + bias + rng.normal(0.0, noise), price_low, price_high))
        price = float(rng.uniform(price_low, price_high))
        buy = int(wtp >= price)
        payoff = (value - price) if buy else 0.0

        return GameOutcome(
            actions={"wtp": wtp, "price": price, "buy": buy},
            payoffs={"A": float(payoff), "B": 0.0},
            trace={"value": value},
        )
