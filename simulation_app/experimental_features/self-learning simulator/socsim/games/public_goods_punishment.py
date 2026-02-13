from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class PublicGoodsWithPunishment(Game):
    name = "public_goods_punishment"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        if b is None:
            raise ValueError("public_goods_punishment requires two players (a and b).")

        endowment = float(spec.get("endowment", 1.0))
        mpcr = float(spec.get("mpcr", 0.6))  # marginal per-capita return
        punish_cost = float(spec.get("punish_cost", 0.05))
        punish_impact = float(spec.get("punish_impact", 0.15))

        def contrib(p: Persona) -> float:
            prosoc = float(p.params.get("prosociality", 0.0))
            norm = float(p.params.get("norm_sensitivity", 0.0))
            raw = 0.35 + 0.35*prosoc + 0.15*norm
            return float(np.clip(raw, 0.0, 1.0))*endowment

        ca = contrib(a)
        cb = contrib(b)

        pot = ca + cb
        ret_each = mpcr * pot

        # punishment stage: punish low contributors relative to own contribution
        def punish(actor: Persona, other_contrib: float, own_contrib: float) -> float:
            punitive = float(actor.params.get("punishment_propensity", 0.0))
            gap = max(0.0, own_contrib - other_contrib)
            raw = punitive * gap / max(endowment, 1e-9)
            return float(np.clip(raw, 0.0, 1.0))  # intensity 0..1

        pa = punish(a, cb, ca)
        pb = punish(b, ca, cb)

        payoff_a = endowment - ca + ret_each - punish_cost*pa - punish_impact*pb
        payoff_b = endowment - cb + ret_each - punish_cost*pb - punish_impact*pa

        trace = {"contribs": {"A": ca, "B": cb}, "punish": {"A": pa, "B": pb}}
        return GameOutcome(
            actions={"contrib_A": ca, "contrib_B": cb, "punish_A": pa, "punish_B": pb},
            payoffs={"A": float(payoff_a), "B": float(payoff_b)},
            trace=trace,
        )
