from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona
from ..reputation import ReputationState, expected_reciprocation, update

class RepeatedTrustGame(Game):
    name = "repeated_trust"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        rounds = int(spec.get("rounds", 5))
        endowment = float(spec.get("endowment", 1.0))
        multiplier = float(spec.get("multiplier", 3.0))

        # belief about partner reciprocation
        state = ReputationState(alpha=1.0, beta=1.0)
        invest_total = 0.0
        return_total = 0.0
        trace = {"rounds": []}

        # Persona parameters
        prosoc = float(a.params.get("prosociality", 0.0))
        trust = float(a.params.get("trust_belief", 0.0))
        risk = float(a.params.get("risk_aversion", 0.0))

        for t in range(rounds):
            p_rec = expected_reciprocation(state)
            # investment propensity: belief + traits
            raw = 0.35 + 0.25*trust + 0.20*prosoc + 0.25*(p_rec-0.5) - 0.15*risk
            invest = float(np.clip(raw, 0.0, 1.0)) * endowment
            invest_total += invest

            # partner reciprocation: if b exists, use their prosociality, else stochastic around belief
            if b is not None:
                b_prosoc = float(b.params.get("prosociality", 0.0))
                recip_prob = float(np.clip(0.35 + 0.25*b_prosoc + 0.25*(p_rec-0.5), 0.05, 0.95))
            else:
                recip_prob = float(np.clip(p_rec, 0.05, 0.95))

            reciprocated = bool(rng.random() < recip_prob)
            if reciprocated:
                # return some share of multiplied amount
                ret_share = float(np.clip(0.25 + 0.30*(b.params.get("prosociality",0.0) if b else 0.0), 0.05, 0.8))
                returned = ret_share * (multiplier * invest)
            else:
                returned = 0.0

            return_total += returned
            state = update(state, reciprocated)

            trace["rounds"].append({"t": t+1, "invest": invest, "returned": returned, "p_rec": p_rec, "recip": reciprocated})

        payoff_a = (rounds * endowment) - invest_total + return_total
        payoff_b = 0.0 if b is None else (multiplier*invest_total - return_total)

        return GameOutcome(
            actions={"invest_total": invest_total, "return_total": return_total, "invest_mean": invest_total/rounds, "returned_mean": return_total/rounds},
            payoffs={"A": float(payoff_a), "B": float(payoff_b)},
            trace=trace,
        )

# Backward-compatible alias
RepeatedTrust = RepeatedTrustGame
