from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

class StagHunt(Game):
    name = "stag_hunt"

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        # payoffs
        r_cc = float(spec.get("r_cc", 4.0))
        r_cd = float(spec.get("r_cd", 0.0))
        r_dc = float(spec.get("r_dc", 3.0))
        r_dd = float(spec.get("r_dd", 2.0))

        def p_coop(p: Persona) -> float:
            trust = float(p.params.get("trust", 0.0))
            risk = float(p.params.get("risk_aversion", 0.0))
            # map to [0,1] using sigmoid-like
            z = 0.0 + 0.7 * trust - 0.7 * risk
            return float(1.0 / (1.0 + np.exp(-z)))

        pa = p_coop(a)
        if b is None:
            pb = float(spec.get("belief_other_coop", 0.5))
            # If no explicit partner, interpret pb as belief about others.
        else:
            pb = p_coop(b)

        a_choice = "C" if rng.random() < pa else "D"
        b_choice = "C" if rng.random() < pb else "D"

        if a_choice == "C" and b_choice == "C":
            pay_a, pay_b = r_cc, r_cc
        elif a_choice == "C" and b_choice == "D":
            pay_a, pay_b = r_cd, r_dc
        elif a_choice == "D" and b_choice == "C":
            pay_a, pay_b = r_dc, r_cd
        else:
            pay_a, pay_b = r_dd, r_dd

        return GameOutcome(
            actions={"choice_A": a_choice, "choice_B": b_choice, "pA": pa, "pB": pb},
            payoffs={"A": float(pay_a), "B": float(pay_b)},
            trace={"spec": {"r_cc": r_cc, "r_cd": r_cd, "r_dc": r_dc, "r_dd": r_dd}}
        )
