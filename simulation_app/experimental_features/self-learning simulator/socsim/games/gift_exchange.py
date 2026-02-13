from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice

class GiftExchangeGame(Game):
    name = "gift_exchange"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        if b is None:
            raise ValueError("Gift exchange requires two personas (employer a, worker b).")

        # Stylized: employer chooses wage w in [w_min,w_max]; worker chooses effort e in [0,1].
        w_min = float(spec.get("w_min", 0.0))
        w_max = float(spec.get("w_max", 10.0))
        step = float(spec.get("step", 1.0))
        wages = np.arange(w_min, w_max + 1e-9, step)

        lam_e = float(a.params.get("noise_lambda", 2.0))
        generosity = float(a.params.get("wage_generosity", 0.5))
        belief_effort_slope = float(a.params.get("strategic_tau", 3.0))

        # Employer expected effort increases with wage, saturating
        exp_effort = 1.0 / (1.0 + np.exp(-belief_effort_slope * ((wages - (w_min+w_max)/2.0) / max(w_max-w_min,1e-9))))
        # Employer payoff: value from effort (v*e) - wage; plus generosity taste
        v = float(spec.get("v", 12.0))
        u_w = v * exp_effort - wages + generosity * wages
        idx_w, probs_w = logit_choice(rng, u_w.astype(float), lam=lam_e)
        w = float(wages[idx_w])

        # Worker chooses effort given wage
        lam_w = float(b.params.get("noise_lambda", 2.0))
        recip = float(b.params.get("reciprocity_r", 0.6))
        effort_cost = float(b.params.get("effort_cost", 1.2))
        efforts = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

        # Worker payoff: wage - cost*e^2; reciprocity increases with wage
        u_e = []
        for e in efforts:
            u_e.append(w - effort_cost * (e**2) + recip * e * (w / max(w_max,1e-9)))
        u_e = np.array(u_e, dtype=float)
        idx_e, probs_e = logit_choice(rng, u_e, lam=lam_w)
        e = float(efforts[idx_e])

        pay_employer = v * e - w
        pay_worker = w - effort_cost * (e**2)

        return GameOutcome(
            actions={"wage": w, "effort": e},
            payoffs={"a": float(pay_employer), "b": float(pay_worker)},
            trace={"wage_options": wages.tolist(), "wage_probs": probs_w.tolist(), "effort_probs": probs_e.tolist()},
        )
