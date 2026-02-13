from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice
from ..models.social_prefs import fehr_schmidt_utility, norm_utility

class PrisonersDilemma(Game):
    name = "pd"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        R = float(spec.get("R", 3.0))
        T = float(spec.get("T", 5.0))
        S = float(spec.get("S", 0.0))
        P = float(spec.get("P", 1.0))
        pC = float(spec.get("belief_partner_C", 0.5))

        lam = float(a.params["noise_lambda"])
        w_norm = float(a.params.get("norm_weight", 0.0))
        alpha = float(a.params["fairness_alpha"])
        beta  = float(a.params["fairness_beta"])
        b0 = float(a.params.get("baseline_prosocial", 0.0))

        piA_C = pC * R + (1-pC) * S
        piB_C = pC * R + (1-pC) * T
        uC = fehr_schmidt_utility(piA_C, piB_C, alpha=alpha, beta=beta) + b0 + norm_utility(component=1.0, norm_weight=w_norm)

        piA_D = pC * T + (1-pC) * P
        piB_D = pC * S + (1-pC) * P
        uD = fehr_schmidt_utility(piA_D, piB_D, alpha=alpha, beta=beta) + norm_utility(component=0.0, norm_weight=w_norm)

        u = np.array([uD, uC], dtype=float)
        idx, probs = logit_choice(rng, u, lam=lam)
        act = "C" if idx == 1 else "D"

        if b is not None:
            pb = float(spec.get("belief_partner_C_for_B", pC))
            lam_b = float(b.params["noise_lambda"])
            alpha_b = float(b.params["fairness_alpha"])
            beta_b  = float(b.params["fairness_beta"])
            b0_b = float(b.params.get("baseline_prosocial", 0.0))
            w_norm_b = float(b.params.get("norm_weight", 0.0))

            piB_C = pb * R + (1-pb) * S
            piA_C = pb * R + (1-pb) * T
            uC_b = fehr_schmidt_utility(piB_C, piA_C, alpha=alpha_b, beta=beta_b) + b0_b + norm_utility(component=1.0, norm_weight=w_norm_b)

            piB_D = pb * T + (1-pb) * P
            piA_D = pb * S + (1-pb) * P
            uD_b = fehr_schmidt_utility(piB_D, piA_D, alpha=alpha_b, beta=beta_b) + norm_utility(component=0.0, norm_weight=w_norm_b)

            u_b = np.array([uD_b, uC_b], dtype=float)
            idx_b, probs_b = logit_choice(rng, u_b, lam=lam_b)
            act_b = "C" if idx_b == 1 else "D"
        else:
            act_b = "C" if rng.random() < pC else "D"
            probs_b = None

        if act == "C" and act_b == "C":
            payA, payB = R, R
        elif act == "C" and act_b == "D":
            payA, payB = S, T
        elif act == "D" and act_b == "C":
            payA, payB = T, S
        else:
            payA, payB = P, P

        return GameOutcome(
            actions={"A": act, "B": act_b, "pC_A": float(probs[1]), "pC_B": float(probs_b[1]) if probs_b is not None else None},
            payoffs={"A": payA, "B": payB},
            trace={"u": [float(uD), float(uC)], "probs": probs.tolist(), "belief_partner_C": pC},
        )
