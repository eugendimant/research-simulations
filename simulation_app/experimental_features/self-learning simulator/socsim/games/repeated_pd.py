from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any
from .base import Game, GameOutcome
from ..decision import logit_choice

class RepeatedPD(Game):
    """Repeated prisoner's dilemma with simple EWA-style reinforcement on cooperate/defect.
    Parameters:
      rounds (int)
      payoffs T,R,P,S
      phi (0..1) decay, rho (0..1) experience weight
    """
    name = "repeated_pd"

    def simulate_one(self, rng, a, b=None, params: Optional[Dict[str, Any]] = None) -> GameOutcome:
        p = params or {}
        rounds = int(p.get("rounds", 10))
        T = float(p.get("T", 5.0)); R = float(p.get("R", 3.0)); P = float(p.get("P", 1.0)); S = float(p.get("S", 0.0))
        phi = float(p.get("phi", 0.9))
        rho = float(p.get("rho", 0.5))

        beta_a = float(a.params.get("choice_precision", 2.0))
        beta_b = float((b.params.get("choice_precision", 2.0) if b is not None else beta_a))

        # initial attractions: tilt by prosociality and reciprocity
        def init_attr(persona):
            pros = float(persona.params.get("prosociality", 0.0))
            rec = float(persona.params.get("reciprocity", 0.0))
            # attraction to C increases with pros and with reciprocity, D decreases
            return np.array([0.2 + pros + 0.5*rec, 0.2 - pros], dtype=float)

        A_attr = init_attr(a)
        B_attr = init_attr(b) if b is not None else init_attr(a)

        actsA, actsB = [], []
        payA = 0.0; payB = 0.0
        traces = []

        for t in range(1, rounds+1):
            actA = "C" if logit_choice(rng, A_attr, beta=beta_a)==0 else "D"
            actB = "C" if logit_choice(rng, B_attr, beta=beta_b)==0 else "D"
            actsA.append(actA); actsB.append(actB)

            if actA=="C" and actB=="C":
                rA, rB = R, R
            elif actA=="C" and actB=="D":
                rA, rB = S, T
            elif actA=="D" and actB=="C":
                rA, rB = T, S
            else:
                rA, rB = P, P

            payA += rA; payB += rB

            # EWA update: A_attr = phi*A_attr + payoff_of_action (with counterfactual weight rho)
            # Counterfactual payoffs: what if other action
            if actB=="C":
                cfA_C, cfA_D = R, T
            else:
                cfA_C, cfA_D = S, P
            if actA=="C":
                realized_idx = 0
            else:
                realized_idx = 1
            cf_vec_A = np.array([cfA_C, cfA_D], dtype=float)
            A_attr = phi*A_attr + (1-rho)*np.eye(2)[realized_idx]*rA + rho*cf_vec_A

            if actA=="C":
                cfB_C, cfB_D = R, T if actA=="C" else None
            # compute B counterfactual properly
            if actA=="C":
                cfB_C, cfB_D = R, T
            else:
                cfB_C, cfB_D = S, P
            cf_vec_B = np.array([cfB_C, cfB_D], dtype=float)
            realized_idx_b = 0 if actB=="C" else 1
            B_attr = phi*B_attr + (1-rho)*np.eye(2)[realized_idx_b]*rB + rho*cf_vec_B

            traces.append({"round": t, "A_attr": A_attr.tolist(), "B_attr": B_attr.tolist(), "rA": rA, "rB": rB})

        return GameOutcome(
            actions={"A": actsA, "B": actsB, "rounds": rounds},
            payoffs={"A": float(payA), "B": float(payB)},
            trace={"round_traces": traces}
        )
