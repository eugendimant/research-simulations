from __future__ import annotations
from typing import Dict
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice
from ..models.social_prefs import fehr_schmidt_utility, ingroup_adjustment, norm_utility

class DictatorGame(Game):
    name = "dictator"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        E = float(spec.get("endowment", 10.0))
        step = float(spec.get("step", 1.0))
        is_ingroup = int(spec.get("is_ingroup", 1))
        norm_target = float(spec.get("norm_target_share", 0.5))
        amounts = np.arange(0.0, E + 1e-9, step)

        alpha = float(a.params["fairness_alpha"])
        beta = float(a.params["fairness_beta"])
        b0 = float(a.params.get("baseline_prosocial", 0.0))
        bias = ingroup_adjustment(float(a.params.get("ingroup_bias", 0.0)), is_ingroup)
        lam = float(a.params["noise_lambda"])
        w_norm = float(a.params.get("norm_weight", 0.0))

        utilities, components = [], []
        for g in amounts:
            pi_i, pi_j = E - g, g
            u_fs = fehr_schmidt_utility(pi_i, pi_j, alpha=alpha, beta=beta)
            u_bias = bias * (g / (E + 1e-9))
            share = g / (E + 1e-9)
            u_norm = norm_utility(component=-(share - norm_target) ** 2, norm_weight=w_norm)
            u = u_fs + b0 + u_bias + u_norm
            utilities.append(u)
            components.append({"u_fs": u_fs, "u_bias": u_bias, "u_norm": u_norm, "b0": b0})

        utilities = np.array(utilities, dtype=float)
        idx, probs = logit_choice(rng, utilities, lam=lam)
        g = float(amounts[idx])

        return GameOutcome(
            actions={"give": g, "share": g/(E+1e-9)},
            payoffs={"A": E - g, "B": g},
            trace={"grid": amounts.tolist(), "utilities": utilities.tolist(), "probs": probs.tolist(), "components": components, "is_ingroup": is_ingroup},
        )
