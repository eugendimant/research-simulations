from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice
from ..models.social_prefs import fehr_schmidt_utility

class TrustGame(Game):
    name = "trust"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        if b is None:
            raise ValueError("Trust game requires two personas (trustor a, trustee b).")

        E = float(spec.get("endowment", 10.0))
        mult = float(spec.get("multiplier", 3.0))
        step = float(spec.get("step", 1.0))

        invests = np.arange(0.0, E + 1e-9, step)

        lam_a = float(a.params.get("noise_lambda", 2.0))
        alpha_a = float(a.params.get("fairness_alpha", 0.8))
        beta_a = float(a.params.get("fairness_beta", 0.2))
        trust_prop = float(a.params.get("trust_propensity", 0.5))
        belief_return = float(a.params.get("belief_return_frac", 0.35))

        # Trustor expected utility: keep + expected return, plus mild prosocial taste
        u = []
        for x in invests:
            sent = x
            received_by_b = mult * sent
            exp_return = belief_return * received_by_b
            pi_a = (E - sent) + exp_return
            pi_b = received_by_b - exp_return
            u.append(fehr_schmidt_utility(pi_a, pi_b, alpha=alpha_a, beta=beta_a) + 0.5 * trust_prop * sent)
        u = np.array(u, dtype=float)
        idx_x, probs_x = logit_choice(rng, u, lam=lam_a)
        x = float(invests[idx_x])

        # Trustee return decision: choose return fraction on grid
        lam_b = float(b.params.get("noise_lambda", 2.0))
        alpha_b = float(b.params.get("fairness_alpha", 0.8))
        beta_b = float(b.params.get("fairness_beta", 0.2))
        recip = float(b.params.get("reciprocity_r", 0.6))
        ret_prop = float(b.params.get("return_propensity", 0.45))

        received = mult * x
        fracs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        u_ret = []
        for f in fracs:
            y = f * received
            pi_b = received - y
            pi_a = (E - x) + y
            # reciprocity: higher return rewarded if received is high
            u_ret.append(fehr_schmidt_utility(pi_b, pi_a, alpha=alpha_b, beta=beta_b) + recip * f + 0.5 * ret_prop * f)
        u_ret = np.array(u_ret, dtype=float)
        idx_f, probs_f = logit_choice(rng, u_ret, lam=lam_b)
        f = float(fracs[idx_f])
        y = float(f * received)

        pay_a = (E - x) + y
        pay_b = received - y

        return GameOutcome(
            actions={"invest": x, "return": y, "return_frac": f},
            payoffs={"a": float(pay_a), "b": float(pay_b)},
            trace={
                "invest_options": invests.tolist(),
                "invest_probs": probs_x.tolist(),
                "return_fracs": fracs.tolist(),
                "return_probs": probs_f.tolist(),
                "params_a": {"belief_return_frac": belief_return, "trust_propensity": trust_prop},
                "params_b": {"reciprocity_r": recip, "return_propensity": ret_prop},
            },
        )
