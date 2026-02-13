from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice
from ..models.social_prefs import fehr_schmidt_utility

class UltimatumGame(Game):
    name = "ultimatum"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        if b is None:
            raise ValueError("Ultimatum requires two personas (proposer a, responder b).")

        E = float(spec.get("endowment", 10.0))
        step = float(spec.get("step", 1.0))
        offers = np.arange(0.0, E + 1e-9, step)

        lam = float(a.params.get("noise_lambda", 2.0))
        alpha = float(a.params.get("fairness_alpha", 0.8))
        beta = float(a.params.get("fairness_beta", 0.2))

        # Proposer belief about acceptance threshold (share)
        thr = float(a.params.get("belief_min_accept_share", 0.3))
        tau = float(a.params.get("strategic_tau", 8.0))

        # approximate acceptance probability as smooth step around threshold
        shares = offers / max(E, 1e-9)
        p_accept = 1.0 / (1.0 + np.exp(-tau * (shares - thr)))

        u_accept = np.array([fehr_schmidt_utility(E - o, o, alpha=alpha, beta=beta) for o in offers], dtype=float)
        u_reject = 0.0
        u_exp = p_accept * u_accept + (1.0 - p_accept) * u_reject

        idx_offer, offer_probs = logit_choice(rng, u_exp, lam=lam)
        offer = float(offers[idx_offer])

        # Responder decision
        lam_b = float(b.params.get("noise_lambda", 2.0))
        alpha_b = float(b.params.get("fairness_alpha", 0.8))
        beta_b = float(b.params.get("fairness_beta", 0.2))
        min_share = float(b.params.get("min_accept_share", 0.3))

        accept_utility = fehr_schmidt_utility(offer, E - offer, alpha=alpha_b, beta=beta_b)
        # add a soft "fairness threshold" disutility below min_share
        share = offer / max(E, 1e-9)
        accept_utility -= 5.0 * max(min_share - share, 0.0)

        reject_utility = 0.0

        idx_acc, acc_probs = logit_choice(rng, np.array([reject_utility, accept_utility], dtype=float), lam=lam_b)
        accepted = bool(idx_acc == 1)

        if accepted:
            pay_a = E - offer
            pay_b = offer
        else:
            pay_a = 0.0
            pay_b = 0.0

        return GameOutcome(
            actions={"offer": offer, "accepted": accepted},
            payoffs={"a": float(pay_a), "b": float(pay_b)},
            trace={
                "offers": offers.tolist(),
                "offer_probs": offer_probs.tolist(),
                "p_accept_est": p_accept.tolist(),
                "accept_probs": acc_probs.tolist(),
                "params_a": {"alpha": alpha, "beta": beta, "lam": lam, "thr": thr, "tau": tau},
                "params_b": {"alpha": alpha_b, "beta": beta_b, "lam": lam_b, "min_share": min_share},
            },
        )
