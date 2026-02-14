"""Money Request 11-20 game.

The "11-20 Money Request" game (Arad & Rubinstein, 2012):
  - Two players each request an amount between 11 and 20
  - Each player receives their requested amount
  - BONUS: If one player requests exactly 1 less than the other,
    the lower-requesting player gets a bonus of 20
  - This creates tension between requesting high (more base pay)
    and requesting low (chance of bonus)

Parameterised version supports rule variations:
  - min_request, max_request: range of valid requests
  - bonus: size of the undercutting bonus
  - bonus_gap: how much lower you must be to get the bonus (default 1)
  - bonus_rule: "exactly_one_less" | "any_lower" | "closest_lower"

References:
  Arad & Rubinstein (2012). The 11-20 Money Request Game.
  AER 102(7):3561-3573.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice
from ..persona import Persona


class MoneyRequest1120(Game):
    name = "money_request_11_20"

    def simulate_one(
        self,
        rng: np.random.Generator,
        a: Persona,
        b: Persona | None,
        spec: Dict[str, Any],
    ) -> GameOutcome:
        min_req = int(spec.get("min_request", 11))
        max_req = int(spec.get("max_request", 20))
        bonus = float(spec.get("bonus", 20.0))
        bonus_gap = int(spec.get("bonus_gap", 1))
        bonus_rule = str(spec.get("bonus_rule", "exactly_one_less"))

        requests = list(range(min_req, max_req + 1))
        n = len(requests)

        # --- Player A's decision ---
        lam_a = float(a.params.get("noise_lambda", 1.0))
        depth_a = float(a.params.get("strategic_depth", 1.0))
        risk_a = float(a.params.get("risk_aversion", 0.5))

        # Expected payoff for each request, assuming opponent plays uniformly
        # (Level-0 belief) then adjusting for strategic depth
        utils_a = np.zeros(n, dtype=float)
        for i, r in enumerate(requests):
            base_pay = float(r)
            # Probability of getting the bonus: how likely is opponent
            # to request exactly r + bonus_gap?
            if bonus_rule == "exactly_one_less":
                bonus_target = r + bonus_gap
                if min_req <= bonus_target <= max_req:
                    # Level-0: uniform probability
                    p_bonus = 1.0 / n
                    # Adjust for strategic depth: higher depth → expect
                    # opponent to also reason strategically
                    p_bonus *= max(0.3, 1.0 - depth_a * 0.1)
                else:
                    p_bonus = 0.0
            elif bonus_rule == "any_lower":
                # Bonus if you're lower than opponent
                n_higher = sum(1 for rr in requests if rr > r)
                p_bonus = n_higher / n * max(0.3, 1.0 - depth_a * 0.1)
            else:
                p_bonus = 1.0 / n * 0.5

            expected = base_pay + p_bonus * bonus
            # Risk aversion: penalise variance
            variance = p_bonus * (1 - p_bonus) * bonus ** 2
            utils_a[i] = expected - risk_a * 0.01 * variance

        idx_a, probs_a = logit_choice(rng, utils_a, lam=lam_a)
        req_a = requests[idx_a]

        # --- Player B's decision ---
        if b is not None:
            lam_b = float(b.params.get("noise_lambda", 1.0))
            depth_b = float(b.params.get("strategic_depth", 1.0))
            risk_b = float(b.params.get("risk_aversion", 0.5))

            utils_b = np.zeros(n, dtype=float)
            for i, r in enumerate(requests):
                base_pay = float(r)
                if bonus_rule == "exactly_one_less":
                    bonus_target = r + bonus_gap
                    if min_req <= bonus_target <= max_req:
                        p_bonus = 1.0 / n * max(0.3, 1.0 - depth_b * 0.1)
                    else:
                        p_bonus = 0.0
                elif bonus_rule == "any_lower":
                    n_higher = sum(1 for rr in requests if rr > r)
                    p_bonus = n_higher / n * max(0.3, 1.0 - depth_b * 0.1)
                else:
                    p_bonus = 1.0 / n * 0.5
                expected = base_pay + p_bonus * bonus
                variance = p_bonus * (1 - p_bonus) * bonus ** 2
                utils_b[i] = expected - risk_b * 0.01 * variance

            idx_b, probs_b = logit_choice(rng, utils_b, lam=lam_b)
            req_b = requests[idx_b]
        else:
            # Solo version: request against a random opponent
            req_b = int(rng.choice(requests))
            probs_b = np.ones(n) / n

        # --- Compute payoffs ---
        pay_a = float(req_a)
        pay_b = float(req_b)

        if bonus_rule == "exactly_one_less":
            if req_a == req_b - bonus_gap:
                pay_a += bonus
            if b is not None and req_b == req_a - bonus_gap:
                pay_b += bonus
        elif bonus_rule == "any_lower":
            if req_a < req_b:
                pay_a += bonus
            elif req_b < req_a and b is not None:
                pay_b += bonus

        return GameOutcome(
            actions={"request_A": req_a, "request_B": req_b},
            payoffs={"A": pay_a, "B": pay_b},
            trace={
                "requests": requests,
                "utils_A": utils_a.tolist(),
                "probs_A": probs_a.tolist(),
                "probs_B": probs_b.tolist() if isinstance(probs_b, np.ndarray) else [],
                "bonus_rule": bonus_rule,
                "bonus": bonus,
            },
        )
