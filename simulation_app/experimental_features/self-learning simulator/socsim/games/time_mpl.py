from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice

def u(x: float) -> float:
    # linear utility placeholder (you can swap in CRRA or prospect theory later)
    return float(x)

class TimeMPL(Game):
    name = "time_mpl"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        # rows: each is {sooner: (amount, delay), later: (amount, delay)}
        rows = spec.get("rows", [])
        if not rows:
            rows = [
                {"sooner": [10, 0], "later": [11, 7]},
                {"sooner": [10, 0], "later": [12, 14]},
                {"sooner": [10, 0], "later": [14, 30]},
            ]
        beta = float(a.params.get("present_bias_beta", 1.0))   # quasi-hyperbolic present bias
        delta = float(a.params.get("discount_delta", 0.99))    # exponential factor
        lam = float(a.params["noise_lambda"])

        choices, traces = [], []
        realized_utils = []
        for row in rows:
            x_s, t_s = map(float, row["sooner"])
            x_l, t_l = map(float, row["later"])

            disc_s = (beta if t_s > 0 else 1.0) * (delta ** t_s)
            disc_l = (beta if t_l > 0 else 1.0) * (delta ** t_l)
            EU_s = disc_s * u(x_s)
            EU_l = disc_l * u(x_l)

            uvec = np.array([EU_s, EU_l], dtype=float)
            idx, probs = logit_choice(rng, uvec, lam=lam)
            ch = "sooner" if idx == 0 else "later"
            choices.append(ch)
            traces.append({"EU_sooner": float(EU_s), "EU_later": float(EU_l), "probs": probs.tolist(), "row": row})
            realized_utils.append(float(uvec[idx]))

        return GameOutcome(
            actions={"choices": choices, "later_share": float(sum(1 for c in choices if c=="later")/len(choices))},
            payoffs={"A": float(sum(realized_utils)/len(realized_utils))},
            trace={"rows": traces, "beta": beta, "delta": delta},
        )
