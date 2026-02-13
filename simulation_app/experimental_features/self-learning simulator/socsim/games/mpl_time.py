from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .base import Game, GameOutcome
from ._choice import logit_choice

class MPLTime(Game):
    """Multiple price list time preference elicitation.
    Each row: choose smaller-sooner (SS) vs larger-later (LL).
    Outputs switch row.
    """
    name = "mpl_time"

    def simulate_one(self, rng, a, b=None, params: Optional[Dict] = None) -> GameOutcome:
        p = params or {}
        rows = p.get("rows", None)
        if rows is None:
            # Example: $10 today vs $X in 30 days
            xs = [10,11,12,13,14,15,16,18,20,22]
            rows = [{"SS_amt": 10.0, "SS_days": 0, "LL_amt": float(x), "LL_days": 30} for x in xs]

        beta = float(a.params.get("choice_precision", 2.0))
        delta = float(a.params.get("discount_factor", 0.98))  # per day
        present_bias = float(a.params.get("present_bias", 1.0))  # beta in (0,1], 1=no bias
        present_bias = min(max(present_bias, 0.1), 1.0)

        def disc(days: int) -> float:
            d = max(0, int(days))
            if d==0:
                return 1.0
            return present_bias*(delta**d)

        choices = []
        Uss, Ull = [], []
        for r in rows:
            ss_amt = float(r["SS_amt"]); ss_d = int(r["SS_days"])
            ll_amt = float(r["LL_amt"]); ll_d = int(r["LL_days"])
            EU_SS = disc(ss_d)*ss_amt
            EU_LL = disc(ll_d)*ll_amt
            idx = logit_choice(rng, np.array([EU_SS, EU_LL], dtype=float), beta=beta)
            choices.append("SS" if idx==0 else "LL")
            Uss.append(float(EU_SS)); Ull.append(float(EU_LL))

        switch = None
        for i,c in enumerate(choices, start=1):
            if c=="LL":
                switch = i
                break

        return GameOutcome(
            actions={"choices": choices, "switch_row": switch if switch is not None else 11},
            payoffs={"A": float("nan")},
            trace={"U_SS": Uss, "U_LL": Ull}
        )
