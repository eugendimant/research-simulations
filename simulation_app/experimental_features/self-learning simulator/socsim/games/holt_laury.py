from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .base import Game, GameOutcome
from ._choice import logit_choice

class HoltLauryRisk(Game):
    """Multiple price list risk elicitation (Holt-Laury style).
    Outputs only the switch row or chosen option per row.
    This is a stylized implementation intended for simulation and calibration scaffolding.
    """
    name = "holt_laury"

    def simulate_one(self, rng, a, b=None, params: Optional[Dict] = None) -> GameOutcome:
        p = params or {}
        # rows: list of dicts with p_high, A_high, A_low, B_high, B_low
        rows = p.get("rows", None)
        if rows is None:
            # canonical 10-row increasing probabilities
            probs = [0.1*i for i in range(1,11)]
            rows = []
            for pr in probs:
                rows.append({"p_high": pr, "A_high": 2.0, "A_low": 1.6, "B_high": 3.85, "B_low": 0.1})

        beta = float(a.params.get("choice_precision", 2.0))
        ra = float(a.params.get("risk_aversion", 0.0))
        # CRRA utility for x>0; if ra ~ 1, use log
        def u(x: float) -> float:
            x = max(1e-9, float(x))
            if abs(ra-1.0) < 1e-6:
                return float(np.log(x))
            return float((x**(1.0-ra)) / (1.0-ra))

        choices = []
        utils_A = []
        utils_B = []
        for r in rows:
            ph = float(r["p_high"])
            Ah, Al = float(r["A_high"]), float(r["A_low"])
            Bh, Bl = float(r["B_high"]), float(r["B_low"])
            EU_A = ph*u(Ah) + (1-ph)*u(Al)
            EU_B = ph*u(Bh) + (1-ph)*u(Bl)
            idx = logit_choice(rng, np.array([EU_A, EU_B], dtype=float), beta=beta)
            choices.append("A" if idx==0 else "B")
            utils_A.append(float(EU_A)); utils_B.append(float(EU_B))

        # switch row: first B after As
        switch = None
        for i,c in enumerate(choices, start=1):
            if c=="B":
                switch = i
                break

        return GameOutcome(
            actions={"choices": choices, "switch_row": switch if switch is not None else 11},
            payoffs={"A": float("nan")},
            trace={"EU_A": utils_A, "EU_B": utils_B}
        )
