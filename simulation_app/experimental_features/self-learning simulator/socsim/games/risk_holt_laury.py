from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any, List
from .base import Game, GameOutcome
from ..decision import logit_choice

class HoltLauryRisk(Game):
    """Holt-Laury multiple price list risk task (metadata-agnostic simulation).

    Output:
      - choices: list of 'A'/'B' per row
      - switch_row: first row where 'B' chosen (11 if never switches)
    """
    name = "risk_holt_laury"

    def simulate_one(self, rng, a, b=None, params: Optional[Dict[str, Any]] = None) -> GameOutcome:
        p = params or {}
        rows = p.get("rows", None)
        if rows is None:
            probs = [0.1*i for i in range(1,11)]
            rows = []
            for pr in probs:
                rows.append({"p_high": pr, "A_high": 2.0, "A_low": 1.6, "B_high": 3.85, "B_low": 0.1})

        beta = float(a.params.get("choice_precision", 2.0))
        ra = float(a.params.get("risk_aversion", 0.0))

        def u(x: float) -> float:
            x = max(1e-9, float(x))
            if abs(ra-1.0) < 1e-6:
                return float(np.log(x))
            return float((x**(1.0-ra)) / (1.0-ra))

        choices: List[str] = []
        EU_A: List[float] = []
        EU_B: List[float] = []

        for r in rows:
            ph = float(r["p_high"])
            Ah, Al = float(r["A_high"]), float(r["A_low"])
            Bh, Bl = float(r["B_high"]), float(r["B_low"])
            euA = ph*u(Ah) + (1-ph)*u(Al)
            euB = ph*u(Bh) + (1-ph)*u(Bl)
            idx = logit_choice(rng, np.array([euA, euB], dtype=float), beta=beta)
            choices.append("A" if idx == 0 else "B")
            EU_A.append(float(euA)); EU_B.append(float(euB))

        switch = None
        for i, c in enumerate(choices, start=1):
            if c == "B":
                switch = i
                break

        return GameOutcome(
            actions={"choices": choices, "switch_row": switch if switch is not None else 11},
            payoffs={"A": float("nan")},
            trace={"EU_A": EU_A, "EU_B": EU_B},
        )
