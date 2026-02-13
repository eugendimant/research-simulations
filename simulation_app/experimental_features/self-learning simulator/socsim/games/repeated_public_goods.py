from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from .base import Game, GameOutcome
from ..persona import Persona

@dataclass
class RepeatedPublicGoods(Game):
    name: str = "repeated_public_goods"

    def simulate_group(self, rng: np.random.Generator, agents: List[Persona], params: Dict[str, Any]) -> Tuple[List[Dict[str, float]], Dict[str, float], Dict[str, Any]]:
        endowment = float(params.get("endowment", 10.0))
        multiplier = float(params.get("multiplier", 1.6))
        rounds = int(params.get("rounds", 10))
        n = len(agents)

        contrib_hist: List[List[float]] = []
        round_means: List[float] = []

        for r in range(rounds):
            contribs: List[float] = []
            for a in agents:
                base = float(a.params.get("conditional_coop", 0.0)) + float(a.params.get("prosociality", 0.0))
                drift = float(a.params.get("belief_drift", 0.0))
                x = base - drift * (r / max(1, rounds - 1))
                p = 1.0 / (1.0 + np.exp(-x))
                # soft contribution
                c = float(np.clip(endowment * p + rng.normal(0.0, 0.5), 0.0, endowment))
                contribs.append(c)
            contrib_hist.append(contribs)
            round_means.append(float(np.mean(contribs)))

        total_last = float(sum(contrib_hist[-1]))
        public_return_last = multiplier * total_last / float(n)
        pay_mean = float(np.mean([endowment - c + public_return_last for c in contrib_hist[-1]]))

        rows: List[Dict[str, float]] = []
        for i in range(n):
            rows.append({
                "act::contrib_r1": float(contrib_hist[0][i]),
                "act::contrib_rlast": float(contrib_hist[-1][i]),
                "act::contrib": float(np.mean([contrib_hist[t][i] for t in range(rounds)])),
            })

        traces = {"round_means": round_means}
        return rows, {"mean_payoff": pay_mean}, traces

    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona, params: Dict[str, Any]) -> GameOutcome:
        return GameOutcome(actions={}, payoffs={})

# Backward-compatible alias
RepeatedPublicGoodsGame = RepeatedPublicGoods
