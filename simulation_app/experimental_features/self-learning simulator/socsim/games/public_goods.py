from __future__ import annotations
import numpy as np
from typing import Dict, Optional, List, Any, Tuple

from .base import Game, GameOutcome
from ..decision import logit_choice

class PublicGoodsGame(Game):
    name = "public_goods"

    def simulate_one(self, rng, a, b=None, params: Optional[Dict[str, Any]] = None) -> GameOutcome:
        p = params or {}
        n = int(p.get("group_size", 4))
        endowment = float(p.get("endowment", 20.0))
        mpcr = float(p.get("mpcr", 0.4))
        grid = p.get("grid", [0, 5, 10, 15, 20])
        grid = np.array([float(x) for x in grid], dtype=float)

        beta = float(a.params.get("choice_precision", 2.0))
        cond = float(a.params.get("conditional_coop", 0.0))
        pros = float(a.params.get("prosociality", 0.0))
        belief_others = float(a.params.get("belief_others_contrib", endowment/4))
        belief_others = float(np.clip(belief_others, 0.0, endowment))

        c = grid
        total = c + (n - 1) * belief_others
        payoff = (endowment - c) + mpcr * total

        # Utility: own payoff + pros*(average group payoff) - cond*|c - belief_others|
        group_total_pay = n * endowment - total + n * mpcr * total
        util = payoff + pros * (group_total_pay / n) - cond * np.abs(c - belief_others)

        idx = logit_choice(rng, util, beta=beta)
        contrib = float(c[idx])

        realized_total = contrib + (n - 1) * belief_others
        realized_pay = (endowment - contrib) + mpcr * realized_total

        return GameOutcome(
            actions={"contribute": contrib},
            payoffs={"A": float(realized_pay)},
            trace={"grid": c.tolist(), "belief_others": float(belief_others), "util": util.tolist()}
        )

    def simulate_group(self, rng, personas: List[Any], params: Optional[Dict[str, Any]] = None) -> Tuple[List[float], List[float], Dict[str, Any]]:
        """Simulate a full group (one-shot), returning contributions and payoffs per player.

        This is intentionally simple: each player i forms a belief about others based on their own
        `belief_others_contrib` parameter, then chooses via the same utility used in simulate_one.
        """
        p = params or {}
        n = len(personas)
        endowment = float(p.get("endowment", 20.0))
        mpcr = float(p.get("mpcr", 0.4))
        grid = p.get("grid", [0, 5, 10, 15, 20])
        grid = np.array([float(x) for x in grid], dtype=float)

        contribs: List[float] = []
        utils: List[List[float]] = []
        beliefs: List[float] = []

        for i, person in enumerate(personas):
            beta = float(person.params.get("choice_precision", 2.0))
            cond = float(person.params.get("conditional_coop", 0.0))
            pros = float(person.params.get("prosociality", 0.0))
            belief_others = float(person.params.get("belief_others_contrib", endowment/4))
            belief_others = float(np.clip(belief_others, 0.0, endowment))
            beliefs.append(belief_others)

            c = grid
            total = c + (n - 1) * belief_others
            payoff = (endowment - c) + mpcr * total
            group_total_pay = n * endowment - total + n * mpcr * total
            util = payoff + pros * (group_total_pay / n) - cond * np.abs(c - belief_others)

            idx = logit_choice(rng, util, beta=beta)
            contribs.append(float(c[idx]))
            utils.append([float(x) for x in util.tolist()])

        total_contrib = float(sum(contribs))
        payoffs = [(endowment - c) + mpcr * total_contrib for c in contribs]

        trace = {"beliefs": beliefs, "utils": utils, "total_contribution": total_contrib}
        return contribs, [float(x) for x in payoffs], trace
