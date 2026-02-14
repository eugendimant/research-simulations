"""Parameterised family generator for Public Goods games.

Generates variants by combining:
  1. Group size (2-10 players)
  2. MPCR (Marginal Per-Capita Return)
  3. Punishment mechanism (none / peer / institutional)
  4. Rounds (1 / 5 / 10)
  5. Communication (none / cheap_talk / binding)
  6. Endowment equality (equal / unequal)

Grounded in meta-analytic findings:
  - Zelmer (2003): Meta-analysis of voluntary contributions, 27 studies
  - Chaudhuri (2011): Sustaining cooperation in laboratory PG experiments
  - Fehr & Gächter (2000): Cooperation and punishment in PG experiments

Each combination produces a canonical FamilySpec with a stable hash.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List

import numpy as np

from .money_request_family import FamilySpec


class PublicGoodsFamily:
    """Generate parameterised variants of the public goods game.

    Core variation dimensions:
      1. group_size: [2, 3, 4, 5, 8, 10]
      2. mpcr: [0.3, 0.4, 0.5, 0.6, 0.75]
      3. punishment: none, peer, institutional
      4. rounds: [1, 5, 10]
      5. communication: none, cheap_talk, binding
      6. endowment_equality: equal, unequal
    """

    DEFAULT_RANGES = {
        "group_size": [2, 3, 4, 5, 8, 10],
        "mpcr": [0.3, 0.4, 0.5, 0.6, 0.75],
        "punishment": ["none", "peer", "institutional"],
        "rounds": [1, 5, 10],
        "communication": ["none", "cheap_talk", "binding"],
        "endowment_equality": ["equal", "unequal"],
    }

    # Published effect sizes by condition (Cohen's d)
    # Source: Zelmer (2003), Chaudhuri (2011), Fehr & Gächter (2000)
    CONDITION_EFFECTS: Dict[str, float] = {
        "peer_punishment": 0.55,         # peer punishment increases contributions
        "institutional_punishment": 0.70, # institutional punishment even stronger
        "cheap_talk": 0.35,              # communication increases cooperation
        "binding": 0.60,                 # binding communication strongest
        "high_mpcr": 0.30,              # MPCR > 0.5 boosts contributions
        "large_group": -0.15,           # larger groups slightly less cooperative
        "unequal_endowment": -0.20,     # inequality reduces average contribution
        "repeated": 0.10,               # slight initial increase, then decline
    }

    def __init__(self, ranges: Dict[str, list] | None = None) -> None:
        self.ranges = ranges or self.DEFAULT_RANGES

    def generate_all(self) -> List[FamilySpec]:
        specs = []
        for combo in product(
            self.ranges["group_size"],
            self.ranges["mpcr"],
            self.ranges["punishment"],
            self.ranges["rounds"],
            self.ranges["communication"],
            self.ranges["endowment_equality"],
        ):
            gs, mpcr, pun, rounds, comm, eq = combo
            params = {
                "group_size": gs,
                "mpcr": mpcr,
                "punishment": pun,
                "rounds": rounds,
                "communication": comm,
                "endowment_equality": eq,
                "endowment": 20.0,  # standard endowment
            }
            specs.append(FamilySpec(game_name="public_goods", params=params))
        return specs

    def sample(self, rng: np.random.Generator, n: int) -> List[FamilySpec]:
        all_specs = self.generate_all()
        if n >= len(all_specs):
            return all_specs
        indices = rng.choice(len(all_specs), size=n, replace=False)
        return [all_specs[i] for i in indices]

    def generate_manifest(
        self, specs: List[FamilySpec], seed: int,
    ) -> Dict[str, Any]:
        return {
            "family": "public_goods",
            "n_specs": len(specs),
            "seed": seed,
            "specs": [s.to_dict() for s in specs],
            "condition_effects": self.CONDITION_EFFECTS,
        }
