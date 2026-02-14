"""Parameterised family generator for the Dictator game.

Generates variants by combining:
  1. Endowment size
  2. Recipient visibility (anonymous / observed / peer-observed)
  3. Framing (give / take / invest)
  4. Social distance (stranger / acquaintance / ingroup / outgroup)
  5. Dictator role framing (earned / windfall)

Grounded in meta-analytic findings:
  - Engel (2011): Dictator game meta-analysis, N=616 studies
  - List (2007): On the interpretation of giving in dictator games
  - Bardsley (2008): Dictator game giving: altruism or artefact?

Each combination produces a canonical FamilySpec with a stable hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List

import numpy as np

from .money_request_family import FamilySpec


class DictatorFamily:
    """Generate parameterised variants of the dictator game.

    Core variation dimensions:
      1. endowment: [5, 10, 20, 50, 100]
      2. visibility: anonymous, observed, peer_observed
      3. framing: give, take, invest
      4. social_distance: stranger, acquaintance, ingroup, outgroup
      5. endowment_source: windfall, earned
    """

    DEFAULT_RANGES = {
        "endowment": [5.0, 10.0, 20.0, 50.0, 100.0],
        "visibility": ["anonymous", "observed", "peer_observed"],
        "framing": ["give", "take", "invest"],
        "social_distance": ["stranger", "acquaintance", "ingroup", "outgroup"],
        "endowment_source": ["windfall", "earned"],
    }

    # Published effect sizes by condition (Cohen's d relative to baseline)
    # Source: Engel (2011) meta-analysis
    CONDITION_EFFECTS: Dict[str, float] = {
        "observed": 0.35,       # observation increases giving ~35%
        "peer_observed": 0.45,  # peer observation stronger than experimenter
        "take": -0.25,          # take frame reduces giving (List 2007)
        "invest": 0.15,         # invest frame slightly increases allocation
        "ingroup": 0.32,        # Balliet et al. (2014)
        "outgroup": -0.28,      # Iyengar & Westwood (2015) for political
        "earned": -0.20,        # earned endowments reduce giving (Cherry 2002)
    }

    def __init__(self, ranges: Dict[str, list] | None = None) -> None:
        self.ranges = ranges or self.DEFAULT_RANGES

    def generate_all(self) -> List[FamilySpec]:
        """Generate all possible combinations."""
        specs = []
        for combo in product(
            self.ranges["endowment"],
            self.ranges["visibility"],
            self.ranges["framing"],
            self.ranges["social_distance"],
            self.ranges["endowment_source"],
        ):
            endow, vis, frame, social, source = combo
            params = {
                "endowment": endow,
                "visibility": vis,
                "framing": frame,
                "social_distance": social,
                "endowment_source": source,
            }
            specs.append(FamilySpec(game_name="dictator", params=params))
        return specs

    def sample(self, rng: np.random.Generator, n: int) -> List[FamilySpec]:
        """Sample n unique specs from the family."""
        all_specs = self.generate_all()
        if n >= len(all_specs):
            return all_specs
        indices = rng.choice(len(all_specs), size=n, replace=False)
        return [all_specs[i] for i in indices]

    def generate_manifest(
        self, specs: List[FamilySpec], seed: int,
    ) -> Dict[str, Any]:
        return {
            "family": "dictator",
            "n_specs": len(specs),
            "seed": seed,
            "specs": [s.to_dict() for s in specs],
            "condition_effects": self.CONDITION_EFFECTS,
        }
