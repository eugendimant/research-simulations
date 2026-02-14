"""Parameterised family generator for the Money Request game.

Generates a large space of game variants by combining:
  1. Request range (min, max)
  2. Bonus size
  3. Bonus gap
  4. Bonus rule type
  5. Number of players
  6. Information structure

Each combination produces a canonical GameSpec with a stable hash
for reproducible evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FamilySpec:
    """A single game specification within a family."""
    game_name: str
    params: Dict[str, Any]
    family_id: str = ""
    spec_hash: str = ""

    def __post_init__(self) -> None:
        if not self.spec_hash:
            self.spec_hash = self._compute_hash()
        if not self.family_id:
            self.family_id = f"{self.game_name}_{self.spec_hash[:8]}"

    def _compute_hash(self) -> str:
        canonical = json.dumps(
            {"game": self.game_name, "params": self.params},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_name": self.game_name,
            "params": self.params,
            "family_id": self.family_id,
            "spec_hash": self.spec_hash,
        }


class MoneyRequestFamily:
    """Generate parameterised variants of the 11-20 money request game.

    The 6 core variation components:
      1. min_request: [5, 8, 11, 15]
      2. max_request range: [5, 10, 15, 20]  (added to min)
      3. bonus: [5, 10, 20, 50]
      4. bonus_gap: [1, 2, 3]
      5. bonus_rule: ["exactly_one_less", "any_lower"]
      6. info: ["symmetric", "asymmetric"]
    """

    # Default variation ranges
    DEFAULT_RANGES = {
        "min_request": [5, 8, 11, 15],
        "range_size": [5, 10, 15, 20],
        "bonus": [5.0, 10.0, 20.0, 50.0],
        "bonus_gap": [1, 2, 3],
        "bonus_rule": ["exactly_one_less", "any_lower"],
        "info_structure": ["symmetric", "asymmetric"],
    }

    def __init__(
        self,
        ranges: Dict[str, list] | None = None,
    ) -> None:
        self.ranges = ranges or self.DEFAULT_RANGES

    def generate_all(self) -> List[FamilySpec]:
        """Generate all possible combinations."""
        specs = []
        for combo in product(
            self.ranges["min_request"],
            self.ranges["range_size"],
            self.ranges["bonus"],
            self.ranges["bonus_gap"],
            self.ranges["bonus_rule"],
            self.ranges["info_structure"],
        ):
            min_req, range_sz, bonus, gap, rule, info = combo
            max_req = min_req + range_sz

            params = {
                "min_request": min_req,
                "max_request": max_req,
                "bonus": bonus,
                "bonus_gap": gap,
                "bonus_rule": rule,
                "info_structure": info,
            }
            specs.append(FamilySpec(game_name="money_request_11_20", params=params))

        return specs

    def sample(self, rng: np.random.Generator, n: int) -> List[FamilySpec]:
        """Sample n unique specs from the family."""
        all_specs = self.generate_all()
        if n >= len(all_specs):
            return all_specs
        indices = rng.choice(len(all_specs), size=n, replace=False)
        return [all_specs[i] for i in indices]

    def generate_manifest(
        self,
        specs: List[FamilySpec],
        seed: int,
    ) -> Dict[str, Any]:
        """Create a manifest for a set of specs."""
        return {
            "family": "money_request_11_20",
            "n_specs": len(specs),
            "seed": seed,
            "specs": [s.to_dict() for s in specs],
            "hashes": [s.spec_hash for s in specs],
        }
