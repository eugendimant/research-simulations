"""Base class for all offline strategies.

Includes empirical effect size calibration tables from meta-analyses
to ground simulation output in published research.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ...persona import Persona
from ...agents.backend import GameState


# ── Empirical effect size calibration ────────────────────────────────────
# Maps game type → published meta-analytic statistics.
# Used by strategies to scale outputs to realistic ranges.
#
# Sources:
#   Engel (2011): Dictator game meta-analysis — mean giving ~28%, SD ~22%
#   Johnson & Mislin (2011): Trust game meta — mean trust ~50%, SD ~33%
#   Zelmer (2003): Public goods meta — mean contribution ~40%, SD ~16%
#   Oosterbeek et al. (2004): Ultimatum meta — mean offer ~40%, SD ~12%
#   Frey et al. (2011): Dictator condition effects d ≈ 0.3-0.6
#   Balliet et al. (2014): Ingroup cooperation meta d ≈ 0.32
#   Dimant (2024): Political intergroup d ≈ 0.6-0.9

EMPIRICAL_CALIBRATION: Dict[str, Dict[str, float]] = {
    "dictator": {
        "mean_share": 0.28, "sd_share": 0.22,
        "condition_d": 0.45, "ingroup_d": 0.32,
    },
    "trust": {
        "mean_share": 0.50, "sd_share": 0.33,
        "condition_d": 0.40, "ingroup_d": 0.35,
    },
    "ultimatum": {
        "mean_offer": 0.40, "sd_offer": 0.12,
        "condition_d": 0.25, "rejection_rate": 0.16,
    },
    "public_goods": {
        "mean_contribution": 0.40, "sd_contribution": 0.16,
        "condition_d": 0.40, "framing_d": 0.40,
    },
    "prisoners_dilemma": {
        "mean_cooperation": 0.47, "sd_cooperation": 0.25,
        "condition_d": 0.35,
    },
    "beauty_contest": {
        "mean_guess_fraction": 0.45, "sd_guess_fraction": 0.20,
    },
    "money_request_11_20": {
        "modal_request": 17.0, "mean_request": 16.2, "sd_request": 2.5,
    },
    "common_pool_resource": {
        "mean_extraction": 0.55, "sd_extraction": 0.18,
    },
}

# Strategy-game validated matches (which strategies are empirically
# validated for which game types)
STRATEGY_VALIDATED_GAMES: Dict[str, List[str]] = {
    "level_k": ["beauty_contest", "coordination_min_effort", "matching_pennies"],
    "cognitive_hierarchy": ["beauty_contest", "coordination_min_effort", "matching_pennies"],
    "qre": ["matching_pennies", "coordination", "ultimatum", "trust"],
    "inequity_aversion": ["dictator", "ultimatum", "trust", "public_goods"],
    "reciprocity": ["trust", "public_goods", "prisoners_dilemma"],
    "reputation_sensitive": ["dictator", "trust", "public_goods"],
    "norm_sensitive": ["dictator", "ultimatum", "coordination"],
    "rule_following": ["dictator", "trust", "common_pool_resource"],
    "noisy_best_response": ["dictator", "trust", "ultimatum", "public_goods"],
    "rl_bandit": ["prisoners_dilemma", "public_goods", "coordination"],
}


class Strategy(ABC):
    """Abstract base for a strategic decision module.

    Subclasses implement ``action_probabilities`` which returns a dict
    mapping action names to probabilities.  These MUST sum to 1.0 and
    be non-negative.
    """

    name: str = "base"

    @abstractmethod
    def action_probabilities(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        """Return {action_name: probability} dict.  Must sum to 1."""
        ...

    def _normalise(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Safety normalisation to ensure valid distribution."""
        total = sum(max(0.0, v) for v in dist.values())
        if total <= 0:
            n = len(dist) or 1
            return {k: 1.0 / n for k in dist}
        return {k: max(0.0, v) / total for k, v in dist.items()}

    def _action_values(self, state: GameState) -> List[float]:
        """Extract numeric values from available actions."""
        vals: List[float] = []
        for a in state.available_actions:
            if isinstance(a.value, (int, float)):
                vals.append(float(a.value))
            else:
                vals.append(0.0)
        return vals

    def _get_calibration(self, game_name: str) -> Dict[str, float]:
        """Return empirical calibration for the game, or empty dict."""
        return EMPIRICAL_CALIBRATION.get(game_name, {})
