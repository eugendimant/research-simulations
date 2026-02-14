"""Level-k reasoning strategy.

Implements strategic depth thinking:
  Level-0: uniformly random
  Level-1: best-responds to Level-0 (assumes opponents random)
  Level-2: best-responds to Level-1
  Level-k: best-responds to Level-(k-1)

The persona's ``strategic_depth`` parameter (0-3, continuous) determines
the mixture over reasoning levels.

References:
  Stahl & Wilson (1994), Nagel (1995), Costa-Gomes & Crawford (2006)
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class LevelKStrategy(Strategy):
    name = "level_k"

    def action_probabilities(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        actions = state.available_actions
        if not actions:
            return {}

        k = float(persona.params.get("strategic_depth", 1.0))
        k = max(0.0, min(k, 4.0))

        n = len(actions)
        action_names = [a.name for a in actions]

        # Level-0: uniform
        level0 = np.ones(n) / n

        if k < 0.5:
            # Pure Level-0
            return self._normalise(dict(zip(action_names, level0.tolist())))

        # For games with numeric actions (beauty contest, dictator, etc.)
        # Level-k best-responds by predicting what level-(k-1) would do
        vals = self._action_values(state)
        has_numeric = any(v != 0.0 for v in vals)

        if has_numeric and state.game_name in ("beauty_contest", "coordination_min_effort"):
            # Beauty contest: Level-k guesses (2/3)^k * mean_of_range
            p = float(state.game_params.get("p_fraction", 2.0 / 3.0))
            max_val = max(vals) if vals else 100.0
            level0_mean = max_val / 2.0  # Level-0 mean guess
            target = level0_mean * (p ** k)
            # Concentrate probability around target
            probs = np.array([math.exp(-((v - target) ** 2) / (max_val * 0.1 + 1e-9)) for v in vals])
        else:
            # For other games: interpolate between uniform and concentrated
            # Higher k → more concentrated on "rational" actions
            noise_lambda = float(persona.params.get("noise_lambda", 1.0))
            lam = noise_lambda * (1 + k)
            utilities = np.zeros(n, dtype=float)
            # Use prosociality as a utility proxy
            prosoc = float(persona.params.get("prosociality", 0.0))
            for i, v in enumerate(vals):
                utilities[i] = prosoc * v
            z = lam * (utilities - np.max(utilities))
            probs = np.exp(z)

        probs = probs / (probs.sum() + 1e-12)
        return self._normalise(dict(zip(action_names, probs.tolist())))
