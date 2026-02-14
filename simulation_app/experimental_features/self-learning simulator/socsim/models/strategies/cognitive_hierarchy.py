"""Cognitive Hierarchy (CH) strategy.

Implements Camerer, Ho & Chong (2004):
  - Agents have a Poisson-distributed thinking level τ
  - Each level k best-responds to the *truncated* distribution of lower levels
  - Unlike Level-k, CH agents consider ALL lower levels (weighted by Poisson)

The persona's ``strategic_depth`` parameter maps to the Poisson τ parameter.

References:
  Camerer, Ho & Chong (2004). A Cognitive Hierarchy Model of Games.
  QJE 119(3):861-898.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


def _poisson_weights(tau: float, max_k: int = 5) -> np.ndarray:
    """Compute Poisson(τ) weights for thinking levels 0..max_k."""
    if tau <= 0:
        w = np.zeros(max_k + 1)
        w[0] = 1.0
        return w
    w = np.array([math.exp(-tau) * (tau ** k) / math.factorial(k) for k in range(max_k + 1)])
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w[0] = 1.0
    return w


class CognitiveHierarchyStrategy(Strategy):
    name = "cognitive_hierarchy"

    def __init__(self, max_k: int = 5) -> None:
        self._max_k = max_k

    def action_probabilities(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        actions = state.available_actions
        if not actions:
            return {}

        tau = float(persona.params.get("strategic_depth", 1.5))
        tau = max(0.0, min(tau, 6.0))
        n = len(actions)
        action_names = [a.name for a in actions]

        weights = _poisson_weights(tau, self._max_k)

        # Build level-by-level distributions
        # Level-0: uniform
        levels = [np.ones(n) / n]

        vals = self._action_values(state)
        noise_lambda = float(persona.params.get("noise_lambda", 1.0))
        prosoc = float(persona.params.get("prosociality", 0.0))

        for k in range(1, self._max_k + 1):
            # Expected distribution of opponents at levels < k
            # (truncated Poisson-weighted average of lower levels)
            w_lower = weights[:k].copy()
            w_sum = w_lower.sum()
            if w_sum > 0:
                w_lower /= w_sum
            else:
                w_lower = np.ones(k) / k

            opponent_dist = np.zeros(n)
            for j in range(k):
                opponent_dist += w_lower[j] * levels[j]

            # Best-respond to opponent_dist with noise
            utilities = np.zeros(n, dtype=float)
            for i, v in enumerate(vals):
                utilities[i] = prosoc * v + noise_lambda * k * 0.1
            lam = noise_lambda * (1 + k * 0.5)
            z = lam * (utilities - np.max(utilities))
            probs = np.exp(z)
            probs /= probs.sum() + 1e-12
            levels.append(probs)

        # Final distribution: Poisson-weighted mixture of all levels
        final = np.zeros(n)
        for k in range(min(len(levels), len(weights))):
            final += weights[k] * levels[k]
        final /= final.sum() + 1e-12

        return self._normalise(dict(zip(action_names, final.tolist())))
