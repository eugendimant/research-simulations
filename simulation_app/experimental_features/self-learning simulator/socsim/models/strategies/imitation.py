"""Imitation / Social Learning strategy.

Implements observational conformity:
  - Agents copy the most common (modal) action from observed history
  - Bandwagon effect: probability of choosing an action scales with its frequency
  - Granovetter threshold model: agents switch when enough others have

Parameters from persona:
  - conformity: how strongly to imitate the majority (0=independent, 1=full conformist)
  - noise_lambda: rationality in interpreting observed frequencies

References:
  Bandura (1977). Social Learning Theory. Prentice Hall.
  Cialdini (2005). Basic Social Influence is Underestimated. Psych Inquiry 16(4).
  Granovetter (1978). Threshold Models of Collective Behavior. AJS 83(6).
  Fischbacher et al. (2001). Are People Conditionally Cooperative? Econ Letters 71.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState

logger = logging.getLogger(__name__)


class ImitationStrategy(Strategy):
    """Imitate the most popular action from observed peer behavior."""

    name = "imitation"

    def action_probabilities(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        actions = state.available_actions
        if not actions:
            return {}

        n = len(actions)
        action_names = [a.name for a in actions]
        conformity = float(persona.params.get("conformity", 0.6))
        lam = float(persona.params.get("noise_lambda", 1.0))

        # If no history, fall back to uniform with slight prosocial bias
        if not state.history:
            prosoc = float(persona.params.get("prosociality", 0.0))
            vals = self._action_values(state)
            if any(v != 0.0 for v in vals):
                utils = np.array([prosoc * v for v in vals])
                z = 0.3 * (utils - np.max(utils))
                probs = np.exp(z)
                probs /= probs.sum() + 1e-12
                return self._normalise(dict(zip(action_names, probs.tolist())))
            return {nm: 1.0 / n for nm in action_names}

        # Count observed action frequencies from history
        freq = np.zeros(n, dtype=float)
        for round_data in state.history:
            obs_action = round_data.get("observed_action", round_data.get("opponent_action"))
            if obs_action is not None:
                for i, a in enumerate(actions):
                    if a.name == obs_action or a.value == obs_action:
                        freq[i] += 1.0
                        break

        total_obs = freq.sum()
        if total_obs == 0:
            # No observed actions in history, use uniform
            return {nm: 1.0 / n for nm in action_names}

        # Normalize to empirical frequencies
        empirical = freq / total_obs

        # Blend: conformity * empirical + (1-conformity) * uniform
        uniform = np.ones(n) / n
        blended = conformity * empirical + (1 - conformity) * uniform

        # Apply noise via softmax
        z = lam * (blended - np.max(blended))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
