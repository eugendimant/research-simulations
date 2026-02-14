"""Inequity Aversion strategy.

Implements Fehr & Schmidt (1999):
  - Agents experience disutility from advantageous and disadvantageous
    inequity between themselves and others
  - α: sensitivity to disadvantageous inequity (other earns more)
  - β: sensitivity to advantageous inequity (self earns more)
  - Typically α > β (people dislike being behind more than being ahead)

Also incorporates Bolton & Ockenfels (2000) ERC model as an option:
  - Agents care about relative share rather than absolute differences

References:
  Fehr & Schmidt (1999). A Theory of Fairness, Competition, and Cooperation.
  QJE 114(3):817-868.
  Bolton & Ockenfels (2000). ERC: A Theory of Equity, Reciprocity, and Competition.
  AER 90(1):166-193.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState
from ..social_prefs import fehr_schmidt_utility


class InequityAversionStrategy(Strategy):
    name = "inequity_aversion"

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
        vals = self._action_values(state)

        alpha = float(persona.params.get("fairness_alpha", 0.8))
        beta = float(persona.params.get("fairness_beta", 0.2))
        lam = float(persona.params.get("noise_lambda", 1.0))
        endow = float(state.game_params.get("endowment", 10.0))

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            pi_other = v
            utilities[i] = fehr_schmidt_utility(pi_self, pi_other, alpha, beta)

        # Softmax with noise
        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
