"""Reciprocity strategy.

Implements Rabin (1993) and Dufwenberg & Kirchsteiger (2004):
  - Agents form beliefs about opponents' intentions
  - Kind intentions → reciprocate with kindness
  - Unkind intentions → reciprocate with punishment
  - Reciprocity strength is modulated by persona's ``reciprocity`` parameter

Also supports conditional cooperation (Fischbacher et al., 2001):
  - In public goods games, cooperate proportionally to expected others' contributions

References:
  Rabin (1993). Incorporating Fairness into Game Theory and Economics. AER 83(5).
  Dufwenberg & Kirchsteiger (2004). A Theory of Sequential Reciprocity. GEB 47(2).
  Fischbacher, Gächter & Fehr (2001). Are People Conditionally Cooperative?
  Economics Letters 71(3):397-404.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class ReciprocityStrategy(Strategy):
    name = "reciprocity"

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

        recip = float(persona.params.get("reciprocity", 0.5))
        lam = float(persona.params.get("noise_lambda", 1.0))
        prosoc = float(persona.params.get("prosociality", 0.0))
        endow = float(state.game_params.get("endowment", 10.0))

        # Assess opponent's kindness from history
        kindness = 0.0
        if state.history:
            last = state.history[-1]
            opp_action = float(last.get("opponent_action", 0.0))
            opp_max = float(last.get("opponent_max_action", endow))
            if opp_max > 0:
                kindness = (opp_action / opp_max) - 0.5  # [-0.5, +0.5]
        else:
            # Prior belief: slight positive expectation
            kindness = 0.1

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            pi_other = v
            share = v / (endow + 1e-9)

            # Base material utility
            u_material = pi_self

            # Reciprocity: reward/punish based on perceived kindness
            u_recip = recip * kindness * share

            # Prosociality
            u_prosoc = prosoc * pi_other * 0.1

            utilities[i] = u_material + u_recip + u_prosoc

        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
