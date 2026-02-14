"""Norm-sensitive strategy.

Agents follow injunctive and descriptive norms:
  - Injunctive norms: what people think *should* be done (fairness norms)
  - Descriptive norms: what people *actually* do (empirical behaviour)
  - Norm sensitivity varies by persona (norm_weight parameter)

Implements Krupka & Weber (2013) norm framework:
  - Each action has a "social appropriateness" rating
  - Agents trade off material payoff against norm compliance

References:
  Krupka & Weber (2013). Identifying Social Norms Using Coordination Games.
  J Economic Psychology 36:187-199.
  Bicchieri (2006). The Grammar of Society. Cambridge UP.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class NormSensitiveStrategy(Strategy):
    name = "norm_sensitive"

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

        lam = float(persona.params.get("noise_lambda", 1.0))
        norm_weight = float(persona.params.get("norm_weight", 0.5))
        prosoc = float(persona.params.get("prosociality", 0.0))
        endow = float(state.game_params.get("endowment", 10.0))

        # Injunctive norm: "fair share" target (Krupka & Weber 2013)
        norm_share = float(state.game_params.get("norm_target_share", 0.5))

        # Descriptive norm: what others typically do (from context)
        desc_mean = float(state.context.get("descriptive_norm_mean", norm_share))

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            share = v / (endow + 1e-9)

            # Material payoff
            u_material = pi_self

            # Injunctive norm: quadratic cost of deviation
            u_injunctive = -norm_weight * (share - norm_share) ** 2

            # Descriptive norm: pull toward empirical behaviour
            u_descriptive = -norm_weight * 0.5 * (share - desc_mean) ** 2

            # Prosocial component
            u_prosoc = prosoc * v * 0.1

            utilities[i] = u_material + u_injunctive + u_descriptive + u_prosoc

        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
