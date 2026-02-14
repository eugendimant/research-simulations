"""Reputation-sensitive strategy.

Agents care about how their actions affect their reputation, especially
in repeated or observed interactions.

Implements reputation-concern models:
  - Benabou & Tirole (2006): image motivation in prosocial behaviour
  - Andreoni & Bernheim (2009): social image in dictator games
  - Agents trade off material payoff against reputational benefit

References:
  Benabou & Tirole (2006). Incentives and Prosocial Behavior. AER 96(5).
  Andreoni & Bernheim (2009). Social Image and the 50-50 Norm. Econometrica 77(5).
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class ReputationSensitiveStrategy(Strategy):
    name = "reputation_sensitive"

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
        prosoc = float(persona.params.get("prosociality", 0.0))
        rep_concern = float(persona.params.get("reputation_concern", 0.5))
        endow = float(state.game_params.get("endowment", 10.0))
        is_observed = bool(state.context.get("observed", True))

        # Social norm: typically 50/50 in dictator-type games
        norm_share = float(state.game_params.get("norm_target_share", 0.5))

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            pi_other = v
            share = v / (endow + 1e-9)

            # Material utility
            u_material = pi_self

            # Prosocial utility
            u_prosoc = prosoc * pi_other * 0.1

            # Reputation: Benabou-Tirole image motivation
            # Deviation from norm reduces reputation
            if is_observed:
                u_rep = -rep_concern * (share - norm_share) ** 2
            else:
                # Anonymity reduces reputation concern (Andreoni & Bernheim 2009)
                u_rep = -rep_concern * 0.2 * (share - norm_share) ** 2

            utilities[i] = u_material + u_prosoc + u_rep

        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
