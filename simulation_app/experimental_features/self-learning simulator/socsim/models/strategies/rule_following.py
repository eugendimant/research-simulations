"""Rule-following strategy.

Agents who follow explicit rules or instructions more literally than others.
Relevant for:
  - Compliance experiments (Milgram-style)
  - Instruction-following tasks
  - Games with suggested actions or recommendations
  - Anchoring effects from stated reference points

The ``rule_salience`` parameter governs how strongly agents follow rules.

References:
  Kimbrough & Vostroknutov (2016). Norms Make Preferences Social. JEBO 132.
  Tyler (2006). Why People Obey the Law. Princeton UP.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class RuleFollowingStrategy(Strategy):
    name = "rule_following"

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
        rule_salience = float(persona.params.get("rule_salience", 0.5))
        prosoc = float(persona.params.get("prosociality", 0.0))
        endow = float(state.game_params.get("endowment", 10.0))

        # Rule/recommendation from game parameters
        rule_target = state.game_params.get("recommended_action", None)
        rule_share = float(state.game_params.get("suggested_share", 0.5))

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            share = v / (endow + 1e-9)

            # Material utility
            u_material = pi_self

            # Rule compliance: follow the recommended action/share
            if rule_target is not None:
                # Distance to recommended action
                try:
                    target_val = float(rule_target)
                    u_rule = -rule_salience * (v - target_val) ** 2
                except (ValueError, TypeError):
                    u_rule = -rule_salience * (share - rule_share) ** 2
            else:
                u_rule = -rule_salience * (share - rule_share) ** 2

            u_prosoc = prosoc * v * 0.1
            utilities[i] = u_material + u_rule + u_prosoc

        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
