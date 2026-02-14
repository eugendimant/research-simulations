"""Intergroup Bias strategy.

Implements full intergroup discrimination model:
  - Ingroup favoritism: agents give more to / cooperate more with ingroup
  - Outgroup derogation: agents give less to / punish outgroup members
  - Political identity amplification: partisan identity increases bias
  - Sign-dependent effects: giving vs. punishment differ

Parameters from persona:
  - ingroup_bias: base intergroup discrimination strength (0=no bias, 1=full)
  - prosociality: baseline generosity (modulated by group membership)
  - noise_lambda: rationality parameter

Context from GameState:
  - context["ingroup_partner"]: 1 if partner is ingroup, 0 if outgroup
  - context["political_identity"]: True if political identity salient
  - context["partner_group"]: label of partner's group

References:
  Iyengar & Westwood (2015). Fear and Loathing Across Party Lines.
  AJPS 59(3):690-707.
  Balliet et al. (2014). Ingroup Favoritism in Cooperation.
  Psych Bull 140(6):1556-1581.  (meta-analysis: d ≈ 0.32)
  Dimant (2024). Political intergroup effects d ≈ 0.6-0.9.
  Fershtman & Gneezy (2001). Discrimination in trust/dictator games.
  Tajfel et al. (1971). Social categorization and intergroup behaviour.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState

logger = logging.getLogger(__name__)

# Effect size multipliers by context type (Balliet et al. 2014; Dimant 2024)
_CONTEXT_MULTIPLIERS: Dict[str, float] = {
    "political_identity": 1.6,   # Dimant (2024): d ≈ 0.6-0.9
    "racial_identity": 1.3,      # Fershtman & Gneezy (2001)
    "minimal_group": 1.0,        # Tajfel et al. (1971): baseline
    "national_identity": 1.2,
    "religious_identity": 1.3,
    "default": 1.0,
}


class IntergroupBiasStrategy(Strategy):
    """Full intergroup discrimination model.

    Differentiates between ingroup favoritism (positive bias toward ingroup)
    and outgroup derogation (negative bias toward outgroup). These are
    empirically separable effects (Balliet et al. 2014).
    """

    name = "intergroup_bias"

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

        ig_bias = float(persona.params.get("ingroup_bias", 0.3))
        prosoc = float(persona.params.get("prosociality", 0.0))
        lam = float(persona.params.get("noise_lambda", 1.0))
        alpha = float(persona.params.get("fairness_alpha", 0.8))
        beta = float(persona.params.get("fairness_beta", 0.2))

        is_ingroup = bool(state.context.get("ingroup_partner", True))
        endow = float(state.game_params.get("endowment", 10.0))

        # Determine context multiplier
        identity_type = "default"
        if state.context.get("political_identity"):
            identity_type = "political_identity"
        elif state.context.get("racial_identity"):
            identity_type = "racial_identity"
        elif state.context.get("minimal_group"):
            identity_type = "minimal_group"
        context_mult = _CONTEXT_MULTIPLIERS.get(identity_type, 1.0)

        # Compute effective bias with context amplification
        effective_bias = ig_bias * context_mult

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v if v <= endow else endow
            pi_other = v
            share = v / (endow + 1e-9)

            # Baseline utility (Fehr-Schmidt)
            u = pi_self - alpha * max(pi_other - pi_self, 0) - beta * max(pi_self - pi_other, 0)

            # Prosocial component
            u += prosoc * pi_other * 0.1

            # Intergroup modulation
            if is_ingroup:
                # Ingroup favoritism: boost utility of generous actions
                # Balliet et al. (2014): ingroup cooperation d ≈ 0.32
                u += effective_bias * share * 0.35
            else:
                # Outgroup derogation: penalize generous actions
                # Iyengar & Westwood (2015): outgroup discrimination
                u -= effective_bias * share * 0.40

            utilities[i] = u

        # Softmax selection
        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
