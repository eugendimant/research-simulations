"""Noisy Best Response (NBR) strategy.

The simplest rational-agent model:
  - Compute utilities for each action
  - Apply logit noise (softmax) to choose
  - No strategic reasoning about opponents (unlike Level-k, CH, QRE)

This is the "default" strategy for most one-shot games where
the persona's utility function directly determines behaviour.

Parameters used from persona:
  - noise_lambda: rationality/noise level
  - fairness_alpha, fairness_beta: inequity aversion
  - prosociality: weight on other's payoff
  - ingroup_bias: in/outgroup differential treatment
  - norm_weight: sensitivity to social norms
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState
from ..social_prefs import fehr_schmidt_utility, ingroup_adjustment, norm_utility


class NoisyBestResponseStrategy(Strategy):
    name = "noisy_best_response"

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
        prosoc = float(persona.params.get("prosociality", 0.0))
        lam = float(persona.params.get("noise_lambda", 1.0))
        ig_bias = float(persona.params.get("ingroup_bias", 0.0))
        is_ingroup = int(state.context.get("ingroup_partner", 1))
        w_norm = float(persona.params.get("norm_weight", 0.0))
        norm_target = float(state.game_params.get("norm_target_share", 0.5))
        endow = float(state.game_params.get("endowment", 10.0))

        utilities = np.zeros(n, dtype=float)
        for i, v in enumerate(vals):
            pi_self = endow - v
            pi_other = v
            share = v / (endow + 1e-9)

            u_fs = fehr_schmidt_utility(pi_self, pi_other, alpha, beta)
            u_bias = ingroup_adjustment(ig_bias, is_ingroup) * share
            u_norm = norm_utility(component=-(share - norm_target) ** 2, norm_weight=w_norm)
            u_prosoc = prosoc * pi_other * 0.05

            utilities[i] = u_fs + u_bias + u_norm + u_prosoc

        z = lam * (utilities - np.max(utilities))
        probs = np.exp(z)
        probs /= probs.sum() + 1e-12

        return self._normalise(dict(zip(action_names, probs.tolist())))
