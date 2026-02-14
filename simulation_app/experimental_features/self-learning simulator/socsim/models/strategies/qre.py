"""Quantal Response Equilibrium (QRE) strategy.

Implements McKelvey & Palfrey (1995):
  - Agents best-respond with noise (logit errors)
  - The noise parameter λ governs rationality: λ→0 is random, λ→∞ is rational
  - QRE is a fixed-point concept: agents best-respond to *others' noisy play*

The persona's ``noise_lambda`` parameter directly maps to the QRE λ.

References:
  McKelvey & Palfrey (1995). Quantal Response Equilibria for Normal Form Games.
  Games and Economic Behavior 10(1):6-38.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class QREStrategy(Strategy):
    name = "qre"

    def __init__(self, iterations: int = 10) -> None:
        """
        Parameters
        ----------
        iterations:
            Number of fixed-point iterations for QRE convergence.
        """
        self._iterations = iterations

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
        lam = float(persona.params.get("noise_lambda", 1.0))
        lam = max(0.01, min(lam, 20.0))

        vals = self._action_values(state)

        # Compute utilities based on persona traits
        alpha = float(persona.params.get("fairness_alpha", 0.8))
        beta = float(persona.params.get("fairness_beta", 0.2))
        prosoc = float(persona.params.get("prosociality", 0.0))

        # Start with uniform belief about opponent
        belief = np.ones(n) / n

        for _ in range(self._iterations):
            # Compute expected utilities given opponent plays according to `belief`
            utilities = np.zeros(n, dtype=float)
            for i, v in enumerate(vals):
                # Self-payoff proxy
                endow = float(state.game_params.get("endowment", 10.0))
                pi_self = endow - v if v <= endow else endow
                pi_other = v

                # Fehr-Schmidt utility
                u = pi_self - alpha * max(pi_other - pi_self, 0) - beta * max(pi_self - pi_other, 0)
                u += prosoc * pi_other * 0.1
                utilities[i] = u

            # QRE response: softmax with λ
            z = lam * (utilities - np.max(utilities))
            probs = np.exp(z)
            probs /= probs.sum() + 1e-12
            belief = probs  # fixed-point update

        return self._normalise(dict(zip(action_names, belief.tolist())))
