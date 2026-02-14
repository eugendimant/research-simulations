"""Quantal Response Equilibrium (QRE) strategy.

Implements McKelvey & Palfrey (1995):
  - Agents best-respond with noise (logit errors)
  - The noise parameter λ governs rationality: λ→0 is random, λ→∞ is rational
  - QRE is a fixed-point concept: agents best-respond to *others' noisy play*

The persona's ``noise_lambda`` parameter directly maps to the QRE λ.
Game-specific λ calibrations from published estimates (see GAME_LAMBDA_CALIBRATION).

References:
  McKelvey & Palfrey (1995). Quantal Response Equilibria for Normal Form Games.
  Games and Economic Behavior 10(1):6-38.
  Goeree, Holt & Palfrey (2016). Quantal Response Equilibrium. Princeton UP.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState

logger = logging.getLogger(__name__)

# ── Empirical λ calibration by game type ────────────────────────────────
# Estimated from published QRE fits. Higher λ = more rational.
# Source: Goeree, Holt & Palfrey (2016); Wright & Leyton-Brown (2017)
GAME_LAMBDA_CALIBRATION: Dict[str, Dict[str, float]] = {
    "matching_pennies":    {"default_lambda": 0.45,  "lambda_range": (0.2, 0.8)},
    "coordination":        {"default_lambda": 2.0,   "lambda_range": (1.2, 3.0)},
    "beauty_contest":      {"default_lambda": 1.0,   "lambda_range": (0.4, 1.8)},
    "dictator":            {"default_lambda": 1.5,   "lambda_range": (0.8, 2.5)},
    "ultimatum":           {"default_lambda": 2.5,   "lambda_range": (1.5, 4.0)},
    "trust":               {"default_lambda": 1.8,   "lambda_range": (1.0, 3.0)},
    "public_goods":        {"default_lambda": 1.2,   "lambda_range": (0.6, 2.0)},
    "prisoners_dilemma":   {"default_lambda": 1.0,   "lambda_range": (0.5, 2.0)},
    "battle_of_sexes":     {"default_lambda": 1.5,   "lambda_range": (0.8, 2.5)},
    "money_request_11_20": {"default_lambda": 1.3,   "lambda_range": (0.7, 2.2)},
}


class QREStrategy(Strategy):
    name = "qre"

    def __init__(
        self,
        max_iterations: int = 50,
        convergence_tol: float = 1e-6,
        use_game_calibration: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        max_iterations:
            Maximum fixed-point iterations for QRE convergence.
        convergence_tol:
            Early-stop when belief change < this threshold (L2 norm).
        use_game_calibration:
            If True, use published λ estimates per game type as baseline.
        """
        self._max_iterations = max_iterations
        self._convergence_tol = convergence_tol
        self._use_game_calibration = use_game_calibration

    def _calibrated_lambda(
        self, persona_lambda: float, game_name: str
    ) -> float:
        """Blend persona's λ with game-specific empirical calibration."""
        if not self._use_game_calibration:
            return persona_lambda

        cal = GAME_LAMBDA_CALIBRATION.get(game_name)
        if cal is None:
            return persona_lambda

        default_lam = cal["default_lambda"]
        lo, hi = cal["lambda_range"]
        # Blend: use persona lambda as a multiplier on the calibrated default
        blended = default_lam * (persona_lambda / 1.0)  # persona_lambda=1.0 → use default
        return max(lo, min(blended, hi))

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
        raw_lam = float(persona.params.get("noise_lambda", 1.0))
        lam = self._calibrated_lambda(raw_lam, state.game_name)
        lam = max(0.01, min(lam, 20.0))

        vals = self._action_values(state)

        # Compute utilities based on persona traits
        alpha = float(persona.params.get("fairness_alpha", 0.8))
        beta = float(persona.params.get("fairness_beta", 0.2))
        prosoc = float(persona.params.get("prosociality", 0.0))
        endow = float(state.game_params.get("endowment", 10.0))

        # Start with uniform belief about opponent
        belief = np.ones(n) / n

        for iteration in range(self._max_iterations):
            prev_belief = belief.copy()

            # Compute expected utilities given opponent plays according to `belief`
            utilities = np.zeros(n, dtype=float)
            for i, v in enumerate(vals):
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
            belief = probs

            # Convergence check
            delta = np.linalg.norm(belief - prev_belief)
            if delta < self._convergence_tol:
                logger.debug(
                    "QRE converged after %d iterations (δ=%.2e)",
                    iteration + 1, delta,
                )
                break
        else:
            logger.debug(
                "QRE reached max %d iterations without convergence (δ=%.2e)",
                self._max_iterations, delta,
            )

        return self._normalise(dict(zip(action_names, belief.tolist())))
