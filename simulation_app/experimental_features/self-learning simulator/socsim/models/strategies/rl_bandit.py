"""Reinforcement Learning / Multi-Armed Bandit strategy.

Implements experience-based learning for repeated games:
  - EWA (Experience-Weighted Attraction): Camerer & Ho (1999)
  - Propensity reinforcement: Erev & Roth (1998)
  - Surprise-weighted learning: surprising outcomes update more strongly
  - Attention-modulated learning rate (Charness & Levin 2005)
  - ε-greedy exploration for early rounds

Parameters from persona:
  - learning_rate: how quickly to update attractions (φ in EWA)
  - exploration: ε for ε-greedy exploration
  - memory: decay factor for past attractions (δ in EWA)
  - attention: modulates effective learning rate (distracted agents learn slower)

References:
  Camerer & Ho (1999). Experience-Weighted Attraction Learning in Normal Form Games.
  Econometrica 67(4):827-874.
  Erev & Roth (1998). Predicting How People Play Games. AER 88(4):848-881.
  Charness & Levin (2005). When Optimal Choices Feel Wrong. AER 95(4):1300-1309.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState

logger = logging.getLogger(__name__)


class RLBanditStrategy(Strategy):
    name = "rl_bandit"

    def __init__(self) -> None:
        # Attraction vectors per game (persistent across rounds for same agent)
        self._attractions: Dict[str, np.ndarray] = {}
        # Running payoff expectation for surprise computation
        self._expected_payoff: Dict[str, float] = {}

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

        base_lr = float(persona.params.get("learning_rate", 0.3))
        eps = float(persona.params.get("exploration", 0.1))
        memory = float(persona.params.get("memory", 0.9))
        lam = float(persona.params.get("noise_lambda", 1.0))
        attention = float(persona.params.get("attention", 0.7))

        game_key = f"{state.game_name}_{persona.id}"

        # Initialise or retrieve attractions
        if game_key not in self._attractions or len(self._attractions[game_key]) != n:
            self._attractions[game_key] = np.zeros(n, dtype=float)

        A = self._attractions[game_key]

        # Update from history (EWA-style with surprise weighting)
        if state.history:
            last = state.history[-1]
            last_action_idx = last.get("action_idx", None)
            last_payoff = float(last.get("payoff", 0.0))

            if last_action_idx is not None and 0 <= last_action_idx < n:
                # Surprise factor: deviation from expected payoff (Erev & Roth 1998)
                expected = self._expected_payoff.get(game_key, last_payoff)
                surprise = min(abs(last_payoff - expected) / (abs(expected) + 1e-9), 3.0)
                surprise_boost = 1.0 + 0.5 * surprise  # surprising outcomes learn 1-2.5× faster

                # Attention modulation (Charness & Levin 2005): distracted agents learn slower
                effective_lr = base_lr * attention * surprise_boost

                # EWA update with modulated learning rate
                A *= memory
                A[last_action_idx] += effective_lr * last_payoff

                # Update running payoff expectation (exponential moving average)
                self._expected_payoff[game_key] = (
                    0.8 * expected + 0.2 * last_payoff
                )

        # ε-greedy + softmax hybrid
        z = lam * (A - np.max(A)) if np.any(A != 0) else np.zeros(n)
        exploit = np.exp(z)
        exploit /= exploit.sum() + 1e-12

        # Exploration decays with experience (more history → less exploration)
        n_rounds = len(state.history) if state.history else 0
        effective_eps = eps * max(0.1, 1.0 / (1.0 + 0.1 * n_rounds))

        probs = (1 - effective_eps) * exploit + effective_eps * np.ones(n) / n

        self._attractions[game_key] = A
        return self._normalise(dict(zip(action_names, probs.tolist())))
