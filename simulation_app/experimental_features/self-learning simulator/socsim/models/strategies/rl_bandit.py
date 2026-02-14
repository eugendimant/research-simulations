"""Reinforcement Learning / Multi-Armed Bandit strategy.

Implements experience-based learning for repeated games:
  - EWA (Experience-Weighted Attraction): Camerer & Ho (1999)
  - Propensity reinforcement: Erev & Roth (1998)
  - ε-greedy exploration for early rounds

Parameters from persona:
  - learning_rate: how quickly to update attractions (φ in EWA)
  - exploration: ε for ε-greedy exploration
  - memory: decay factor for past attractions (δ in EWA)

References:
  Camerer & Ho (1999). Experience-Weighted Attraction Learning in Normal Form Games.
  Econometrica 67(4):827-874.
  Erev & Roth (1998). Predicting How People Play Games. AER 88(4):848-881.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Strategy
from ...persona import Persona
from ...agents.backend import GameState


class RLBanditStrategy(Strategy):
    name = "rl_bandit"

    def __init__(self) -> None:
        # Attraction vectors per game (persistent across rounds for same agent)
        self._attractions: Dict[str, np.ndarray] = {}

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

        lr = float(persona.params.get("learning_rate", 0.3))
        eps = float(persona.params.get("exploration", 0.1))
        memory = float(persona.params.get("memory", 0.9))
        lam = float(persona.params.get("noise_lambda", 1.0))

        game_key = f"{state.game_name}_{persona.id}"

        # Initialise or retrieve attractions
        if game_key not in self._attractions or len(self._attractions[game_key]) != n:
            self._attractions[game_key] = np.zeros(n, dtype=float)

        A = self._attractions[game_key]

        # Update from history (EWA-style)
        if state.history:
            last = state.history[-1]
            last_action_idx = last.get("action_idx", None)
            last_payoff = float(last.get("payoff", 0.0))

            if last_action_idx is not None and 0 <= last_action_idx < n:
                # EWA update: A_i(t) = memory * A_i(t-1) + lr * payoff * indicator
                A *= memory
                A[last_action_idx] += lr * last_payoff

        # ε-greedy + softmax hybrid
        # With probability ε: explore uniformly
        # With probability 1-ε: softmax over attractions
        z = lam * (A - np.max(A)) if np.any(A != 0) else np.zeros(n)
        exploit = np.exp(z)
        exploit /= exploit.sum() + 1e-12

        probs = (1 - eps) * exploit + eps * np.ones(n) / n

        self._attractions[game_key] = A  # persist
        return self._normalise(dict(zip(action_names, probs.tolist())))
