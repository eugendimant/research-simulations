"""OfflineBackend — pure rule-based strategy execution.

Always available, deterministic given seed, no network calls.
Uses the strategy library in ``socsim.models.strategies`` to produce
action distributions from persona traits and game state.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..backend import AgentBackend, GameState, Action, ActionResult
from ...persona import Persona
from ...decision import softmax


class OfflineBackend(AgentBackend):
    """Deterministic, offline-first inference backend.

    Uses persona parameters + strategy router to select actions
    via utility-maximization with logit noise.
    """

    def __init__(self, strategy_router: Any = None) -> None:
        self._router = strategy_router

    # ------------------------------------------------------------------
    def sample_action(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator,
    ) -> ActionResult:
        dist = self.action_distribution(state, persona)
        if not dist:
            # Fallback: uniform
            n = max(len(state.available_actions), 1)
            names = [a.name for a in state.available_actions] or ["default"]
            dist = {nm: 1.0 / n for nm in names}

        names = list(dist.keys())
        probs = np.array([dist[n] for n in names], dtype=float)
        probs = probs / probs.sum()  # safety normalise
        idx = int(rng.choice(len(names), p=probs))
        chosen_name = names[idx]

        # Find the matching Action object
        chosen = Action(name=chosen_name, value=chosen_name)
        for a in state.available_actions:
            if a.name == chosen_name:
                chosen = a
                break

        return ActionResult(
            chosen=chosen,
            distribution=dict(zip(names, probs.tolist())),
            trace={"backend": "offline", "strategy_router": self._router is not None},
        )

    # ------------------------------------------------------------------
    def action_distribution(
        self,
        state: GameState,
        persona: Persona,
    ) -> Dict[str, float]:
        if self._router is not None:
            return self._router.distribution(state, persona)

        # Default: softmax over action index using noise_lambda
        n = len(state.available_actions)
        if n == 0:
            return {}
        lam = float(persona.params.get("noise_lambda", 1.0))
        utilities = np.zeros(n, dtype=float)
        # Simple utility heuristic: prosociality-weighted linear
        prosoc = float(persona.params.get("prosociality", 0.0))
        for i, act in enumerate(state.available_actions):
            val = act.value if isinstance(act.value, (int, float)) else float(i) / max(n - 1, 1)
            utilities[i] = prosoc * val  # higher prosociality → prefer larger actions
        probs = softmax(utilities, lam=lam)
        return {act.name: float(p) for act, p in zip(state.available_actions, probs)}

    # ------------------------------------------------------------------
    def supports_text(self) -> bool:
        return False

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "offline",
            "version": "1.0.0",
            "strategy_router": type(self._router).__name__ if self._router else "none",
            "network_required": False,
        }
