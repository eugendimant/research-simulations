"""RemoteLLMBackend — external paid API inference (opt-in only).

This backend must NEVER be used unless explicitly enabled by the user.
It wraps the existing LLMResponseGenerator provider chain and requires:
  1. SOCSIM_REMOTE_LLM_ENABLED=true
  2. SOCSIM_ALLOW_NETWORK=true
  3. A user-supplied API key (no embedded/default keys)
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ..backend import AgentBackend, GameState, Action, ActionResult
from ...persona import Persona
from ...config import get_config, CostPolicy
from ...network_guard import assert_network_allowed, NetworkBlockedError

logger = logging.getLogger(__name__)


class RemoteLLMBackend(AgentBackend):
    """Paid remote LLM backend — opt-in only, never default.

    Raises NetworkBlockedError if the user hasn't explicitly enabled it.
    """

    def __init__(self, fallback: AgentBackend | None = None) -> None:
        self._fallback = fallback

    def _check_enabled(self) -> None:
        """Raise if remote access is not explicitly enabled."""
        cfg = get_config()
        if not cfg.remote_llm_enabled:
            raise NetworkBlockedError(
                "RemoteLLMBackend is disabled. Set SOCSIM_REMOTE_LLM_ENABLED=true "
                "and SOCSIM_ALLOW_NETWORK=true to enable paid remote LLM calls."
            )
        assert_network_allowed("RemoteLLMBackend inference call")

    def sample_action(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator,
    ) -> ActionResult:
        try:
            self._check_enabled()
        except NetworkBlockedError:
            if self._fallback:
                return self._fallback.sample_action(state, persona, rng)
            raise

        dist = self.action_distribution(state, persona)
        names = list(dist.keys())
        probs = np.array([dist[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(names), p=probs))
        chosen_name = names[idx]
        chosen = Action(name=chosen_name, value=chosen_name)
        for a in state.available_actions:
            if a.name == chosen_name:
                chosen = a
                break
        return ActionResult(chosen=chosen, distribution=dict(zip(names, probs.tolist())),
                            trace={"backend": "remote_llm"})

    def action_distribution(
        self,
        state: GameState,
        persona: Persona,
    ) -> Dict[str, float]:
        try:
            self._check_enabled()
        except NetworkBlockedError:
            if self._fallback:
                return self._fallback.action_distribution(state, persona)
            raise

        # Placeholder — actual implementation would call the remote provider
        # For now, delegate to fallback (this is the opt-in pathway)
        if self._fallback:
            return self._fallback.action_distribution(state, persona)
        n = max(len(state.available_actions), 1)
        return {a.name: 1.0 / n for a in state.available_actions}

    def generate_open_ended(
        self,
        prompt_spec: Dict[str, Any],
        persona: Persona,
        rng: np.random.Generator,
    ) -> str:
        try:
            self._check_enabled()
        except NetworkBlockedError:
            if self._fallback:
                return self._fallback.generate_open_ended(prompt_spec, persona, rng)
            return ""
        # Placeholder for actual remote call
        if self._fallback:
            return self._fallback.generate_open_ended(prompt_spec, persona, rng)
        return ""

    def supports_text(self) -> bool:
        try:
            self._check_enabled()
            return True
        except NetworkBlockedError:
            return self._fallback.supports_text() if self._fallback else False

    def metadata(self) -> Dict[str, Any]:
        cfg = get_config()
        return {
            "backend": "remote_llm",
            "enabled": cfg.remote_llm_enabled,
            "network_allowed": cfg.allow_network_calls,
            "cost_policy": cfg.cost_policy.value,
            "billing": True,
            "warning": "May incur per-token costs",
        }
