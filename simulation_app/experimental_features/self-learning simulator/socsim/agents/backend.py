"""AgentBackend — abstract interface that all SocSim inference backends implement.

Three concrete implementations:
  * OfflineBackend  — deterministic rule-based strategies (always available)
  * LocalLLMBackend — local open-source model inference (optional, no billing)
  * RemoteLLMBackend — external paid APIs (opt-in only)

The simulator calls ``sample_action`` for game choices and optionally
``generate_open_ended`` for text responses.  Each backend must be able
to produce both action distributions and (optionally) text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..persona import Persona


# ---------------------------------------------------------------------------
# Lightweight protocol types
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A single action a player can take."""
    name: str
    value: Any  # numeric, categorical, or structured

    def __repr__(self) -> str:
        return f"Action({self.name}={self.value})"


@dataclass
class GameState:
    """Everything the agent can observe when deciding."""
    game_name: str
    game_params: Dict[str, Any] = field(default_factory=dict)
    available_actions: List[Action] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    round_number: int = 0
    opponent_info: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """The output of a backend's action-selection step."""
    chosen: Action
    distribution: Dict[str, float]  # action_name → probability
    trace: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AgentBackend(ABC):
    """Abstract base class for all SocSim inference backends."""

    @abstractmethod
    def sample_action(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator,
    ) -> ActionResult:
        """Choose an action given the game state and persona."""
        ...

    @abstractmethod
    def action_distribution(
        self,
        state: GameState,
        persona: Persona,
    ) -> Dict[str, float]:
        """Return the full probability distribution over actions (no sampling)."""
        ...

    def generate_open_ended(
        self,
        prompt_spec: Dict[str, Any],
        persona: Persona,
        rng: np.random.Generator,
    ) -> str:
        """Generate a free-text response.  Optional — returns empty string by default."""
        return ""

    def supports_text(self) -> bool:
        """Whether this backend can generate meaningful open-ended text."""
        return False

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return backend metadata for run manifests."""
        ...
