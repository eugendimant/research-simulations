"""Base class for all offline strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ...persona import Persona
from ...agents.backend import GameState


class Strategy(ABC):
    """Abstract base for a strategic decision module.

    Subclasses implement ``action_probabilities`` which returns a dict
    mapping action names to probabilities.  These MUST sum to 1.0 and
    be non-negative.
    """

    name: str = "base"

    @abstractmethod
    def action_probabilities(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        """Return {action_name: probability} dict.  Must sum to 1."""
        ...

    def _normalise(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Safety normalisation to ensure valid distribution."""
        total = sum(max(0.0, v) for v in dist.values())
        if total <= 0:
            n = len(dist) or 1
            return {k: 1.0 / n for k in dist}
        return {k: max(0.0, v) / total for k, v in dist.items()}

    def _action_values(self, state: GameState) -> List[float]:
        """Extract numeric values from available actions."""
        vals = []
        for a in state.available_actions:
            if isinstance(a.value, (int, float)):
                vals.append(float(a.value))
            else:
                vals.append(0.0)
        return vals
