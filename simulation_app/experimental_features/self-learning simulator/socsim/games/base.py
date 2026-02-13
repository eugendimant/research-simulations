from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from ..persona import Persona

@dataclass
class GameOutcome:
    actions: Dict[str, Any]
    payoffs: Dict[str, float]
    trace: Dict[str, Any]

class Game(ABC):
    name: str

    @abstractmethod
    def simulate_one(self, rng: np.random.Generator, a: Persona, b: Persona | None, spec: Dict[str, Any]) -> GameOutcome:
        raise NotImplementedError
