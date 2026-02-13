from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class GRMItem:
    name: str
    a: float
    thresholds: List[float]  # ordered

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))

def sample_grm(theta: float, item: GRMItem, rng: np.random.Generator) -> int:
    a = float(item.a)
    bs = list(item.thresholds)
    m = len(bs) + 1
    # P(Y>=k) for k=1..m-1
    p_ge = [_logistic(a * (theta - b)) for b in bs]
    probs = []
    probs.append(1.0 - p_ge[0])
    for k in range(1, m - 1):
        probs.append(p_ge[k-1] - p_ge[k])
    probs.append(p_ge[-1])
    probs = np.clip(np.array(probs, dtype=float), 1e-9, 1.0)
    probs = probs / probs.sum()
    return int(rng.choice(np.arange(m), p=probs))

class SurveySimulator:
    def __init__(self, items: List[GRMItem]) -> None:
        self.items = items

    def simulate(self, theta: float, rng: np.random.Generator) -> Dict[str, int]:
        return {it.name: sample_grm(theta, it, rng) for it in self.items}
