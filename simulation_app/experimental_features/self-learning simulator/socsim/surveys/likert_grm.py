from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class LikertItem:
    id: str
    a: float  # discrimination
    thresholds: List[float]  # ordered: b1 < b2 < ... < b_{K-1}

def grm_category_probs(theta: float, item: LikertItem) -> np.ndarray:
    # Samejima-style graded response: P(Y >= k) = logistic(a*(theta - b_k))
    b = np.array(item.thresholds, dtype=float)
    K = len(b) + 1
    P_ge = np.zeros(K+1, dtype=float)
    P_ge[1:K] = expit(item.a * (theta - b))
    P_ge[0] = 1.0
    P_ge[K] = 0.0
    probs = np.clip(P_ge[:K] - P_ge[1:K+1], 1e-12, 1.0)
    probs = probs / probs.sum()
    return probs

def sample_likert(rng: np.random.Generator, theta: float, item: LikertItem) -> tuple[int, np.ndarray]:
    probs = grm_category_probs(theta, item)
    y = int(rng.choice(len(probs), p=probs)) + 1  # categories 1..K
    return y, probs

def simulate_likert_block(rng: np.random.Generator, theta: float, items: List[LikertItem]) -> Dict[str, Any]:
    responses = {}
    traces = {}
    for it in items:
        y, p = sample_likert(rng, theta, it)
        responses[it.id] = y
        traces[it.id] = {"probs": p.tolist(), "a": it.a, "thresholds": it.thresholds}
    return {"responses": responses, "trace": traces}
