from __future__ import annotations
import numpy as np

def softmax(u: np.ndarray, lam: float) -> np.ndarray:
    z = lam * (u - np.max(u))
    e = np.exp(z)
    return e / np.sum(e)

def logit_choice(rng: np.random.Generator, utilities: np.ndarray, lam: float) -> tuple[int, np.ndarray]:
    probs = softmax(utilities, lam=lam)
    idx = int(rng.choice(len(probs), p=probs))
    return idx, probs
