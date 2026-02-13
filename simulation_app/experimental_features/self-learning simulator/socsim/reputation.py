from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ReputationState:
    alpha: float
    beta: float

def expected_reciprocation(state: ReputationState) -> float:
    return float(state.alpha) / (float(state.alpha) + float(state.beta))

def update(state: ReputationState, reciprocated: bool) -> ReputationState:
    if reciprocated:
        return ReputationState(alpha=state.alpha + 1.0, beta=state.beta)
    return ReputationState(alpha=state.alpha, beta=state.beta + 1.0)
