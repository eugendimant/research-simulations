from __future__ import annotations

def fehr_schmidt_utility(pi_self: float, pi_other: float, alpha: float, beta: float) -> float:
    return float(pi_self - alpha * max(pi_other - pi_self, 0.0) - beta * max(pi_self - pi_other, 0.0))

def norm_utility(component: float, norm_weight: float) -> float:
    return float(norm_weight * component)

def ingroup_adjustment(weight: float, is_ingroup: int) -> float:
    return float(weight * (1.0 if is_ingroup else -1.0))
