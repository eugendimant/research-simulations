from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import random

@dataclass
class LikertSpec:
    k: int = 7
    threshold_sd: float = 0.9

def likert_response(latent: float, spec: LikertSpec, rng: random.Random) -> int:
    z = latent + rng.gauss(0.0, spec.threshold_sd)
    u = (math.tanh(z) + 1.0) / 2.0
    idx = int(u * spec.k)
    if idx >= spec.k:
        idx = spec.k - 1
    return idx + 1

@dataclass
class ConjointSpec:
    utilities: Dict[str, Dict[str, float]]
    noise_sd: float = 0.6

def conjoint_choose(profile_a: Dict[str, str], profile_b: Dict[str, str], spec: ConjointSpec, rng: random.Random) -> str:
    def u(profile: Dict[str, str]) -> float:
        s = 0.0
        for a, lvl in profile.items():
            s += float(spec.utilities.get(a, {}).get(lvl, 0.0))
        s += rng.gauss(0.0, spec.noise_sd)
        return s
    return "A" if u(profile_a) >= u(profile_b) else "B"

@dataclass
class ListExperimentSpec:
    base_count_mean: float = 2.0
    base_count_sd: float = 1.0
    sensitive_prob: float = 0.25

def list_experiment_count(treated: bool, spec: ListExperimentSpec, rng: random.Random) -> int:
    base = max(0, int(round(rng.gauss(spec.base_count_mean, spec.base_count_sd))))
    if treated:
        base += 1 if rng.random() < spec.sensitive_prob else 0
    return base

@dataclass
class RandomizedResponseSpec:
    p_truth: float = 0.7

def randomized_response(truth: bool, spec: RandomizedResponseSpec, rng: random.Random) -> bool:
    if rng.random() < spec.p_truth:
        return truth
    return rng.random() < 0.5

@dataclass
class EndorsementSpec:
    base_support: float = 0.5
    endorsement_shift: float = 0.1

def endorsement_support(endorsement: bool, spec: EndorsementSpec, latent: float, rng: random.Random) -> bool:
    p = spec.base_support + (spec.endorsement_shift if endorsement else 0.0) + 0.15 * math.tanh(latent)
    p = max(0.0, min(1.0, p))
    return rng.random() < p
