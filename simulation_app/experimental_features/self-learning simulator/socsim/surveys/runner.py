from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import random

from .models import LikertSpec, likert_response, ConjointSpec, conjoint_choose, ListExperimentSpec, list_experiment_count, RandomizedResponseSpec, randomized_response, EndorsementSpec, endorsement_support

@dataclass
class SurveySpec:
    name: str
    params: Dict[str, Any]

def simulate_survey(rng: random.Random, persona: Dict[str, float], context: Dict[str, Any], spec: SurveySpec) -> Dict[str, Any]:
    name = spec.name
    p = spec.params or {}
    out: Dict[str, Any] = {}
    if name == "likert":
        s = LikertSpec(**p)
        latent = float(persona.get("attitude", 0.0))
        out["resp::likert"] = likert_response(latent, s, rng)
        return out
    if name == "conjoint_binary":
        s = ConjointSpec(**p)
        a = p.get("profile_a") or {}
        b = p.get("profile_b") or {}
        out["resp::choice"] = conjoint_choose(a, b, s, rng)
        return out
    if name == "list_experiment":
        s = ListExperimentSpec(**p)
        treated = bool(context.get("treated", False))
        out["resp::count"] = list_experiment_count(treated, s, rng)
        return out
    if name == "randomized_response":
        truth = bool(p.get("truth", False))
        spec_params = {k:v for k,v in p.items() if k != "truth"}
        s = RandomizedResponseSpec(**spec_params)
        out["resp::rr"] = int(randomized_response(truth, s, rng))
        return out
    if name == "endorsement":
        s = EndorsementSpec(**p)
        endorsement_flag = bool(context.get("endorsement", False))
        latent = float(persona.get("attitude", 0.0))
        out["resp::support"] = int(endorsement_support(endorsement_flag, s, latent, rng))
        return out
    raise ValueError(f"Unknown survey: {name}")
