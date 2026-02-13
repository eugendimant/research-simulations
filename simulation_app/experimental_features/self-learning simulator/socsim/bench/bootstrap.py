from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .scout import scout_openalex, hits_to_registry_candidates

DEFAULT_QUERIES = [
    ("dictator", "dictator game dataset"),
    ("ultimatum", "ultimatum game dataset"),
    ("trust", "trust game dataset"),
    ("public_goods", "public goods game dataset"),
    ("public_goods_punishment", "public goods punishment dataset"),
    ("prisoners_dilemma", "prisoner's dilemma experiment dataset"),
    ("risk", "risk preference experiment dataset"),
    ("time", "time preference experiment dataset"),
    ("honesty", "dishonesty task experiment dataset"),
    ("social_norms", "social norms experiment dataset"),
]

def bootstrap_candidates(out_path: Path, per_query: int = 10, mailto: Optional[str] = None, user_agent: str = "socsim/0.12.0 (bootstrap)") -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for game, q in DEFAULT_QUERIES:
        hits = scout_openalex(q, per_page=per_query, mailto=mailto, user_agent=user_agent)
        candidates.extend(hits_to_registry_candidates(hits, default_game=game))
    payload = {"version":"0.12.0", "per_query": per_query, "candidates": candidates}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
