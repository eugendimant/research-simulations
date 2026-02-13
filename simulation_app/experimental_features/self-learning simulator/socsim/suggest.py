from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class Suggestion:
    feature: str
    values: List[Any]
    reason: str

def suggest_experiments(features: Dict[str, Any], conflict_report: Dict[str, Any], matched_evidence: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # Heuristic: if there are conflicts, propose toggling the most common missing/variable context keys in applicability blocks.
    suggestions: List[Suggestion] = []
    conflicts = (conflict_report or {}).get("conflicts", []) or []
    if not conflicts:
        return {"n_suggestions": 0, "suggestions": []}

    counts: Dict[str, Dict[str, int]] = {}
    if matched_evidence:
        for u in matched_evidence:
            app = (u.get("applicability") or {})
            cf = (app.get("context_features") or {})
            for k, v in cf.items():
                counts.setdefault(k, {})
                sv = str(v)
                counts[k][sv] = counts[k].get(sv, 0) + 1

    # Rank keys by diversity of values
    ranked: List[Tuple[str, int]] = []
    for k, vv in counts.items():
        ranked.append((k, len(vv)))
    ranked.sort(key=lambda x: -x[1])

    for k, div in ranked[:10]:
        vals = sorted(counts[k].keys())
        suggestions.append(Suggestion(feature=k, values=vals, reason="conflicting evidence appears conditioned on this context feature"))

    # fallback: propose increasing stakes and anonymity toggles
    if not suggestions:
        for k in ["ctx::anonymity", "ctx::repeated", "ctx::stakes_level"]:
            if k in features:
                suggestions.append(Suggestion(feature=k, values=[features[k]], reason="baseline; add orthogonal manipulation"))
            else:
                suggestions.append(Suggestion(feature=k, values=[0,1] if "anonymity" in k or "repeated" in k else ["low","high"], reason="common moderator"))

    return {"n_suggestions": len(suggestions), "suggestions": [s.__dict__ for s in suggestions]}
