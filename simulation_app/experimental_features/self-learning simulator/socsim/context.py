from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from .evidence import EvidenceStore, EvidenceUnit
from .quality import score_quality
from .transport import transportability, attenuate

@dataclass
class Conflict:
    parameter: str
    pos_weight: float
    neg_weight: float
    sources: List[str]

@dataclass
class ContextResult:
    features: Dict[str, Any]
    mean_shifts: Dict[str, float]
    extra_sd: Dict[str, float]
    matched_evidence_ids: list[str]
    conflict_report: Dict[str, Any]

class ContextEngine:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def build(self, features: Dict[str, Any]) -> ContextResult:
        shifts: Dict[str, float] = {}
        extra_sd: Dict[str, float] = {}
        contributions: Dict[str, Dict[str, float]] = {}

        matched = self.store.match(features)
        # collect weighted shifts
        for u in matched:
            if u.type != "param_shift":
                continue

            q = score_quality(getattr(u, "quality", None))
            expected = None
            app = getattr(u, "applicability", None)
            if isinstance(app, dict):
                expected = app.get("context_features")
            t = transportability(features, expected)
            w_eff = float(u.weight) * float(q.weight) * (float(t.score) ** 1.5)

            for p, delta in u.effect.items():
                d_att = attenuate(float(delta), t.score)
                shifts[p] = shifts.get(p, 0.0) + w_eff * d_att
                contributions.setdefault(p, {"pos":0.0, "neg":0.0})
                if d_att > 0:
                    contributions[p]["pos"] += w_eff
                elif d_att < 0:
                    contributions[p]["neg"] += w_eff

            if u.uncertainty and "se" in u.uncertainty:
                for p in u.effect.keys():
                    extra_sd[p] = max(extra_sd.get(p, 0.0), float(u.uncertainty["se"]))

        # conflicts
        conflicts: List[Conflict] = []
        for p, s in contributions.items():
            pos = float(s.get("pos", 0.0))
            neg = float(s.get("neg", 0.0))
            total = pos + neg
            if total <= 0:
                continue
            if pos/total > 0.35 and neg/total > 0.35:
                conflicts.append(Conflict(parameter=p, pos_weight=pos, neg_weight=neg, sources=[m.id for m in matched]))

        return ContextResult(
            features=features,
            mean_shifts=shifts,
            extra_sd=extra_sd,
            matched_evidence_ids=[m.id for m in matched],
            conflict_report={
                "n_conflicts": len(conflicts),
                "conflicts": [c.__dict__ for c in conflicts],
            },
        )
