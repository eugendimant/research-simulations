from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set

from .evidence import EvidenceStore

@dataclass
class CoverageReport:
    matched_units: int
    unique_sources: int
    matched_domain_tags: List[str]

def coverage(store: EvidenceStore, features: Dict) -> CoverageReport:
    matched = store.match(features)
    sources: Set[str] = set()
    tags: Set[str] = set()
    for u in matched:
        src = u.data.get("source", {})
        sources.add(f"{src.get('type','')}:{src.get('ref','')}")
        for t in u.data.get("domain_tags", []):
            tags.add(str(t))
    return CoverageReport(
        matched_units=len(matched),
        unique_sources=len([s for s in sources if s.strip(":")]),
        matched_domain_tags=sorted(tags)[:50],
    )
