"""Deduplication for game family specs.

Canonicalises specs, hashes them, and removes duplicates.
Stable across runs and platforms.
"""
from __future__ import annotations

from typing import List

from .money_request_family import FamilySpec


def deduplicate_specs(specs: List[FamilySpec]) -> List[FamilySpec]:
    """Remove duplicate specs based on canonical hash.

    Returns deduplicated list in original order (first occurrence kept).
    """
    seen: set = set()
    result: List[FamilySpec] = []
    for s in specs:
        h = s.spec_hash
        if h not in seen:
            seen.add(h)
            result.append(s)
    return result
