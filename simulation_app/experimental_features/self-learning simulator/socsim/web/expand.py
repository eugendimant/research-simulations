from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .http import WebClient
from .harvest import HarvestConfig, harvest_bibliography, bibitem_to_atomic_unit
from ..corpus.store import CorpusStore

@dataclass
class ExpandResult:
    added_or_updated: int

def expand_corpus_from_web(
    corpus_path: Path,
    queries: Sequence[str],
    sources: Sequence[str] = ("crossref", "openalex", "semanticscholar"),
    rows_per_source: int = 20,
    mailto: Optional[str] = None,
    user_agent: str = "socsim/0.13.0 (corpus expand)",
    timeout_s: int = 30,
    min_interval_s: float = 1.0,
    cache_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> ExpandResult:
    corpus = CorpusStore.load(corpus_path)
    client = WebClient(user_agent=user_agent, timeout_s=int(timeout_s), min_interval_s=float(min_interval_s), cache_path=cache_path)
    cfg = HarvestConfig(sources=tuple(sources), per_source_rows=int(rows_per_source), mailto=mailto)
    total = 0
    for q in queries:
        items = harvest_bibliography(client, q, cfg)
        for it in items:
            unit = bibitem_to_atomic_unit(it, query=q, fetched=None)
            corpus.add_dict(unit, schema_path=schema_path)
            total += 1
    corpus.save(corpus_path)
    return ExpandResult(added_or_updated=total)
