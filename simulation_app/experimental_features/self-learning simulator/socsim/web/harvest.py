from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from datetime import datetime, timezone

from .http import WebClient, FetchResult
from .sources import CrossrefSource, OpenAlexSource, SemanticScholarSource, BibItem

DEFAULT_SOURCES = ("crossref", "openalex", "semanticscholar")

def normalize_doi(doi: str) -> str:
    d = (doi or "").strip()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    d = d.replace("doi:", "").strip().lower()
    return d

def stable_bib_uid(title: str, ref: str, source: str) -> str:
    raw = (title or "") + "||" + (ref or "") + "||" + (source or "")
    h = __import__('hashlib').sha256(raw.encode('utf-8')).hexdigest()[:16]
    return f"bib_{h}"


@dataclass
class HarvestConfig:
    sources: Sequence[str] = DEFAULT_SOURCES
    per_source_rows: int = 20
    mailto: Optional[str] = None

def harvest_bibliography(client: WebClient, query: str, cfg: HarvestConfig) -> List[BibItem]:
    out: List[BibItem] = []
    if "crossref" in cfg.sources:
        out.extend(CrossrefSource().search(client, query, rows=cfg.per_source_rows, mailto=cfg.mailto))
    if "openalex" in cfg.sources:
        out.extend(OpenAlexSource().search(client, query, per_page=cfg.per_source_rows))
    if "semanticscholar" in cfg.sources:
        out.extend(SemanticScholarSource().search(client, query, limit=cfg.per_source_rows))

    seen = set()
    dedup: List[BibItem] = []
    for it in out:
        k = ("doi:" + it.doi.lower()) if it.doi else ("title:" + it.title.lower())
        if k in seen:
            continue
        seen.add(k)
        dedup.append(it)
    return dedup

def bibitem_to_atomic_unit(it: BibItem, query: str, fetched: Optional[FetchResult] = None) -> Dict:
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doi_norm = normalize_doi(it.doi) if it.doi else ""
    ref_type = "doi" if doi_norm else "url"
    ref = doi_norm or (it.url or "")
    uid = stable_bib_uid(it.title, ref, it.source)
    prov = {
        "added_by": "socsim.web.harvest",
        "added_at_utc": retrieved,
        "extraction_method": "programmatic_metadata_only",
        "notes": "Metadata-only import from public APIs; no effect sizes extracted.",
    }
    if fetched is not None:
        prov["retrieved_at_utc"] = fetched.retrieved_at_utc
        prov["body_sha256"] = fetched.body_sha256
    return {
        "id": uid,
        "kind": "bibliography",
        "title": it.title,
        "source": {"ref_type": ref_type, "ref": ref,
            "ref_raw": it.doi or it.url or "", "origin": it.source, "url": it.url or ""},
        "tags": ["bibliography", "metadata_only"],
        "payload": {"year": it.year, "venue": it.venue, "authors": it.authors, "query": query},
        "provenance": prov,
    }
