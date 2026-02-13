from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import requests
import re

OPENALEX = "https://api.openalex.org/works"

@dataclass
class ScoutHit:
    title: str
    year: int | None
    doi: str | None
    openalex_id: str | None
    landing_page: str | None
    data_links: List[str]

def _extract_data_links(text: str) -> List[str]:
    if not text:
        return []
    # crude url extraction
    urls = re.findall(r"https?://[^\s)\]]+", text)
    # keep likely repos
    keep = []
    for u in urls:
        ul = u.lower()
        if any(k in ul for k in ["dryad", "dataverse", "osf.io", "figshare", "zenodo", "openicpsr", "github.com"]):
            keep.append(u.rstrip(".,;"))
    # de-dup while preserving order
    out = []
    seen = set()
    for u in keep:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def scout_openalex(query: str, per_page: int = 25, mailto: Optional[str] = None, user_agent: str = "socsim/0.12.0 (bench_scout)") -> List[ScoutHit]:
    sess = requests.Session()
    headers = {"User-Agent": user_agent}
    params = {"search": query, "per-page": int(per_page)}
    if mailto:
        params["mailto"] = mailto
    r = sess.get(OPENALEX, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: List[ScoutHit] = []
    for w in data.get("results", []):
        title = w.get("title")
        year = w.get("publication_year")
        doi = w.get("doi")
        openalex_id = w.get("id")
        landing = None
        # OpenAlex has primary_location with landing_page_url
        pl = (w.get("primary_location") or {})
        landing = (pl.get("landing_page_url") if isinstance(pl, dict) else None)
        # gather data links from open_access, locations, host_venue, etc.
        links = []
        for loc in (w.get("locations") or []):
            if isinstance(loc, dict):
                lp = loc.get("landing_page_url")
                if lp:
                    links.extend(_extract_data_links(lp))
        oa = w.get("open_access") or {}
        if isinstance(oa, dict):
            links.extend(_extract_data_links(oa.get("oa_url") or ""))
        out.append(ScoutHit(
            title=title or "",
            year=int(year) if year is not None else None,
            doi=doi,
            openalex_id=openalex_id,
            landing_page=landing,
            data_links=links,
        ))
    return out

def hits_to_registry_candidates(hits: List[ScoutHit], default_game: str) -> List[Dict[str, Any]]:
    candidates = []
    for h in hits:
        # best-effort guess provider from links
        provider = None
        payload: Dict[str, Any] = {"provider":"url", "url": h.landing_page or ""}
        for u in h.data_links:
            ul = u.lower()
            if "dryad" in ul:
                provider = "dryad"
                # Dryad DOI is not always in URL; keep as url unless found elsewhere
            elif "dataverse" in ul:
                provider = "dataverse"
            elif "osf.io" in ul:
                provider = "osf"
            if provider:
                payload = {"provider": provider, "url": u}
                break
        candidates.append({
            "id": "candidate__" + (h.doi or (h.openalex_id or "unknown")).replace("https://doi.org/","").replace("/","_").replace(":","_"),
            "game": default_game,
            "source": payload,
            "context_features": {},
            "notes": f"Candidate from OpenAlex: {h.title} ({h.year}) DOI={h.doi}"
        })
    return candidates
