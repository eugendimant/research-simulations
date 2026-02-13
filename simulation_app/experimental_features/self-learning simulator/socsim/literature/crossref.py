from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import requests

@dataclass
class BibItem:
    title: str
    doi: Optional[str]
    url: Optional[str]
    year: Optional[int]
    container: Optional[str]
    authors: List[str]

def search_crossref(query: str, rows: int = 20, mailto: Optional[str] = None) -> List[BibItem]:
    headers = {"User-Agent": "socsim/0.5.0 (mailto:unknown)"}
    if mailto:
        headers["User-Agent"] = f"socsim/0.5.0 (mailto:{mailto})"
    params = {"query": query, "rows": int(rows)}
    r = requests.get("https://api.crossref.org/works", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("message", {})
    out: List[BibItem] = []
    for it in data.get("items", []):
        title = (it.get("title") or [""])[0]
        doi = it.get("DOI")
        url = it.get("URL")
        year = None
        issued = it.get("issued", {}).get("date-parts", None)
        if issued and issued[0] and isinstance(issued[0][0], int):
            year = int(issued[0][0])
        container = (it.get("container-title") or [""])[0] if it.get("container-title") else None
        authors = []
        for a in it.get("author", [])[:12]:
            nm = ((a.get("given","") + " " + a.get("family","")).strip())
            if nm:
                authors.append(nm)
        out.append(BibItem(title=title, doi=doi, url=url, year=year, container=container, authors=authors))
    return out
