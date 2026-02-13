from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any
from urllib.parse import quote

from .http import WebClient

@dataclass
class BibItem:
    title: str
    doi: Optional[str]
    url: Optional[str]
    year: Optional[int]
    venue: Optional[str]
    authors: List[str]
    source: str

def _safe_year(y: Any) -> Optional[int]:
    try:
        if y is None:
            return None
        y = int(y)
        if 1500 <= y <= 2100:
            return y
    except Exception:
        return None
    return None

class CrossrefSource:
    base = "https://api.crossref.org/works"

    def search(self, client: WebClient, query: str, rows: int = 20, mailto: Optional[str] = None) -> List[BibItem]:
        q = quote(query)
        url = f"{self.base}?query={q}&rows={int(rows)}"
        if mailto:
            url += f"&mailto={quote(mailto)}"
        res = client.fetch_text(url)
        if res.status != 200:
            return []
        import json
        data = json.loads(res.body_text)
        items = data.get("message", {}).get("items", []) or []
        out: List[BibItem] = []
        for it in items:
            title = (it.get("title") or [""])[0] if isinstance(it.get("title"), list) else (it.get("title") or "")
            doi = it.get("DOI")
            url2 = it.get("URL")
            venue = (it.get("container-title") or [""])[0] if isinstance(it.get("container-title"), list) else it.get("container-title")
            year = None
            issued = it.get("issued", {}).get("date-parts", None)
            if issued and isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
                year = _safe_year(issued[0][0])
            authors = []
            for a in it.get("author", []) or []:
                given = a.get("given", "")
                family = a.get("family", "")
                nm = (given + " " + family).strip() or family or given
                if nm:
                    authors.append(nm)
            out.append(BibItem(title=title or query, doi=doi, url=url2, year=year, venue=venue, authors=authors, source="crossref"))
        return out

class OpenAlexSource:
    base = "https://api.openalex.org/works"

    def search(self, client: WebClient, query: str, per_page: int = 25) -> List[BibItem]:
        q = quote(query)
        url = f"{self.base}?search={q}&per_page={int(per_page)}"
        res = client.fetch_text(url)
        if res.status != 200:
            return []
        import json
        data = json.loads(res.body_text)
        items = data.get("results", []) or []
        out: List[BibItem] = []
        for it in items:
            title = it.get("title") or query
            doi = None
            ids = it.get("ids", {}) or {}
            doi_url = ids.get("doi")
            if doi_url and isinstance(doi_url, str) and "doi.org/" in doi_url:
                doi = doi_url.split("doi.org/")[-1]
            url2 = it.get("id") or ids.get("openalex")
            year = _safe_year(it.get("publication_year"))
            host = it.get("host_venue", {}) or {}
            venue = host.get("display_name")
            authors = []
            for a in it.get("authorships", []) or []:
                au = (a.get("author") or {})
                nm = au.get("display_name")
                if nm:
                    authors.append(nm)
            out.append(BibItem(title=title, doi=doi, url=url2, year=year, venue=venue, authors=authors, source="openalex"))
        return out

class SemanticScholarSource:
    base = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, client: WebClient, query: str, limit: int = 20) -> List[BibItem]:
        q = quote(query)
        fields = quote("title,year,venue,authors,externalIds,url")
        url = f"{self.base}?query={q}&limit={int(limit)}&fields={fields}"
        res = client.fetch_text(url)
        if res.status != 200:
            return []
        import json
        data = json.loads(res.body_text)
        items = data.get("data", []) or []
        out: List[BibItem] = []
        for it in items:
            title = it.get("title") or query
            year = _safe_year(it.get("year"))
            venue = it.get("venue")
            authors = [a.get("name") for a in (it.get("authors") or []) if a.get("name")]
            ext = it.get("externalIds", {}) or {}
            doi = ext.get("DOI")
            url2 = it.get("url")
            out.append(BibItem(title=title, doi=doi, url=url2, year=year, venue=venue, authors=authors, source="semanticscholar"))
        return out
