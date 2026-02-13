from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv

@dataclass
class ExtractedEffect:
    parameter: str
    delta: float
    se: Optional[float]
    n: Optional[int]
    moderator: Optional[str]
    value: Optional[str]

class CSVEvidenceExtractor:
    """Audited extractor for a strict CSV template (no PDFs, no free text).

    Required columns:
    - parameter
    - delta

    Optional columns:
    - se
    - n
    - moderator
    - value
    """
    REQUIRED = {"parameter", "delta"}

    def extract(self, path: Path) -> List[ExtractedEffect]:
        out: List[ExtractedEffect] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            fields = {h.strip() for h in reader.fieldnames}
            missing = self.REQUIRED - fields
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")

            for r in reader:
                param = (r.get("parameter") or "").strip()
                if not param:
                    continue
                delta = float(r.get("delta"))
                se = r.get("se")
                se_v = float(se) if se not in (None, "", "NA", "na") else None
                n = r.get("n")
                n_v = int(float(n)) if n not in (None, "", "NA", "na") else None
                mod = (r.get("moderator") or "").strip() or None
                val = (r.get("value") or "").strip() or None
                out.append(ExtractedEffect(parameter=param, delta=delta, se=se_v, n=n_v, moderator=mod, value=val))
        return out
