from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from jsonschema import validate as js_validate

def _matches(pattern: Dict[str, Any], features: Dict[str, Any]) -> bool:
    for k, v in pattern.items():
        if k not in features:
            return False
        fv = features[k]
        if isinstance(v, list):
            if fv not in v:
                return False
        else:
            if fv != v:
                return False
    return True

@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    type: str
    when: Dict[str, Any]
    effect: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    weight: float = 1.0
    notes: str = ""
    provenance: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    applicability: Optional[Dict[str, Any]] = None

class EvidenceStore:
    def __init__(self, units: List[EvidenceUnit]):
        self.units = units
        self._by_id = {u.id: u for u in units}
        # simple inverted index for exact scalar matches in unit.when
        self._index: Dict[tuple[str, str], List[str]] = {}
        for u in units:
            if isinstance(u.when, dict):
                for k, v in u.when.items():
                    if isinstance(v, (str, int, float, bool)):
                        key = (str(k), str(v))
                        self._index.setdefault(key, []).append(u.id)
                    elif isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, (str, int, float, bool)):
                                key = (str(k), str(vv))
                                self._index.setdefault(key, []).append(u.id)

    @staticmethod
    def load(path: Path) -> "EvidenceStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        units = [EvidenceUnit(**x) for x in data.get("evidence_units", [])]
        return EvidenceStore(units)

    def save(self, path: Path) -> None:
        data = {"evidence_units": [u.__dict__ for u in self.units]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def validate_unit_dict(unit: Dict[str, Any], schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        js_validate(instance=unit, schema=schema)

    def add_dict(self, unit: Dict[str, Any], schema_path: Path) -> None:
        # validate first
        self.validate_unit_dict(unit, schema_path=schema_path)
        u = EvidenceUnit(**unit)
        if u.id in self._by_id:
            # update existing in-place
            for i, old in enumerate(self.units):
                if old.id == u.id:
                    self.units[i] = u
                    break
        else:
            self.units.append(u)
        self._by_id[u.id] = u

    def match(self, features: Dict[str, Any]) -> List[EvidenceUnit]:
        # Use the inverted index to narrow candidates when possible.
        candidate_ids: Optional[set[str]] = None
        for k, v in (features or {}).items():
            if isinstance(v, (str, int, float, bool)):
                key = (str(k), str(v))
                if key in self._index:
                    ids = set(self._index[key])
                    candidate_ids = ids if candidate_ids is None else (candidate_ids & ids)

        if candidate_ids is None:
            candidates = self.units
        else:
            candidates = [self._by_id[i] for i in candidate_ids if i in self._by_id]

        out: List[EvidenceUnit] = []
        for u in candidates:
            if _matches(u.when, features):
                out.append(u)
        return out

