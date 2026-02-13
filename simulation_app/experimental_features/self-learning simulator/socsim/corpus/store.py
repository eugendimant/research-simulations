from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from jsonschema import validate as js_validate

@dataclass(frozen=True)
class AtomicUnit:
    id: str
    kind: str
    title: str
    source: Dict[str, Any]
    tags: List[str]
    payload: Dict[str, Any]
    provenance: Dict[str, Any]

class CorpusStore:
    def __init__(self, units: List[AtomicUnit]):
        self.units = units
        self._by_id = {u.id: u for u in units}
        self._by_ref = {}
        for u in units:
            src = u.source or {}
            ref = (src.get('ref') or '').strip().lower()
            if ref:
                self._by_ref[ref] = u.id

    @staticmethod
    def load(path: Path) -> "CorpusStore":
        if not path.exists():
            return CorpusStore([])
        data = json.loads(path.read_text(encoding="utf-8"))
        units = [AtomicUnit(**x) for x in data.get("atomic_units", [])]
        return CorpusStore(units)

    def save(self, path: Path) -> None:
        data = {"atomic_units": [u.__dict__ for u in self.units]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def validate_unit_dict(unit: Dict[str, Any], schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        js_validate(instance=unit, schema=schema)

    def add_dict(self, unit: Dict[str, Any], schema_path: Path) -> None:
        self.validate_unit_dict(unit, schema_path=schema_path)
        u = AtomicUnit(**unit)
        if u.id in self._by_id:
            for i, old in enumerate(self.units):
                if old.id == u.id:
                    self.units[i] = u
                    break
        else:
            self.units.append(u)
        self._by_id[u.id] = u
