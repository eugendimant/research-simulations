from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from ..validate import validate_json

@dataclass
class InsightStore:
    items: List[Dict[str, Any]]

    @staticmethod
    def load(path: Path) -> "InsightStore":
        if not path.exists():
            return InsightStore(items=[])
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            return InsightStore(items=list(data["items"]))
        if isinstance(data, list):
            return InsightStore(items=list(data))
        return InsightStore(items=[])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"items": self.items}, indent=2), encoding="utf-8")

    def upsert(self, item: Dict[str, Any], schema_path: Optional[Path] = None) -> None:
        if schema_path:
            validate_json(item, schema_path)
        iid = item.get("id")
        for i, it in enumerate(self.items):
            if it.get("id") == iid:
                self.items[i] = item
                return
        self.items.append(item)

    def search(self, game: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in self.items:
            sc = it.get("scope") or {}
            games = sc.get("games") or []
            topics = sc.get("topics") or []
            if game and game not in games:
                continue
            if topic and topic not in topics:
                continue
            out.append(it)
        return out
