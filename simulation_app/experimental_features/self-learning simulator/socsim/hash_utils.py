from __future__ import annotations
import json, hashlib
from typing import Any, Dict

def stable_json_sha256(obj: Dict[str, Any]) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
