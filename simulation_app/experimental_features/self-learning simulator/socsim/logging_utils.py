from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import os
import uuid

@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at_utc: str

def new_run() -> RunContext:
    rid = os.environ.get("SOCSIM_RUN_ID") or str(uuid.uuid4())
    t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return RunContext(run_id=rid, started_at_utc=t)

def log_event(event: str, payload: Dict[str, Any], path: Optional[str] = None) -> None:
    rec = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": str(event),
        "payload": payload,
    }
    line = json.dumps(rec, ensure_ascii=False)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
