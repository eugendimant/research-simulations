"""Automatic error logging pipeline for generation failures.

v1.2.0.4: Captures all generation errors with full context (traceback, session
state snapshot, timing, generation method, etc.) and stores them as JSON in
the _error_logs/ directory.  On software updates, the CLAUDE.md-guided agent
reads these logs and implements fixes.

Usage:
    from utils.error_logger import log_generation_error, get_error_summary, get_pending_errors

    # In the generation except handler:
    log_generation_error(exception, context_dict)

    # On startup (check for pending errors to fix):
    pending = get_pending_errors()
    summary = get_error_summary()
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__version__ = "1.2.0.6"

# Directory for error logs — sibling to the app directory
_ERROR_LOG_DIR = Path(__file__).resolve().parent.parent / "_error_logs"
_ERROR_LOG_DIR.mkdir(exist_ok=True)

# Index file for quick lookups
_INDEX_FILE = _ERROR_LOG_DIR / "_index.json"

# Max errors to keep (prevent disk bloat)
_MAX_ERRORS = 200

# Keys to NEVER log from session state (sensitive data)
_SENSITIVE_KEYS = frozenset({
    "user_llm_api_key", "user_groq_api_key", "LLM_API_KEY",
    "admin_password", "_admin_auth",
    "prereg_pdf_content", "qsf_raw_content",  # binary blobs
    "last_zip",  # large binary
})


def _safe_serialize(obj: Any, depth: int = 0) -> Any:
    """Recursively convert obj to JSON-serializable form, truncating large values."""
    if depth > 5:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:500] if len(obj) > 500 else obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, (list, tuple)):
        if len(obj) > 20:
            return [_safe_serialize(v, depth + 1) for v in obj[:20]] + [f"... +{len(obj) - 20} more"]
        return [_safe_serialize(v, depth + 1) for v in obj]
    if isinstance(obj, dict):
        result = {}
        for k, v in list(obj.items())[:30]:
            sk = str(k)
            if sk in _SENSITIVE_KEYS:
                result[sk] = "<redacted>"
            else:
                result[sk] = _safe_serialize(v, depth + 1)
        if len(obj) > 30:
            result["__truncated__"] = f"+{len(obj) - 30} keys"
        return result
    # Fallback
    try:
        s = str(obj)
        return s[:200] if len(s) > 200 else s
    except Exception:
        return f"<{type(obj).__name__}>"


def _error_fingerprint(error_type: str, error_message: str, tb_text: str) -> str:
    """Generate a stable fingerprint for deduplication.

    Uses the error type + the last frame of the traceback (where the error
    actually occurred) so that the same root cause maps to the same fingerprint
    even if the full traceback varies slightly.
    """
    # Extract the last meaningful frame from traceback
    lines = [l for l in tb_text.strip().splitlines() if l.strip().startswith("File ")]
    last_frame = lines[-1].strip() if lines else ""
    raw = f"{error_type}|{error_message[:100]}|{last_frame}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_index() -> Dict[str, Any]:
    """Load or create the error index."""
    if _INDEX_FILE.exists():
        try:
            return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"errors": [], "total_logged": 0, "last_updated": None}


def _save_index(index: Dict[str, Any]) -> None:
    """Save the error index."""
    index["last_updated"] = datetime.now().isoformat()
    try:
        _INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception:
        pass


def log_generation_error(
    exception: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
    session_state_snapshot: Optional[Dict[str, Any]] = None,
    generation_method: str = "",
    app_version: str = "",
    phase: str = "generation",
    traceback_text: str = "",
) -> Optional[str]:
    """Log a generation error with full context.

    Returns the error ID (fingerprint) or None if logging failed.
    """
    try:
        error_type = type(exception).__name__
        error_message = str(exception)
        if not traceback_text:
            traceback_text = traceback.format_exc()

        fingerprint = _error_fingerprint(error_type, error_message, traceback_text)
        timestamp = datetime.now().isoformat()

        # Build the error record
        record: Dict[str, Any] = {
            "id": fingerprint,
            "timestamp": timestamp,
            "unix_time": time.time(),
            "error_type": error_type,
            "error_message": error_message[:1000],
            "traceback": traceback_text[:5000],
            "phase": phase,
            "generation_method": generation_method,
            "app_version": app_version,
            "status": "pending",  # pending | acknowledged | fixed
            "fix_version": None,  # version that fixed this
        }

        # Add context
        if context:
            record["context"] = _safe_serialize(context)

        # Add session state snapshot (sanitized)
        if session_state_snapshot:
            _snap = {}
            for k, v in session_state_snapshot.items():
                if k in _SENSITIVE_KEYS:
                    continue
                if k.startswith("_") and k not in (
                    "_generation_phase", "_use_socsim_experimental",
                    "_llm_exhausted_pending", "_llm_exhausted_resume",
                ):
                    continue
                _snap[k] = _safe_serialize(v)
            record["session_state"] = _snap

        # Write individual error file
        error_file = _ERROR_LOG_DIR / f"error_{fingerprint}_{int(time.time())}.json"
        error_file.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

        # Update index
        index = _load_index()
        # Check for duplicate fingerprint
        existing = [e for e in index["errors"] if e.get("id") == fingerprint]
        if existing:
            existing[0]["count"] = existing[0].get("count", 1) + 1
            existing[0]["last_seen"] = timestamp
        else:
            index["errors"].append({
                "id": fingerprint,
                "error_type": error_type,
                "error_message": error_message[:200],
                "first_seen": timestamp,
                "last_seen": timestamp,
                "count": 1,
                "status": "pending",
                "phase": phase,
                "generation_method": generation_method,
            })
        index["total_logged"] = index.get("total_logged", 0) + 1

        # Trim old entries if over limit
        if len(index["errors"]) > _MAX_ERRORS:
            # Keep most recent + highest count
            index["errors"].sort(key=lambda e: (e.get("status") == "pending", e.get("count", 0), e.get("last_seen", "")), reverse=True)
            index["errors"] = index["errors"][:_MAX_ERRORS]

        _save_index(index)
        return fingerprint

    except Exception:
        # Error logging itself should NEVER crash the app
        return None


def get_pending_errors() -> List[Dict[str, Any]]:
    """Get all pending (unfixed) errors, newest first.

    Called on software update to identify errors that need fixing.
    """
    index = _load_index()
    pending = [e for e in index.get("errors", []) if e.get("status") == "pending"]
    pending.sort(key=lambda e: e.get("last_seen", ""), reverse=True)
    return pending


def get_error_summary() -> Dict[str, Any]:
    """Get a summary of all logged errors for diagnostics.

    Returns:
        Dict with total_errors, pending_count, fixed_count, top_errors,
        and error_by_phase breakdown.
    """
    index = _load_index()
    errors = index.get("errors", [])

    pending = [e for e in errors if e.get("status") == "pending"]
    fixed = [e for e in errors if e.get("status") == "fixed"]
    acknowledged = [e for e in errors if e.get("status") == "acknowledged"]

    # Group by phase
    by_phase: Dict[str, int] = {}
    for e in pending:
        phase = e.get("phase", "unknown")
        by_phase[phase] = by_phase.get(phase, 0) + e.get("count", 1)

    # Top errors by frequency
    top_errors = sorted(pending, key=lambda e: e.get("count", 0), reverse=True)[:10]

    return {
        "total_logged": index.get("total_logged", 0),
        "unique_errors": len(errors),
        "pending_count": len(pending),
        "fixed_count": len(fixed),
        "acknowledged_count": len(acknowledged),
        "errors_by_phase": by_phase,
        "top_errors": top_errors,
        "last_updated": index.get("last_updated"),
    }


def get_error_detail(error_id: str) -> Optional[Dict[str, Any]]:
    """Get full detail for a specific error by its fingerprint ID.

    Reads the most recent error file for this fingerprint.
    """
    try:
        files = sorted(_ERROR_LOG_DIR.glob(f"error_{error_id}_*.json"), reverse=True)
        if files:
            return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def mark_error_fixed(error_id: str, fix_version: str) -> bool:
    """Mark an error as fixed in a specific version.

    Called after implementing a fix. Updates the index and the error file.
    """
    try:
        index = _load_index()
        for e in index.get("errors", []):
            if e.get("id") == error_id:
                e["status"] = "fixed"
                e["fix_version"] = fix_version
                break
        _save_index(index)

        # Also update the individual error file
        files = sorted(_ERROR_LOG_DIR.glob(f"error_{error_id}_*.json"), reverse=True)
        if files:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            data["status"] = "fixed"
            data["fix_version"] = fix_version
            files[0].write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


def mark_error_acknowledged(error_id: str) -> bool:
    """Mark an error as acknowledged (seen but not yet fixed)."""
    try:
        index = _load_index()
        for e in index.get("errors", []):
            if e.get("id") == error_id:
                e["status"] = "acknowledged"
                break
        _save_index(index)
        return True
    except Exception:
        return False


def generate_fix_report() -> str:
    """Generate a report of pending errors suitable for an AI agent to fix.

    This is the key function that bridges error logging and self-healing:
    it produces a structured report that the CLAUDE.md protocol can consume
    during software updates.

    Returns:
        Markdown-formatted report of pending errors with full context.
    """
    pending = get_pending_errors()
    if not pending:
        return "# Error Report\n\nNo pending errors found. All clear!"

    lines = [
        "# Pending Error Report",
        f"\n**Generated:** {datetime.now().isoformat()}",
        f"**Pending errors:** {len(pending)}\n",
        "---\n",
    ]

    for i, err in enumerate(pending, 1):
        detail = get_error_detail(err["id"])
        lines.append(f"## Error #{i}: {err.get('error_type', 'Unknown')}")
        lines.append(f"- **ID:** `{err['id']}`")
        lines.append(f"- **Count:** {err.get('count', 1)} occurrence(s)")
        lines.append(f"- **Phase:** {err.get('phase', 'unknown')}")
        lines.append(f"- **Method:** {err.get('generation_method', 'unknown')}")
        lines.append(f"- **First seen:** {err.get('first_seen', 'unknown')}")
        lines.append(f"- **Last seen:** {err.get('last_seen', 'unknown')}")
        lines.append(f"- **Message:** `{err.get('error_message', 'N/A')}`")

        if detail:
            tb = detail.get("traceback", "")
            if tb:
                lines.append(f"\n### Traceback\n```python\n{tb[:3000]}\n```\n")
            ctx = detail.get("context", {})
            if ctx:
                lines.append(f"### Context\n```json\n{json.dumps(ctx, indent=2, default=str)[:2000]}\n```\n")
            ss = detail.get("session_state", {})
            if ss:
                lines.append(f"### Session State (relevant keys)\n```json\n{json.dumps(ss, indent=2, default=str)[:2000]}\n```\n")

        lines.append("---\n")

    lines.append("\n## Instructions for Fix Agent\n")
    lines.append("1. For each error above, identify the root cause from the traceback and context.")
    lines.append("2. Implement a fix in the relevant file(s).")
    lines.append("3. After fixing, call `mark_error_fixed(error_id, version)` for each fixed error.")
    lines.append("4. Run tests to verify the fix doesn't introduce regressions.")

    return "\n".join(lines)
