# simulation_app/app.py
"""
Behavioral Experiment Simulation Tool (Streamlit App)
============================================================

Design goal: students should be able to run a simulation quickly with minimal
manual configuration. The app infers as much as possible from a Qualtrics QSF
export. Advanced mode exposes additional controls for power users.

Deployment:
- This file is intended to be used as `simulation_app/app.py` on Streamlit Cloud.
- The `utils/` directory must be a Python package (must contain __init__.py).

Email (Free SMTP - Gmail, Outlook, etc.):
- For automatic email delivery, set these Streamlit secrets:
  - SMTP_SERVER (e.g., "smtp.gmail.com")
  - SMTP_PORT (e.g., 587)
  - SMTP_USERNAME (your email address)
  - SMTP_PASSWORD (app password for Gmail)
  - SMTP_FROM_EMAIL (sender email)
  - (optional) INSTRUCTOR_NOTIFICATION_EMAIL

Gmail Setup:
1. Go to Google Account > Security > 2-Step Verification (enable if not already)
2. Go to Google Account > Security > App passwords
3. Create a new app password for "Mail"
4. Use that 16-character password as SMTP_PASSWORD
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as _st_components

# =============================================================================
# MODULE VERSION VERIFICATION
# =============================================================================
# This section ensures Streamlit Cloud loads the correct module versions.
# Addresses known issue: https://github.com/streamlit/streamlit/issues/366
# Where deeply imported modules don't hot-reload properly.

REQUIRED_UTILS_VERSION = "1.0.6.8"
BUILD_ID = "20260212-v10608-fix-missing-logger-import"  # Change this to force cache invalidation

# NOTE: Previously _verify_and_reload_utils() purged utils.* from sys.modules
# before every import.  This caused KeyError crashes on Streamlit Cloud when
# concurrent sessions imported the same modules simultaneously.  Removed in
# v1.4.3.1 — the version-check on line ~103 already warns on mismatch, and
# BUILD_ID changes handle cache invalidation safely.

from utils.group_management import GroupManager, APIKeyManager
from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult
from utils.schema_validator import validate_schema
from utils.github_qsf_collector import collect_qsf_async, is_collection_enabled
from utils.instructor_report import InstructorReportGenerator, ComprehensiveInstructorReport
from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedCondition, ParsedScale, KNOWN_SCALES, AVAILABLE_DOMAINS, generate_qsf_from_design
from utils.persona_library import PersonaLibrary, Persona
from utils.enhanced_simulation_engine import (
    EnhancedSimulationEngine,
    EffectSizeSpec,
    ExclusionCriteria,
)
from utils.condition_identifier import (
    DesignAnalysisResult,
    VariableRole,
    analyze_qsf_design,
)
import utils

# v1.8.8.0: Correlation matrix for cross-DV correlations
try:
    from utils.correlation_matrix import (
        infer_correlation_matrix,
        detect_construct_types,
        get_correlation_summary,
    )
    _HAS_CORRELATION_MODULE = True
except ImportError:
    _HAS_CORRELATION_MODULE = False

# Verify correct version loaded — if stale, force a targeted reload
if hasattr(utils, '__version__') and utils.__version__ != REQUIRED_UTILS_VERSION:
    try:
        # Purge stale utils sub-modules from sys.modules so fresh code is loaded
        _stale_keys = [k for k in sys.modules if k == "utils" or k.startswith("utils.")]
        for _k in _stale_keys:
            del sys.modules[_k]
        # Re-import with fresh code
        import utils  # noqa: F811
        from utils.group_management import GroupManager, APIKeyManager  # noqa: F811
        from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult  # noqa: F811
        from utils.schema_validator import validate_schema  # noqa: F811
        from utils.github_qsf_collector import collect_qsf_async, is_collection_enabled  # noqa: F811
        from utils.instructor_report import InstructorReportGenerator, ComprehensiveInstructorReport  # noqa: F811
        from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedCondition, ParsedScale, KNOWN_SCALES, AVAILABLE_DOMAINS, generate_qsf_from_design  # noqa: F811
        from utils.persona_library import PersonaLibrary, Persona  # noqa: F811
        from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec, ExclusionCriteria  # noqa: F811
        from utils.condition_identifier import DesignAnalysisResult, VariableRole, analyze_qsf_design  # noqa: F811
    except Exception:
        st.warning(f"Utils version mismatch: expected {REQUIRED_UTILS_VERSION}, got {getattr(utils, '__version__', '?')}. Please restart the app.")


# -----------------------------
# App constants
# -----------------------------
APP_TITLE = "Behavioral Experiment Simulation Tool"
APP_SUBTITLE = "Fast, standardized pilot simulations from your Qualtrics QSF or study description"
APP_VERSION = "1.0.6.8"  # v1.0.6.8: Fix missing logger import in simulation engine
APP_BUILD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

BASE_STORAGE = Path("data")
BASE_STORAGE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# v1.0.5.8: Session-scoped API key encryption helpers
# Keys are XOR-encrypted with a random session key so they're never stored
# as plaintext in st.session_state (visible in admin Session State Explorer).
# The session key exists only in memory for the browser session lifetime.
# ---------------------------------------------------------------------------
import secrets as _secrets

def _get_session_cipher_key() -> bytes:
    """Return a 32-byte random key unique to this browser session."""
    _k = st.session_state.get("_cipher_key_bytes")
    if _k is None:
        _k = _secrets.token_bytes(32)
        st.session_state["_cipher_key_bytes"] = _k
    return _k

def _encrypt_api_key(plaintext: str) -> str:
    """XOR-encrypt an API key for session storage. Returns hex string."""
    if not plaintext:
        return ""
    key = _get_session_cipher_key()
    data = plaintext.encode("utf-8")
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return encrypted.hex()

def _decrypt_api_key(ciphertext_hex: str) -> str:
    """Decrypt a session-encrypted API key. Returns plaintext."""
    if not ciphertext_hex:
        return ""
    try:
        key = _get_session_cipher_key()
        data = bytes.fromhex(ciphertext_hex)
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        return decrypted.decode("utf-8")
    except Exception:
        return ""

MAX_SIMULATED_N = 10000

STANDARD_DEFAULTS = {
    "demographics": {"gender_quota": 50, "age_mean": 35, "age_sd": 12},
    "attention_rate": 0.95,
    "random_responder_rate": 0.05,
    "exclusion_criteria": {
        "attention_check_threshold": 0.0,
        "completion_time_min_seconds": 60,
        "completion_time_max_seconds": 1800,
        "straight_line_threshold": 10,
        "duplicate_ip_check": True,
        "exclude_careless_responders": False,
    },
}

ADVANCED_DEFAULTS = {
    "demographics": {"gender_quota": 50, "age_mean": 35, "age_sd": 12},
    "attention_rate": 0.95,
    "random_responder_rate": 0.05,
}


# -----------------------------
# Utilities
# -----------------------------
def _safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


# v1.5.0: Removed unused validation helpers, SimulationError class, and related
# dead code (~180 lines). Validation is handled inline where needed.



def _markdown_to_html(markdown_text: str, title: str = "Study Summary") -> str:
    """
    Convert markdown text to a well-formatted, standalone HTML document.
    Uses simple regex-based conversion for common markdown patterns.
    """
    import re

    # Start HTML document with styling
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>{title}</title>',
        '<style>',
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; ',
        '       max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }',
        'h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }',
        'h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }',
        'h3 { color: #7f8c8d; margin-top: 25px; }',
        'table { border-collapse: collapse; width: 100%; margin: 15px 0; }',
        'th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }',
        'th { background-color: #f8f9fa; font-weight: 600; }',
        'tr:nth-child(even) { background-color: #f9f9f9; }',
        'code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }',
        'pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }',
        'blockquote { border-left: 4px solid #3498db; margin: 15px 0; padding: 10px 20px; background: #f8f9fa; }',
        'ul, ol { padding-left: 25px; }',
        'li { margin: 5px 0; }',
        '.info-box { background: #e8f4fd; border: 1px solid #3498db; border-radius: 5px; padding: 15px; margin: 15px 0; }',
        '.warning-box { background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin: 15px 0; }',
        '@media print { body { max-width: 100%; margin: 20px; } }',
        '</style>',
        '</head>',
        '<body>',
    ]

    content = markdown_text

    # Convert headers
    content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)

    # Convert bold and italic
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)

    # Convert tables (simple markdown tables)
    lines = content.split('\n')
    in_table = False
    new_lines = []
    for line in lines:
        if '|' in line and not line.strip().startswith('```'):
            cells = [c.strip() for c in line.split('|')[1:-1]]  # Remove empty first/last
            if cells:
                if all(c.replace('-', '').replace(':', '') == '' for c in cells):
                    # This is a separator row, skip it
                    continue
                if not in_table:
                    new_lines.append('<table>')
                    # First row is header
                    new_lines.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
                    in_table = True
                else:
                    new_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            else:
                if in_table:
                    new_lines.append('</table>')
                    in_table = False
                new_lines.append(line)
        else:
            if in_table:
                new_lines.append('</table>')
                in_table = False
            new_lines.append(line)
    if in_table:
        new_lines.append('</table>')
    content = '\n'.join(new_lines)

    # Convert bullet lists
    content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
    content = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', content)

    # Convert line breaks to paragraphs (for non-HTML content)
    paragraphs = []
    current_para = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('<') or not line:
            if current_para:
                paragraphs.append('<p>' + ' '.join(current_para) + '</p>')
                current_para = []
            if line:
                paragraphs.append(line)
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append('<p>' + ' '.join(current_para) + '</p>')

    content = '\n'.join(paragraphs)

    html_parts.append(content)
    html_parts.append('</body>')
    html_parts.append('</html>')

    return '\n'.join(html_parts)


def _bytes_to_zip(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ========================================
# INTERNAL USAGE COUNTER (for admin tracking only)
# ========================================
USAGE_COUNTER_FILE = Path(__file__).resolve().parent / ".usage_counter.json"

def _get_usage_count() -> Dict[str, Any]:
    """Get the current usage statistics from the counter file."""
    try:
        if USAGE_COUNTER_FILE.exists():
            with open(USAGE_COUNTER_FILE, "r") as f:
                data = json.load(f)
                return data
        return {"total_simulations": 0, "simulations_by_date": {}, "first_use": None, "last_use": None}
    except Exception:
        return {"total_simulations": 0, "simulations_by_date": {}, "first_use": None, "last_use": None}


def _increment_usage_counter() -> Dict[str, Any]:
    """Increment the usage counter and return updated stats."""
    try:
        stats = _get_usage_count()
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        stats["total_simulations"] = stats.get("total_simulations", 0) + 1

        if "simulations_by_date" not in stats:
            stats["simulations_by_date"] = {}
        stats["simulations_by_date"][today] = stats["simulations_by_date"].get(today, 0) + 1

        if not stats.get("first_use"):
            stats["first_use"] = now.isoformat()
        stats["last_use"] = now.isoformat()

        # Save to file
        with open(USAGE_COUNTER_FILE, "w") as f:
            json.dump(stats, f, indent=2)

        return stats
    except Exception as e:
        # If we can't write to file (e.g., read-only filesystem), just return current count
        return {"total_simulations": "unknown", "error": str(e)}


def _get_usage_summary() -> str:
    """Get a formatted summary of usage statistics for email."""
    stats = _get_usage_count()
    total = stats.get("total_simulations", 0)
    first_use = stats.get("first_use", "N/A")
    last_use = stats.get("last_use", "N/A")

    # Get recent daily counts
    by_date = stats.get("simulations_by_date", {})
    recent_dates = sorted(by_date.keys(), reverse=True)[:7]  # Last 7 days with activity
    recent_summary = "\n".join([f"  {d}: {by_date[d]} simulation(s)" for d in recent_dates]) if recent_dates else "  No recent data"

    return f"""
=== INTERNAL USAGE STATS (admin only) ===
Total simulations: {total}
First use: {first_use}
Last use: {last_use}
Recent activity:
{recent_summary}
========================================="""


def _sanitize_prereg_text(raw_text: str) -> Tuple[str, List[str]]:
    """
    Remove hypothesis-like language to avoid biasing simulation settings.

    Returns sanitized text and list of removed lines.
    """
    if not raw_text:
        return "", []

    removed: List[str] = []
    kept: List[str] = []
    flagged_tokens = ("hypothesis", "hypotheses", "predict", "prediction", "expect", "expectation", "h1", "h2", "h3")
    for line in raw_text.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if any(token in normalized.lower() for token in flagged_tokens):
            removed.append(line)
        else:
            kept.append(line)
    return "\n".join(kept).strip(), removed


def _split_comma_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_simulation_output(df: pd.DataFrame, metadata: dict, scales: list) -> Dict[str, Any]:
    """
    v1.2.4: Simulation quality validation gate.

    This function MUST be called after every simulation to verify data quality.
    It serves as the automated "self-improvement workflow" that ensures
    simulations are never broken and data quality is maintained with every version.

    Returns a dict with quality metrics and pass/fail status.
    """
    results = {
        "passed": True,
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    # CHECK 1: DataFrame is not empty
    if df is None or df.empty:
        results["passed"] = False
        results["errors"].append("DataFrame is empty - no data generated")
        return results
    results["checks"].append(f"✅ Data generated: {df.shape[0]} rows × {df.shape[1]} columns")

    # CHECK 2: Expected columns exist
    expected_cols = ["PARTICIPANT_ID", "CONDITION"]
    for col in expected_cols:
        if col in df.columns:
            results["checks"].append(f"✅ Column '{col}' present")
        else:
            results["passed"] = False
            results["errors"].append(f"❌ Missing required column: {col}")

    # CHECK 3: Scale values within bounds
    # v1.0.6.1: Also try variable_name for column lookup (engine prefers it)
    import re as _re_dqc
    def _dqc_clean_col(name: str) -> str:
        s = str(name).strip().replace(' ', '_')
        s = _re_dqc.sub(r'[^A-Za-z0-9_]', '_', s)
        return _re_dqc.sub(r'_+', '_', s).strip('_') or 'Scale'

    for scale in (scales or []):
        _raw_name = scale.get("variable_name", "") or scale.get("name", "")
        s_name = _dqc_clean_col(_raw_name)
        s_min = scale.get("scale_min", 1)
        s_max = scale.get("scale_max", 7)
        n_items = scale.get("num_items", 5)
        try:
            s_min = int(s_min) if not isinstance(s_min, dict) else 1
            s_max = int(s_max) if not isinstance(s_max, dict) else 7
            n_items = int(n_items) if not isinstance(n_items, dict) else 5
        except (ValueError, TypeError):
            continue

        for item_num in range(1, n_items + 1):
            col = f"{s_name}_{item_num}"
            # Also try simple name variant
            _alt = str(scale.get("name", "")).strip().replace(" ", "_")
            _alt_col = f"{_alt}_{item_num}"
            if col not in df.columns and _alt_col in df.columns:
                col = _alt_col
            if col in df.columns:
                try:
                    actual_min = int(df[col].min())
                    actual_max = int(df[col].max())
                    if actual_min < s_min or actual_max > s_max:
                        results["warnings"].append(
                            f"⚠️ {col}: values [{actual_min}-{actual_max}] outside expected [{s_min}-{s_max}]"
                        )
                    else:
                        results["checks"].append(f"✅ {col}: range [{actual_min}-{actual_max}] within bounds")
                except Exception as _val_err:
                    results["warnings"].append(f"⚠️ {col}: validation check skipped ({_val_err})")

    # CHECK 4: Open-ended response uniqueness (>= 90% unique)
    # v1.0.6.1: Expanded exclusion list to avoid checking non-OE text columns
    oe_cols = [c for c in df.columns if df[c].dtype == object and c not in [
        'CONDITION', 'PARTICIPANT_ID', 'RUN_ID', 'SIMULATION_MODE', 'SIMULATION_SEED',
        'Gender', '_PERSONA', 'EXCLUSION_REASON',
    ]]
    for col in oe_cols:
        responses = df[col].dropna().tolist()
        if responses:
            unique_pct = len(set(responses)) / len(responses) * 100
            if unique_pct < 90:
                results["warnings"].append(f"⚠️ {col}: only {unique_pct:.1f}% unique responses")
            else:
                results["checks"].append(f"✅ {col}: {unique_pct:.1f}% unique responses")

    # CHECK 5: Condition balance
    if "CONDITION" in df.columns:
        condition_counts = df["CONDITION"].value_counts()
        if len(condition_counts) > 1:
            balance_ratio = condition_counts.min() / condition_counts.max()
            if balance_ratio < 0.8:
                results["warnings"].append(
                    f"⚠️ Condition imbalance: ratio {balance_ratio:.2f} (counts: {dict(condition_counts)})"
                )
            else:
                results["checks"].append(f"✅ Condition balance: ratio {balance_ratio:.2f}")

    # CHECK 6: No NaN in scale columns
    scale_cols = [c for c in df.columns if any(
        c.startswith(str(s.get("name", "")).strip().replace(" ", "_") + "_")
        for s in (scales or [])
    )]
    nan_count = df[scale_cols].isna().sum().sum() if scale_cols else 0
    if nan_count > 0:
        results["warnings"].append(f"⚠️ Found {nan_count} NaN values in scale columns")
    else:
        results["checks"].append(f"✅ No NaN values in scale columns")

    # CHECK 7: Metadata integrity
    if metadata:
        if "persona_distribution" in metadata:
            results["checks"].append("✅ Persona distribution recorded in metadata")
        if "effect_sizes_observed" in metadata:
            results["checks"].append("✅ Observed effect sizes recorded")

    return results


# Map builder scale types to engine-expected types
# The conversational builder uses "likert", "slider", "numeric", "binary"
# but the engine and QSF path use "matrix", "slider", "numeric_input", "single_item"
_BUILDER_TO_ENGINE_TYPE: Dict[str, str] = {
    "likert": "matrix",         # Multi-item Likert -> matrix
    "slider": "slider",         # Slider stays slider
    "numeric": "numeric_input", # Numeric input
    "binary": "single_item",    # Binary -> single item
}


def _normalize_scale_specs(scales: List[Any]) -> List[Dict[str, Any]]:
    """
    SINGLE SOURCE OF TRUTH for scale normalization.

    This is the ONE place where scale specs are validated and normalized.
    After this function, every scale dict is guaranteed to have:
      - name: str (non-empty)
      - variable_name: str (non-empty, underscore-separated)
      - num_items: int >= 1
      - scale_points: int >= 2 and <= 1001
      - reverse_items: list
      - type: str (engine-compatible scale type)
      - _validated: True (contract flag - engine MUST NOT re-default these)

    Preserves scale_points from source (QSF or user input) - only defaults when missing.
    Maps builder scale types to engine-expected types via _BUILDER_TO_ENGINE_TYPE.
    Deduplicates by variable name AND display name to prevent extra DVs
    and column collisions.
    """
    normalized: List[Dict[str, Any]] = []
    seen_names: set = set()  # Track to prevent duplicates by variable_name
    seen_display: set = set()  # Track to prevent duplicates by display_name

    for scale in scales or []:
        if isinstance(scale, str):
            name = scale.strip()
            if not name:
                continue
            name_key = name.lower().replace(" ", "_").replace("-", "_")
            if name_key in seen_names or name_key in seen_display:
                continue
            seen_names.add(name_key)
            seen_display.add(name_key)
            normalized.append({
                "name": name,
                "variable_name": name.replace(" ", "_"),
                "num_items": 5,
                "scale_points": 7,
                "reverse_items": [],
                "_validated": True,
            })
            continue

        if isinstance(scale, dict):
            name = str(scale.get("name", "")).strip()
            if not name:
                continue

            # Deduplicate by variable name AND display name to prevent column collisions
            var_name = str(scale.get("variable_name", name)).strip() or name
            name_key = var_name.lower().replace(" ", "_").replace("-", "_")
            display_key = name.lower().replace(" ", "_").replace("-", "_")
            if name_key in seen_names or display_key in seen_display:
                continue
            seen_names.add(name_key)
            seen_display.add(display_key)

            # Carefully extract scale_points - preserve from source
            # v1.4.0: Handle dict-contaminated values (e.g. {"value": 7})
            raw_points = scale.get("scale_points")
            if raw_points is None or (isinstance(raw_points, float) and np.isnan(raw_points)):
                # For numeric/slider scales, compute from scale_min/scale_max if available
                scale_type = str(scale.get("type", "likert")).lower()
                if scale_type in ("numeric", "slider", "number"):
                    # Numeric scales: scale_points = range + 1 (or default 101 for sliders)
                    _raw_max = scale.get("scale_max")
                    _raw_min = scale.get("scale_min")
                    if _raw_max is not None and _raw_min is not None:
                        try:
                            _p_max = int(_raw_max) if not isinstance(_raw_max, dict) else int(_raw_max.get("value", 100))
                            _p_min = int(_raw_min) if not isinstance(_raw_min, dict) else int(_raw_min.get("value", 0))
                            scale_points = max(2, _p_max - _p_min + 1)
                        except (ValueError, TypeError):
                            scale_points = 101 if scale_type == "slider" else 7
                    else:
                        scale_points = 101 if scale_type == "slider" else 7
                else:
                    scale_points = 7
            elif isinstance(raw_points, dict):
                # Handle dict-contaminated scale_points (e.g. {"value": 7, "label": "..."})
                try:
                    scale_points = int(raw_points.get("value", 7)) if "value" in raw_points else 7
                except (ValueError, TypeError):
                    scale_points = 7
            elif isinstance(raw_points, (list, tuple)):
                # Handle list-contaminated values - take first element or default
                try:
                    scale_points = int(raw_points[0]) if raw_points else 7
                except (ValueError, TypeError, IndexError):
                    scale_points = 7
            else:
                try:
                    scale_points = int(raw_points)
                except (ValueError, TypeError):
                    scale_points = 7

            # Extract num_items - check BOTH "num_items" and "items" keys for compatibility
            # v1.4.0: Handle dict-contaminated values
            raw_items = scale.get("num_items")
            if raw_items is None:
                raw_items = scale.get("items")  # Fallback to QSF detection key
            if raw_items is None or (isinstance(raw_items, float) and np.isnan(raw_items)):
                num_items = 5
            elif isinstance(raw_items, dict):
                # Handle dict-contaminated num_items
                try:
                    num_items = int(raw_items.get("value", 5)) if "value" in raw_items else 5
                except (ValueError, TypeError):
                    num_items = 5
            elif isinstance(raw_items, (list, tuple)):
                # Handle list-contaminated values
                try:
                    num_items = int(raw_items[0]) if raw_items else 5
                except (ValueError, TypeError, IndexError):
                    num_items = 5
            else:
                try:
                    num_items = int(raw_items)
                except (ValueError, TypeError):
                    num_items = 5

            # v1.2.1: Extract scale_min and scale_max with proper numeric conversion
            raw_min = scale.get("scale_min")
            if raw_min is None or (isinstance(raw_min, float) and np.isnan(raw_min)):
                scale_min = 1
            elif isinstance(raw_min, dict):
                # Handle case where scale_min is incorrectly a dict - extract value or default
                scale_min = int(raw_min.get("value", 1)) if "value" in raw_min else 1
            else:
                try:
                    scale_min = int(raw_min)
                except (ValueError, TypeError):
                    scale_min = 1

            raw_max = scale.get("scale_max")
            if raw_max is None or (isinstance(raw_max, float) and np.isnan(raw_max)):
                scale_max = scale_points  # Default to scale_points
            elif isinstance(raw_max, dict):
                # Handle case where scale_max is incorrectly a dict - extract value or default
                scale_max = int(raw_max.get("value", scale_points)) if "value" in raw_max else scale_points
            else:
                try:
                    scale_max = int(raw_max)
                except (ValueError, TypeError):
                    scale_max = scale_points

            # v1.4.0: Map builder scale types to engine-expected types
            raw_type = scale.get("type", "likert")
            mapped_type = _BUILDER_TO_ENGINE_TYPE.get(raw_type, raw_type)

            normalized.append(
                {
                    "name": name,
                    "variable_name": var_name.replace(" ", "_"),
                    "num_items": max(1, num_items),
                    "scale_points": max(2, min(1001, scale_points)),
                    "scale_min": max(0, scale_min),  # v1.2.1: Preserve scale_min
                    "scale_max": max(1, scale_max),  # v1.2.1: Preserve scale_max
                    "reverse_items": scale.get("reverse_items", []) or [],
                    "type": mapped_type,  # v1.4.0: Mapped from builder types to engine types
                    "_validated": True,
                }
            )

    return normalized


def _normalize_factor_specs(factors: List[Any], fallback_conditions: List[str]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for factor in factors or []:
        if isinstance(factor, str):
            name = factor.strip()
            if name:
                normalized.append({"name": name, "levels": fallback_conditions})
            continue
        if isinstance(factor, dict):
            name = str(factor.get("name", "")).strip() or "Condition"
            levels = factor.get("levels", fallback_conditions)
            if isinstance(levels, str):
                levels_list = [lvl.strip() for lvl in levels.split(",") if lvl.strip()]
            else:
                levels_list = [str(lvl).strip() for lvl in (levels or []) if str(lvl).strip()]
            normalized.append({"name": name, "levels": levels_list or fallback_conditions})
    return normalized or [{"name": "Condition", "levels": fallback_conditions}]


def _normalize_condition_label(label: str) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", str(label))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^(condition|treatment|group|arm|variant|scenario|manipulation)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -:")


def _extract_conditions_from_prereg(prereg_iv: str, prereg_notes: str, prereg_pdf_text: str) -> List[str]:
    """
    DEPRECATED: This function is no longer used for condition extraction.
    Conditions should ONLY come from QSF Randomizer structure.
    Kept for backwards compatibility but returns empty list.
    """
    return []


# =============================================================================
# v1.0.0: PRE-REGISTRATION PARSING (OSF, AEA, AsPredicted)
# =============================================================================

# Pre-registration format patterns for parsing
PREREG_SECTION_PATTERNS = {
    'osf': {
        'study_info': [
            r'(?:1|A)\.?\s*(?:Study|Title|Registration)',
            r'Study\s*Information',
            r'Study\s*Title',
        ],
        'hypotheses': [
            r'(?:2|B)\.?\s*Hypothes[ie]s',
            r'Prediction[s]?',
            r'Expected\s*Results?',
        ],
        'design': [
            r'(?:3|C)\.?\s*(?:Study|Research)\s*Design',
            r'Design\s*Plan',
            r'Experimental\s*Design',
        ],
        'sample': [
            r'(?:4|D)\.?\s*(?:Sample|Sampling|Data\s*Collection)',
            r'Sample\s*Size',
            r'Participants?',
        ],
        'variables': [
            r'(?:5|E)\.?\s*(?:Variables?|Measures?)',
            r'Manipulated\s*Variables?',
            r'Measured\s*Variables?',
            r'Independent\s*Variables?',
            r'Dependent\s*Variables?',
        ],
        'analysis': [
            r'(?:6|F)\.?\s*Analysis\s*Plan',
            r'Statistical\s*Analysis',
            r'Planned\s*Analyses?',
        ],
        'prereg_number': [
            r'Registration\s*(?:Number|ID|#)',
            r'OSF\s*(?:Prereg|Registration)',
            r'https?://osf\.io/[a-z0-9]+',
        ],
    },
    'aea': {
        'title': [r'Title\s*:?', r'(?:1|I)\.?\s*Title'],
        'investigators': [r'(?:Principal\s*)?Investigators?\s*:?'],
        'sample': [r'Sample\s*Size\s*:?', r'Power\s*Calculation'],
        'outcomes': [r'Primary\s*Outcome', r'Secondary\s*Outcome'],
        'hypotheses': [r'Hypothes[ie]s\s*:?', r'Pre-Analysis\s*Plan'],
        'prereg_number': [
            r'AEA\s*(?:RCT\s*)?Registry',
            r'AEARCTR-\d+',
            r'Registration\s*(?:Number|ID)',
        ],
    },
    'aspredicted': {
        'main_question': [r'(?:1|A)\.?\s*(?:What.s the main question|Main\s*Question)'],
        'dvs': [r'(?:2|B)\.?\s*(?:Dependent\s*Variable|What are you measuring|DV)'],
        'conditions': [r'(?:3|C)\.?\s*(?:Conditions|Treatment|How many conditions)'],
        'analyses': [r'(?:4|D)\.?\s*(?:Analyses|Statistical\s*tests)'],
        'outliers': [r'(?:5|E)\.?\s*(?:Outliers|Exclusion)'],
        'sample_size': [r'(?:6|F)\.?\s*(?:Sample\s*size|How many observations)'],
        'other': [r'(?:7|G)\.?\s*(?:Other|Anything else)'],
        'prereg_number': [
            r'AsPredicted\s*#?\d+',
            r'aspredicted\.org/[a-z0-9]+',
            r'Pre-registration\s*(?:Number|ID)',
        ],
    },
}


def _detect_prereg_format(text: str) -> str:
    """Detect the pre-registration format from text content."""
    text_lower = text.lower()

    # Check for AsPredicted
    if 'aspredicted' in text_lower or 'as predicted' in text_lower:
        return 'aspredicted'

    # Check for AEA Registry
    if 'aea' in text_lower and ('registry' in text_lower or 'rct' in text_lower):
        return 'aea'
    if 'aearctr' in text_lower:
        return 'aea'

    # Check for OSF (most common)
    if 'osf' in text_lower or 'open science framework' in text_lower:
        return 'osf'

    # Check for structural patterns
    aspredicted_count = sum(1 for p in PREREG_SECTION_PATTERNS['aspredicted']['main_question']
                           if re.search(p, text, re.IGNORECASE))
    osf_count = sum(1 for p in PREREG_SECTION_PATTERNS['osf']['hypotheses']
                   if re.search(p, text, re.IGNORECASE))

    if aspredicted_count > osf_count:
        return 'aspredicted'

    return 'osf'  # Default to OSF format


def _extract_prereg_number(text: str, format_type: str) -> Optional[str]:
    """Extract the pre-registration number/ID from text."""
    patterns = PREREG_SECTION_PATTERNS.get(format_type, {}).get('prereg_number', [])

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Try to extract the actual number/ID
            if 'osf.io' in pattern.lower():
                url_match = re.search(r'osf\.io/([a-z0-9]+)', text, re.IGNORECASE)
                if url_match:
                    return f"OSF: {url_match.group(1)}"
            elif 'aearctr' in pattern.lower():
                id_match = re.search(r'AEARCTR-(\d+)', text, re.IGNORECASE)
                if id_match:
                    return f"AEA: AEARCTR-{id_match.group(1)}"
            elif 'aspredicted' in pattern.lower():
                num_match = re.search(r'#?(\d+)', match.group())
                if num_match:
                    return f"AsPredicted: #{num_match.group(1)}"
            return match.group()

    return None


def _parse_prereg_sections(text: str, format_type: str) -> Dict[str, str]:
    """Parse pre-registration text into sections based on format."""
    sections = {}
    patterns = PREREG_SECTION_PATTERNS.get(format_type, {})

    for section_name, section_patterns in patterns.items():
        if section_name == 'prereg_number':
            continue  # Handle separately

        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Extract content after the header until next section or end
                start_pos = match.end()
                # Find the next section header
                next_match = None
                next_pos = len(text)
                for other_name, other_patterns in patterns.items():
                    if other_name == section_name:
                        continue
                    for other_pattern in other_patterns:
                        other_match = re.search(other_pattern, text[start_pos:], re.IGNORECASE)
                        if other_match and (start_pos + other_match.start()) < next_pos:
                            next_pos = start_pos + other_match.start()

                content = text[start_pos:next_pos].strip()
                # Clean up the content
                content = re.sub(r'^\s*[:]\s*', '', content)  # Remove leading colons
                content = content[:2000]  # Limit length
                if content:
                    sections[section_name] = content
                break

    return sections


def _extract_prereg_variables(sections: Dict[str, str]) -> Dict[str, List[str]]:
    """Extract variables from parsed pre-registration sections."""
    variables = {
        'ivs': [],
        'dvs': [],
        'mediators': [],
        'moderators': [],
        'covariates': [],
    }

    # Keywords to identify variable types
    iv_keywords = ['independent', 'manipulat', 'condition', 'treatment', 'iv']
    dv_keywords = ['dependent', 'outcome', 'measure', 'dv']
    mediator_keywords = ['mediat', 'mechanism', 'process']
    moderator_keywords = ['moderat', 'boundary', 'interact']
    covariate_keywords = ['covariate', 'control', 'demographic']

    for section_name, content in sections.items():
        content_lower = content.lower()

        # Check IVs
        if any(kw in section_name.lower() or kw in content_lower for kw in iv_keywords):
            # Extract variable names (simple heuristic)
            lines = content.split('\n')
            for line in lines[:10]:  # Limit to first 10 lines
                line = line.strip()
                if line and len(line) > 3 and len(line) < 100:
                    # Clean up bullet points and numbering
                    clean = re.sub(r'^[\d\.\-\*\•]\s*', '', line)
                    if clean and clean not in variables['ivs']:
                        variables['ivs'].append(clean[:50])

        # Check DVs
        if any(kw in section_name.lower() or kw in content_lower for kw in dv_keywords):
            lines = content.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if line and len(line) > 3 and len(line) < 100:
                    clean = re.sub(r'^[\d\.\-\*\•]\s*', '', line)
                    if clean and clean not in variables['dvs']:
                        variables['dvs'].append(clean[:50])

        # Check mediators
        if any(kw in content_lower for kw in mediator_keywords):
            match = re.search(r'mediat\w*[:\s]+([^\.]+)', content, re.IGNORECASE)
            if match:
                variables['mediators'].append(match.group(1).strip()[:50])

        # Check moderators
        if any(kw in content_lower for kw in moderator_keywords):
            match = re.search(r'moderat\w*[:\s]+([^\.]+)', content, re.IGNORECASE)
            if match:
                variables['moderators'].append(match.group(1).strip()[:50])

    return variables


def _check_prereg_consistency(prereg_data: Dict[str, Any], design_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check consistency between pre-registration and current design."""
    warnings = []

    prereg_vars = prereg_data.get('variables', {})
    design_conditions = design_data.get('conditions', [])
    design_scales = design_data.get('scales', [])

    # Check sample size
    prereg_sample = prereg_data.get('sample_size')
    design_sample = design_data.get('sample_size')
    if prereg_sample and design_sample:
        try:
            _ps, _ds = int(prereg_sample), int(design_sample)
        except (ValueError, TypeError):
            _ps, _ds = 0, 0
        if _ps and _ds and abs(_ps - _ds) > 10:
            warnings.append({
                'type': 'sample_size',
                'severity': 'warning',
                'message': f"Sample size mismatch: Pre-reg={prereg_sample}, Current={design_sample}",
                'recommendation': "Verify the sample size matches your pre-registration"
            })

    # Check conditions
    prereg_ivs = prereg_vars.get('ivs', [])
    if prereg_ivs and design_conditions:
        # Check if any pre-registered IVs are missing
        missing_ivs = []
        for iv in prereg_ivs:
            iv_lower = iv.lower()
            found = any(iv_lower in cond.lower() or cond.lower() in iv_lower
                       for cond in design_conditions)
            if not found:
                missing_ivs.append(iv)
        if missing_ivs:
            warnings.append({
                'type': 'missing_ivs',
                'severity': 'info',
                'message': f"Pre-registered IVs not found in conditions: {', '.join(missing_ivs[:3])}",
                'recommendation': "Verify all pre-registered conditions are included"
            })

    # Check DVs
    prereg_dvs = prereg_vars.get('dvs', [])
    if prereg_dvs and design_scales:
        scale_names = [s.get('name', '').lower() for s in design_scales]
        missing_dvs = []
        for dv in prereg_dvs:
            dv_lower = dv.lower()
            found = any(dv_lower in name or name in dv_lower for name in scale_names)
            if not found:
                missing_dvs.append(dv)
        if missing_dvs:
            warnings.append({
                'type': 'missing_dvs',
                'severity': 'info',
                'message': f"Pre-registered DVs not found in scales: {', '.join(missing_dvs[:3])}",
                'recommendation': "Verify all pre-registered outcome measures are included"
            })

    # Check mediators
    prereg_mediators = prereg_vars.get('mediators', [])
    if prereg_mediators:
        warnings.append({
            'type': 'mediators_detected',
            'severity': 'info',
            'message': f"Pre-registration mentions mediators: {', '.join(prereg_mediators[:3])}",
            'recommendation': "Consider adding mediator variables in the design"
        })

    return warnings


# =============================================================================
# v1.0.0: DIFFICULTY LEVELS FOR DATA QUALITY
# =============================================================================

DIFFICULTY_LEVELS = {
    'easy': {
        'name': 'Easy (Clean Data)',
        'description': 'Minimal noise, clear patterns, high attention rates',
        'attention_rate': 0.98,
        'random_responder_rate': 0.02,
        'careless_rate': 0.02,
        'straightline_rate': 0.03,
        'text_quality': 'high',
        'text_effort': 0.9,
        'numeric_noise': 0.1,
        'effect_clarity': 1.2,  # Amplify effects for clearer patterns
    },
    'medium': {
        'name': 'Medium (Realistic)',
        'description': 'Standard noise levels typical of online samples',
        'attention_rate': 0.92,
        'random_responder_rate': 0.05,
        'careless_rate': 0.05,
        'straightline_rate': 0.08,
        'text_quality': 'medium',
        'text_effort': 0.7,
        'numeric_noise': 0.25,
        'effect_clarity': 1.0,
    },
    'hard': {
        'name': 'Hard (Noisy Data)',
        'description': 'Higher noise, more careless responding, unclear patterns',
        'attention_rate': 0.85,
        'random_responder_rate': 0.10,
        'careless_rate': 0.10,
        'straightline_rate': 0.15,
        'text_quality': 'low',
        'text_effort': 0.5,
        'numeric_noise': 0.4,
        'effect_clarity': 0.8,
    },
    'expert': {
        'name': 'Expert (Very Noisy)',
        'description': 'Challenging data quality requiring advanced cleaning',
        'attention_rate': 0.75,
        'random_responder_rate': 0.15,
        'careless_rate': 0.15,
        'straightline_rate': 0.20,
        'text_quality': 'very_low',
        'text_effort': 0.3,
        'numeric_noise': 0.5,
        'effect_clarity': 0.6,
    },
}


def _get_difficulty_settings(level: str) -> Dict[str, Any]:
    """Get simulation settings for a difficulty level."""
    return DIFFICULTY_LEVELS.get(level, DIFFICULTY_LEVELS['medium'])


# =============================================================================
# v1.0.0: LIVE DATA PREVIEW GENERATOR
# =============================================================================

def _generate_preview_data(
    conditions: List[str],
    scales: List[Dict[str, Any]],
    open_ended: List[Dict[str, Any]],
    n_rows: int = 5,
    difficulty: str = 'medium',
    study_title: str = '',
    study_description: str = '',
) -> pd.DataFrame:
    """Generate a preview of simulated data (5 rows by default).

    v1.0.3.8: Now accepts study_title and study_description for context-aware
    open-ended response generation. Responses are grounded in the actual
    question text and context instead of generic meta-commentary.
    """
    preview_data = {}
    difficulty_settings = _get_difficulty_settings(difficulty)

    # Add participant ID (matches actual engine output column name)
    preview_data['PARTICIPANT_ID'] = [f"P{i+1:03d}" for i in range(n_rows)]

    # Add condition assignment (matches actual engine output column name)
    if conditions:
        preview_data['CONDITION'] = [conditions[i % len(conditions)] for i in range(n_rows)]

    # Add scale responses - use same defaults as actual generation
    # v1.4.2.1: Safe handling for None values in scale properties (builder/QSF compat)
    for scale in scales[:5]:  # Limit to 5 scales for preview
        if not isinstance(scale, dict):
            continue
        scale_name = scale.get('name', 'Scale') or 'Scale'

        # Safely parse scale_points (may be None for numeric/slider scales from builder)
        raw_pts = scale.get('scale_points')
        try:
            scale_points = int(raw_pts) if raw_pts is not None else 7
        except (ValueError, TypeError):
            scale_points = 7
        scale_points = max(2, scale_points)

        # Check both "num_items" and "items" for compatibility with QSF and builder
        items = scale.get('num_items')
        if items is None:
            _raw_items = scale.get('items', 5)
            # 'items' can be a list of item names (builder) or an int (QSF)
            if isinstance(_raw_items, (list, tuple)):
                items = len(_raw_items) if _raw_items else 5
            else:
                items = _raw_items
        try:
            items = int(items) if items is not None else 5
        except (ValueError, TypeError):
            items = 5
        items = max(1, items)

        # Use scale_min/scale_max if available (builder always provides these)
        # Safely handle None values to prevent int(None) TypeError
        raw_s_min = scale.get('scale_min')
        raw_s_max = scale.get('scale_max')
        try:
            _s_min = int(raw_s_min) if raw_s_min is not None else 1
        except (ValueError, TypeError):
            _s_min = 1
        try:
            _s_max = int(raw_s_max) if raw_s_max is not None else scale_points
        except (ValueError, TypeError):
            _s_max = scale_points
        if _s_max <= _s_min:
            _s_max = _s_min + scale_points - 1

        if items == 1:
            # Single item
            var_name = scale_name.replace(' ', '_')
            values = []
            for row_idx in range(n_rows):
                val = np.random.randint(_s_min, _s_max + 1)
                # v1.0.1.3: Apply condition-aware shifts to preview data
                # so researchers see realistic between-condition differences
                if conditions:
                    condition = conditions[row_idx % len(conditions)]
                    cond_lower = condition.lower()
                    # Determine directional shift based on condition semantics
                    cond_shift = 0
                    if any(w in cond_lower.split() for w in ['treatment', 'high', 'positive', 'gain', 'hedonic']):
                        cond_shift = 1  # Shift up
                    elif any(w in cond_lower.split() for w in ['negative', 'loss', 'low', 'bad']):
                        cond_shift = -1  # Shift down
                    # Apply shift (clamped to scale bounds)
                    val = max(_s_min, min(_s_max, val + cond_shift))
                values.append(val)
            preview_data[var_name] = values
        else:
            # Multi-item scale - show first item and composite
            var_name = scale_name.replace(' ', '_')
            item1_values = []
            mean_values = []
            for row_idx in range(n_rows):
                val1 = np.random.randint(_s_min, _s_max + 1)
                val_mean = np.random.uniform(_s_min, _s_max)
                # v1.0.1.3: Apply condition-aware shifts to preview data
                # so researchers see realistic between-condition differences
                if conditions:
                    condition = conditions[row_idx % len(conditions)]
                    cond_lower = condition.lower()
                    # Determine directional shift based on condition semantics
                    cond_shift = 0
                    if any(w in cond_lower.split() for w in ['treatment', 'high', 'positive', 'gain', 'hedonic']):
                        cond_shift = 1  # Shift up
                    elif any(w in cond_lower.split() for w in ['negative', 'loss', 'low', 'bad']):
                        cond_shift = -1  # Shift down
                    # Apply shift (clamped to scale bounds)
                    val1 = max(_s_min, min(_s_max, val1 + cond_shift))
                    val_mean = max(_s_min, min(_s_max, val_mean + cond_shift))
                item1_values.append(val1)
                mean_values.append(round(val_mean, 2))
            preview_data[f"{var_name}_1"] = item1_values
            preview_data[f"{var_name}_mean"] = mean_values

    # v1.0.6.7: Open-ended columns EXCLUDED from preview.
    # OE responses are always generated via LLM API calls during full simulation.
    # Template-based preview responses were misleading, so we only show numeric
    # columns + demographics in the preview to set correct expectations.
    if open_ended:
        # Show a note in the preview that OE columns will appear in the full dataset
        _oe_names = []
        for oe in open_ended[:3]:
            if isinstance(oe, str):
                _oe_names.append(oe)
            elif isinstance(oe, dict):
                _oe_names.append(oe.get('variable_name', oe.get('name', 'OE')))
        if _oe_names:
            _oe_note = ", ".join(_oe_names)
            if len(open_ended) > 3:
                _oe_note += f", ... (+{len(open_ended) - 3} more)"
            # Store note for display after the preview table
            preview_data['_oe_note'] = _oe_note

    # Add demographics
    preview_data['age'] = [np.random.randint(18, 65) for _ in range(n_rows)]
    preview_data['gender'] = [np.random.choice(['Male', 'Female', 'Other']) for _ in range(n_rows)]

    # Add attention check
    preview_data['attention_check_pass'] = [
        1 if np.random.random() < difficulty_settings['attention_rate'] else 0
        for _ in range(n_rows)
    ]

    return pd.DataFrame(preview_data)


def _get_sample_text_response(
    quality: str,
    participant_idx: int,
    question_name: str = "",
    question_text: str = "",
    question_context: str = "",
    condition: str = "",
    study_title: str = "",
    study_description: str = "",
) -> str:
    """Generate context-aware sample text responses for preview.

    v1.0.3.8: COMPLETE REWRITE — responses are now grounded in the actual
    question content. Uses question_text, question_context, condition, and
    study_title to generate topical responses that actually answer the question,
    instead of generic meta-commentary like 'it was an interesting task'.

    The function extracts key themes/subjects from the question context and
    builds response templates that reference those specific themes. For
    questions without context, it still tries to extract meaning from the
    question text / variable name.

    Args:
        quality: Response quality level (high/medium/low/very_low)
        participant_idx: Index of the participant (0-based)
        question_name: Unique identifier for the question
        question_text: The actual question text
        question_context: Researcher-provided context explaining the question
        condition: Experimental condition for this participant
        study_title: Title of the study
        study_description: Description of the study

    Returns:
        Context-aware response for this participant-question combination
    """
    import random as random_module
    import re as _re

    # CRITICAL: Create a unique seed that combines participant index AND question identity
    if question_name:
        name_hash = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(question_name[:100]))
    else:
        name_hash = 0
    unique_seed = (participant_idx * 100003) + name_hash
    local_rng = random_module.Random(unique_seed)

    # ------------------------------------------------------------------
    # Extract the SUBJECT of the question from context/text/name
    # ------------------------------------------------------------------
    subject = ""
    topic_words: List[str] = []

    # v1.0.4.7: Unified stop word list — includes researcher-instruction vocabulary
    # to prevent "primed thinking Trump telling" instead of just "Trump".
    _stop = {
        # Articles & determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'it',
        # Pronouns
        'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
        # Prepositions
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'into', 'about',
        # Conjunctions
        'and', 'or', 'but', 'not', 'no', 'so', 'nor',
        # Auxiliary/modal verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        # Question words
        'how', 'what', 'who', 'why', 'when', 'where', 'which',
        # Common non-topical verbs
        'want', 'wants', 'understand', 'think', 'feel', 'tell', 'share', 'describe',
        'explain', 'ask', 'asked', 'give', 'gave', 'get', 'gets', 'make', 'makes',
        'say', 'said', 'know', 'knew', 'see', 'saw', 'come', 'came', 'take', 'took',
        # ── Researcher instruction vocabulary (v1.0.4.7) ──
        'participants', 'respondents', 'subjects', 'people', 'person', 'individuals',
        'primed', 'priming', 'prime', 'exposed', 'exposure', 'exposing',
        'presented', 'presenting', 'presentation', 'shown', 'showing', 'show',
        'told', 'telling', 'instructed', 'instructions', 'instruction',
        'assigned', 'randomly', 'random', 'randomized', 'allocation',
        'thinking', 'reading', 'viewing', 'watching', 'completing', 'answering',
        'reporting', 'sharing', 'responding',
        'before', 'after', 'during', 'following', 'prior',
        'then', 'next', 'first', 'second', 'third',
        'stories', 'story', 'experience', 'experiences', 'experienced',
        'whether', 'toward', 'towards', 'regarding', 'concerning',
        # Survey/study metadata
        'question', 'questions', 'context', 'study', 'survey', 'experiment',
        'condition', 'conditions', 'topic', 'measure', 'measured', 'measuring',
        'response', 'responses', 'answer', 'answers', 'item', 'items',
        'scale', 'rating', 'rate', 'rated', 'open', 'ended', 'text', 'variable',
        # Common adjectives/adverbs (non-topical)
        'much', 'more', 'most', 'very', 'really', 'just', 'also', 'please',
        'better', 'deeply', 'held', 'quite', 'thoughts', 'feelings',
        # v1.0.6.3: Additional stop words
        'here', 'there', 'well', 'like', 'even', 'still', 'let',
        'only', 'some', 'such', 'each', 'every', 'any', 'all', 'both',
        'many', 'few', 'own', 'other', 'another', 'same', 'different',
        'something', 'anything', 'everything', 'nothing',
        'however', 'therefore', 'moreover', 'furthermore', 'indeed',
        'certain', 'particular', 'specific', 'general', 'overall',
    }

    # Priority 1: Use question_context (researcher-provided, most specific)
    if question_context and question_context.strip():
        subject = question_context.strip()
        # Strip researcher framing before extracting topic words
        _ctx_clean = _re.sub(
            r'^(?:participants?\s+(?:are|were|will\s+be)\s+)',
            '', subject, flags=_re.IGNORECASE).strip()
        _words = _re.findall(r'\b[a-zA-Z]{3,}\b', _ctx_clean.lower())
        topic_words = [w for w in _words if w not in _stop][:8]

    # Priority 2: Use question_text if context is missing
    elif question_text and question_text.strip():
        # If question_text looks like a variable name, humanize it
        _qt = question_text.strip()
        if ' ' not in _qt:
            subject = _re.sub(r'[_\-]+', ' ', _qt).strip()
        else:
            subject = _qt
        _words = _re.findall(r'\b[a-zA-Z]{3,}\b', subject.lower())
        topic_words = [w for w in _words if w not in _stop][:6]

    # v1.0.4.8: Strategy 2 — Phrase-level extraction for richer topic understanding
    # Look for "feelings about X", "thoughts on X", "how you feel about X" patterns
    _phrase_topic = ""
    _source_text = question_context or question_text or ""
    _phrase_patterns = [
        r'(?:feelings?|thoughts?|opinions?|views?|attitudes?)\s+(?:about|toward|towards|on|regarding)\s+(.+?)(?:\.|$|\?)',
        r'(?:describe|explain|tell\s+us)\s+(?:about|how\s+you\s+feel\s+about)\s+(.+?)(?:\.|$|\?)',
        r'(?:how\s+do\s+you\s+feel\s+about)\s+(.+?)(?:\.|$|\?)',
        r'(?:what\s+do\s+you\s+think\s+(?:about|of))\s+(.+?)(?:\.|$|\?)',
    ]
    for _pp in _phrase_patterns:
        _pm = _re.search(_pp, _source_text, flags=_re.IGNORECASE)
        if _pm:
            _phrase_topic = _pm.group(1).strip()[:60]
            break

    # Add study title words as fallback context
    if study_title and study_title.strip():
        _title_words = _re.findall(r'\b[a-zA-Z]{4,}\b', study_title.lower())
        _stop_title = {'study', 'experiment', 'survey', 'research', 'behavioral', 'simulation'}
        _title_topics = [w for w in _title_words if w not in _stop_title]
        if not topic_words:
            topic_words = _title_topics[:4]

    # v1.0.4.8: Strategy 3 — Condition-aware topic enrichment
    _cond_topic_words: List[str] = []
    if condition and condition.strip():
        _cond_clean = _re.sub(r'[_\-,]+', ' ', condition).strip()
        _cw = _re.findall(r'\b[a-zA-Z]{3,}\b', _cond_clean.lower())
        _cond_stop_extra = {'control', 'baseline', 'treatment', 'group', 'condition',
                            'level', 'high', 'low', 'cell'}
        _cond_topic_words = [w for w in _cw if w not in (_stop | _cond_stop_extra)][:3]

    # v1.0.3.10: Build a compact subject phrase for template insertion.
    # v1.0.5.5: Entity-first topic construction — use named entities as primary
    # topic to avoid word-salad ("love hate Trump" → "Trump").
    if not topic_words and question_name:
        _name_clean = _re.sub(r'[_\-]+', ' ', question_name).strip()
        _name_words = _re.findall(r'\b[a-zA-Z]{3,}\b', _name_clean.lower())
        _name_stop = {'open', 'ended', 'text', 'question', 'response', 'answer'}
        topic_words = [w for w in _name_words if w not in _name_stop][:4]

    # v1.0.6.4: GENERAL-PURPOSE entity detection via heuristics. Detects
    # capitalized words and acronyms from the ORIGINAL (non-lowered) source text.
    # Works for ANY topic — not just pre-listed names.
    _orig_source = f"{question_context or ''} {question_text or ''} {condition or ''}"
    _detected_entities: set = set()
    # Heuristic 1: Capitalized words mid-sentence
    for _cw in _re.findall(r'(?<=[a-z]\s)([A-Z][a-zA-Z]{2,})', _orig_source):
        if _cw.lower() not in _stop:
            _detected_entities.add(_cw.lower())
    # Heuristic 2: ALL-CAPS acronyms (2-6 chars)
    for _acr in _re.findall(r'\b([A-Z]{2,6})\b', _orig_source):
        if _acr.lower() not in _stop:
            _detected_entities.add(_acr.lower())
    # Heuristic 3: First capitalized word in source (often the topic)
    _first_cap = _re.findall(r'^([A-Z][a-zA-Z]{3,})', _orig_source.strip())
    for _fc in _first_cap:
        if _fc.lower() not in _stop:
            _detected_entities.add(_fc.lower())

    # Prefer named entities as topic, then max 2 content words
    _entities_in_words = [w for w in topic_words if w.lower() in _detected_entities]
    _content_in_words = [w for w in topic_words if w.lower() not in _detected_entities]

    if _phrase_topic:
        subject_phrase = _phrase_topic
    elif _entities_in_words:
        # Named entity is the cleanest topic: "Trump", "Biden", etc.
        subject_phrase = _entities_in_words[0]
    elif _content_in_words:
        # Max 2 content words to avoid word-salad
        subject_phrase = ' '.join(_content_in_words[:2])
    elif _cond_topic_words:
        subject_phrase = ' '.join(_cond_topic_words[:2])
    else:
        subject_phrase = 'the questions asked'

    # v1.0.6.5: Derive proper nouns from detected entities (fixes undefined _proper_nouns bug)
    _proper_nouns = _detected_entities  # already lowercased set from heuristic detection
    subject_words_list = subject_phrase.split()
    subject_words_list = [w.capitalize() if w.lower() in _proper_nouns else w for w in subject_words_list]
    subject_phrase = ' '.join(subject_words_list)

    # Also prepare condition-aware modifiers
    cond_phrase = ""
    if condition and condition.strip():
        _cond_humanized = _re.sub(r'[_\-]+', ' ', condition).strip()
        cond_phrase = _cond_humanized

    # ------------------------------------------------------------------
    # Generate context-aware responses based on quality level
    # ------------------------------------------------------------------
    if quality in ('low', 'very_low'):
        # Low quality: short but STILL about the topic (never generic)
        if quality == 'very_low':
            _short_subj = subject_phrase[:15].strip()
            low_templates = [
                f"{_short_subj}",
                f"{_short_subj} idk",
                f"{_short_subj} ok",
                f"idk {_short_subj}",
                f"{_short_subj} i guess",
                f"{_short_subj} whatever",
                f"meh {_short_subj}",
            ]
            return local_rng.choice(low_templates)
        else:
            low_templates = [
                f"{subject_phrase} is ok i guess",
                f"idk {subject_phrase} seems fine",
                f"whatever about {subject_phrase}",
                f"yeah {subject_phrase}",
                f"its fine",
                f"didnt think much about {subject_phrase}",
                f"{subject_phrase} is alright",
                f"meh {subject_phrase}",
                f"not sure about {subject_phrase}",
                f"don't really care about {subject_phrase}",
            ]
            return local_rng.choice(low_templates)

    # v1.0.5.5: Compositional preview templates — compose from opener + core +
    # elaboration for high diversity instead of fixed templates.

    # Sentiment variation based on participant index
    sentiment_idx = participant_idx % 5
    sentiments = ['positive', 'neutral', 'negative', 'positive', 'mixed']
    sentiment = sentiments[sentiment_idx]

    _openers = [
        "Honestly", "I gotta say", "For me personally", "I mean",
        "To be real", "Look", "I'll be honest", "The way I see it",
        "I have to say", "From my perspective", "Thinking about it",
        "Being honest here", "In my view", "So basically",
    ]

    if quality == 'high':
        if sentiment == 'positive':
            _cores = [
                f"I feel good about {subject_phrase}",
                f"{subject_phrase} is something I care about",
                f"I'm supportive of {subject_phrase}",
                f"my views on {subject_phrase} are positive",
                f"I think {subject_phrase} is headed in the right direction",
                f"I see {subject_phrase} favorably",
                f"{subject_phrase} matters to me and I'm generally optimistic",
            ]
            _elaborations = [
                "It really made me think about what matters to me personally.",
                "My experiences have shaped a pretty strong view on this.",
                "I tried to express that honestly in my answers.",
                "I feel like this is heading the right way and I wanted to say so.",
                "I've thought about it a lot and I stand by my feelings.",
                "I spent time thinking about each question because it resonates with me.",
            ]
        elif sentiment == 'negative':
            _cores = [
                f"I have real concerns about {subject_phrase}",
                f"{subject_phrase} frustrates me",
                f"I'm disappointed with {subject_phrase}",
                f"my views on {subject_phrase} are pretty critical",
                f"I don't think {subject_phrase} is working well",
                f"there are serious problems with {subject_phrase}",
                f"{subject_phrase} needs to change",
            ]
            _elaborations = [
                "There are problems that aren't being addressed and that bothers me.",
                "I tried to express my concerns honestly in my responses.",
                "Things could be so much better and I feel that needs to be said.",
                "My personal experience with this has been negative.",
                "I've tried to stay open-minded but I keep coming back to the same issues.",
                "I feel strongly that people need to pay more attention to this.",
            ]
        elif sentiment == 'mixed':
            _cores = [
                f"I have mixed feelings about {subject_phrase}",
                f"{subject_phrase} is complicated for me",
                f"I'm torn about {subject_phrase}",
                f"I see both sides when it comes to {subject_phrase}",
                f"my thoughts on {subject_phrase} are nuanced",
            ]
            _elaborations = [
                "There are things I appreciate but also things that worry me.",
                "I can see both the positives and the problems.",
                "It's not black-and-white for me and I tried to capture that.",
                "I've gone back and forth on this honestly.",
                "My answers might seem contradictory but that reflects how I actually feel.",
            ]
        else:  # neutral
            _cores = [
                f"I don't have super strong opinions about {subject_phrase}",
                f"I'm fairly neutral on {subject_phrase}",
                f"my views on {subject_phrase} are moderate",
                f"I don't feel strongly about {subject_phrase} either way",
                f"{subject_phrase} is something I thought about but don't feel extreme about",
            ]
            _elaborations = [
                "I just tried to give honest answers based on what I actually think.",
                "I can see different perspectives and tried to represent my actual views.",
                "Nothing too extreme either way, just my honest take.",
                "I tried to think carefully and respond authentically.",
                "I didn't force myself to have a stronger opinion than I actually do.",
            ]

        core = local_rng.choice(_cores)
        opener = local_rng.choice(_openers)
        elab = local_rng.choice(_elaborations)
        if local_rng.random() < 0.6:
            return f"{opener}, {core}. {elab}"
        else:
            return f"{core[0].upper()}{core[1:]}. {elab}"

    else:  # medium quality
        if sentiment == 'positive':
            _cores = [
                f"I feel good about {subject_phrase}",
                f"{subject_phrase} is fine with me",
                f"I'm positive about {subject_phrase}",
                f"{subject_phrase} seems good to me",
                f"no issues with {subject_phrase}",
                f"I think {subject_phrase} is alright",
                f"{subject_phrase} works for me",
            ]
        elif sentiment == 'negative':
            _cores = [
                f"not a fan of {subject_phrase}",
                f"I have problems with {subject_phrase}",
                f"{subject_phrase} concerns me",
                f"I'm not happy about {subject_phrase}",
                f"there are issues with {subject_phrase}",
                f"{subject_phrase} could be a lot better",
            ]
        elif sentiment == 'mixed':
            _cores = [
                f"mixed feelings about {subject_phrase}",
                f"{subject_phrase} has pros and cons",
                f"I see both sides of {subject_phrase}",
                f"not sure how I feel about {subject_phrase}",
                f"{subject_phrase} is complicated",
            ]
        else:  # neutral
            _cores = [
                f"I don't feel strongly about {subject_phrase}",
                f"{subject_phrase} is whatever honestly",
                f"no strong opinion on {subject_phrase}",
                f"just answered what I think about {subject_phrase}",
                f"I'm somewhere in the middle on {subject_phrase}",
                f"{subject_phrase} - gave my honest answer",
            ]

        core = local_rng.choice(_cores)
        # Medium quality: shorter, sometimes with opener, sometimes not
        if local_rng.random() < 0.3:
            opener = local_rng.choice(["Honestly", "I mean", "Yeah", "Look", "Idk"])
            return f"{opener}, {core}."
        else:
            return f"{core[0].upper()}{core[1:]}."


def _merge_condition_sources(qsf_conditions: List[str], prereg_conditions: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Merge condition sources. Now only uses QSF conditions (prereg_conditions is always empty).
    Returns (merged_conditions, source_info_list)
    """
    # Only use QSF conditions - preregistration parsing produced garbage
    conditions = []
    sources = []
    seen = set()

    for cond in qsf_conditions:
        key = cond.strip().lower()
        if key and key not in seen:
            seen.add(key)
            conditions.append(cond.strip())
            sources.append({"Condition": cond.strip(), "Source": "QSF"})

    return conditions, sources


def _extract_qsf_payload(uploaded_bytes: bytes) -> Tuple[bytes, str]:
    """
    Return JSON bytes from a QSF upload (supports raw JSON or ZIP wrappers).
    """
    if zipfile.is_zipfile(io.BytesIO(uploaded_bytes)):
        with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
            candidates = [n for n in zf.namelist() if n.lower().endswith((".qsf", ".json"))]
            if not candidates:
                raise ValueError("ZIP did not contain a .qsf or .json file.")
            selected = candidates[0]
            return zf.read(selected), selected
    return uploaded_bytes, "uploaded.qsf"


def _extract_pdf_text(uploaded_bytes: bytes) -> str:
    """
    Extract text from a PDF using pypdf if available.
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(uploaded_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    except Exception:
        return ""


def _extract_scale_info_from_pdf(pdf_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract scale information from survey PDF text.

    Looks for patterns like:
    - "1 = Strongly Disagree ... 7 = Strongly Agree" (indicates 7-point scale)
    - "Scale: 1-5" or "(1-7)" (explicit range)
    - Numbered response options
    """
    scale_info: Dict[str, Dict[str, Any]] = {}

    if not pdf_text:
        return scale_info

    # Pattern 1: Explicit scale ranges like "1-7", "(1-5)", "scale: 1 to 7"
    range_pattern = r'(?:scale[:\s]*)?[(\[]?\s*(\d)\s*[-–to]+\s*(\d)\s*[)\]]?'
    for match in re.finditer(range_pattern, pdf_text, re.IGNORECASE):
        try:
            low = int(match.group(1))
            high = int(match.group(2))
            if 1 <= low < high <= 11:
                # This is likely a scale definition
                context = pdf_text[max(0, match.start() - 100):match.end() + 50]
                scale_info[f"scale_range_{match.start()}"] = {
                    "points": high - low + 1,
                    "range": f"{low}-{high}",
                    "context": context.strip()
                }
        except (ValueError, IndexError):
            pass

    # Pattern 2: Likert anchor patterns
    likert_patterns = [
        (r'strongly\s+disagree.*?strongly\s+agree', 7),
        (r'not\s+at\s+all.*?extremely', 7),
        (r'very\s+unlikely.*?very\s+likely', 7),
        (r'never.*?always', 5),
        (r'poor.*?excellent', 5),
    ]

    for pattern, default_points in likert_patterns:
        if re.search(pattern, pdf_text, re.IGNORECASE | re.DOTALL):
            scale_info[f"likert_{pattern[:20]}"] = {
                "points": default_points,
                "type": "likert",
                "pattern": pattern[:30]
            }

    # Pattern 3: Numbered response sequences like "1 2 3 4 5 6 7" or "1. Strongly..."
    numbered_pattern = r'(?:^|\s)(\d)\s+(\d)\s+(\d)\s+(\d)(?:\s+(\d))?(?:\s+(\d))?(?:\s+(\d))?'
    for match in re.finditer(numbered_pattern, pdf_text):
        nums = [int(g) for g in match.groups() if g]
        if len(nums) >= 4 and nums == list(range(nums[0], nums[-1] + 1)):
            scale_info[f"numbered_{match.start()}"] = {
                "points": len(nums),
                "type": "numbered_sequence"
            }

    return scale_info


def _infer_default_scale_points(survey_pdf_text: str, prereg_text: str) -> int:
    """
    Infer the most likely default scale points from available text.

    Returns the most common scale point value found, or 7 as default.
    """
    scale_info = _extract_scale_info_from_pdf(survey_pdf_text)

    if scale_info:
        # Count occurrences of each scale point value
        point_counts: Dict[int, int] = {}
        for info in scale_info.values():
            points = info.get("points", 7)
            point_counts[points] = point_counts.get(points, 0) + 1

        # Return most common, defaulting to 7
        if point_counts:
            return max(point_counts, key=point_counts.get)

    # Check preregistration text for scale hints
    prereg_combined = f"{prereg_text}"
    if "5-point" in prereg_combined.lower() or "5 point" in prereg_combined.lower():
        return 5
    if "7-point" in prereg_combined.lower() or "7 point" in prereg_combined.lower():
        return 7

    return 7  # Default


def _parse_warnings_for_display(warnings: List[str]) -> List[Dict[str, Any]]:
    """
    Parse warning messages to extract actionable information.

    Returns structured data with fix suggestions.
    """
    parsed = []

    for warning in warnings:
        parsed_warning = {
            "message": warning,
            "type": "general",
            "fix_suggestion": None,
            "fix_action": None,
        }

        # Scale point warnings
        if "unclear scale points" in warning.lower():
            # Extract question ID
            q_id_match = re.search(r'Question\s+(\S+)', warning)
            q_id = q_id_match.group(1) if q_id_match else None

            parsed_warning["type"] = "scale_points"
            parsed_warning["question_id"] = q_id
            parsed_warning["fix_suggestion"] = "Set scale points to common values (5 or 7)"
            parsed_warning["fix_options"] = [5, 7, 9, 11]

        # Condition warnings
        elif "no experimental conditions" in warning.lower():
            parsed_warning["type"] = "no_conditions"
            parsed_warning["fix_suggestion"] = "Add conditions manually in the Design Setup step"

        # Attention check warnings
        elif "no attention check" in warning.lower():
            parsed_warning["type"] = "no_attention_check"
            parsed_warning["fix_suggestion"] = "Consider adding attention checks to your survey"

        parsed.append(parsed_warning)

    return parsed


def _create_scale_point_fixes(warnings: List[str], default_points: int = 7) -> Dict[str, int]:
    """
    Create a dictionary of suggested scale point fixes.
    """
    fixes = {}
    for warning in warnings:
        if "unclear scale points" in warning.lower():
            q_id_match = re.search(r'Question\s+(\S+)', warning)
            if q_id_match:
                q_id = q_id_match.group(1)
                fixes[q_id] = default_points
    return fixes


def _categorize_variable_by_name(var_name: str) -> str:
    """
    Categorize a variable based on its name patterns.
    Returns: Timing, Demographics, Scale, Attention, or Other
    """
    var_lower = var_name.lower()

    # Timing variables
    timing_patterns = [
        'duration', 'startdate', 'enddate', 'recordeddate', 'timing',
        'time', 'q_totaltime', 'q_recaptchascore', 'progress', 'finished',
        'responseid', 'ipaddress', 'locationlatitude', 'locationlongitude',
        'userlanguage', 'distributionchannel', 'externalreference',
        'status', 'recipientlastname', 'recipientfirstname', 'recipientemail',
        'gc', '_first_click', '_last_click', '_page_submit', '_click_count',
    ]
    for pattern in timing_patterns:
        if pattern in var_lower or var_lower.startswith('q_'):
            return "Timing/Meta"

    # Demographics
    demo_patterns = ['age', 'gender', 'sex', 'income', 'education', 'race', 'ethnicity', 'occupation', 'demo']
    for pattern in demo_patterns:
        if pattern in var_lower:
            return "Demographics"

    # Attention checks
    attention_patterns = ['attention', 'check', 'attn', 'ac_', 'attention_']
    for pattern in attention_patterns:
        if pattern in var_lower:
            return "Attention check"

    return "Survey Question"


def _build_variable_review_rows(
    inferred: Dict[str, Any],
    prereg_outcomes: str,
    prereg_iv: str,
    design_analysis: Optional[DesignAnalysisResult] = None,
) -> List[Dict[str, Any]]:
    """
    Build variable review rows from inferred design and preregistration info.

    This creates a comprehensive list of variables for user review and correction.
    Variables are categorized by type for easier filtering and organization.
    """
    rows: List[Dict[str, Any]] = []
    seen_vars: set = set()

    # If we have enhanced design analysis, use it
    if design_analysis and design_analysis.variables:
        for var in design_analysis.variables:
            if var.variable_id in seen_vars:
                continue
            seen_vars.add(var.variable_id)

            # Map VariableRole enum to display string with better categorization
            role_map = {
                VariableRole.CONDITION: "Condition",
                VariableRole.INDEPENDENT_VARIABLE: "Independent variable",
                VariableRole.PRIMARY_OUTCOME: "Primary outcome",
                VariableRole.SECONDARY_OUTCOME: "Secondary outcome",
                VariableRole.MEDIATOR: "Mediator",
                VariableRole.MODERATOR: "Moderator",
                VariableRole.COVARIATE: "Covariate",
                VariableRole.DEMOGRAPHICS: "Demographics",
                VariableRole.ATTENTION_CHECK: "Attention check",
                VariableRole.MANIPULATION_CHECK: "Manipulation check",
                VariableRole.OPEN_ENDED: "Open-ended",
                VariableRole.FILLER: "Filler",
                VariableRole.OTHER: "Other",
            }

            # Determine category based on variable name patterns
            category = _categorize_variable_by_name(var.variable_id)
            role = role_map.get(var.role, "Other")

            # Override role for timing/meta variables
            if category == "Timing/Meta":
                role = "Timing/Meta"
            elif category == "Demographics" and role == "Other":
                role = "Demographics"

            rows.append({
                "Variable": var.variable_id,
                "Display Name": var.display_name,
                "Type": category,
                "Role": role,
                "Question Text": var.question_text[:60] + "..." if len(var.question_text) > 60 else var.question_text,
            })

        # Sort: Survey Questions first, then by role importance
        role_order = {
            "Primary outcome": 0, "Secondary outcome": 1, "Condition": 2,
            "Independent variable": 3, "Mediator": 4, "Moderator": 5,
            "Manipulation check": 6, "Attention check": 7, "Open-ended": 8,
            "Demographics": 9, "Covariate": 10, "Timing/Meta": 11,
            "Filler": 12, "Other": 13,
        }
        type_order = {"Survey Question": 0, "Demographics": 1, "Attention check": 2, "Timing/Meta": 3}
        rows.sort(key=lambda r: (type_order.get(r["Type"], 99), role_order.get(r["Role"], 99)))
        return rows

    # Fallback: Build from inferred design
    # Add conditions
    for cond in inferred.get("conditions", []):
        if cond and cond not in seen_vars:
            seen_vars.add(cond)
            rows.append({
                "Variable": cond,
                "Display Name": cond,
                "Type": "Survey Question",
                "Role": "Condition",
                "Question Text": "",
            })

    # Add scales
    for scale in inferred.get("scales", []):
        name = scale.get("name", "")
        if name and name not in seen_vars:
            seen_vars.add(name)
            # Check if this matches preregistration outcomes
            role = "Primary outcome"
            if prereg_outcomes:
                name_lower = name.lower()
                outcomes_lower = prereg_outcomes.lower()
                if name_lower not in outcomes_lower:
                    role = "Secondary outcome"

            rows.append({
                "Variable": name,
                "Display Name": name.replace("_", " ").title(),
                "Type": "Survey Question",
                "Role": role,
                "Question Text": f"{scale.get('num_items', 0)} items, {scale.get('scale_points', 7)}-pt",
            })

    # Add open-ended questions
    for q in inferred.get("open_ended_questions", []):
        if q and q not in seen_vars:
            seen_vars.add(q)
            rows.append({
                "Variable": q,
                "Display Name": q.replace("_", " ").title(),
                "Type": "Survey Question",
                "Role": "Open-ended",
                "Question Text": "",
            })

    # If no rows, add a placeholder
    if not rows:
        rows.append({
            "Variable": "Main_DV",
            "Display Name": "Main DV",
            "Type": "Survey Question",
            "Role": "Primary outcome",
            "Question Text": "",
        })

    return rows


def _perform_enhanced_analysis(
    qsf_content: bytes,
    prereg_outcomes: str = "",
    prereg_iv: str = "",
    prereg_text: str = "",
    prereg_pdf_text: str = "",
) -> Optional[DesignAnalysisResult]:
    """
    Perform enhanced design analysis using the condition identifier.

    This provides deep analysis of:
    - Randomization structure
    - Condition identification
    - Variable role classification
    - Scale detection
    """
    try:
        return analyze_qsf_design(
            qsf_content=qsf_content,
            prereg_outcomes=prereg_outcomes,
            prereg_iv=prereg_iv,
            prereg_text=prereg_text,
            prereg_pdf_text=prereg_pdf_text,
        )
    except Exception:
        # Log error but don't crash
        return None


def _design_analysis_to_inferred(
    analysis: DesignAnalysisResult,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert DesignAnalysisResult to the inferred design format.
    """
    if not analysis:
        return fallback

    # Extract conditions
    conditions = [c.name for c in analysis.conditions] if analysis.conditions else fallback.get("conditions", [])
    if not conditions:
        conditions = ["Condition A"]

    # Extract factors
    factors = [
        {"name": f.name, "levels": f.levels}
        for f in analysis.factors
    ] if analysis.factors else fallback.get("factors", [])

    # Extract scales
    scales = [
        {
            "name": s.name,
            "variable_name": s.variable_name,
            "num_items": s.num_items,
            "scale_points": s.scale_points,
            "reverse_items": s.reverse_items,
        }
        for s in analysis.scales
    ] if analysis.scales else fallback.get("scales", [])

    # Extract open-ended questions
    open_ended = analysis.open_ended_questions if analysis.open_ended_questions else fallback.get("open_ended_questions", [])

    return {
        "conditions": conditions,
        "factors": factors,
        "scales": scales,
        "open_ended_questions": open_ended,
        "attention_checks": analysis.attention_checks if analysis else [],
        "manipulation_checks": analysis.manipulation_checks if analysis else [],
        "randomization_level": analysis.randomization.level.value if analysis and analysis.randomization else "Participant-level",
    }


# ========================================
# FEEDBACK/BUG REPORT SYSTEM
# ========================================

FEEDBACK_EMAIL = "edimant@sas.upenn.edu"

def _render_analytics_dashboard(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    scales: List[Dict[str, Any]],
) -> None:
    """Render the professional analytics dashboard.

    v1.8.8.0: Password-protected, advanced-mode only.
    Uses Plotly for publication-quality interactive visualizations.
    Sections: Descriptive stats, condition comparisons, correlations,
    effect sizes, distributions, data quality.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        _has_plotly = True
    except ImportError:
        _has_plotly = False

    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);'
        'padding:20px 24px;border-radius:12px;margin:16px 0 20px 0;">'
        '<h3 style="color:#e8e8e8;margin:0 0 4px 0;font-weight:700;letter-spacing:0.02em;">'
        'Analytics Dashboard</h3>'
        '<p style="color:#8896ab;margin:0;font-size:0.82rem;">'
        'Professional statistical analysis of your simulated dataset</p></div>',
        unsafe_allow_html=True,
    )

    if not _has_plotly:
        st.warning("Plotly is required for the analytics dashboard. Install with: `pip install plotly`")
        return

    # ── Theme configuration ──
    _THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#374151"),
        margin=dict(l=50, r=30, t=45, b=40),
        colorway=["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
                   "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1"],
    )

    conditions = sorted(df["CONDITION"].unique()) if "CONDITION" in df.columns else []
    n_conditions = len(conditions)

    # ──────────────────────────────────────────────────────────────
    # v1.8.9: SAMPLE ADEQUACY SUMMARY (quick at-a-glance metrics)
    # ──────────────────────────────────────────────────────────────
    _adeq_cols = st.columns(4)
    _total_n = len(df)
    _per_cell_n = _total_n // max(n_conditions, 1) if n_conditions > 0 else _total_n
    _adequacy_label = "Excellent" if _per_cell_n >= 50 else "Good" if _per_cell_n >= 30 else "Adequate" if _per_cell_n >= 20 else "Low"
    _adequacy_color = "#22c55e" if _per_cell_n >= 30 else "#f59e0b" if _per_cell_n >= 20 else "#ef4444"
    _adeq_cols[0].metric("Total N", _total_n)
    _adeq_cols[1].metric("Conditions", n_conditions)
    _adeq_cols[2].metric("Per Cell", _per_cell_n)
    _adeq_cols[3].markdown(
        f'<div style="text-align:center;padding-top:8px;">'
        f'<span style="font-size:0.78rem;color:#6b7280;">Sample Adequacy</span><br>'
        f'<span style="font-size:1.3rem;font-weight:700;color:{_adequacy_color};">{_adequacy_label}</span></div>',
        unsafe_allow_html=True,
    )

    # Identify composite columns (scale means)
    _composite_cols = [c for c in df.columns if c.endswith("_mean") and df[c].dtype in ("float64", "float32", "int64")]
    if not _composite_cols:
        # Fallback: look for item columns and compute means
        for s in scales:
            sname = str(s.get("variable_name", s.get("name", ""))).strip().replace(" ", "_")
            n_items = int(s.get("num_items", 1))
            _item_cols = [f"{sname}_{i}" for i in range(1, n_items + 1) if f"{sname}_{i}" in df.columns]
            if _item_cols:
                _mean_col = f"{sname}_mean"
                if _mean_col not in df.columns:
                    df[_mean_col] = df[_item_cols].mean(axis=1)
                _composite_cols.append(_mean_col)

    # ──────────────────────────────────────────────────────────────
    # SECTION 1: Descriptive Statistics
    # ──────────────────────────────────────────────────────────────
    with st.expander("Descriptive Statistics", expanded=True):
        if _composite_cols and conditions:
            _desc_rows = []
            for col in _composite_cols:
                _clean = col.replace("_mean", "").replace("_", " ")
                for cond in conditions:
                    _subset = df.loc[df["CONDITION"] == cond, col].dropna()
                    if len(_subset) > 0:
                        _mean = float(_subset.mean())
                        _sd = float(_subset.std())
                        _n = len(_subset)
                        # v1.8.9: Add 95% CI for means
                        _se = _sd / np.sqrt(_n) if _n > 1 else 0
                        _ci_lo = round(_mean - 1.96 * _se, 3)
                        _ci_hi = round(_mean + 1.96 * _se, 3)
                        _desc_rows.append({
                            "Scale": _clean,
                            "Condition": cond,
                            "N": _n,
                            "Mean": round(_mean, 3),
                            "SD": round(_sd, 3),
                            "95% CI": f"[{_ci_lo}, {_ci_hi}]",
                            "Min": round(float(_subset.min()), 2),
                            "Max": round(float(_subset.max()), 2),
                            "Skewness": round(float(_subset.skew()), 3) if _n > 2 else 0,
                        })
            if _desc_rows:
                _desc_df = pd.DataFrame(_desc_rows)
                st.dataframe(
                    _desc_df.style.format({"Mean": "{:.3f}", "SD": "{:.3f}", "Skewness": "{:.3f}"}),
                    use_container_width=True,
                    height=min(400, 40 + len(_desc_rows) * 35),
                )
        else:
            st.info("No composite scale columns found for descriptive statistics.")

    # ──────────────────────────────────────────────────────────────
    # SECTION 2: Condition Comparisons (Violin + Box)
    # ──────────────────────────────────────────────────────────────
    if _composite_cols and n_conditions >= 2:
        with st.expander("Condition Comparisons", expanded=True):
            _sel_dv = st.selectbox(
                "Select DV", _composite_cols,
                format_func=lambda x: x.replace("_mean", "").replace("_", " "),
                key="dashboard_dv_select",
            )
            if _sel_dv:
                fig = go.Figure()
                for i, cond in enumerate(conditions):
                    _vals = df.loc[df["CONDITION"] == cond, _sel_dv].dropna()
                    fig.add_trace(go.Violin(
                        y=_vals,
                        name=cond,
                        box_visible=True,
                        meanline_visible=True,
                        points="outliers",
                        marker_color=_THEME["colorway"][i % len(_THEME["colorway"])],
                        opacity=0.85,
                    ))
                fig.update_layout(
                    **_THEME,
                    title=dict(text=f"Distribution of {_sel_dv.replace('_mean','').replace('_',' ')} by Condition",
                               font=dict(size=15, color="#1f2937")),
                    yaxis_title="Score",
                    xaxis_title="Condition",
                    showlegend=False,
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Quick statistical test
                if n_conditions == 2:
                    from scipy import stats as _sp_stats
                    _g1 = df.loc[df["CONDITION"] == conditions[0], _sel_dv].dropna()
                    _g2 = df.loc[df["CONDITION"] == conditions[1], _sel_dv].dropna()
                    if len(_g1) > 1 and len(_g2) > 1:
                        _t, _p = _sp_stats.ttest_ind(_g1, _g2, equal_var=False)
                        _pooled_var = (float(_g1.std())**2 + float(_g2.std())**2) / 2
                        _d = (float(_g1.mean()) - float(_g2.mean())) / np.sqrt(_pooled_var) if _pooled_var > 0 else 0.0
                        _sig = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else "ns"
                        st.markdown(
                            f"**Welch's t-test:** t({len(_g1)+len(_g2)-2}) = {_t:.3f}, "
                            f"p = {_p:.4f} {_sig} | **Cohen's d** = {_d:.3f}"
                        )
                elif n_conditions > 2:
                    from scipy import stats as _sp_stats
                    _groups = [df.loc[df["CONDITION"] == c, _sel_dv].dropna().values for c in conditions]
                    _groups = [g for g in _groups if len(g) > 1]
                    if len(_groups) >= 2:
                        _f, _p = _sp_stats.f_oneway(*_groups)
                        _grand_mean = df[_sel_dv].dropna().mean()
                        _ss_b = sum(len(g) * (g.mean() - _grand_mean)**2 for g in _groups)
                        _ss_t = sum((df[_sel_dv].dropna() - _grand_mean)**2)
                        _eta2 = float(_ss_b / _ss_t) if _ss_t > 0 else 0
                        _sig = "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else "ns"
                        st.markdown(
                            f"**One-way ANOVA:** F({len(_groups)-1}, {sum(len(g) for g in _groups)-len(_groups)}) "
                            f"= {_f:.3f}, p = {_p:.4f} {_sig} | **η²** = {_eta2:.3f}"
                        )

    # ──────────────────────────────────────────────────────────────
    # SECTION 3: Correlation Heatmap (with significance annotations)
    # ──────────────────────────────────────────────────────────────
    if len(_composite_cols) >= 2:
        with st.expander("Correlation Matrix", expanded=True):
            from scipy import stats as _sp_stats_corr
            _corr_data = df[_composite_cols].corr()
            _labels = [c.replace("_mean", "").replace("_", " ") for c in _composite_cols]
            _z = _corr_data.values
            _n_obs = len(df)

            # Compute p-values for each correlation
            _p_matrix = np.ones_like(_z)
            _text_annot = np.empty_like(_z, dtype=object)
            for _ri in range(len(_composite_cols)):
                for _ci in range(len(_composite_cols)):
                    if _ri == _ci:
                        _text_annot[_ri][_ci] = "1.00"
                        continue
                    _pair_data = df[[_composite_cols[_ri], _composite_cols[_ci]]].dropna()
                    if len(_pair_data) > 2:
                        _r_val, _p_val = _sp_stats_corr.pearsonr(
                            _pair_data.iloc[:, 0], _pair_data.iloc[:, 1]
                        )
                        _p_matrix[_ri, _ci] = _p_val
                        _sig_star = "***" if _p_val < 0.001 else "**" if _p_val < 0.01 else "*" if _p_val < 0.05 else ""
                        _text_annot[_ri][_ci] = f"{_z[_ri, _ci]:.2f}{_sig_star}"
                    else:
                        _text_annot[_ri][_ci] = f"{_z[_ri, _ci]:.2f}"

            # Mask upper triangle for cleaner display
            _mask = np.triu(np.ones_like(_z, dtype=bool), k=1)
            _z_masked = np.where(_mask, np.nan, _z)
            _text_masked = np.where(_mask, "", _text_annot)

            fig = go.Figure(data=go.Heatmap(
                z=_z_masked,
                x=_labels,
                y=_labels,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=_text_masked,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#1f2937"),
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
                colorbar=dict(title="r", thickness=15, len=0.8),
            ))
            fig.update_layout(
                **_THEME,
                title=dict(text="Inter-Scale Correlation Matrix (with significance)",
                           font=dict(size=15, color="#1f2937")),
                height=max(350, 100 + len(_composite_cols) * 40),
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("\\* p < .05, \\*\\* p < .01, \\*\\*\\* p < .001")

            # Target vs Realized correlation comparison
            _target_corr = metadata.get("cross_dv_correlation", {}).get("correlation_matrix")
            if _target_corr is not None:
                _target_names = metadata.get("cross_dv_correlation", {}).get("scale_names", [])
                if len(_target_names) == len(_composite_cols):
                    st.markdown("**Target vs. Realized Correlations**")
                    _target_arr = np.array(_target_corr)
                    _comp_rows = []
                    for _ri in range(len(_composite_cols)):
                        for _ci in range(_ri + 1, len(_composite_cols)):
                            _comp_rows.append({
                                "Scale Pair": f"{_labels[_ri]} ↔ {_labels[_ci]}",
                                "Target r": round(float(_target_arr[_ri, _ci]), 3),
                                "Realized r": round(float(_z[_ri, _ci]), 3),
                                "Δ": round(float(_z[_ri, _ci] - _target_arr[_ri, _ci]), 3),
                            })
                    if _comp_rows:
                        st.dataframe(
                            pd.DataFrame(_comp_rows).style.format(
                                {"Target r": "{:+.3f}", "Realized r": "{:+.3f}", "Δ": "{:+.3f}"}
                            ),
                            use_container_width=True,
                            height=min(300, 40 + len(_comp_rows) * 35),
                        )

    # ──────────────────────────────────────────────────────────────
    # SECTION 4: Effect Size Forest Plot (with specified vs observed)
    # ──────────────────────────────────────────────────────────────
    if _composite_cols and n_conditions >= 2:
        with st.expander("Effect Size Analysis", expanded=True):
            _es_rows = []
            _specified_effects = metadata.get("effect_sizes_configured", metadata.get("effect_sizes", {}))
            for col in _composite_cols:
                _clean = col.replace("_mean", "").replace("_", " ")
                if n_conditions == 2:
                    _g1 = df.loc[df["CONDITION"] == conditions[0], col].dropna()
                    _g2 = df.loc[df["CONDITION"] == conditions[1], col].dropna()
                    if len(_g1) > 1 and len(_g2) > 1:
                        _pooled_sd = np.sqrt((float(_g1.std())**2 + float(_g2.std())**2) / 2)
                        _d = (float(_g1.mean()) - float(_g2.mean())) / _pooled_sd if _pooled_sd > 0 else 0
                        _se = np.sqrt((len(_g1) + len(_g2)) / (len(_g1) * len(_g2)) + _d**2 / (2 * (len(_g1) + len(_g2))))
                        _es_rows.append({
                            "Scale": _clean,
                            "Cohen's d": round(_d, 3),
                            "SE": round(_se, 3),
                            "CI_lo": round(_d - 1.96 * _se, 3),
                            "CI_hi": round(_d + 1.96 * _se, 3),
                        })
                else:
                    # For multi-condition: compare each pair vs. first condition
                    _g_ref = df.loc[df["CONDITION"] == conditions[0], col].dropna()
                    for cond in conditions[1:]:
                        _g_cmp = df.loc[df["CONDITION"] == cond, col].dropna()
                        if len(_g_ref) > 1 and len(_g_cmp) > 1:
                            _pooled_sd = np.sqrt((float(_g_ref.std())**2 + float(_g_cmp.std())**2) / 2)
                            _d = (float(_g_cmp.mean()) - float(_g_ref.mean())) / _pooled_sd if _pooled_sd > 0 else 0
                            _se = np.sqrt((len(_g_ref) + len(_g_cmp)) / (len(_g_ref) * len(_g_cmp)) + _d**2 / (2 * (len(_g_ref) + len(_g_cmp))))
                            _es_rows.append({
                                "Scale": f"{_clean} ({cond} vs {conditions[0]})",
                                "Cohen's d": round(_d, 3),
                                "SE": round(_se, 3),
                                "CI_lo": round(_d - 1.96 * _se, 3),
                                "CI_hi": round(_d + 1.96 * _se, 3),
                            })

            if _es_rows:
                fig = go.Figure()
                _names = [r["Scale"] for r in _es_rows]
                _ds = [r["Cohen's d"] for r in _es_rows]
                _ci_lo = [r["CI_lo"] for r in _es_rows]
                _ci_hi = [r["CI_hi"] for r in _es_rows]

                fig.add_trace(go.Scatter(
                    x=_ds, y=_names,
                    mode="markers",
                    marker=dict(size=10, color="#3b82f6", line=dict(width=1.5, color="#1e40af")),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[h - d for d, h in zip(_ds, _ci_hi)],
                        arrayminus=[d - lo for d, lo in zip(_ds, _ci_lo)],
                        color="#93c5fd",
                        thickness=2,
                        width=6,
                    ),
                    hovertemplate="<b>%{y}</b><br>d = %{x:.3f}<br>95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
                    customdata=list(zip(_ci_lo, _ci_hi)),
                ))
                # Zero reference line
                fig.add_vline(x=0, line_dash="dash", line_color="#9ca3af", line_width=1)
                # Cohen's d benchmarks
                for _bench, _label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                    fig.add_vline(x=_bench, line_dash="dot", line_color="#d1d5db", line_width=0.8,
                                  annotation_text=_label, annotation_position="top",
                                  annotation_font_size=9, annotation_font_color="#9ca3af")

                fig.update_layout(
                    **_THEME,
                    title=dict(text="Effect Size Forest Plot (Cohen's d with 95% CI)",
                               font=dict(size=15, color="#1f2937")),
                    xaxis_title="Cohen's d",
                    height=max(300, 80 + len(_es_rows) * 40),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.dataframe(
                    pd.DataFrame(_es_rows).style.format({"Cohen's d": "{:.3f}", "SE": "{:.3f}", "CI_lo": "{:.3f}", "CI_hi": "{:.3f}"}),
                    use_container_width=True,
                )

    # ──────────────────────────────────────────────────────────────
    # v1.8.9: SECTION 4b: Post-Hoc Power Analysis Summary
    # ──────────────────────────────────────────────────────────────
    if _composite_cols and n_conditions >= 2 and _es_rows:
        with st.expander("Post-Hoc Power Estimates", expanded=False):
            st.caption(
                "Approximate power for detecting the observed effect sizes at α = .05 (two-tailed). "
                "These are post-hoc estimates — interpret with caution."
            )
            _power_rows = []
            for row in _es_rows:
                _d_obs = abs(row["Cohen's d"])
                _n_per_group = _per_cell_n
                # Approximate power using normal approximation for two-sample t-test
                # noncentrality parameter: δ = d * sqrt(n/2)
                _ncp = _d_obs * np.sqrt(_n_per_group / 2) if _n_per_group > 0 else 0
                # Power ≈ Φ(δ - z_α/2) for large samples
                from scipy import stats as _sp_pwr
                _z_crit = 1.96
                _power = float(_sp_pwr.norm.cdf(_ncp - _z_crit))
                _power_label = "High" if _power >= 0.80 else "Moderate" if _power >= 0.50 else "Low"
                _power_rows.append({
                    "Comparison": row["Scale"],
                    "|d|": round(_d_obs, 3),
                    "n/group": _n_per_group,
                    "Est. Power": f"{_power:.0%}",
                    "Adequacy": _power_label,
                })
            if _power_rows:
                _pwr_df = pd.DataFrame(_power_rows)
                st.dataframe(
                    _pwr_df.style.map(
                        lambda v: "background-color: #dcfce7" if v == "High"
                        else "background-color: #fef9c3" if v == "Moderate"
                        else "background-color: #fee2e2" if v in ("Low",) else "",
                        subset=["Adequacy"],
                    ),
                    use_container_width=True,
                )
                _underpowered = sum(1 for r in _power_rows if r["Adequacy"] == "Low")
                if _underpowered > 0:
                    st.info(
                        f"{_underpowered} comparison(s) appear underpowered (< 50%). "
                        "Consider increasing sample size or targeting larger effects."
                    )

    # ──────────────────────────────────────────────────────────────
    # SECTION 5: Normality Assessment
    # ──────────────────────────────────────────────────────────────
    if _composite_cols and conditions:
        with st.expander("Normality Assessment", expanded=False):
            from scipy import stats as _sp_stats_norm
            _norm_rows = []
            for col in _composite_cols:
                _clean = col.replace("_mean", "").replace("_", " ")
                _vals = df[col].dropna()
                if len(_vals) >= 8:
                    _skew = float(_vals.skew())
                    _kurt = float(_vals.kurtosis())
                    # Shapiro-Wilk (up to 5000 samples)
                    _test_vals = _vals.values[:5000] if len(_vals) > 5000 else _vals.values
                    _w, _p_sw = _sp_stats_norm.shapiro(_test_vals)
                    _verdict = "Normal" if _p_sw >= 0.05 else "Non-normal"
                    _norm_rows.append({
                        "Scale": _clean,
                        "N": len(_vals),
                        "Skewness": round(_skew, 3),
                        "Kurtosis": round(_kurt, 3),
                        "Shapiro-Wilk W": round(float(_w), 4),
                        "p-value": round(float(_p_sw), 4),
                        "Verdict": _verdict,
                    })
            if _norm_rows:
                _norm_df = pd.DataFrame(_norm_rows)
                st.dataframe(
                    _norm_df.style.map(
                        lambda v: "background-color: #dcfce7" if v == "Normal"
                        else "background-color: #fee2e2" if v == "Non-normal" else "",
                        subset=["Verdict"],
                    ).format({"Skewness": "{:.3f}", "Kurtosis": "{:.3f}",
                              "Shapiro-Wilk W": "{:.4f}", "p-value": "{:.4f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "Shapiro-Wilk test: p ≥ .05 → Normal. "
                    "Skewness within ±1 and kurtosis within ±2 are generally acceptable."
                )

    # ──────────────────────────────────────────────────────────────
    # SECTION 6: Distribution Plots
    # ──────────────────────────────────────────────────────────────
    if _composite_cols and conditions:
        with st.expander("Distribution Analysis", expanded=False):
            _sel_dist_dv = st.selectbox(
                "Select scale for distribution analysis",
                _composite_cols,
                format_func=lambda x: x.replace("_mean", "").replace("_", " "),
                key="dashboard_dist_select",
            )
            if _sel_dist_dv:
                fig = make_subplots(
                    rows=1, cols=n_conditions,
                    subplot_titles=[c for c in conditions],
                    shared_yaxes=True,
                )
                for i, cond in enumerate(conditions):
                    _vals = df.loc[df["CONDITION"] == cond, _sel_dist_dv].dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=_vals, nbinsx=20,
                            marker_color=_THEME["colorway"][i % len(_THEME["colorway"])],
                            opacity=0.8,
                            name=cond,
                        ),
                        row=1, col=i + 1,
                    )
                fig.update_layout(
                    **_THEME,
                    title=dict(text=f"Response Distributions: {_sel_dist_dv.replace('_mean','').replace('_',' ')}",
                               font=dict(size=15, color="#1f2937")),
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    # SECTION 7: Data Quality Metrics
    # ──────────────────────────────────────────────────────────────
    with st.expander("Data Quality Metrics", expanded=False):
        _q_cols = st.columns(4)

        # Missing data rate
        _total_cells = df.shape[0] * df.shape[1]
        _missing_cells = int(df.isna().sum().sum())
        _missing_pct = (_missing_cells / _total_cells * 100) if _total_cells > 0 else 0
        _q_cols[0].metric("Missing Data", f"{_missing_pct:.1f}%", help="Percentage of cells with missing values")

        # Attention check pass rate
        if "Attention_Pass_Rate" in df.columns:
            _att_mean = float(df["Attention_Pass_Rate"].mean())
            _q_cols[1].metric("Attention Pass", f"{_att_mean:.0%}")
        else:
            _q_cols[1].metric("Attention Pass", "N/A")

        # Exclusion rate
        if "Exclude_Recommended" in df.columns:
            _excl_rate = float(df["Exclude_Recommended"].mean())
            _q_cols[2].metric("Exclusion Rate", f"{_excl_rate:.1%}")
        else:
            _q_cols[2].metric("Exclusion Rate", "N/A")

        # Condition balance
        if "CONDITION" in df.columns:
            _cond_counts = df["CONDITION"].value_counts()
            _balance = 1 - (_cond_counts.max() - _cond_counts.min()) / max(1, len(df))
            _q_cols[3].metric("Balance Score", f"{_balance:.2f}")

        # v1.8.9: Data completeness per scale
        if _composite_cols:
            st.markdown("**Scale Completeness**")
            _comp_rows = []
            for col in _composite_cols:
                _n_valid = int(df[col].notna().sum())
                _n_total = len(df)
                _pct = (_n_valid / _n_total * 100) if _n_total > 0 else 0
                _comp_rows.append({
                    "Scale": col.replace("_mean", "").replace("_", " "),
                    "Valid": _n_valid,
                    "Missing": _n_total - _n_valid,
                    "Complete %": round(_pct, 1),
                })
            if _comp_rows:
                st.dataframe(
                    pd.DataFrame(_comp_rows).style.map(
                        lambda v: "background-color: #dcfce7" if isinstance(v, float) and v >= 95
                        else "background-color: #fef9c3" if isinstance(v, float) and v >= 80
                        else "background-color: #fee2e2" if isinstance(v, (int, float)) else "",
                        subset=["Complete %"],
                    ),
                    use_container_width=True,
                )

        # Missing data detail
        _md_meta = metadata.get("missing_data", {})
        if _md_meta.get("total_missing_rate", 0) > 0 or _md_meta.get("dropout_count", 0) > 0:
            st.markdown("**Missing Data Details**")
            _md_cols = st.columns(3)
            _md_cols[0].metric(
                "Item-level Missing",
                f"{_md_meta.get('missing_data_rate', 0) * 100:.1f}% (configured)",
                delta=f"{_md_meta.get('total_missing_rate', 0) * 100:.1f}% realized",
                delta_color="off",
            )
            _md_cols[1].metric(
                "Survey Dropouts",
                f"{_md_meta.get('dropout_count', 0)} participants",
                delta=f"{_md_meta.get('dropout_rate', 0) * 100:.0f}% configured",
                delta_color="off",
            )
            _md_cols[2].metric(
                "Mechanism",
                _md_meta.get("mechanism", "realistic").capitalize(),
            )

        # Condition demographics breakdown
        if "CONDITION" in df.columns and "Gender" in df.columns and conditions:
            st.markdown("**Condition Demographics**")
            _demo_rows = []
            for cond in conditions:
                _cond_df = df[df["CONDITION"] == cond]
                _n_cond = len(_cond_df)
                _age_mean = float(_cond_df["Age"].mean()) if "Age" in df.columns and _n_cond > 0 else 0
                _age_sd = float(_cond_df["Age"].std()) if "Age" in df.columns and _n_cond > 1 else 0
                _gender_counts = _cond_df["Gender"].value_counts()
                _pct_female = float(_gender_counts.get("Female", 0) / max(_n_cond, 1) * 100)
                _pct_male = float(_gender_counts.get("Male", 0) / max(_n_cond, 1) * 100)
                _demo_rows.append({
                    "Condition": cond,
                    "N": _n_cond,
                    "Age M (SD)": f"{_age_mean:.1f} ({_age_sd:.1f})",
                    "% Female": round(_pct_female, 1),
                    "% Male": round(_pct_male, 1),
                })
            if _demo_rows:
                st.dataframe(pd.DataFrame(_demo_rows), use_container_width=True)

        # Reliability estimates (Cronbach's alpha)
        _alpha_rows = []
        for s in scales:
            sname = str(s.get("variable_name", s.get("name", ""))).strip().replace(" ", "_")
            n_items = int(s.get("num_items", 1))
            if n_items < 2:
                continue
            _item_cols = [f"{sname}_{i}" for i in range(1, n_items + 1) if f"{sname}_{i}" in df.columns]
            if len(_item_cols) >= 2:
                _item_data = df[_item_cols].dropna()
                if len(_item_data) > 10:
                    _k = len(_item_cols)
                    _item_vars = _item_data.var(axis=0, ddof=1)
                    _total_var = _item_data.sum(axis=1).var(ddof=1)
                    if _total_var > 0:
                        _alpha = (_k / (_k - 1)) * (1 - _item_vars.sum() / _total_var)
                        _alpha_rows.append({"Scale": sname.replace("_", " "), "Items": _k, "Cronbach's α": round(float(_alpha), 3)})

        if _alpha_rows:
            st.markdown("**Scale Reliability (Cronbach's α)**")
            _alpha_df = pd.DataFrame(_alpha_rows)
            st.dataframe(
                _alpha_df.style.map(
                    lambda v: "background-color: #dcfce7" if isinstance(v, float) and v >= 0.7
                    else "background-color: #fef9c3" if isinstance(v, float) and v >= 0.6
                    else "background-color: #fee2e2" if isinstance(v, float) else "",
                    subset=["Cronbach's α"],
                ),
                use_container_width=True,
            )


def _render_feedback_button() -> None:
    """
    Render a compact feedback/bug report section at the bottom of each page.
    v1.5.0: Made compact — no more bulky header, just a clean expander.
    """
    st.markdown("---")

    with st.expander("Report a bug or send feedback", expanded=False):
        feedback_type = st.radio(
            "What would you like to do?",
            options=["Report a bug 🐛", "Send a recommendation 💡", "Both (bug + recommendation) 🔧"],
            horizontal=True,
            key="feedback_type_radio"
        )

        feedback_subject_prefix = {
            "Report a bug 🐛": "[BUG REPORT]",
            "Send a recommendation 💡": "[RECOMMENDATION]",
            "Both (bug + recommendation) 🔧": "[BUG + RECOMMENDATION]"
        }

        user_email = st.text_input(
            "Your email (optional, for follow-up)",
            placeholder="your.email@example.com",
            key="feedback_user_email"
        )

        feedback_message = st.text_area(
            "Describe the bug or your recommendation",
            placeholder="Please provide as much detail as possible. For bugs: what were you trying to do? What happened instead? For recommendations: what feature would you like to see?",
            height=150,
            key="feedback_message"
        )

        if st.button("📧 Send Feedback", type="primary", key="send_feedback_btn"):
            if not feedback_message.strip():
                st.error("Please enter a message before sending.")
            else:
                # Prepare email
                prefix = feedback_subject_prefix.get(feedback_type, "[FEEDBACK]")
                subject = f"{prefix} Behavioral Experiment Simulation Tool v{APP_VERSION}"

                body = f"""
{prefix} - Behavioral Experiment Simulation Tool
{'='*60}

FEEDBACK TYPE: {feedback_type}

FROM: {user_email if user_email else 'Anonymous'}

MESSAGE:
{feedback_message}

{'='*60}
SYSTEM INFO:
- App Version: {APP_VERSION}
- Build ID: {BUILD_ID}
- Timestamp: {datetime.now().isoformat()}
- Study Title: {st.session_state.get('study_title', 'N/A')}
"""

                # Try to send via SMTP
                ok, msg = _send_email(
                    to_email=FEEDBACK_EMAIL,
                    subject=subject,
                    body_text=body,
                )

                if ok:
                    st.success("✅ Thank you! Your feedback has been sent successfully.")
                    st.balloons()
                else:
                    # Fallback: show mailto link
                    mailto_subject = subject.replace(" ", "%20")
                    mailto_body = feedback_message.replace("\n", "%0A").replace(" ", "%20")
                    mailto_link = f"mailto:{FEEDBACK_EMAIL}?subject={mailto_subject}&body={mailto_body}"

                    st.warning(
                        f"Email service unavailable. Please send your feedback manually:\n\n"
                        f"**Email:** {FEEDBACK_EMAIL}\n\n"
                        f"**Subject:** {subject}"
                    )
                    st.markdown(f"[📧 Click here to open your email client]({mailto_link})")

        st.caption(f"Feedback is sent to Dr. Eugen Dimant ({FEEDBACK_EMAIL})")


def _send_email_with_smtp(
    to_email: str,
    subject: str,
    body_text: str,
    attachments: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[bool, str]:
    """
    Send an email using SMTP (free alternative to SendGrid).

    Supports Gmail, Google Workspace, Outlook, or any SMTP provider.

    Required Streamlit secrets:
        - SMTP_SERVER (e.g., "smtp.gmail.com")
        - SMTP_PORT (e.g., 587)
        - SMTP_USERNAME (your email address)
        - SMTP_PASSWORD (app password for Gmail/Google Workspace)
        - SMTP_FROM_EMAIL (sender email, usually same as username)

    Optional:
        - SMTP_FROM_NAME (display name)
        - SMTP_USE_TLS (default True)

    Returns: (ok, message)
    """
    import smtplib
    import ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

    # Get SMTP configuration from secrets
    smtp_server = st.secrets.get("SMTP_SERVER", "")
    smtp_port = int(st.secrets.get("SMTP_PORT", 587))
    smtp_username = st.secrets.get("SMTP_USERNAME", "")
    smtp_password = st.secrets.get("SMTP_PASSWORD", "")
    from_email = st.secrets.get("SMTP_FROM_EMAIL", smtp_username)
    from_name = st.secrets.get("SMTP_FROM_NAME", "Behavioral Experiment Simulation Tool")
    use_tls = st.secrets.get("SMTP_USE_TLS", True)

    if not smtp_server or not smtp_username or not smtp_password:
        return False, "Email not configured. Contact the administrator."

    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = f"{from_name} <{from_email}>" if from_name else from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach body text
        msg.attach(MIMEText(body_text, 'plain'))

        # Attach files if provided
        if attachments:
            for filename, data in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(data)
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(part)

        # Create SSL context for secure connection
        context = ssl.create_default_context()

        # Connect and send
        if smtp_port == 465:
            # SSL connection (less common)
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30, context=context)
        else:
            # TLS connection (most common - Gmail, Google Workspace, Outlook)
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.ehlo()
            if use_tls:
                server.starttls(context=context)
                server.ehlo()

        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()

        return True, "Email sent successfully!"

    except smtplib.SMTPAuthenticationError as e:
        error_msg = str(e)
        if "Username and Password not accepted" in error_msg or "535" in error_msg:
            return False, "Authentication failed. For Gmail/Google Workspace: use an App Password (not your regular password). Go to Google Account > Security > App passwords."
        return False, f"Authentication failed: {error_msg}"
    except smtplib.SMTPConnectError:
        return False, f"Could not connect to {smtp_server}:{smtp_port}. Check server settings."
    except smtplib.SMTPRecipientsRefused:
        return False, f"Recipient address rejected: {to_email}"
    except ssl.SSLError as e:
        return False, f"SSL/TLS error: {str(e)}. Try changing SMTP_PORT to 465."
    except Exception as e:
        return False, f"Email failed: {str(e)}"


def _send_email(
    to_email: str,
    subject: str,
    body_text: str,
    attachments: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[bool, str]:
    """
    Send an email using the configured method (SMTP).

    This is the main email function that should be called throughout the app.
    It uses free SMTP (Gmail, Outlook, etc.) instead of paid services.

    Returns: (ok, message)
    """
    return _send_email_with_smtp(to_email, subject, body_text, attachments)


def _clean_condition_name(condition: str) -> str:
    """Remove common suffixes, prefixes, and clean up condition names.

    Enhanced cleaning (Iteration 4):
    - Removes HTML tags and entities
    - Strips Qualtrics artifact patterns
    - Normalizes whitespace and special characters
    - Preserves meaningful content while removing noise
    """
    if not condition:
        return ""

    import re
    import html

    # First, decode HTML entities
    cleaned = html.unescape(str(condition))

    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)

    # Remove common Qualtrics suffixes/artifacts
    patterns_to_remove = [
        r'\s*\(new\)',           # (new)
        r'\s*\(copy\)',          # (copy)
        r'\s*\(old\)',           # (old)
        r'\s*- copy\s*$',        # - copy at end
        r'\s*copy\s+of\s+',      # copy of at start
        r'\s*_v\d+$',            # _v1, _v2, etc.
        r'\s*v\d+$',             # v1, v2 at end
        r'\s*_old$',             # _old
        r'\s*_new$',             # _new
        r'\s*_backup$',          # _backup
        r'\s*\(unused\)',        # (unused)
        r'\s*\[unused\]',        # [unused]
        r'\s*_\d+$',             # trailing _1, _2 (but not if it's the whole ID like "Condition_1")
        r'\s*\(\d+\)\s*$',       # trailing (1), (2)
        r'\s*#\d+$',             # trailing #1, #2
    ]

    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove common Qualtrics block prefixes
    prefix_patterns = [
        r'^Block\s*\d+\s*[-:]\s*',           # Block 1:
        r'^BL_\w+\s*[-:]\s*',                # BL_abc123:
        r'^FL_\w+\s*[-:]\s*',                # FL_abc123:
        r'^Default\s*Block\s*[-:]\s*',       # Default Block:
        r'^Untitled\s*Block\s*[-:]\s*',      # Untitled Block:
        r'^\[\w+\]\s*',                      # [Block], [Randomizer], etc.
    ]

    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Normalize separators - convert various separators to consistent format
    # but preserve meaningful patterns like "AI × Hedonic"
    cleaned = re.sub(r'\s*[|]\s*', ' | ', cleaned)  # Normalize pipe
    cleaned = re.sub(r'\s*[×xX]\s*', ' × ', cleaned)  # Normalize multiplication

    # Clean up multiple spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip(' -_:')

    # If cleaning removed everything, return a sanitized version of original
    if not cleaned:
        return re.sub(r'[^\w\s-]', '', str(condition)).strip()[:50] or "Condition"

    return cleaned


def _normalize_condition_for_comparison(condition: str) -> str:
    """Normalize a condition name for duplicate detection."""
    if not condition:
        return ""
    # Lowercase, remove spaces/underscores/hyphens, strip common words
    normalized = condition.lower()
    normalized = re.sub(r'[\s_\-]+', '', normalized)
    # Remove common prefixes/suffixes that don't differentiate conditions
    normalized = re.sub(r'^(condition|cond|group|arm|treatment|trt)', '', normalized)
    return normalized


def _infer_factor_name(levels: List[str]) -> str:
    """
    Try to infer a meaningful factor name from its levels.

    This is designed to work with ANY QSF file by using general heuristics
    rather than hardcoded patterns. The approach:
    1. Check for common experimental design patterns
    2. Look for semantic patterns (presence/absence, high/low, etc.)
    3. Find common terms across levels
    4. Fall back to generic names
    """
    if not levels:
        return "Factor"

    # Normalize levels for comparison
    level_set = set(l.lower().strip() for l in levels)
    levels_lower = [l.lower().strip() for l in levels]

    # ==========================================================
    # PATTERN 1: Presence/Absence patterns (AI, Robot, Human, etc.)
    # ==========================================================
    presence_patterns = {
        ('ai', 'no ai', 'noai', 'without ai', 'no_ai'): "AI Presence",
        ('robot', 'no robot', 'human', 'norobot'): "Agent Type",
        ('chatbot', 'no chatbot', 'human agent'): "Agent Type",
        ('personalization', 'no personalization', 'personalized', 'not personalized'): "Personalization",
        ('disclosure', 'no disclosure', 'disclosed', 'not disclosed'): "Disclosure",
        ('incentive', 'no incentive', 'incentivized'): "Incentive",
        ('feedback', 'no feedback'): "Feedback",
        ('warning', 'no warning'): "Warning",
        ('label', 'no label', 'labeled', 'unlabeled'): "Labeling",
    }

    for patterns, factor_name in presence_patterns.items():
        if any(any(p in l for p in patterns) for l in level_set):
            return factor_name

    # ==========================================================
    # PATTERN 2: Control/Treatment designs
    # ==========================================================
    if 'control' in level_set or 'treatment' in level_set or 'placebo' in level_set:
        return "Condition"

    # ==========================================================
    # PATTERN 3: High/Low/Medium patterns
    # ==========================================================
    has_high = any('high' in l for l in level_set)
    has_low = any('low' in l for l in level_set)
    has_medium = any('medium' in l or 'mid' in l or 'moderate' in l for l in level_set)

    if has_high or has_low or has_medium:
        # Try to find what's being varied
        for level in levels_lower:
            for term in ['high', 'low', 'medium', 'mid', 'moderate']:
                if term in level:
                    base = level.replace(term, '').strip(' -_')
                    if base and len(base) > 1:
                        return base.title().replace('_', ' ')
        return "Level"

    # ==========================================================
    # PATTERN 4: Product/Category types
    # ==========================================================
    product_patterns = {
        ('hedonic', 'utilitarian', 'functional', 'experiential', 'luxury', 'necessity'): "Product Type",
        ('food', 'electronics', 'clothing', 'service'): "Category",
        ('positive', 'negative', 'neutral'): "Valence",
        ('happy', 'sad', 'angry', 'neutral'): "Emotion",
        ('gain', 'loss'): "Frame",
        ('promotion', 'prevention'): "Regulatory Focus",
    }

    for patterns, factor_name in product_patterns.items():
        if any(p in level_set for p in patterns):
            return factor_name

    # ==========================================================
    # PATTERN 5: Price/Money patterns
    # ==========================================================
    if any('price' in l or '$' in l or 'expensive' in l or 'cheap' in l for l in level_set):
        return "Price"

    # ==========================================================
    # PATTERN 6: Brand patterns
    # ==========================================================
    if any('brand' in l for l in level_set):
        return "Brand"

    # ==========================================================
    # PATTERN 7: Time-related patterns
    # ==========================================================
    if any(t in level_set for t in ['past', 'present', 'future', 'before', 'after', 'now', 'later']):
        return "Time Frame"

    # ==========================================================
    # PATTERN 8: Source/Origin patterns
    # ==========================================================
    if any(s in level_set for s in ['local', 'foreign', 'domestic', 'imported', 'organic', 'conventional']):
        return "Source"

    # ==========================================================
    # PATTERN 9: Social patterns
    # ==========================================================
    if any(s in level_set for s in ['social', 'individual', 'group', 'alone', 'public', 'private']):
        return "Social Context"

    # ==========================================================
    # PATTERN 10: Try to find common differentiating terms
    # ==========================================================
    if len(levels) >= 2:
        # Find what makes levels different (the varying part)
        words_lists = [l.split() for l in levels_lower]

        # Find words that appear in some but not all levels
        all_words = set()
        for wl in words_lists:
            all_words.update(wl)

        varying_words = set()
        for word in all_words:
            count = sum(1 for wl in words_lists if word in wl)
            if 0 < count < len(words_lists):
                varying_words.add(word)

        # Filter out noise words
        noise_words = {'the', 'a', 'an', 'with', 'without', 'no', 'vs', 'and', 'or', 'in', 'on', 'for', 'to', 'of', 'is'}
        varying_words -= noise_words

        if varying_words:
            # Use the longest varying word as potential factor name
            best_word = max(varying_words, key=len)
            if len(best_word) > 2:
                return best_word.title()

        # Find common words across all levels (the constant part)
        words_sets = [set(wl) for wl in words_lists]
        common_words = set.intersection(*words_sets) if words_sets else set()
        common_words -= noise_words

        if common_words:
            # Use common words as factor name
            common_str = ' '.join(sorted(common_words, key=len, reverse=True)[:2])
            if common_str and len(common_str) > 2:
                return common_str.title()

    # ==========================================================
    # FALLBACK: Generic factor name
    # ==========================================================
    return "Factor"


def _infer_factors_from_conditions(conditions: List[str]) -> List[Dict[str, Any]]:
    """
    Parse factorial design from condition names.

    This function is designed to work with ANY QSF file by:
    1. Normalizing separator patterns first (× x X etc. → standard ×)
    2. Handling underscore-separated names (e.g., "AI_Hedonic")
    3. Detecting numeric suffixes as potential factor levels
    4. Handling factorial + control designs (where one condition may not fit the pattern)
    5. NEVER including interaction terms as factor levels
    6. Falling back gracefully to single-factor designs

    Examples:
    - "No AI x Utilitarian", "AI x Hedonic" → 2 factors: [AI, No AI] and [Hedonic, Utilitarian]
    - "Control", "Treatment" → 1 factor: [Control, Treatment]
    - "Cond_1_A", "Cond_1_B", "Cond_2_A", "Cond_2_B" → 2 factors
    """
    if not conditions:
        return [{"name": "Condition", "levels": ["Condition A"]}]

    # Clean all condition names first
    conditions = [_clean_condition_name(c) for c in conditions]
    conditions = [c for c in conditions if c]  # Remove empty

    if not conditions:
        return [{"name": "Condition", "levels": ["Condition A"]}]

    # If only one condition, return single factor
    if len(conditions) == 1:
        return [{"name": "Condition", "levels": conditions}]

    # ==========================================================
    # NORMALIZE ALL SEPARATOR VARIATIONS TO STANDARD " × "
    # ==========================================================
    # This is critical to prevent mixed separators from causing issues
    normalized_conditions = []
    for c in conditions:
        norm = c
        # Replace all separator variations with standard " × "
        for sep in [" x ", " X ", " | ", "_x_", " * ", " & "]:
            norm = norm.replace(sep, " × ")
        normalized_conditions.append(norm)

    # ==========================================================
    # DETECT FACTORIAL STRUCTURE
    # ==========================================================
    # Count conditions with the normalized separator
    separator = " × "
    factorial_conditions = [c for c in normalized_conditions if separator in c]
    non_matching = [conditions[i] for i, c in enumerate(normalized_conditions) if separator not in c]

    # ==========================================================
    # PARSE USING FACTORIAL SEPARATOR
    # ==========================================================
    if len(factorial_conditions) >= 2:
        split_rows = [c.split(separator) for c in factorial_conditions]
        max_parts = max(len(r) for r in split_rows)

        # Check if splitting is consistent among matching conditions
        consistent_rows = [r for r in split_rows if len(r) == max_parts]

        if len(consistent_rows) >= 2 and max_parts > 1:
            # Extract unique levels for each factor position
            # IMPORTANT: Only include single-level terms, never interaction terms
            factors: List[Dict[str, Any]] = []
            for j in range(max_parts):
                raw_levels = [r[j].strip() for r in consistent_rows if len(r) > j and r[j].strip()]
                # Filter out any interaction-looking terms (contain × or x)
                levels = sorted(list({
                    l for l in raw_levels
                    if " × " not in l and " x " not in l.lower() and " X " not in l
                }))
                if not levels:
                    continue
                factor_name = _infer_factor_name(levels)
                # Make factor name unique if needed
                existing_names = [f.get("name") for f in factors]
                if factor_name in existing_names:
                    factor_name = f"{factor_name} {j+1}"
                factors.append({"name": factor_name, "levels": levels})

            if factors and len(factors) >= 2:
                # Store info about non-matching conditions (potential controls)
                if non_matching:
                    for f in factors:
                        f["_non_matching_conditions"] = non_matching
                return factors

    # ==========================================================
    # TRY UNDERSCORE-BASED PARSING (e.g., "AI_Hedonic", "NoAI_Utilitarian")
    # ==========================================================
    # Only if conditions have underscores and consistent structure
    underscore_rows = [c.split('_') for c in conditions if '_' in c]
    if len(underscore_rows) >= len(conditions) - 1 and len(underscore_rows) > 1:  # Allow 1 non-matching
        parts_count = [len(r) for r in underscore_rows]
        most_common_parts = max(set(parts_count), key=parts_count.count)
        consistent_rows = [r for r in underscore_rows if len(r) == most_common_parts]

        if len(consistent_rows) >= 2 and most_common_parts > 1:
            factors = []
            for j in range(most_common_parts):
                levels = sorted(list({r[j].strip() for r in consistent_rows if r[j].strip()}))
                if len(levels) > 1:  # Only include if there's variation
                    factor_name = _infer_factor_name(levels)
                    existing_names = [f.get("name") for f in factors]
                    if factor_name in existing_names:
                        factor_name = f"{factor_name} {j+1}"
                    factors.append({"name": factor_name, "levels": levels})

            if len(factors) > 1:
                return factors

    # ==========================================================
    # TRY TO DETECT NUMERIC SUFFIXES AS FACTOR LEVELS
    # ==========================================================
    # e.g., "Cond1A", "Cond1B", "Cond2A", "Cond2B" → 2 factors
    import re
    numeric_pattern = re.compile(r'^(.+?)(\d+)(.*)$')
    parsed_parts = []
    for c in conditions:
        match = numeric_pattern.match(c)
        if match:
            parsed_parts.append((match.group(1), match.group(2), match.group(3)))
        else:
            parsed_parts.append((c, '', ''))

    # Check if we have consistent numeric patterns
    if all(p[1] for p in parsed_parts):  # All have numbers
        # Check if we have multiple levels in each "position"
        prefixes = set(p[0] for p in parsed_parts)
        numbers = set(p[1] for p in parsed_parts)
        suffixes = set(p[2] for p in parsed_parts)

        if len(numbers) > 1 and len(suffixes) > 1:
            # Potential 2-factor design
            factors = []
            if len(numbers) > 1:
                factor_name = _infer_factor_name(list(numbers))
                factors.append({"name": factor_name if factor_name != "Factor" else "Factor 1", "levels": sorted(list(numbers))})
            if len(suffixes) > 1:
                factor_name = _infer_factor_name(list(suffixes))
                factors.append({"name": factor_name if factor_name != "Factor" else "Factor 2", "levels": sorted(list(suffixes))})

            if len(factors) > 1:
                return factors

    # ==========================================================
    # FALLBACK: Single factor design
    # ==========================================================
    factor_name = _infer_factor_name(conditions)
    return [{"name": factor_name if factor_name != "Factor" else "Condition", "levels": conditions}]


def _render_manual_factor_config(
    all_conditions: List[str],
    auto_detected: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Render manual factor configuration UI.

    This allows users to define factors and their levels directly,
    with validation against the detected conditions.
    Supports up to 3x3x3 designs (3 factors with up to 3 levels each).
    """
    import streamlit as st

    # Always allow up to 3 factors for factorial designs
    # User can select 1, 2, or 3 factors regardless of condition count
    default_num = len(auto_detected) if auto_detected else 2

    num_factors = st.selectbox(
        "Number of factors",
        options=[1, 2, 3],
        index=min(default_num - 1, 2),
        key="manual_num_factors",
        help="Select how many independent variables (factors) you have. Supports up to 3x3x3 designs.",
    )

    factors = []

    # Parse condition names to extract potential levels
    # IMPORTANT: Normalize separators first and only include conditions that match the pattern
    # This prevents interaction terms from being included as factor levels

    # Normalize all separator variations to a standard one
    normalized_conditions = []
    for c in all_conditions:
        # Replace various separator types with standard " × "
        norm = c.replace(" x ", " × ").replace(" X ", " × ").replace(" | ", " × ").replace("_x_", " × ")
        normalized_conditions.append(norm)

    # Detect if we have factorial structure (separator present)
    separator = " × "
    factorial_conditions = [c for c in normalized_conditions if separator in c]
    non_factorial_conditions = [c for c in normalized_conditions if separator not in c]

    # Parse only factorial conditions (exclude controls/non-matching)
    parsed_conditions = []
    if factorial_conditions:
        for c in factorial_conditions:
            parts = [_clean_condition_name(p.strip()) for p in c.split(separator)]
            parsed_conditions.append(parts)

    # Extract unique levels for each factor position (excluding interaction terms)
    max_parts = max(len(p) for p in parsed_conditions) if parsed_conditions else 1

    for f_idx in range(num_factors):
        st.markdown(f"---")
        st.markdown(f"**Factor {f_idx + 1}**")

        # Default name from auto-detection or generic
        default_name = (
            auto_detected[f_idx].get("name", f"Factor {f_idx + 1}")
            if auto_detected and f_idx < len(auto_detected)
            else f"Factor {f_idx + 1}"
        )

        factor_name = st.text_input(
            "Factor name",
            value=default_name,
            key=f"manual_factor_name_{f_idx}",
            placeholder="e.g., AI Presence, Product Type",
        )

        # Extract potential levels from parsed conditions ONLY
        # This ensures we only get single factor levels, not interaction combinations
        if f_idx < max_parts and parsed_conditions:
            potential_levels = sorted(list({p[f_idx] for p in parsed_conditions if len(p) > f_idx and p[f_idx]}))
            # Filter out any remaining interaction-looking terms (contain × or x)
            potential_levels = [l for l in potential_levels if " × " not in l and " x " not in l.lower()]
        else:
            potential_levels = []

        # Show levels input - user can type or select
        st.markdown("**Factor levels** (the different values this factor can take):")

        if potential_levels:
            st.caption(f"Detected from conditions: {', '.join(potential_levels)}")
            use_detected = st.checkbox(
                f"Use detected levels for Factor {f_idx + 1}",
                value=True,
                key=f"use_detected_levels_{f_idx}",
            )
            if use_detected:
                levels = potential_levels
            else:
                levels_input = st.text_input(
                    "Enter levels (comma-separated)",
                    value=", ".join(potential_levels),
                    key=f"manual_levels_input_{f_idx}",
                    placeholder="e.g., AI, No AI",
                )
                levels = [l.strip() for l in levels_input.split(",") if l.strip()]
        else:
            levels_input = st.text_input(
                "Enter levels (comma-separated)",
                value="",
                key=f"manual_levels_input_{f_idx}",
                placeholder="e.g., AI, No AI",
            )
            levels = [l.strip() for l in levels_input.split(",") if l.strip()]

        if levels:
            st.info(f"Levels: {' | '.join(levels)}")
            factors.append({"name": factor_name, "levels": levels})
        else:
            st.warning(f"Please define levels for Factor {f_idx + 1}")

    # Show expected condition combinations
    if len(factors) > 1:
        st.markdown("---")
        st.markdown("**Expected condition combinations:**")
        import itertools
        all_combos = list(itertools.product(*[f["levels"] for f in factors]))
        combo_names = [" × ".join(combo) for combo in all_combos]
        st.caption(f"{len(combo_names)} combinations: {', '.join(combo_names[:8])}{'...' if len(combo_names) > 8 else ''}")

        # Validate against actual conditions
        if len(combo_names) != len(all_conditions):
            st.warning(
                f"Note: {len(combo_names)} factorial combinations vs {len(all_conditions)} actual conditions. "
                "This may indicate a non-pure factorial design (e.g., with control)."
            )

    return factors


def _render_factorial_design_table(
    detected_conditions: List[str],
    session_key_prefix: str = "factorial_table",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Render an enhanced factorial design table UI with clear visual feedback.

    Creates a visual table where users assign conditions to factors.
    Returns factors and crossed condition names, saved to session state for persistence.

    Features:
    - Color-coded factor assignment
    - Real-time condition count
    - Visual crossing table
    - Clear validation feedback
    """
    import streamlit as st
    import itertools

    # Clean condition names
    clean_conditions = [_clean_condition_name(c) for c in detected_conditions]

    # Header with design type selector
    st.markdown("**Configure your factorial design:**")

    # Number of factors - compact inline with helpful text
    col_factors, col_help = st.columns([2, 3])
    with col_factors:
        num_factors = st.radio(
            "Number of factors",
            options=[2, 3],
            index=0,
            horizontal=True,
            key=f"{session_key_prefix}_num_factors",
            help="2 factors = 2×2, 2×3, 3×3 designs. 3 factors = 2×2×2, etc."
        )
    with col_help:
        if num_factors == 2:
            st.caption("Examples: 2×2 (4 cells), 2×3 (6 cells), 3×3 (9 cells)")
        else:
            st.caption("Examples: 2×2×2 (8 cells), 2×2×3 (12 cells)")

    st.markdown("---")

    # Factor 1 Configuration - with visual indicator
    st.markdown("**Factor 1** (rows in design table)")
    col_f1_name, col_f1_levels = st.columns([1, 3])
    with col_f1_name:
        factor1_name = st.text_input(
            "Name",
            value=st.session_state.get(f"{session_key_prefix}_f1_name", "Factor 1"),
            key=f"{session_key_prefix}_f1_name_input",
            label_visibility="collapsed",
            placeholder="e.g., Game Type"
        )
        st.session_state[f"{session_key_prefix}_f1_name"] = factor1_name
    with col_f1_levels:
        factor1_levels = st.multiselect(
            "Levels",
            options=clean_conditions,
            default=st.session_state.get(f"{session_key_prefix}_factor1_levels", []),
            key=f"{session_key_prefix}_f1_levels_select",
            label_visibility="collapsed",
            placeholder="Select levels for Factor 1..."
        )
        st.session_state[f"{session_key_prefix}_factor1_levels"] = factor1_levels
    if factor1_levels:
        st.caption(f"✓ {len(factor1_levels)} level(s): {', '.join(factor1_levels)}")

    # Factor 2 Configuration
    remaining_conditions = [c for c in clean_conditions if c not in factor1_levels]
    st.markdown("**Factor 2** (columns in design table)")
    col_f2_name, col_f2_levels = st.columns([1, 3])
    with col_f2_name:
        factor2_name = st.text_input(
            "Name",
            value=st.session_state.get(f"{session_key_prefix}_f2_name", "Factor 2"),
            key=f"{session_key_prefix}_f2_name_input",
            label_visibility="collapsed",
            placeholder="e.g., Partner Type"
        )
        st.session_state[f"{session_key_prefix}_f2_name"] = factor2_name
    with col_f2_levels:
        factor2_levels = st.multiselect(
            "Levels",
            options=remaining_conditions,
            default=[c for c in st.session_state.get(f"{session_key_prefix}_factor2_levels", []) if c in remaining_conditions],
            key=f"{session_key_prefix}_f2_levels_select",
            label_visibility="collapsed",
            placeholder="Select levels for Factor 2..."
        )
        st.session_state[f"{session_key_prefix}_factor2_levels"] = factor2_levels
    if factor2_levels:
        st.caption(f"✓ {len(factor2_levels)} level(s): {', '.join(factor2_levels)}")

    # Factor 3 Configuration (if 3 factors)
    factor3_name = ""
    factor3_levels = []
    if num_factors == 3:
        remaining_for_f3 = [c for c in remaining_conditions if c not in factor2_levels]
        st.markdown("**Factor 3** (layers in design)")
        col_f3_name, col_f3_levels = st.columns([1, 3])
        with col_f3_name:
            factor3_name = st.text_input(
                "Name",
                value=st.session_state.get(f"{session_key_prefix}_f3_name", "Factor 3"),
                key=f"{session_key_prefix}_f3_name_input",
                label_visibility="collapsed",
                placeholder="e.g., Feedback Type"
            )
            st.session_state[f"{session_key_prefix}_f3_name"] = factor3_name
        with col_f3_levels:
            factor3_levels = st.multiselect(
                "Levels",
                options=remaining_for_f3,
                default=[c for c in st.session_state.get(f"{session_key_prefix}_factor3_levels", []) if c in remaining_for_f3],
                key=f"{session_key_prefix}_f3_levels_select",
                label_visibility="collapsed",
                placeholder="Select levels for Factor 3..."
            )
            st.session_state[f"{session_key_prefix}_factor3_levels"] = factor3_levels
        if factor3_levels:
            st.caption(f"✓ {len(factor3_levels)} level(s): {', '.join(factor3_levels)}")

    # Build factors list
    factors = []
    if factor1_levels:
        factors.append({"name": factor1_name or "Factor 1", "levels": factor1_levels})
    if factor2_levels:
        factors.append({"name": factor2_name or "Factor 2", "levels": factor2_levels})
    if num_factors == 3 and factor3_levels:
        factors.append({"name": factor3_name or "Factor 3", "levels": factor3_levels})

    # Save factors to session state for persistence
    st.session_state["factorial_table_factors"] = factors

    # Generate crossed conditions
    crossed_conditions = []
    if len(factors) >= 2:
        all_combos = list(itertools.product(*[f["levels"] for f in factors]))
        crossed_conditions = [" × ".join(combo) for combo in all_combos]

        # Display the factorial design visualization
        st.markdown("---")

        if num_factors == 2 and factor1_levels and factor2_levels:
            # Design summary
            design_str = f"{len(factor1_levels)}×{len(factor2_levels)}"
            st.success(f"**{design_str} Factorial Design = {len(crossed_conditions)} conditions**")

            # Build enhanced visual table with condition names
            st.markdown(f"**Design Table** ({factor1_name} × {factor2_name}):")

            # Create table header
            header_row = f"| **{factor1_name}** \\ **{factor2_name}** |"
            for f2_level in factor2_levels:
                header_row += f" {f2_level} |"
            separator = "|" + "---|" * (len(factor2_levels) + 1)

            table_md = header_row + "\n" + separator + "\n"

            # Create table rows with cell numbers
            cell_num = 1
            for f1_level in factor1_levels:
                row = f"| **{f1_level}** |"
                for f2_level in factor2_levels:
                    row += f" Cell {cell_num} |"
                    cell_num += 1
                table_md += row + "\n"

            st.markdown(table_md)

        elif num_factors == 3 and factor1_levels and factor2_levels and factor3_levels:
            design_str = f"{len(factor1_levels)}×{len(factor2_levels)}×{len(factor3_levels)}"
            st.success(f"**{design_str} Factorial Design = {len(crossed_conditions)} conditions**")
            st.caption(f"Factors: {factor1_name} × {factor2_name} × {factor3_name}")

        # Show resulting conditions in expandable section
        st.markdown("**Resulting condition combinations:**")
        with st.expander(f"View all {len(crossed_conditions)} conditions", expanded=True):
            # Display in columns for better readability
            n_cols = min(3, len(crossed_conditions))
            cols = st.columns(n_cols)
            for i, combo in enumerate(crossed_conditions):
                with cols[i % n_cols]:
                    st.markdown(f"**{i+1}.** {combo}")

    else:
        # Helpful validation message
        missing = []
        if not factor1_levels:
            missing.append("Factor 1")
        if not factor2_levels:
            missing.append("Factor 2")
        st.warning(f"Select levels for {' and '.join(missing)} to generate the design table.")

    return factors, crossed_conditions


def _preview_to_engine_inputs(preview: QSFPreviewResult) -> Dict[str, Any]:
    """
    Convert QSFPreviewResult to engine-ready inputs with minimal assumptions.

    IMPORTANT: Only includes scales actually detected from QSF - does NOT fabricate DVs.
    Scale points are preserved from QSF detection; only defaults when truly unknown.
    """
    conditions = (preview.detected_conditions or [])[:]
    if not conditions:
        conditions = ["Condition A"]

    factors = _infer_factors_from_conditions(conditions)

    scales: List[Dict[str, Any]] = []
    seen_scale_names: set = set()  # Track to prevent duplicates

    for s in (preview.detected_scales or []):
        # Use variable_name if available (more precise), fall back to name
        name = str(s.get("variable_name", s.get("name", "Scale"))).strip() or "Scale"
        display_name = str(s.get("name", name)).strip() or name

        # Deduplicate by normalized name
        name_key = name.lower().replace(" ", "_").replace("-", "_")
        if name_key in seen_scale_names:
            continue
        seen_scale_names.add(name_key)

        # Use detected values, with reasonable defaults only when truly missing
        # Check BOTH "items" (QSF detection key) and "num_items" (engine key)
        num_items = s.get("num_items")
        if num_items is None:
            num_items = s.get("items")  # QSF detection uses "items" key
        if num_items is None or (isinstance(num_items, float) and np.isnan(num_items)):
            num_items = 5  # Default only if truly not detected
        else:
            try:
                num_items = int(num_items)
            except (ValueError, TypeError):
                num_items = 5

        scale_points = s.get("scale_points")
        if scale_points is None or (isinstance(scale_points, float) and np.isnan(scale_points)):
            # IMPORTANT: Default to 7 only when QSF doesn't specify
            scale_points = 7
        else:
            try:
                scale_points = int(scale_points)
            except (ValueError, TypeError):
                scale_points = 7

        scales.append(
            {
                "name": display_name,
                "variable_name": name,
                "num_items": max(1, num_items),
                "scale_points": max(2, min(1001, scale_points)),
                "reverse_items": s.get("reverse_items", []) or [],
                "detected_from_qsf": s.get("scale_points") is not None,
                "_validated": True,
            }
        )

    # Only add default if NO scales were detected at all
    # This prevents fabricating extra DVs
    if not scales:
        scales = [{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 5, "scale_points": 7, "reverse_items": [], "detected_from_qsf": False, "_validated": True}]

    open_ended = getattr(preview, "open_ended_questions", None) or []

    # Extract detailed open-ended info for context-aware text generation
    open_ended_details = getattr(preview, "open_ended_details", None) or []

    # Extract study context for context-aware text generation
    study_context = getattr(preview, "study_context", None) or {}

    # Extract embedded data conditions (for surveys using embedded data for randomization)
    embedded_data_conditions = getattr(preview, "embedded_data_conditions", None) or []

    return {
        "conditions": conditions,
        "factors": factors,
        "scales": scales,
        "open_ended_questions": open_ended,
        "open_ended_details": open_ended_details,
        "study_context": study_context,
        "embedded_data_conditions": embedded_data_conditions,
    }


@st.cache_resource
def _get_group_manager() -> GroupManager:
    return GroupManager()


@st.cache_resource
def _get_api_key_manager() -> APIKeyManager:
    return APIKeyManager()


@st.cache_resource
def _get_qsf_preview_parser() -> QSFPreviewParser:
    return QSFPreviewParser()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# v1.0.1.7: Minimal header on wizard pages — title is secondary to progress bar
_current_page = st.session_state.get("active_page", -1)
if _current_page >= 0:
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:-8px;">'
        f'<span style="font-size:0.85rem;font-weight:600;color:#6B7280;">{APP_TITLE}</span>'
        f'<span style="font-size:0.7rem;color:#D1D5DB;">v{APP_VERSION}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# v1.5.0: Removed legacy STEP_LABELS, STEP_DESCRIPTIONS, STEP_HELP constants
# (replaced by SECTION_META in flow navigation)

# User guidance messages for better UX
UI_GUIDANCE = {
    "study_info_complete": "Study information complete. Proceed to upload your QSF file.",
    "study_info_incomplete": "Please fill in all required study information fields.",
    "qsf_upload_success": "QSF file parsed successfully. Review detected conditions below.",
    "qsf_upload_needed": "Upload your Qualtrics survey file (.qsf) to continue.",
    "design_ready": "Design configuration complete. Ready to generate simulated data.",
    "design_needs_conditions": "Please select or define at least one experimental condition.",
    "design_needs_outcome": "Please identify at least one primary outcome variable.",
    "generation_ready": "All requirements met. Click 'Generate Simulation' to create your data package.",
    "no_conditions_detected": "No experimental conditions detected. Define conditions manually below.",
    "conditions_detected": "Experimental conditions detected from your survey flow.",
    # v1.1.0: New validation messages
    "sample_size_low": "Sample size is below recommended minimum. Consider at least 30 per condition.",
    "sample_size_ok": "Sample size is adequate for reliable statistical analyses.",
    "scales_detected": "Detected scales will generate realistic Likert-type responses.",
    "open_ended_detected": "Open-ended questions will generate unique, topic-relevant responses.",
}


def _finalize_builder_design(
    parser: "SurveyDescriptionParser",
    parsed_conditions: list,
    parsed_scales: list,
    parsed_oe: list,
    design_type: str,
    sample_size: int,
    title: str,
    description: str,
    participant_desc: str = "",
    raw_inputs: Optional[Dict[str, str]] = None,
) -> bool:
    """Build inferred_design from parsed builder data and store everything in session state.

    This consolidates the shared logic between the auto-build path (example studies)
    and the manual "Build Study Specification" submit path so neither can drift.

    Returns True on success, False on failure (with st.error displayed).
    """
    _cond_text = " ".join(c.name for c in parsed_conditions)
    _scale_text = " ".join(s.name for s in parsed_scales)
    domain = parser.detect_research_domain(
        title, description,
        conditions_text=_cond_text, scales_text=_scale_text,
    )
    factors = parser.detect_factorial_structure(parsed_conditions)

    parsed_design = ParsedDesign(
        conditions=parsed_conditions,
        scales=parsed_scales,
        open_ended=parsed_oe,
        factors=factors,
        design_type=design_type,
        sample_size=sample_size,
        research_domain=domain,
        study_title=title,
        study_description=description,
        participant_characteristics=participant_desc,
    )

    try:
        inferred = parser.build_inferred_design(parsed_design)
    except Exception as e:
        st.error(f"Error building study design: {e}")
        _log(f"build_inferred_design error: {e}", level="error")
        return False

    if not inferred.get("conditions"):
        st.warning("No conditions detected. Please check your condition description.")
        return False

    # Core design state
    st.session_state["inferred_design"] = inferred
    st.session_state["builder_parsed_design"] = parsed_design
    st.session_state["conversational_builder_complete"] = True

    # v1.0.4.7: Save builder text to shadow keys for edit-state persistence.
    # Streamlit removes widget keys when their widgets aren't rendered,
    # so by the time the user clicks "Edit" on the Design page, the widget
    # keys may have been cleaned up. Shadow keys survive across pages.
    st.session_state["_saved_conditions_text"] = st.session_state.get("builder_conditions_text", "")
    st.session_state["_saved_scales_text"] = st.session_state.get("builder_scales_text", "")
    st.session_state["_saved_oe_text"] = st.session_state.get("builder_oe_text", "")

    # Condition/scale/OE state for Design & Generate tabs
    st.session_state["selected_conditions"] = [c.name for c in parsed_conditions]
    st.session_state["confirmed_scales"] = inferred.get("scales", [])
    st.session_state["scales_confirmed"] = True
    st.session_state["confirmed_open_ended"] = inferred.get("open_ended_questions", [])
    st.session_state["open_ended_confirmed"] = True

    # v1.0.4.5: Clean up stale QSF design page state to prevent mixed UI
    st.session_state["_builder_oe_context_complete"] = True
    st.session_state.pop("_br_scale_version", None)
    st.session_state.pop("_br_oe_version", None)

    # Allocation
    st.session_state["condition_allocation"] = inferred.get("condition_allocation", {})
    _cond_names = [c.name for c in parsed_conditions]
    _n_conds = max(len(_cond_names), 1)
    _per_cell = sample_size // _n_conds
    _remainder = sample_size % _n_conds
    st.session_state["condition_allocation_n"] = {
        c: _per_cell + (1 if i < _remainder else 0)
        for i, c in enumerate(_cond_names)
    }
    st.session_state["builder_design_type"] = inferred.get("design_type", "between")
    st.session_state["builder_sample_size"] = sample_size
    st.session_state["sample_size"] = sample_size

    # Design feedback suggestions
    _feedback = parser.generate_feedback(parsed_design)
    if _feedback:
        st.session_state["_builder_feedback"] = _feedback

    # Collect synthetic QSF training data (never block user on failure)
    try:
        _ri = raw_inputs or {
            "conditions_text": _cond_text,
            "scales_text": _scale_text,
            "open_ended_text": " | ".join(q.question_text for q in parsed_oe) if parsed_oe else "",
            "study_title": title,
            "study_description": description,
            "participant_desc": participant_desc,
            "design_type": design_type,
            "sample_size": sample_size,
        }
        synthetic_qsf = generate_qsf_from_design(parsed_design, raw_inputs=_ri)
        safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title or 'untitled')[:60].strip().replace(' ', '_')
        _date_prefix = datetime.now().strftime("%Y_%m_%d")
        collect_qsf_async(f"{_date_prefix}_{safe_title}.qsf", synthetic_qsf)
    except Exception:
        pass  # Never let collection errors affect the user workflow

    return True


def _render_conversational_builder() -> None:
    """
    Render the conversational study builder interface.
    Allows users to describe their experiment in natural language
    instead of uploading a QSF file.
    """
    parser = SurveyDescriptionParser()

    # ── Pending autofill: apply BEFORE widgets render ────────────────
    # This is the reliable Streamlit pattern for programmatically setting
    # widget values. The pending data is set by the example button handler,
    # then applied here at the TOP of the next rerun — before any text_area
    # widgets are created — so st.session_state[widget_key] is correctly
    # picked up by each widget.
    _pending = st.session_state.pop("_pending_autofill_example", None)
    if _pending:
        st.session_state["builder_conditions_text"] = _pending["conditions"]
        st.session_state["builder_scales_text"] = _pending["scales"]
        st.session_state["builder_oe_text"] = _pending.get("open_ended", "")
        st.session_state["study_title"] = _pending["title"]
        _pending_desc = _pending.get(
            "description",
            f"Investigating {_pending.get('domain', _pending['title'].lower())}",
        )
        st.session_state["study_description"] = _pending_desc

        # ── Auto-build: automatically submit the builder for example studies ──
        # Examples are curated and always valid, so skip the manual "Build Study"
        # step to provide a smooth one-click experience.
        try:
            _auto_conds, _ = parser.parse_conditions(_pending["conditions"])
            _auto_scales = parser.parse_scales(_pending["scales"])
            _auto_oe = parser.parse_open_ended(_pending.get("open_ended", "")) if _pending.get("open_ended", "").strip() else []
            if len(_auto_conds) >= 2 and len(_auto_scales) >= 1:
                _auto_sample = int(st.session_state.get("builder_sample_size", 100))
                _raw_inputs = {
                    "conditions_text": _pending["conditions"],
                    "scales_text": _pending["scales"],
                    "open_ended_text": _pending.get("open_ended", ""),
                    "study_title": _pending["title"],
                    "study_description": _pending_desc,
                    "participant_desc": "",
                    "design_type": "between",
                    "sample_size": _auto_sample,
                }
                if _finalize_builder_design(
                    parser, _auto_conds, _auto_scales, _auto_oe,
                    "between", _auto_sample, _pending["title"], _pending_desc,
                    raw_inputs=_raw_inputs,
                ):
                    _navigate_to(2)
        except Exception:
            pass  # Fall through to manual builder if auto-build fails

    # ── Pending scales autofill ──────────────────────────────────────
    _pending_scales = st.session_state.pop("_pending_autofill_scales", None)
    if _pending_scales:
        st.session_state["builder_scales_text"] = _pending_scales

    # v1.0.4.7: Restore builder text from shadow keys if widget keys were
    # cleaned up by Streamlit (happens when navigating away from this page).
    for _wk, _sk in [("builder_conditions_text", "_saved_conditions_text"),
                      ("builder_scales_text", "_saved_scales_text"),
                      ("builder_oe_text", "_saved_oe_text")]:
        if _wk not in st.session_state and _sk in st.session_state:
            st.session_state[_wk] = st.session_state[_sk]

    # Check if builder is already complete
    if st.session_state.get("conversational_builder_complete"):
        if st.button("Continue to Design \u2192", key="nav_next_builder", type="primary", use_container_width=True):
            _navigate_to(2)
        if st.button("Edit my description", key="builder_reopen_edit"):
            st.session_state["conversational_builder_complete"] = False
            st.rerun()
        return

    # v1.6.1: Cleaner builder intro with inline checklist
    _has_conds = bool(st.session_state.get("builder_conditions_text", "").strip())
    _has_scales = bool(st.session_state.get("builder_scales_text", "").strip())

    st.markdown("### Describe Your Experiment")
    _check_conds = "✅" if _has_conds else "⬜"
    _check_scales = "✅" if _has_scales else "⬜"
    st.markdown(
        f"Fill in the two required fields to build your study specification:  \n"
        f"{_check_conds} Conditions &nbsp;&nbsp; {_check_scales} Scales/DVs"
    )

    # v1.6.1: Examples — prominent when empty, compact when filled
    examples = SurveyDescriptionParser.generate_example_descriptions()
    if not _has_conds and not _has_scales:
        st.markdown("**Quick start — click any example to auto-fill everything:**")
        _ex_cols = st.columns(min(len(examples), 3))
        for idx, ex in enumerate(examples):
            with _ex_cols[idx % len(_ex_cols)]:
                _ex_label = ex["title"]
                if st.button(_ex_label, key=f"example_btn_{idx}", use_container_width=True):
                    st.session_state["_pending_autofill_example"] = ex
                    _navigate_to(1)
    else:
        with st.expander("Load a different example study", expanded=False):
            _ex_cols = st.columns(min(len(examples), 3))
            for idx, ex in enumerate(examples):
                with _ex_cols[idx % len(_ex_cols)]:
                    if st.button(ex["title"], key=f"example_btn_{idx}", use_container_width=True):
                        st.session_state["_pending_autofill_example"] = ex
                        _navigate_to(1)

    # ── Section 1: Experimental Conditions ──────────────────────────────
    st.markdown("")
    st.markdown("#### Experimental Conditions *")

    # v1.0.5.1: Dual-mode condition input — structured (default) vs text (advanced)
    _cond_input_mode = st.radio(
        "How would you like to define conditions?",
        options=["Structured (recommended)", "Text / Factorial notation"],
        index=0 if not st.session_state.get("builder_conditions_text", "").strip() else 1,
        key="cond_input_mode",
        horizontal=True,
        help="Structured mode: add conditions one by one with descriptions. Text mode: type conditions as text or use factorial notation.",
    )

    if _cond_input_mode == "Structured (recommended)":
        # ── Structured condition input ──
        st.caption(
            "Add each condition with a short name and a description of what participants experience. "
            "Descriptions help the simulator understand your treatments."
        )

        # Initialize structured conditions list in session state
        if "builder_structured_conditions" not in st.session_state:
            st.session_state["builder_structured_conditions"] = []

        _struct_conds = st.session_state["builder_structured_conditions"]

        # Display existing structured conditions
        _struct_to_remove = None
        for _si, _sc in enumerate(_struct_conds):
            _sc_col1, _sc_col2, _sc_col3 = st.columns([2, 5, 0.5])
            with _sc_col1:
                st.markdown(
                    f'<div style="background:{"#E8F5E9" if _sc.get("is_control") else "#E3F2FD"};'
                    f'border-radius:6px;padding:6px 10px;margin-top:4px;">'
                    f'<strong>{_sc["name"]}</strong><br>'
                    f'<span style="font-size:0.8em;color:#666;">{"Control" if _sc.get("is_control") else "Treatment"}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _sc_col2:
                st.markdown(
                    f'<div style="color:#555;font-size:0.9em;padding-top:8px;">{_sc.get("description", "")}</div>',
                    unsafe_allow_html=True,
                )
            with _sc_col3:
                if st.button("X", key=f"rm_struct_cond_{_si}"):
                    _struct_to_remove = _si

        if _struct_to_remove is not None:
            _struct_conds.pop(_struct_to_remove)
            st.session_state["builder_structured_conditions"] = _struct_conds
            st.rerun()

        # Add new condition form
        st.markdown("---")
        # v1.0.6.1: Use versioned keys so fields reliably reset after adding a condition.
        # st.session_state.pop() doesn't always clear widget values in Streamlit;
        # incrementing the version creates a brand-new widget with an empty default.
        _struct_ver = st.session_state.get("_struct_cond_version", 0)
        _add_c1, _add_c2 = st.columns([1, 2])
        with _add_c1:
            _new_cond_name = st.text_input(
                "Condition name",
                key=f"struct_cond_name_input_v{_struct_ver}",
                placeholder="e.g., Republican identity visible",
            )
            _new_cond_type = st.selectbox(
                "Type",
                options=["Treatment", "Control"],
                key=f"struct_cond_type_input_v{_struct_ver}",
            )
        with _add_c2:
            _new_cond_desc = st.text_area(
                "What do participants experience in this condition?",
                key=f"struct_cond_desc_input_v{_struct_ver}",
                placeholder="e.g., Participants see a profile picture with a visible Republican pin. They then play a dictator game where they decide how much to share with this person.",
                height=100,
                help="Describe the manipulation or experience. This helps the simulator choose appropriate behavioral patterns, personas, and effect sizes.",
            )

        _can_add_cond = bool(_new_cond_name.strip())
        if st.button("Add condition", key=f"struct_add_cond_btn_v{_struct_ver}", disabled=not _can_add_cond, type="primary"):
            _existing_names = [c["name"].lower() for c in _struct_conds]
            if _new_cond_name.strip().lower() in _existing_names:
                st.warning(f"Condition '{_new_cond_name.strip()}' already exists.")
            else:
                _struct_conds.append({
                    "name": _new_cond_name.strip(),
                    "is_control": _new_cond_type == "Control",
                    "description": _new_cond_desc.strip(),
                })
                st.session_state["builder_structured_conditions"] = _struct_conds
                # Increment version to create fresh widgets on next rerun
                st.session_state["_struct_cond_version"] = _struct_ver + 1
                st.rerun()

        # Convert structured conditions to parsed conditions for downstream
        parsed_conditions = []
        cond_warnings = []
        for _sc in _struct_conds:
            parsed_conditions.append(ParsedCondition(
                name=_sc["name"],
                is_control=_sc.get("is_control", False),
            ))

        # Build the text representation for downstream compatibility
        if parsed_conditions:
            conditions_text = ", ".join(c.name for c in parsed_conditions)
            st.session_state["builder_conditions_text"] = conditions_text
        else:
            conditions_text = ""

        # Store condition descriptions and control flags for simulation engine
        if _struct_conds:
            st.session_state["builder_condition_descriptions"] = {
                sc["name"]: sc.get("description", "") for sc in _struct_conds
            }
            st.session_state["builder_condition_controls"] = {
                sc["name"]: sc.get("is_control", False) for sc in _struct_conds
            }

        # Status display
        if parsed_conditions:
            st.caption(f"**{len(parsed_conditions)}** condition(s) defined")
            if len(parsed_conditions) < 2:
                st.warning("At least 2 conditions needed for an experiment.")

    else:
        # ── Original text-based input (for factorial and advanced users) ──
        st.caption(
            "List your groups (comma-separated) or describe a factorial design. "
            "NxM notation, factor labels, and crossed conditions are auto-detected."
        )

        conditions_placeholder = (
            "Control, Treatment A, Treatment B\n"
            "\n"
            "Or factorial: 2 (Source: AI vs Human) × 3 (Frame: Gain, Loss, Neutral)"
        )

        conditions_text = st.text_area(
            "Conditions",
            placeholder=conditions_placeholder,
            height=100,
            key="builder_conditions_text",
            help=(
                "Describe your experimental design. Supports:\n"
                "- Simple lists: 'Control, Treatment A, Treatment B'\n"
                "- Factorial notation: '2x2', '3×2', '2x2x2'\n"
                "- Labeled factorial: '3 (Source: AI vs Human vs None) × 2 (Product: Hedonic vs Utilitarian)'\n"
                "- Parenthetical: 'Trust (high, low) and Risk (high, low)'\n"
                "Any NxM (and NxMxK, etc.) factorial is automatically detected and crossed."
            ),
        )

        # Live parsing preview for conditions
        if conditions_text.strip():
            parsed_conditions, cond_warnings = parser.parse_conditions(conditions_text)
        else:
            parsed_conditions, cond_warnings = [], []
        for cw in cond_warnings:
            st.warning(cw)

        # Check for duplicate conditions
        if parsed_conditions:
            _cond_names_lower = [c.name.lower().strip() for c in parsed_conditions]
            _seen = set()
            _dupes = []
            for cn in _cond_names_lower:
                if cn in _seen:
                    _dupes.append(cn)
                _seen.add(cn)
            if _dupes:
                st.error(f"Duplicate condition(s) detected: {', '.join(set(_dupes))}. Each condition must be unique.")

        if parsed_conditions:
            cond_names = [c.name for c in parsed_conditions]
            # Detect if factorial (conditions contain ×)
            _is_factorial = any(" × " in cn for cn in cond_names)
            if _is_factorial:
                # Count factors from first condition
                _n_facs = len(cond_names[0].split(" × "))
                st.caption(
                    f"Detected **{_n_facs}-factor factorial** design "
                    f"with **{len(cond_names)}** crossed conditions"
                )
            else:
                st.caption(f"Detected **{len(cond_names)}** conditions: {', '.join(cond_names)}")

            # Feedback for unusual condition counts
            if len(cond_names) == 1:
                st.warning(
                    "Only 1 condition detected. An experiment needs at least 2 conditions "
                    "(e.g., control vs treatment)."
                )
            elif len(cond_names) > 16:
                st.warning(
                    f"{len(cond_names)} conditions detected. This is unusually many "
                    "-- verify this is correct. Large designs need larger sample sizes."
                )

            # Check for factorial structure
            factors = parser.detect_factorial_structure(parsed_conditions)
            if factors:
                st.caption(
                    f"Factorial design detected: "
                    + " x ".join(f"{f['name']} ({', '.join(f['levels'])})" for f in factors)
                )

    # ── Section 2: Dependent Variables / Scales ─────────────────────────
    st.markdown("")
    st.markdown("#### Scales / DVs *")
    st.caption(
        "One scale per line. Include name, item count, and range. "
        "Validated instruments (Big Five, PANAS, etc.) are auto-recognized."
    )

    scales_placeholder = (
        "Trust scale, 5 items, 1-7 Likert\n"
        "Purchase intention, 3 items, 7-point\n"
        "Willingness to pay in dollars (0-100)"
    )

    scales_text = st.text_area(
        "Scales / Dependent Variables",
        placeholder=scales_placeholder,
        height=120,
        key="builder_scales_text",
        help="Describe each scale on a new line or separated by semicolons. Include item count and range when possible.",
    )

    # Live parsing preview for scales
    parsed_scales = parser.parse_scales(scales_text) if scales_text.strip() else []
    if parsed_scales:
        # Type badge mapping
        _type_badges = {
            "likert": "[Likert]",
            "slider": "[Slider]",
            "binary": "[Binary]",
            "numeric": "[Numeric]",
        }
        for s in parsed_scales:
            badge = _type_badges.get(s.scale_type, f"[{s.scale_type.title()}]")
            # Check if scale name matches a known validated instrument
            _s_lower = s.name.lower().strip()
            is_validated = any(
                _s_lower == k or _s_lower == v.get("label", "").lower()
                for k, v in KNOWN_SCALES.items()
            )
            validated_tag = " *(validated instrument)*" if is_validated else ""
            st.caption(
                f"  {badge} **{s.name}**: {s.num_items} item(s), "
                f"{s.scale_min}-{s.scale_max}{validated_tag}"
            )

    # Smart scale auto-fill based on conditions and domain
    if parsed_conditions and not scales_text.strip():
        _builder_title = st.session_state.get("study_title", "")
        _builder_desc = st.session_state.get("study_description", "")
        if _builder_title or _builder_desc or conditions_text.strip():
            _sug_domain = parser.detect_research_domain(
                _builder_title, _builder_desc,
                conditions_text=conditions_text,
            )
            _auto_scales = parser.suggest_scales_for_domain(_sug_domain)
            if _auto_scales:
                st.markdown("**Suggested scales for your domain:**")
                for _sug in _auto_scales[:4]:
                    st.caption(f"- **{_sug['name']}** ({_sug['items']} items, {_sug['range']}) -- {_sug['description']}")
                # Auto-fill button — uses pending pattern for reliable widget update
                _auto_text = "\n".join(f"{s['name']}, {s['items']} items, {s['range']}" for s in _auto_scales[:3])
                if st.button("Auto-fill suggested scales", key="auto_fill_scales_btn"):
                    st.session_state["_pending_autofill_scales"] = _auto_text
                    _navigate_to(1)

    # ── Section 3: Open-Ended Questions (Optional) ─────────────────────
    st.markdown("")
    st.markdown("#### Open-Ended Questions *(optional)*")
    st.caption("Free-text questions to simulate. Leave empty if your study has none.")

    oe_text = st.text_area(
        "Open-ended questions",
        placeholder=(
            "Examples:\n"
            "• Please explain why you made this choice\n"
            "• Describe your experience with the product\n"
            "• What are your thoughts on this policy?"
        ),
        height=100,
        key="builder_oe_text",
        help="Enter each open-ended question on a new line. Leave empty if none.",
    )

    parsed_oe = parser.parse_open_ended(oe_text) if oe_text.strip() else []
    if parsed_oe:
        st.caption(f"Detected **{len(parsed_oe)}** open-ended question(s)")

    # ── Section 4 & 5: Sample Size + Design Type (side by side) ────────
    st.markdown("")
    _cfg_col1, _cfg_col2 = st.columns([1, 1], gap="large")

    with _cfg_col1:
        st.markdown("#### Sample Size")
        builder_sample = st.number_input(
            "Total participants to simulate",
            min_value=10,
            max_value=10000,
            value=int(st.session_state.get("builder_sample_size", 100)),
            step=10,
            key="builder_sample_input",
            help="Total simulated participants across all conditions.",
        )
        st.session_state["builder_sample_size"] = builder_sample
        st.session_state["sample_size"] = builder_sample

        # Compact power guidance
        if parsed_conditions:
            n_conds = len(parsed_conditions)
            per_cell = builder_sample // max(n_conds, 1)
            recommended_min = max(30 * n_conds, 50)
            if builder_sample < recommended_min:
                st.warning(f"~{per_cell}/cell. Recommend {recommended_min}+ ({30}+ per condition).")
            else:
                st.caption(f"~{per_cell} participants per condition")

    with _cfg_col2:
        st.markdown("#### Design Type")
        # Auto-detect design type from condition structure
        _auto_design: str = "between"
        if parsed_conditions:
            cond_names_lower = " ".join(c.name.lower() for c in parsed_conditions)
            if any(w in cond_names_lower for w in [
                "pre", "post", "before", "after", "time 1", "time 2",
                "baseline", "follow", "wave 1", "wave 2", "session 1", "session 2",
            ]):
                _auto_design = "within"
            elif any(w in cond_names_lower for w in ["mixed", "repeated"]):
                _auto_design = "mixed"
        if not st.session_state.get("_design_type_manually_set"):
            st.session_state["builder_design_type"] = _auto_design

        design_options = {
            "between": "Between-subjects",
            "within": "Within-subjects",
            "mixed": "Mixed design",
        }
        design_type = st.radio(
            "Experimental design",
            options=list(design_options.keys()),
            format_func=lambda x: design_options[x],
            index=["between", "within", "mixed"].index(
                st.session_state.get("builder_design_type", "between")
            ),
            key="builder_design_type_input",
            help=(
                "Between = each participant sees one condition. "
                "Within = each participant sees all conditions. "
                "Mixed = combination of both."
            ),
        )
        st.session_state["builder_design_type"] = design_type
        if design_type != _auto_design:
            st.session_state["_design_type_manually_set"] = True

    # ── Demographics + Participants (collapsible) ─────────────────────
    st.markdown("")
    with st.expander("Demographics & Participants *(optional — defaults work well)*", expanded=False):
        _demo_col1, _demo_col2, _demo_col3 = st.columns(3)
        with _demo_col1:
            age_mean = st.number_input(
                "Mean age", min_value=18, max_value=80, value=35,
                key="builder_age_mean",
            )
        with _demo_col2:
            age_sd = st.number_input(
                "Age SD", min_value=1, max_value=30, value=12,
                key="builder_age_sd",
            )
        with _demo_col3:
            gender_pct = st.slider(
                "Male %", min_value=0, max_value=100, value=50,
                key="builder_gender_pct",
                help="Percentage of male participants (engine uses this as gender_quota)",
            )
        st.session_state["demographics_config"] = {
            "age_mean": age_mean,
            "age_sd": age_sd,
            "gender_quota": gender_pct,
        }

        participant_desc = st.text_area(
            "Expected participant type *(optional)*",
            placeholder="e.g., College students, MTurk workers, tech professionals...",
            height=60,
            key="builder_participant_desc",
            help="Helps calibrate behavioral personas. Leave blank for general population defaults.",
        )

    # ── Validation & Submission ─────────────────────────────────────────
    st.markdown("")
    st.markdown("#### Review & Build")

    # Build a preliminary ParsedDesign for validation
    _pre_factors = parser.detect_factorial_structure(parsed_conditions) if parsed_conditions else []
    _pre_design = ParsedDesign(
        conditions=parsed_conditions,
        scales=parsed_scales,
        open_ended=parsed_oe,
        factors=_pre_factors,
        design_type=design_type,
        sample_size=builder_sample,
    )
    validation = parser.validate_full_design(_pre_design)
    errors = validation["errors"]
    warnings = validation["warnings"]

    if not parsed_oe:
        warnings.append("No open-ended questions detected (this is fine if your study doesn't have any)")

    # Show summary — compact metrics + detail
    if parsed_conditions and parsed_scales:
        # v1.0.2.3: Compact inline metrics (consistent styling)
        st.markdown(
            f'<div style="display:flex;gap:20px;font-size:0.85rem;color:#6B7280;margin-bottom:4px;">'
            f'<span><strong style="color:#374151;">{len(parsed_conditions)}</strong> conditions</span>'
            f'<span><strong style="color:#374151;">{len(parsed_scales)}</strong> scales</span>'
            f'<span><strong style="color:#374151;">{len(parsed_oe)}</strong> open-ended</span>'
            f'<span><strong style="color:#374151;">{builder_sample}</strong> participants</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Detect domain
        title = st.session_state.get("study_title", "")
        desc = st.session_state.get("study_description", "")
        _cond_str = " ".join(c.name for c in parsed_conditions)
        _scale_str = " ".join(s.name for s in parsed_scales)
        domain = parser.detect_research_domain(
            title, desc,
            conditions_text=_cond_str,
            scales_text=_scale_str,
        )

        # Recommended analysis
        if len(parsed_conditions) == 2 and len(parsed_scales) >= 1:
            _rec_analysis = "Independent samples t-test"
        elif len(parsed_conditions) > 2 and not any(" \u00d7 " in c.name for c in parsed_conditions):
            _rec_analysis = "One-way ANOVA"
        elif any(" \u00d7 " in c.name for c in parsed_conditions):
            n_factors = len(parsed_conditions[0].name.split(" \u00d7 "))
            _rec_analysis = f"{n_factors}-way factorial ANOVA"
        else:
            _rec_analysis = ""

        _summary_parts = [f"**Design:** {design_options[design_type]}", f"**Domain:** {domain}"]
        if _rec_analysis:
            _summary_parts.append(f"**Suggested analysis:** {_rec_analysis}")
        st.caption(" · ".join(_summary_parts))

        with st.expander("Full design details", expanded=False):
            st.markdown(f"**Conditions:** {', '.join(c.name for c in parsed_conditions)}")
            for s in parsed_scales:
                st.caption(f"- {s.name}: {s.num_items} item(s), {s.scale_min}-{s.scale_max} ({s.scale_type})")
            if parsed_oe:
                for q in parsed_oe:
                    _qt = q.question_text
                    st.caption(f"- {_qt[:80]}{'...' if len(_qt) > 80 else ''}")

    for w in warnings:
        st.warning(w)
    for e in errors:
        st.error(e)

    # Guidance when there are warnings but no blocking errors
    if not errors and warnings:
        st.info("There are warnings but no errors -- you can proceed if the warnings are acceptable.")

    # Submit button
    can_submit = len(errors) == 0 and len(parsed_conditions) >= 2 and len(parsed_scales) >= 1
    if st.button(
        "Build Study Specification",
        type="primary",
        disabled=not can_submit,
        key="builder_submit_btn",
    ):
        title = st.session_state.get("study_title", "")
        desc = st.session_state.get("study_description", "")

        # Auto-generate description if user left it empty
        if not desc.strip() and title.strip():
            desc = parser.generate_smart_description(
                title, parsed_conditions, parsed_scales, design_type
            )
            st.session_state["study_description"] = desc

        _raw_inputs = {
            "conditions_text": st.session_state.get("builder_conditions_text", ""),
            "scales_text": st.session_state.get("builder_scales_text", ""),
            "open_ended_text": st.session_state.get("builder_oe_text", ""),
            "study_title": title,
            "study_description": desc,
            "participant_desc": participant_desc,
            "design_type": design_type,
            "sample_size": builder_sample,
        }
        if _finalize_builder_design(
            parser, parsed_conditions, parsed_scales, parsed_oe,
            design_type, builder_sample, title, desc,
            participant_desc=participant_desc,
            raw_inputs=_raw_inputs,
        ):
            st.success("Study specification built successfully! Use the buttons at the top to proceed to **Design**.")
            _navigate_to(2)


def _render_builder_design_review() -> None:
    """
    Render a design review/edit interface for studies created with the conversational builder.
    This replaces the QSF-based design configuration in Step 3.
    """
    inferred = st.session_state.get("inferred_design", {})
    if not inferred:
        st.warning("No study design found. Go back to **Study Input** and describe your study.")
        return

    # ── Go back to edit button (Issue #19) ─────────────────────────────
    if st.button("← Edit my study description", key="builder_go_back_edit"):
        st.session_state["conversational_builder_complete"] = False
        # Clear stale design data so user must re-submit
        st.session_state.pop("inferred_design", None)
        st.session_state.pop("builder_effect_sizes", None)
        st.session_state.pop("builder_parsed_design", None)
        _navigate_to(1)

    st.markdown("### Review Your Study Design")

    # ── Quick design summary metrics ──────────────────────────────────
    _summary_conditions = inferred.get("conditions", [])
    _summary_scales = inferred.get("scales", [])
    _summary_oe = inferred.get("open_ended_questions", [])
    _n_conds = len(_summary_conditions) if isinstance(_summary_conditions, list) else 0
    _n_scales = len(_summary_scales) if isinstance(_summary_scales, list) else 0
    _n_oe = len(_summary_oe) if isinstance(_summary_oe, list) else 0
    _n_sample = st.session_state.get("sample_size", st.session_state.get("builder_sample_size", 100))

    # v1.0.2.3: Compact inline metrics (consistent styling)
    st.markdown(
        f'<div style="display:flex;gap:20px;font-size:0.85rem;color:#6B7280;margin-bottom:4px;">'
        f'<span><strong style="color:#374151;">{_n_conds}</strong> conditions</span>'
        f'<span><strong style="color:#374151;">{_n_scales}</strong> scales/DVs</span>'
        f'<span><strong style="color:#374151;">{_n_oe}</strong> open-ended</span>'
        f'<span><strong style="color:#374151;">{_n_sample}</strong> sample size</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # v1.0.5.1: Use a placeholder for the design completeness message so it shows
    # CURRENT values (conditions/scales may be added/removed below on this page).
    _design_status_placeholder = st.container()

    # Show design improvement suggestions if available
    _builder_feedback = st.session_state.get("_builder_feedback", [])
    if _builder_feedback:
        with st.expander("Design Improvement Suggestions", expanded=True):
            for _fb in _builder_feedback:
                st.info(_fb)
        # Clear after display so it doesn't reappear on every rerun
        st.session_state.pop("_builder_feedback", None)

    # ── Detected Research Domain (Iteration 2: dropdown) ──────────────
    detected_domain = inferred.get("study_context", {}).get("domain", "")

    # Build options list: put detected domain first if valid
    _domain_options = list(AVAILABLE_DOMAINS)
    if detected_domain and detected_domain.lower().strip() not in [d.lower() for d in _domain_options]:
        _domain_options.insert(0, detected_domain)  # Keep custom domain if not in standard list

    _default_idx = 0
    for _di, _dn in enumerate(_domain_options):
        if _dn.lower().strip() == detected_domain.lower().strip():
            _default_idx = _di
            break

    domain_override = st.selectbox(
        "Research Domain",
        options=_domain_options,
        index=_default_idx,
        key="builder_review_domain",
        help=(
            "The research domain determines which behavioral personas are included "
            "in your simulation. Choose the domain that best matches your study."
        ),
    )
    if domain_override.strip():
        if "study_context" not in inferred:
            inferred["study_context"] = {}
        inferred["study_context"]["domain"] = domain_override.strip()
        inferred["study_context"]["study_domain"] = domain_override.strip()
        st.session_state["inferred_design"] = inferred

    # ── Persona Preview (Iteration 1) ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### Expected Participant Personas")
    st.caption(
        "Based on your research domain, these behavioral archetypes will be included "
        "in the simulation. Each persona has unique response patterns grounded in "
        "published behavioral science research."
    )

    # Get persona preview based on domain
    try:
        _persona_domains = SurveyDescriptionParser.get_persona_domain_keys(
            domain_override.strip() if domain_override.strip() else detected_domain
        )
        _temp_library = PersonaLibrary(seed=42)
        _preview_personas = _temp_library.get_personas_for_domains(_persona_domains)

        if _preview_personas:
            # Separate response style (universal) from domain-specific
            _response_style = {k: v for k, v in _preview_personas.items() if v.category == "response_style"}
            _domain_specific = {k: v for k, v in _preview_personas.items() if v.category != "response_style"}

            col_rs, col_ds = st.columns(2)
            with col_rs:
                st.markdown("**Universal Response Styles**")
                for _pname, _persona in sorted(_response_style.items(), key=lambda x: x[1].weight, reverse=True):
                    _pct = _persona.weight * 100
                    st.markdown(f"- **{_persona.name}** ({_pct:.0f}%)")
            with col_ds:
                st.markdown("**Domain-Specific Personas**")
                if _domain_specific:
                    for _pname, _persona in sorted(_domain_specific.items(), key=lambda x: x[1].weight, reverse=True)[:8]:
                        _pct = _persona.weight * 100
                        st.markdown(f"- **{_persona.name}** ({_pct:.0f}%)")
                    if len(_domain_specific) > 8:
                        st.caption(f"... and {len(_domain_specific) - 8} more")
                else:
                    st.caption("No domain-specific personas for this domain (universal styles will be used)")

            st.caption(
                f"Total: **{len(_preview_personas)}** personas active for "
                f"domain '{domain_override.strip() if domain_override.strip() else detected_domain}'"
            )

            # Store for potential custom weights UI
            st.session_state["_preview_persona_names"] = {
                k: {"name": v.name, "weight": v.weight, "category": v.category, "description": v.description}
                for k, v in _preview_personas.items()
            }
        else:
            st.info("No domain-specific personas found. Universal response styles will be used.")
    except Exception as _pe:
        st.caption(f"Persona preview unavailable: {_pe}")

    # ── Custom Persona Weights (Iteration 4) ──────────────────────────
    _preview_info = st.session_state.get("_preview_persona_names", {})
    if _preview_info:
        with st.expander("Advanced: Customize Persona Distribution", expanded=False):
            st.caption(
                "Override the default persona proportions to simulate specific "
                "sample compositions (e.g., more engaged responders, fewer satisficers)."
            )

            _custom_enabled = st.checkbox(
                "Use custom persona weights",
                value=bool(st.session_state.get("custom_persona_weights")),
                key="enable_custom_persona_weights",
            )

            if _custom_enabled:
                _custom_weights = {}
                # Group by category
                _rs_personas = {k: v for k, v in _preview_info.items() if v["category"] == "response_style"}
                _ds_personas = {k: v for k, v in _preview_info.items() if v["category"] != "response_style"}

                st.markdown("**Response Style Personas**")
                _rs_cols = st.columns(min(len(_rs_personas), 3)) if _rs_personas else []
                for _ci, (_pk, _pv) in enumerate(_rs_personas.items()):
                    with _rs_cols[_ci % len(_rs_cols)] if _rs_cols else st.container():
                        _w = st.number_input(
                            _pv["name"],
                            min_value=0,
                            max_value=100,
                            value=int(float(_pv.get("weight", 0)) * 100),
                            step=5,
                            key=f"pw_{_pk}",
                            help=_pv.get("description", "")[:200],
                        )
                        _custom_weights[_pk] = _w / 100.0

                if _ds_personas:
                    st.markdown("**Domain-Specific Personas**")
                    _ds_cols = st.columns(min(len(_ds_personas), 3))
                    for _ci, (_pk, _pv) in enumerate(list(_ds_personas.items())[:9]):
                        with _ds_cols[_ci % len(_ds_cols)]:
                            _w = st.number_input(
                                _pv["name"],
                                min_value=0,
                                max_value=100,
                                value=int(float(_pv.get("weight", 0)) * 100),
                                step=5,
                                key=f"pw_ds_{_pk}",
                                help=_pv.get("description", "")[:200],
                            )
                            _custom_weights[_pk] = _w / 100.0

                _total_w = sum(_custom_weights.values())
                if abs(_total_w - 1.0) > 0.05 and _total_w > 0:
                    st.warning(f"Weights sum to {_total_w*100:.0f}% (will be normalized to 100%)")
                elif _total_w <= 0:
                    st.error("At least one persona must have weight > 0")

                # Normalize and store
                if _total_w > 0:
                    _normalized = {k: v / _total_w for k, v in _custom_weights.items()}
                    st.session_state["custom_persona_weights"] = _normalized
            else:
                st.session_state["custom_persona_weights"] = None

    conditions = inferred.get("conditions", [])
    scales = inferred.get("scales", [])
    open_ended = inferred.get("open_ended_questions", [])
    factors = inferred.get("factors", [])

    # ── Sample Size (moved BEFORE factorial table to fix NameError) ────
    st.markdown("---")
    st.markdown("#### Sample Size")
    sample = st.number_input(
        "Total participants",
        min_value=10,
        max_value=10000,
        value=int(st.session_state.get("sample_size", 100)),
        step=10,
        key="builder_review_sample",
    )
    st.session_state["sample_size"] = sample

    # Per-cell size guidance
    if conditions and len(conditions) >= 2:
        _per_cell_n = sample // len(conditions)
        if _per_cell_n < 20:
            st.warning(
                f"With {sample} participants across {len(conditions)} conditions, "
                f"each cell gets ~{_per_cell_n} participants. "
                f"Consider at least {len(conditions) * 20} for adequate statistical power."
            )
        else:
            st.caption(f"~{_per_cell_n} participants per condition")

    # ── Conditions ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Conditions")

    # v1.0.5.1: Initialize condition descriptions from builder structured input or empty
    if "builder_condition_descriptions" not in st.session_state:
        st.session_state["builder_condition_descriptions"] = {}
    _cond_descs = st.session_state["builder_condition_descriptions"]

    # v1.0.5.4: Load stored is_control flags from structured input
    _cond_controls: dict[str, bool] = st.session_state.get("builder_condition_controls", {})

    if conditions:
        # Build a lookup: condition name → factor-level decomposition for factorial designs
        _cond_factor_map: dict[str, list[str]] = {}
        if factors and len(factors) >= 2:
            for _cond_name in conditions:
                _parts: list[str] = []
                for _fac in factors:
                    _fac_name = _fac.get("name", "Factor")
                    for _lev in _fac.get("levels", []):
                        if _lev.lower() in _cond_name.lower():
                            _parts.append(f"{_fac_name}: {_lev}")
                            break
                if _parts:
                    _cond_factor_map[_cond_name] = _parts

        cond_to_remove = None
        _br_cond_ver = st.session_state.get("_br_cond_version", 0)
        for i, cond in enumerate(conditions):
            # v1.0.5.4: Condition card — name + badge on top, description below, remove button on right
            _cond_card_cols = st.columns([8.5, 0.5])
            with _cond_card_cols[0]:
                _factor_info = _cond_factor_map.get(cond, [])
                # v1.0.5.4: Use stored is_control flag (from structured input), fall back to keyword matching
                _is_ctrl = _cond_controls.get(cond, any(w in cond.lower() for w in ('control', 'baseline', 'placebo', 'no treatment', 'neutral')))
                _type_badge = "Control" if _is_ctrl else "Treatment"
                _badge_color = "#16A34A" if _type_badge == "Control" else "#2563EB"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<strong>{i+1}. {cond}</strong>'
                    f'<span style="background:{_badge_color};color:white;font-size:0.7em;'
                    f'padding:2px 8px;border-radius:10px;">{_type_badge}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _factor_info:
                    st.caption(" | ".join(_factor_info))
                # v1.0.5.4: Show full description below name (not in narrow column)
                _existing_desc = _cond_descs.get(cond, "")
                _new_desc = st.text_input(
                    "Description",
                    value=_existing_desc,
                    key=f"br_cond_desc_v{_br_cond_ver}_{i}",
                    placeholder="What do participants experience?",
                    label_visibility="collapsed",
                )
                if _new_desc != _existing_desc:
                    _cond_descs[cond] = _new_desc
                    st.session_state["builder_condition_descriptions"] = _cond_descs
            with _cond_card_cols[1]:
                # v1.0.5.1: Allow removing ANY condition
                if st.button("X", key=f"br_remove_cond_{i}", help=f"Remove '{cond}'"):
                    cond_to_remove = i

        if cond_to_remove is not None:
            _removed_cond = conditions.pop(cond_to_remove)
            # Clean up description for removed condition
            _cond_descs.pop(_removed_cond, None)
            st.session_state["builder_condition_descriptions"] = _cond_descs
            inferred["conditions"] = conditions
            inferred["factors"] = _infer_factors_from_conditions(conditions)
            st.session_state["inferred_design"] = inferred
            st.session_state["selected_conditions"] = conditions
            st.session_state["builder_effect_sizes"] = []
            _n = max(len(conditions), 1)
            _samp = int(st.session_state.get("sample_size", 100))
            _per = _samp // _n
            _rem = _samp % _n
            st.session_state["condition_allocation_n"] = {
                c: _per + (1 if idx < _rem else 0) for idx, c in enumerate(conditions)
            }
            st.session_state["condition_allocation"] = {c: round(100.0 / _n, 1) for c in conditions}
            st.session_state["_br_cond_version"] = _br_cond_ver + 1
            st.rerun()

        # v1.0.5.1: Show guidance about descriptions
        _n_with_desc = sum(1 for c in conditions if _cond_descs.get(c, "").strip())
        if _n_with_desc < len(conditions):
            st.caption(
                f"{_n_with_desc}/{len(conditions)} conditions have descriptions. "
                "Adding descriptions helps generate more realistic simulated data."
            )
        else:
            st.caption("All conditions have descriptions.")
    else:
        st.warning("No conditions defined. Add conditions below to proceed.")

    # Allow adding custom conditions — with name and optional description
    # v1.0.6.1: Use _br_cond_version in widget keys so fields reset after adding
    _br_add_ver = st.session_state.get("_br_cond_version", 0)
    with st.expander("Add condition", expanded=not bool(conditions) or len(conditions) < 2):
        _add_c1, _add_c2 = st.columns([1, 2])
        with _add_c1:
            extra_cond_name = st.text_input(
                "Condition name",
                key=f"builder_review_extra_conds_v{_br_add_ver}",
                placeholder="e.g., Democrat identity visible",
                help="Enter the name for the new condition.",
            )
        with _add_c2:
            extra_cond_desc = st.text_area(
                "What do participants experience? (recommended)",
                key=f"builder_review_extra_desc_v{_br_add_ver}",
                placeholder="e.g., Participants see a profile with visible Democrat campaign pin, then play an economic sharing game.",
                height=68,
                help="Describe the manipulation. This helps the simulator generate scientifically realistic behavioral patterns.",
            )
        _can_add = bool(extra_cond_name.strip())
        if st.button("Add condition", key=f"builder_add_cond_btn_v{_br_add_ver}", disabled=not _can_add, type="primary"):
            _nc = extra_cond_name.strip()
            if _nc in conditions:
                st.warning(f"Condition '{_nc}' already exists.")
            elif " × " in _nc or " x " in _nc.lower():
                st.warning("Cannot add factorial conditions manually. Use text mode on the Study Input page.")
            elif len(_nc) > 200:
                st.warning("Condition name is too long (max 200 characters).")
            else:
                conditions.append(_nc)
                # Save description
                if extra_cond_desc.strip():
                    _cond_descs[_nc] = extra_cond_desc.strip()
                    st.session_state["builder_condition_descriptions"] = _cond_descs
                inferred["conditions"] = conditions
                inferred["factors"] = _infer_factors_from_conditions(conditions)
                st.session_state["inferred_design"] = inferred
                st.session_state["selected_conditions"] = conditions
                st.session_state["builder_effect_sizes"] = []
                _n = max(len(conditions), 1)
                _samp = int(st.session_state.get("sample_size", 100))
                _per = _samp // _n
                _rem = _samp % _n
                st.session_state["condition_allocation_n"] = {
                    c: _per + (1 if idx < _rem else 0) for idx, c in enumerate(conditions)
                }
                st.session_state["condition_allocation"] = {c: round(100.0 / _n, 1) for c in conditions}
                # Increment version to create fresh widgets on next rerun
                st.session_state["_br_cond_version"] = _br_add_ver + 1
                st.rerun()

    # ── Factorial Structure ─────────────────────────────────────────────
    if factors:
        st.markdown("---")
        st.markdown("#### Factorial Structure")
        for f in factors:
            st.markdown(f"**{f.get('name', 'Factor')}**: {', '.join(f.get('levels', []))}")

        # Visual design table for 2-factor designs
        if len(factors) == 2:
            row_factor = factors[0]
            col_factor = factors[1]
            row_levels = row_factor.get("levels", [])
            col_levels = col_factor.get("levels", [])
            n_cells = len(row_levels) * len(col_levels)
            per_cell = sample // n_cells if n_cells > 0 else 0

            header = f"| | " + " | ".join(f"**{lv}**" for lv in col_levels) + " |"
            separator = "|--" + "|".join("---" for _ in col_levels) + "|"
            table_rows = [header, separator]
            cell_num = 1
            for rl in row_levels:
                cells = []
                for _ in col_levels:
                    cells.append(f"Cell {cell_num} (n={per_cell})")
                    cell_num += 1
                row_str = f"| **{rl}** | " + " | ".join(cells) + " |"
                table_rows.append(row_str)

            st.markdown(
                f"**Design Table** ({row_factor['name']} x {col_factor['name']}):"
            )
            st.markdown("\n".join(table_rows))

        elif len(factors) >= 3:
            # For 3+ factors, show a numbered list of all crossed conditions
            import itertools as _iter3
            n_cells = 1
            for f in factors:
                n_cells *= len(f.get("levels", []))
            per_cell = sample // n_cells if n_cells > 0 else 0
            dims_str = "×".join(str(len(f.get("levels", []))) for f in factors)
            st.success(f"**{dims_str} Factorial Design = {n_cells} conditions** (~{per_cell} per cell)")
            # Show as compact table
            all_combos = list(_iter3.product(*[f["levels"] for f in factors]))
            table_header = "| # | " + " | ".join(f"**{f['name']}**" for f in factors) + " |"
            table_sep = "|--" + "|".join("---" for _ in factors) + "|"
            rows = [table_header, table_sep]
            for idx, combo in enumerate(all_combos, 1):
                row = f"| {idx} | " + " | ".join(combo) + " |"
                rows.append(row)
            st.markdown("\n".join(rows))

    # ── Scales / DVs ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Dependent Variables / Scales")
    # v1.0.1.5: Version counter for builder scale widget keys to prevent stale values after removal
    _br_scale_ver = st.session_state.get("_br_scale_version", 0)
    if scales:
        scale_to_remove = None
        for i, scale in enumerate(scales):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 0.5])
            with col1:
                new_name = st.text_input(
                    "Name", value=scale.get("name", ""), key=f"br_scale_name_v{_br_scale_ver}_{i}"
                )
                if new_name.strip():
                    scale["name"] = new_name.strip()
                    # Regenerate variable name from updated name
                    _vn = re.sub(r'[^a-zA-Z0-9\s]', ' ', new_name)
                    _vn = re.sub(r'\s+', ' ', _vn).strip()
                    # Remove orphaned numbers (digits not part of a known abbreviation)
                    _vn = re.sub(r'\b\d+\b', '', _vn).strip()
                    _vn = re.sub(r'\s+', '_', _vn)[:30]
                    scale["variable_name"] = _vn or "Scale"
                else:
                    st.caption("Name cannot be empty")
            with col2:
                new_items = st.number_input(
                    "Items", value=scale.get("num_items", 1),
                    min_value=1, max_value=50, key=f"br_scale_items_v{_br_scale_ver}_{i}"
                )
                scale["num_items"] = new_items
                scale["items"] = [
                    f"{scale.get('variable_name', 'var')}_{j+1}" for j in range(new_items)
                ]
            with col3:
                new_min = st.number_input(
                    "Min", value=scale.get("scale_min", 1),
                    min_value=-1000, max_value=100, key=f"br_scale_min_v{_br_scale_ver}_{i}"
                )
                scale["scale_min"] = new_min
            with col4:
                # Ensure max > min (allow 0-1 for binary scales)
                _min_for_max = new_min + 1
                _cur_max = scale.get("scale_max", 7)
                _safe_max = max(_cur_max, _min_for_max)
                new_max = st.number_input(
                    "Max", value=_safe_max,
                    min_value=_min_for_max, max_value=1000, key=f"br_scale_max_v{_br_scale_ver}_{i}"
                )
                scale["scale_max"] = new_max
                # scale_points: discrete count for likert/binary, raw max for numeric/slider
                _stype = scale.get("type", "likert")
                if _stype in ("numeric", "slider"):
                    scale["scale_points"] = new_max
                else:
                    scale["scale_points"] = new_max - new_min + 1
            with col5:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove", key=f"br_remove_scale_v{_br_scale_ver}_{i}", help=f"Remove '{scale.get('name', 'scale')}'"):
                    scale_to_remove = i

        if scale_to_remove is not None:
            scales.pop(scale_to_remove)
            inferred["scales"] = scales
            st.session_state["inferred_design"] = inferred
            st.session_state["confirmed_scales"] = scales
            # v1.0.1.5: Increment version + rerun to avoid scroll-to-top and stale widget keys
            st.session_state["_br_scale_version"] = st.session_state.get("_br_scale_version", 0) + 1
            st.rerun()

        inferred["scales"] = scales
        st.session_state["inferred_design"] = inferred
        st.session_state["confirmed_scales"] = scales

        # Domain-specific scale suggestions
        _review_domain = (domain_override.strip() if domain_override.strip() else detected_domain)
        if _review_domain and scales:
            _existing_parsed = [
                ParsedScale(name=s.get("name", ""), scale_type=s.get("type", "likert"))
                for s in scales
            ]
            _suggestions = SurveyDescriptionParser().suggest_additional_measures(
                _review_domain, _existing_parsed
            )
            if _suggestions:
                with st.expander("Suggested additional measures for your domain"):
                    for _sug in _suggestions:
                        _s_col1, _s_col2 = st.columns([5, 1])
                        with _s_col1:
                            st.markdown(f"**{_sug['name']}** — {_sug['description']}")
                            st.caption(f"Why: {_sug['why']}")
                        with _s_col2:
                            if st.button("Add", key=f"br_add_sug_{_sug['name'][:15]}"):
                                # Add suggested scale to the design
                                _sug_lower = _sug['name'].lower().strip()
                                _known = KNOWN_SCALES.get(_sug_lower, {})
                                _new_scale = {
                                    "name": _sug['name'],
                                    "variable_name": re.sub(r'[^a-zA-Z0-9]', '_', _sug['name'])[:30],
                                    "num_items": _known.get("items", 4),
                                    "scale_min": _known.get("min", 1),
                                    "scale_max": _known.get("max", 7),
                                    "scale_points": _known.get("max", 7) - _known.get("min", 1) + 1,
                                    "type": "likert",
                                    "reverse_items": [],
                                    "reliability": 0.85,
                                    "detected_from_qsf": False,
                                    "description": _sug['description'],
                                    "items": [
                                        f"{re.sub(r'[^a-zA-Z0-9]', '_', _sug['name'])[:20]}_{j+1}"
                                        for j in range(_known.get("items", 4))
                                    ],
                                }
                                scales.append(_new_scale)
                                inferred["scales"] = scales
                                st.session_state["inferred_design"] = inferred
                                st.session_state["confirmed_scales"] = scales
                                # v1.0.1.5: Increment version + rerun to avoid scroll-to-top
                                st.session_state["_br_scale_version"] = st.session_state.get("_br_scale_version", 0) + 1
                                st.rerun()
    else:
        st.warning("No scales detected")

    # ── Open-Ended Questions ────────────────────────────────────────────
    # v1.0.1.4: Version counter for builder OE widget keys to prevent stale values after removal
    _br_oe_ver = st.session_state.get("_br_oe_version", 0)
    if open_ended:
        st.markdown("---")
        st.markdown("#### Open-Ended Questions")
        # v1.0.1.2: For word-based experiments, context is MANDATORY (not optional)
        st.markdown(
            '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 14px;margin:6px 0 12px 0;">'
            '<span style="color:#1E40AF;font-weight:600;">Context is required for word-based experiments</span><br>'
            '<span style="color:#1E40AF;font-size:0.9em;">'
            'Since there is no QSF file, the AI relies entirely on your context descriptions to understand '
            'what each question is asking. For each question, write 1-2 sentences explaining: '
            '(1) what the participant just experienced, (2) what kind of response is expected, and '
            '(3) how it relates to your experimental conditions. '
            'This is the single most important factor for realistic AI-generated responses that '
            'reflect different participant personas.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        oe_to_remove = None
        _br_oe_missing_ctx = []
        for i, oe in enumerate(open_ended):
            oe_col1, oe_col2 = st.columns([8, 1])
            with oe_col1:
                new_text = st.text_input(
                    f"Question {i+1} (`{oe.get('variable_name', '')}`)",
                    value=oe.get("question_text", ""),
                    key=f"br_oe_text_v{_br_oe_ver}_{i}",
                )
                if new_text and new_text != oe.get("question_text", ""):
                    oe["question_text"] = new_text
                    # Regenerate variable name from updated text
                    _stop = {'the', 'a', 'an', 'of', 'for', 'in', 'on', 'to', 'by',
                             'your', 'you', 'this', 'that', 'what', 'how', 'why',
                             'did', 'please', 'would', 'about', 'any', 'with', 'and'}
                    _words = [w.lower() for w in re.findall(r'[a-zA-Z]+', new_text)
                              if len(w) > 2 and w.lower() not in _stop]
                    oe["variable_name"] = '_'.join(_words[:3])[:30] if _words else f"OE_{i+1}"
                # v1.0.1.2: Context is mandatory for builder path — enhanced UI
                _existing_ctx = oe.get("question_context", "")
                _br_ctx_placeholder = "e.g., 'Participants explain their feelings toward Donald Trump after reading a polarizing article'"
                _q_txt = oe.get("question_text", "").strip()
                _study_d = (st.session_state.get("study_description") or "")[:80]
                if _q_txt and len(_q_txt) > 10:
                    _br_ctx_placeholder = f"e.g., 'Participants respond to: {_q_txt[:50]}'"
                    if _study_d:
                        _br_ctx_placeholder += f" (study about {_study_d[:30]}...)"
                new_ctx = st.text_input(
                    f"Context for Q{i+1} (required)",
                    value=_existing_ctx,
                    key=f"br_oe_ctx_v{_br_oe_ver}_{i}",
                    placeholder=_br_ctx_placeholder,
                    help=(
                        "REQUIRED: 1-2 sentences explaining what this question is really asking. "
                        "Include: (1) what the participant experienced, (2) what kind of response "
                        "is expected, (3) how it connects to conditions. This context shapes how "
                        "each simulated persona (engaged, satisficer, etc.) responds."
                    ),
                )
                if new_ctx != _existing_ctx:
                    oe["question_context"] = new_ctx
                # v1.0.1.2: Visual feedback for missing context
                if not new_ctx.strip():
                    st.markdown(
                        '<span style="color:#DC2626;font-size:0.85em;">Context is required for this question.</span>',
                        unsafe_allow_html=True,
                    )
                    _br_oe_missing_ctx.append(i + 1)
                else:
                    _ctx_w = len(new_ctx.strip().split())
                    if _ctx_w < 3:
                        st.caption(f"{_ctx_w} words — consider adding more detail.")
                    elif _ctx_w <= 30:
                        st.caption(f"{_ctx_w} words")
                    else:
                        st.caption(f"{_ctx_w} words — concise is best; consider trimming to 1-2 sentences.")
            with oe_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove", key=f"br_remove_oe_v{_br_oe_ver}_{i}", help=f"Remove question {i+1}"):
                    oe_to_remove = i
        # v1.0.1.2: Summary of missing context with blocking warning
        if _br_oe_missing_ctx:
            _q_nums = ", ".join(f"Q{n}" for n in _br_oe_missing_ctx)
            st.warning(
                f"Missing context for: {_q_nums}. All open-ended questions require context "
                f"when describing your study in words. Add context above to proceed."
            )
            st.session_state["_builder_oe_context_complete"] = False
        else:
            if open_ended:
                st.markdown(
                    '<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:8px 12px;margin:6px 0;">'
                    '<span style="color:#166534;font-weight:600;">All open-ended questions have context</span>'
                    ' &mdash; AI personas will generate well-tailored responses.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            st.session_state["_builder_oe_context_complete"] = True
        if oe_to_remove is not None:
            open_ended.pop(oe_to_remove)
            inferred["open_ended_questions"] = open_ended
            st.session_state["inferred_design"] = inferred
            st.session_state["confirmed_open_ended"] = open_ended
            # v1.0.1.4: Increment version counter and rerun to refresh widget keys
            st.session_state["_br_oe_version"] = _br_oe_ver + 1
            st.rerun()

        # v1.0.1.4: "Remove All" button for builder open-ended questions
        if open_ended and len(open_ended) > 1:
            if st.button(
                f"Remove All Open-Ended Questions ({len(open_ended)})",
                key=f"br_remove_all_oe_v{_br_oe_ver}",
                help="Remove all open-ended questions at once",
            ):
                inferred["open_ended_questions"] = []
                st.session_state["inferred_design"] = inferred
                st.session_state["confirmed_open_ended"] = []
                st.session_state["_builder_oe_context_complete"] = True
                st.session_state["_br_oe_version"] = _br_oe_ver + 1
                st.rerun()

        # Persist edits
        inferred["open_ended_questions"] = open_ended
        st.session_state["inferred_design"] = inferred
        st.session_state["confirmed_open_ended"] = open_ended

    # ── Condition Allocation ────────────────────────────────────────────
    if conditions and len(conditions) >= 2:
        st.markdown("---")
        st.markdown("#### Condition Allocation")

        n_conds = len(conditions)
        sample_size = int(sample)

        # Track changes to conditions/sample to know when to recalculate
        _br_prev_sample = st.session_state.get("_br_prev_sample_size", 0)
        _br_prev_n_conds = st.session_state.get("_br_prev_n_conditions", 0)
        _br_prev_conds = st.session_state.get("_br_prev_conditions", [])

        _br_conds_changed = (
            n_conds != _br_prev_n_conds or
            set(_br_prev_conds) != set(conditions)
        )
        _br_sample_changed = _br_prev_sample != sample_size and _br_prev_sample > 0

        _br_needs_recalc = (
            "condition_allocation_n" not in st.session_state or
            _br_conds_changed
        )

        if _br_needs_recalc:
            # Fresh allocation: equal distribution
            n_per = sample_size // n_conds
            remainder = sample_size % n_conds
            st.session_state["condition_allocation_n"] = {
                cond: n_per + (1 if i < remainder else 0)
                for i, cond in enumerate(conditions)
            }
            _br_alloc_ver = st.session_state.get("_br_alloc_version", 0) + 1
            st.session_state["_br_alloc_version"] = _br_alloc_ver
        elif _br_sample_changed:
            # Proportionally adjust existing allocations to match new total
            old_alloc = st.session_state.get("condition_allocation_n", {})
            old_total = sum(old_alloc.values())
            if old_total > 0 and sample_size > 0:
                scale_factor = sample_size / old_total
                new_alloc = {}
                running_total = 0
                sorted_conds = list(conditions)
                for j, c in enumerate(sorted_conds[:-1]):
                    new_alloc[c] = max(1, round(old_alloc.get(c, sample_size // n_conds) * scale_factor))
                    running_total += new_alloc[c]
                new_alloc[sorted_conds[-1]] = max(1, sample_size - running_total)
                st.session_state["condition_allocation_n"] = new_alloc
                _br_alloc_ver = st.session_state.get("_br_alloc_version", 0) + 1
                st.session_state["_br_alloc_version"] = _br_alloc_ver

        # Update tracking variables
        st.session_state["_br_prev_sample_size"] = sample_size
        st.session_state["_br_prev_n_conditions"] = n_conds
        st.session_state["_br_prev_conditions"] = list(conditions)

        allocation_n = st.session_state["condition_allocation_n"]
        _br_alloc_ver = st.session_state.get("_br_alloc_version", 0)

        # Render number input widgets per condition
        st.markdown("**Participants per condition:**")
        cols_per_row = min(n_conds, 3)
        input_cols = st.columns(cols_per_row)

        new_allocation_n = {}
        for i, cond in enumerate(conditions):
            col_idx = i % cols_per_row
            with input_cols[col_idx]:
                display_name = _clean_condition_name(cond)
                full_name = display_name  # Keep full name for tooltip
                # v1.0.5.1: Wider truncation (35 chars) + full name in help tooltip
                if len(display_name) > 37:
                    display_name = display_name[:35] + "..."
                current_n = allocation_n.get(cond, sample_size // n_conds)
                current_n = min(max(1, current_n), sample_size)

                new_n = st.number_input(
                    display_name,
                    min_value=1,
                    max_value=sample_size,
                    value=current_n,
                    step=1,
                    key=f"br_alloc_n_v{_br_alloc_ver}_{i}",
                    help=f"Participants in: {full_name}"
                )
                new_allocation_n[cond] = new_n

        # Validate totals
        total_n = sum(new_allocation_n.values())
        diff = total_n - sample_size

        _br_btn_col1, _br_btn_col2 = st.columns(2)
        if diff != 0:
            st.warning(f"Allocations sum to **{total_n}** (should be {sample_size}). Difference: {'+' if diff > 0 else ''}{diff}")
            with _br_btn_col1:
                if st.button("Auto-balance to match total", key=f"br_auto_balance_v{_br_alloc_ver}"):
                    _scale_f = sample_size / total_n if total_n > 0 else 1
                    normalized = {}
                    _running = 0
                    _sorted = list(conditions)
                    for j, c in enumerate(_sorted[:-1]):
                        normalized[c] = max(1, round(new_allocation_n[c] * _scale_f))
                        _running += normalized[c]
                    normalized[_sorted[-1]] = max(1, sample_size - _running)
                    st.session_state["condition_allocation_n"] = normalized
                    st.session_state["_br_alloc_version"] = _br_alloc_ver + 1
                    st.rerun()
            with _br_btn_col2:
                if st.button("Reset to equal", key=f"br_equal_reset_v{_br_alloc_ver}"):
                    _per = sample_size // n_conds
                    _rem = sample_size % n_conds
                    st.session_state["condition_allocation_n"] = {
                        c: _per + (1 if idx < _rem else 0) for idx, c in enumerate(conditions)
                    }
                    st.session_state["_br_alloc_version"] = _br_alloc_ver + 1
                    st.rerun()
        else:
            st.success(f"Allocations sum to {sample_size}")
            st.session_state["condition_allocation_n"] = new_allocation_n
            _min_cell_n = min(new_allocation_n.values()) if new_allocation_n else 0
            if _min_cell_n < 20:
                st.caption(f"Smallest cell has {_min_cell_n} participants. Consider at least 20 per cell for adequate statistical power.")
            with _br_btn_col2:
                if st.button("Reset to equal", key=f"br_equal_reset_ok_v{_br_alloc_ver}"):
                    _per = sample_size // n_conds
                    _rem = sample_size % n_conds
                    st.session_state["condition_allocation_n"] = {
                        c: _per + (1 if idx < _rem else 0) for idx, c in enumerate(conditions)
                    }
                    st.session_state["_br_alloc_version"] = _br_alloc_ver + 1
                    st.rerun()

        # Convert to percentage-based allocation for the simulation engine
        st.session_state["condition_allocation"] = {
            cond: (new_allocation_n.get(cond, 0) / sample_size * 100) if sample_size > 0 else 0
            for cond in conditions
        }

    # ── Difficulty Level ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Data Quality / Difficulty")
    difficulty = st.select_slider(
        "How 'messy' should the simulated data be?",
        options=["easy", "medium", "hard", "expert"],
        value=st.session_state.get("builder_difficulty_level", st.session_state.get("difficulty_level", "medium")),
        key="builder_review_difficulty",
        help="Easy = clean data with high attention. Hard = more noise, careless respondents, missing data.",
    )
    st.session_state["builder_difficulty_level"] = difficulty

    # ── Effect Sizes (Optional) ─────────────────────────────────────────
    if conditions and scales and len(conditions) >= 2:
        st.markdown("---")
        st.markdown("##### Effect Sizes")
        st.caption(
            "Effect sizes control how different conditions are from each other. "
            "Common benchmarks: **Small** (d=0.2), **Medium** (d=0.5), **Large** (d=0.8). "
            "Most behavioral experiments find medium effects."
        )
        with st.expander("Configure Expected Effect Sizes", expanded=False):
            st.caption(
                "Specify expected differences between conditions. "
                "This makes the simulated data reflect realistic experimental effects."
            )
            add_effect = st.checkbox("Add an expected effect", key="builder_add_effect")
            if add_effect:
                effect_dv = st.selectbox(
                    "Dependent variable",
                    [s.get("name", "Unknown") for s in scales],
                    key="builder_effect_dv",
                )
                col_hi, col_lo = st.columns(2)
                with col_hi:
                    level_high = st.selectbox(
                        "Higher-scoring condition",
                        conditions,
                        key="builder_effect_high",
                    )
                with col_lo:
                    remaining = [c for c in conditions if c != level_high]
                    level_low = st.selectbox(
                        "Lower-scoring condition",
                        remaining if remaining else conditions,
                        key="builder_effect_low",
                    )
                cohens_d = st.slider(
                    "Cohen's d (effect size)",
                    min_value=0.0, max_value=1.5, value=0.5, step=0.1,
                    key="builder_cohens_d",
                    help="0.2 = small, 0.5 = medium, 0.8 = large effect",
                )
                # Store the effect size spec
                st.session_state["builder_effect_sizes"] = [
                    {
                        "variable": effect_dv,
                        "factor": "condition",
                        "level_high": level_high,
                        "level_low": level_low,
                        "cohens_d": cohens_d,
                        "direction": "positive",
                    }
                ]
                st.caption(
                    f"Effect: **{effect_dv}** will be ~{cohens_d}d higher in "
                    f"'{level_high}' than '{level_low}'"
                )
            else:
                # Clear stale effect sizes when checkbox is unchecked
                st.session_state["builder_effect_sizes"] = []

    # v1.0.5.1: Fill the design status placeholder with CURRENT values
    # (all conditions/scales/OE edits above have been applied by this point)
    with _design_status_placeholder:
        _final_issues: list[str] = []
        _final_n_conds = len(conditions) if conditions else 0
        _final_n_scales = len(scales) if scales else 0
        _final_n_oe = len(open_ended) if open_ended else 0
        if _final_n_conds < 2:
            _final_issues.append(f"At least 2 conditions needed (currently {_final_n_conds})")
        if _final_n_scales < 1:
            _final_issues.append("At least 1 scale/DV needed")
        if _final_n_oe > 0 and not st.session_state.get("_builder_oe_context_complete", False):
            _final_issues.append("Open-ended questions need context (see below)")
        if _final_issues:
            st.warning("Design incomplete: " + "; ".join(_final_issues))
        else:
            st.success("Design complete! Use the buttons at the top to proceed to **Generate**.")

    # ── Confirmation ────────────────────────────────────────────────────
    st.markdown("---")
    if conditions and scales and len(conditions) >= 2:
        st.info(
            f"Design ready: **{len(conditions)}** conditions, **{len(scales)}** scale(s), "
            f"**{sample}** participants. Use the **Continue** button at the top to proceed."
        )
    elif not conditions or len(conditions) < 2:
        st.error(f"Need at least 2 conditions (currently {len(conditions) if conditions else 0}). Add conditions above.")
    elif not scales:
        st.error("Need at least 1 scale/DV. Go back to Study Input and add scales.")

    # ── Proceed to Generate guidance ───────────────────────────────────
    if conditions and scales:
        # What happens next (Iteration 8)
        with st.expander("What happens when you generate?", expanded=False):
            _oe_line = f"4. Generate **{len(open_ended)}** open-ended text response(s)\n" if open_ended else ""
            _fx_line = "5. Apply effect sizes to create condition differences\n" if st.session_state.get("builder_effect_sizes") else ""
            st.markdown(
                f"**Your simulation will:**\n"
                f"1. Create **{sample}** virtual participants with realistic behavioral profiles\n"
                f"2. Assign each to one of **{len(conditions)}** conditions\n"
                f"3. Generate responses to **{len(scales)}** scale(s) based on persona traits\n"
                f"{_oe_line}"
                f"{_fx_line}\n"
                f"**Personas** will be selected based on the '{domain_override}' research domain, "
                f"producing realistic variation in attention, engagement, response styles, and more."
            )

        st.markdown("---")
        st.markdown("#### Ready to generate?")
        st.markdown(
            "Click **Generate** above to simulate your data. "
            "You can always come back here to adjust your design."
        )


def _get_step_completion() -> Dict[str, bool]:
    """Get completion status for each step.

    v1.8.9: Accounts for both QSF-upload and conversational-builder paths.
    """
    preview = st.session_state.get("qsf_preview", None)

    # Check variable roles for primary outcome and independent variable
    variable_rows = st.session_state.get("variable_review_rows", [])
    has_primary_outcome = any(r.get("Role") == "Primary outcome" for r in variable_rows)
    has_independent_var = any(r.get("Role") == "Condition" for r in variable_rows)

    # v1.8.9: Builder path may not have variable_review_rows — check confirmed_scales instead
    is_builder = bool(st.session_state.get("conversational_builder_complete"))
    confirmed_scales = st.session_state.get("confirmed_scales", [])
    if is_builder and not has_primary_outcome and confirmed_scales:
        has_primary_outcome = True  # Builder always creates scales as primary outcomes
    if is_builder and not has_independent_var:
        inferred = st.session_state.get("inferred_design", {})
        has_independent_var = bool(inferred.get("conditions"))

    # v1.8.9: Also consider conditions from inferred_design (covers builder path)
    _conditions_set = bool(
        st.session_state.get("selected_conditions")
        or st.session_state.get("custom_conditions")
        or (is_builder and st.session_state.get("inferred_design", {}).get("conditions"))
    )

    # v1.8.9: Check DVs are configured (confirmed_scales present)
    _dvs_configured = bool(confirmed_scales) or bool(
        st.session_state.get("inferred_design", {}).get("scales")
    )

    return {
        "study_title": bool((st.session_state.get("study_title") or st.session_state.get("_p_study_title", "")).strip()),
        "study_description": bool((st.session_state.get("study_description") or st.session_state.get("_p_study_description", "")).strip()),
        "sample_size": int(st.session_state.get("sample_size", 0)) >= 10,
        "qsf_uploaded": bool(preview and preview.success) or is_builder,
        "conditions_set": _conditions_set,
        "primary_outcome": has_primary_outcome,
        "independent_var": has_independent_var,
        "design_ready": bool(st.session_state.get("inferred_design")),
        "dvs_configured": _dvs_configured,
    }


def _preflight_validation() -> List[str]:
    """v1.4.14: Pre-generation validation. Returns list of error messages.

    Catches issues that would crash the simulation engine BEFORE starting
    the expensive generation process. Returns an empty list if all checks pass.
    """
    errors = []
    inferred = st.session_state.get("inferred_design", {})
    if not inferred:
        errors.append("No experiment design configured. Complete the Design page first.")
        return errors

    # Check conditions
    conditions = inferred.get("conditions", [])
    if not conditions:
        errors.append("No conditions defined. Go to Design to add conditions.")

    # Check scales
    scales = inferred.get("scales", [])
    if not scales:
        errors.append("No scales/DVs defined. Go to Design to add dependent variables.")
    for s in scales:
        s_min = s.get("scale_min", 1)
        s_max = s.get("scale_max", 7)
        if isinstance(s_min, (int, float)) and isinstance(s_max, (int, float)):
            if s_min >= s_max:
                errors.append(f"Scale '{s.get('name', '?')}' has min ({s_min}) >= max ({s_max}). Fix in Design.")

    # Check sample size
    n = st.session_state.get("sample_size", 0)
    if not isinstance(n, (int, float)) or n < 10:
        errors.append(f"Sample size ({n}) is too small. Minimum is 10.")

    # Check allocation sums
    alloc_n = st.session_state.get("condition_allocation_n", {})
    if alloc_n and conditions:
        total = sum(alloc_n.values())
        if total != n:
            errors.append(
                f"Condition allocation sums to {total} but sample size is {n}. "
                "Go to Design and click 'Auto-balance'."
            )
        # v1.8.9: Warn about dangerously small cell sizes
        min_cell = min(alloc_n.values()) if alloc_n else 0
        if min_cell < 5 and n >= 10:
            errors.append(
                f"Smallest condition has only {min_cell} participants. "
                "Minimum recommended is 5 per cell for basic analyses."
            )

    # v1.8.9: Check for confirmed scales consistency
    confirmed = st.session_state.get("confirmed_scales", [])
    for s in confirmed:
        s_min = s.get("scale_min", 1)
        s_max = s.get("scale_max", 7)
        if isinstance(s_min, (int, float)) and isinstance(s_max, (int, float)):
            if s_min >= s_max:
                errors.append(f"Scale '{s.get('name', '?')}' has min ({s_min}) >= max ({s_max}). Fix in Design.")

    return errors


# v1.4.14: _save_step_state() and _restore_step_state() REMOVED.
# Page-based rendering keeps all state in st.session_state directly.
# No snapshot/restore cycle needed since only one page renders at a time.



def _reset_generation_state() -> None:
    """Reset all generation-related state to allow a fresh generation run.

    v1.4.16: Extracted from duplicate inline blocks to ensure consistent cleanup.
    """
    st.session_state["is_generating"] = False
    st.session_state["has_generated"] = False
    st.session_state["generation_requested"] = False
    st.session_state["_generation_phase"] = 0
    st.session_state["last_df"] = None
    st.session_state["last_zip"] = None
    st.session_state["last_metadata"] = None
    st.session_state["generated_metadata"] = None
    st.session_state["_quality_checks"] = []
    st.session_state["_validation_results"] = None
    st.session_state["_llm_connectivity_status"] = None  # Re-check LLM on next run
    st.session_state.pop("preview_df", None)


def _navigate_to(page_index: int) -> None:
    """Navigate to a section by index and rerun.

    v1.7.0: Persists widget values before rerun so they survive when
    their widgets are not rendered on the target page.  Uses pending-nav
    pattern for reliable section switching.
    """
    # v1.7.0: Persist widget values that would be lost when their page
    # is not rendered (Streamlit removes unrendered widget keys).
    _widget_persist_keys = [
        "study_title", "study_description", "team_name", "team_members_raw",
    ]
    for _wk in _widget_persist_keys:
        _wv = st.session_state.get(_wk)
        if _wv is not None:
            st.session_state[f"_p_{_wk}"] = _wv

    clamped = max(-1, min(page_index, 3))
    st.session_state["_pending_nav"] = clamped
    st.session_state["_page_just_changed"] = True
    st.session_state["_force_scroll_top"] = True
    if clamped == 2:
        st.session_state["_page_just_changed_design"] = True
    st.rerun()


# ── Section metadata for flow navigation ──────────────────────────────
SECTION_META: List[Dict[str, str]] = [
    {"title": "Setup", "icon": "📋", "desc": "Study details & team"},
    {"title": "Study Input", "icon": "📁", "desc": "Upload QSF or describe study"},
    {"title": "Design", "icon": "🔬", "desc": "Conditions, factors & outcomes"},
    {"title": "Generate", "icon": "⚡", "desc": "Run simulation & download"},
]


def _section_summary(idx: int) -> str:
    """One-line summary for a completed section."""
    if idx == 0:
        return "Complete" if (st.session_state.get("study_title") or st.session_state.get("_p_study_title")) else ""
    elif idx == 1:
        if st.session_state.get("conversational_builder_complete"):
            return "Complete"
        if st.session_state.get("qsf_preview"):
            return "Complete"
        return ""
    elif idx == 2:
        conds = st.session_state.get("selected_conditions", [])
        _n_scales = len(st.session_state.get("confirmed_scales", []))
        if conds and _n_scales:
            return "Complete"
        elif conds:
            return f"{len(conds)} conditions"
        return ""
    elif idx == 3:
        return "Complete" if st.session_state.get("has_generated") else ""
    return ""


# ── Flow navigation CSS (v1.7.0 — Professional minimal design) ───────
_FLOW_NAV_CSS = """<style>
/* === v1.8.0: Premium landing page + segmented progress === */

/* v1.0.3.5: Hidden-button infrastructure removed. Navigation via visible Next buttons. */

/* Base layout */
section.main .block-container {
    max-width: 960px;
    padding-top: 1.2rem;
}
section.main h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; color: #111827; }
section.main h2 { font-size: 1.35rem; font-weight: 600; color: #1F2937; }
section.main h3 { font-size: 1.1rem; font-weight: 600; color: #1F2937; }
section.main h4 {
    font-size: 1rem; font-weight: 600; margin-bottom: 0.6rem; color: #111827;
    padding-bottom: 8px; border-bottom: 2px solid #F3F4F6;
    margin-top: 1.2rem;
}
section[data-testid="stSidebar"] .stCaption { line-height: 1.4; }

/* v1.0.3.2: Cleaner expander styling */
details[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #374151 !important;
}
details[data-testid="stExpander"] {
    border-radius: 10px !important;
    border-color: #E5E7EB !important;
    background: white !important;
}
details[data-testid="stExpander"][open] {
    border-color: #D1D5DB !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}

/* ─── Landing page hero ─── */
.landing-hero {
    text-align: center;
    padding: 64px 20px 40px;
    max-width: 720px;
    margin: 0 auto;
    animation: heroFadeIn 0.6s ease-out;
}
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.landing-hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.035em;
    margin: 0 0 18px 0;
    line-height: 1.08;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #334155 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.landing-hero .subtitle {
    font-size: 1.2rem;
    color: #6B7280;
    font-weight: 400;
    margin: 0 0 8px 0;
    line-height: 1.6;
}
.landing-hero .creator {
    font-size: 0.85rem;
    color: #9CA3AF;
    margin: 0 0 36px 0;
}
.landing-hero .creator a {
    color: #6B7280;
    text-decoration: underline;
    text-decoration-color: #D1D5DB;
    text-underline-offset: 3px;
    transition: color 0.15s ease;
}
.landing-hero .creator a:hover { color: #FF4B4B; }

/* ─── Feature cards ─── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    max-width: 780px;
    margin: 0 auto 40px;
    padding: 0 20px;
    animation: cardsSlideUp 0.6s ease-out 0.15s both;
}
@keyframes cardsSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feature-card {
    padding: 22px 20px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF4B4B, #FF8585);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.feature-card:hover {
    border-color: #D1D5DB;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transform: translateY(-2px);
}
.feature-card:hover::before { opacity: 1; }
.feature-card .fc-icon {
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    font-size: 1.25rem;
    margin-bottom: 12px;
}
.feature-card h4 {
    font-size: 0.95rem;
    font-weight: 650;
    color: #111827;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.feature-card p {
    font-size: 0.82rem;
    color: #6B7280;
    margin: 0;
    line-height: 1.55;
}

/* ─── Trust signals strip ─── */
.trust-strip {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    max-width: 680px;
    margin: 8px auto 0;
    padding: 16px 24px;
    background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.trust-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    padding: 0 16px;
}
.trust-num {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.trust-label {
    font-size: 0.7rem;
    color: #64748B;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 2px;
}
.trust-divider {
    width: 1px;
    height: 32px;
    background: #CBD5E1;
    flex-shrink: 0;
}

/* ─── How it works ─── */
.how-it-works {
    max-width: 780px;
    margin: 0 auto 40px;
    padding: 0 20px;
    text-align: center;
    animation: cardsSlideUp 0.6s ease-out 0.3s both;
}
.how-it-works h3 {
    font-size: 0.78rem;
    font-weight: 600;
    color: #374151;
    margin: 0 0 24px 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.hiw-steps {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0;
}
.hiw-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    max-width: 160px;
    padding: 0 8px;
}
.hiw-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #F9FAFB, #F3F4F6);
    color: #6B7280;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700;
    margin-bottom: 10px;
    border: 1.5px solid #E5E7EB;
    transition: all 0.2s ease;
}
.hiw-step:hover .hiw-num {
    background: #FF4B4B;
    color: white;
    border-color: #FF4B4B;
}
.hiw-step .hiw-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
    margin: 0 0 3px 0;
}
.hiw-step .hiw-desc {
    font-size: 0.75rem;
    color: #9CA3AF;
    line-height: 1.4;
}
.hiw-arrow {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 10px;
    flex-shrink: 0;
    font-weight: 300;
}

/* ─── Research list (inside expander) ─── */
.research-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: 640px;
    margin: 0 auto;
    text-align: left;
}
.research-item {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px 10px;
    padding: 10px 14px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    text-decoration: none;
    transition: all 0.15s ease;
    cursor: pointer;
}
.research-item:hover {
    border-color: #D1D5DB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transform: translateY(-1px);
}
.ri-authors {
    font-size: 0.82rem;
    font-weight: 600;
    color: #1F2937;
}
.ri-venue {
    font-size: 0.72rem;
    color: #9CA3AF;
    font-style: italic;
}
.ri-insight {
    font-size: 0.78rem;
    color: #6B7280;
    width: 100%;
    line-height: 1.4;
}

/* ─── Landing footer ─── */
.landing-footer {
    text-align: center;
    padding: 32px 0 12px;
    border-top: 1px solid #F1F5F9;
    max-width: 400px;
    margin: 24px auto 0;
}
.footer-institution {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
}
.footer-version {
    font-size: 0.7rem;
    color: #CBD5E1;
}

/* ─── Landing info tabs ─── */
.landing-tabs-container {
    max-width: 780px;
    margin: 32px auto 0;
    padding: 0 20px;
}
.landing-tab-content {
    padding: 24px 0 16px;
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.65;
}
.landing-tab-content h4 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9CA3AF;
    margin: 0 0 16px;
    font-weight: 600;
}
.use-case-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 12px;
}
.use-case-card {
    padding: 16px;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background: #FAFAFA;
    transition: all 0.15s ease;
}
.use-case-card:hover {
    border-color: #D1D5DB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.use-case-card strong {
    display: block;
    font-size: 0.85rem;
    color: #1F2937;
    margin-bottom: 4px;
}
.use-case-card span {
    font-size: 0.8rem;
    color: #6B7280;
    line-height: 1.5;
}
.capability-item {
    display: flex;
    gap: 14px;
    padding: 16px 0;
    border-bottom: 1px solid #F3F4F6;
}
.capability-item:last-child { border-bottom: none; }
.cap-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #FEF2F2, #FFF1F2);
    border-radius: 10px;
}
.cap-text strong {
    display: block;
    font-size: 0.88rem;
    color: #1F2937;
    margin-bottom: 3px;
}
.cap-text span {
    font-size: 0.82rem;
    color: #6B7280;
    line-height: 1.55;
}
.step-detail-item {
    display: flex;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #F3F4F6;
}
.step-detail-item:last-child { border-bottom: none; }
.step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #FF4B4B, #FF6B6B);
    color: white;
    font-weight: 700;
    font-size: 0.78rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-detail-text strong {
    display: block;
    font-size: 0.85rem;
    color: #1F2937;
    margin-bottom: 2px;
}
.step-detail-text span {
    font-size: 0.8rem;
    color: #6B7280;
    line-height: 1.55;
}
@media (max-width: 768px) {
    .use-case-grid { grid-template-columns: 1fr; }
}

/* ─── v1.0.3.2: Redesigned stepper — large, readable, primary navigation ─── */
.stepper-nav {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 22px 16px 16px;
    position: relative;
    background: white;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stepper-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
    z-index: 1;
    max-width: 200px;
}
/* Connecting line between steps */
.stepper-step:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 24px;
    left: calc(50% + 26px);
    width: calc(100% - 52px);
    height: 3px;
    background: #E5E7EB;
    z-index: 0;
    border-radius: 2px;
    transition: background 0.3s ease;
}
.stepper-step.st-done:not(:last-child)::after {
    background: linear-gradient(90deg, #22C55E, #4ADE80);
}
.stepper-step.st-active:not(:last-child)::after {
    background: linear-gradient(90deg, #3B82F6 0%, #E5E7EB 100%);
}
.stepper-step.st-active.step-done:not(:last-child)::after {
    background: linear-gradient(90deg, #22C55E 0%, #F97316 100%);
}
/* Circle indicator — 48px for clear readability */
.stepper-circle {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 10px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    z-index: 2;
    cursor: default;
    border: 2.5px solid transparent;
}
/* Done state — green with checkmark */
.stepper-step.st-done .stepper-circle {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    color: white;
    border-color: #22C55E;
    box-shadow: 0 3px 12px rgba(34, 197, 94, 0.3);
}
/* Active state — blue with number */
.stepper-step.st-active .stepper-circle {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    border-color: #2563EB;
    box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.15);
    animation: stepPulse 2.5s ease-in-out infinite;
}
@keyframes stepPulse {
    0%, 100% { box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.15); }
    50% { box-shadow: 0 0 0 9px rgba(37, 99, 235, 0.07); }
}
/* Active + done — green circle with checkmark */
.stepper-step.st-active.step-done .stepper-circle {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    border-color: #22C55E;
    box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.2);
    animation: activeDonePulse 2.5s ease-in-out infinite;
}
.stepper-step.st-active.step-done .stepper-title { color: #15803D; }
.stepper-step.st-active.step-done .stepper-desc { color: #16A34A; }
@keyframes activeDonePulse {
    0%, 100% { box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.2); }
    50% { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0.08); }
}
/* Upcoming (reachable) — light gray */
.stepper-step.st-upcoming .stepper-circle {
    background: #F9FAFB;
    color: #6B7280;
    border: 2px solid #D1D5DB;
}
/* Next step — orange pulsing */
.stepper-step.st-next-target .stepper-circle {
    background: #FFF7ED;
    color: #EA580C;
    border-color: #F97316;
    box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.15);
    animation: nextPulse 2s ease-in-out infinite;
}
.stepper-step.st-next-target .stepper-title { color: #C2410C; font-weight: 700; }
.stepper-step.st-next-target .stepper-desc { color: #EA580C; font-weight: 500; }
@keyframes nextPulse {
    0%, 100% { box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.15); }
    50% { box-shadow: 0 0 0 8px rgba(249, 115, 22, 0.07); }
}
/* Locked — dimmed */
.stepper-step.st-locked { cursor: default; }
.stepper-step.st-locked .stepper-circle {
    background: #F9FAFB;
    color: #D1D5DB;
    border-color: #E5E7EB;
    cursor: default;
    opacity: 0.45;
}
/* v1.0.3.5: Stepper is visual-only — clickable hover styles removed */
/* Step title — readable, prominent */
.stepper-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #6B7280;
    text-align: center;
    transition: color 0.2s ease;
    line-height: 1.3;
    max-width: 150px;
    cursor: default;
    user-select: none;
}
.stepper-step.st-done .stepper-title { color: #15803D; font-weight: 700; }
.stepper-step.st-active .stepper-title { color: #1D4ED8; font-weight: 700; }
.stepper-step.st-locked .stepper-title { color: #D1D5DB; }
/* Step description under title */
.stepper-desc {
    font-size: 0.76rem;
    color: #9CA3AF;
    text-align: center;
    max-width: 150px;
    line-height: 1.35;
    margin-top: 3px;
}
.stepper-step.st-done .stepper-desc { color: #4ADE80; }
.stepper-step.st-active .stepper-desc { color: #60A5FA; }
.stepper-step.st-locked .stepper-desc { color: #E5E7EB; }

/* Responsive stepper */
@media (max-width: 600px) {
    .stepper-circle { width: 38px; height: 38px; font-size: 14px; }
    .stepper-title { font-size: 0.78rem; max-width: 85px; }
    .stepper-desc { display: none; }
    .stepper-step:not(:last-child)::after { top: 19px; left: calc(50% + 21px); width: calc(100% - 42px); }
    .stepper-nav { padding: 14px 6px 10px; }
}

/* v1.0.3.4: flow-section wrapper removed (empty div didn't wrap content).
   Card styling now applied to section-guide + Streamlit content container. */

/* v1.0.3.2: Section guide — clean step header with subtle left accent */
.section-guide {
    font-size: 0.92rem;
    color: #374151;
    margin: 0 0 20px 0;
    padding: 16px 22px;
    background: #F8FAFC;
    border-radius: 12px;
    border-left: 4px solid #3B82F6;
    line-height: 1.55;
}
.section-guide strong {
    color: #1E40AF;
    font-size: 0.92rem;
    font-weight: 700;
}

/* v1.0.3.7: Design readiness checklist — prominent, visual progress */
.design-checklist {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 16px 0;
    animation: bannerSlideIn 0.3s ease-out;
}
.design-checklist-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
.design-checklist-item.done {
    background: #F0FDF4;
    border: 1.5px solid #BBF7D0;
    color: #166534;
}
.design-checklist-item.pending {
    background: #FFF7ED;
    border: 1.5px solid #FED7AA;
    color: #9A3412;
}

/* v1.0.3.7: Confirmation action banners for DV/OE checkboxes */
.confirm-banner {
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 12px 0 4px 0;
}
.confirm-banner.pending {
    background: #FFFBEB;
    border-left: 4px solid #F59E0B;
    color: #92400E;
}
.confirm-banner.done {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    color: #166534;
}

/* v1.0.3.6: Back to top link styling */
.btt-link {
    font-size: 0.85rem;
    color: #6B7280;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s ease;
}
.btt-link:hover {
    color: #374151;
    text-decoration: underline;
}

/* v1.0.3.2: Section complete banner — clean success state */
.section-done-banner {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 22px;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #22C55E;
    border-radius: 12px;
    margin: 4px 0 20px 0;
    font-size: 0.88rem; color: #166534; font-weight: 500;
    animation: bannerSlideIn 0.3s ease-out;
}
@keyframes bannerSlideIn {
    from { opacity: 0; transform: translateY(-3px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* v1.0.3.2: Design page cards — polished container look */
.design-card {
    padding: 18px 20px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    margin: 10px 0;
    transition: all 0.2s ease;
}
.design-card:hover {
    border-color: #D1D5DB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.design-card-header {
    font-size: 0.92rem; font-weight: 600;
    color: #111827; margin: 0 0 10px 0;
}
.config-status {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; padding: 3px 10px;
    border-radius: 12px; font-weight: 500;
}
.config-status.ready {
    background: #F0FDF4; color: #166534;
    border: 1px solid #D1FAE5;
}
.config-status.needed {
    background: #FEF3C7; color: #92400E;
    border: 1px solid #FDE68A;
}

/* v1.0.3.2: Better form element spacing */
section.main .stTextInput > div, section.main .stTextArea > div,
section.main .stSelectbox > div, section.main .stMultiSelect > div,
section.main .stNumberInput > div {
    margin-bottom: 2px;
}
section.main .stTextInput label, section.main .stTextArea label,
section.main .stSelectbox label, section.main .stMultiSelect label,
section.main .stNumberInput label {
    font-weight: 500;
    color: #374151;
    font-size: 0.88rem;
}
/* Cleaner radio buttons */
section.main .stRadio > div {
    margin-bottom: 4px;
}
section.main .stRadio label {
    font-weight: 500;
    font-size: 0.88rem;
}
/* Better checkbox styling */
section.main .stCheckbox label span {
    font-size: 0.88rem;
    color: #374151;
}

/* v1.8.7.4: nav-btn-row removed (CSS selector never matched Streamlit DOM) */

/* Compact feedback link */
.feedback-bar {
    display: flex; align-items: center; justify-content: center;
    gap: 8px; padding: 10px 16px; margin-top: 28px;
    border-top: 1px solid #F3F4F6;
    font-size: 11px; color: #9CA3AF;
}
.feedback-bar a { color: #6B7280; text-decoration: none; }
.feedback-bar a:hover { color: #FF4B4B; }

/* v1.0.3.0: Back-to-top links removed — stepper handles navigation */

/* ─── v1.0.3.2: Polished button styles ─── */
.stButton button[kind="secondary"] {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    padding: 8px 18px !important;
    border-color: #D1D5DB !important;
}
.stButton button[kind="secondary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    border-color: #9CA3AF !important;
    background: #F9FAFB !important;
}
.stButton button[kind="primary"] {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    padding: 12px 28px !important;
    letter-spacing: 0.01em !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(255,75,75,0.25) !important;
}
/* Warning/info/success/error message polish */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .feature-grid { grid-template-columns: 1fr; }
    .hiw-steps { flex-direction: column; align-items: center; gap: 12px; }
    .hiw-arrow { transform: rotate(90deg); }
    .landing-hero h1 { font-size: 1.8rem; }
    .landing-hero { padding: 32px 16px 24px; }
    .research-section { padding: 20px 16px; }
}

/* v1.0.1.7: Removed pulsing animation — cleaner, more professional look */

/* v1.0.3.0: Confirmation checkpoint CSS removed — using simple checkboxes now */
</style>"""


# =====================================================================
# v1.0.4.7: ADMIN DASHBOARD — Hidden, password-protected diagnostics
# Access: Add ?admin=1 to the URL. Password required.
# Shows: LLM stats, simulation history, system diagnostics, session state.
# =====================================================================
_ADMIN_PASSWORD_HASH = "19465e8fc94da7f22aec392a5514a6494a3e090ce0ba3bd1773c1c9e339dcfac"  # SHA-256 of "Dimant_Admin"


# v1.0.6.3: File-based admin persistence so simulation history survives browser refresh
_ADMIN_PERSIST_DIR = Path(os.environ.get("ADMIN_DATA_DIR", "/tmp/.sim_admin"))
_ADMIN_PERSIST_FILE = _ADMIN_PERSIST_DIR / "sim_history.json"


def _save_admin_history() -> None:
    """Persist admin simulation history to disk."""
    try:
        _ADMIN_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _history = st.session_state.get("_admin_sim_history", [])
        _llm_stats = st.session_state.get("_last_llm_stats", {})
        _engine_log = st.session_state.get("_admin_engine_log", [])
        _data = {
            "sim_history": _history,
            "last_llm_stats": _llm_stats,
            "engine_log": _engine_log,
            "total_exhaustions": st.session_state.get("_admin_total_exhaustions", 0),
            "dialog_shown": st.session_state.get("_admin_llm_exhaust_dialog_shown", 0),
            "user_key_activations": st.session_state.get("_admin_user_key_activations", 0),
        }
        _ADMIN_PERSIST_FILE.write_text(json.dumps(_data, default=str))
    except Exception:
        pass  # File persistence is best-effort


def _load_admin_history() -> None:
    """Load persisted admin simulation history from disk into session state."""
    if st.session_state.get("_admin_history_loaded"):
        return  # Already loaded this session
    try:
        if _ADMIN_PERSIST_FILE.exists():
            _data = json.loads(_ADMIN_PERSIST_FILE.read_text())
            # Merge: session data takes priority, file fills in gaps
            if not st.session_state.get("_admin_sim_history"):
                st.session_state["_admin_sim_history"] = _data.get("sim_history", [])
            else:
                # Append persisted entries that aren't in current session
                _existing = st.session_state.get("_admin_sim_history", [])
                _persisted = _data.get("sim_history", [])
                _existing_timestamps = {e.get("timestamp") for e in _existing}
                for _pe in _persisted:
                    if _pe.get("timestamp") not in _existing_timestamps:
                        _existing.insert(0, _pe)  # Prepend older entries
                st.session_state["_admin_sim_history"] = _existing
            if not st.session_state.get("_last_llm_stats"):
                st.session_state["_last_llm_stats"] = _data.get("last_llm_stats", {})
            if not st.session_state.get("_admin_engine_log"):
                st.session_state["_admin_engine_log"] = _data.get("engine_log", [])
            # Restore cumulative counters
            _stored_exhaustions = _data.get("total_exhaustions", 0)
            _current_exhaustions = st.session_state.get("_admin_total_exhaustions", 0)
            st.session_state["_admin_total_exhaustions"] = max(_stored_exhaustions, _current_exhaustions)
    except Exception:
        pass  # File persistence is best-effort
    st.session_state["_admin_history_loaded"] = True


def _render_admin_dashboard() -> None:
    """Render the hidden admin dashboard with full diagnostic info."""
    import hashlib
    from datetime import datetime

    # v1.0.6.3: Load persisted history on dashboard render
    _load_admin_history()

    st.markdown(
        '<h1 style="text-align:center;">Admin Dashboard</h1>'
        '<p style="text-align:center;color:#6B7280;font-size:0.9rem;">'
        'Behavioral Data Simulation Tool — Internal Diagnostics</p>',
        unsafe_allow_html=True,
    )

    # ── Password Gate ─────────────────────────────────────────────────
    if not st.session_state.get("_admin_authenticated"):
        st.markdown("---")
        _pw_col1, _pw_col2, _pw_col3 = st.columns([1, 2, 1])
        with _pw_col2:
            st.markdown("### Authentication Required")
            _pw = st.text_input("Admin Password", type="password", key="_admin_pw_input")
            if st.button("Authenticate", type="primary", use_container_width=True, key="_admin_auth_btn"):
                _hash = hashlib.sha256(_pw.encode()).hexdigest()
                if _hash == _ADMIN_PASSWORD_HASH:
                    st.session_state["_admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid password.")
        return

    # ── Top metrics bar ───────────────────────────────────────────────
    # v1.0.5.7: Compute CUMULATIVE stats across all simulations, not just
    # the last one.  Also fall back to per-history-entry llm_stats if
    # _last_llm_stats is empty (handles page refresh after simulation).
    st.markdown("---")
    _m1, _m2, _m3, _m4 = st.columns(4)
    _sim_history = st.session_state.get("_admin_sim_history", [])
    _current_llm_stats = st.session_state.get("_last_llm_stats", {})

    # Cumulative stats from ALL simulation runs in this session
    _cum_calls = 0
    _cum_pool = 0
    _cum_fallbacks = 0
    _last_provider = "none"
    for _hist_entry in _sim_history:
        _hls = _hist_entry.get("llm_stats", {})
        _cum_calls += _hls.get("llm_calls", 0)
        _cum_pool += _hls.get("pool_size", 0)
        _cum_fallbacks += _hls.get("fallback_uses", 0)
        _hp = _hls.get("active_provider", "")
        if _hp:
            _last_provider = _hp
    # If no history entries had stats, fall back to _last_llm_stats
    if _cum_calls == 0 and _cum_pool == 0 and _current_llm_stats:
        _cum_calls = _current_llm_stats.get("llm_calls", 0)
        _cum_pool = _current_llm_stats.get("pool_size", 0)
        _cum_fallbacks = _current_llm_stats.get("fallback_uses", 0)
        _last_provider = _current_llm_stats.get("active_provider", "none")

    with _m1:
        st.metric("Total Simulations", len(_sim_history))
    with _m2:
        st.metric("LLM API Calls", _cum_calls)
    with _m3:
        st.metric("LLM Pool Size", _cum_pool)
    with _m4:
        _fb_pct = f"{(_cum_fallbacks / max(1, _cum_pool + _cum_fallbacks)) * 100:.0f}%" if (_cum_pool + _cum_fallbacks) > 0 else "N/A"
        st.metric("Template Fallback", _fb_pct)

    # ── Tabs ──────────────────────────────────────────────────────────
    _tab_llm, _tab_failures, _tab_history, _tab_session, _tab_system = st.tabs([
        "LLM Pipeline", "Failure Analytics", "Simulation History", "Session State", "System Info"
    ])

    # ── TAB 1: LLM Pipeline ──────────────────────────────────────────
    with _tab_llm:
        st.markdown("### LLM Provider Chain")
        if _cum_calls > 0 or _cum_pool > 0 or _current_llm_stats:
            st.markdown(f"**Active Provider:** `{_last_provider}`")
            st.markdown(f"**Total API Calls (cumulative):** {_cum_calls}")
            st.markdown(f"**Response Pool Size (cumulative):** {_cum_pool}")
            st.markdown(f"**Template Fallbacks (cumulative):** {_cum_fallbacks}")

            # Provider breakdown from most recent simulation
            _providers = _current_llm_stats.get("providers", {})
            if _providers:
                st.markdown("#### Provider Breakdown (last run)")
                _prov_data = []
                for _pname, _pinfo in _providers.items():
                    _prov_data.append({
                        "Provider": _pname,
                        "Calls": _pinfo.get("calls", 0),
                        "Available": "Yes" if _pinfo.get("available", False) else "No",
                    })
                if _prov_data:
                    st.dataframe(_prov_data, use_container_width=True, hide_index=True)

            # LLM generation quality
            if _cum_pool > 0:
                _total_gen = _cum_pool + _cum_fallbacks
                _llm_pct = (_cum_pool / _total_gen) * 100
                st.markdown("#### Generation Quality")
                st.progress(min(1.0, _llm_pct / 100))
                st.caption(f"{_llm_pct:.1f}% AI-generated, {100 - _llm_pct:.1f}% template fallback")
        else:
            st.info("No LLM statistics available yet. Run a simulation first.")

        # Engine initialization log
        st.markdown("#### Engine Log")
        _engine_log = st.session_state.get("_admin_engine_log", [])
        if _engine_log:
            for _entry in _engine_log[-20:]:
                st.text(_entry)
        else:
            st.caption("No engine log entries. Run a simulation to populate.")

    # ── TAB 2: Failure Analytics (v1.0.5.8) ───────────────────────────
    with _tab_failures:
        st.markdown("### LLM Provider Failure Analytics")
        st.markdown(
            "Tracks when built-in AI providers fail and users are prompted "
            "to provide their own API keys."
        )

        # Aggregate failure stats from all simulation runs
        _total_exhaustions = 0
        _total_user_key_activations = 0
        _total_fallback_uses_all = 0
        _total_pool_all = 0
        _all_exhaust_timestamps: list = []
        for _he in _sim_history:
            _hls = _he.get("llm_stats", {})
            _total_exhaustions += _hls.get("provider_exhaustions", 0)
            _total_user_key_activations += _hls.get("user_key_activations", 0)
            _total_fallback_uses_all += _hls.get("fallback_uses", 0)
            _total_pool_all += _hls.get("pool_size", 0)
            _all_exhaust_timestamps.extend(_hls.get("exhaustion_timestamps", []))

        # Also include session-level counters from UI interactions
        _dialog_shown = st.session_state.get("_admin_llm_exhaust_dialog_shown", 0)
        _ui_key_activations = st.session_state.get("_admin_user_key_activations", 0)
        _session_total_exhaustions = st.session_state.get("_admin_total_exhaustions", 0)

        # Top metrics
        _fa1, _fa2, _fa3, _fa4 = st.columns(4)
        with _fa1:
            st.metric("Provider Exhaustions", max(_total_exhaustions, _session_total_exhaustions))
        with _fa2:
            st.metric("User Key Activations", max(_total_user_key_activations, _ui_key_activations))
        with _fa3:
            st.metric("Fallback Dialog Shown", _dialog_shown)
        with _fa4:
            if (_total_pool_all + _total_fallback_uses_all) > 0:
                _exhaust_rate = (_total_fallback_uses_all / (_total_pool_all + _total_fallback_uses_all)) * 100
                st.metric("Template Fallback Rate", f"{_exhaust_rate:.1f}%")
            else:
                st.metric("Template Fallback Rate", "N/A")

        # Per-run exhaustion breakdown
        st.markdown("#### Per-Simulation Breakdown")
        _exhaust_data = []
        for _idx, _he in enumerate(_sim_history, 1):
            _hls = _he.get("llm_stats", {})
            _ex = _hls.get("provider_exhaustions", 0)
            _fb = _hls.get("fallback_uses", 0)
            _pool = _hls.get("pool_size", 0)
            _uk = _hls.get("user_key_activations", 0)
            _exhaust_data.append({
                "Run": f"#{_idx}",
                "Title": _he.get("title", "Untitled")[:30],
                "Exhaustions": _ex,
                "Fallbacks": _fb,
                "AI Generated": _pool,
                "User Key Used": "Yes" if _uk > 0 else "No",
                "Fallback %": f"{(_fb / max(1, _pool + _fb)) * 100:.0f}%" if (_pool + _fb) > 0 else "N/A",
            })
        if _exhaust_data:
            st.dataframe(_exhaust_data, use_container_width=True, hide_index=True)
        else:
            st.info("No simulation runs recorded yet. Run a simulation to see failure analytics.")

        # Exhaustion timeline
        if _all_exhaust_timestamps:
            st.markdown("#### Exhaustion Event Timeline")
            _timeline_data = []
            for _ts in sorted(_all_exhaust_timestamps):
                try:
                    _dt = datetime.fromtimestamp(_ts)
                    _timeline_data.append({"Time": _dt.strftime("%H:%M:%S"), "Event": "All providers exhausted"})
                except Exception:
                    pass
            if _timeline_data:
                st.dataframe(_timeline_data[-20:], use_container_width=True, hide_index=True)
                st.caption(f"Showing last {min(20, len(_timeline_data))} of {len(_timeline_data)} events")

    # ── TAB 3: Simulation History ─────────────────────────────────────
    with _tab_history:
        st.markdown("### Simulation Run History")
        if _sim_history:
            for _idx, _run in enumerate(reversed(_sim_history), 1):
                with st.expander(
                    f"Run #{len(_sim_history) - _idx + 1} — "
                    f"{_run.get('title', 'Untitled')} "
                    f"({_run.get('timestamp', 'unknown')})",
                    expanded=(_idx == 1),
                ):
                    _rc1, _rc2, _rc3 = st.columns(3)
                    with _rc1:
                        st.metric("Sample Size", _run.get("sample_size", "?"))
                    with _rc2:
                        st.metric("Conditions", _run.get("n_conditions", "?"))
                    with _rc3:
                        st.metric("Scales", _run.get("n_scales", "?"))

                    _run_llm = _run.get("llm_stats", {})
                    if _run_llm:
                        st.markdown("**LLM Stats:**")
                        st.json(_run_llm)

                    _run_conds = _run.get("conditions", [])
                    if _run_conds:
                        st.markdown(f"**Conditions:** {', '.join(_run_conds)}")

                    _run_domain = _run.get("detected_domains", [])
                    if _run_domain:
                        st.markdown(f"**Detected Domains:** {', '.join(_run_domain)}")

                    _run_personas = _run.get("personas_used", [])
                    if _run_personas:
                        st.markdown(f"**Personas Used:** {', '.join(_run_personas[:10])}")
                        if len(_run_personas) > 10:
                            st.caption(f"... and {len(_run_personas) - 10} more")
        else:
            st.info("No simulation history found.")
            st.caption("Run a simulation to see entries here. History persists across browser refreshes.")

    # ── TAB 4: Session State Explorer ─────────────────────────────────
    with _tab_session:
        st.markdown("### Session State Explorer")
        _filter = st.text_input("Filter keys (contains):", key="_admin_state_filter")
        _all_keys = sorted(st.session_state.keys())
        if _filter:
            _all_keys = [k for k in _all_keys if _filter.lower() in k.lower()]

        st.caption(f"Showing {len(_all_keys)} keys")
        _state_data = []
        for _k in _all_keys:
            _v = st.session_state.get(_k)
            _type = type(_v).__name__
            _val_str = str(_v)
            if len(_val_str) > 200:
                _val_str = _val_str[:200] + "..."
            _state_data.append({"Key": _k, "Type": _type, "Value": _val_str})
        if _state_data:
            st.dataframe(_state_data, use_container_width=True, hide_index=True, height=400)

    # ── TAB 5: System Info ────────────────────────────────────────────
    with _tab_system:
        st.markdown("### System Information")
        _sys_col1, _sys_col2 = st.columns(2)
        with _sys_col1:
            st.markdown(f"**App Version:** `{APP_VERSION}`")
            st.markdown(f"**Required Utils Version:** `{REQUIRED_UTILS_VERSION}`")
            st.markdown(f"**Build ID:** `{BUILD_ID}`")
            try:
                _actual_utils_ver = utils.__version__ if hasattr(utils, '__version__') else "unknown"
                st.markdown(f"**Actual Utils Version:** `{_actual_utils_ver}`")
                _match = _actual_utils_ver == REQUIRED_UTILS_VERSION
                if _match:
                    st.success("Utils version matches")
                else:
                    st.warning(f"Utils version MISMATCH: expected {REQUIRED_UTILS_VERSION}")
            except Exception:
                st.markdown("**Actual Utils Version:** `unknown`")
        with _sys_col2:
            import platform
            st.markdown(f"**Python:** `{platform.python_version()}`")
            st.markdown(f"**Platform:** `{platform.platform()}`")
            try:
                import streamlit as _st_ver
                st.markdown(f"**Streamlit:** `{_st_ver.__version__}`")
            except Exception:
                pass
            st.markdown(f"**Timestamp:** `{datetime.now().isoformat()}`")

        # LLM provider chain info
        st.markdown("### LLM Provider Chain Configuration")
        try:
            from utils.llm_response_generator import LLMResponseGenerator
            _temp_gen = LLMResponseGenerator(seed=42)
            _prov_info = []
            for _p in _temp_gen._providers:
                _key_preview = (_p.api_key[:6] + "..." + _p.api_key[-4:]) if _p.api_key and len(_p.api_key) > 12 else "(none)"
                _prov_info.append({
                    "Provider": _p.name,
                    "Model": _p.model,
                    "API Key": _key_preview,
                    "Available": "Yes" if _p.available else "No",
                    "Rate Limit": f"{_p.max_rpm} RPM" if hasattr(_p, 'max_rpm') else "N/A",
                })
            if _prov_info:
                st.dataframe(_prov_info, use_container_width=True, hide_index=True)
                st.caption(f"API Available: {'Yes' if _temp_gen.is_llm_available else 'No'}")
            else:
                st.caption("No providers configured")
        except Exception as _pe:
            st.error(f"Could not load provider info: {_pe}")
            st.caption("Check that utils/llm_response_generator.py imports correctly.")

    # ── Logout ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Logout", key="_admin_logout_btn"):
        st.session_state["_admin_authenticated"] = False
        st.rerun()


def _render_flow_nav(active: int, done: List[bool]) -> None:
    """Render clickable stepper progress bar for direct step navigation.

    v1.0.2.5: Clickable stepper — click any completed/accessible step to jump.
    Hidden Streamlit buttons + JS wiring via _st_components.html().
    - Completed steps: green circle + checkmark (CLICKABLE)
    - Active step: blue circle + number with subtle pulse (current page)
    - Active+done: green circle + checkmark (current page, done)
    - Next-target step: orange pulsing circle (CLICKABLE)
    - Upcoming steps: gray outline (CLICKABLE if accessible)
    - Locked steps: dimmed with tooltip (not clickable)
    Includes #btt-anchor for "Back to top" links at the bottom of pages.
    """
    while len(done) < len(SECTION_META):
        done.append(False)

    # Anchor target for "Back to top" HTML links at the bottom of pages.
    # Pure client-side scrollIntoView() — no st.rerun() needed.
    st.markdown('<div id="btt-anchor"></div>', unsafe_allow_html=True)

    # Determine state for each step:
    # - done: completed (i < active and done[i], OR done[i] is True for any visited step)
    # - active: currently viewing
    # - next-target: the immediate next step when current step is done (pulsing CTA)
    # - upcoming: reachable (all prior steps done, or step is active+1)
    # - locked: not yet reachable
    _active_done = done[active] if 0 <= active < len(done) else False
    step_states: List[str] = []
    for i in range(len(SECTION_META)):
        if done[i] and i != active:
            step_states.append("done")
        elif i == active:
            step_states.append("active")
        elif i <= active:
            # Past step that isn't done — treat as upcoming (revisitable)
            step_states.append("upcoming")
        elif i == active + 1 and _active_done:
            # v1.0.2.0: Next step is the target — show pulsing CTA
            step_states.append("next-target")
        elif i == active + 1:
            # Next step is reachable but current not done yet
            step_states.append("upcoming")
        else:
            # Check if all prior steps are done
            all_prior_done = all(done[j] for j in range(i))
            step_states.append("upcoming" if all_prior_done else "locked")

    # v1.0.3.1: Larger SVGs to match 46px circles
    check_svg = (
        '<svg width="22" height="22" viewBox="0 0 22 22" fill="none" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path d="M4.5 11.5L9 16L17.5 6" stroke="white" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    # Lock icon for locked steps
    lock_svg = (
        '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<rect x="4" y="8" width="10" height="7" rx="2" stroke="#D1D5DB" stroke-width="1.5"/>'
        '<path d="M6.5 8V6C6.5 4.34 7.84 3 9.5 3V3C11.16 3 12.5 4.34 12.5 6V8" '
        'stroke="#D1D5DB" stroke-width="1.5" stroke-linecap="round"/>'
        '</svg>'
    )

    html = '<div class="stepper-nav">'
    for i, sm in enumerate(SECTION_META):
        state = step_states[i]
        state_cls = f"st-{state}"

        # Circle content: checkmark for done/active-done, number for active/upcoming, lock for locked
        if state == "done":
            circle_content = check_svg
        elif i == active and _active_done:
            circle_content = check_svg  # v1.0.2.1: Show checkmark on active step when complete
        elif state == "locked":
            circle_content = lock_svg
        else:
            circle_content = str(i + 1)

        # Summary text: done steps show summary, next-target shows CTA
        if state == "done":
            summary = _section_summary(i)
        elif i == active and _active_done:
            summary = _section_summary(i) or "Complete"  # v1.0.2.1: Show summary on active-done
        elif state == "next-target":
            summary = "Up next"
        else:
            summary = sm["desc"]

        # v1.0.2.0: Add step-done class to active step when it's complete
        _extra_cls = " step-done" if (i == active and _active_done) else ""
        # v1.0.3.5: Stepper is visual-only (no clickable class, no JS wiring)
        html += f'<div class="stepper-step {state_cls}{_extra_cls}" data-step="{i}" data-state="{state}">'
        html += f'<div class="stepper-circle">{circle_content}</div>'
        html += f'<div class="stepper-title">{sm["title"]}</div>'
        html += f'<div class="stepper-desc">{summary}</div>'
        html += '</div>'
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

    # v1.0.3.5: Stepper is visual-only. Navigation via visible Next buttons on each page.


_SCROLL_TO_TOP_JS = """<script>
(function() {
    var doc = window.parent.document;
    function scrollUp() {
        doc.querySelectorAll('section.main,[data-testid="stAppViewContainer"],[data-testid="stVerticalBlock"],.main .block-container,.main').forEach(function(e){e.scrollTop=0;});
        window.parent.scrollTo(0, 0);
        try { doc.body.scrollTop = 0; doc.documentElement.scrollTop = 0; } catch(e) {}
        try {
            var first = doc.querySelector('.main .block-container > div:first-child, [data-testid="stVerticalBlock"] > div:first-child');
            if (first) first.scrollIntoView({behavior: 'instant', block: 'start'});
        } catch(e) {}
    }
    // v1.8.7.5: Fire aggressively with timers AND a MutationObserver.
    // The observer catches Streamlit's async widget rendering which can
    // restore scroll position after our initial scrollUp calls.
    var delays = [0, 10, 30, 60, 100, 200, 400, 700, 1200, 2000, 3000];
    for (var d = 0; d < delays.length; d++) { setTimeout(scrollUp, delays[d]); }
    // MutationObserver: keep scrolling to top for 4s as DOM changes
    try {
        var target = doc.querySelector('section.main') || doc.body;
        var endTime = Date.now() + 4000;
        var obs = new MutationObserver(function() {
            if (Date.now() < endTime) { scrollUp(); } else { obs.disconnect(); }
        });
        obs.observe(target, {childList: true, subtree: true});
        setTimeout(function(){ obs.disconnect(); }, 4500);
    } catch(e) {}
})();
</script>"""

# v1.8.4: Additional inline scroll-to-top that can be injected via st.markdown
# (works alongside the components.html version for double coverage)
_SCROLL_TO_TOP_INLINE = """<script>
(function(){
    var doc = window.parent.document;
    function s(){
        doc.querySelectorAll('section.main,[data-testid="stAppViewContainer"]').forEach(function(e){e.scrollTop=0;});
        window.parent.scrollTo(0,0);
        try{doc.documentElement.scrollTop=0;}catch(e){}
        try{var f=doc.querySelector('.main .block-container > div:first-child');if(f)f.scrollIntoView({behavior:'instant',block:'start'});}catch(e){}
    }
    [0,50,150,350,700,1200,2000,3000].forEach(function(d){setTimeout(s,d);});
    try{var t=doc.querySelector('section.main')||doc.body;var end=Date.now()+4000;
    var o=new MutationObserver(function(){if(Date.now()<end){s();}else{o.disconnect();}});
    o.observe(t,{childList:true,subtree:true});setTimeout(function(){o.disconnect();},4500);}catch(e){}
})();
</script>"""


def _inject_scroll_to_top_js() -> None:
    """Inject scroll-to-top via both components.html and inline markdown.

    v1.8.4: Double injection strategy — uses both components.html (fires in
    an iframe) and inline markdown (fires in the parent frame). This ensures
    scroll resets even when one method is blocked or delayed by Streamlit's
    rendering pipeline. Fires 14+ times over 3 seconds.
    v1.8.7.5: Added MutationObserver to keep scrolling as widgets render.
    """
    _st_components.html(_SCROLL_TO_TOP_JS, height=0)
    st.markdown(_SCROLL_TO_TOP_INLINE, unsafe_allow_html=True)



# v1.7.0: Restore persisted widget values EARLY — before sidebar or any code
# that reads study_title, study_description, etc.  Streamlit removes widget keys
# from session_state when their widgets are not rendered on the current page.
for _pk in ["study_title", "study_description", "team_name", "team_members_raw"]:
    _saved_val = st.session_state.get(f"_p_{_pk}")
    if _saved_val is not None and _pk not in st.session_state:
        st.session_state[_pk] = _saved_val

with st.sidebar:
    # Mini brand
    st.markdown(
        f'<div style="font-size:0.82rem;font-weight:600;color:#374151;padding:0 0 4px 0;">'
        f'{APP_TITLE}</div>',
        unsafe_allow_html=True,
    )

    advanced_mode = st.toggle("Advanced mode", value=st.session_state.get("advanced_mode", False))
    st.session_state["advanced_mode"] = advanced_mode

    st.divider()

    # Study snapshot
    title = st.session_state.get('study_title') or st.session_state.get('_p_study_title', '') or ''
    sample_size = st.session_state.get('sample_size', 0)
    snapshot_conditions = st.session_state.get("selected_conditions") or []
    snapshot_conditions = snapshot_conditions + st.session_state.get("custom_conditions", [])
    snapshot_conditions = list(dict.fromkeys([c for c in snapshot_conditions if str(c).strip()]))
    inferred = st.session_state.get("inferred_design", {})
    snapshot_scales = inferred.get("scales", [])

    if title:
        st.markdown(f"**{title[:48]}{'...' if len(title) > 48 else ''}**")
    else:
        st.caption("No study configured yet")

    detected_factors = _infer_factors_from_conditions(snapshot_conditions) if snapshot_conditions else []
    if len(detected_factors) == 2:
        f1_levels = len(detected_factors[0].get("levels", []))
        f2_levels = len(detected_factors[1].get("levels", []))
        design_hint = f"{f1_levels}\u00d7{f2_levels} factorial"
    elif len(detected_factors) > 2:
        design_hint = f"{len(detected_factors)}-factor"
    elif snapshot_conditions:
        design_hint = f"{len(snapshot_conditions)}-condition"
    else:
        design_hint = "\u2014"

    _snapshot_lines = []
    _snapshot_lines.append(f"N = {sample_size}" if sample_size else "N = \u2014")
    _snapshot_lines.append(f"Design: {design_hint}")
    if snapshot_conditions and len(snapshot_conditions) <= 5:
        _snapshot_lines.append(f"Conditions: {', '.join(snapshot_conditions)}")
    elif snapshot_conditions:
        _snapshot_lines.append(f"Conditions: {len(snapshot_conditions)} groups")
    if snapshot_scales:
        _n_scales = len(snapshot_scales)
        _first = snapshot_scales[0].get("name", "Scale")[:18] if snapshot_scales else ""
        _snapshot_lines.append(
            f"DVs: {_first}" + (f" +{_n_scales - 1} more" if _n_scales > 1 else "")
        )
    for _sl in _snapshot_lines:
        st.caption(_sl)

    # Start Over — compact
    st.divider()
    _confirm_reset = st.session_state.get("_confirm_reset", False)
    if not _confirm_reset:
        if st.button("Start Over", key="start_over_btn", use_container_width=True, type="secondary"):
            st.session_state["_confirm_reset"] = True
            _navigate_to(-1)
    else:
        st.warning("Clear all entries and start fresh?")
        _c1, _c2 = st.columns(2)
        with _c1:
            if st.button("Yes, clear", key="confirm_reset_yes", use_container_width=True, type="primary"):
                st.session_state["_pending_reset"] = True
                _navigate_to(-1)
        with _c2:
            if st.button("Cancel", key="confirm_reset_no", use_container_width=True):
                st.session_state["_confirm_reset"] = False
                _navigate_to(-1)


# v1.4.14: Pending reset handler — MUST run before ANY widgets render.
# The "Start Over" button sets the flag; we clear state here so that
# Streamlit widgets see empty session_state on this render cycle.
if st.session_state.pop("_pending_reset", False):
    _keys_to_preserve = {"active_page"}
    for _k in list(st.session_state.keys()):
        if _k not in _keys_to_preserve:
            try:
                del st.session_state[_k]
            except Exception:
                pass
    st.session_state["active_page"] = -1  # Reset to landing page
    st.session_state["_page_just_changed"] = True  # Scroll to top

# =====================================================================
# v1.5.0: FLOW NAVIGATION — Modern section-based single-page flow.
# Replaces the old wizard stepper + clickable buttons + Next/Back.
# One visual ribbon + one row of buttons. No redundant navigation.
# =====================================================================

# v1.8.7.5: Two-phase scroll-then-nav removed (replaced with direct navigation)
# Clean up any stale flags from previous versions
st.session_state.pop("_scroll_then_nav_target", None)
st.session_state.pop("_review_at_top", None)

# ── v1.0.3.5: Navigation via visible Next buttons on each page ──────
# ── Apply pending navigation (from auto-advance or _navigate_to) ─────
_pending_nav = st.session_state.pop("_pending_nav", None)
if _pending_nav is not None:
    st.session_state["active_page"] = _pending_nav

if "active_page" not in st.session_state:
    st.session_state["active_page"] = -1  # Landing page

# ── v1.0.4.7: Admin dashboard — hidden page via ?admin=1 query param ──
_admin_mode = False
try:
    _qp = st.query_params
    _admin_mode = _qp.get("admin", "") == "1"
except Exception:
    pass

if _admin_mode:
    _render_admin_dashboard()
    st.stop()

# Guard: clamp active_page to valid range (-1 = landing, 0-3 = wizard)
active_page = max(-1, min(int(st.session_state.get("active_page", -1)), 3))
st.session_state["active_page"] = active_page

# ── Scroll to top on section change ──────────────────────────────────
# v1.8.4: Fire scroll-to-top for both _page_just_changed AND _force_scroll_top
# This ensures scrolling happens regardless of which navigation method was used
if st.session_state.pop("_page_just_changed", False) or st.session_state.pop("_force_scroll_top", False):
    _inject_scroll_to_top_js()

# ── Inject flow navigation CSS ───────────────────────────────────────
st.markdown(_FLOW_NAV_CSS, unsafe_allow_html=True)

# ── Completion status (single call, reused for ribbon + nav) ──────────
_step_completion = _get_step_completion()
_step_done = [
    _step_completion["study_title"] and _step_completion["study_description"],
    _step_completion["qsf_uploaded"],
    _step_completion["conditions_set"] and _step_completion["design_ready"] and _step_completion.get("dvs_configured", True),
    bool(st.session_state.get("has_generated")),
]

# ── Flow navigation (v1.8.0: segmented progress bar, only on wizard pages) ──
if active_page >= 0:
    _render_flow_nav(active_page, _step_done)

    # v1.0.3.0: All navigation via clickable stepper. No redundant forward buttons.
    _top_fwd_ph = None


def _get_condition_candidates(
    preview: Optional[QSFPreviewResult],
    enhanced_analysis: Optional[DesignAnalysisResult],
) -> List[str]:
    """
    Extract condition candidates from QSF analysis.

    IMPORTANT: Filters out all trash/unused/admin blocks to prevent
    false condition detection. Uses comprehensive exclusion patterns.
    """
    # Comprehensive exclusion patterns - must NOT be conditions
    excluded_lower = {
        # Trash/unused
        'trash', 'unused', 'deleted', 'archived', 'old', 'deprecated',
        'trash / unused questions', 'trash/unused questions', 'do not use',
        # Generic
        'block', 'default', 'default question block', 'standard', 'main',
        'block 1', 'block 2', 'block 3', 'block 4', 'block 5',
        # Intro/consent
        'intro', 'introduction', 'welcome', 'consent', 'informed consent',
        # Instructions
        'instructions', 'directions', 'guidelines', 'tutorial',
        # Quality control
        'attention check', 'manipulation check', 'quality check', 'screener',
        'captcha', 'bot check',
        # Demographics/end
        'demographics', 'background', 'debrief', 'debriefing',
        'end', 'ending', 'thank you', 'thanks', 'completion',
        # Structural
        'feedback', 'comments', 'practice', 'training', 'timer', 'timing',
        # Payment
        'payment', 'compensation', 'prolific', 'mturk', 'code',
    }

    # Patterns that indicate exclusion
    excluded_patterns = ['trash', 'unused', 'delete', 'archive', 'old question',
                         'not use', 'hidden', 'disabled', 'draft', 'test']

    def is_excluded(name: str) -> bool:
        """Check if a name should be excluded from conditions."""
        name_lower = name.lower().strip()
        if name_lower in excluded_lower:
            return True
        if any(pat in name_lower for pat in excluded_patterns):
            return True
        # Exclude generic numbered blocks
        if re.match(r'^block\s*\d*$', name_lower):
            return True
        return False

    candidates: List[str] = []

    # Extract from enhanced analysis (highest quality source)
    if enhanced_analysis and enhanced_analysis.conditions:
        for cond in enhanced_analysis.conditions:
            if cond.source in ("QSF Randomizer", "QSF Block Name"):
                if not is_excluded(cond.name):
                    candidates.append(cond.name)

    # Extract from QSF blocks
    if preview and preview.blocks:
        for block in preview.blocks:
            block_name = block.block_name.strip()
            if block_name and not is_excluded(block_name):
                # Also check block type
                if hasattr(block, 'block_type') and block.block_type in ('Trash', 'Default'):
                    continue
                candidates.append(block_name)

    # Extract from embedded data conditions (may indicate randomization)
    if preview and hasattr(preview, 'embedded_data_conditions'):
        for edc in (preview.embedded_data_conditions or []):
            if isinstance(edc, dict):
                name = edc.get('name', '') or edc.get('field', '')
                if name and not is_excluded(name):
                    candidates.append(name)

    # Deduplicate while preserving order
    return list(dict.fromkeys([c for c in candidates if c.strip()]))


def _get_qsf_identifiers(preview: Optional[QSFPreviewResult]) -> List[str]:
    """
    Extract all identifiers from QSF that could potentially be condition indicators.
    Returns alphabetically sorted list excluding time/standard Qualtrics variables.
    """
    identifiers: List[str] = []

    # Standard Qualtrics variables to exclude
    excluded_patterns = {
        'startdate', 'enddate', 'status', 'ipaddress', 'progress',
        'duration', 'finished', 'recordeddate', 'responseid', 'recipientlastname',
        'recipientfirstname', 'recipientemail', 'externalreference', 'locationlatitude',
        'locationlongitude', 'distributionchannel', 'userlanguage', 'q_', 'gc', 'q_recaptchascore',
        'q_relevantidduplicate', 'q_relevantidduplicatescore', 'q_relevantidfraudulentscore',
        'q_relevantidlasttimesurvey', 'q_totalcount', 'q_url', 'q_ballotboxstuffing',
        'q_ees', 'q_r', 'q_preview', 'q_qualtricssurvey', 'timing', '_first', '_last',
        '_pagesubmit', '_click', 'timestamp', 'datetime', 'date', 'time', 'browser',
        'version', 'operating', 'resolution', 'devicetype', 'consent', 'captcha',
    }

    if not preview:
        return []

    # Extract embedded data fields
    if preview.embedded_data:
        for ed in preview.embedded_data:
            if ed and isinstance(ed, str):
                ed_lower = ed.lower()
                # Check if it's a standard/time variable
                if not any(excl in ed_lower for excl in excluded_patterns):
                    identifiers.append(ed)

    # Extract from blocks (question IDs and data export tags could be condition indicators)
    if preview.blocks:
        for block in preview.blocks:
            # Block names could be conditions
            block_name = block.block_name.strip()
            if block_name and block_name.lower() not in (
                "default question block", "trash / unused questions", "block",
                "intro", "introduction", "consent", "demographics", "debrief",
            ):
                identifiers.append(f"[Block] {block_name}")

            # Question IDs from blocks
            if hasattr(block, 'questions'):
                for q in block.questions:
                    q_id = q.question_id
                    if q_id:
                        q_lower = q_id.lower()
                        if not any(excl in q_lower for excl in excluded_patterns):
                            identifiers.append(q_id)

    # Extract from randomizer info if available
    if hasattr(preview, 'randomizer_info') and preview.randomizer_info:
        rand_info = preview.randomizer_info
        if rand_info.get('randomizers'):
            for rand in rand_info['randomizers']:
                desc = rand.get('description', '')
                if desc and desc.strip():
                    identifiers.append(f"[Randomizer] {desc}")

    # Dedupe and sort alphabetically
    unique_ids = list(dict.fromkeys(identifiers))
    return sorted(unique_ids, key=lambda x: x.lower().lstrip('['))


# v1.5.0: _render_step_navigation removed — top flow nav handles all navigation


def _get_total_conditions() -> int:
    """Get total number of conditions from all sources."""
    selected = st.session_state.get("selected_conditions", [])
    custom = st.session_state.get("custom_conditions", [])
    return len(set(selected + custom))



# =====================================================================
# LANDING PAGE (active_page == -1)
# =====================================================================
if active_page == -1:
    # Hide sidebar on landing page
    st.markdown("""<style>
        section[data-testid="stSidebar"] { display: none; }
        section.main .block-container { max-width: 1000px; }
        /* Landing page CTA button styling */
        [data-testid="stButton"] button[kind="primary"] {
            font-size: 1.05rem !important;
            padding: 12px 32px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stButton"] button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(255,75,75,0.3) !important;
        }
    </style>""", unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div class="landing-hero">'
        '<h1>Behavioral Experiment<br>Simulation Tool</h1>'
        '<p class="subtitle">Generate realistic pilot datasets to build and test<br>'
        'your analysis pipeline before collecting real data</p>'
        '<p class="creator">Created by Dr. <a href="https://eugendimant.github.io/">Eugen Dimant</a></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature cards
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="fc-icon">\U0001F9EA</div>'
        '<h4>Test Before You Collect</h4>'
        '<p>Catch analysis errors, verify variable coding, and debug scripts '
        'on realistic synthetic data — before spending resources on real participants.</p></div>'

        '<div class="feature-card"><div class="fc-icon">\U0001F4AC</div>'
        '<h4>Realistic Open-Ended Responses</h4>'
        '<p>AI-generated free-text answers that match numeric ratings, '
        'built from 50+ behavioral personas across 225+ research domains.</p></div>'

        '<div class="feature-card"><div class="fc-icon">\U0001F4CA</div>'
        '<h4>Ready-to-Run Analysis Code</h4>'
        '<p>Get R and Python scripts tailored to your exact design — ANOVAs, t-tests, '
        'regressions, mediation — ready for immediate execution.</p></div>'

        '<div class="feature-card"><div class="fc-icon">\U0001F393</div>'
        '<h4>Built for Research & Teaching</h4>'
        '<p>Instructor reports, group management, and pre-registration consistency checks '
        'make this ideal for both active research and classroom use.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # v1.0.1.3: Enhanced trust signals with bolder numbers
    st.markdown(
        '<div class="trust-strip">'
        '<div class="trust-item"><span class="trust-num">225+</span><span class="trust-label">Research Domains</span></div>'
        '<div class="trust-divider"></div>'
        '<div class="trust-item"><span class="trust-num">40</span><span class="trust-label">Question Types</span></div>'
        '<div class="trust-divider"></div>'
        '<div class="trust-item"><span class="trust-num">5</span><span class="trust-label">Analysis Languages</span></div>'
        '<div class="trust-divider"></div>'
        '<div class="trust-item"><span class="trust-num">50+</span><span class="trust-label">Behavioral Personas</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="max-width:680px;margin:0 auto;height:24px;"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div class="how-it-works">'
        '<h3>How It Works</h3>'
        '<div class="hiw-steps">'
        '<div class="hiw-step"><div class="hiw-num">1</div>'
        '<div class="hiw-title">Name Your Study</div>'
        '<div class="hiw-desc">Title, description &amp; team</div></div>'
        '<div class="hiw-arrow">\u2192</div>'
        '<div class="hiw-step"><div class="hiw-num">2</div>'
        '<div class="hiw-title">Provide Your Design</div>'
        '<div class="hiw-desc">Upload QSF or describe in words</div></div>'
        '<div class="hiw-arrow">\u2192</div>'
        '<div class="hiw-step"><div class="hiw-num">3</div>'
        '<div class="hiw-title">Configure Experiment</div>'
        '<div class="hiw-desc">Conditions, factors &amp; DVs</div></div>'
        '<div class="hiw-arrow">\u2192</div>'
        '<div class="hiw-step"><div class="hiw-num">4</div>'
        '<div class="hiw-title">Generate &amp; Download</div>'
        '<div class="hiw-desc">CSV, scripts &amp; reports</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # v1.0.3.1: CTA + prominent Methods PDF download
    _cta_col1, _cta_col2, _cta_col3 = st.columns([1, 2, 1])
    with _cta_col2:
        if st.button("Start Your Simulation  \u2192", type="primary", use_container_width=True, key="landing_cta"):
            _navigate_to(0)

    # v1.0.3.1: Prominent Methods PDF — styled download section
    methods_pdf_path = Path(__file__).resolve().parent.parent / "docs" / "papers" / "methods_summary.pdf"
    if methods_pdf_path.exists():
        st.markdown(
            '<div style="text-align:center;margin:16px 0 8px 0;">'
            '<span style="font-size:0.88rem;color:#4B5563;">'
            '\U0001F4C4 Read our methods paper for the full scientific approach'
            '</span></div>',
            unsafe_allow_html=True,
        )
        _pdf_dl1, _pdf_dl2, _pdf_dl3 = st.columns([1, 2, 1])
        with _pdf_dl2:
            st.download_button(
                "\u2B07 Download Methods Paper (PDF)",
                data=methods_pdf_path.read_bytes(),
                file_name=methods_pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
                key="landing_methods_pdf",
            )

    # v1.9.0: Professional tabbed info sections (replacing generic expanders)
    st.markdown('<div class="landing-tabs-container">', unsafe_allow_html=True)

    _info_tab1, _info_tab2, _info_tab3, _info_tab4 = st.tabs([
        "Who Uses This",
        "Capabilities",
        "Step-by-Step",
        "Research & Citations",
    ])

    with _info_tab1:
        st.markdown(
            '<div class="landing-tab-content">'
            '<div class="use-case-grid">'

            '<div class="use-case-card">'
            '<strong>Researchers & PIs</strong>'
            '<span>Validate analysis pipelines before IRB data collection. Simulate expected '
            'effect sizes and power for grant proposals.</span></div>'

            '<div class="use-case-card">'
            '<strong>University Instructors</strong>'
            '<span>Assign realistic data analysis exercises where every student gets unique '
            'datasets. Instructor reports provide answer keys.</span></div>'

            '<div class="use-case-card">'
            '<strong>Market Research Teams</strong>'
            '<span>Prototype survey instruments and verify analysis code handles real-world '
            'response patterns before fielding.</span></div>'

            '<div class="use-case-card">'
            '<strong>UX Research Groups</strong>'
            '<span>Generate pilot data for A/B tests, preference studies, and usability '
            'scales before committing to participant panels.</span></div>'

            '<div class="use-case-card">'
            '<strong>Government & Policy Labs</strong>'
            '<span>Simulate citizen survey responses across demographic segments to validate '
            'measurement instruments and analysis plans.</span></div>'

            '<div class="use-case-card">'
            '<strong>Graduate Students</strong>'
            '<span>Practice statistical analysis with realistic datasets before collecting '
            'real data. Learn data cleaning and analysis pipelines.</span></div>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    with _info_tab2:
        st.markdown(
            '<div class="landing-tab-content">'

            '<div class="capability-item">'
            '<div class="cap-icon">\U0001f9ea</div>'
            '<div class="cap-text"><strong>Test Before You Collect</strong>'
            '<span>Generate a publication-ready CSV with realistic Likert-scale responses, attention check '
            'failures, individual differences, and demographic distributions. The data mirrors real Qualtrics '
            'output format so your analysis scripts work identically on both simulated and real data.</span></div></div>'

            '<div class="capability-item">'
            '<div class="cap-icon">\U0001f4ac</div>'
            '<div class="cap-text"><strong>Realistic Open-Ended Responses</strong>'
            '<span>Uses a multi-provider LLM failover chain with 50+ behavioral personas to generate unique, '
            'context-aware free-text responses. Each response aligns with the participant\'s numeric ratings '
            'and assigned persona. Supports 225+ research domains and 40 question types.</span></div></div>'

            '<div class="capability-item">'
            '<div class="cap-icon">\U0001f4ca</div>'
            '<div class="cap-text"><strong>Ready-to-Run Analysis Code</strong>'
            '<span>Automatically generates scripts in R, Python, Julia, SPSS, and Stata \u2014 tailored to your '
            'specific experimental design. Includes data loading, variable coding, condition comparisons, '
            'and appropriate statistical tests.</span></div></div>'

            '<div class="capability-item">'
            '<div class="cap-icon">\U0001f393</div>'
            '<div class="cap-text"><strong>Built for Research & Teaching</strong>'
            '<span>Instructor-only reports include effect sizes, power estimates, and visualizations that '
            'students don\'t see. Group management tracks team usage. Pre-registration consistency checks '
            'compare your design against OSF, AEA, or AsPredicted specifications.</span></div></div>'

            '</div>',
            unsafe_allow_html=True,
        )

    with _info_tab3:
        st.markdown(
            '<div class="landing-tab-content">'

            '<div class="step-detail-item">'
            '<div class="step-num">1</div>'
            '<div class="step-detail-text"><strong>Name Your Study</strong>'
            '<span>Enter your study title and a description of your experiment\'s purpose, manipulation, '
            'and main outcomes. This information is embedded in all generated outputs (data files, '
            'analysis scripts, reports).</span></div></div>'

            '<div class="step-detail-item">'
            '<div class="step-num">2</div>'
            '<div class="step-detail-text"><strong>Provide Your Design</strong>'
            '<span>Upload a Qualtrics .qsf file for automatic detection of conditions, scales, randomizers, '
            'and embedded data. Or describe your study in plain text using the guided builder \u2014 it parses '
            'natural language descriptions of your experimental design.</span></div></div>'

            '<div class="step-detail-item">'
            '<div class="step-num">3</div>'
            '<div class="step-detail-text"><strong>Configure Experiment</strong>'
            '<span>Review auto-detected conditions, configure factorial designs (2\u00d72, 2\u00d73, etc.), set sample '
            'sizes with per-condition allocation, and verify dependent variables. The tool detects matrix '
            'scales, Likert items, sliders, and numeric inputs automatically.</span></div></div>'

            '<div class="step-detail-item">'
            '<div class="step-num">4</div>'
            '<div class="step-detail-text"><strong>Generate & Download</strong>'
            '<span>Choose a difficulty level (easy to expert) that controls noise, attention check failure rates, '
            'and response quality. Generate your complete data package \u2014 CSV, codebook, analysis scripts in 5 '
            'languages, summary reports, and metadata.</span></div></div>'

            '</div>',
            unsafe_allow_html=True,
        )

    with _info_tab4:
        st.markdown(
            '<div class="landing-tab-content">'
            '<div class="research-list">'

            '<a class="research-item" href="https://doi.org/10.1017/pan.2023.2" target="_blank">'
            '<span class="ri-authors">Argyle et al. (2023)</span>'
            '<span class="ri-venue">Political Analysis</span>'
            '<span class="ri-insight">LLMs replicate human survey responses across demographics</span></a>'

            '<a class="research-item" href="https://www.nber.org/papers/w31122" target="_blank">'
            '<span class="ri-authors">Horton (2023)</span>'
            '<span class="ri-venue">NBER</span>'
            '<span class="ri-insight">LLM agents as stand-ins for human subjects in experiments</span></a>'

            '<a class="research-item" href="https://doi.org/10.48550/arXiv.2301.07543" target="_blank">'
            '<span class="ri-authors">Aher, Arriaga &amp; Kalai (2023)</span>'
            '<span class="ri-venue">ICML</span>'
            '<span class="ri-insight">Simulating classic behavioral experiments with LLMs</span></a>'

            '<a class="research-item" href="https://doi.org/10.1145/3586183.3606763" target="_blank">'
            '<span class="ri-authors">Park et al. (2023)</span>'
            '<span class="ri-venue">ACM UIST</span>'
            '<span class="ri-insight">Generative agents with believable human behavior</span></a>'

            '<a class="research-item" href="https://doi.org/10.1073/pnas.2218523120" target="_blank">'
            '<span class="ri-authors">Binz &amp; Schulz (2023)</span>'
            '<span class="ri-venue">PNAS</span>'
            '<span class="ri-insight">LLMs match human cognitive biases and decision-making</span></a>'

            '<a class="research-item" href="https://doi.org/10.1016/j.tics.2023.04.008" target="_blank">'
            '<span class="ri-authors">Dillion et al. (2023)</span>'
            '<span class="ri-venue">Trends in Cognitive Sciences</span>'
            '<span class="ri-insight">Can AI language models replace human participants?</span></a>'

            '<a class="research-item" href="https://doi.org/10.1073/pnas.2317245121" target="_blank">'
            '<span class="ri-authors">Westwood (2025)</span>'
            '<span class="ri-venue">PNAS</span>'
            '<span class="ri-insight">Validating LLM-generated survey responses at scale</span></a>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="landing-footer">'
        f'<div class="footer-version">v{APP_VERSION}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 1: STUDY SETUP
# =====================================================================
if active_page == 0:
    # v1.8.0: Hero card removed — now on landing page
    st.markdown(
        '<div class="section-guide">'
        '<strong>Step 1 &middot; Setup</strong> &mdash; '
        'Name your study and describe its purpose.</div>',
        unsafe_allow_html=True,
    )

    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]

    # v1.0.3.7: Navigation row — Continue at top when step is complete
    if step1_done:
        if st.button("Continue to Study Input \u2192", key="nav_next_0", type="primary"):
            _navigate_to(1)

    # v1.7.0: Clean form — study details first, team optional below
    study_title = st.text_input(
        "Study title *",
        placeholder="e.g., Effect of AI Labels on Consumer Trust",
        help="Appears in the report and simulated data outputs.",
        key="study_title",
    )
    study_description = st.text_area(
        "Study description *",
        height=120,
        placeholder=(
            "Describe your study's purpose, manipulation, and main outcomes.\n"
            "Example: 'We examine whether AI-generated content labels "
            "affect consumer trust in product reviews, comparing labeled vs. "
            "unlabeled conditions on a 7-point trust scale.'"
        ),
        help="Include your manipulation, population, and intended outcomes.",
        key="study_description",
    )

    # v1.8.9: Always persist widget values (including empty) so clearing a field sticks
    st.session_state["_p_study_title"] = study_title or ""
    st.session_state["_p_study_description"] = study_description or ""

    with st.expander("Team information (optional)"):
        _t1, _t2 = st.columns(2)
        with _t1:
            team_name = st.text_input(
                "Team name",
                placeholder="e.g., Team Alpha",
                help="Helps instructors identify your team.",
                key="team_name",
            )
        with _t2:
            members = st.text_area(
                "Team members (one per line)",
                height=80,
                placeholder="John Doe\nJane Smith",
                key="team_members_raw",
            )
        # v1.8.9: Persist team fields too (was missing _p_ pattern)
        st.session_state["_p_team_name"] = team_name or ""
        st.session_state["_p_team_members_raw"] = members or ""

    # v1.0.2.1: Structured completion indicator (bottom of form)
    if not step1_done:
        _title_ok = "\u2705" if completion["study_title"] else "\u2B1C"
        _desc_ok = "\u2705" if completion["study_description"] else "\u2B1C"
        st.caption(f"{_title_ok} Study title &nbsp;&nbsp; {_desc_ok} Study description")

    # v1.0.3.7: Setup page is short — no scroll button needed


# =====================================================================
# PAGE 2: FILE UPLOAD / STUDY BUILDER
# =====================================================================
if active_page == 1:
    st.markdown(
        '<div class="section-guide">'
        '<strong>Step 2 &middot; Study Input</strong> &mdash; '
        'Upload a Qualtrics QSF file or describe your study design.</div>',
        unsafe_allow_html=True,
    )
    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]
    step2_done = completion["qsf_uploaded"]

    # v1.0.3.7: Navigation row — Back + Continue at top
    _nav1_left, _nav1_right = st.columns([1, 1])
    with _nav1_left:
        if st.button("\u2190 Back to Setup", key="nav_back_1"):
            _navigate_to(0)
    with _nav1_right:
        if step2_done:
            if st.button("Continue to Design \u2192", key="nav_next_1", type="primary"):
                _navigate_to(2)

    if not step1_done:
        st.warning("Complete **Setup** first \u2014 fill in the study title and description on the Setup page.")

    # v1.6.0: Cleaner mode selector with descriptive labels
    _mode_options = {
        "upload_qsf": "Upload QSF file",
        "describe_study": "Describe in words",
    }
    input_mode = st.radio(
        "Input method",
        options=list(_mode_options.keys()),
        format_func=lambda x: _mode_options[x],
        index=0 if st.session_state.get("study_input_mode", "upload_qsf") == "upload_qsf" else 1,
        key="study_input_mode_radio",
        horizontal=True,
        help="No QSF file? Choose 'Describe in words' for a guided form.",
    )
    st.session_state["study_input_mode"] = input_mode

    # PATH A: CONVERSATIONAL STUDY BUILDER
    if input_mode == "describe_study":
        _render_conversational_builder()

    # PATH B: QSF FILE UPLOAD
    _show_qsf_upload = (input_mode == "upload_qsf")

    # v1.0.3.0: Removed redundant sub-header — radio label is sufficient

    # QSF file upload section (only shown in upload_qsf mode)
    if _show_qsf_upload:
        existing_qsf_name = st.session_state.get("qsf_file_name")
        existing_qsf_content = st.session_state.get("qsf_raw_content")

        # v1.0.3.0: Full-width QSF upload — help instructions in expander below
        with st.container():
            if existing_qsf_name and existing_qsf_content:
                st.success(f"**{existing_qsf_name}** uploaded ({len(existing_qsf_content):,} bytes)")
                change_qsf = st.checkbox("Upload a different QSF file", value=False, key="change_qsf")
            else:
                change_qsf = True

            if change_qsf or not existing_qsf_content:
                qsf_file = st.file_uploader(
                    "QSF file",
                    type=["qsf", "zip", "json"],
                    help=(
                        "Export from Qualtrics via Tools → Import/Export → Export Survey. "
                        "Upload the .qsf (or .zip) here."
                    ),
                )
            else:
                qsf_file = None

        parser = _get_qsf_preview_parser()
        preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

        stored_preview = st.session_state.get("qsf_preview", None)
        is_new_upload = qsf_file is not None and (
            not stored_preview or
            st.session_state.get("qsf_file_name") != qsf_file.name
        )

        if qsf_file is not None and is_new_upload:
            try:
                content = qsf_file.read()
                payload, payload_name = _extract_qsf_payload(content)
                preview = parser.parse(payload)
                st.session_state["qsf_preview"] = preview
                st.session_state["qsf_raw_content"] = payload
                st.session_state["qsf_file_name"] = qsf_file.name
                # v1.8.9: Clear cached condition candidates on new upload
                st.session_state.pop("condition_candidates", None)
                st.session_state.pop("selected_conditions", None)
                st.session_state.pop("custom_conditions", None)
                # v1.0.1.5: Clear stale design state from previous QSF or builder path
                st.session_state.pop("scales_confirmed", None)
                st.session_state.pop("confirmed_scales", None)
                st.session_state.pop("confirmed_open_ended", None)
                st.session_state.pop("inferred_design", None)
                st.session_state.pop("open_ended_confirmed", None)
                st.session_state.pop("_oe_version", None)
                st.session_state.pop("_dv_version", None)

                if preview.success:
                    # Naming: YYYY_MM_DD_OriginalFilename.qsf
                    _upload_date = datetime.now().strftime("%Y_%m_%d")
                    _upload_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', qsf_file.name)
                    collect_qsf_async(f"{_upload_date}_{_upload_name}", payload)
                    with st.spinner("Analyzing experimental design..."):
                        enhanced_analysis = _perform_enhanced_analysis(
                            qsf_content=payload,
                            prereg_outcomes=st.session_state.get("prereg_outcomes", ""),
                            prereg_iv=st.session_state.get("prereg_iv", ""),
                            prereg_text=st.session_state.get("prereg_text_sanitized", ""),
                            prereg_pdf_text=st.session_state.get("prereg_pdf_text", ""),
                        )
                        if enhanced_analysis:
                            st.session_state["enhanced_analysis"] = enhanced_analysis
                    _navigate_to(1)
                else:
                    st.error("QSF parsed but validation failed. See warnings below.")
            except Exception as e:
                import traceback
                st.session_state["qsf_preview"] = None
                st.session_state["enhanced_analysis"] = None
                error_details = traceback.format_exc()
                st.error(f"QSF parsing failed: {e}")
                with st.expander("Error details (for debugging)"):
                    st.code(error_details)

    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    # ========================================
    # OPTIONAL: Survey PDF Export
    # ========================================
    st.markdown("")
    st.markdown("#### Survey Materials *(optional)*")
    st.caption("Upload PDF exports or screenshots for better question identification.")

    with st.expander("How to export from Qualtrics", expanded=False):
        st.markdown("""
1. Open survey in Qualtrics → **Tools** → **Import/Export** → **Print Survey**
2. Save as PDF and upload here
""")

    # Check for existing Survey PDF
    existing_survey_pdf_name = st.session_state.get("survey_pdf_name")
    existing_survey_pdf_content = st.session_state.get("survey_pdf_content")

    if existing_survey_pdf_name and existing_survey_pdf_content:
        st.success(f"✓ **{existing_survey_pdf_name}** uploaded ({len(existing_survey_pdf_content):,} bytes)")
        change_survey_pdf = st.checkbox("Upload a different Survey PDF", value=False, key="change_survey_pdf")
    else:
        change_survey_pdf = True

    if change_survey_pdf or not existing_survey_pdf_content:
        survey_files = st.file_uploader(
            "Survey Materials",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="PDF exports, screenshots, or photos of your survey (optional but recommended)",
            key="survey_pdf_uploader",
        )
    else:
        survey_files = None

    if survey_files:
        # Process multiple files
        all_contents = []
        all_names = []
        all_text = ""

        for survey_file in survey_files:
            try:
                file_content = survey_file.read()
                file_name = survey_file.name
                file_ext = file_name.lower().split('.')[-1]

                all_contents.append(file_content)
                all_names.append(file_name)

                # Extract text from PDFs
                if file_ext == 'pdf':
                    file_text = ""
                    extraction_method = None

                    # Method 1: Try PyMuPDF (fitz)
                    try:
                        import fitz
                        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                        try:
                            for page in pdf_doc:
                                file_text += page.get_text()
                            extraction_method = "PyMuPDF"
                        finally:
                            pdf_doc.close()
                    except ImportError:
                        pass
                    except Exception:
                        pass

                    # Method 2: Try pypdf as fallback
                    if not file_text.strip():
                        try:
                            file_text = _extract_pdf_text(file_content)
                            if file_text.strip():
                                extraction_method = "pypdf"
                        except Exception:
                            pass

                    # Method 3: Try pdfplumber as last resort
                    if not file_text.strip():
                        try:
                            import pdfplumber
                            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        file_text += page_text + "\n"
                            extraction_method = "pdfplumber"
                        except Exception:
                            pass

                    all_text += f"\n--- {file_name} ---\n{file_text}"
                    if file_text.strip():
                        st.success(f"✓ {file_name} uploaded (text extracted via {extraction_method})")
                    else:
                        st.info(f"✓ {file_name} uploaded (image-based PDF, no text extracted)")
                else:
                    # PNG/JPG - just store the file, no text extraction
                    st.success(f"✓ {file_name} uploaded (image file)")

            except Exception as e:
                st.error(f"Failed to process {survey_file.name}: {e}")

        # Store all files in session state
        if all_contents:
            st.session_state["survey_pdf_content"] = all_contents[0] if len(all_contents) == 1 else all_contents
            st.session_state["survey_pdf_name"] = ", ".join(all_names)
            st.session_state["survey_materials"] = list(zip(all_names, all_contents))
            st.session_state["survey_pdf_text"] = all_text.strip()

    # ========================================
    # OPTIONAL: Preregistration / AsPredicted
    # ========================================
    st.markdown("")
    st.markdown("#### Preregistration *(optional)*")
    st.caption("Upload AsPredicted or preregistration PDFs to improve simulation accuracy.")

    with st.expander("Add preregistration for better simulations", expanded=False):
        col_prereg1, col_prereg2 = st.columns(2)

        with col_prereg1:
            prereg_files = st.file_uploader(
                "AsPredicted / Preregistration PDF(s)",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more preregistration documents (PDF format)",
                key="prereg_pdf_uploader",
            )

            if prereg_files:
                # Process multiple preregistration files
                all_prereg_contents = []
                all_prereg_names = []
                all_prereg_text = ""

                for prereg_file in prereg_files:
                    try:
                        # Store the PDF content for later analysis
                        pdf_content = prereg_file.read()
                        all_prereg_contents.append(pdf_content)
                        all_prereg_names.append(prereg_file.name)

                        # Try to extract text from PDF using multiple methods
                        pdf_text = ""
                        extraction_method = None

                        # Method 1: Try PyMuPDF (fitz) - best quality
                        try:
                            import fitz
                            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
                            try:
                                for page in pdf_doc:
                                    pdf_text += page.get_text()
                                extraction_method = "PyMuPDF"
                            finally:
                                pdf_doc.close()
                        except ImportError:
                            pass
                        except Exception:
                            pass

                        # Method 2: Try pypdf as fallback
                        if not pdf_text.strip():
                            try:
                                pdf_text = _extract_pdf_text(pdf_content)
                                if pdf_text.strip():
                                    extraction_method = "pypdf"
                            except Exception:
                                pass

                        # Method 3: Try pdfplumber as last resort
                        if not pdf_text.strip():
                            try:
                                import pdfplumber
                                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                                    for page in pdf.pages:
                                        page_text = page.extract_text()
                                        if page_text:
                                            pdf_text += page_text + "\n"
                                extraction_method = "pdfplumber"
                            except ImportError:
                                pass
                            except Exception:
                                pass

                        if pdf_text.strip():
                            all_prereg_text += f"\n\n--- {prereg_file.name} ---\n\n{pdf_text}"

                    except Exception as e:
                        st.error(f"Failed to process {prereg_file.name}: {e}")

                # Store all prereg files in session state
                if all_prereg_contents:
                    # Store first file for backwards compatibility
                    st.session_state["prereg_pdf_content"] = all_prereg_contents[0]
                    st.session_state["prereg_pdf_name"] = all_prereg_names[0]
                    # Store all files
                    st.session_state["prereg_pdf_contents"] = all_prereg_contents
                    st.session_state["prereg_pdf_names"] = all_prereg_names
                    st.session_state["prereg_materials"] = list(zip(all_prereg_names, all_prereg_contents))
                    st.session_state["prereg_pdf_text"] = all_prereg_text.strip()

                    st.success(f"✓ {len(all_prereg_names)} preregistration file(s) uploaded: {', '.join(all_prereg_names)}")
                    if all_prereg_text.strip():
                        st.info(f"Text extracted from {len([t for t in all_prereg_text.split('---') if t.strip()])} document(s).")
                    else:
                        st.warning("PDFs uploaded but text extraction failed. Files will still be included in metadata.")

        with col_prereg2:
            prereg_outcomes = st.text_area(
                "Primary outcome variables",
                placeholder="e.g., purchase_intention, brand_attitude",
                help="List your main dependent variables (comma-separated)",
                key="prereg_outcomes",
            )

            prereg_iv = st.text_area(
                "Independent variables / Manipulations",
                placeholder="e.g., AI_recommendation (yes/no), product_type (utilitarian/hedonic)",
                help="List your manipulated variables and levels",
                key="prereg_iv",
            )

            # Sanitize preregistration text to remove hypothesis-biasing language
            combined_prereg = f"{prereg_outcomes}\n{prereg_iv}".strip()
            sanitized_text, removed_lines = _sanitize_prereg_text(combined_prereg)
            st.session_state["prereg_text_sanitized"] = sanitized_text

    if preview:
        condition_candidates = _get_condition_candidates(
            preview=preview,
            enhanced_analysis=st.session_state.get("enhanced_analysis"),
        )
        st.session_state["condition_candidates"] = condition_candidates

        # No longer extracting from prereg - conditions come from QSF only
        prereg_conditions: List[str] = []
        merged_conditions, condition_sources = _merge_condition_sources(
            preview.detected_conditions or [],
            prereg_conditions,
        )
        if merged_conditions:
            preview.detected_conditions = merged_conditions
            preview.validation_errors = [
                err for err in (preview.validation_errors or [])
                if "No experimental conditions detected" not in err
            ]
            preview.validation_warnings = [
                warn for warn in (preview.validation_warnings or [])
                if "No experimental conditions automatically detected" not in warn
                and "Trash" not in warn
                and "Unused" not in warn
                and "trash" not in warn.lower()
            ]
        st.session_state["condition_sources"] = condition_sources

        selected_conditions = st.session_state.get("selected_conditions") or []
        st.markdown("#### QSF Analysis")
        _n_questions = int(getattr(preview, "total_questions", 0) or 0)
        _n_scales = int(len(getattr(preview, "detected_scales", []) or []))
        _n_candidates = int(len(st.session_state.get("condition_candidates", []) or []))
        warnings = getattr(preview, "validation_warnings", []) or []
        # v1.0.2.3: Compact inline metrics (consistent across all pages)
        st.markdown(
            f'<div style="display:flex;gap:20px;font-size:0.85rem;color:#6B7280;margin-bottom:4px;">'
            f'<span><strong style="color:#374151;">{_n_questions}</strong> questions</span>'
            f'<span><strong style="color:#374151;">{_n_scales}</strong> scales</span>'
            f'<span><strong style="color:#374151;">{_n_candidates}</strong> conditions</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if _n_questions > 0 and not warnings:
            st.caption(f"QSF parsed successfully. Use the **Continue** button at the top to proceed.")

        errors = getattr(preview, "validation_errors", []) or []

        if errors:
            with st.expander("Show QSF errors"):
                for err in errors:
                    st.error(err)

        if warnings:
            with st.expander("Show QSF warnings", expanded=True):
                # Parse warnings for actionable display
                parsed_warnings = _parse_warnings_for_display(warnings)

                # Check if there are scale point warnings
                scale_warnings = [w for w in parsed_warnings if w["type"] == "scale_points"]

                if scale_warnings:
                    st.markdown("**Scale Point Clarification** - Please confirm scale points for these questions:")
                    st.caption("The QSF file should contain this information. If these are showing, the question format may be unusual.")

                    # Initialize scale_point_overrides in session state if not exists
                    if "scale_point_overrides" not in st.session_state:
                        st.session_state["scale_point_overrides"] = {}

                    # Show each question with individual scale point setting
                    for sw in scale_warnings:
                        q_id = sw.get("question_id", "Unknown")
                        if q_id and q_id != "Unknown":
                            col_q, col_scale = st.columns([3, 1])
                            with col_q:
                                st.text(f"{q_id}")
                            with col_scale:
                                current_val = st.session_state["scale_point_overrides"].get(q_id, 7)
                                new_val = st.selectbox(
                                    "Points",
                                    options=[2, 3, 4, 5, 6, 7, 9, 10, 11],
                                    index=[2, 3, 4, 5, 6, 7, 9, 10, 11].index(current_val) if current_val in [2, 3, 4, 5, 6, 7, 9, 10, 11] else 3,
                                    key=f"scale_{q_id}",
                                    label_visibility="collapsed",
                                )
                                st.session_state["scale_point_overrides"][q_id] = new_val

                # Show other warnings
                other_warnings = [w for w in parsed_warnings if w["type"] not in ["scale_points"]]
                for w in other_warnings:
                    st.warning(w["message"])
                    if w.get("fix_suggestion"):
                        st.caption(f"Suggested fix: {w['fix_suggestion']}")

        if selected_conditions:
            st.success(f"✓ Conditions: {', '.join(selected_conditions)}")

    # v1.0.3.6: Back to top — ALWAYS present at bottom of every page
    st.markdown("---")
    st.markdown(
        '<a href="#btt-anchor" '
        'onclick="var el=document.getElementById(\'btt-anchor\');'
        'if(el){el.scrollIntoView({behavior:\'smooth\',block:\'start\'});}return false;" '
        'class="btt-link">\u2191 Back to top</a>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 3: DESIGN CONFIGURATION
# =====================================================================
if active_page == 2:
    # v1.8.4: Extra scroll-to-top for the heavy Design page (widgets render async)
    if st.session_state.pop("_page_just_changed_design", None):
        _inject_scroll_to_top_js()
    st.markdown(
        '<div class="section-guide">'
        '<strong>Step 3 &middot; Design</strong> &mdash; '
        'Configure conditions, DVs, and sample size.</div>',
        unsafe_allow_html=True,
    )
    # v1.0.3.7: Nav + checklist use st.container() placeholders at top,
    # filled at bottom of page with CURRENT session_state values.
    # Fixes stale-data bug: checkboxes below set values AFTER the top renders,
    # so reading session_state at the top showed PREVIOUS rerun's values.
    _nav_placeholder = st.container()
    _checklist_placeholder = st.container()

    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)
    enhanced_analysis: Optional[DesignAnalysisResult] = st.session_state.get("enhanced_analysis", None)

    _builder_complete = st.session_state.get("conversational_builder_complete", False)

    # Determine which path to show (builder vs QSF)
    # CRITICAL: Do NOT use st.stop() here — it would kill the Generate tab too
    _skip_qsf_design = False

    if not preview and not _builder_complete:
        # Tailor the message to the user's chosen input method
        _user_input_mode = st.session_state.get("study_input_mode", "upload_qsf")
        if _user_input_mode == "describe_study":
            st.warning("Complete **Study Input** first \u2014 describe your study, then click **Build Study Specification**.")
        else:
            st.warning("Complete **Study Input** first \u2014 upload a QSF file or describe your study.")
        _skip_qsf_design = True
    elif _builder_complete and not preview:
        # Conversational builder path — show review mode, skip QSF config
        _render_builder_design_review()
        _skip_qsf_design = True

    if not _skip_qsf_design:
        # If we reach here, preview exists (QSF path)
        inferred = st.session_state.get("inferred_design", {})

        # v1.7.0: Compact status at top — no redundant banner
        _design_conds = st.session_state.get("selected_conditions", [])
        _design_scales = st.session_state.get("confirmed_scales", [])
        _design_n = st.session_state.get("sample_size", 0)
        _design_ready = bool(_design_conds) and len(_design_conds) >= 1 and bool(_design_scales) and st.session_state.get("scales_confirmed", False)

        # v1.0.3.0: Simplified progress — only show if truly missing major items
        if not _design_ready and not _design_conds and not _design_scales:
            st.info("Select conditions and configure DVs below to complete this step.")

        # ── Conditions ─────────────────────────────────────────────────
        st.markdown("#### Experimental Conditions")

        # Condition selection
        condition_candidates = st.session_state.get("condition_candidates")
        if not condition_candidates:
            condition_candidates = _get_condition_candidates(preview, enhanced_analysis)
            st.session_state["condition_candidates"] = condition_candidates

        all_possible_conditions = condition_candidates or []

        # Get all QSF identifiers for the dropdown option
        qsf_identifiers = st.session_state.get("qsf_identifiers")
        if qsf_identifiers is None:
            qsf_identifiers = _get_qsf_identifiers(preview)
            st.session_state["qsf_identifiers"] = qsf_identifiers

        # Initialize selected conditions in session state
        if "selected_conditions" not in st.session_state:
            st.session_state["selected_conditions"] = condition_candidates[:] if condition_candidates else []

        # Get current custom conditions
        custom_conditions = st.session_state.get("custom_conditions", [])

        if all_possible_conditions:

            # Multi-select from available options
            selected = st.multiselect(
                "Auto-detected conditions",
                options=all_possible_conditions,
                default=st.session_state.get("selected_conditions", []),
                help="Pick the block names that correspond to experimental conditions.",
                key="condition_multiselect",
            )
            st.session_state["selected_conditions"] = selected
        else:
            st.caption("No conditions auto-detected. Add conditions below.")
            selected = []
            st.session_state["selected_conditions"] = []

        # Additional ways to add conditions
        with st.expander("Add more conditions", expanded=not bool(all_possible_conditions) or not bool(selected)):

            if qsf_identifiers:
                st.markdown("**From QSF identifiers**")

                col_id1, col_id2 = st.columns([3, 1])
                with col_id1:
                    id_to_add = st.selectbox(
                        "Select identifier",
                        options=[""] + qsf_identifiers,
                        index=0,
                        key="qsf_identifier_select",
                        help="Select an identifier from your QSF to add as a condition.",
                    )
                with col_id2:
                    if st.button("Add →", key="add_identifier_btn", disabled=not id_to_add):
                        # Clean up the identifier (remove [Block] or [Randomizer] prefix if present)
                        clean_id = id_to_add
                        if clean_id.startswith("[Block] "):
                            clean_id = clean_id[8:]
                        elif clean_id.startswith("[Randomizer] "):
                            clean_id = clean_id[13:]

                        if clean_id and clean_id not in custom_conditions:
                            custom_conditions.append(clean_id)
                            st.session_state["custom_conditions"] = custom_conditions
                            st.rerun()

            st.markdown("**Type manually**")

            # v1.0.6.1: Use versioned keys so the field resets after adding
            _qsf_cond_ver = st.session_state.get("_qsf_cond_version", 0)
            col_add1, col_add2 = st.columns([3, 1])
            with col_add1:
                new_condition = st.text_input(
                    "New condition name",
                    key=f"new_condition_input_v{_qsf_cond_ver}",
                    placeholder="e.g., Control, Treatment, High, Low",
                    help="Type any condition name you want to add.",
                )
            with col_add2:
                if st.button("Add →", key=f"add_condition_btn_v{_qsf_cond_ver}", disabled=not new_condition.strip()):
                    _nc = new_condition.strip()
                    # v1.8.9: Case-insensitive duplicate check across all sources
                    _all_existing = [c.lower() for c in custom_conditions + selected]
                    if _nc and _nc.lower() not in _all_existing:
                        custom_conditions.append(_nc)
                        st.session_state["custom_conditions"] = custom_conditions
                        # Increment version to create fresh widget on next rerun
                        st.session_state["_qsf_cond_version"] = _qsf_cond_ver + 1
                        st.rerun()
                    elif _nc and _nc.lower() in _all_existing:
                        st.warning(f"Condition '{_nc}' already exists (case-insensitive match).")

            # Show custom conditions with remove buttons
            if custom_conditions:
                st.markdown("---")
                st.markdown("**Conditions added:**")
                for i, cc in enumerate(custom_conditions):
                    col_cc, col_rm = st.columns([4, 1])
                    with col_cc:
                        st.text(f"• {cc}")
                    with col_rm:
                        # Use condition name in key for stability when list changes
                        safe_key = cc.replace(" ", "_").replace(".", "_")[:20]
                        if st.button("✕", key=f"rm_custom_{safe_key}_{i}"):
                            # Defensive: check if item still exists before removing
                            if cc in custom_conditions:
                                custom_conditions.remove(cc)
                                st.session_state["custom_conditions"] = custom_conditions
                            st.rerun()

        # Combine selected and custom conditions
        # Safety: ensure `selected` is always defined even if neither branch above ran
        if 'selected' not in dir():
            selected = st.session_state.get("selected_conditions", [])
        custom_conditions = st.session_state.get("custom_conditions", [])
        all_conditions = list(dict.fromkeys(selected + custom_conditions))

        # Show final conditions summary
        if all_conditions:
            clean_names = [_clean_condition_name(c) for c in all_conditions]
            st.caption(f"{len(all_conditions)} condition(s): {', '.join(clean_names)}")
        else:
            st.error("No conditions defined. Select or add at least one condition above.")
            all_conditions = []

        # v1.0.5.1: Condition descriptions (optional, improves simulation realism)
        if all_conditions and len(all_conditions) >= 2:
            if "builder_condition_descriptions" not in st.session_state:
                st.session_state["builder_condition_descriptions"] = {}
            _qsf_cond_descs = st.session_state["builder_condition_descriptions"]
            with st.expander("Describe your conditions (recommended for better simulation)", expanded=False):
                st.caption(
                    "Briefly describe what participants experience in each condition. "
                    "This helps the simulator choose appropriate behavioral patterns and effect sizes."
                )
                for _ci, _cond in enumerate(all_conditions):
                    _c_clean = _clean_condition_name(_cond)
                    _existing_d = _qsf_cond_descs.get(_cond, "")
                    _new_d = st.text_input(
                        f"{_c_clean}",
                        value=_existing_d,
                        key=f"qsf_cond_desc_{_ci}",
                        placeholder="e.g., Participants read a positive framing of the product",
                    )
                    if _new_d != _existing_d:
                        _qsf_cond_descs[_cond] = _new_d
                        st.session_state["builder_condition_descriptions"] = _qsf_cond_descs

        # ── Design Structure ────────────────────────────────────────────
        st.markdown("#### Design Structure")
        col_design1, col_design2 = st.columns(2)

        # Auto-detect factors from condition names
        auto_detected_factors = _infer_factors_from_conditions(all_conditions)
        auto_num_factors = len(auto_detected_factors)

        with col_design1:
            # Design type selection
            design_type = st.selectbox(
                "Experimental design type",
                options=[
                    "Between-subjects (each participant sees one condition)",
                    "Within-subjects (each participant sees all conditions)",
                    "Mixed design",
                    "Simple comparison (2 groups)",
                ],
                index=0,
                key="design_type_select",
                help="How are conditions assigned to participants? Most experiments use between-subjects designs.",
            )

            # Show auto-detected design info
            if auto_num_factors > 1:
                factor_levels_str = " × ".join([str(len(f.get("levels", []))) for f in auto_detected_factors])
                st.success(f"Auto-detected: {factor_levels_str} factorial design ({auto_num_factors} factors)")
            elif len(all_conditions) > 1:
                st.info(f"Detected: {len(all_conditions)}-condition single-factor design")

        with col_design2:
            # Randomization level
            rand_level = st.selectbox(
                "Randomization level",
                options=[
                    "Participant-level (standard)",
                    "Group/Cluster-level",
                    "Not randomized / observational",
                ],
                index=0,
                key="rand_level_select",
                help="How participants are assigned to conditions.",
            )

        # ── Sample Size & Allocation ────────────────────────────────────
        st.markdown("#### Sample Size")
        default_sample_size = int(st.session_state.get("sample_size", 200))
        sample_size = st.number_input(
            "Target sample size (N)",
            min_value=10,
            max_value=MAX_SIMULATED_N,
            value=default_sample_size,
            step=10,
            help=f"From your power analysis. Max: {MAX_SIMULATED_N:,}.",
            key="sample_size_step3",
        )
        st.session_state["sample_size"] = int(sample_size)

        if all_conditions and len(all_conditions) > 1:

            n_conditions = len(all_conditions)
            prev_sample_size = st.session_state.get("_prev_sample_size", 0)
            prev_n_conditions = st.session_state.get("_prev_n_conditions", 0)
            prev_conditions = st.session_state.get("_prev_conditions", [])

            # Detect what changed
            conditions_changed = (
                n_conditions != prev_n_conditions or
                set(prev_conditions) != set(all_conditions)
            )
            sample_size_changed = prev_sample_size != sample_size and prev_sample_size > 0

            # Recalculate allocations if:
            # 1. No allocation exists, OR
            # 2. Number of conditions changed, OR
            # 3. Condition names changed
            needs_recalc = (
                "condition_allocation_n" not in st.session_state or
                conditions_changed
            )

            if needs_recalc:
                # Fresh allocation: equal distribution for current sample size
                n_per = sample_size // n_conditions
                remainder = sample_size % n_conditions
                st.session_state["condition_allocation_n"] = {
                    cond: n_per + (1 if i < remainder else 0)
                    for i, cond in enumerate(all_conditions)
                }
                # Clear any stale widget state by incrementing a version counter
                alloc_version = st.session_state.get("_alloc_version", 0) + 1
                st.session_state["_alloc_version"] = alloc_version
            elif sample_size_changed:
                # Proportionally adjust existing allocations to match new total
                old_alloc = st.session_state.get("condition_allocation_n", {})
                old_total = sum(old_alloc.values())
                if old_total > 0 and sample_size > 0:
                    scale = sample_size / old_total
                    new_alloc = {}
                    running_total = 0
                    sorted_conds = list(all_conditions)
                    for j, c in enumerate(sorted_conds[:-1]):
                        new_alloc[c] = max(1, round(old_alloc.get(c, sample_size // n_conditions) * scale))
                        running_total += new_alloc[c]
                    # Last condition gets the remainder to ensure sum matches exactly
                    new_alloc[sorted_conds[-1]] = max(1, sample_size - running_total)
                    st.session_state["condition_allocation_n"] = new_alloc
                    # Increment version to force widget refresh
                    alloc_version = st.session_state.get("_alloc_version", 0) + 1
                    st.session_state["_alloc_version"] = alloc_version

            # Update tracking variables
            st.session_state["_prev_sample_size"] = sample_size
            st.session_state["_prev_n_conditions"] = n_conditions
            st.session_state["_prev_conditions"] = list(all_conditions)

            allocation_n = st.session_state["condition_allocation_n"]

            # Create a clean display table with number inputs
            st.markdown("**Participants per condition:**")

            # Use columns for layout (max 3 per row)
            cols_per_row = min(n_conditions, 3)
            input_cols = st.columns(cols_per_row)

            # Get version for unique widget keys
            alloc_version = st.session_state.get("_alloc_version", 0)

            new_allocation_n = {}
            for i, cond in enumerate(all_conditions):
                col_idx = i % cols_per_row
                with input_cols[col_idx]:
                    # Clean display name
                    display_name = _clean_condition_name(cond)
                    full_name = display_name  # Keep full name for tooltip
                    # v1.0.5.1: Wider truncation (35 chars) + full name in help tooltip
                    if len(display_name) > 37:
                        display_name = display_name[:35] + "..."

                    current_n = allocation_n.get(cond, sample_size // n_conditions)
                    # Ensure current_n doesn't exceed sample_size
                    current_n = min(max(1, current_n), sample_size)

                    # Use version in key to force widget refresh when conditions change
                    new_n = st.number_input(
                        display_name,
                        min_value=1,
                        max_value=sample_size,
                        value=current_n,
                        step=1,
                        key=f"alloc_n_v{alloc_version}_{i}",
                        help=f"Participants in: {full_name}"
                    )
                    new_allocation_n[cond] = new_n

            # Calculate total and show status
            total_n = sum(new_allocation_n.values())
            diff = total_n - sample_size

            if diff != 0:
                st.warning(f"⚠️ Allocations sum to **{total_n}** (should be {sample_size}). Difference: {'+' if diff > 0 else ''}{diff}")
                # Auto-balance button
                if st.button("Auto-balance to match total N", key=f"auto_balance_n_btn_v{alloc_version}"):
                    # Scale proportionally to match sample_size
                    scale = sample_size / total_n if total_n > 0 else 1
                    normalized = {}
                    running_total = 0
                    sorted_conds = list(all_conditions)
                    for j, c in enumerate(sorted_conds[:-1]):
                        normalized[c] = max(1, round(new_allocation_n[c] * scale))
                        running_total += normalized[c]
                    # Last condition gets the remainder
                    normalized[sorted_conds[-1]] = max(1, sample_size - running_total)
                    st.session_state["condition_allocation_n"] = normalized
                    # Increment version to force widget refresh
                    st.session_state["_alloc_version"] = alloc_version + 1
                    st.rerun()
            else:
                st.success(f"✓ Allocations sum to {sample_size}")
                st.session_state["condition_allocation_n"] = new_allocation_n
                # v1.8.9: Per-cell-N power warning
                _min_cell_n = min(new_allocation_n.values()) if new_allocation_n else 0
                if _min_cell_n < 20:
                    st.caption(f"⚠️ Smallest cell has {_min_cell_n} participants. Consider at least 20 per cell for adequate statistical power.")

            # Convert to percentage-based allocation for the simulation engine
            st.session_state["condition_allocation"] = {
                cond: (new_allocation_n.get(cond, 0) / sample_size * 100) if sample_size > 0 else 0
                for cond in all_conditions
            }

        elif all_conditions:
            # Single condition - show simple info
            st.info(f"Single condition design: all {sample_size} participants in '{_clean_condition_name(all_conditions[0])}'")
            st.session_state["condition_allocation"] = {all_conditions[0]: 100}
            st.session_state["condition_allocation_n"] = {all_conditions[0]: sample_size}

        st.markdown("#### Factor Structure")

        # Detect design type
        design_options = [
            "Factorial design (2×2, 2×3, 3×3, etc.)",
            "Factorial + control (factorial + 1 control condition)",
            "Simple multi-arm (independent conditions, no factorial)",
        ]

        # Determine default index - use saved value if exists, otherwise auto-detect
        if "design_structure_select" in st.session_state and st.session_state["design_structure_select"] in design_options:
            default_design_idx = design_options.index(st.session_state["design_structure_select"])
        else:
            default_design_idx = 0 if auto_num_factors > 1 else 2

        design_structure = st.selectbox(
            "Design structure",
            options=design_options,
            index=default_design_idx,
            key="design_structure_select",
            help=(
                "• **Factorial**: Conditions are factor level combinations (2×2=4, 2×3=6, 3×3=9, 2×2×2=8)\n"
                "• **Factorial + control**: Factorial design plus a separate control condition\n"
                "• **Simple multi-arm**: Independent conditions without factorial structure"
            ),
        )

        # Show specific factorial design selection when factorial is chosen
        if "Factorial" in design_structure and "multi-arm" not in design_structure:
            # Detect if conditions appear to be from separate randomizers (not already crossed)
            # This happens when conditions don't contain separators like " x ", " × ", " + "
            separators = [" x ", " X ", " × ", " + ", " | "]
            conditions_already_crossed = any(
                any(sep in c for sep in separators)
                for c in all_conditions
            )

            # If conditions are NOT already crossed, use the visual factorial table
            if not conditions_already_crossed and len(all_conditions) >= 2:
                st.caption("Assign your detected conditions to factors to create crossed combinations.")

                # Use the visual factorial design table
                factors, crossed_conditions = _render_factorial_design_table(
                    all_conditions,
                    session_key_prefix="factorial_design"
                )
                st.session_state["use_factorial_table"] = True

                # Update all_conditions to use the crossed conditions
                if crossed_conditions:
                    st.session_state["factorial_crossed_conditions"] = crossed_conditions
                    st.session_state["use_crossed_conditions"] = True

                    # Update condition allocation for crossed conditions
                    n_crossed = len(crossed_conditions)
                    if n_crossed > 0 and sample_size > 0:
                        n_per = sample_size // n_crossed
                        remainder = sample_size % n_crossed
                        st.session_state["condition_allocation"] = {
                            cond: ((n_per + (1 if i < remainder else 0)) / sample_size * 100)
                            for i, cond in enumerate(crossed_conditions)
                        }
                        st.session_state["condition_allocation_n"] = {
                            cond: n_per + (1 if i < remainder else 0)
                            for i, cond in enumerate(crossed_conditions)
                        }
            else:
                # Conditions already have factorial structure - use traditional approach
                st.session_state["use_crossed_conditions"] = False
                st.session_state["use_factorial_table"] = False

            # Only show additional configuration if NOT using factorial table with crossed conditions
            if not st.session_state.get("use_factorial_table") or conditions_already_crossed:
                factorial_designs = [
                    "2×2 (2 factors, 2 levels each = 4 conditions)",
                    "2×3 (2 factors: 2 and 3 levels = 6 conditions)",
                    "3×2 (2 factors: 3 and 2 levels = 6 conditions)",
                    "3×3 (2 factors, 3 levels each = 9 conditions)",
                    "2×2×2 (3 factors, 2 levels each = 8 conditions)",
                    "2×2×3 (3 factors: 2, 2, and 3 levels = 12 conditions)",
                    "3×3×3 (3 factors, 3 levels each = 27 conditions)",
                    "Custom (define factors manually)",
                ]

                # Auto-detect which factorial design matches
                auto_design = "Custom (define factors manually)"
                if auto_num_factors == 2:
                    f1_levels = len(auto_detected_factors[0].get("levels", [])) if len(auto_detected_factors) > 0 else 0
                    f2_levels = len(auto_detected_factors[1].get("levels", [])) if len(auto_detected_factors) > 1 else 0
                    if f1_levels == 2 and f2_levels == 2:
                        auto_design = "2×2 (2 factors, 2 levels each = 4 conditions)"
                    elif (f1_levels == 2 and f2_levels == 3) or (f1_levels == 3 and f2_levels == 2):
                        auto_design = "2×3 (2 factors: 2 and 3 levels = 6 conditions)" if f1_levels == 2 else "3×2 (2 factors: 3 and 2 levels = 6 conditions)"
                    elif f1_levels == 3 and f2_levels == 3:
                        auto_design = "3×3 (2 factors, 3 levels each = 9 conditions)"
                elif auto_num_factors == 3:
                    levels = [len(f.get("levels", [])) for f in auto_detected_factors[:3]]
                    if levels == [2, 2, 2]:
                        auto_design = "2×2×2 (3 factors, 2 levels each = 8 conditions)"
                    elif sorted(levels) == [2, 2, 3]:
                        auto_design = "2×2×3 (3 factors: 2, 2, and 3 levels = 12 conditions)"
                    elif levels == [3, 3, 3]:
                        auto_design = "3×3×3 (3 factors, 3 levels each = 27 conditions)"

                default_factorial_idx = factorial_designs.index(auto_design) if auto_design in factorial_designs else len(factorial_designs) - 1

                selected_factorial = st.selectbox(
                    "Factorial design type",
                    options=factorial_designs,
                    index=default_factorial_idx,
                    key="factorial_design_type",
                    help="Select your specific factorial design.",
                )

        factors = []

        # Check if we already have factors from the factorial table
        use_factorial_table = st.session_state.get("use_factorial_table", False)
        factorial_table_factors = st.session_state.get("factorial_table_factors", [])

        if use_factorial_table and factorial_table_factors:
            # Use factors from the factorial table - no need to show redundant config
            factors = factorial_table_factors
        elif "Factorial" in design_structure and "multi-arm" not in design_structure:
            # Factorial design (with or without control)
            has_control = "control" in design_structure.lower()

            # Show auto-detected factors if available
            if auto_num_factors > 1:
                st.success(f"Auto-detected {auto_num_factors}-factor design from condition names")

                # Show detected factors with editable names
                st.markdown("**Detected factors:**")
                detected_factors_display = []
                for i, f in enumerate(auto_detected_factors):
                    col_name, col_levels = st.columns([1, 2])
                    with col_name:
                        new_name = st.text_input(
                            f"Factor {i+1}",
                            value=f.get("name", f"Factor {i+1}"),
                            key=f"detected_factor_name_{i}",
                            label_visibility="collapsed",
                        )
                    with col_levels:
                        levels = f.get("levels", [])
                        # Clean level names for display and filter out any interaction terms
                        clean_levels = []
                        for l in levels:
                            clean_l = _clean_condition_name(l)
                            # Skip if it looks like an interaction term (contains × or x)
                            if " × " not in clean_l and " x " not in clean_l.lower():
                                clean_levels.append(clean_l)
                        st.markdown(f"**Levels:** {' | '.join(clean_levels)}")
                    detected_factors_display.append({"name": new_name, "levels": [l for l in levels if " × " not in _clean_condition_name(l) and " x " not in _clean_condition_name(l).lower()]})

                factors = detected_factors_display

                # Option to manually adjust
                if st.checkbox("Manually adjust factors", value=False, key="manual_adjust_factors"):
                    factors = _render_manual_factor_config(all_conditions, auto_detected_factors)
            else:
                # No auto-detection - manual configuration
                st.info("Could not auto-detect factorial structure. Please configure manually.")
                factors = _render_manual_factor_config(all_conditions, None)

            # Handle control condition for "Factorial + control"
            if has_control:
                st.markdown("**Control condition:**")

                # Smart control condition detection:
                # 1. Check if any factor has "_non_matching_conditions" (conditions that didn't fit pattern)
                # 2. Look for conditions containing "control"
                # 3. Otherwise, find the condition that doesn't fit the factorial crossing
                control_candidates = []

                # Check for non-matching conditions from factor detection
                for f in auto_detected_factors:
                    non_match = f.get("_non_matching_conditions", [])
                    if non_match:
                        control_candidates.extend(non_match)
                        break

                # If no non-matching found, look for conditions with "control" in name
                if not control_candidates:
                    control_candidates = [c for c in all_conditions if 'control' in _clean_condition_name(c).lower()]

                # If still no candidates, find conditions that don't match the factorial pattern
                if not control_candidates and len(factors) >= 2:
                    # Get all factor levels
                    all_factor_levels = []
                    for f in factors:
                        all_factor_levels.extend([l.lower() for l in f.get("levels", [])])

                    # Find conditions where not all parts match known factor levels
                    for cond in all_conditions:
                        clean_cond = _clean_condition_name(cond).lower()
                        # Check if this condition can be explained by the factorial crossing
                        parts_match = sum(1 for level in all_factor_levels if level in clean_cond)
                        if parts_match < len(factors):  # Doesn't match enough factor levels
                            control_candidates.append(cond)

                # Default to first condition if no candidates found
                if not control_candidates and all_conditions:
                    control_candidates = [all_conditions[0]]

                # Determine default index
                default_ctrl_idx = 0
                if control_candidates and control_candidates[0] in all_conditions:
                    default_ctrl_idx = all_conditions.index(control_candidates[0])

                # Store detected control for display
                if control_candidates:
                    detected_ctrl_name = _clean_condition_name(control_candidates[0])
                    st.caption(f"Auto-detected potential control: **{detected_ctrl_name}**")

                control_cond = st.selectbox(
                    "Select the control condition",
                    options=all_conditions,
                    index=default_ctrl_idx,
                    key="control_condition_select",
                    format_func=_clean_condition_name,
                    help="This condition will be treated separately from the factorial structure",
                )
                st.session_state["control_condition"] = control_cond

        else:
            # Simple multi-arm design - each condition is independent
            st.caption("Each condition is an independent level of a single factor.")
            factors = [{"name": "Condition", "levels": all_conditions}]

        # Ensure we have at least one factor
        if not factors:
            factors = [{"name": "Condition", "levels": all_conditions}]

        # Get scales from inferred design
        scales = inferred.get("scales", []) if inferred else []

        # Filter out empty scales
        scales = [s for s in scales if s.get("name", "").strip()]
        if not scales:
            scales = [{"name": "Main_DV", "num_items": 5, "scale_points": 7}]

        # ── Dependent Variables ──────────────────────────────────────────
        st.markdown("#### Dependent Variables")

        # Initialize DV state - use a version counter to handle deletions
        dv_version = st.session_state.get("_dv_version", 0)

        if "confirmed_scales" not in st.session_state:
            st.session_state["confirmed_scales"] = scales.copy()
            st.session_state["_dv_version"] = 0
        if "scales_confirmed" not in st.session_state:
            st.session_state["scales_confirmed"] = False

        # Get current confirmed DVs
        confirmed_scales = st.session_state.get("confirmed_scales", scales.copy())

        # Type badge colors
        type_badges = {
            'matrix': '🔢 Matrix Scale',
            'numbered_items': '📊 Multi-Item Scale',
            'likert': '📈 Likert Scale',
            'slider': '🎚️ Slider',
            'single_item': '📍 Single Item',
            'numeric_input': '🔣 Numeric Input',
            'constant_sum': '➕ Constant Sum',
        }

        # Display each DV with edit and remove options
        updated_scales = []
        scales_to_remove = []

        if confirmed_scales:
            st.markdown(f"**{len(confirmed_scales)} DV(s) detected.** Review and adjust as needed:")

            # Column headers
            hdr1, hdr2, hdr3a, hdr3b, hdr4, hdr5 = st.columns([2.5, 0.8, 0.6, 0.6, 1.5, 0.4])
            hdr1.markdown("**Name**")
            hdr2.markdown("**Items**")
            hdr3a.markdown("**Min**")
            hdr3b.markdown("**Max**")
            hdr4.markdown("**Type**")
            hdr5.caption("")

            for i, scale in enumerate(confirmed_scales):
                dv_type = scale.get("type", "likert")
                type_badge = type_badges.get(dv_type, "📊 Scale")

                # Get item names and scale info for display
                item_names = scale.get("item_names", [])
                scale_min = scale.get("scale_min", 1)
                scale_max = scale.get("scale_max", scale.get("scale_points", 7))
                scale_anchors = scale.get("scale_anchors", {})

                with st.container():
                    # Main row: Name, # Items, Scale Min, Scale Max, Type, Remove
                    # v1.2.0: Improved layout with separate min/max for flexible scale ranges
                    col1, col2, col3a, col3b, col4, col5 = st.columns([2.5, 0.8, 0.6, 0.6, 1.5, 0.4])

                    # v1.2.0: Use variable_name from QSF as default, fall back to name
                    qsf_var_name = scale.get("variable_name", scale.get("name", f"DV_{i+1}"))
                    display_name = scale.get("name", qsf_var_name)

                    with col1:
                        scale_name = st.text_input(
                            f"DV {i+1} Name",
                            value=qsf_var_name,  # Use exact QSF variable name
                            key=f"dv_name_v{dv_version}_{i}",
                            label_visibility="collapsed",
                            help=f"QSF Variable: {qsf_var_name} | Question: {display_name[:50]}..."
                        )

                    with col2:
                        # Get items value, handle both 'items' and 'num_items' keys
                        items_val = scale.get("items", scale.get("num_items", 1))
                        # v1.2.0: Clearer label based on DV type
                        items_label = "Items" if dv_type in ['matrix', 'numbered_items'] else "Qs"
                        items_help = {
                            'matrix': "Number of items in this matrix/battery (e.g., 5-item attitude scale)",
                            'numbered_items': "Number of numbered items (e.g., Scale_1 through Scale_5)",
                            'likert': "Number of Likert-type questions grouped together",
                            'slider': "Number of slider questions",
                            'single_item': "Single question (always 1)",
                            'numeric_input': "Number of numeric input fields",
                        }.get(dv_type, "Number of questions/items in this scale")
                        num_items = st.number_input(
                            items_label,
                            min_value=1,
                            max_value=50,
                            value=int(items_val) if items_val else 1,
                            key=f"dv_items_v{dv_version}_{i}",
                            help=items_help
                        )

                    # v1.2.0: Two separate inputs for min and max scale values (1-100 range)
                    with col3a:
                        # Get current min, default to 1
                        current_min = int(scale_min) if scale_min is not None else 1
                        if dv_type == 'numeric_input':
                            # Numeric inputs can have any range (incl. negative for games with taking)
                            new_scale_min = st.number_input(
                                "Min",
                                min_value=-1000,
                                max_value=1000,
                                value=current_min,
                                key=f"dv_min_v{dv_version}_{i}",
                                help="Minimum value (e.g., 0 for slider, -10 for games with taking option)"
                            )
                        else:
                            new_scale_min = st.number_input(
                                "Min",
                                min_value=-1000,
                                max_value=100,
                                value=current_min,
                                key=f"dv_min_v{dv_version}_{i}",
                                help="Minimum scale value (e.g., 0 or 1, negative for bipolar scales)"
                            )

                    with col3b:
                        # Get current max from scale_max or scale_points
                        current_max = int(scale_max) if scale_max is not None else (int(scale.get("scale_points", 7)) if scale.get("scale_points") else 7)
                        if dv_type == 'numeric_input':
                            # Numeric inputs can have any range
                            new_scale_max = st.number_input(
                                "Max",
                                min_value=1,
                                max_value=10000,
                                value=current_max,
                                key=f"dv_max_v{dv_version}_{i}",
                                help="Maximum value (e.g., 100 for percentage, 1000 for WTP)"
                            )
                        else:
                            new_scale_max = st.number_input(
                                "Max",
                                min_value=2,
                                max_value=100,
                                value=min(current_max, 100),  # Cap at 100 for display
                                key=f"dv_max_v{dv_version}_{i}",
                                help="Maximum scale value (e.g., 5, 7, 10, 100)"
                            )

                    # v1.8.9: Validate min < max and warn user
                    if new_scale_max <= new_scale_min:
                        st.warning(f"⚠️ Max ({new_scale_max}) must be greater than Min ({new_scale_min}). Adjusting Max to {new_scale_min + 1}.")
                        new_scale_max = new_scale_min + 1

                    # Calculate scale_points from min/max for compatibility
                    scale_points = new_scale_max  # Used for data generation

                    with col4:
                        # Type badge with descriptive tooltip
                        type_descriptions = {
                            'matrix': 'Multi-item scale detected as a matrix (e.g., 5-item attitude battery)',
                            'numbered_items': 'Scale detected from numbered items (e.g., Scale_1, Scale_2)',
                            'likert': 'Likert-type scale with multiple response options',
                            'slider': 'Visual analog scale or slider (typically 0-100)',
                            'single_item': 'Single standalone question',
                            'numeric_input': 'Open-ended numeric input (e.g., WTP, quantity)',
                            'constant_sum': 'Constant sum allocation task',
                        }
                        type_desc = type_descriptions.get(dv_type, 'Scale type')
                        st.markdown(
                            f'<small title="{type_desc}">{type_badge}</small>',
                            unsafe_allow_html=True
                        )
                        if scale.get("detected_from_qsf", True):
                            st.caption("*Auto-detected*")

                    with col5:
                        # Remove button with clear help text
                        if st.button("✕", key=f"rm_dv_v{dv_version}_{i}", help="Remove this DV from the simulation. Click to delete."):
                            scales_to_remove.append(i)

                    # Show scale anchors if available
                    if scale_anchors:
                        anchor_text = " | ".join([f"{k}={v}" for k, v in sorted(scale_anchors.items(), key=lambda x: str(x[0]))])
                        if anchor_text:
                            st.caption(f"📏 *Scale: {anchor_text}*")

                    # Show question text if available
                    q_text = scale.get("question_text", "")
                    if q_text and not item_names:
                        st.caption(f"*\"{q_text[:80]}{'...' if len(q_text) > 80 else ''}\"*")

                    # Show individual items in an expander if there are multiple items
                    if item_names and len(item_names) > 0:
                        with st.expander(f"📋 View {len(item_names)} scale item(s)", expanded=False):
                            for j, item_text in enumerate(item_names, 1):
                                if item_text:
                                    # Truncate long item text
                                    display_text = item_text[:100] + "..." if len(item_text) > 100 else item_text
                                    st.markdown(f"**{j}.** {display_text}")

                if scale_name.strip() and i not in scales_to_remove:
                    updated_scales.append({
                        "name": scale_name.strip(),
                        "variable_name": scale_name.strip(),  # v1.2.0: Use user-entered name as variable name
                        "question_text": scale.get("question_text", ""),
                        "items": num_items,
                        "num_items": num_items,  # Keep both for compatibility
                        "scale_points": scale_points,  # Max value for data generation
                        "type": dv_type,
                        "reverse_items": scale.get("reverse_items", []),
                        "detected_from_qsf": scale.get("detected_from_qsf", True),
                        # Preserve new fields for item display and simulation accuracy
                        "item_names": scale.get("item_names", []),
                        "scale_anchors": scale.get("scale_anchors", {}),
                        # v1.2.0: Save user-specified min/max for accurate simulation
                        "scale_min": new_scale_min,
                        "scale_max": new_scale_max,
                    })
        else:
            st.warning("No DVs auto-detected. Click **+ Add DV** below to define them manually.")

        # Handle removals with visual feedback
        if scales_to_remove:
            _removed_names = [confirmed_scales[i].get("name", f"DV {i+1}") for i in scales_to_remove if i < len(confirmed_scales)]
            # v1.0.6.1: Clean up orphaned widget state from old version to prevent memory bloat
            for _ci in range(max(10, len(confirmed_scales))):
                for _prefix in ("dv_name_v", "dv_items_v", "dv_min_v", "dv_max_v", "dv_type_v"):
                    st.session_state.pop(f"{_prefix}{dv_version}_{_ci}", None)
            st.session_state["confirmed_scales"] = updated_scales
            st.session_state["_dv_version"] = dv_version + 1
            st.session_state["_dv_removal_notice"] = f"Removed: {', '.join(_removed_names)}"
            # v1.0.1.4: Use st.rerun() to avoid scroll-to-top on DV removal
            st.rerun()

        # Show removal notice from previous rerun
        _removal_notice = st.session_state.pop("_dv_removal_notice", None)
        if _removal_notice:
            st.info(_removal_notice)

        add_clicked = st.button("+ Add DV", key=f"add_dv_btn_v{dv_version}", help="Add a new dependent variable manually.")

        if add_clicked:
            new_dv = {
                "name": f"New_DV_{len(confirmed_scales)+1}",
                "variable_name": f"New_DV_{len(confirmed_scales)+1}",
                "question_text": "",
                "items": 1,
                "num_items": 1,
                "scale_points": 7,
                "type": "single_item",
                "reverse_items": [],
                "detected_from_qsf": False,
                "item_names": [],
                "scale_anchors": {},
                "scale_min": 1,
                "scale_max": 7,
            }
            confirmed_scales.append(new_dv)
            st.session_state["confirmed_scales"] = confirmed_scales
            st.session_state["_dv_version"] = dv_version + 1
            # v1.0.1.4: Use st.rerun() to avoid scroll-to-top on DV add
            st.rerun()

        # Update session state with edited DVs
        st.session_state["confirmed_scales"] = updated_scales
        scales = updated_scales if updated_scales else scales

        # v1.0.3.7: Prominent DV confirmation — styled banner + checkbox
        _n_updated = len(updated_scales)
        _confirm_label = f"DVs verified ({_n_updated})" if _n_updated > 0 else "DVs verified"
        # Read widget key for CURRENT value (avoids stale session_state)
        _dv_live = bool(st.session_state.get(
            f"dv_confirm_checkbox_v{dv_version}",
            st.session_state.get("scales_confirmed", False),
        ))
        if not _dv_live:
            st.markdown(
                '<div class="confirm-banner pending">'
                '\u2611\uFE0F Please confirm your DVs / scales to proceed</div>',
                unsafe_allow_html=True,
            )
        scales_confirmed = st.checkbox(
            _confirm_label,
            value=_dv_live,
            key=f"dv_confirm_checkbox_v{dv_version}",
            help="Check once you've verified variable names, items, and scale ranges.",
        )
        if scales_confirmed:
            st.markdown(
                '<div class="confirm-banner done">'
                '\u2705 DVs / scales confirmed</div>',
                unsafe_allow_html=True,
            )
        st.session_state["scales_confirmed"] = scales_confirmed

        # ========================================
        # STEP 5: OPEN-ENDED QUESTIONS
        # ========================================
        st.markdown("")
        _oe_count_display = len(st.session_state.get("confirmed_open_ended", []))
        # v1.0.1.2: Pre-calculate context quality for header + expander label
        _oe_pre_list = st.session_state.get("confirmed_open_ended", [])
        _ctx_pre_count = sum(1 for _oq in _oe_pre_list if _oq.get("question_context", "").strip()) if _oe_pre_list else 0
        _ctx_pre_missing = len(_oe_pre_list) - _ctx_pre_count

        if _oe_count_display > 0:
            _oe_label = f"#### Open-Ended Questions ({_oe_count_display})"
        else:
            _oe_label = "#### Open-Ended Questions"
        st.markdown(_oe_label)

        # v1.0.1.4: Show removal notice OUTSIDE expander so it's visible even when collapsed
        _oe_removal_notice = st.session_state.pop("_oe_removal_notice", None)
        if _oe_removal_notice:
            st.info(_oe_removal_notice)

        with st.expander(
            f"Review & edit open-ended questions ({_oe_count_display}) — {_ctx_pre_count}/{_oe_count_display} have context"
            if _oe_count_display > 0
            else "Add open-ended questions (optional)",
            expanded=False,
        ):

            # Initialize open-ended state
            oe_version = st.session_state.get("_oe_version", 0)

            # Get detected open-ended questions from preview
            detected_open_ended = []
            if preview and hasattr(preview, 'open_ended_details') and preview.open_ended_details:
                detected_open_ended = preview.open_ended_details
            elif preview and hasattr(preview, 'open_ended_questions') and preview.open_ended_questions:
                detected_open_ended = [
                    {"variable_name": q, "question_text": "", "source_type": "detected"}
                    for q in preview.open_ended_questions
                ]

            if "confirmed_open_ended" not in st.session_state:
                st.session_state["confirmed_open_ended"] = detected_open_ended.copy()
                st.session_state["_oe_version"] = 0
            if "open_ended_confirmed" not in st.session_state:
                st.session_state["open_ended_confirmed"] = False

            confirmed_open_ended = st.session_state.get("confirmed_open_ended", detected_open_ended.copy())

            source_badges = {
                'text_entry': '📝 Text Entry', 'essay': '📄 Essay Box',
                'mc_text_entry': '🔘 MC + Text', 'form_field': '📋 Form Field',
                'matrix_text': '🔢 Matrix Text', 'detected': '🔍 Detected',
            }

            updated_open_ended = []
            oe_to_remove = []

            if confirmed_open_ended:
                _ctx_count = sum(1 for _oe in confirmed_open_ended if _oe.get("question_context", "").strip())
                _missing_ctx = len(confirmed_open_ended) - _ctx_count
                st.markdown(f"**{len(confirmed_open_ended)} open-ended question(s):**")
                st.caption("Add context (1-2 sentences) to each question to improve AI response quality.")
                for i, oe in enumerate(confirmed_open_ended):
                    source_type = oe.get("source_type", "detected")
                    source_badge = source_badges.get(source_type, "📝 Text")
                    col1, col2, col3 = st.columns([3, 2, 0.5])
                    with col1:
                        var_name = st.text_input(
                            f"Variable {i+1}",
                            value=oe.get("variable_name", oe.get("name", f"OpenEnded_{i+1}")),
                            key=f"oe_name_v{oe_version}_{i}",
                            label_visibility="collapsed",
                        )
                    with col2:
                        st.markdown(f"<small>{source_badge}</small>", unsafe_allow_html=True)
                    with col3:
                        if st.button("✕", key=f"rm_oe_v{oe_version}_{i}", help="Remove"):
                            _removed_name = oe.get("variable_name", oe.get("name", f"OpenEnded_{i+1}"))
                            st.session_state["_oe_removal_notice"] = f"Removed open-ended question: {_removed_name}"
                            oe_to_remove.append(i)
                    q_text = oe.get("question_text", "")
                    if q_text:
                        st.caption(f"*\"{q_text[:80]}{'...' if len(q_text) > 80 else ''}\"*")
                    # v1.0.1.2: Enhanced context input with auto-suggest and rich guidance
                    _oe_ctx_existing = oe.get("question_context", "")
                    # Auto-generate a suggested context from question text if user hasn't provided one
                    _auto_placeholder = "e.g., 'Participants describe their emotional reaction to the news article about climate policy'"
                    _q_text_raw = oe.get("question_text", "").strip()
                    _study_desc_hint = (st.session_state.get("study_description") or st.session_state.get("_p_study_description", ""))[:80]
                    if _q_text_raw and len(_q_text_raw) > 10:
                        _auto_placeholder = f"e.g., 'Participants respond to: {_q_text_raw[:60]}'"
                        if _study_desc_hint:
                            _auto_placeholder += f" (in a study about {_study_desc_hint[:40]}...)"
                    _ctx_label = f"Context for {var_name or f'Q{i+1}'}"
                    if not _oe_ctx_existing:
                        _ctx_label = f"Add context for {var_name or f'Q{i+1}'} (recommended)"
                    _oe_ctx = st.text_input(
                        _ctx_label,
                        value=_oe_ctx_existing,
                        key=f"oe_ctx_v{oe_version}_{i}",
                        placeholder=_auto_placeholder,
                        help=(
                            "1-2 sentences explaining what this question is really asking. "
                            "Include: (1) who the participant is, (2) what they experienced/read, "
                            "(3) what kind of answer is expected. This is the #1 factor in AI response quality."
                        ),
                        label_visibility="collapsed",
                    )
                    # v1.0.1.7: Minimal context feedback
                    if not _oe_ctx.strip():
                        st.caption("No context yet")

                    if var_name.strip() and i not in oe_to_remove:
                        updated_open_ended.append({
                            "variable_name": var_name.strip(),
                            "name": var_name.strip(),
                            "question_text": oe.get("question_text", ""),
                            "question_context": _oe_ctx.strip(),
                            "source_type": source_type,
                            "force_response": oe.get("force_response", False),
                            "context_type": oe.get("context_type", "general"),
                            "min_chars": oe.get("min_chars"),
                            "block_name": oe.get("block_name", ""),
                        })
            else:
                st.info("No open-ended questions detected. Add any below.")

            if oe_to_remove:
                # v1.0.6.1: Clean up orphaned widget state from old version
                for _ci in range(max(10, len(confirmed_open_ended))):
                    for _prefix in ("oe_name_v", "oe_ctx_v", "oe_type_v"):
                        st.session_state.pop(f"{_prefix}{oe_version}_{_ci}", None)
                st.session_state["confirmed_open_ended"] = updated_open_ended
                st.session_state["_oe_version"] = oe_version + 1
                # v1.0.1.4: Use st.rerun() instead of _navigate_to(2) to avoid
                # closing the expander and scrolling to top on removal
                st.rerun()

            # v1.0.1.4: Action buttons row — Add + Remove All
            _oe_btn_col1, _oe_btn_col2, _oe_spacer = st.columns([1, 1, 2])
            with _oe_btn_col1:
                if st.button("Add Open-Ended Question", key=f"add_oe_btn_v{oe_version}"):
                    new_oe = {
                        "variable_name": f"OpenEnded_{len(confirmed_open_ended)+1}",
                        "name": f"OpenEnded_{len(confirmed_open_ended)+1}",
                        "question_text": "", "question_context": "",
                        "source_type": "manual",
                        "force_response": False, "context_type": "general",
                    }
                    confirmed_open_ended.append(new_oe)
                    st.session_state["confirmed_open_ended"] = confirmed_open_ended
                    st.session_state["_oe_version"] = oe_version + 1
                    st.rerun()
            with _oe_btn_col2:
                if confirmed_open_ended and len(confirmed_open_ended) > 1:
                    if st.button(
                        f"Remove All ({len(confirmed_open_ended)})",
                        key=f"rm_all_oe_v{oe_version}",
                        help="Remove all open-ended questions at once",
                    ):
                        _n_removed = len(confirmed_open_ended)
                        st.session_state["confirmed_open_ended"] = []
                        st.session_state["_oe_version"] = oe_version + 1
                        st.session_state["_oe_removal_notice"] = f"Removed all {_n_removed} open-ended questions."
                        st.rerun()

            # Only update confirmed_open_ended if there were actual changes
            # This prevents silent data loss from rendering edge cases
            if updated_open_ended or not confirmed_open_ended:
                st.session_state["confirmed_open_ended"] = updated_open_ended
            elif confirmed_open_ended and not updated_open_ended:
                # User cleared all variable names — warn but don't silently delete
                st.warning("All open-ended question names are empty. Fill in names or remove them explicitly.")

        # v1.0.6.7: Prominent context recommendation (OUTSIDE expander) for QSF path
        _oe_final = st.session_state.get("confirmed_open_ended", [])
        _oe_version_outer = st.session_state.get("_oe_version", 0)
        if _oe_final:
            _ctx_filled = sum(1 for _oq in _oe_final if _oq.get("question_context", "").strip())
            _ctx_missing = len(_oe_final) - _ctx_filled
            if _ctx_missing > 0:
                st.markdown(
                    '<div style="background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;'
                    'padding:10px 14px;margin:6px 0 10px 0;">'
                    '<span style="color:#92400E;font-weight:600;">Tip: Add context to your open-ended questions</span><br>'
                    '<span style="color:#92400E;font-size:0.9em;">'
                    f'{_ctx_missing} of {len(_oe_final)} open-ended question(s) have no context yet. '
                    'Expand the section above and add 1\u20132 sentences per question explaining what the '
                    'participant experienced and what kind of response you expect. '
                    'This is the single biggest factor in getting realistic AI-generated open-text responses.'
                    '</span></div>',
                    unsafe_allow_html=True,
                )
            elif _ctx_filled == len(_oe_final):
                st.caption(f"All {len(_oe_final)} open-ended question(s) have context \u2014 great for AI response quality")
        # v1.0.3.7: Prominent OE confirmation — styled banner + checkbox
        if _oe_final:
            _oe_live = bool(st.session_state.get(
                f"oe_confirm_checkbox_v{_oe_version_outer}",
                st.session_state.get("open_ended_confirmed", False),
            ))
            if not _oe_live:
                st.markdown(
                    '<div class="confirm-banner pending">'
                    '\u2611\uFE0F Please confirm your open-ended questions</div>',
                    unsafe_allow_html=True,
                )
            open_ended_confirmed = st.checkbox(
                f"Open-ended questions verified ({len(_oe_final)})",
                value=_oe_live,
                key=f"oe_confirm_checkbox_v{_oe_version_outer}",
            )
            if open_ended_confirmed:
                st.markdown(
                    '<div class="confirm-banner done">'
                    '\u2705 Open-ended questions confirmed</div>',
                    unsafe_allow_html=True,
                )
            st.session_state["open_ended_confirmed"] = open_ended_confirmed
        else:
            st.session_state["open_ended_confirmed"] = True  # Nothing to confirm

        # ── Attention & Manipulation Checks (collapsed) ──────────────
        preview = st.session_state.get("qsf_preview")
        detected_attention = preview.attention_checks if preview and hasattr(preview, 'attention_checks') else []
        detected_manipulation = []
        enhanced_analysis = st.session_state.get("enhanced_analysis")
        if enhanced_analysis and hasattr(enhanced_analysis, 'manipulation_checks'):
            detected_manipulation = enhanced_analysis.manipulation_checks

        _attn_count = len(detected_attention)
        _manip_count = len(detected_manipulation)
        _checks_label = f"Attention & manipulation checks ({_attn_count + _manip_count} detected)" if (_attn_count + _manip_count) > 0 else "Attention & manipulation checks"
        with st.expander(_checks_label, expanded=False):
            col_attn, col_manip = st.columns(2)
            with col_attn:
                st.markdown("**Attention Checks**")
                if detected_attention:
                    for check in detected_attention[:5]:
                        st.caption(f"\u2713 {check[:50]}{'...' if len(str(check)) > 50 else ''}")
                else:
                    st.caption("None detected")
                manual_attention = st.text_area(
                    "Add attention checks (one per line)",
                    key="manual_attention_checks",
                    height=80,
                    placeholder="Q15_Attention"
                )
            with col_manip:
                st.markdown("**Manipulation Checks**")
                if detected_manipulation:
                    for check in detected_manipulation[:5]:
                        st.caption(f"\u2713 {check[:50]}{'...' if len(str(check)) > 50 else ''}")
                else:
                    st.caption("None detected (optional)")
                manual_manipulation = st.text_area(
                    "Add manipulation checks (one per line)",
                    key="manual_manipulation_checks",
                    height=80,
                    placeholder="Q20_ManipCheck"
                )

        # ── Design Summary ──────────────────────────────────────────────
        st.markdown("")
        st.markdown("#### Summary")
        display_conditions = all_conditions
        if st.session_state.get("use_crossed_conditions") and st.session_state.get("factorial_crossed_conditions"):
            display_conditions = st.session_state["factorial_crossed_conditions"]

        confirmed_oe = st.session_state.get("confirmed_open_ended", [])
        oe_count = len(confirmed_oe)

        # v1.0.2.3: Compact inline summary (consistent with page 4 styling)
        _summary_n = st.session_state.get("sample_size", 0)
        st.markdown(
            f'<div style="display:flex;gap:20px;font-size:0.85rem;color:#6B7280;margin-bottom:4px;">'
            f'<span><strong style="color:#374151;">{len(display_conditions)}</strong> conditions</span>'
            f'<span><strong style="color:#374151;">{len(scales)}</strong> DVs</span>'
            f'<span><strong style="color:#374151;">{_summary_n}</strong> participants</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Validate and lock design - require conditions, scales, scale confirmation, and correct allocation
        _alloc_n = st.session_state.get("condition_allocation_n", {})
        _alloc_sum = sum(_alloc_n.values()) if _alloc_n else 0
        _alloc_ok = not _alloc_n or _alloc_sum == sample_size  # OK if no custom allocation or if sums match
        # v1.0.1.6: Also require OE confirmation when open-ended questions exist
        _oe_confirmed_ok = st.session_state.get("open_ended_confirmed", True)
        design_valid = len(display_conditions) >= 1 and len(scales) >= 1 and scales_confirmed and _alloc_ok and _oe_confirmed_ok

        if design_valid:
            # Save to session state
            # If using factorial crossed conditions from the visual table, use those instead
            if st.session_state.get("use_crossed_conditions") and st.session_state.get("factorial_crossed_conditions"):
                final_conditions = st.session_state["factorial_crossed_conditions"]
                # Update condition allocation for the crossed conditions
                n_crossed = len(final_conditions)
                if n_crossed > 0 and sample_size > 0 and "condition_allocation" not in st.session_state:
                    n_per = sample_size // n_crossed
                    remainder = sample_size % n_crossed
                    st.session_state["condition_allocation"] = {
                        cond: ((n_per + (1 if i < remainder else 0)) / sample_size * 100)
                        for i, cond in enumerate(final_conditions)
                    }
            else:
                final_conditions = all_conditions
            final_factors = _normalize_factor_specs(factors, final_conditions)
            final_scales = _normalize_scale_specs(scales)
            # Use user-confirmed open-ended questions instead of auto-detected
            final_open_ended = st.session_state.get("confirmed_open_ended", inferred.get("open_ended_questions", []))

            # Determine randomization level string
            rand_mapping = {
                "Participant-level (standard)": "Participant-level",
                "Group/Cluster-level": "Group/Cluster-level",
                "Not randomized / observational": "Not randomized",
            }
            final_rand_level = rand_mapping.get(rand_level, "Participant-level")

            # v1.0.0: Get pre-computed visibility map from QSF parser for accurate simulation
            qsf_preview = st.session_state.get("qsf_preview")
            visibility_map = {}
            if qsf_preview and hasattr(qsf_preview, 'condition_visibility_map'):
                visibility_map = qsf_preview.condition_visibility_map or {}

            st.session_state["inferred_design"] = {
                "conditions": final_conditions,
                "factors": final_factors,
                "scales": final_scales,
                "open_ended_questions": final_open_ended,
                "randomization_level": final_rand_level,
                "condition_visibility_map": visibility_map,  # v1.0.0: For accurate condition-specific simulation
            }
            st.session_state["randomization_level"] = final_rand_level

            # Initialize variable review rows if not already set
            if not st.session_state.get("variable_review_rows"):
                prereg_outcomes = st.session_state.get("prereg_outcomes", "")
                prereg_iv = st.session_state.get("prereg_iv", "")
                st.session_state["variable_review_rows"] = _build_variable_review_rows(
                    st.session_state["inferred_design"], prereg_outcomes, prereg_iv, enhanced_analysis
                )

            # ── v1.8.8.0: Auto-infer cross-DV correlation matrix ──────────
            if _HAS_CORRELATION_MODULE and len(final_scales) > 1:
                try:
                    _construct_types = detect_construct_types(final_scales)
                    _corr_matrix, _corr_names = infer_correlation_matrix(
                        final_scales, _construct_types
                    )
                    st.session_state["_auto_corr_matrix"] = _corr_matrix
                    st.session_state["_auto_corr_names"] = _corr_names
                    st.session_state["_auto_construct_types"] = _construct_types

                    # Store in inferred_design for engine use
                    st.session_state["inferred_design"]["correlation_matrix"] = _corr_matrix.tolist()
                    st.session_state["inferred_design"]["construct_types"] = _construct_types

                    # Default mode: small info badge
                    if not st.session_state.get("advanced_mode", False):
                        _n_pairs = len(final_scales) * (len(final_scales) - 1) // 2
                        _n_typed = sum(1 for v in _construct_types.values() if v != "general")
                        st.caption(
                            f"Inter-scale correlations: auto-inferred from {_n_typed}/{len(final_scales)} "
                            f"recognized construct types ({_n_pairs} pairs)"
                        )

                    # Advanced mode: show editable correlation matrix
                    if st.session_state.get("advanced_mode", False):
                        with st.expander("Inter-scale correlation matrix (advanced)", expanded=False):
                            st.caption(
                                "Auto-inferred from empirical meta-analytic data. "
                                "Edit values to customize (matrix will be automatically validated)."
                            )
                            # Show construct type badges inline
                            _badge_parts = []
                            for _sn, _ct in _construct_types.items():
                                if _ct != "general":
                                    _badge_parts.append(
                                        f"<span style='background:#E8F4FD;padding:2px 8px;border-radius:4px;"
                                        f"font-size:0.78rem;margin-right:4px;display:inline-block;"
                                        f"margin-bottom:4px;'>"
                                        f"<b>{_sn}</b> → {_ct.replace('_', ' ')}</span>"
                                    )
                            if _badge_parts:
                                st.markdown(
                                    "<div style='line-height:1.8;margin-bottom:8px;'>"
                                    + " ".join(_badge_parts) + "</div>",
                                    unsafe_allow_html=True,
                                )

                            # Editable correlation matrix
                            _corr_df = pd.DataFrame(
                                _corr_matrix,
                                index=_corr_names,
                                columns=_corr_names,
                            )
                            _edited_corr = st.data_editor(
                                _corr_df,
                                use_container_width=True,
                                key="corr_matrix_editor",
                                height=min(400, 50 + len(_corr_names) * 35),
                            )
                            # Save user edits
                            _user_corr = _edited_corr.values
                            # Symmetrize (take lower triangle)
                            _user_corr = np.tril(_user_corr) + np.tril(_user_corr, -1).T
                            np.fill_diagonal(_user_corr, 1.0)

                            # Validate and auto-fix if needed
                            _needs_fix = False
                            try:
                                np.linalg.cholesky(_user_corr)
                            except np.linalg.LinAlgError:
                                _needs_fix = True
                            if np.any(np.abs(_user_corr) > 1.0 + 1e-8):
                                _needs_fix = True

                            if _needs_fix:
                                from utils.correlation_matrix import nearest_positive_definite
                                _user_corr = nearest_positive_definite(_user_corr)
                                st.info(
                                    "Matrix was adjusted to ensure positive-definiteness "
                                    "(required for valid simulation). Extreme values were moderated."
                                )

                            st.session_state["inferred_design"]["correlation_matrix"] = _user_corr.tolist()
                except Exception as _corr_err:
                    _log(f"Correlation matrix inference failed: {_corr_err}", level="warning")

            # ── v1.8.8.0: Missing data settings ──────────────────────────
            # Default: automatic realistic missingness (no UI controls)
            # Advanced: configurable sliders
            _missing_rate = 0.0  # Default: 0% — DVs are forced-response
            _dropout_rate = 0.0  # Default: 0% — no survey dropout
            if st.session_state.get("advanced_mode", False):
                with st.expander("Data realism settings (advanced)", expanded=False):
                    st.caption(
                        "Simulate realistic missing data patterns found in online surveys "
                        "(MTurk, Prolific). Persona-driven: careless responders skip more items."
                    )
                    _c_miss1, _c_miss2 = st.columns(2)
                    with _c_miss1:
                        _missing_rate = st.slider(
                            "Item-level missingness",
                            min_value=0.0, max_value=0.15, value=0.03, step=0.01,
                            key="missing_data_rate",
                            help="Probability that any individual item is missing (3% typical for online surveys)",
                        )
                    with _c_miss2:
                        _dropout_rate = st.slider(
                            "Survey dropout rate",
                            min_value=0.0, max_value=0.20, value=0.05, step=0.01,
                            key="dropout_rate",
                            help="Proportion of participants who abandon the survey mid-way (5-7% typical)",
                        )
            st.session_state["inferred_design"]["missing_data_rate"] = _missing_rate
            st.session_state["inferred_design"]["dropout_rate"] = _dropout_rate

            st.caption("Design ready. Use the **Continue** button at the top to proceed.")
        else:
            # v1.0.3.0: Concise missing-item notice
            missing_bits = []
            if not all_conditions:
                missing_bits.append("conditions")
            if not scales:
                missing_bits.append("DVs")
            if not scales_confirmed:
                missing_bits.append("DV confirmation")
            if not _alloc_ok:
                missing_bits.append("sample allocation")
            if not _oe_confirmed_ok:
                missing_bits.append("open-ended confirmation")
            if missing_bits:
                st.warning(f"Still needed: {', '.join(missing_bits)}")

        # v1.7.0: Variable review — hidden behind Advanced toggle
        if st.session_state.get("advanced_mode", False):
            with st.expander("Variable roles (advanced)"):
                prereg_outcomes = st.session_state.get("prereg_outcomes", "")
                prereg_iv = st.session_state.get("prereg_iv", "")
                default_rows = _build_variable_review_rows(inferred, prereg_outcomes, prereg_iv, enhanced_analysis)
                current_rows = st.session_state.get("variable_review_rows")
                if not current_rows:
                    current_rows = default_rows
                filtered_rows = [r for r in current_rows if r.get("Type") != "Timing/Meta"]
                if filtered_rows:
                    variable_df = st.data_editor(
                        pd.DataFrame(filtered_rows),
                        num_rows="fixed",
                        use_container_width=True,
                        column_config={
                            "Variable": st.column_config.TextColumn("Variable", disabled=True),
                            "Display Name": st.column_config.TextColumn("Name", disabled=True),
                            "Type": st.column_config.TextColumn("Type", disabled=True),
                            "Role": st.column_config.SelectboxColumn(
                                "Role",
                                options=["Primary outcome", "Secondary outcome", "Condition", "Mediator",
                                         "Moderator", "Attention check", "Demographics", "Timing/Meta", "Other"],
                            ),
                            "Question Text": st.column_config.TextColumn("Question", disabled=True),
                        },
                        key="adv_variable_editor",
                        height=300,
                    )
                    st.session_state["variable_review_rows"] = variable_df.to_dict(orient="records")

    # v1.0.3.7: Fill nav + checklist placeholders with CURRENT values
    # (all widgets above have now executed and set their session_state values)
    _builder_active = (
        st.session_state.get("conversational_builder_complete", False)
        and not st.session_state.get("qsf_preview")
    )
    if _builder_active:
        # Builder path: readiness based on inferred_design content
        _b_design = st.session_state.get("inferred_design", {})
        _b_conds = _b_design.get("conditions", []) if isinstance(_b_design, dict) else []
        _b_scales = _b_design.get("scales", []) if isinstance(_b_design, dict) else []
        _b_oe = _b_design.get("open_ended_questions", []) if isinstance(_b_design, dict) else []
        _b_oe_ctx_ok = st.session_state.get("_builder_oe_context_complete", True) if _b_oe else True
        _p3_ready = len(_b_conds) >= 2 and len(_b_scales) >= 1 and _b_oe_ctx_ok
        _chk_items_final: list = [
            (len(_b_conds) >= 2, f"Conditions ({len(_b_conds)})"),
            (len(_b_scales) >= 1, f"DVs / scales ({len(_b_scales)})"),
        ]
        if _b_oe and not _b_oe_ctx_ok:
            _chk_items_final.append((False, "OE context needed"))
    else:
        # QSF path: readiness based on checkboxes and design state
        _chk_has_conds = bool(
            st.session_state.get("selected_conditions")
            or st.session_state.get("custom_conditions")
        )
        _chk_design_ok = bool(st.session_state.get("inferred_design"))
        _chk_dvs_ok = bool(st.session_state.get("scales_confirmed", False))
        _chk_oe_ok = bool(st.session_state.get("open_ended_confirmed", True))
        _p3_ready = _chk_has_conds and _chk_design_ok and _chk_dvs_ok and _chk_oe_ok
        _chk_items_final = [
            (_chk_has_conds, "Conditions defined"),
            (_chk_design_ok, "Design configured"),
            (_chk_dvs_ok, "DVs / scales confirmed"),
        ]
        if not _chk_oe_ok:
            _chk_items_final.append((False, "Open-ended confirmed"))

    with _nav_placeholder:
        _nav2_left, _nav2_right = st.columns([1, 1])
        with _nav2_left:
            if st.button("\u2190 Back to Study Input", key="nav_back_2"):
                _navigate_to(1)
        with _nav2_right:
            if _p3_ready:
                if st.button("Continue to Generate \u2192", key="nav_next_2", type="primary"):
                    _navigate_to(3)

    with _checklist_placeholder:
        _cl_html = '<div class="design-checklist">'
        for _d, _l in _chk_items_final:
            if _d:
                _cl_html += f'<div class="design-checklist-item done">\u2705 {_l}</div>'
            else:
                _cl_html += f'<div class="design-checklist-item pending">\u26A0\uFE0F {_l}</div>'
        _cl_html += '</div>'
        st.markdown(_cl_html, unsafe_allow_html=True)

    # v1.0.3.6: Back to top — ALWAYS present at bottom of every page
    st.markdown("---")
    st.markdown(
        '<a href="#btt-anchor" '
        'onclick="var el=document.getElementById(\'btt-anchor\');'
        'if(el){el.scrollIntoView({behavior:\'smooth\',block:\'start\'});}return false;" '
        'class="btt-link">\u2191 Back to top</a>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 4: GENERATE SIMULATION
# =====================================================================
if active_page == 3:
    st.markdown(
        '<div class="section-guide">'
        '<strong>Step 4 &middot; Generate</strong> &mdash; '
        'Set difficulty, preview data, and generate your dataset.</div>',
        unsafe_allow_html=True,
    )
    # v1.0.3.7: Back button at top (no Continue — this is the last step)
    if st.button("\u2190 Back to Design", key="nav_back_3"):
        _navigate_to(2)
    inferred = st.session_state.get("inferred_design", None)
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    # v1.4.16: Read generation state FIRST — these variables are used throughout
    # the entire Generate page (difficulty selector, design summary, generate button).
    # Previously _has_generated was referenced at the difficulty selector before
    # being assigned, causing a NameError on fresh sessions.
    if "generation_requested" not in st.session_state:
        st.session_state["generation_requested"] = False
    _is_generating = st.session_state.get("is_generating", False)
    _has_generated = st.session_state.get("has_generated", False)

    # Pre-validation
    completion = _get_step_completion()
    all_required_complete = (
        completion["study_title"] and
        completion["study_description"] and
        completion["sample_size"] and
        completion["qsf_uploaded"] and
        completion["conditions_set"]
    )

    if not inferred:
        st.warning("No experiment design configured. Complete **Study Input** and **Design** first.")
        st.stop()

    # v1.8.9: Readiness checklist — compact inline, includes DVs
    _gen_mode = st.session_state.get("study_input_mode", "upload_qsf")
    _input_label = "Study described" if _gen_mode == "describe_study" else "QSF uploaded"
    _has_dvs = bool(st.session_state.get("confirmed_scales")) or bool(inferred.get("scales"))
    # v1.0.1.5: Use _p_ persist fallback — widget keys may not survive cross-page navigation
    _gen_title = (st.session_state.get("study_title") or st.session_state.get("_p_study_title", "")).strip()
    _gen_desc = (st.session_state.get("study_description") or st.session_state.get("_p_study_description", "")).strip()
    required_fields = {
        "Study title": bool(_gen_title),
        "Study description": bool(_gen_desc),
        "Sample size (\u226510)": int(st.session_state.get("sample_size", 0)) >= 10,
        _input_label: bool(preview and preview.success) or bool(st.session_state.get("conversational_builder_complete")),
        "Design configured": bool(inferred),
        "DVs defined": _has_dvs,
    }

    missing_fields = [label for label, ok in required_fields.items() if not ok]
    if missing_fields:
        st.warning(f"Missing: {', '.join(missing_fields)}. Complete earlier steps first.")

    # Quick summary — metrics row
    conditions = inferred.get('conditions', [])
    scales = st.session_state.get('confirmed_scales', []) or inferred.get('scales', [])
    factors = inferred.get('factors', [])
    scale_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]
    _sample_n = st.session_state.get('sample_size', 0)

    # v1.0.2.3: Compact study title + inline metrics with consistent sizing
    _study_title_display = st.session_state.get('study_title', 'Untitled')
    _per_cell = _sample_n // max(len(conditions), 1) if _sample_n else 0
    st.markdown(
        f'<div style="font-size:0.95rem;font-weight:600;color:#1F2937;margin-bottom:4px;">{_study_title_display}</div>'
        f'<div style="display:flex;gap:24px;font-size:0.85rem;color:#6B7280;margin-bottom:8px;">'
        f'<span><strong style="color:#374151;">{_sample_n}</strong> participants ({_per_cell}/cell)</span>'
        f'<span><strong style="color:#374151;">{len(conditions)}</strong> conditions</span>'
        f'<span><strong style="color:#374151;">{len(scale_names)}</strong> DVs</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ========================================
    # v1.0.0: DIFFICULTY LEVEL SELECTOR
    # v1.4.16: Hidden after generation completes; value read from session state key
    # ========================================
    # Use builder_difficulty_level as fallback if set from design review
    _diff_options = ['easy', 'medium', 'hard', 'expert']
    _builder_diff = st.session_state.get("builder_difficulty_level", "")
    _diff_default = _builder_diff if _builder_diff in _diff_options else "medium"
    _diff_index = _diff_options.index(_diff_default)

    if not _has_generated and not _is_generating:
        st.markdown("")
        st.markdown("#### Data Quality Difficulty")

        difficulty_col1, difficulty_col2 = st.columns([1, 2])
        with difficulty_col1:
            difficulty_level = st.selectbox(
                "Select difficulty level",
                options=_diff_options,
                format_func=lambda x: DIFFICULTY_LEVELS[x]['name'],
                index=_diff_index,
                key="difficulty_level",
                help="Controls the amount of noise and data quality issues in the simulated data"
            )

        with difficulty_col2:
            diff_settings = _get_difficulty_settings(difficulty_level)
            st.caption(f"**{diff_settings['description']}**")
            st.caption(f"Attention rate: {diff_settings['attention_rate']:.0%} | "
                      f"Careless: {diff_settings['careless_rate']:.0%} | "
                      f"Text quality: {diff_settings['text_quality']}")
    else:
        # After generation / during generation: read from session state key
        difficulty_level = st.session_state.get("difficulty_level", _diff_default)
        # v1.8.7.5: Guard against corrupted session state values
        if difficulty_level not in DIFFICULTY_LEVELS:
            difficulty_level = _diff_default

    # ========================================
    # v1.0.0: PRE-REGISTRATION CONSISTENCY CHECK
    # ========================================
    prereg_text = st.session_state.get("prereg_text_sanitized", "")
    prereg_pdf_names = st.session_state.get("prereg_pdf_names", [])

    if prereg_text or prereg_pdf_names:
        st.markdown("")
        st.markdown("#### Pre-registration Consistency Check")

        # Parse the pre-registration
        prereg_format = _detect_prereg_format(prereg_text)
        prereg_number = _extract_prereg_number(prereg_text, prereg_format)
        prereg_sections = _parse_prereg_sections(prereg_text, prereg_format)
        prereg_variables = _extract_prereg_variables(prereg_sections)

        # Show detected pre-registration info
        if prereg_number:
            st.info(f"**Pre-registration ID:** {prereg_number}")

        # Build design data for comparison
        design_data = {
            'conditions': conditions,
            'scales': scales,
            'sample_size': st.session_state.get('sample_size', 0),
        }

        prereg_data = {
            'variables': prereg_variables,
            'sample_size': None,  # Would need to extract from sections
            'format': prereg_format,
        }

        # Run consistency check
        consistency_warnings = _check_prereg_consistency(prereg_data, design_data)

        if consistency_warnings:
            for warning in consistency_warnings:
                if warning['severity'] == 'warning':
                    st.warning(f"**{warning['type'].replace('_', ' ').title()}**: {warning['message']}")
                else:
                    st.info(f"**{warning['type'].replace('_', ' ').title()}**: {warning['message']}")
                if warning.get('recommendation'):
                    st.caption(f"💡 {warning['recommendation']}")
        else:
            st.success("No consistency issues detected between pre-registration and design.")

        # Show detected variables from pre-registration (only if any found)
        _has_prereg_vars = any(prereg_variables.get(k) for k in ('ivs', 'dvs', 'mediators', 'moderators'))
        if _has_prereg_vars:
            with st.expander("View detected pre-registration variables"):
                if prereg_variables.get('ivs'):
                    st.markdown(f"**Independent Variables:** {', '.join(prereg_variables['ivs'][:5])}")
                if prereg_variables.get('dvs'):
                    st.markdown(f"**Dependent Variables:** {', '.join(prereg_variables['dvs'][:5])}")
                if prereg_variables.get('mediators'):
                    st.markdown(f"**Mediators:** {', '.join(prereg_variables['mediators'][:3])}")
                if prereg_variables.get('moderators'):
                    st.markdown(f"**Moderators:** {', '.join(prereg_variables['moderators'][:3])}")

    # ========================================
    # v1.0.0: LIVE DATA PREVIEW
    # ========================================
    st.markdown("")
    st.markdown("#### Live Data Preview")
    st.caption("Preview 5 rows of simulated data before full generation")

    if st.button("Generate Preview (5 rows)", key="preview_button"):
        open_ended_for_preview = st.session_state.get('confirmed_open_ended', [])
        # v1.0.3.8: Pass study context for context-aware open-text preview
        _preview_study_title = st.session_state.get('study_title', '') or st.session_state.get('_p_study_title', '')
        _preview_study_desc = st.session_state.get('study_description', '') or st.session_state.get('_p_study_description', '')
        preview_df = _generate_preview_data(
            conditions=conditions,
            scales=scales,
            open_ended=open_ended_for_preview,
            n_rows=5,
            difficulty=difficulty_level,
            study_title=_preview_study_title,
            study_description=_preview_study_desc,
        )
        st.session_state['preview_df'] = preview_df

    if 'preview_df' in st.session_state and st.session_state['preview_df'] is not None:
        _preview_show = st.session_state['preview_df']
        # v1.0.6.7: Extract OE note before displaying, then drop the meta column
        _preview_oe_note = None
        if '_oe_note' in _preview_show.columns:
            _preview_oe_note = _preview_show['_oe_note'].iloc[0] if len(_preview_show) > 0 else None
            _preview_show = _preview_show.drop(columns=['_oe_note'])
        st.dataframe(_preview_show, use_container_width=True, height=200)
        _diff_info = DIFFICULTY_LEVELS.get(difficulty_level, DIFFICULTY_LEVELS.get("medium", {}))
        st.caption(f"Preview at difficulty: **{_diff_info.get('name', difficulty_level)}**")
        if _preview_oe_note:
            st.caption(f"Open-ended columns ({_preview_oe_note}) will be generated via AI in the full dataset.")

    if not st.session_state.get("advanced_mode", False):
        # v1.0.0: Use difficulty level settings
        diff_settings = _get_difficulty_settings(difficulty_level)
        # Use builder demographics if available, otherwise defaults
        builder_demo = st.session_state.get("demographics_config")
        if builder_demo and isinstance(builder_demo, dict):
            demographics = {
                "gender_quota": builder_demo.get("gender_quota", 50),
                "age_mean": builder_demo.get("age_mean", 35),
                "age_sd": builder_demo.get("age_sd", 12),
            }
        else:
            demographics = STANDARD_DEFAULTS["demographics"].copy()
        attention_rate = diff_settings['attention_rate']  # From difficulty level
        random_responder_rate = diff_settings['random_responder_rate']  # From difficulty level
        exclusion = ExclusionCriteria(**STANDARD_DEFAULTS["exclusion_criteria"])
        # Use builder effect sizes if available, validating against current conditions
        builder_effects = st.session_state.get("builder_effect_sizes", [])
        effect_sizes: List[EffectSizeSpec] = []
        _current_conds = set(conditions)
        _invalid_effects = []
        for be in builder_effects:
            try:
                _hi = be["level_high"]
                _lo = be["level_low"]
                # Check that referenced conditions still exist
                if _hi not in _current_conds or _lo not in _current_conds:
                    _invalid_effects.append(f"{be.get('variable','?')}: '{_hi}' vs '{_lo}'")
                    continue
                effect_sizes.append(EffectSizeSpec(
                    variable=be["variable"],
                    factor=be.get("factor", "condition"),
                    level_high=_hi,
                    level_low=_lo,
                    cohens_d=float(be.get("cohens_d", 0.5)),
                    direction=be.get("direction", "positive"),
                ))
            except (KeyError, ValueError):
                pass
        if _invalid_effects:
            st.warning(
                f"Some effect sizes reference conditions that no longer exist "
                f"and were skipped: {', '.join(_invalid_effects)}. "
                f"Go back to Design Review to reconfigure."
            )
        custom_persona_weights = st.session_state.get("custom_persona_weights", None)

        # Store difficulty settings in session state for engine
        st.session_state['difficulty_settings'] = diff_settings

        with st.expander("View settings", expanded=False):
            col_std1, col_std2 = st.columns(2)
            with col_std1:
                st.markdown("**Demographics**")
                st.markdown(f"- Gender balance: {demographics['gender_quota']}% male / {100-demographics['gender_quota']}% female")
                st.markdown(f"- Age: M = {demographics['age_mean']}, SD = {demographics['age_sd']}")

                st.markdown("**Data Quality**")
                st.markdown(f"- Attention check pass rate: {attention_rate:.0%}")
                st.markdown(f"- Random/careless responders: {random_responder_rate:.0%}")

            with col_std2:
                st.markdown("**Exclusion Criteria**")
                st.markdown(f"- Min completion time: {exclusion.completion_time_min_seconds}s (1 min)")
                st.markdown(f"- Max completion time: {exclusion.completion_time_max_seconds}s (30 min)")
                st.markdown(f"- Straight-line threshold: {exclusion.straight_line_threshold}+ items")
                st.markdown(f"- Check duplicate IPs: {'Yes' if exclusion.duplicate_ip_check else 'No'}")

                st.markdown("**Effect Sizes**")
                if effect_sizes:
                    for es in effect_sizes:
                        st.markdown(f"- **{es.variable}**: d={es.cohens_d} ({es.level_high} > {es.level_low})")
                else:
                    st.markdown("- No directional effects (null hypothesis)")
                    st.caption("Specify effects in the Design tab or enable Advanced mode")
    else:
        st.markdown("#### Advanced Settings")

        c1, c2, c3 = st.columns(3)
        with c1:
            male_pct = st.slider("Male %", 0, 100, int(ADVANCED_DEFAULTS["demographics"]["gender_quota"]))
            age_mean = st.number_input("Mean age", 18, 70, int(ADVANCED_DEFAULTS["demographics"]["age_mean"]))
            age_sd = st.number_input("Age SD", 1, 30, int(ADVANCED_DEFAULTS["demographics"]["age_sd"]))
            demographics = {"gender_quota": int(male_pct), "age_mean": float(age_mean), "age_sd": float(age_sd)}

        with c2:
            attention_rate = st.slider("Attention check pass rate", 0.50, 1.00, float(ADVANCED_DEFAULTS["attention_rate"]), 0.01)
            random_responder_rate = st.slider("Random responder rate", 0.00, 0.30, float(ADVANCED_DEFAULTS["random_responder_rate"]), 0.01)

        with c3:
            min_sec = st.number_input("Min completion time (sec)", 10, 600, 60)
            max_sec = st.number_input("Max completion time (sec)", 300, 7200, 1800)
            straight = st.number_input("Straight-line threshold", 3, 40, 10)
            exclusion = ExclusionCriteria(
                attention_check_threshold=0.0,
                completion_time_min_seconds=int(min_sec),
                completion_time_max_seconds=int(max_sec),
                straight_line_threshold=int(straight),
                duplicate_ip_check=True,
                exclude_careless_responders=False,
            )

        st.markdown("#### Expected Effect Sizes *(optional)*")
        st.caption(
            "Specify the expected effect size for your main hypothesis. "
            "This makes the simulated data reflect a directional hypothesis rather than a null effect."
        )

        effect_sizes = []

        # Pre-populate from builder if available, validating conditions
        builder_effects = st.session_state.get("builder_effect_sizes", [])
        _adv_conds = set(conditions)
        for be in builder_effects:
            try:
                _hi = be["level_high"]
                _lo = be["level_low"]
                if _hi not in _adv_conds or _lo not in _adv_conds:
                    continue  # Skip stale effect sizes
                effect_sizes.append(EffectSizeSpec(
                    variable=be["variable"],
                    factor=be.get("factor", "condition"),
                    level_high=_hi,
                    level_low=_lo,
                    cohens_d=float(be.get("cohens_d", 0.5)),
                    direction=be.get("direction", "positive"),
                ))
            except (KeyError, ValueError):
                pass

        # Get available scales and factors for selection
        available_scales = [s.get("name", "Main_DV") for s in scales] if scales else ["Main_DV"]
        available_factors = [f.get("name", "Condition") for f in factors] if factors else ["Condition"]

        add_effect = st.checkbox("Add an expected effect size", value=False, key="add_effect_checkbox")

        if add_effect:
            st.markdown("**Configure Effect**")

            eff_col1, eff_col2 = st.columns(2)

            with eff_col1:
                # Select DV
                effect_variable = st.selectbox(
                    "Dependent variable (scale)",
                    options=available_scales,
                    key="effect_variable",
                    help="Which outcome variable should show the effect?"
                )

                # Select factor
                effect_factor = st.selectbox(
                    "Factor (independent variable)",
                    options=available_factors,
                    key="effect_factor",
                    help="Which factor creates the effect?"
                )

                # Get levels for selected factor
                selected_factor_data = next((f for f in factors if f.get("name") == effect_factor), None)
                if selected_factor_data:
                    factor_levels = selected_factor_data.get("levels", [])
                else:
                    factor_levels = conditions if conditions else []

                # Ensure we have at least some levels to work with
                if not factor_levels:
                    factor_levels = ["Control", "Treatment"]  # Fallback defaults

            with eff_col2:
                # Effect direction and magnitude
                st.markdown("**Effect magnitude (Cohen's d)**")
                effect_d = st.slider(
                    "Cohen's d",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.3,
                    step=0.05,
                    key="effect_cohens_d",
                    help="0.2 = small, 0.5 = medium, 0.8 = large effect"
                )

                # Visual guide for effect size
                if effect_d < 0.3:
                    st.caption("📊 Small effect")
                elif effect_d < 0.6:
                    st.caption("📊 Medium effect")
                else:
                    st.caption("📊 Large effect")

                effect_direction = st.radio(
                    "Direction",
                    options=["Higher in treatment", "Lower in treatment"],
                    key="effect_direction",
                    horizontal=True
                )

            # Level selection
            if len(factor_levels) >= 2:
                lev_col1, lev_col2 = st.columns(2)
                with lev_col1:
                    level_high = st.selectbox(
                        "Higher-scoring condition",
                        options=factor_levels,
                        key="effect_level_high",
                        help="Which condition should have higher scores?"
                    )
                with lev_col2:
                    other_levels = [l for l in factor_levels if l != level_high]
                    level_low = st.selectbox(
                        "Lower-scoring condition",
                        options=other_levels if other_levels else factor_levels,
                        key="effect_level_low",
                        help="Which condition should have lower scores?"
                    )

                # Build effect spec
                if effect_variable and effect_factor and level_high and level_low:
                    effect_sizes.append(
                        EffectSizeSpec(
                            variable=effect_variable,
                            factor=effect_factor,
                            level_high=level_high,
                            level_low=level_low,
                            cohens_d=effect_d,
                            direction="positive" if "Higher" in effect_direction else "negative",
                        )
                    )
                    st.success(
                        f"Effect configured: **{effect_variable}** will be {effect_d:.2f}d higher "
                        f"in '{level_high}' vs '{level_low}'"
                    )
            else:
                st.warning(
                    "⚠️ Need at least 2 condition levels to configure effect sizes. "
                    "Please ensure your study has multiple conditions defined."
                )

        custom_persona_weights = st.session_state.get("custom_persona_weights", None)

    # ========================================
    # v1.0.0: FINAL DESIGN SUMMARY
    # Complete overview of what will be simulated
    # v1.4.5: Hidden during generation to show progress immediately
    # ========================================

    # v1.0.3.0: Collapsed design summary — only shown in expander to reduce noise
    if not _is_generating and not _has_generated:
        pass  # Summary moved to expander below

    # Get all relevant design info (needed for both display and generation)
    display_conditions = conditions
    if st.session_state.get("use_crossed_conditions") and st.session_state.get("factorial_crossed_conditions"):
        display_conditions = st.session_state["factorial_crossed_conditions"]

    confirmed_oe = st.session_state.get("confirmed_open_ended", [])
    oe_count = len(confirmed_oe)

    # v1.0.3.0: Collapsed design review — keeps page focused on generation
    if not _is_generating and not _has_generated:
        with st.expander("Review full design", expanded=False):
            n_conds = len(display_conditions)
            clean_cond_names = [_clean_condition_name(c) for c in display_conditions]
            dv_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]
            N = st.session_state.get("sample_size", 200)
            n_per_cell = N // max(1, len(display_conditions))

            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**Conditions ({n_conds}):** {', '.join(clean_cond_names[:6])}")
                if len(clean_cond_names) > 6:
                    st.caption(f"+{len(clean_cond_names) - 6} more")
                st.markdown(f"**DVs ({len(dv_names)}):** {', '.join(dv_names[:5]) if dv_names else 'Main_DV'}")
            with detail_col2:
                if oe_count > 0:
                    oe_names = [oe.get('variable_name', oe.get('name', '')) for oe in confirmed_oe[:5]]
                    st.markdown(f"**Open-Ended ({oe_count}):** {', '.join(oe_names)}")
                st.markdown(f"**Sample:** {N} total (~{n_per_cell}/cell)")
                if st.session_state.get("add_effect_checkbox", False):
                    effect_d = st.session_state.get("effect_cohens_d", 0.5)
                    st.markdown(f"**Effect Size:** d = {effect_d:.2f}")

    # ========================================
    # GENERATE BUTTON - with proper state management
    # v1.4.5: Progress spinner shown FIRST, design summary hidden during generation
    # ========================================
    st.markdown("")

    # v1.4.9: LLM status indicator with multi-provider support and single key input
    # v1.0.1.5: Fixed key — inferred_design uses "open_ended_questions", not "open_ended"
    _has_open_ended = bool(st.session_state.get("inferred_design", {}).get("open_ended_questions", []))
    if not _has_open_ended:
        _has_open_ended = bool(st.session_state.get("confirmed_open_ended", []))

    if _has_open_ended and not _is_generating and not _has_generated:
        # Check LLM connectivity (cached in session state to avoid repeated API calls)
        _llm_status = st.session_state.get("_llm_connectivity_status")
        if _llm_status is None:
            try:
                from utils.llm_response_generator import LLMResponseGenerator
                _user_key = st.session_state.get("user_llm_api_key", "")
                _test_gen = LLMResponseGenerator(api_key=_user_key or None, seed=1)
                _llm_status = _test_gen.check_connectivity()
                _llm_status["provider_display"] = _test_gen.provider_display_name
                st.session_state["_llm_connectivity_status"] = _llm_status
            except Exception as _llm_err:
                _log(f"LLM connectivity check failed: {_llm_err}", level="warning")
                _llm_status = {"available": False, "provider": "none", "error": f"Init failed: {_llm_err}"}
                st.session_state["_llm_connectivity_status"] = _llm_status

        if _llm_status.get("available"):
            st.markdown(
                '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                'padding:12px 16px;margin-bottom:16px;">'
                '<span style="font-size:1.1em;font-weight:600;color:#166534;">'
                'AI-Powered Open-Ended Responses: Active</span><br>'
                '<span style="color:#15803d;font-size:0.9em;">'
                'Open-ended responses will be generated by an AI language model tailored to your specific '
                'survey questions and experimental conditions.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            # v1.0.5.8: Enhanced LLM fallback dialog — prominent "ask first" UI with
            # model selection, encrypted key storage, and tracking.
            # Track that we showed this dialog (for admin analytics)
            _exhaust_shown_count = st.session_state.get("_admin_llm_exhaust_dialog_shown", 0)
            st.session_state["_admin_llm_exhaust_dialog_shown"] = _exhaust_shown_count + 1

            st.markdown(
                '<div style="background:#fef2f2;border:2px solid #fca5a5;border-radius:10px;'
                'padding:16px 20px;margin-bottom:16px;">'
                '<span style="font-size:1.2em;font-weight:700;color:#991b1b;">'
                'All Built-in AI Providers Are Currently Unavailable</span><br>'
                '<span style="color:#b91c1c;font-size:0.95em;">'
                'All 6 built-in AI services have reached their rate limits. '
                'You have two options:</span>'
                '<ol style="color:#7f1d1d;font-size:0.9em;margin-top:8px;">'
                '<li><strong>Provide your own API key</strong> (free or paid) — '
                'AI-generated open-ended responses, highest quality</li>'
                '<li><strong>Proceed without a key</strong> — template-based responses '
                '(still realistic, 225+ research domains)</li>'
                '</ol></div>',
                unsafe_allow_html=True,
            )

            # v1.0.5.8: Enhanced key input with model/provider selection
            with st.expander("Use your own API key for AI-powered responses", expanded=True):
                st.markdown(
                    "**Get a free key in 30 seconds** from any of these providers:\n\n"
                    "| Provider | Free Tier | Sign Up |\n"
                    "|----------|-----------|--------|\n"
                    "| **Groq** (recommended) | 14,400 req/day | [console.groq.com](https://console.groq.com) |\n"
                    "| **Google AI Studio** | Free Gemini | [aistudio.google.com](https://aistudio.google.com) |\n"
                    "| **Cerebras** | 1M tokens/day | [cloud.cerebras.ai](https://cloud.cerebras.ai) |\n"
                    "| **OpenRouter** | Free models | [openrouter.ai](https://openrouter.ai) |\n"
                    "| **Poe** | 3K points/day | [poe.com/api_key](https://poe.com/api_key) |\n"
                    "| **OpenAI** | Paid (~$0.15/1M tokens) | [platform.openai.com](https://platform.openai.com) |\n"
                )

                # Provider/model selection dropdown
                _provider_options = [
                    "Auto-detect from key (recommended)",
                    "Groq (Llama 3.3 70B) — Free",
                    "Cerebras (Llama 3.3 70B) — Free",
                    "Google AI (Gemini Flash) — Free",
                    "Poe (GPT-4o-mini) — Free tier",
                    "OpenRouter (Mistral) — Free tier",
                    "OpenAI (GPT-4o-mini)",
                ]
                _selected_provider = st.selectbox(
                    "Provider / Model",
                    options=_provider_options,
                    index=0,
                    key="user_llm_provider_select",
                    help="Select your provider or leave on auto-detect. The provider is usually detected from the key prefix.",
                )

                # Decrypt existing key for display (shows masked in password field)
                _existing_encrypted = st.session_state.get("_user_llm_key_encrypted", "")
                _existing_plain = _decrypt_api_key(_existing_encrypted) if _existing_encrypted else ""
                _user_key_input = st.text_input(
                    "API Key",
                    value=_existing_plain,
                    type="password",
                    key="user_llm_key_input",
                    placeholder="Paste your key here (e.g., gsk_..., csk-..., sk-or-..., poe.com key, sk-...)",
                )

                # Detect key change and encrypt + store
                if _user_key_input != _existing_plain:
                    if _user_key_input.strip():
                        # Encrypt and store — key never sits as plaintext in session state
                        st.session_state["_user_llm_key_encrypted"] = _encrypt_api_key(_user_key_input.strip())
                        # Also set in env for the generator (env vars are process-scoped, not persisted)
                        os.environ["LLM_API_KEY"] = _user_key_input.strip()
                        # Track user key activation for admin
                        _user_key_count = st.session_state.get("_admin_user_key_activations", 0)
                        st.session_state["_admin_user_key_activations"] = _user_key_count + 1
                    else:
                        st.session_state["_user_llm_key_encrypted"] = ""
                        if "LLM_API_KEY" in os.environ:
                            del os.environ["LLM_API_KEY"]
                    # Maintain backward compat with old key storage
                    st.session_state["user_llm_api_key"] = _user_key_input.strip() if _user_key_input else ""
                    # Store selected provider for engine to use
                    st.session_state["_user_llm_provider_choice"] = _selected_provider
                    # Clear cached status so it re-checks
                    st.session_state["_llm_connectivity_status"] = None
                    st.rerun()

                st.markdown(
                    '<span style="color:#6b7280;font-size:0.8em;">'
                    'Your key is encrypted in memory for this session only. '
                    'It is never saved to disk, never logged, and transmitted only '
                    'to the selected provider\'s API over HTTPS.</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;'
                'padding:10px 14px;margin-top:4px;margin-bottom:16px;">'
                '<span style="font-weight:600;color:#0369a1;font-size:0.95em;">'
                'No key? No problem.</span><br>'
                '<span style="color:#0369a1;font-size:0.9em;">'
                'Without an AI key, your open-ended responses are generated by a '
                'comprehensive behavioral response engine trained on <strong>225+ research '
                'domains</strong> (from behavioral economics to health psychology) and '
                '<strong>40 question types</strong>. Each response is unique, '
                'persona-aligned (matching the participant\'s verbosity, formality, and '
                'engagement level), and condition-specific. The engine has been calibrated '
                'against real survey data to produce realistic variation in response '
                'quality, length, and content — including hedging, off-topic tangents, and '
                'minimal-effort answers from careless participants.</span></div>',
                unsafe_allow_html=True,
            )

    is_generating = _is_generating  # Use the early-read variable
    has_generated = _has_generated

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # v1.4.5: Show prominent animated progress indicator IMMEDIATELY when generating
    # (design summary is hidden above, so this is the first thing users see)
    if is_generating:
        with status_placeholder.container():
            st.markdown("""
            <style>
                @keyframes pulse {
                    0%, 100% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.1); opacity: 0.8; }
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .progress-spinner {
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    border: 4px solid rgba(255,255,255,0.3);
                    border-top: 4px solid #ffffff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-bottom: 15px;
                }
                .progress-container {
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                    border-radius: 15px;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                .progress-title {
                    color: white;
                    margin: 0 0 10px 0;
                    font-size: 24px;
                    animation: pulse 2s ease-in-out infinite;
                }
                .progress-subtitle {
                    color: #a0c4e8;
                    font-size: 16px;
                    margin: 0;
                }
            </style>
            <div class="progress-container">
                <div class="progress-spinner"></div>
                <h2 class="progress-title">Generating Your Dataset...</h2>
                <p class="progress-subtitle">Creating realistic behavioral data. This typically takes 10-30 seconds.</p>
                <p class="progress-subtitle" style="margin-top: 10px; font-size: 14px;">Please don't close or refresh this page.</p>
            </div>
            """, unsafe_allow_html=True)

    if has_generated:
        # v1.0.6.7: Generation warnings stored for display in quality report expander
        # instead of showing as prominent yellow warnings that confuse users.
        _gen_meta = st.session_state.get("generated_metadata", {})
        _gen_warns = _gen_meta.get("generation_warnings", []) if isinstance(_gen_meta, dict) else []
        if _gen_warns:
            _existing_notes = st.session_state.get("_gen_quality_notes", [])
            for _gw in _gen_warns:
                if _gw not in _existing_notes:
                    _existing_notes.append(_gw)
            st.session_state["_gen_quality_notes"] = _existing_notes

    # v1.4.5: CSS to prevent disabled buttons from appearing clickable
    st.markdown("""
    <style>
        /* Disabled buttons should NOT look clickable */
        button[disabled], button:disabled,
        [data-testid="stBaseButton-primary"] button:disabled,
        .stButton button:disabled {
            cursor: not-allowed !important;
            opacity: 0.6 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # v1.4.14: Preflight validation — catch issues before expensive generation
    _preflight_errors = _preflight_validation() if all_required_complete else []
    if _preflight_errors and not is_generating and not has_generated:
        st.error("**Pre-generation checks failed:**")
        for _pe in _preflight_errors:
            st.markdown(f"- {_pe}")

    # Button: disabled if not ready, generating, or already generated
    can_generate = all_required_complete and not is_generating and not has_generated and not _preflight_errors

    # v1.4.5: No generate button during generation — just show the progress + reset
    if is_generating:
        # Only show reset button during generation
        if st.button("Cancel & Reset", use_container_width=True, key="reset_generate_btn"):
            _reset_generation_state()
            _navigate_to(3)
    else:
        # Create button row with generate + reset
        if has_generated:
            # After generation: only show reset button (download section below)
            if st.button("Reset & Generate New", use_container_width=True, key="reset_after_gen_btn"):
                _reset_generation_state()
                _navigate_to(3)
        else:
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                if st.button("Generate simulated dataset", type="primary", disabled=not can_generate, use_container_width=True, key="generate_dataset_btn"):
                    st.session_state["generation_requested"] = False
                    st.session_state["is_generating"] = True
                    st.session_state["_generation_phase"] = 1
                    _navigate_to(3)

    # v1.2.1 / v1.3.4: Legacy fallback — handle generation_requested if set elsewhere
    if st.session_state.get("generation_requested") and not is_generating:
        st.session_state["generation_requested"] = False
        st.session_state["is_generating"] = True
        st.session_state["_generation_phase"] = 1
        _navigate_to(3)

    # Phase 2: Actually generate (progress UI is now visible)
    if is_generating and st.session_state.get("_generation_phase", 0) == 1:
        st.session_state["_generation_phase"] = 2  # Move to generation phase
        progress_bar = progress_placeholder.progress(5, text="Preparing simulation inputs...")
        status_placeholder.info("🔄 Preparing simulation inputs...")
        # v1.0.1.5: Use _p_ persist fallback — widget keys may not survive cross-page navigation
        title = (st.session_state.get("study_title") or st.session_state.get("_p_study_title", "")) or "Untitled Study"
        desc = (st.session_state.get("study_description") or st.session_state.get("_p_study_description", "")) or ""
        try:
            requested_n = int(st.session_state.get("sample_size", 200))
        except (ValueError, TypeError):
            requested_n = 200
        if requested_n > MAX_SIMULATED_N:
            st.info(
                f"Requested N ({requested_n}) exceeds the cap ({MAX_SIMULATED_N}). "
                "Using the capped value for standardization."
            )
        N = min(requested_n, MAX_SIMULATED_N)

        # v1.4.2.1: Safety assertion — missing_fields should never be true here
        # because the Generate button is disabled when all_required_complete is False.
        # The legacy generation_requested fallback is the only theoretical path, so
        # guard defensively rather than leaving dead code.
        if missing_fields:
            st.session_state["is_generating"] = False
            st.session_state["generation_requested"] = False
            progress_placeholder.empty()
            st.warning(
                "Cannot generate: missing required fields "
                f"({', '.join(missing_fields)}). Please complete all steps first."
            )
            st.stop()

        prereg_text = st.session_state.get("prereg_text_sanitized", "")

        # ========================================
        # QSF VARIABLE VALIDATION
        # Use only user-confirmed scales from the scale confirmation UI
        # This ensures we ONLY simulate variables that exist in the QSF
        # ========================================
        confirmed_scales = st.session_state.get("confirmed_scales", [])
        if confirmed_scales:
            # User has confirmed scales - use those exclusively
            clean_scales = _normalize_scale_specs(confirmed_scales)
            # Log the validated scales for debugging
            validated_scale_names = [s.get("name", "Unknown") for s in clean_scales]
            st.session_state["_validated_scales_log"] = validated_scale_names
        else:
            # Fallback to inferred scales if no confirmation (shouldn't happen with new UI)
            clean_scales = _normalize_scale_specs(inferred.get("scales", []))

        # Additional validation: warn if no scales detected
        if not clean_scales:
            st.warning("No scales detected or confirmed. A default scale will be used. Please check your study configuration.")
            clean_scales = [{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 5, "scale_points": 7, "reverse_items": [], "_validated": True}]

        clean_factors = _normalize_factor_specs(inferred.get("factors", []), inferred.get("conditions", []))

        # Get condition allocation from session state
        condition_allocation = st.session_state.get("condition_allocation", None)

        # Build open-ended questions with full context for text generation
        # PRIORITY: Use user-confirmed open-ended questions from Step 3
        # This ensures only questions the user verified will have text generated
        # v1.4.2.1: Robust handling for both dict and string elements in confirmed_open_ended
        confirmed_open_ended = st.session_state.get("confirmed_open_ended", [])
        if confirmed_open_ended:
            # Use user-confirmed questions with full context
            open_ended_questions_for_engine = []
            for oe in confirmed_open_ended:
                if isinstance(oe, dict):
                    open_ended_questions_for_engine.append({
                        "name": oe.get("variable_name", oe.get("name", "")),
                        "variable_name": oe.get("variable_name", oe.get("name", "")),
                        "question_text": oe.get("question_text", ""),
                        "question_context": oe.get("question_context", ""),  # v1.8.7.1
                        "context_type": oe.get("context_type", "general"),
                        "type": oe.get("source_type", "text"),
                        "force_response": oe.get("force_response", False),
                        "min_chars": oe.get("min_chars"),
                        "block_name": oe.get("block_name", ""),
                    })
                elif isinstance(oe, str) and oe.strip():
                    # Handle legacy/fallback case where open-ended is a plain string
                    open_ended_questions_for_engine.append({
                        "name": oe, "variable_name": oe,
                        "question_text": oe, "context_type": "general",
                        "type": "text", "force_response": False,
                        "min_chars": None, "block_name": "",
                    })
        else:
            # Fallback to inferred detailed info if available
            open_ended_details = inferred.get("open_ended_details", [])
            if open_ended_details:
                open_ended_questions_for_engine = open_ended_details
            else:
                # Final fallback to basic list (variable names only)
                basic_open_ended = inferred.get("open_ended_questions", [])
                open_ended_questions_for_engine = []
                for q in basic_open_ended:
                    if isinstance(q, dict):
                        open_ended_questions_for_engine.append(q)
                    elif isinstance(q, str) and q.strip():
                        open_ended_questions_for_engine.append(
                            {"name": q, "question_text": q, "context_type": "general"}
                        )

        # ========================================
        # v1.2.3: PRE-FLIGHT VALIDATION
        # Sanitize all inputs before passing to engine to prevent
        # float()/int() crashes on unexpected types (dicts, None, etc.)
        # ========================================
        def _preflight_sanitize_scales(scales_list: list) -> list:
            """Ensure all scale dicts have clean numeric values.

            v1.4.0: Enhanced to handle NaN, list, and dict contamination for all fields.
            Also handles None scale_points for numeric/slider scales from the builder.
            """
            sanitized = []
            for s in scales_list:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name", "")).strip()
                if not name:
                    continue

                def _clean_int(val: Any, default: int) -> int:
                    """Safely convert any value to int, handling dicts, lists, NaN, None."""
                    if val is None:
                        return default
                    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                        return default
                    if isinstance(val, dict):
                        for key in ("value", "count", "mean"):
                            if key in val:
                                try:
                                    return int(val[key])
                                except (ValueError, TypeError):
                                    pass
                        return default
                    if isinstance(val, (list, tuple)):
                        try:
                            return int(val[0]) if val else default
                        except (ValueError, TypeError, IndexError):
                            return default
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default

                # Determine scale type to handle None scale_points for numeric scales
                scale_type = str(s.get("type", "likert")).lower()

                # Force all numeric fields to proper types
                raw_pts = s.get("scale_points")
                if raw_pts is None and scale_type in ("numeric", "slider", "number"):
                    # Numeric scales: derive from min/max or use sensible default
                    _tmp_min = _clean_int(s.get("scale_min"), 0)
                    _tmp_max = _clean_int(s.get("scale_max"), 100)
                    pts = max(2, _tmp_max - _tmp_min + 1)
                else:
                    pts = _clean_int(raw_pts, 7)

                n_items = _clean_int(s.get("num_items"), 5)

                # Handle scale_min/scale_max
                s_min = _clean_int(s.get("scale_min", 1), 1)
                s_max = _clean_int(s.get("scale_max", pts), pts)

                # Ensure s_max > s_min
                if s_max <= s_min:
                    s_max = s_min + max(1, pts - 1)

                sanitized.append({
                    **s,
                    "name": name,
                    "scale_points": max(2, min(1001, pts)),
                    "num_items": max(1, n_items),
                    "scale_min": max(0, s_min),
                    "scale_max": max(1, s_max),
                    "_validated": True,
                })
            return sanitized

        clean_scales = _preflight_sanitize_scales(clean_scales)

        # Sanitize demographics dict
        if isinstance(demographics, dict):
            for key in ("age_mean", "age_sd", "gender_quota"):
                val = demographics.get(key)
                if isinstance(val, dict):
                    demographics[key] = val.get("value", {"age_mean": 35, "age_sd": 12, "gender_quota": 50}.get(key, 0))
                elif val is not None:
                    try:
                        demographics[key] = float(val)
                    except (ValueError, TypeError):
                        demographics[key] = {"age_mean": 35, "age_sd": 12, "gender_quota": 50}.get(key, 0)

        # v1.4.9: Ensure user's API key (if provided) is in env for the engine
        _user_llm_key = st.session_state.get("user_llm_api_key", "") or st.session_state.get("user_groq_api_key", "")
        if _user_llm_key:
            os.environ["LLM_API_KEY"] = _user_llm_key

        # v1.8.8.0: Retrieve correlation matrix and missing data settings
        _engine_corr_matrix = None
        _raw_corr = inferred.get("correlation_matrix")
        if _raw_corr is not None:
            try:
                _engine_corr_matrix = np.array(_raw_corr, dtype=float)
            except Exception:
                _engine_corr_matrix = None
        _engine_missing_rate = float(inferred.get("missing_data_rate", 0.0))
        _engine_dropout_rate = float(inferred.get("dropout_rate", 0.0))

        # v1.0.5.1: Inject condition descriptions into study_context for the engine
        _engine_study_context = dict(inferred.get("study_context", {}))
        _cond_descs_for_engine = st.session_state.get("builder_condition_descriptions", {})
        if _cond_descs_for_engine:
            _engine_study_context["condition_descriptions"] = _cond_descs_for_engine

        engine = EnhancedSimulationEngine(
            study_title=title,
            study_description=desc,
            sample_size=N,
            conditions=inferred.get("conditions", []),
            factors=clean_factors,
            scales=clean_scales,
            additional_vars=[],
            demographics=demographics,
            attention_rate=attention_rate,
            random_responder_rate=random_responder_rate,
            effect_sizes=effect_sizes,
            exclusion_criteria=exclusion,
            custom_persona_weights=custom_persona_weights,
            open_ended_questions=open_ended_questions_for_engine,
            study_context=_engine_study_context,
            condition_allocation=condition_allocation,
            seed=None,
            mode="pilot" if not st.session_state.get("advanced_mode", False) else "final",
            # v1.0.0: Pass pre-computed visibility map for accurate survey flow simulation
            precomputed_visibility=inferred.get("condition_visibility_map", {}),
            # v1.8.8.0: Cross-DV correlations and missing data
            correlation_matrix=_engine_corr_matrix,
            missing_data_rate=_engine_missing_rate,
            dropout_rate=_engine_dropout_rate,
        )

        # v1.2.4: Show detected research domains
        if hasattr(engine, 'detected_domains') and engine.detected_domains:
            domain_text = ", ".join([d.replace("_", " ").title() for d in engine.detected_domains[:5]])
            st.info(f"🔬 **Detected research domain(s):** {domain_text}")

        try:
            # v1.2.0: Enhanced progress indicator with prominent visual display
            # Clear the status placeholder and show progress bar
            status_placeholder.empty()

            # Show large, visible progress container
            progress_container = st.container()
            with progress_container:
                st.markdown("""
                <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 15px; margin: 15px 0;">
                    <h3 style="color: white; margin: 0 0 8px 0;">Simulation In Progress</h3>
                    <p style="color: #a0c4e8; font-size: 14px; margin: 0;">Generating realistic behavioral data for your experiment...</p>
                </div>
                """, unsafe_allow_html=True)

            progress_bar.progress(10, text="Step 1/5 — Initializing simulation engine...")

            # Show animated spinner during the actual data generation
            with st.spinner("Generating participant responses... Please wait."):
                progress_bar.progress(25, text="Step 2/5 — Generating participant responses...")
                df, metadata = engine.generate()

            # v1.4.9: Inject LLM stats into metadata for the instructor report
            if hasattr(engine, 'llm_generator') and engine.llm_generator is not None:
                metadata['llm_stats'] = engine.llm_generator.stats

            # v1.0.5.5: Track simulation run for admin dashboard
            # Fix: engine metadata uses "sample_size", "conditions", "scales" —
            # not "n_participants", "conditions_used", "scales_used".
            _llm_run_stats = metadata.get('llm_stats', metadata.get('llm_response_stats', {}))
            st.session_state["_last_llm_stats"] = _llm_run_stats
            _admin_history = st.session_state.get("_admin_sim_history", [])
            _admin_history.append({
                "title": st.session_state.get("study_title", metadata.get("study_title", "Untitled")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sample_size": metadata.get("sample_size", metadata.get("n_participants", 0)),
                "n_conditions": len(metadata.get("conditions", metadata.get("conditions_used", []))),
                "n_scales": len(metadata.get("scales", metadata.get("scales_used", []))),
                "conditions": metadata.get("conditions", metadata.get("conditions_used", [])),
                "detected_domains": list(metadata.get("detected_domains", [])),
                "personas_used": metadata.get("personas_used", []),
                "llm_stats": _llm_run_stats,
            })
            st.session_state["_admin_sim_history"] = _admin_history
            # Store engine validation log for admin
            if hasattr(engine, 'validation_log'):
                st.session_state["_admin_engine_log"] = engine.validation_log[-50:]
            # v1.0.6.3: Persist admin history to disk for cross-session survival
            _save_admin_history()

            # v1.0.6.7: Store LLM exhaustion info in session state for post-generation display
            # instead of showing a yellow banner that flashes and disappears during generation.
            _run_exhaustions = _llm_run_stats.get("provider_exhaustions", 0)
            _run_fallbacks = _llm_run_stats.get("fallback_uses", 0)
            _run_pool = _llm_run_stats.get("pool_size", 0)
            if _run_exhaustions > 0 and _run_fallbacks > 0:
                _fb_pct = (_run_fallbacks / max(1, _run_pool + _run_fallbacks)) * 100
                _cum_exhaustions = st.session_state.get("_admin_total_exhaustions", 0)
                st.session_state["_admin_total_exhaustions"] = _cum_exhaustions + _run_exhaustions
                st.session_state["_gen_llm_exhaustion_note"] = (
                    f"AI providers experienced {_run_exhaustions} exhaustion event(s). "
                    f"{_fb_pct:.0f}% of open-ended responses used template fallback instead of AI. "
                    f"Provide your own API key and re-run for 100% AI-generated responses."
                )

            # v1.2.4: Run simulation quality validation
            validation_results = _validate_simulation_output(df, metadata, clean_scales)
            st.session_state["_validation_results"] = validation_results
            # v1.0.6.7: Store validation warnings in session state for post-generation display
            # instead of showing them inline during generation (they flash and disappear).
            _gen_quality_notes: List[str] = []
            if not validation_results["passed"]:
                for err in validation_results["errors"]:
                    _gen_quality_notes.append(f"Error: {err}")
            for warn in validation_results.get("warnings", []):
                if "unique responses" in warn:
                    continue  # suppress — expected for template-based OE generation
                _gen_quality_notes.append(warn)
            st.session_state["_gen_quality_notes"] = _gen_quality_notes

            # v1.2.5: Show quick data quality summary
            progress_bar.progress(50, text="Step 3/5 — Validating data quality...")  # v1.4.2.1: Fixed duplicate step numbering
            status_placeholder.info("🔍 Validating generated data...")
            quality_checks = []
            # Check scale ranges
            for scale in clean_scales:
                sname = str(scale.get("name", "")).strip().replace(" ", "_")
                s_min = int(scale.get("scale_min", 1))
                s_max = int(scale.get("scale_max", 7))
                n_items = int(scale.get("num_items", 5))
                for item_num in range(1, n_items + 1):
                    col = f"{sname}_{item_num}"
                    if col in df.columns:
                        actual_min = df[col].min()
                        actual_max = df[col].max()
                        if actual_min >= s_min and actual_max <= s_max:
                            quality_checks.append(f"✅ {col}: range [{actual_min}-{actual_max}] within [{s_min}-{s_max}]")
                        else:
                            quality_checks.append(f"⚠️ {col}: range [{actual_min}-{actual_max}] outside [{s_min}-{s_max}]")
            st.session_state["_quality_checks"] = quality_checks

            # Increment internal usage counter (admin tracking)
            try:
                usage_stats = _increment_usage_counter()
            except Exception:
                usage_stats = {}

            # Add usage stats to metadata for instructor report
            metadata["usage_stats"] = usage_stats

            progress_bar.progress(70, text="Step 4/5 — Packaging downloads & reports...")
            status_placeholder.info("📦 Packaging downloads and reports...")
            try:
                explainer = engine.generate_explainer()
            except Exception as _e:
                _log(f"Explainer generation failed: {_e}", level="warning")
                explainer = f"# Explainer generation failed: {_e}\n# See metadata for study details"
            try:
                r_script = engine.generate_r_export(df)
            except Exception as _e:
                _log(f"R export failed: {_e}", level="warning")
                r_script = f"# R export generation failed: {_e}"
            try:
                python_script = engine.generate_python_export(df)
            except Exception as _e:
                _log(f"Python export failed: {_e}", level="warning")
                python_script = f"# Python export generation failed: {_e}"
            try:
                julia_script = engine.generate_julia_export(df)
            except Exception as _e:
                _log(f"Julia export failed: {_e}", level="warning")
                julia_script = f"# Julia export generation failed: {_e}"
            try:
                spss_script = engine.generate_spss_export(df)
            except Exception as _e:
                _log(f"SPSS export failed: {_e}", level="warning")
                spss_script = f"* SPSS export generation failed: {_e}"
            try:
                stata_script = engine.generate_stata_export(df)
            except Exception as _e:
                _log(f"Stata export failed: {_e}", level="warning")
                stata_script = f"// Stata export generation failed: {_e}"

            metadata["preregistration_summary"] = {
                "outcomes": st.session_state.get("prereg_outcomes", ""),
                "independent_variables": st.session_state.get("prereg_iv", ""),
                "exclusion_criteria": st.session_state.get("prereg_exclusions", ""),
                "analysis_plan": st.session_state.get("prereg_analysis", ""),
                "notes_sanitized": st.session_state.get("prereg_text_sanitized", ""),
            }
            metadata["design_review"] = {
                "variable_roles": st.session_state.get("variable_review_rows", []),
                "randomization_level": st.session_state.get("randomization_level", ""),
            }

            try:
                schema_results = validate_schema(
                    df=df,
                    expected_conditions=inferred.get("conditions", []),
                    expected_scales=clean_scales,
                    expected_n=N,
                )
            except Exception as _schema_err:
                # v1.0.6.1: Log the error instead of silently swallowing
                _log(f"Schema validation error: {_schema_err}", level="warning")
                schema_results = {"passed": True, "checks": [], "warnings": [f"Schema validation encountered an error: {_schema_err}"], "errors": []}

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            meta_bytes = _safe_json(metadata).encode("utf-8")
            explainer_bytes = explainer.encode("utf-8")
            r_bytes = r_script.encode("utf-8")
            # v2.4.5: Encode additional analysis scripts
            python_bytes = python_script.encode("utf-8")
            julia_bytes = julia_script.encode("utf-8")
            spss_bytes = spss_script.encode("utf-8")
            stata_bytes = stata_script.encode("utf-8")
            # v1.2.3: Wrap report generation in try/except to prevent report errors
            # from crashing the entire simulation. Data generation succeeded at this point.
            try:
                # User study summary (included in user's download ZIP)
                instructor_report = InstructorReportGenerator().generate_markdown_report(
                    df=df,
                    metadata=metadata,
                    schema_validation=schema_results,
                    prereg_text=st.session_state.get("prereg_text_sanitized", ""),
                    team_info={
                        "team_name": st.session_state.get("team_name", ""),
                        "team_members": st.session_state.get("team_members_raw", ""),
                    },
                )
                instructor_bytes = instructor_report.encode("utf-8")
            except Exception as report_err:
                instructor_report = f"# Study Summary\n\nReport generation encountered an error: {report_err}\n\nData was generated successfully."
                instructor_bytes = instructor_report.encode("utf-8")

            try:
                # COMPREHENSIVE instructor report (for instructor email ONLY - not included in user download)
                # This includes detailed statistical analysis, hypothesis testing, and recommendations
                comprehensive_reporter = ComprehensiveInstructorReport()
                team_info_dict = {
                    "team_name": st.session_state.get("team_name", ""),
                    "team_members": st.session_state.get("team_members_raw", ""),
                }
                prereg_text_report = st.session_state.get("prereg_text_sanitized", "")

                # Markdown version (text-based)
                comprehensive_report = comprehensive_reporter.generate_comprehensive_report(
                    df=df,
                    metadata=metadata,
                    schema_validation=schema_results,
                    prereg_text=prereg_text_report,
                    team_info=team_info_dict,
                )
                comprehensive_bytes = comprehensive_report.encode("utf-8")

                # HTML version with visualizations and statistical tests
                comprehensive_html = comprehensive_reporter.generate_html_report(
                    df=df,
                    metadata=metadata,
                    schema_validation=schema_results,
                    prereg_text=prereg_text_report,
                    team_info=team_info_dict,
                )
                comprehensive_html_bytes = comprehensive_html.encode("utf-8")
            except Exception as comp_report_err:
                comprehensive_report = f"# Comprehensive Report\n\nReport generation encountered an error: {comp_report_err}\n\nData was generated successfully."
                comprehensive_bytes = comprehensive_report.encode("utf-8")
                comprehensive_html = f"<html><body><h1>Report Error</h1><p>{comp_report_err}</p></body></html>"
                comprehensive_html_bytes = comprehensive_html.encode("utf-8")

            # Generate HTML version of study summary (easy to open and well-formatted)
            try:
                study_title = metadata.get('study_title', 'Study Summary')
                instructor_html = _markdown_to_html(instructor_report, title=f"User Study Summary: {study_title}")
                instructor_html_bytes = instructor_html.encode("utf-8")
            except Exception:
                instructor_html_bytes = f"<html><body><pre>{instructor_report}</pre></body></html>".encode("utf-8")

            files = {
                "Simulated_Data.csv": csv_bytes,
                "Metadata.json": meta_bytes,
                "Data_Codebook_Handbook.txt": explainer_bytes,  # Explains all variable coding
                "R_Prepare_Data.R": r_bytes,
                "Python_Prepare_Data.py": python_bytes,  # v2.4.5: Python/pandas script
                "Julia_Prepare_Data.jl": julia_bytes,  # v2.4.5: Julia/DataFrames script
                "SPSS_Prepare_Data.sps": spss_bytes,  # v2.4.5: SPSS syntax file
                "Stata_Prepare_Data.do": stata_bytes,  # v2.4.5: Stata do-file
                "Schema_Validation.json": _safe_json(schema_results).encode("utf-8"),
                "User_Study_Summary.md": instructor_bytes,  # Summary report in Markdown
                "User_Study_Summary.html": instructor_html_bytes,  # Same summary in HTML (easy to view in browser)
            }

            # Include uploaded source files in "Source_Files" subfolder
            qsf_content = st.session_state.get("qsf_raw_content")
            if qsf_content:
                qsf_name = st.session_state.get("qsf_file_name", "survey.qsf")
                files[f"Source_Files/{qsf_name}"] = qsf_content if isinstance(qsf_content, bytes) else qsf_content.encode("utf-8")

            # Include all preregistration files
            prereg_materials = st.session_state.get("prereg_materials", [])
            if prereg_materials:
                for prereg_name, prereg_content in prereg_materials:
                    files[f"Source_Files/Preregistration/{prereg_name}"] = prereg_content
            else:
                # Fallback to single file for backwards compatibility
                prereg_pdf = st.session_state.get("prereg_pdf_content")
                if prereg_pdf:
                    prereg_name = st.session_state.get("prereg_pdf_name", "preregistration.pdf")
                    files[f"Source_Files/Preregistration/{prereg_name}"] = prereg_pdf

            # Include all survey materials
            survey_materials = st.session_state.get("survey_materials", [])
            if survey_materials:
                for survey_name, survey_content in survey_materials:
                    files[f"Source_Files/Survey_Materials/{survey_name}"] = survey_content
            else:
                # Fallback to single file for backwards compatibility
                survey_pdf = st.session_state.get("survey_pdf_content")
                if survey_pdf:
                    survey_pdf_name = st.session_state.get("survey_pdf_name", "survey_export.pdf")
                    files[f"Source_Files/Survey_Materials/{survey_pdf_name}"] = survey_pdf

            # Include preregistration text if provided
            prereg_outcomes = st.session_state.get("prereg_outcomes", "")
            prereg_iv = st.session_state.get("prereg_iv", "")
            if prereg_outcomes or prereg_iv:
                prereg_summary = f"# Preregistration Summary\n\n## Primary Outcomes\n{prereg_outcomes}\n\n## Independent Variables\n{prereg_iv}"
                files["Source_Files/Preregistration_Summary.txt"] = prereg_summary.encode("utf-8")

            zip_bytes = _bytes_to_zip(files)

            st.session_state["last_df"] = df
            st.session_state["last_zip"] = zip_bytes
            st.session_state["last_metadata"] = metadata

            progress_bar.progress(85, text="Step 5/5 — Finalizing output...")
            status_placeholder.info("Packaging your data...")
            # (success message consolidated into download section banner below)

            # v1.0.6.7: Schema validation results stored in quality notes instead of flashing
            if not schema_results.get("valid", True):
                _existing_qn = st.session_state.get("_gen_quality_notes", [])
                _existing_qn.append("Schema validation failed. Review Schema_Validation.json in the download.")
                st.session_state["_gen_quality_notes"] = _existing_qn
            elif schema_results.get("warnings"):
                _existing_qn = st.session_state.get("_gen_quality_notes", [])
                _existing_qn.append("Schema validation warnings found. Review Schema_Validation.json in the download.")
                st.session_state["_gen_quality_notes"] = _existing_qn

            # v1.0.0: Enhanced instructor email notification with better diagnostics
            instructor_email = st.secrets.get("INSTRUCTOR_NOTIFICATION_EMAIL", "edimant@sas.upenn.edu")
            subject = f"[Behavioral Simulation] Output ({metadata.get('simulation_mode', 'pilot')}) - {title}"

            # Get usage stats for internal tracking
            usage_summary = _get_usage_summary()

            # Check if SMTP email is configured before attempting to send
            smtp_configured = (
                st.secrets.get("SMTP_SERVER", "") and
                st.secrets.get("SMTP_USERNAME", "") and
                st.secrets.get("SMTP_PASSWORD", "")
            )

            if not smtp_configured:
                pass  # SMTP not configured — skip instructor notification silently
            else:
                body = (
                    "COMPREHENSIVE INSTRUCTOR ANALYSIS ATTACHED\n"
                    "=========================================\n\n"
                    "This email includes detailed statistical analysis that students do NOT receive.\n"
                    "Users get User_Study_Summary.md and User_Study_Summary.html (browser-viewable) in their download ZIP.\n\n"
                    "INSTRUCTOR ATTACHMENTS:\n"
                    "- INSTRUCTOR_Statistical_Report.html - Full visual report with charts, t-tests,\n"
                    "  ANOVA, Mann-Whitney, chi-squared, regression analysis, and effect sizes.\n"
                    "  Open in any web browser for best viewing.\n"
                    "- INSTRUCTOR_Detailed_Analysis.md - Text-based analysis (Markdown format)\n"
                    "- User_Study_Summary.md - What users receive (for reference)\n\n"
                    f"Team: {st.session_state.get('team_name','')}\n"
                    f"Members:\n{st.session_state.get('team_members_raw','')}\n\n"
                    f"Study: {title}\n"
                    f"Sample Size: N={metadata.get('sample_size', 'N/A')}\n"
                    f"Conditions: {len(metadata.get('conditions', []))}\n"
                    f"Generated: {metadata.get('generation_timestamp','')}\n"
                    f"Run ID: {metadata.get('run_id','')}\n\n"
                    "Files in ZIP (what students see):\n"
                    "- Simulated_Data.csv (the data)\n"
                    "- Data_Codebook_Handbook.txt (variable coding)\n"
                    "- User_Study_Summary.md (study summary in Markdown)\n"
                    "- User_Study_Summary.html (same summary - opens in any browser)\n"
                    "- R_Prepare_Data.R (R script)\n"
                    "- Python_Prepare_Data.py (Python/pandas script)\n"
                    "- Julia_Prepare_Data.jl (Julia/DataFrames script)\n"
                    "- SPSS_Prepare_Data.sps (SPSS syntax)\n"
                    "- Stata_Prepare_Data.do (Stata do-file)\n"
                    "- Metadata.json, Schema_Validation.json\n"
                    f"\n{usage_summary}\n"
                )

                # Log the email attempt for debugging
                # Instructor notification sent silently (user should not see this)

                ok, msg = _send_email(
                    to_email=instructor_email,
                    subject=subject,
                    body_text=body,
                    attachments=[
                        ("simulation_output.zip", zip_bytes),
                        ("INSTRUCTOR_Statistical_Report.html", comprehensive_html_bytes),  # HTML report with visualizations
                        ("INSTRUCTOR_Detailed_Analysis.md", comprehensive_bytes),  # Markdown fallback
                        ("User_Study_Summary.md", instructor_bytes),  # What users receive (for reference)
                    ],
                )
                if ok:
                    pass  # Instructor notification sent silently
                else:
                    pass  # Instructor notification failed silently — not shown to user

            progress_bar.progress(100, text="Complete — your dataset is ready to download.")
            status_placeholder.success("Simulation complete.")
            st.session_state["has_generated"] = True
            st.session_state["is_generating"] = False
            _navigate_to(3)  # Refresh to show download section
        except Exception as e:
            import traceback as _tb
            progress_bar.progress(100, text="Simulation failed.")
            status_placeholder.error("Simulation failed.")
            # v1.2.3 / v1.9.1: Categorized error messages with actionable guidance
            error_tb = _tb.format_exc()
            error_str = str(e).lower()

            # Categorize the error for a more helpful user-facing message
            if "api" in error_str or "auth" in error_str or "401" in error_str or "403" in error_str:
                st.error(
                    "**LLM API Error:** Could not connect to the AI provider. "
                    "If you provided an API key, please verify it is correct and has not expired. "
                    "The simulation will still work without an AI key using the built-in response engine."
                )
            elif "timeout" in error_str or "timed out" in error_str or "connection" in error_str:
                st.error(
                    "**Connection Timeout:** The AI provider did not respond in time. "
                    "This is usually temporary. Please try again in a moment, or proceed without an API key."
                )
            elif "memory" in error_str or "overflow" in error_str:
                st.error(
                    f"**Resource Error:** The simulation ran out of resources (N={N}). "
                    "Try reducing the sample size or the number of scales and retry."
                )
            elif "scale" in error_str or "column" in error_str:
                st.error(
                    f"**Data Configuration Error:** {e}. "
                    "Please go back to the Design step and verify your DV names and scale settings match your QSF."
                )
            else:
                st.error(f"**Simulation failed:** {e}")

            with st.expander("Error details (for debugging)", expanded=False):
                st.code(error_tb, language="python")
                # Show input summary for debugging
                st.markdown("**Input Summary:**")
                st.markdown(f"- Scales: {len(clean_scales)} configured")
                _cond_list = inferred.get('conditions', [])
                st.markdown(f"- Conditions: {len(_cond_list)} ({', '.join(str(c) for c in _cond_list[:5])}{'...' if len(_cond_list) > 5 else ''})")
                st.markdown(f"- Sample size: {N}")
                st.markdown(f"- Open-ended questions: {len(open_ended_questions_for_engine)}")
                if clean_scales:
                    st.markdown("**Scale details:**")
                    for i, s in enumerate(clean_scales[:5]):
                        st.markdown(f"  - Scale {i+1}: {s.get('name', '?')} (items={s.get('num_items', '?')}, pts={s.get('scale_points', '?')}, min={s.get('scale_min', '?')}, max={s.get('scale_max', '?')})")
                    if len(clean_scales) > 5:
                        st.markdown(f"  - ... and {len(clean_scales) - 5} more scale(s)")

            st.info("You can click **Reset & Generate New** to try again with different settings.")
            st.session_state["is_generating"] = False
            st.session_state["generation_requested"] = False
            # Don't rerun on error - show error message to user

    zip_bytes = st.session_state.get("last_zip", None)
    df = st.session_state.get("last_df", None)
    if zip_bytes and df is not None and len(df) > 0:
        st.markdown(
            '<div class="section-done-banner">Simulation complete — download your dataset below</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div id="download"></div>', unsafe_allow_html=True)
        st.markdown("#### Download")

        # v1.0.2.3: Compact inline download summary (consistent styling)
        _dl_meta = st.session_state.get("last_metadata", {}) or {}
        try:
            _dl_rows = f"{len(df):,}"
            _dl_cols_n = f"{len(df.columns):,}"
        except Exception:
            _dl_rows = "N/A"
            _dl_cols_n = "N/A"
        _dl_n_conds = 0
        try:
            _dl_n_conds = df["CONDITION"].nunique() if "CONDITION" in df.columns else 0
        except Exception:
            _dl_n_conds = 0
        _dl_zip_kb = len(zip_bytes) / 1024 if zip_bytes else 0
        _dl_zip_str = f"{_dl_zip_kb:.0f} KB" if _dl_zip_kb < 1024 else f"{_dl_zip_kb/1024:.1f} MB"
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:20px;font-size:0.85rem;color:#6B7280;margin-bottom:8px;">'
            f'<span><strong style="color:#374151;">{_dl_rows}</strong> rows</span>'
            f'<span><strong style="color:#374151;">{_dl_cols_n}</strong> columns</span>'
            f'<span><strong style="color:#374151;">{_dl_n_conds}</strong> conditions</span>'
            f'<span><strong style="color:#374151;">{_dl_zip_str}</strong> ZIP</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Data realism badges (correlation + missing data)
        _gen_meta = st.session_state.get("last_metadata", {}) or {}
        _gen_badges = []
        _corr_info = _gen_meta.get("cross_dv_correlation", {})
        if _corr_info.get("enabled"):
            _gen_badges.append(
                f"<span style='background:#E8F4FD;padding:3px 10px;border-radius:6px;"
                f"font-size:0.78rem;margin-right:6px;'>Cross-DV correlations: "
                f"{_corr_info.get('num_scales', 0)} scales</span>"
            )
        _miss_info = _gen_meta.get("missing_data", {})
        if _miss_info.get("total_missing_rate", 0) > 0:
            _gen_badges.append(
                f"<span style='background:#FEF3C7;padding:3px 10px;border-radius:6px;"
                f"font-size:0.78rem;margin-right:6px;'>Missing data: "
                f"{_miss_info.get('total_missing_rate', 0) * 100:.1f}% "
                f"({_miss_info.get('dropout_count', 0)} dropouts)</span>"
            )
        if _gen_badges:
            st.markdown(
                "<div style='margin:4px 0 8px 0;'>" + " ".join(_gen_badges) + "</div>",
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download ZIP (CSV + metadata + analysis scripts)",
            data=zip_bytes,
            file_name=f"behavioral_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )

        # v1.4.16: ZIP contents list
        st.caption(
            "Includes: Simulated_Data.csv, Data_Codebook_Handbook.txt, "
            "User_Study_Summary.html, R/Python/Julia/SPSS/Stata scripts, "
            "Metadata.json, Schema_Validation.json"
        )

        with st.expander("Preview (first 20 rows)"):
            try:
                _preview_df = df.head(20)
                if _preview_df is not None and len(_preview_df) > 0:
                    st.dataframe(_preview_df, use_container_width=True)
                else:
                    st.warning("No data rows available for preview.")
            except Exception as _preview_err:
                st.warning(f"Could not render data preview: {_preview_err}")
                st.caption("The data was generated successfully. Download the ZIP to access it.")

        # v1.2.5 / v1.0.6.7: Data Quality Report expander — includes validation warnings + LLM exhaustion notes
        quality_checks = st.session_state.get("_quality_checks", [])
        _gen_quality_notes = st.session_state.get("_gen_quality_notes", [])
        _gen_llm_note = st.session_state.get("_gen_llm_exhaustion_note", "")
        _has_quality_content = quality_checks or _gen_quality_notes or _gen_llm_note
        if _has_quality_content:
            with st.expander("📊 Data Quality Report", expanded=False):
                if _gen_llm_note:
                    st.info(_gen_llm_note)
                for _gqn in _gen_quality_notes:
                    st.markdown(f"- {_gqn}")
                for check in quality_checks:
                    st.markdown(check)

        # ══════════════════════════════════════════════════════════════
        # v1.8.8.0: ANALYTICS DASHBOARD (Advanced mode + password)
        # ══════════════════════════════════════════════════════════════
        if st.session_state.get("advanced_mode", False):
            st.markdown("---")
            _dashboard_pw = st.text_input(
                "Analytics Dashboard (enter access code)",
                type="password",
                key="analytics_dashboard_pw",
                help="Enter the access code to unlock the professional analytics dashboard.",
            )
            _DASHBOARD_PW_HASH = "f35234aa5d24"  # MD5[:12] of "Dimant_Simulation"
            _pw_valid = (
                _dashboard_pw == "Dimant_Simulation"
                or hashlib.md5(_dashboard_pw.encode()).hexdigest()[:12] == _DASHBOARD_PW_HASH
            )
            if _dashboard_pw and not _pw_valid:
                st.warning("Incorrect access code.")
            elif _pw_valid:
                _render_analytics_dashboard(df, metadata, clean_scales)

        st.markdown("")
        st.markdown("#### Email *(optional)*")

        to_email = st.text_input("Send to email", value=st.session_state.get("send_to_email", ""))
        st.session_state["send_to_email"] = to_email

        colE1, colE2 = st.columns([1, 1])
        with colE1:
            if st.button("Send ZIP via email", key="send_zip_email_btn"):
                if not to_email or "@" not in to_email:
                    st.error("Please enter a valid email address.")
                else:
                    subject = f"[Behavioral Simulation] Output: {st.session_state.get('study_title','Untitled Study')}"
                    body = (
                        "Attached is the simulation output ZIP (Simulated.csv, metadata, analysis scripts).\n\n"
                        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
                    )
                    ok, msg = _send_email(
                        to_email=to_email,
                        subject=subject,
                        body_text=body,
                        attachments=[("simulation_output.zip", zip_bytes)],
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        with colE2:
            instructor_email = st.secrets.get("INSTRUCTOR_NOTIFICATION_EMAIL", "")
            if instructor_email:
                if st.button("Send to instructor too", key="send_to_instructor_btn"):
                    subject = f"[Behavioral Simulation] Output (team: {st.session_state.get('team_name','') or 'N/A'})"
                    body = (
                        f"Team: {st.session_state.get('team_name','')}\n"
                        f"Members:\n{st.session_state.get('team_members_raw','')}\n\n"
                        f"Study: {st.session_state.get('study_title','')}\n"
                        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
                    )
                    ok, msg = _send_email(
                        to_email=instructor_email,
                        subject=subject,
                        body_text=body,
                        attachments=[("simulation_output.zip", zip_bytes)],
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.caption("Instructor email not configured in secrets (INSTRUCTOR_NOTIFICATION_EMAIL).")

    # v1.0.2.3: Bottom nav — ONLY "Back to top" (navigation is at the top)
    st.markdown("---")
    _nav_m3_col, = st.columns([1])
    with _nav_m3_col:
        st.markdown(
            '<a href="#btt-anchor" '
            'onclick="var el=document.getElementById(\'btt-anchor\');'
            'if(el){el.scrollIntoView({behavior:\'smooth\',block:\'start\'});}return false;" '
            'class="btt-link">\u2191 Back to top</a>',
            unsafe_allow_html=True,
        )

# ========================================
# FEEDBACK (Shown on wizard pages only)
# v1.5.0: Compact feedback — no more bulky expander at top
# v1.8.2: Only show on wizard pages, not landing page
# ========================================
if active_page >= 0:
    _render_feedback_button()
