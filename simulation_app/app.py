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

REQUIRED_UTILS_VERSION = "1.8.6"
BUILD_ID = "20260209-v186-domain-detection-word-boundary"  # Change this to force cache invalidation

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
from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedScale, KNOWN_SCALES, AVAILABLE_DOMAINS, generate_qsf_from_design
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
        from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedScale, KNOWN_SCALES, AVAILABLE_DOMAINS, generate_qsf_from_design  # noqa: F811
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
APP_VERSION = "1.8.6"  # v1.8.6: Domain detection with word-boundary matching, negative keywords, top-N filtering
APP_BUILD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

BASE_STORAGE = Path("data")
BASE_STORAGE.mkdir(parents=True, exist_ok=True)

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
    for scale in (scales or []):
        s_name = str(scale.get("name", "")).strip().replace(" ", "_")
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
    oe_cols = [c for c in df.columns if df[c].dtype == object and c not in [
        'CONDITION', 'PARTICIPANT_ID', 'RUN_ID', 'SIMULATION_MODE', 'SIMULATION_SEED', 'Gender'
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
    difficulty: str = 'medium'
) -> pd.DataFrame:
    """Generate a preview of simulated data (5 rows by default)."""
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
            preview_data[var_name] = [
                np.random.randint(_s_min, _s_max + 1) for _ in range(n_rows)
            ]
        else:
            # Multi-item scale - show first item and composite
            var_name = scale_name.replace(' ', '_')
            preview_data[f"{var_name}_1"] = [
                np.random.randint(_s_min, _s_max + 1) for _ in range(n_rows)
            ]
            preview_data[f"{var_name}_mean"] = [
                round(np.random.uniform(_s_min, _s_max), 2) for _ in range(n_rows)
            ]

    # Add sample open-ended responses
    # v1.0.0 CRITICAL FIX: Pass question name to ensure UNIQUE responses per question
    # v1.4.2.1: Handle both dict and string elements in open_ended list
    for oe in open_ended[:2]:  # Limit to 2 open-ended for preview
        if isinstance(oe, str):
            var_name = oe
            question_text = oe
        elif isinstance(oe, dict):
            var_name = oe.get('variable_name', oe.get('name', 'OE'))
            question_text = oe.get('question_text', var_name)
        else:
            continue
        # Use both var_name AND question_text to create truly unique identifier
        unique_question_id = f"{var_name}_{question_text[:50]}"
        preview_data[var_name] = [
            _get_sample_text_response(difficulty_settings['text_quality'], i, unique_question_id)
            for i in range(n_rows)
        ]

    # Add demographics
    preview_data['age'] = [np.random.randint(18, 65) for _ in range(n_rows)]
    preview_data['gender'] = [np.random.choice(['Male', 'Female', 'Other']) for _ in range(n_rows)]

    # Add attention check
    preview_data['attention_check_pass'] = [
        1 if np.random.random() < difficulty_settings['attention_rate'] else 0
        for _ in range(n_rows)
    ]

    return pd.DataFrame(preview_data)


def _get_sample_text_response(quality: str, participant_idx: int, question_name: str = "") -> str:
    """Generate sample text responses based on quality level.

    v1.0.0 CRITICAL FIX: Uses proper seeded random generator to GUARANTEE unique
    responses for each participant-question combination. The seed is deterministic
    so the same inputs always produce the same output, but different questions
    will ALWAYS get different responses.

    Args:
        quality: Response quality level (high/medium/low/very_low)
        participant_idx: Index of the participant (0-based)
        question_name: REQUIRED - unique identifier for the question

    Returns:
        Unique response for this participant-question combination
    """
    import random as random_module

    # CRITICAL: Create a unique seed that combines participant index AND question identity
    # Use a deterministic hash that's independent of Python's hash randomization
    if question_name:
        # Create a stable hash from the question name
        name_hash = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(question_name[:100]))
    else:
        name_hash = 0

    # Combine participant and question into a unique seed
    unique_seed = (participant_idx * 100003) + name_hash  # Use prime multiplier

    # Create a LOCAL random generator with this unique seed
    local_rng = random_module.Random(unique_seed)

    # Large response banks for maximum variety - 30+ responses per quality level
    high_quality_responses = [
        "I found this experience to be quite engaging and thought-provoking.",
        "The scenario presented was realistic and made me consider multiple perspectives.",
        "This was an interesting task that required careful deliberation.",
        "I appreciated the clarity of the instructions and the relevance of the topic.",
        "The questions were well-designed and captured my genuine reactions.",
        "Overall, I thought this was a meaningful exercise that made me reflect.",
        "The task was engaging and I tried to answer as honestly as possible.",
        "I gave careful thought to each response and tried to be accurate.",
        "The scenario felt realistic and relevant to real-world situations.",
        "I found the questions interesting and engaging to answer.",
        "This made me think about my own values and preferences.",
        "The experience was well-structured and easy to follow.",
        "I appreciated being asked to share my genuine thoughts.",
        "The questions prompted genuine reflection on my part.",
        "I tried to provide thoughtful and honest answers throughout.",
        "The study made me consider perspectives I hadn't thought of before.",
        "I engaged seriously with the material and gave it due consideration.",
        "The questions were clear and I felt I could express my views well.",
        "This was a valuable exercise that prompted self-reflection.",
        "I found myself thinking carefully about each response.",
        "The task helped me articulate my thoughts on this topic.",
        "I appreciated the opportunity to share my genuine perspective.",
        "The scenario was compelling and held my attention throughout.",
        "I tried to be as accurate as possible in describing my reactions.",
        "This experience made me consider the topic from new angles.",
        "The questions captured aspects of my thinking that I hadn't considered.",
        "I found the exercise to be worthwhile and thought-provoking.",
        "The study was well-constructed and easy to understand.",
        "I gave honest responses based on my true feelings about the topic.",
        "This made me reflect on my own experiences and beliefs.",
    ]

    medium_quality_responses = [
        "It was okay, nothing special.",
        "I thought about it for a bit before deciding.",
        "The task was straightforward.",
        "Made me think about the topic.",
        "Interesting scenario overall.",
        "Seemed like a reasonable exercise.",
        "I tried to answer honestly.",
        "Pretty standard experience overall.",
        "The questions made sense to me.",
        "I gave it some thought before answering.",
        "Nothing too surprising here.",
        "It was a reasonable task.",
        "I did my best to respond accurately.",
        "The scenario was understandable.",
        "It was fine, I suppose.",
        "The questions were clear enough.",
        "I answered based on my initial thoughts.",
        "The experience was okay overall.",
        "I tried to give truthful answers.",
        "Nothing out of the ordinary.",
        "The task was pretty simple.",
        "I understood what was being asked.",
        "My responses reflect my honest opinion.",
        "It was an average experience.",
        "The questions were reasonable.",
        "I completed the task without difficulty.",
        "Fairly straightforward exercise.",
        "I gave it reasonable consideration.",
        "The scenario was acceptable.",
        "Not too different from what I expected.",
    ]

    low_quality_responses = [
        "fine", "good", "ok", "yes", "idk", "sure", "maybe",
        "whatever", "alright", "meh", "k", "yeah", "dunno", "yep",
    ]

    very_low_quality_responses = [
        "asdf", "...", "n/a", "x", "", ".", "na",
        "aaa", "-", "no", "?", "z", "blah", "xyz",
    ]

    # Select appropriate response bank
    if quality == 'high':
        responses = high_quality_responses
    elif quality == 'medium':
        responses = medium_quality_responses
    elif quality == 'low':
        responses = low_quality_responses
    else:  # very_low
        responses = very_low_quality_responses

    # Shuffle the responses using our unique seed, then pick the first one
    # This guarantees different questions get different shuffles and thus different responses
    shuffled = responses.copy()
    local_rng.shuffle(shuffled)

    return shuffled[0]


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

# ── Compact header (only on wizard pages, not landing page) ───────────
_current_page = st.session_state.get("active_page", -1)
if _current_page >= 0:
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:0;">'
        f'<h2 style="margin:0;padding:0;font-size:1.35rem;">{APP_TITLE}</h2>'
        f'<span style="font-size:0.75rem;color:#D1D5DB;">v{APP_VERSION}</span>'
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

    # Condition/scale/OE state for Design & Generate tabs
    st.session_state["selected_conditions"] = [c.name for c in parsed_conditions]
    st.session_state["confirmed_scales"] = inferred.get("scales", [])
    st.session_state["scales_confirmed"] = True
    st.session_state["confirmed_open_ended"] = inferred.get("open_ended_questions", [])
    st.session_state["open_ended_confirmed"] = True

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

    # Check if builder is already complete
    if st.session_state.get("conversational_builder_complete"):
        st.markdown(
            '<div class="section-done-banner">'
            '✓ Study specification built — proceed to Design to review your configuration'
            '</div>',
            unsafe_allow_html=True,
        )
        _done_col1, _done_col2 = st.columns(2)
        with _done_col1:
            if st.button("Continue to Design  \u2192", key="builder_goto_design", type="primary", use_container_width=True):
                _scroll_then_navigate(2)
        with _done_col2:
            if st.button("Edit my description", key="builder_reopen_edit", use_container_width=True):
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
        # v1.6.0: Metrics row for at-a-glance summary
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Conditions", len(parsed_conditions))
        _mc2.metric("Scales", len(parsed_scales))
        _mc3.metric("Open-ended", len(parsed_oe))
        _mc4.metric("N", builder_sample)

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
            st.success("Study specification built successfully! Proceed to **Design** to review.")
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

    _sum_c1, _sum_c2, _sum_c3, _sum_c4 = st.columns(4)
    _sum_c1.metric("Conditions", _n_conds)
    _sum_c2.metric("Scales/DVs", _n_scales)
    _sum_c3.metric("Open-ended", _n_oe)
    _sum_c4.metric("Sample Size", _n_sample)

    # Design completeness check
    _design_issues: list[str] = []
    if _n_conds < 2:
        _design_issues.append("At least 2 conditions needed")
    if _n_scales < 1:
        _design_issues.append("At least 1 scale/DV needed")
    if _design_issues:
        st.warning("Design incomplete: " + "; ".join(_design_issues))
    else:
        # v1.6.0: CTA at top when design is ready — most important action first
        if st.button("Continue to Generate  \u2192", key="review_to_generate", type="primary"):
            _scroll_then_navigate(3)
        st.caption("Review the settings below, or proceed directly to generate your data.")

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
                                key=f"pw_{_pk}",
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
    if conditions:
        # Build a lookup: condition name → factor-level decomposition for factorial designs
        _cond_factor_map: dict[str, list[str]] = {}
        if factors and len(factors) >= 2:
            for _cond_name in conditions:
                _parts: list[str] = []
                for _fac in factors:
                    _fac_name = _fac.get("name", "Factor")
                    for _lev in _fac.get("levels", []):
                        # Check if this level appears in the condition name (case-insensitive)
                        if _lev.lower() in _cond_name.lower():
                            _parts.append(f"{_fac_name}: {_lev}")
                            break
                if _parts:
                    _cond_factor_map[_cond_name] = _parts

        cond_to_remove = None
        for i, cond in enumerate(conditions):
            cond_col, cond_btn_col = st.columns([8, 1])
            with cond_col:
                _factor_info = _cond_factor_map.get(cond, [])
                if _factor_info:
                    # Factorial condition: show name prominently with factor breakdown
                    st.markdown(f"**{i+1}. {cond}**")
                    st.caption(" | ".join(_factor_info))
                else:
                    st.markdown(f"**{i+1}. {cond}**")
            with cond_btn_col:
                # Require at least 2 conditions for a valid experiment
                if len(conditions) > 2 and st.button("X", key=f"br_remove_cond_{i}", help=f"Remove '{cond}'"):
                    cond_to_remove = i
                elif len(conditions) <= 2:
                    st.button("X", key=f"br_remove_cond_dis_{i}", disabled=True,
                              help="Minimum 2 conditions required for an experiment")
        if cond_to_remove is not None:
            conditions.pop(cond_to_remove)
            inferred["conditions"] = conditions
            # Recalculate factors from remaining conditions
            inferred["factors"] = _infer_factors_from_conditions(conditions)
            st.session_state["inferred_design"] = inferred
            st.session_state["selected_conditions"] = conditions
            # Clear stale effect sizes that may reference removed condition
            st.session_state["builder_effect_sizes"] = []
            # Update allocation with proper remainder distribution
            _n = max(len(conditions), 1)
            _samp = int(st.session_state.get("sample_size", 100))
            _per = _samp // _n
            _rem = _samp % _n
            st.session_state["condition_allocation_n"] = {
                c: _per + (1 if i < _rem else 0) for i, c in enumerate(conditions)
            }
            st.session_state["condition_allocation"] = {c: round(100.0 / _n, 1) for c in conditions}
            _navigate_to(2)
    else:
        st.warning("No conditions found. Please go back and describe your conditions.")

    # Allow adding custom conditions
    _add_col1, _add_col2 = st.columns([6, 1])
    with _add_col1:
        extra_conds = st.text_input(
            "Add more conditions (comma-separated)",
            value="",
            key="builder_review_extra_conds",
            help="Add additional conditions not captured from your description.",
        )
    with _add_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        _add_clicked = st.button("Add", key="builder_add_cond_btn")
    if _add_clicked and extra_conds.strip():
        new_conds = [c.strip() for c in extra_conds.split(",") if c.strip()]
        _added = 0
        _skipped = []
        for c in new_conds:
            # Skip if duplicate or contains factorial separator (would break factor parsing)
            if c in conditions:
                continue
            if " × " in c or " x " in c.lower():
                _skipped.append(c)
                continue
            if len(c) > 200:
                _skipped.append(c[:50] + "... (too long)")
                continue
            conditions.append(c)
            _added += 1
        if _skipped:
            st.warning(f"Skipped invalid conditions: {', '.join(_skipped)}")
        if _added > 0:
            inferred["conditions"] = conditions
            # Recalculate factors from updated conditions
            inferred["factors"] = _infer_factors_from_conditions(conditions)
            st.session_state["inferred_design"] = inferred
            st.session_state["selected_conditions"] = conditions
            # Clear stale effect sizes that may reference removed/added conditions
            st.session_state["builder_effect_sizes"] = []
            # Update allocation with proper remainder distribution
            _n = max(len(conditions), 1)
            _samp = int(st.session_state.get("sample_size", 100))
            _per = _samp // _n
            _rem = _samp % _n
            st.session_state["condition_allocation_n"] = {
                c: _per + (1 if i < _rem else 0) for i, c in enumerate(conditions)
            }
            st.session_state["condition_allocation"] = {c: round(100.0 / _n, 1) for c in conditions}
            _navigate_to(2)

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
    if scales:
        scale_to_remove = None
        for i, scale in enumerate(scales):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 0.5])
            with col1:
                new_name = st.text_input(
                    "Name", value=scale.get("name", ""), key=f"br_scale_name_{i}"
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
                    min_value=1, max_value=50, key=f"br_scale_items_{i}"
                )
                scale["num_items"] = new_items
                scale["items"] = [
                    f"{scale.get('variable_name', 'var')}_{j+1}" for j in range(new_items)
                ]
            with col3:
                new_min = st.number_input(
                    "Min", value=scale.get("scale_min", 1),
                    min_value=0, max_value=100, key=f"br_scale_min_{i}"
                )
                scale["scale_min"] = new_min
            with col4:
                # Ensure max > min (allow 0-1 for binary scales)
                _min_for_max = new_min + 1
                _cur_max = scale.get("scale_max", 7)
                _safe_max = max(_cur_max, _min_for_max)
                new_max = st.number_input(
                    "Max", value=_safe_max,
                    min_value=_min_for_max, max_value=1000, key=f"br_scale_max_{i}"
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
                if st.button("Remove", key=f"br_remove_scale_{i}", help=f"Remove '{scale.get('name', 'scale')}'"):
                    scale_to_remove = i

        if scale_to_remove is not None:
            scales.pop(scale_to_remove)
            inferred["scales"] = scales
            st.session_state["inferred_design"] = inferred
            st.session_state["confirmed_scales"] = scales
            _navigate_to(2)

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
                                _navigate_to(2)
    else:
        st.warning("No scales detected")

    # ── Open-Ended Questions ────────────────────────────────────────────
    if open_ended:
        st.markdown("---")
        st.markdown("#### Open-Ended Questions")
        oe_to_remove = None
        for i, oe in enumerate(open_ended):
            oe_col1, oe_col2 = st.columns([8, 1])
            with oe_col1:
                new_text = st.text_input(
                    f"Question {i+1} (`{oe.get('variable_name', '')}`)",
                    value=oe.get("question_text", ""),
                    key=f"br_oe_text_{i}",
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
            with oe_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove", key=f"br_remove_oe_{i}", help=f"Remove question {i+1}"):
                    oe_to_remove = i
        if oe_to_remove is not None:
            open_ended.pop(oe_to_remove)
            inferred["open_ended_questions"] = open_ended
            st.session_state["inferred_design"] = inferred
            st.session_state["confirmed_open_ended"] = open_ended
            _navigate_to(2)
        # Persist edits
        inferred["open_ended_questions"] = open_ended
        st.session_state["inferred_design"] = inferred
        st.session_state["confirmed_open_ended"] = open_ended

    # ── Condition Allocation ────────────────────────────────────────────
    if conditions and len(conditions) >= 2:
        st.markdown("---")
        st.markdown("#### Condition Allocation")
        per_cond = sample // len(conditions)
        remainder = sample % len(conditions)
        alloc = {}
        alloc_n = {}
        for i, cond in enumerate(conditions):
            n = per_cond + (1 if i < remainder else 0)
            alloc_n[cond] = n
            alloc[cond] = round(100.0 * n / sample, 1)
        st.session_state["condition_allocation_n"] = alloc_n
        st.session_state["condition_allocation"] = alloc

        # Visual allocation display
        n_conds = len(conditions)
        cols_per_row = min(n_conds, 4)
        alloc_cols = st.columns(cols_per_row)
        for i, cond in enumerate(conditions):
            with alloc_cols[i % cols_per_row]:
                n_val = alloc_n[cond]
                pct = alloc[cond]
                st.metric(label=cond, value=f"{n_val}", delta=f"{pct}%")
        st.caption("Equal allocation across conditions")

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

    # ── Confirmation ────────────────────────────────────────────────────
    st.markdown("---")
    if conditions and scales:
        st.success(
            f"Design ready: **{len(conditions)}** conditions, **{len(scales)}** scale(s), "
            f"**{sample}** participants."
        )
        st.info(
            "Your design is fully configured. You can proceed directly to the "
            "**Generate** step to simulate your data."
        )
    else:
        st.error("Design incomplete. Please go back and provide conditions and scales.")

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
    """Get completion status for each step."""
    preview = st.session_state.get("qsf_preview", None)

    # Check variable roles for primary outcome and independent variable
    variable_rows = st.session_state.get("variable_review_rows", [])
    has_primary_outcome = any(r.get("Role") == "Primary outcome" for r in variable_rows)
    has_independent_var = any(r.get("Role") == "Condition" for r in variable_rows)

    return {
        "study_title": bool((st.session_state.get("study_title") or st.session_state.get("_p_study_title", "")).strip()),
        "study_description": bool((st.session_state.get("study_description") or st.session_state.get("_p_study_description", "")).strip()),
        "sample_size": int(st.session_state.get("sample_size", 0)) >= 10,
        "qsf_uploaded": bool(preview and preview.success) or bool(st.session_state.get("conversational_builder_complete")),
        "conditions_set": bool(st.session_state.get("selected_conditions") or st.session_state.get("custom_conditions")),
        "primary_outcome": has_primary_outcome,
        "independent_var": has_independent_var,
        "design_ready": bool(st.session_state.get("inferred_design")),
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


def _scroll_then_navigate(target_page: int) -> None:
    """Two-phase navigation: scroll to top first, navigate on next rerun.

    v1.8.5: When called from a button at the bottom of a long page,
    Phase 1 scrolls the current page to the top and reruns.
    Phase 2 (handled at the top of the script) actually navigates.
    This ensures users always see pages start at the top.
    """
    # Persist widget values (same as _navigate_to)
    for _wk in ["study_title", "study_description", "team_name", "team_members_raw"]:
        _wv = st.session_state.get(_wk)
        if _wv is not None:
            st.session_state[f"_p_{_wk}"] = _wv
    st.session_state["_scroll_then_nav_target"] = max(-1, min(target_page, 3))
    st.session_state["_force_scroll_top"] = True
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
        title = st.session_state.get("study_title") or st.session_state.get("_p_study_title", "")
        if title:
            return f'"{title[:28]}…"' if len(title) > 28 else f'"{title}"'
        return ""
    elif idx == 1:
        if st.session_state.get("conversational_builder_complete"):
            return "Builder complete"
        if st.session_state.get("qsf_preview"):
            return "QSF uploaded"
        return ""
    elif idx == 2:
        conds = st.session_state.get("selected_conditions", [])
        n = len(conds)
        if n:
            return f"{n} condition{'s' if n != 1 else ''}"
        return ""
    elif idx == 3:
        return "Data ready" if st.session_state.get("has_generated") else ""
    return ""


# ── Flow navigation CSS (v1.7.0 — Professional minimal design) ───────
_FLOW_NAV_CSS = """<style>
/* === v1.8.0: Premium landing page + segmented progress === */

/* Base layout */
section.main .block-container {
    max-width: 960px;
    padding-top: 1.2rem;
}
section.main h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; color: #111827; }
section.main h2 { font-size: 1.35rem; font-weight: 600; color: #1F2937; }
section.main h3 { font-size: 1.1rem; font-weight: 600; color: #1F2937; }
section.main h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.4rem; color: #374151; }
section[data-testid="stSidebar"] .stCaption { line-height: 1.4; }

/* ─── Landing page hero ─── */
.landing-hero {
    text-align: center;
    padding: 56px 20px 36px;
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
    margin: 0 0 16px 0;
    line-height: 1.1;
    background: linear-gradient(135deg, #111827 0%, #374151 50%, #111827 100%);
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
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    background: #F9FAFB;
    border-radius: 10px;
    font-size: 1.3rem;
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
    color: #D1D5DB;
    font-size: 16px;
    margin-top: 10px;
    flex-shrink: 0;
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
    padding: 24px 0 8px;
    font-size: 0.75rem;
    color: #D1D5DB;
}

/* ─── Segmented progress bar ─── */
.seg-progress {
    display: flex;
    gap: 4px;
    padding: 14px 0 4px;
}
.seg-bar {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: #E5E7EB;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
}
.seg-bar.active {
    background: #FF4B4B;
    box-shadow: 0 0 8px rgba(255,75,75,0.2);
}
.seg-bar.done {
    background: #22C55E;
}
.seg-bar:hover {
    transform: scaleY(1.5);
    opacity: 0.85;
}
.step-label {
    text-align: center;
    font-size: 0.82rem;
    color: #6B7280;
    padding: 6px 0 10px;
    font-weight: 500;
}
.step-label strong {
    color: #1F2937;
}

/* ─── Section wrapper ─── */
.flow-section {
    padding-top: 4px;
    animation: sectionEnter 0.3s ease-out;
}
@keyframes sectionEnter {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Section guidance tagline */
.section-guide {
    font-size: 13px; color: #6B7280;
    margin: 0 0 16px 0;
    padding: 10px 16px;
    background: linear-gradient(135deg, #FAFBFF 0%, #F8FAFC 100%);
    border-radius: 8px;
    border-left: 3px solid #FF4B4B;
}

/* Section complete banner */
.section-done-banner {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border: 1px solid #D1FAE5;
    border-radius: 10px;
    margin: 14px 0 6px 0;
    font-size: 13px; color: #166534; font-weight: 500;
}

/* Design page cards */
.design-card {
    padding: 16px 18px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    margin: 8px 0;
    transition: border-color 0.15s ease;
}
.design-card:hover {
    border-color: #D1D5DB;
}
.design-card-header {
    font-size: 0.9rem; font-weight: 600;
    color: #1F2937; margin: 0 0 8px 0;
}
.config-status {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; padding: 2px 8px;
    border-radius: 12px;
}
.config-status.ready {
    background: #F0FDF4; color: #166534;
    border: 1px solid #D1FAE5;
}
.config-status.needed {
    background: #FEF3C7; color: #92400E;
    border: 1px solid #FDE68A;
}

/* Nav button styling */
.nav-btn-row .stButton button {
    font-size: 0 !important;
    height: 2px !important; min-height: 2px !important;
    padding: 0 !important; margin: 0 !important;
    border: none !important; background: transparent !important;
    box-shadow: none !important; opacity: 0 !important;
    overflow: hidden !important;
}

/* Compact feedback link */
.feedback-bar {
    display: flex; align-items: center; justify-content: center;
    gap: 8px; padding: 8px 16px; margin-top: 20px;
    border-top: 1px solid #F3F4F6;
    font-size: 11px; color: #9CA3AF;
}
.feedback-bar a { color: #6B7280; text-decoration: none; }
.feedback-bar a:hover { color: #FF4B4B; }

/* ─── Wizard page nav buttons ─── */
.stButton button[kind="secondary"] {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 20px !important;
}
.stButton button[kind="secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    border-color: #9CA3AF !important;
}
.stButton button[kind="primary"] {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 20px !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(255,75,75,0.3) !important;
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
</style>"""


def _render_flow_nav(active: int, done: List[bool]) -> None:
    """Render modern segmented progress bar with step label.

    v1.8.0: Thin segmented bar replaces dots+lines stepper.
    Clickable segments for navigation, centered step label below.
    """
    while len(done) < len(SECTION_META):
        done.append(False)

    # Build segmented progress bar
    segments_html = '<div class="seg-progress">'
    for i in range(len(SECTION_META)):
        if i < active and done[i]:
            cls = "done"
        elif i == active:
            cls = "active"
        else:
            cls = ""
        segments_html += f'<div class="seg-bar {cls}" data-step="{i}"></div>'
    segments_html += '</div>'
    st.markdown(segments_html, unsafe_allow_html=True)

    # Step label with completion summary
    meta = SECTION_META[active]
    n_done = sum(1 for d in done if d)
    completion_text = f" \u00b7 {n_done}/{len(SECTION_META)} complete" if n_done > 0 else ""
    st.markdown(
        f'<div class="step-label">Step {active + 1} of {len(SECTION_META)}'
        f' \u2014 <strong>{meta["title"]}</strong>{completion_text}</div>',
        unsafe_allow_html=True,
    )

    # Invisible clickable buttons for navigation (same technique, just styled differently)
    st.markdown('<div class="nav-btn-row">', unsafe_allow_html=True)
    nav_cols = st.columns(len(SECTION_META), gap="small")
    for i, meta in enumerate(SECTION_META):
        with nav_cols[i]:
            if st.button(
                meta["title"],
                key=f"nav_{i}",
                type="secondary",
                use_container_width=True,
            ):
                if i != active:
                    _navigate_to(i)
    st.markdown('</div>', unsafe_allow_html=True)


_SCROLL_TO_TOP_JS = """<script>
(function() {
    var doc = window.parent.document;
    var selectors = [
        'section.main',
        '[data-testid="stAppViewContainer"]',
        '[data-testid="stVerticalBlock"]',
        '.main .block-container',
        '.main'
    ];
    function scrollUp() {
        for (var i = 0; i < selectors.length; i++) {
            var els = doc.querySelectorAll(selectors[i]);
            for (var j = 0; j < els.length; j++) {
                els[j].scrollTop = 0;
            }
        }
        window.parent.scrollTo(0, 0);
        try { doc.body.scrollTop = 0; } catch(e) {}
        try { doc.documentElement.scrollTop = 0; } catch(e) {}
    }
    // Fire aggressively: 14 times over 3s to beat Streamlit's scroll restoration.
    // Streamlit reruns can restore scroll position AFTER initial render, so we
    // need to keep firing well past the initial widget render window.
    var delays = [0, 10, 30, 60, 100, 150, 250, 400, 600, 900, 1300, 1800, 2400, 3000];
    for (var d = 0; d < delays.length; d++) {
        setTimeout(scrollUp, delays[d]);
    }
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
    }
    [0,50,150,350,700,1200,2000,3000].forEach(function(d){setTimeout(s,d);});
})();
</script>"""


def _inject_scroll_to_top_js() -> None:
    """Inject scroll-to-top via both components.html and inline markdown.

    v1.8.4: Double injection strategy — uses both components.html (fires in
    an iframe) and inline markdown (fires in the parent frame). This ensures
    scroll resets even when one method is blocked or delayed by Streamlit's
    rendering pipeline. Fires 14+ times over 3 seconds.
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
    st.caption(
        "Full control over demographics, exclusions, and effect sizes."
        if advanced_mode else
        "Unlocks demographics, exclusions, and effect size controls."
    )

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

# ── Two-phase navigation: scroll to top FIRST, then navigate ──────
# v1.8.5: When user clicks a forward button at the bottom of a page,
# Phase 1: scroll to top + rerun (so user sees current page scroll up)
# Phase 2: on next rerun, actually navigate to the target page
_scroll_then_nav = st.session_state.pop("_scroll_then_nav_target", None)
if _scroll_then_nav is not None:
    # Phase 2: now actually navigate
    st.session_state["_pending_nav"] = _scroll_then_nav
    st.session_state["_page_just_changed"] = True
    st.session_state["_force_scroll_top"] = True
    if _scroll_then_nav == 2:
        st.session_state["_page_just_changed_design"] = True

# ── Apply pending navigation (from auto-advance or _navigate_to) ─────
_pending_nav = st.session_state.pop("_pending_nav", None)
if _pending_nav is not None:
    st.session_state["active_page"] = _pending_nav

if "active_page" not in st.session_state:
    st.session_state["active_page"] = -1  # Landing page
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
    _step_completion["conditions_set"] and _step_completion["design_ready"],
    bool(st.session_state.get("has_generated")),
]

# ── Flow navigation (v1.8.0: segmented progress bar, only on wizard pages) ──
if active_page >= 0:
    _render_flow_nav(active_page, _step_done)


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

    st.markdown('<div style="max-width:780px;margin:0 auto;padding:0 20px;"><hr style="border:none;border-top:1px solid #F3F4F6;margin:0;"></div>', unsafe_allow_html=True)

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

    # Methods PDF download — right after how-it-works
    methods_pdf_path = Path(__file__).resolve().parent.parent / "docs" / "papers" / "methods_summary.pdf"
    if methods_pdf_path.exists():
        _pdf_col1, _pdf_col2, _pdf_col3 = st.columns([1, 2, 1])
        with _pdf_col2:
            st.download_button(
                "Download Methods PDF",
                data=methods_pdf_path.read_bytes(),
                file_name=methods_pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )

    # Primary CTA — right after how-it-works + PDF
    _cta_col1, _cta_col2, _cta_col3 = st.columns([1, 2, 1])
    with _cta_col2:
        if st.button("Start Your Simulation  \u2192", type="primary", use_container_width=True, key="landing_cta"):
            _navigate_to(0)

    # ── Reference sections (expandable, at the bottom) ──
    st.markdown('<div style="max-width:780px;margin:24px auto 0;padding:0 20px;"><hr style="border:none;border-top:1px solid #F3F4F6;margin:0;"></div>', unsafe_allow_html=True)

    with st.expander("Learn more about each capability"):
        st.markdown("""
**Test Before You Collect**
Generate a publication-ready CSV with realistic Likert-scale responses, attention check failures,
individual differences, and demographic distributions. The data mirrors real Qualtrics output format
so your analysis scripts work identically on both simulated and real data.

**Realistic Open-Ended Responses**
Uses a multi-provider LLM failover chain with 50+ behavioral personas
to generate unique, context-aware free-text responses. Each response aligns with the participant's
numeric ratings and assigned persona characteristics. Supports 225+ research domains and 40 question types.

**Ready-to-Run Analysis Code**
Automatically generates scripts in R, Python, Julia, SPSS, and Stata — tailored to your specific
experimental design. Includes data loading, variable coding, condition comparisons, and appropriate
statistical tests (t-tests, ANOVAs, regressions, mediation analyses).

**Built for Research & Teaching**
Instructor-only reports include detailed statistical analysis with effect sizes, power estimates,
and visualization charts that students don't see. Group management tracks team usage.
Pre-registration consistency checks compare your design against OSF, AEA, or AsPredicted specifications.
""")

    with st.expander("Step-by-step details"):
        st.markdown("""
**Step 1 — Name Your Study:** Enter your study title and a description of your experiment's
purpose, manipulation, and main outcomes. This information is embedded in all generated outputs
(data files, analysis scripts, reports). Optionally add team member names for group tracking.

**Step 2 — Provide Your Design:** Upload a Qualtrics .qsf file for automatic detection of conditions,
scales, randomizers, and embedded data. Or describe your study in plain text using the guided builder —
it parses natural language descriptions of your experimental design.

**Step 3 — Configure Experiment:** Review auto-detected conditions, add custom conditions, configure
factorial designs (2x2, 2x3, etc.), set sample sizes with per-condition allocation, and verify
dependent variables. The tool detects matrix scales, Likert items, sliders, and numeric inputs.

**Step 4 — Generate & Download:** Choose a difficulty level (easy to expert) that controls noise,
attention check failure rates, and response quality. Generate your complete data package and download
a ZIP containing the CSV, codebook, analysis scripts in 5 languages, summary reports, and metadata.
""")

    with st.expander("Research foundations & citations"):
        st.markdown(
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

            '</div>',
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        f'<div class="landing-footer">v{APP_VERSION} &middot; February 2026</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 1: STUDY SETUP
# =====================================================================
if active_page == 0:
    st.markdown('<div class="flow-section">', unsafe_allow_html=True)

    # v1.8.0: Hero card removed — now on landing page
    st.markdown(
        '<div class="section-guide">Let\'s set up your study. Enter a title and description — '
        'these appear in your final report, data outputs, and analysis scripts.</div>',
        unsafe_allow_html=True,
    )

    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]

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

    # v1.7.0: Persist widget values immediately after rendering
    if study_title:
        st.session_state["_p_study_title"] = study_title
    if study_description:
        st.session_state["_p_study_description"] = study_description

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

    # Navigation
    if step1_done:
        st.markdown(
            '<div class="section-done-banner">\u2713 Setup complete</div>',
            unsafe_allow_html=True,
        )
    else:
        _missing = []
        if not completion["study_title"]:
            _missing.append("study title")
        if not completion["study_description"]:
            _missing.append("study description")
        if _missing:
            st.caption(f"Fill in {' and '.join(_missing)} to continue.")

    _nav_left, _nav_right = st.columns([1, 1])
    with _nav_left:
        if st.button("\u2190 Home", key="back_0", type="secondary"):
            _navigate_to(-1)
    with _nav_right:
        if step1_done:
            if st.button("Study Input \u2192", key="auto_advance_0", type="primary", use_container_width=True):
                _scroll_then_navigate(1)

    # v1.8.0: Research citations moved to landing page

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 2: FILE UPLOAD / STUDY BUILDER
# =====================================================================
if active_page == 1:
    st.markdown('<div class="flow-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-guide">Provide your experiment design — upload a Qualtrics '
        '.qsf export for automatic detection, or describe your study in plain text.</div>',
        unsafe_allow_html=True,
    )
    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]
    step2_done = completion["qsf_uploaded"]

    if not step1_done:
        st.warning("Complete **Setup** first — enter your study title and description, then return here.")

    # v1.6.0: Cleaner mode selector with descriptive labels
    _mode_options = {
        "upload_qsf": "Upload QSF file",
        "describe_study": "Describe in words",
    }
    input_mode = st.radio(
        "How would you like to provide your study design?",
        options=list(_mode_options.keys()),
        format_func=lambda x: _mode_options[x],
        index=0 if st.session_state.get("study_input_mode", "upload_qsf") == "upload_qsf" else 1,
        key="study_input_mode_radio",
        horizontal=True,
        help="No QSF? Choose 'Describe in words' to set up your experiment through a guided form.",
    )
    st.session_state["study_input_mode"] = input_mode

    # PATH A: CONVERSATIONAL STUDY BUILDER
    if input_mode == "describe_study":
        _render_conversational_builder()

    # PATH B: QSF FILE UPLOAD
    _show_qsf_upload = (input_mode == "upload_qsf")

    if _show_qsf_upload:
        st.markdown("#### Upload Qualtrics Survey File")

    # QSF file upload section (only shown in upload_qsf mode)
    if _show_qsf_upload:
        existing_qsf_name = st.session_state.get("qsf_file_name")
        existing_qsf_content = st.session_state.get("qsf_raw_content")

        col_qsf, col_help = st.columns([2, 1])

        with col_qsf:
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

        with col_help:
            with st.expander("How to export from Qualtrics", expanded=False):
                st.markdown("""
1. Open your survey in Qualtrics
2. Click **Tools** → **Import/Export**
3. Select **Export Survey**
4. Download the .qsf file
5. Upload it here
""")

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
    st.markdown("---")
    st.markdown("### 2. Survey Materials (Optional)")
    st.caption("Upload PDF exports, screenshots, or photos of your survey for better question identification and simulation output quality.")

    with st.expander("How to create a Survey PDF export from Qualtrics", expanded=False):
        st.markdown("""
**Steps to export your survey as PDF:**

1. Open your survey in Qualtrics
2. Click **Tools** in the top menu
3. Select **Import/Export** → **Print Survey**
4. In the print preview, click **Save as PDF** (or use your browser's print-to-PDF feature)
5. Upload the saved PDF file here

**Why this helps:**
- Provides exact question wording for better simulation accuracy
- Helps identify response scale labels and anchors
- Improves the quality of simulated open-ended responses
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
    st.markdown("---")
    st.markdown("### 3. Preregistration Details (Optional)")
    st.caption("Upload your AsPredicted or preregistration PDF(s) to improve simulation quality. The tool uses this to better understand your hypotheses and variables.")

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
        st.markdown("#### QSF Analysis Results")
        _n_questions = int(getattr(preview, "total_questions", 0) or 0)
        _n_scales = int(len(getattr(preview, "detected_scales", []) or []))
        _n_candidates = int(len(st.session_state.get("condition_candidates", []) or []))
        warnings = getattr(preview, "validation_warnings", []) or []
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Questions", _n_questions)
        c2.metric("Scales detected", _n_scales)
        c3.metric("Condition candidates", _n_candidates)
        c4.metric("Conditions selected", int(len(selected_conditions)))
        c5.metric("Warnings", int(len(warnings)))
        if _n_questions > 0 and not warnings:
            st.success(
                f"QSF parsed successfully: {_n_questions} questions, {_n_scales} scales, "
                f"{_n_candidates} condition candidates. Proceed to the **Design** tab to configure."
            )

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

    # v1.6.1: Section-complete banner with prominent CTA
    if step2_done:
        st.markdown(
            '<div class="section-done-banner">\u2713 Study input complete — ready to configure your design</div>',
            unsafe_allow_html=True,
        )

    _nav_left1, _nav_right1 = st.columns([1, 1])
    with _nav_left1:
        if st.button("\u2190 Setup", key="back_1", type="secondary"):
            _navigate_to(0)
    with _nav_right1:
        if step2_done:
            if st.button("Design \u2192", key="auto_advance_1", type="primary", use_container_width=True):
                _scroll_then_navigate(2)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 3: DESIGN CONFIGURATION
# =====================================================================
if active_page == 2:
    # v1.8.4: Extra scroll-to-top for the heavy Design page (widgets render async)
    if st.session_state.pop("_page_just_changed_design", None):
        _inject_scroll_to_top_js()
    st.markdown('<div class="flow-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-guide">Review and configure your experimental conditions, '
        'dependent variables, and sample allocation.</div>',
        unsafe_allow_html=True,
    )
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
            st.warning(
                "Complete the **Study Input** step first — describe your conditions and scales, "
                "then click **Build Study Specification** to proceed."
            )
        else:
            st.warning(
                "Complete the **Study Input** step first — upload a QSF file or describe your study "
                "using the builder."
            )
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

        if not _design_ready:
            _missing = []
            if not _design_conds:
                _missing.append("conditions")
            if not _design_scales:
                _missing.append("DVs/scales")
            if not st.session_state.get("scales_confirmed", False) and _design_scales:
                _missing.append("scale confirmation")
            if _missing:
                st.caption(f"Still needed: {', '.join(_missing)}")

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
            st.info("No conditions auto-detected from the QSF file. Use the options below to add conditions.")
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
                    st.write("")  # Spacer
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
                            _navigate_to(2)

            st.markdown("**Type manually**")

            col_add1, col_add2 = st.columns([3, 1])
            with col_add1:
                new_condition = st.text_input(
                    "New condition name",
                    key="new_condition_input",
                    placeholder="e.g., Control, Treatment, High, Low",
                    help="Type any condition name you want to add.",
                )
            with col_add2:
                st.write("")  # Spacer
                if st.button("Add →", key="add_condition_btn", disabled=not new_condition.strip()):
                    if new_condition.strip() and new_condition.strip() not in custom_conditions:
                        custom_conditions.append(new_condition.strip())
                        st.session_state["custom_conditions"] = custom_conditions
                        _navigate_to(2)

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
                            _navigate_to(2)

        # Combine selected and custom conditions
        custom_conditions = st.session_state.get("custom_conditions", [])
        all_conditions = list(dict.fromkeys(selected + custom_conditions))

        # Show final conditions summary
        st.markdown("")
        if all_conditions:
            # Display cleaned condition names
            clean_names = [_clean_condition_name(c) for c in all_conditions]
            st.success(f"**{len(all_conditions)} condition(s) configured:** {', '.join(clean_names)}")
        else:
            st.error("No conditions defined. Please go back and select or add at least 2 conditions.")
            all_conditions = []

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
                if old_total > 0:
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
                    if len(display_name) > 22:
                        display_name = display_name[:20] + "..."

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
                        help=f"Number of participants in '{display_name}'"
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
                    _navigate_to(2)
            else:
                st.success(f"✓ Allocations sum to {sample_size}")
                st.session_state["condition_allocation_n"] = new_allocation_n

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
            st.info("Each condition is treated as an independent level of a single factor.")
            clean_conditions = [_clean_condition_name(c) for c in all_conditions]
            factors = [{"name": "Condition", "levels": all_conditions}]

            # Display the conditions cleanly
            st.markdown("**Conditions:**")
            for i, (orig, clean) in enumerate(zip(all_conditions, clean_conditions)):
                st.markdown(f"  {i+1}. {clean}")

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
        st.caption("Verify each DV matches your QSF survey. Adjust names, scale ranges, or remove incorrect detections.")

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
            st.markdown(f"**{len(confirmed_scales)} DV(s) detected from your QSF.** Review and adjust as needed:")

            # v1.2.0: Column headers for clarity with tooltips
            hdr1, hdr2, hdr3a, hdr3b, hdr4, hdr5 = st.columns([2.5, 0.8, 0.6, 0.6, 1.5, 0.4])
            with hdr1:
                st.markdown(
                    '<span title="The variable name as it appears in your QSF file. You can edit this to match your analysis script.">**Variable Name**</span>',
                    unsafe_allow_html=True
                )
            with hdr2:
                st.markdown(
                    '<span title="Number of items/questions in this scale. Multi-item scales (e.g., 5-item attitude measure) will generate averaged composite scores.">**Items**</span>',
                    unsafe_allow_html=True
                )
            with hdr3a:
                st.markdown(
                    '<span title="Minimum value on the response scale (e.g., 1 for a 1-7 scale, 0 for a 0-100 slider).">**Min**</span>',
                    unsafe_allow_html=True
                )
            with hdr3b:
                st.markdown(
                    '<span title="Maximum value on the response scale (e.g., 7 for a 1-7 Likert scale, 100 for a 0-100 slider).">**Max**</span>',
                    unsafe_allow_html=True
                )
            with hdr4:
                st.markdown(
                    '<span title="Detection type: Matrix=multi-item battery, Slider=visual analog scale, Single Item=standalone question, Numeric=open-ended number input.">**Type**</span>',
                    unsafe_allow_html=True
                )
            with hdr5:
                st.caption("Del")

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
                            # Numeric inputs can have any range
                            new_scale_min = st.number_input(
                                "Min",
                                min_value=0,
                                max_value=1000,
                                value=current_min,
                                key=f"dv_min_v{dv_version}_{i}",
                                help="Minimum value (e.g., 0 for slider, 1 for Likert)"
                            )
                        else:
                            new_scale_min = st.number_input(
                                "Min",
                                min_value=0,
                                max_value=100,
                                value=current_min,
                                key=f"dv_min_v{dv_version}_{i}",
                                help="Minimum scale value (usually 0 or 1)"
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
            st.info(
                "**No DVs detected from QSF.** This can happen if your survey uses unconventional question formats. "
                "Click **Add DV** below to manually define your dependent variables."
            )

        # Handle removals
        if scales_to_remove:
            st.session_state["confirmed_scales"] = updated_scales
            st.session_state["_dv_version"] = dv_version + 1
            _navigate_to(2)

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
            _navigate_to(2)

        # Update session state with edited DVs
        st.session_state["confirmed_scales"] = updated_scales
        scales = updated_scales if updated_scales else scales

        # Confirmation checkbox with better help
        scales_confirmed = st.checkbox(
            "I confirm these DVs match my survey",
            value=st.session_state.get("scales_confirmed", False),
            key=f"dv_confirm_checkbox_v{dv_version}",
            help="Check this box once you have verified that the variable names, item counts, and scale ranges match your actual Qualtrics survey."
        )
        st.session_state["scales_confirmed"] = scales_confirmed

        if not scales_confirmed and updated_scales:
            st.caption("Please verify the DVs above match your survey, then check the confirmation box to proceed.")

        # ========================================
        # STEP 5: OPEN-ENDED QUESTIONS
        # ========================================
        st.markdown("")
        _oe_count_display = len(st.session_state.get("confirmed_open_ended", []))
        _oe_label = f"#### Open-Ended Questions ({_oe_count_display})" if _oe_count_display > 0 else "#### Open-Ended Questions"
        st.markdown(_oe_label)

        with st.expander(
            f"Review & edit open-ended questions ({_oe_count_display})"
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
                st.markdown(f"**{len(confirmed_open_ended)} open-ended question(s):**")
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
                            oe_to_remove.append(i)
                    q_text = oe.get("question_text", "")
                    if q_text:
                        st.caption(f"*\"{q_text[:80]}{'...' if len(q_text) > 80 else ''}\"*")
                    if var_name.strip() and i not in oe_to_remove:
                        updated_open_ended.append({
                            "variable_name": var_name.strip(),
                            "name": var_name.strip(),
                            "question_text": oe.get("question_text", ""),
                            "source_type": source_type,
                            "force_response": oe.get("force_response", False),
                            "context_type": oe.get("context_type", "general"),
                            "min_chars": oe.get("min_chars"),
                            "block_name": oe.get("block_name", ""),
                        })
            else:
                st.info("No open-ended questions detected. Add any below.")

            if oe_to_remove:
                st.session_state["confirmed_open_ended"] = updated_open_ended
                st.session_state["_oe_version"] = oe_version + 1
                _navigate_to(2)

            if st.button("Add Open-Ended Question", key=f"add_oe_btn_v{oe_version}"):
                new_oe = {
                    "variable_name": f"OpenEnded_{len(confirmed_open_ended)+1}",
                    "name": f"OpenEnded_{len(confirmed_open_ended)+1}",
                    "question_text": "", "source_type": "manual",
                    "force_response": False, "context_type": "general",
                }
                confirmed_open_ended.append(new_oe)
                st.session_state["confirmed_open_ended"] = confirmed_open_ended
                st.session_state["_oe_version"] = oe_version + 1
                _navigate_to(2)

            # Only update confirmed_open_ended if there were actual changes
            # This prevents silent data loss from rendering edge cases
            if updated_open_ended or not confirmed_open_ended:
                st.session_state["confirmed_open_ended"] = updated_open_ended
            elif confirmed_open_ended and not updated_open_ended:
                # User cleared all variable names — warn but don't silently delete
                st.warning("All open-ended question names are empty. Fill in names or remove them explicitly.")

            open_ended_confirmed = st.checkbox(
                "I confirm these open-ended questions match my survey",
                value=st.session_state.get("open_ended_confirmed", False),
                key=f"oe_confirm_checkbox_v{oe_version}",
            )
            st.session_state["open_ended_confirmed"] = open_ended_confirmed

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
        st.markdown("---")

        display_conditions = all_conditions
        if st.session_state.get("use_crossed_conditions") and st.session_state.get("factorial_crossed_conditions"):
            display_conditions = st.session_state["factorial_crossed_conditions"]

        confirmed_oe = st.session_state.get("confirmed_open_ended", [])
        oe_count = len(confirmed_oe)

        _sm1, _sm2, _sm3 = st.columns(3)
        _sm1.metric("Conditions", len(display_conditions))
        _sm2.metric("DVs", len(scales))
        _sm3.metric("Sample N", st.session_state.get("sample_size", 0))

        # Validate and lock design - require conditions, scales, scale confirmation, and correct allocation
        _alloc_n = st.session_state.get("condition_allocation_n", {})
        _alloc_sum = sum(_alloc_n.values()) if _alloc_n else 0
        _alloc_ok = not _alloc_n or _alloc_sum == sample_size  # OK if no custom allocation or if sums match
        design_valid = len(display_conditions) >= 1 and len(scales) >= 1 and scales_confirmed and _alloc_ok

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

            st.success("✅ Design configuration complete. Proceed to the **Generate** step to run the simulation.")
        else:
            missing_bits = []
            if not all_conditions:
                missing_bits.append("conditions")
            if not scales:
                missing_bits.append("scales")
            if not scales_confirmed:
                missing_bits.append("scale confirmation (check the box above)")
            if not _alloc_ok:
                missing_bits.append(f"condition allocation (sums to {_alloc_sum}, need {sample_size})")
            st.error("⚠️ Cannot proceed - missing: " + ", ".join(missing_bits))

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

    # ── Bottom navigation ───────────────────────────────────────────
    _design_can_next = bool(st.session_state.get("inferred_design"))
    if _design_can_next:
        st.markdown(
            '<div class="section-done-banner">\u2713 Design configured — ready to generate</div>',
            unsafe_allow_html=True,
        )

    _nav_left2, _nav_right2 = st.columns([1, 1])
    with _nav_left2:
        if st.button("\u2190 Study Input", key="back_2", type="secondary"):
            _navigate_to(1)
    with _nav_right2:
        if _design_can_next:
            if st.button("Generate \u2192", key="auto_advance_2", type="primary", use_container_width=True):
                _scroll_then_navigate(3)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 4: GENERATE SIMULATION
# =====================================================================
if active_page == 3:
    st.markdown('<div class="flow-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-guide">Choose your difficulty level, review the design summary, '
        'and generate your simulated dataset.</div>',
        unsafe_allow_html=True,
    )
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
        _gen_input_mode = st.session_state.get("study_input_mode", "upload_qsf")
        if _gen_input_mode == "describe_study":
            st.warning(
                "No experiment design configured yet. Please complete **Study Input** "
                "to describe your conditions and scales, click **Build Study Specification**, "
                "then review your design in **Design** before generating data."
            )
        else:
            st.warning(
                "No experiment design configured yet. Please complete **Study Input** "
                "to upload a QSF file (or describe your study), then configure your "
                "design in **Design** before generating data."
            )
        col_back1, col_back2 = st.columns([1, 1])
        with col_back1:
            if st.button("Go to Study Input", key="go_step2_from_generate", use_container_width=True):
                _navigate_to(1)
        with col_back2:
            if st.button("Go to Design", key="go_step3_from_generate", use_container_width=True):
                _navigate_to(2)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # v1.6.0: Readiness checklist — compact inline
    _gen_mode = st.session_state.get("study_input_mode", "upload_qsf")
    _input_label = "Study described" if _gen_mode == "describe_study" else "QSF uploaded"
    required_fields = {
        "Study title": bool(st.session_state.get("study_title", "").strip()),
        "Study description": bool(st.session_state.get("study_description", "").strip()),
        "Sample size (\u226510)": int(st.session_state.get("sample_size", 0)) >= 10,
        _input_label: bool(preview and preview.success) or bool(st.session_state.get("conversational_builder_complete")),
        "Design configured": bool(inferred),
    }

    missing_fields = [label for label, ok in required_fields.items() if not ok]
    if missing_fields:
        st.warning(f"Missing: {', '.join(missing_fields)}")
        _fix_cols = st.columns(len(missing_fields), gap="small")
        for _fi, _field in enumerate(missing_fields):
            with _fix_cols[_fi]:
                _target = 0 if _field in ("Study title", "Study description") else (1 if _field == _input_label else 2)
                if st.button(f"\u2190 Fix: {_field}", key=f"fix_{_fi}_from_gen", use_container_width=True):
                    _navigate_to(_target)

    # Quick summary — metrics row
    conditions = inferred.get('conditions', [])
    scales = st.session_state.get('confirmed_scales', []) or inferred.get('scales', [])
    factors = inferred.get('factors', [])
    scale_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]
    _sample_n = st.session_state.get('sample_size', 0)

    _gc1, _gc2, _gc3, _gc4 = st.columns(4)
    _study_title_display = st.session_state.get('study_title', 'Untitled')
    _gc1.metric("Study", _study_title_display[:40] + ('...' if len(_study_title_display) > 40 else ''))
    _gc2.metric("N", _sample_n)
    _gc3.metric("Conditions", len(conditions))
    _gc4.metric("Scales", len(scale_names))

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
        preview_df = _generate_preview_data(
            conditions=conditions,
            scales=scales,
            open_ended=open_ended_for_preview,
            n_rows=5,
            difficulty=difficulty_level
        )
        st.session_state['preview_df'] = preview_df

    if 'preview_df' in st.session_state and st.session_state['preview_df'] is not None:
        st.dataframe(st.session_state['preview_df'], use_container_width=True, height=200)
        st.caption(f"Preview generated at difficulty level: **{DIFFICULTY_LEVELS[difficulty_level]['name']}**")

    st.markdown("")

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

        with st.expander("View settings (based on difficulty level)"):
            st.markdown("""
These settings are locked to ensure all simulations are comparable across teams.
To customize these parameters, enable **Advanced mode** in the sidebar.
""")
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

    # v1.4.14: Hide design summary during generation AND after generation is complete.
    # After generation, the download section is the primary focus.
    if not _is_generating and not _has_generated:
        st.markdown("")
        st.markdown("#### Final Design Summary")
        st.caption("Review your experimental design before generating data.")

    # Get all relevant design info (needed for both display and generation)
    display_conditions = conditions
    if st.session_state.get("use_crossed_conditions") and st.session_state.get("factorial_crossed_conditions"):
        display_conditions = st.session_state["factorial_crossed_conditions"]

    confirmed_oe = st.session_state.get("confirmed_open_ended", [])
    oe_count = len(confirmed_oe)

    # v1.4.14: Show design summary only when NOT generating AND NOT already generated
    if not _is_generating and not _has_generated:
        # Design metrics row
        summary_cols = st.columns(5)
        summary_cols[0].metric("Conditions", len(display_conditions))
        summary_cols[1].metric("Factors", len(factors))
        summary_cols[2].metric("DVs", len(scales))
        summary_cols[3].metric("Open-Ended", oe_count)

        # Determine design type for display
        n_conds = len(display_conditions)
        if n_conds == 1:
            design_type_str = "Single group"
        elif n_conds == 2:
            design_type_str = "2-group"
        elif n_conds == 4:
            design_type_str = "2×2 factorial"
        elif n_conds == 6:
            design_type_str = "2×3 factorial"
        elif n_conds == 9:
            design_type_str = "3×3 factorial"
        else:
            design_type_str = f"{n_conds}-cell"
        summary_cols[4].metric("Design", design_type_str)

        # Detailed breakdown
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            clean_cond_names = [_clean_condition_name(c) for c in display_conditions]
            if st.session_state.get("use_crossed_conditions"):
                st.markdown(f"**Factorial Conditions:** {', '.join(clean_cond_names[:6])}")
                if len(clean_cond_names) > 6:
                    st.markdown(f"  _{len(clean_cond_names) - 6} more..._")
            else:
                st.markdown(f"**Conditions:** {', '.join(clean_cond_names[:6])}")
                if len(clean_cond_names) > 6:
                    st.markdown(f"  _{len(clean_cond_names) - 6} more..._")

            # DVs
            dv_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]
            st.markdown(f"**DVs:** {', '.join(dv_names[:5]) if dv_names else 'Main_DV (default)'}")
            if len(dv_names) > 5:
                st.markdown(f"  _{len(dv_names) - 5} more..._")

        with detail_col2:
            # Open-ended questions
            if oe_count > 0:
                oe_names = [oe.get('variable_name', oe.get('name', '')) for oe in confirmed_oe[:5]]
                oe_display = ', '.join(oe_names)
                st.markdown(f"**Open-Ended:** {oe_display}")
                if oe_count > 5:
                    st.markdown(f"  _{oe_count - 5} more..._")
            else:
                st.markdown("**Open-Ended:** None configured")

            # Sample info
            N = st.session_state.get("sample_size", 200)
            n_per_cell = N // max(1, len(display_conditions))
            st.markdown(f"**Sample:** {N} total (~{n_per_cell} per condition)")

            # Effect size only if user specified
            if st.session_state.get("add_effect_checkbox", False):
                effect_d = st.session_state.get("effect_cohens_d", 0.5)
                st.markdown(f"**Effect Size:** d = {effect_d:.2f}")

    # ========================================
    # GENERATE BUTTON - with proper state management
    # v1.4.5: Progress spinner shown FIRST, design summary hidden during generation
    # ========================================
    st.markdown("")

    # v1.4.9: LLM status indicator with multi-provider support and single key input
    _has_open_ended = bool(st.session_state.get("inferred_design", {}).get("open_ended", []))
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
            _prov_display = _llm_status.get("provider_display", "AI")
            st.markdown(
                '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                'padding:12px 16px;margin-bottom:16px;">'
                '<span style="font-size:1.1em;font-weight:600;color:#166534;">'
                'AI-Powered Open-Ended Responses: Active</span><br>'
                '<span style="color:#15803d;font-size:0.9em;">'
                'Open-ended responses will be generated by an LLM tailored to your specific '
                'survey questions and experimental conditions.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            # LLM unavailable — show single unified key input
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;'
                'padding:12px 16px;margin-bottom:12px;">'
                '<span style="font-size:1.1em;font-weight:600;color:#92400e;">'
                'AI-Powered Open-Ended Responses: Unavailable</span><br>'
                '<span style="color:#a16207;font-size:0.9em;">'
                'The built-in AI service is currently at capacity. '
                'You can provide your own free API key below, '
                'or proceed with high-quality template-based responses.</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Use your own free API key (optional)", expanded=False):
                st.markdown(
                    "**Get a free key in 30 seconds** from any of these providers:\n\n"
                    "| Provider | Free Tier | Sign Up |\n"
                    "|----------|-----------|--------|\n"
                    "| **Groq** (recommended) | 14,400 req/day | [console.groq.com](https://console.groq.com) |\n"
                    "| **Cerebras** | 1M tokens/day | [cloud.cerebras.ai](https://cloud.cerebras.ai) |\n"
                    "| **OpenRouter** | Free models | [openrouter.ai](https://openrouter.ai) |\n\n"
                    "Simply paste your key below — the provider is detected automatically."
                )
                _user_key_input = st.text_input(
                    "API Key",
                    value=st.session_state.get("user_llm_api_key", ""),
                    type="password",
                    key="user_llm_key_input",
                    placeholder="Paste your key here (e.g., gsk_..., csk-..., sk-or-...)",
                )
                if _user_key_input != st.session_state.get("user_llm_api_key", ""):
                    st.session_state["user_llm_api_key"] = _user_key_input
                    # Set in environment for the generator to pick up (only if non-empty)
                    if _user_key_input.strip():
                        os.environ["LLM_API_KEY"] = _user_key_input.strip()
                    elif "LLM_API_KEY" in os.environ:
                        del os.environ["LLM_API_KEY"]
                    # Clear cached status so it re-checks with the new key
                    st.session_state["_llm_connectivity_status"] = None
                    _navigate_to(3)
                st.markdown(
                    '<span style="color:#6b7280;font-size:0.8em;">'
                    'Your key is used only for this session, kept in memory only, '
                    'and never saved or transmitted anywhere except to the provider\'s API.</span>',
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
        st.success("Simulation complete! Download your files below.")
        # Show generation warnings if any
        _gen_meta = st.session_state.get("generated_metadata", {})
        _gen_warns = _gen_meta.get("generation_warnings", []) if isinstance(_gen_meta, dict) else []
        for _gw in _gen_warns:
            st.warning(_gw)

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
        btn_col1, btn_col2 = st.columns([2, 1])

        with btn_col1:
            if has_generated:
                st.button("Simulation generated", type="primary", disabled=True, use_container_width=True)
            else:
                if st.button("Generate simulated dataset", type="primary", disabled=not can_generate, use_container_width=True, key="generate_dataset_btn"):
                    # v1.3.4: Skip intermediate rerun — go directly to is_generating + phase 1
                    # so the progress spinner appears on the very next render (1 rerun, not 2)
                    st.session_state["generation_requested"] = False
                    st.session_state["is_generating"] = True
                    st.session_state["_generation_phase"] = 1
                    _navigate_to(3)

        with btn_col2:
            # Reset button - allows user to restart after generation
            if has_generated:
                if st.button("Reset & Generate New", use_container_width=True, key="reset_after_gen_btn"):
                    _reset_generation_state()
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
        title = st.session_state.get("study_title", "") or "Untitled Study"
        desc = st.session_state.get("study_description", "") or ""
        requested_n = int(st.session_state.get("sample_size", 200))
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
            study_context=inferred.get("study_context", {}),
            condition_allocation=condition_allocation,
            seed=None,
            mode="pilot" if not st.session_state.get("advanced_mode", False) else "final",
            # v1.0.0: Pass pre-computed visibility map for accurate survey flow simulation
            precomputed_visibility=inferred.get("condition_visibility_map", {}),
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
                    <div style="font-size: 40px; margin-bottom: 10px;">🔄</div>
                    <h3 style="color: white; margin: 0 0 8px 0;">Simulation In Progress</h3>
                    <p style="color: #a0c4e8; font-size: 14px; margin: 0;">Generating realistic behavioral data for your experiment...</p>
                </div>
                """, unsafe_allow_html=True)

            progress_bar.progress(10, text="Step 1/5 — Initializing simulation engine...")

            # Show animated spinner during the actual data generation
            with st.spinner("🧠 Generating participant responses... Please wait."):
                progress_bar.progress(25, text="Step 2/5 — Generating participant responses...")
                df, metadata = engine.generate()

            # v1.4.9: Inject LLM stats into metadata for the instructor report
            if hasattr(engine, 'llm_generator') and engine.llm_generator is not None:
                metadata['llm_stats'] = engine.llm_generator.stats

            # v1.2.4: Run simulation quality validation
            validation_results = _validate_simulation_output(df, metadata, clean_scales)
            st.session_state["_validation_results"] = validation_results
            if not validation_results["passed"]:
                for err in validation_results["errors"]:
                    st.error(err)
                st.warning("⚠️ Simulation validation found issues. Data may need review.")
            for warn in validation_results.get("warnings", []):
                st.warning(warn)

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
            except Exception:
                schema_results = {"passed": True, "checks": [], "warnings": [], "errors": []}

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
                # Standard instructor report (included in download for students)
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
                # COMPREHENSIVE instructor report (for instructor email ONLY - not included in student download)
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
                instructor_html = _markdown_to_html(instructor_report, title=f"Study Summary: {study_title}")
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
                "Study_Summary.md": instructor_bytes,  # Summary report in Markdown
                "Study_Summary.html": instructor_html_bytes,  # Same summary in HTML (easy to view in browser)
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
            st.success("Simulation generated.")
            st.markdown("[Jump to download](#download)")

            if not schema_results.get("valid", True):
                st.error("Schema validation failed. Review Schema_Validation.json in the download.")
            elif schema_results.get("warnings"):
                st.info("Schema validation warnings found. Review Schema_Validation.json in the download.")
            else:
                st.info("Schema validation passed.")

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
                    "Students get Study_Summary.md and Study_Summary.html (browser-viewable) in their download ZIP.\n\n"
                    "INSTRUCTOR ATTACHMENTS:\n"
                    "- INSTRUCTOR_Statistical_Report.html - Full visual report with charts, t-tests,\n"
                    "  ANOVA, Mann-Whitney, chi-squared, regression analysis, and effect sizes.\n"
                    "  Open in any web browser for best viewing.\n"
                    "- INSTRUCTOR_Detailed_Analysis.md - Text-based analysis (Markdown format)\n"
                    "- Student_Study_Summary.md - What students receive (for reference)\n\n"
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
                    "- Study_Summary.md (basic summary in Markdown)\n"
                    "- Study_Summary.html (same summary - opens in any browser)\n"
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
                        ("Student_Study_Summary.md", instructor_bytes),  # What students see (for reference)
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
            # v1.2.3: Enhanced error message with traceback for debugging
            error_tb = _tb.format_exc()
            st.error(f"Simulation failed: {e}")
            with st.expander("Error details (for debugging)", expanded=False):
                st.code(error_tb, language="python")
                # Show input summary for debugging
                st.markdown("**Input Summary:**")
                st.markdown(f"- Scales: {len(clean_scales)} configured")
                st.markdown(f"- Conditions: {inferred.get('conditions', [])}")
                st.markdown(f"- Sample size: {N}")
                if clean_scales:
                    for i, s in enumerate(clean_scales[:5]):
                        st.markdown(f"  - Scale {i+1}: {s.get('name', '?')} (items={s.get('num_items', '?')}, pts={s.get('scale_points', '?')}, min={s.get('scale_min', '?')}, max={s.get('scale_max', '?')})")
            st.session_state["is_generating"] = False
            st.session_state["generation_requested"] = False
            # Don't rerun on error - show error message to user

    zip_bytes = st.session_state.get("last_zip", None)
    df = st.session_state.get("last_df", None)
    if zip_bytes and df is not None:
        st.markdown(
            '<div class="section-done-banner">✓ Simulation complete — download your dataset below</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div id="download"></div>', unsafe_allow_html=True)
        st.markdown("#### Download")

        # v1.4.16: Quick data summary before download button
        _dl_meta = st.session_state.get("last_metadata", {}) or {}
        _dl_cols = st.columns(4)
        _dl_cols[0].metric("Rows", f"{len(df):,}")
        _dl_cols[1].metric("Columns", f"{len(df.columns):,}")
        _dl_n_conds = df["CONDITION"].nunique() if "CONDITION" in df.columns else 0
        _dl_cols[2].metric("Conditions", _dl_n_conds)
        _dl_zip_kb = len(zip_bytes) / 1024
        _dl_cols[3].metric("ZIP Size", f"{_dl_zip_kb:.0f} KB" if _dl_zip_kb < 1024 else f"{_dl_zip_kb/1024:.1f} MB")

        st.download_button(
            "Download ZIP (CSV + metadata + analysis scripts)",
            data=zip_bytes,
            file_name=f"behavioral_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )

        # v1.4.16: ZIP contents list
        st.caption(
            "Includes: Simulated_Data.csv, Data_Codebook_Handbook.txt, "
            "Study_Summary.html, R/Python/Julia/SPSS/Stata scripts, "
            "Metadata.json, Schema_Validation.json"
        )

        with st.expander("Preview (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)

        # v1.2.5: Data Quality Report expander
        quality_checks = st.session_state.get("_quality_checks", [])
        if quality_checks:
            with st.expander("📊 Data Quality Report", expanded=False):
                for check in quality_checks:
                    st.markdown(check)

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

    # v1.4.5: Removed "Done — scroll back to top" button (was unreliable, not needed on last tab)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================
# FEEDBACK (Shown on wizard pages only)
# v1.5.0: Compact feedback — no more bulky expander at top
# v1.8.2: Only show on wizard pages, not landing page
# ========================================
if active_page >= 0:
    _render_feedback_button()
