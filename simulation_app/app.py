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

Email:
- If you want automatic email delivery, set Streamlit secrets:
  - SENDGRID_API_KEY
  - SENDGRID_FROM_EMAIL
  - (optional) SENDGRID_FROM_NAME
  - (optional) INSTRUCTOR_NOTIFICATION_EMAIL
"""

from __future__ import annotations

import base64
import importlib
import io
import json
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

# =============================================================================
# MODULE VERSION VERIFICATION
# =============================================================================
# This section ensures Streamlit Cloud loads the correct module versions.
# Addresses known issue: https://github.com/streamlit/streamlit/issues/366
# Where deeply imported modules don't hot-reload properly.

REQUIRED_UTILS_VERSION = "2.1.6"
BUILD_ID = "20260201-v216-html-reports-stats"  # Change this to force cache invalidation

def _verify_and_reload_utils():
    """Verify utils modules are at correct version, force reload if needed."""
    utils_modules = [
        'utils',
        'utils.qsf_preview',
        'utils.qsf_parser',
        'utils.condition_identifier',
        'utils.enhanced_simulation_engine',
        'utils.group_management',
        'utils.schema_validator',
        'utils.instructor_report',
    ]

    # First, remove all utils modules from sys.modules to force fresh import
    modules_to_remove = [m for m in sys.modules if m.startswith('utils')]
    for mod_name in modules_to_remove:
        del sys.modules[mod_name]

# Force fresh import of utils modules
_verify_and_reload_utils()

from utils.group_management import GroupManager, APIKeyManager
from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult
from utils.schema_validator import validate_schema
from utils.instructor_report import InstructorReportGenerator, ComprehensiveInstructorReport
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

# Verify correct version loaded
if hasattr(utils, '__version__') and utils.__version__ != REQUIRED_UTILS_VERSION:
    st.warning(f"⚠️ Utils version mismatch: expected {REQUIRED_UTILS_VERSION}, got {utils.__version__}")


# -----------------------------
# App constants
# -----------------------------
APP_TITLE = "Behavioral Experiment Simulation Tool"
APP_SUBTITLE = "Fast, standardized pilot simulations from your Qualtrics QSF"
APP_VERSION = "2.1.6"  # Major UX overhaul: factor detection, file names, effect size UI, file persistence, Start Over
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


def _bytes_to_zip(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


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


def _normalize_scale_specs(scales: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize scales to consistent format with deduplication.

    Preserves scale_points from source (QSF or user input) - only defaults when missing.
    Deduplicates by variable name to prevent extra DVs.
    """
    normalized: List[Dict[str, Any]] = []
    seen_names: set = set()  # Track to prevent duplicates

    for scale in scales or []:
        if isinstance(scale, str):
            name = scale.strip()
            if not name:
                continue
            name_key = name.lower().replace(" ", "_").replace("-", "_")
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            normalized.append({
                "name": name,
                "variable_name": name.replace(" ", "_"),
                "num_items": 5,
                "scale_points": 7,
                "reverse_items": []
            })
            continue

        if isinstance(scale, dict):
            name = str(scale.get("name", "")).strip()
            if not name:
                continue

            # Deduplicate by variable name (more precise) or name
            var_name = str(scale.get("variable_name", name)).strip() or name
            name_key = var_name.lower().replace(" ", "_").replace("-", "_")
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

            # Carefully extract scale_points - preserve from source
            raw_points = scale.get("scale_points")
            if raw_points is None or (isinstance(raw_points, float) and np.isnan(raw_points)):
                scale_points = 7
            else:
                try:
                    scale_points = int(raw_points)
                except (ValueError, TypeError):
                    scale_points = 7

            # Same for num_items
            raw_items = scale.get("num_items")
            if raw_items is None or (isinstance(raw_items, float) and np.isnan(raw_items)):
                num_items = 5
            else:
                try:
                    num_items = int(raw_items)
                except (ValueError, TypeError):
                    num_items = 5

            normalized.append(
                {
                    "name": name,
                    "variable_name": var_name.replace(" ", "_"),
                    "num_items": max(1, num_items),
                    "scale_points": max(2, min(11, scale_points)),  # Reasonable bounds
                    "reverse_items": scale.get("reverse_items", []) or [],
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


def _send_email_with_sendgrid(
    to_email: str,
    subject: str,
    body_text: str,
    attachments: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[bool, str]:
    """
    Send an email using SendGrid.

    Returns: (ok, message)
    """
    api_key = st.secrets.get("SENDGRID_API_KEY", "")
    from_email = st.secrets.get("SENDGRID_FROM_EMAIL", "")
    from_name = st.secrets.get("SENDGRID_FROM_NAME", "Behavioral Experiment Simulation Tool")

    if not api_key or not from_email:
        return False, "Missing SENDGRID_API_KEY or SENDGRID_FROM_EMAIL in Streamlit secrets."

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Mail,
            Email,
            To,
            Content,
            Attachment,
            FileContent,
            FileName,
            FileType,
            Disposition,
        )
    except Exception as e:
        return False, f"SendGrid library import failed: {e}"

    mail = Mail(
        from_email=Email(from_email, from_name),
        to_emails=To(to_email),
        subject=subject,
        plain_text_content=Content("text/plain", body_text),
    )

    if attachments:
        for filename, data in attachments:
            encoded = base64.b64encode(data).decode("utf-8")
            att = Attachment(
                FileContent(encoded),
                FileName(filename),
                FileType("application/octet-stream"),
                Disposition("attachment"),
            )
            mail.add_attachment(att)

    try:
        sg = SendGridAPIClient(api_key)
        resp = sg.send(mail)
        status_code = getattr(resp, "status_code", 0)
        if 200 <= int(status_code) < 300:
            return True, f"Sent (status {status_code})."
        return False, f"SendGrid error (status {status_code})."
    except Exception as e:
        return False, f"SendGrid send failed: {e}"


def _render_email_setup_diagnostics() -> None:
    api_key = st.secrets.get("SENDGRID_API_KEY", "")
    from_email = st.secrets.get("SENDGRID_FROM_EMAIL", "")
    from_name = st.secrets.get("SENDGRID_FROM_NAME", "")
    instructor_email = st.secrets.get("INSTRUCTOR_NOTIFICATION_EMAIL", "")

    with st.expander("Email setup diagnostics"):
        st.markdown("**SendGrid configuration**")
        st.markdown(
            "\n".join(
                [
                    f"- API key: {'✅ configured' if api_key else '❌ missing'}",
                    f"- From email: {'✅ configured' if from_email else '❌ missing'}",
                    f"- From name: {'✅ configured' if from_name else 'ℹ️ optional'}",
                    f"- Instructor notification email: {'✅ configured' if instructor_email else 'ℹ️ optional'}",
                ]
            )
        )
        if api_key and not from_email:
            st.warning(
                "SendGrid API key is set, but the sender email is missing. "
                "Add SENDGRID_FROM_EMAIL in Streamlit secrets using a verified sender address."
            )
        elif not api_key:
            st.warning("SendGrid API key is missing. Add SENDGRID_API_KEY in Streamlit secrets.")
        else:
            st.success("SendGrid basics look configured.")


def _clean_condition_name(condition: str) -> str:
    """Remove common suffixes and clean up condition names."""
    import re
    # Remove common suffixes like (new), (copy), etc.
    cleaned = re.sub(r'\s*\(new\)\s*$', '', condition, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(copy\)\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*- copy\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*_\d+$', '', cleaned)  # Remove trailing _1, _2, etc.
    cleaned = re.sub(r'\s*\(\d+\)\s*$', '', cleaned)  # Remove trailing (1), (2), etc.
    # Remove common Qualtrics block prefixes
    cleaned = re.sub(r'^Block\s*\d+\s*[-:]\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^BL_\w+\s*[-:]\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


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
    1. Trying multiple separator patterns
    2. Handling underscore-separated names (e.g., "AI_Hedonic")
    3. Detecting numeric suffixes as potential factor levels
    4. Falling back gracefully to single-factor designs

    Examples:
    - "No AI x Utilitarian", "AI x Hedonic" → 2 factors
    - "Control", "Treatment" → 1 factor
    - "Cond_1_A", "Cond_1_B", "Cond_2_A", "Cond_2_B" → 2 factors
    - "High_Price", "Low_Price" → 1 factor (Price)
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
    # TRY MULTIPLE SEPARATOR PATTERNS
    # ==========================================================
    # Order matters: try more specific separators first
    separator_patterns = [
        " x ",     # Most common factorial separator
        " X ",     # Uppercase version
        " × ",     # Unicode multiplication sign
        " | ",     # Pipe separator
        " / ",     # Slash separator (but not in numbers like 1/2)
        " - ",     # Dash separator (be careful with hyphenated words)
        " * ",     # Asterisk
        " & ",     # Ampersand
        "_x_",     # Underscore x underscore
        "__",      # Double underscore
    ]

    chosen_sep = None
    for sep in separator_patterns:
        # Count how many conditions contain this separator
        matches = sum(1 for c in conditions if sep in c)
        # Need at least 2 conditions with the separator, or all conditions
        if matches >= 2 or matches == len(conditions):
            chosen_sep = sep
            break

    # ==========================================================
    # PARSE USING FOUND SEPARATOR
    # ==========================================================
    if chosen_sep:
        split_rows = [c.split(chosen_sep) for c in conditions]
        max_parts = max(len(r) for r in split_rows)

        # Check if splitting is consistent
        if all(len(r) == max_parts for r in split_rows) and max_parts > 1:
            # Extract unique levels for each factor position
            factors: List[Dict[str, Any]] = []
            for j in range(max_parts):
                levels = sorted(list({r[j].strip() for r in split_rows if r[j].strip()}))
                if not levels:
                    continue
                factor_name = _infer_factor_name(levels)
                # Make factor name unique if needed
                existing_names = [f.get("name") for f in factors]
                if factor_name in existing_names:
                    factor_name = f"{factor_name} {j+1}"
                factors.append({"name": factor_name, "levels": levels})

            if factors:
                return factors

    # ==========================================================
    # TRY UNDERSCORE-BASED PARSING (e.g., "AI_Hedonic", "NoAI_Utilitarian")
    # ==========================================================
    # Only if conditions have underscores and consistent structure
    underscore_rows = [c.split('_') for c in conditions if '_' in c]
    if len(underscore_rows) == len(conditions) and len(underscore_rows) > 1:
        parts_count = [len(r) for r in underscore_rows]
        if len(set(parts_count)) == 1 and parts_count[0] > 1:
            # All conditions split into same number of parts
            max_parts = parts_count[0]
            factors = []
            for j in range(max_parts):
                levels = sorted(list({r[j].strip() for r in underscore_rows if r[j].strip()}))
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
        num_items = s.get("items")
        if num_items is None or (isinstance(num_items, float) and np.isnan(num_items)):
            num_items = 5  # Default only if truly not detected
        else:
            num_items = int(num_items)

        scale_points = s.get("scale_points")
        if scale_points is None or (isinstance(scale_points, float) and np.isnan(scale_points)):
            # IMPORTANT: Default to 7 only when QSF doesn't specify
            # This will be logged for transparency in instructor report
            scale_points = 7
        else:
            scale_points = int(scale_points)

        scales.append(
            {
                "name": display_name,
                "variable_name": name,
                "num_items": max(1, num_items),
                "scale_points": max(2, min(11, scale_points)),  # Reasonable bounds
                "reverse_items": s.get("reverse_items", []) or [],
                "detected_from_qsf": s.get("scale_points") is not None,  # Track source
            }
        )

    # Only add default if NO scales were detected at all
    # This prevents fabricating extra DVs
    if not scales:
        scales = [{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 5, "scale_points": 7, "reverse_items": [], "detected_from_qsf": False}]

    open_ended = getattr(preview, "open_ended_questions", None) or []

    return {
        "conditions": conditions,
        "factors": factors,
        "scales": scales,
        "open_ended_questions": open_ended,
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

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.markdown(
    "Created by Dr. [Eugen Dimant](https://eugendimant.github.io/) · "
    "This tool is designed to make behavioral experiment simulation fast, comparable, and reproducible."
)
st.caption(f"Version {APP_VERSION} · Build {APP_BUILD_TIMESTAMP}")

STEP_LABELS = ["Study Info", "Upload Files", "Design Setup", "Generate"]
STEP_DESCRIPTIONS = [
    "Basic study details",
    "Upload your QSF",
    "Configure design",
    "Run simulation"
]


def _get_step_completion() -> Dict[str, bool]:
    """Get completion status for each step."""
    preview = st.session_state.get("qsf_preview", None)

    # Check variable roles for primary outcome and independent variable
    variable_rows = st.session_state.get("variable_review_rows", [])
    has_primary_outcome = any(r.get("Role") == "Primary outcome" for r in variable_rows)
    has_independent_var = any(r.get("Role") == "Condition" for r in variable_rows)

    return {
        "study_title": bool(st.session_state.get("study_title", "").strip()),
        "study_description": bool(st.session_state.get("study_description", "").strip()),
        "sample_size": int(st.session_state.get("sample_size", 0)) >= 10,
        "qsf_uploaded": bool(preview and preview.success),
        "conditions_set": bool(st.session_state.get("selected_conditions") or st.session_state.get("custom_conditions")),
        "primary_outcome": has_primary_outcome,
        "independent_var": has_independent_var,
        "design_ready": bool(st.session_state.get("inferred_design")),
    }


def _go_to_step(step_index: int) -> None:
    st.session_state["active_step"] = max(0, min(step_index, len(STEP_LABELS) - 1))
    st.rerun()


with st.expander("What this tool delivers", expanded=True):
    st.markdown("""
### Overview

This tool generates **realistic synthetic pilot data** for behavioral experiments. Upload your Qualtrics survey file and receive a complete data package ready for analysis practice.

**Key capabilities:**
- **Automatic survey parsing**: Extracts conditions, factors, and scales from your QSF file
- **Behaviorally realistic responses**: Uses theory-grounded personas to simulate human-like variance including satisficing, extreme responding, and engaged responding
- **Open-ended response generation**: Produces variable, contextually appropriate text responses that align with numeric response patterns
- **Attention checks and exclusions**: Simulates realistic attention check failures and flags participants for exclusion
- **Complete output package**: CSV data, R script, metadata, schema validation, and instructor report

**Why use simulated pilot data?**
Simulated data allows you to:
- Test your analysis pipeline before collecting real data
- Practice data cleaning and exclusion procedures
- Verify your survey logic and variable coding
- Develop your R/analysis scripts on realistic data structures
""")

with st.expander("Research foundations and citations", expanded=False):
    st.markdown("""
### Methodological foundations

This tool implements simulation approaches validated in recent computational social science research:

**Core LLM Simulation Research:**
- **Argyle et al. (2023)** - "Out of One, Many: Using Language Models to Simulate Human Samples" *Political Analysis*, 31(3), 337-351. [DOI: 10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)
- **Horton (2023)** - "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?" *NBER Working Paper* No. 31122. [DOI: 10.3386/w31122](https://doi.org/10.3386/w31122)
- **Aher, Arriaga & Kalai (2023)** - "Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies" *ICML*, PMLR 202:337-371. [Paper](https://proceedings.mlr.press/v202/aher23a.html)

**High-Impact Validation Studies:**
- **Park et al. (2023)** - "Generative Agents: Interactive Simulacra of Human Behavior" *ACM UIST*. [DOI: 10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
- **Binz & Schulz (2023)** - "Using cognitive psychology to understand GPT-3" *PNAS*, 120(6). [DOI: 10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120)
- **Dillion et al. (2023)** - "Can AI language models replace human participants?" *Trends in Cognitive Sciences*, 27, 597-600. [DOI: 10.1016/j.tics.2023.04.008](https://doi.org/10.1016/j.tics.2023.04.008)

**On LLM Detection & Survey Validity:**
- **Westwood (2025)** - "The potential existential threat of large language models to online survey research" *PNAS*, 122(47). [DOI: 10.1073/pnas.2518075122](https://doi.org/10.1073/pnas.2518075122) — Demonstrates why rigorous simulation standards matter for distinguishing human from AI responses.

**Additional Resources:**
- **Brand, Israeli & Ngwe (2023)** - "Using GPT for Market Research" *Harvard Business School Working Paper* 23-062. [Paper](https://www.hbs.edu/ris/download.aspx?name=23-062.pdf)

### How personas work

The simulator automatically assigns behavioral personas to simulated participants based on the study domain. Each persona has trait parameters calibrated from computational social science research on LLM response patterns. This creates realistic individual differences without requiring manual configuration.
""")
    # Prefer PDF over markdown for methods summary
    methods_pdf_path = Path(__file__).resolve().parent / "docs" / "Behavioral Experiment Simulation Tool - Methods Summary.pdf"
    methods_md_path = Path(__file__).resolve().parent / "docs" / "methods_summary.md"

    if methods_pdf_path.exists():
        methods_updated = datetime.utcfromtimestamp(methods_pdf_path.stat().st_mtime).strftime("%Y-%m-%d")
        st.caption(f"Methods summary (PDF) · Last updated: {methods_updated}")

        # Two options: download or view in browser
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.download_button(
                "📥 Download PDF",
                data=methods_pdf_path.read_bytes(),
                file_name=methods_pdf_path.name,
                mime="application/pdf",
            )
        with col_m2:
            # Create a data URL for opening in new tab
            pdf_bytes = methods_pdf_path.read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode()
            pdf_data_url = f"data:application/pdf;base64,{pdf_b64}"
            st.markdown(
                f'<a href="{pdf_data_url}" target="_blank" rel="noopener noreferrer">'
                f'📄 Open in new tab</a>',
                unsafe_allow_html=True
            )
    elif methods_md_path.exists():
        methods_updated = datetime.utcfromtimestamp(methods_md_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Methods summary updated: {methods_updated}")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.download_button(
                "📥 Download Markdown",
                data=methods_md_path.read_bytes(),
                file_name=methods_md_path.name,
                mime="text/markdown",
            )
        with col_m2:
            with st.expander("📄 View Methods Summary"):
                st.markdown(methods_md_path.read_text(encoding="utf-8"))

with st.sidebar:
    st.subheader("Mode")

    advanced_mode = st.toggle("Advanced mode", value=st.session_state.get("advanced_mode", False))
    st.session_state["advanced_mode"] = advanced_mode

    st.divider()
    st.subheader("Simple vs. Advanced")
    st.write(
        "- **Simple mode** uses standardized defaults for demographics, attention checks, and exclusions "
        "so outputs are comparable across teams.\n"
        "- **Advanced mode** unlocks controls for demographics, exclusions, and effect sizes when you need "
        "a custom setup."
    )

    st.divider()
    st.subheader("Study Snapshot")

    # Gather all snapshot data
    snapshot_conditions = st.session_state.get("current_conditions") or st.session_state.get("selected_conditions") or []
    snapshot_conditions = snapshot_conditions + st.session_state.get("custom_conditions", [])
    snapshot_conditions = list(dict.fromkeys([c for c in snapshot_conditions if str(c).strip()]))
    inferred = st.session_state.get("inferred_design", {})
    snapshot_scales = inferred.get("scales", [])
    snapshot_factors = inferred.get("factors", [])

    # Study title
    title = st.session_state.get('study_title', '—') or '—'
    st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")

    # Key metrics in compact format
    sample_size = st.session_state.get('sample_size', '—')
    st.caption(f"N = {sample_size}")

    # Detect design type from conditions
    detected_factors = _infer_factors_from_conditions(snapshot_conditions) if snapshot_conditions else []
    if len(detected_factors) == 1:
        design_hint = f"{len(snapshot_conditions)}-condition"
    elif len(detected_factors) == 2:
        f1_levels = len(detected_factors[0].get("levels", []))
        f2_levels = len(detected_factors[1].get("levels", []))
        design_hint = f"{f1_levels}×{f2_levels} factorial"
    elif len(detected_factors) > 2:
        design_hint = f"{len(detected_factors)}-factor"
    else:
        design_hint = "—"

    st.caption(f"Design: {design_hint}")

    # Conditions (collapsible if many)
    if snapshot_conditions:
        if len(snapshot_conditions) <= 4:
            st.caption(f"Conditions: {', '.join(snapshot_conditions)}")
        else:
            with st.expander(f"Conditions ({len(snapshot_conditions)})"):
                for c in snapshot_conditions:
                    st.caption(f"• {c}")

    # Scales with scale points
    if snapshot_scales:
        scale_summary = []
        for s in snapshot_scales[:3]:
            name = s.get("name", "Scale")[:20]
            pts = s.get("scale_points", "?")
            scale_summary.append(f"{name} ({pts}pt)")
        scales_text = ", ".join(scale_summary)
        if len(snapshot_scales) > 3:
            scales_text += f" +{len(snapshot_scales) - 3} more"
        st.caption(f"DVs: {scales_text}")
    else:
        st.caption("DVs: Not configured")

    # Factors (if detected)
    if len(detected_factors) > 1:
        factors_text = " × ".join([f.get("name", "Factor") for f in detected_factors[:3]])
        st.caption(f"Factors: {factors_text}")

    st.divider()
    st.subheader("Progress")
    completion = _get_step_completion()
    step1_ready = completion["study_title"] and completion["study_description"] and completion["sample_size"]
    step2_ready = completion["qsf_uploaded"]
    step3_ready = completion["conditions_set"]
    step4_ready = completion["design_ready"]

    # Calculate overall progress
    steps_complete = sum([step1_ready, step2_ready, step3_ready, step4_ready])
    st.progress(steps_complete / 4, text=f"{steps_complete}/4 steps complete")

    # Quick jump buttons for incomplete steps
    if not step1_ready:
        if st.button("Complete Step 1", key="jump_step1", use_container_width=True):
            _go_to_step(0)
    elif not step2_ready:
        if st.button("Complete Step 2", key="jump_step2", use_container_width=True):
            _go_to_step(1)
    elif not step3_ready:
        if st.button("Complete Step 3", key="jump_step3", use_container_width=True):
            _go_to_step(2)
    elif not step4_ready:
        if st.button("Go to Generate", key="jump_step4", use_container_width=True):
            _go_to_step(3)

    # Start Over button
    st.divider()
    if st.button("🔄 Start Over", key="start_over_btn", use_container_width=True, type="secondary"):
        # Clear all session state
        keys_to_clear = [
            "study_title", "study_description", "sample_size", "team_name", "team_members_raw",
            "qsf_preview", "qsf_raw_content", "qsf_file_name",
            "survey_pdf_content", "survey_pdf_name",
            "prereg_pdf_content", "prereg_pdf_name", "prereg_text_sanitized", "prereg_pdf_text",
            "prereg_outcomes", "prereg_iv", "prereg_exclusions", "prereg_analysis",
            "enhanced_analysis", "inferred_design", "condition_allocation",
            "selected_conditions", "current_conditions", "custom_conditions",
            "last_df", "last_zip", "last_metadata",
            "has_generated", "is_generating", "generation_requested",
            "active_step",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.caption("Clear all entries and start fresh")


if "active_step" not in st.session_state:
    st.session_state["active_step"] = 0

active_step = st.session_state["active_step"]

# Get step completion status
completion = _get_step_completion()
step_complete = [
    completion["study_title"] and completion["study_description"],
    completion["qsf_uploaded"],
    completion["conditions_set"] and completion["primary_outcome"] and completion["independent_var"] and completion["sample_size"],
    completion["design_ready"],
]


def _render_workflow_stepper():
    """Render a visual workflow stepper with progress indicators."""
    # Inject CSS for step styling
    st.markdown("""
    <style>
    .step-circle-completed {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #10b981;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0 auto 0.5rem auto;
    }
    .step-circle-current {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #3b82f6;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0 auto 0.5rem auto;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.3);
    }
    .step-circle-pending {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #9ca3af;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0 auto 0.5rem auto;
    }
    .step-container {
        text-align: center;
        padding: 0.5rem 0;
    }
    .step-label-current {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1f2937;
        text-align: center;
    }
    .step-label-default {
        font-weight: 500;
        font-size: 0.9rem;
        color: #374151;
        text-align: center;
    }
    .step-desc {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render stepper using Streamlit columns for interactivity
    cols = st.columns(len(STEP_LABELS))

    for i, (label, desc) in enumerate(zip(STEP_LABELS, STEP_DESCRIPTIONS)):
        with cols[i]:
            # Determine step status
            is_completed = step_complete[i]
            is_current = i == active_step

            # Step indicator and status
            if is_completed:
                status_icon = "✓"
                circle_class = "step-circle-completed"
            elif is_current:
                status_icon = str(i + 1)
                circle_class = "step-circle-current"
            else:
                status_icon = str(i + 1)
                circle_class = "step-circle-pending"

            label_class = "step-label-current" if is_current else "step-label-default"

            # Create step display using clean HTML with CSS classes
            step_html = f'''<div class="step-container">
<div class="{circle_class}">{status_icon}</div>
<div class="{label_class}">{label}</div>
<div class="step-desc">{desc}</div>
</div>'''
            st.markdown(step_html, unsafe_allow_html=True)

            # Navigation button under each step
            if is_current:
                st.caption("You are here")
            elif is_completed or i <= active_step + 1:
                if st.button(f"Go to Step {i + 1}", key=f"stepper_nav_{i}", use_container_width=True):
                    _go_to_step(i)


# Render the workflow stepper
_render_workflow_stepper()

# Show what's missing for current step
st.markdown("---")


def _get_condition_candidates(
    preview: Optional[QSFPreviewResult],
    enhanced_analysis: Optional[DesignAnalysisResult],
) -> List[str]:
    candidates: List[str] = []
    if enhanced_analysis and enhanced_analysis.conditions:
        for cond in enhanced_analysis.conditions:
            if cond.source in ("QSF Randomizer", "QSF Block Name"):
                candidates.append(cond.name)

    if preview and preview.blocks:
        for block in preview.blocks:
            block_name = block.block_name.strip()
            if block_name and block_name.lower() not in (
                "default question block",
                "trash / unused questions",
                "block",
            ):
                candidates.append(block_name)

    return list(dict.fromkeys([c for c in candidates if c.strip()]))


def _render_step_navigation(step_index: int, can_next: bool, next_label: str) -> None:
    """Render navigation buttons at the bottom of each step."""
    st.markdown("---")

    # Create a prominent navigation section
    if step_index < len(STEP_LABELS) - 1:
        # Not on final step
        col_back, col_spacer, col_next = st.columns([1, 1, 2])
        with col_back:
            if step_index > 0:
                if st.button("← Previous Step", key=f"nav_back_{step_index}", use_container_width=True):
                    _go_to_step(step_index - 1)
        with col_next:
            if can_next:
                if st.button(f"Continue to {next_label} →", key=f"nav_next_{step_index}", type="primary", use_container_width=True):
                    _go_to_step(step_index + 1)
            else:
                st.button(f"Continue to {next_label} →", key=f"nav_next_disabled_{step_index}", disabled=True, use_container_width=True)
                st.caption("Complete required fields above to continue")
    else:
        # On final step - just show back button
        col_back, col_spacer = st.columns([1, 3])
        with col_back:
            if st.button("← Previous Step", key=f"nav_back_{step_index}", use_container_width=True):
                _go_to_step(step_index - 1)


def _get_total_conditions() -> int:
    """Get total number of conditions from all sources."""
    selected = st.session_state.get("selected_conditions", [])
    custom = st.session_state.get("custom_conditions", [])
    return len(set(selected + custom))


# ========================================
# UNIFIED STATUS PANEL - Shows progress across all steps
# ========================================
def _render_status_panel():
    """Render a unified status panel showing all required fields."""
    completion = _get_step_completion()

    # Calculate overall progress
    required_items = ["study_title", "study_description", "sample_size", "qsf_uploaded",
                      "primary_outcome", "independent_var", "conditions_set"]
    completed_count = sum(1 for k in required_items if completion.get(k, False))
    total_count = len(required_items)

    # Progress bar
    progress = completed_count / total_count
    st.progress(progress, text=f"Setup progress: {completed_count}/{total_count} required fields")

    # Missing fields with clickable guidance
    missing = []
    if not completion["study_title"]:
        missing.append(("Study title", "Step 1"))
    if not completion["study_description"]:
        missing.append(("Study description", "Step 1"))
    if not completion["sample_size"]:
        missing.append(("Sample size (minimum 10)", "Step 3"))
    if not completion["qsf_uploaded"]:
        missing.append(("QSF file upload", "Step 2"))
    if not completion["primary_outcome"]:
        missing.append(("Primary outcome variable", "Step 2"))
    if not completion["independent_var"]:
        missing.append(("Independent variable", "Step 2"))
    if not completion["conditions_set"]:
        missing.append(("Experimental conditions", "Step 3"))

    if missing:
        missing_text = " · ".join([f"**{name}** ({loc})" for name, loc in missing])
        st.warning(f"Missing: {missing_text}")

    # Current status summary
    preview = st.session_state.get("qsf_preview", None)
    total_conditions = _get_total_conditions()
    inferred = st.session_state.get("inferred_design", None)
    num_scales = len(inferred.get("scales", [])) if inferred else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Size", st.session_state.get("sample_size", 0))
    col2.metric("Conditions", total_conditions)
    col3.metric("Scales", num_scales)
    col4.metric("QSF Status", "✓ Uploaded" if preview and preview.success else "Not uploaded")


# -----------------------------
# Step 1: Study Info (required basics)
# -----------------------------
if active_step == 0:
    # Step header with status
    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]

    st.markdown("### Step 1: Study Information")

    if step1_done:
        st.success("All required fields complete. Click **Continue to Upload Files** below to proceed.")
    else:
        missing = []
        if not completion["study_title"]:
            missing.append("Study title")
        if not completion["study_description"]:
            missing.append("Study description")
        st.info(f"**Complete these fields to continue:** {', '.join(missing)}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Study Details")
        study_title = st.text_input(
            "Study title *",
            value=st.session_state.get("study_title", ""),
            placeholder="e.g., Effect of AI Labels on Consumer Trust",
            help="Appears in the report and simulated data outputs.",
        )
        study_description = st.text_area(
            "Study description *",
            value=st.session_state.get("study_description", ""),
            height=150,
            placeholder="Describe your study's purpose, manipulation, and main outcomes. This helps with domain detection and persona selection.",
            help="Include your manipulation, population, and intended outcomes (helps domain detection).",
        )

        st.session_state["study_title"] = study_title
        st.session_state["study_description"] = study_description

    with col2:
        st.markdown("### Team Information (optional)")
        team_name = st.text_input(
            "Team name",
            value=st.session_state.get("team_name", ""),
            placeholder="e.g., Team Alpha",
            help="Optional. Helps instructors identify your team.",
        )
        members = st.text_area(
            "Team members (one per line)",
            value=st.session_state.get("team_members_raw", ""),
            height=100,
            placeholder="John Doe\nJane Smith",
            help="Optional. List team members for the report.",
        )
        st.session_state["team_name"] = team_name
        st.session_state["team_members_raw"] = members

    _render_step_navigation(0, step1_done, "Upload Files")


# -----------------------------
# Step 2: Upload Files (QSF + Study Design Info)
# -----------------------------
if active_step == 1:
    # Check if Step 1 is complete
    completion = _get_step_completion()
    step1_done = completion["study_title"] and completion["study_description"]
    step2_done = completion["qsf_uploaded"]

    st.markdown("### Step 2: Upload Files")

    if not step1_done:
        st.error("Please complete **Step 1: Study Info** first before uploading files.")
        if st.button("← Go to Step 1: Study Info", key="go_step1_from_step2", type="primary"):
            _go_to_step(0)
        st.stop()

    if step2_done:
        st.success("QSF file uploaded successfully. Click **Continue to Design Setup** below to proceed.")
    else:
        st.info("**Upload your Qualtrics QSF file below to continue.**")

    st.markdown("---")

    # ========================================
    # REQUIRED: QSF FILE
    # ========================================
    st.markdown("### 1. Upload Qualtrics Survey File *")

    # Check if we already have a file uploaded (persistence)
    existing_qsf_name = st.session_state.get("qsf_file_name")
    existing_qsf_content = st.session_state.get("qsf_raw_content")

    col_qsf, col_help = st.columns([2, 1])

    with col_qsf:
        # Show existing file info if available
        if existing_qsf_name and existing_qsf_content:
            st.success(f"✓ **{existing_qsf_name}** uploaded ({len(existing_qsf_content):,} bytes)")
            change_qsf = st.checkbox("Upload a different QSF file", value=False, key="change_qsf")
        else:
            change_qsf = True  # No existing file, show uploader

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
            qsf_file = None  # Use existing

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

    # Initialize preview variable for QSF processing
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    # Check if this is a new file upload (different from what's already stored)
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
            st.session_state["qsf_raw_content"] = payload  # Store for enhanced analysis
            st.session_state["qsf_file_name"] = qsf_file.name  # Track file name

            if preview.success:
                # Perform enhanced design analysis
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

                # Rerun to update navigation buttons with new completion status
                st.rerun()
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
    st.markdown("### 2. Survey PDF Export (Optional)")
    st.caption("Upload a PDF export of your survey for better question identification and simulation output quality.")

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
        survey_pdf = st.file_uploader(
            "Survey PDF",
            type=["pdf"],
            help="PDF export of your Qualtrics survey (optional but recommended)",
            key="survey_pdf_uploader",
        )
    else:
        survey_pdf = None

    if survey_pdf is not None:
        try:
            survey_pdf_content = survey_pdf.read()
            st.session_state["survey_pdf_content"] = survey_pdf_content
            st.session_state["survey_pdf_name"] = survey_pdf.name

            # Try to extract text from PDF
            survey_pdf_text = ""
            extraction_method = None

            # Method 1: Try PyMuPDF (fitz)
            try:
                import fitz
                pdf_doc = fitz.open(stream=survey_pdf_content, filetype="pdf")
                try:
                    for page in pdf_doc:
                        survey_pdf_text += page.get_text()
                    extraction_method = "PyMuPDF"
                finally:
                    pdf_doc.close()
            except ImportError:
                pass
            except Exception:
                pass

            # Method 2: Try pypdf as fallback
            if not survey_pdf_text.strip():
                try:
                    survey_pdf_text = _extract_pdf_text(survey_pdf_content)
                    if survey_pdf_text.strip():
                        extraction_method = "pypdf"
                except Exception:
                    pass

            # Method 3: Try pdfplumber as last resort
            if not survey_pdf_text.strip():
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(survey_pdf_content)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                survey_pdf_text += page_text + "\n"
                    extraction_method = "pdfplumber"
                except Exception:
                    pass

            st.session_state["survey_pdf_text"] = survey_pdf_text
            if survey_pdf_text.strip():
                st.success(f"Survey PDF uploaded: {survey_pdf.name} (text extracted via {extraction_method})")
            else:
                st.warning(f"Survey PDF uploaded: {survey_pdf.name} (text extraction failed, but file will be included in output)")
        except Exception as e:
            st.error(f"Failed to process survey PDF: {e}")

    # ========================================
    # OPTIONAL: Preregistration / AsPredicted
    # ========================================
    st.markdown("---")
    st.markdown("### 3. Preregistration Details (Optional)")
    st.caption("Upload your AsPredicted or preregistration PDF to improve simulation quality. The tool uses this to better understand your hypotheses and variables.")

    with st.expander("Add preregistration for better simulations", expanded=False):
        col_prereg1, col_prereg2 = st.columns(2)

        with col_prereg1:
            prereg_file = st.file_uploader(
                "AsPredicted / Preregistration PDF",
                type=["pdf"],
                help="Upload your preregistration document (PDF format)",
                key="prereg_pdf_uploader",
            )

            if prereg_file is not None:
                try:
                    # Store the PDF content for later analysis
                    pdf_content = prereg_file.read()
                    st.session_state["prereg_pdf_content"] = pdf_content
                    st.session_state["prereg_pdf_name"] = prereg_file.name
                    st.success(f"Preregistration uploaded: {prereg_file.name}")

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

                    st.session_state["prereg_pdf_text"] = pdf_text
                    if pdf_text.strip():
                        st.info(f"PDF text extracted successfully ({extraction_method}).")
                    else:
                        st.warning("PDF uploaded but text extraction failed. The file will still be included in metadata.")
                except Exception as e:
                    st.error(f"Failed to process preregistration PDF: {e}")

        with col_prereg2:
            prereg_outcomes = st.text_area(
                "Primary outcome variables",
                value=st.session_state.get("prereg_outcomes", ""),
                placeholder="e.g., purchase_intention, brand_attitude",
                help="List your main dependent variables (comma-separated)",
                key="prereg_outcomes_input",
            )
            st.session_state["prereg_outcomes"] = prereg_outcomes

            prereg_iv = st.text_area(
                "Independent variables / Manipulations",
                value=st.session_state.get("prereg_iv", ""),
                placeholder="e.g., AI_recommendation (yes/no), product_type (utilitarian/hedonic)",
                help="List your manipulated variables and levels",
                key="prereg_iv_input",
            )
            st.session_state["prereg_iv"] = prereg_iv

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
            ]
        st.session_state["condition_sources"] = condition_sources

        current_conditions = st.session_state.get("current_conditions") or []
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Questions", int(getattr(preview, "total_questions", 0) or 0))
        c2.metric("Scales detected", int(len(getattr(preview, "detected_scales", []) or [])))
        c3.metric("Condition candidates", int(len(st.session_state.get("condition_candidates", []) or [])))
        c4.metric("Conditions selected", int(len(current_conditions)))
        warnings = getattr(preview, "validation_warnings", []) or []
        c5.metric("Warnings", int(len(warnings)))

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

        if current_conditions:
            st.success(f"Current conditions saved: {', '.join(current_conditions)}")
        else:
            st.info("No conditions selected yet. Configure them in **Design Setup**.")

        # Navigation to next step
        st.markdown("---")
        st.success("**QSF parsed successfully!** Proceed to the **Design Setup** step to configure your experimental conditions.")

    _render_step_navigation(1, step2_done, "Design Setup")


# -----------------------------
# Step 3: Design Setup (conditions, factors, scales)
# -----------------------------
if active_step == 2:
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)
    enhanced_analysis: Optional[DesignAnalysisResult] = st.session_state.get("enhanced_analysis", None)

    st.markdown("### Step 3: Design Setup")

    if not preview:
        st.error("Please upload a QSF file first before configuring your design.")
        if st.button("← Go to Step 2: Upload Files", key="go_step2_from_step3", type="primary"):
            _go_to_step(1)
        _render_step_navigation(2, False, "Generate")
        st.stop()  # Don't continue rendering Step 3 if no preview

    # If we reach here, preview exists
    inferred = st.session_state.get("inferred_design", {})

    st.markdown("---")

    # ========================================
    # STEP 1: CONDITION SETUP
    # ========================================
    st.markdown("### 1. Define Your Experimental Conditions")

    # Help text in expander (separate from selection)
    with st.expander("ℹ️ What are conditions? (click for help)", expanded=False):
        st.markdown("""
**Conditions** are the different groups/treatments in your experiment.

**Examples:**
- A simple A/B test has 2 conditions: Control vs. Treatment
- A 2x2 design has 4 conditions: Control-Low, Control-High, Treatment-Low, Treatment-High

**Where do conditions come from?**
- The simulator detects conditions from your QSF file's **Randomizer** or **BlockRandomizer** elements
- If your QSF doesn't use randomization, select the block names that represent your conditions
- Condition names should match what's in your QSF (they appear in the dropdown below)
""")

    # Condition selection (OUTSIDE the help expander)
    condition_candidates = st.session_state.get("condition_candidates")
    if not condition_candidates:
        condition_candidates = _get_condition_candidates(preview, enhanced_analysis)
        st.session_state["condition_candidates"] = condition_candidates

    all_possible_conditions = condition_candidates or []

    # Initialize selected conditions in session state
    if "selected_conditions" not in st.session_state:
        # Default to QSF-detected conditions if any
        st.session_state["selected_conditions"] = condition_candidates[:] if condition_candidates else []

    # Check if we have any possible conditions to select from
    if all_possible_conditions:
        st.markdown("**Select which blocks/groups represent your experimental conditions:**")
        st.caption(
            "These were detected from your QSF file's randomizer and block structure. "
            "Select the exact block names if you want them to match your survey flow."
        )

        # Multi-select from available options
        selected = st.multiselect(
            "Select conditions from QSF",
            options=all_possible_conditions,
            default=st.session_state.get("selected_conditions", []),
            help="Pick the block names that correspond to experimental conditions (matches the QSF flow).",
            key="condition_multiselect",
        )
        st.session_state["selected_conditions"] = selected

        if not selected:
            st.warning("Please select at least one condition, or add custom conditions below.")
    else:
        st.info("No condition blocks detected in the QSF file.")
        selected = []
        st.session_state["selected_conditions"] = []

    # Option to add custom conditions (but using a controlled interface)
    with st.expander("Add custom conditions (if not in QSF)", expanded=not bool(all_possible_conditions)):
        st.caption(
            "If your conditions aren't detected above, add them here. Use names that match your survey flow "
            "if you want alignment, but any descriptive labels will still work for simulation."
        )

        # Get current custom conditions
        custom_conditions = st.session_state.get("custom_conditions", [])

        col_add1, col_add2 = st.columns([3, 1])
        with col_add1:
            new_condition = st.text_input(
                "New condition name",
                key="new_condition_input",
                placeholder="e.g., Control, Treatment, High, Low",
                help="Use the same spelling as your QSF block names if you want exact alignment.",
            )
        with col_add2:
            st.write("")  # Spacer
            if st.button("Add", key="add_condition_btn", disabled=not new_condition.strip()):
                if new_condition.strip() and new_condition.strip() not in custom_conditions:
                    custom_conditions.append(new_condition.strip())
                    st.session_state["custom_conditions"] = custom_conditions
                    st.rerun()

            # Show custom conditions with remove buttons
            if custom_conditions:
                st.markdown("**Custom conditions added:**")
                for i, cc in enumerate(custom_conditions):
                    col_cc, col_rm = st.columns([4, 1])
                    with col_cc:
                        st.text(f"• {cc}")
                    with col_rm:
                        # Use condition name in key for stability when list changes
                        safe_key = cc.replace(" ", "_").replace(".", "_")[:20]
                        if st.button("✕", key=f"rm_custom_{safe_key}_{i}"):
                            custom_conditions.remove(cc)
                            st.session_state["custom_conditions"] = custom_conditions
                            st.rerun()

    # Combine selected and custom conditions
    custom_conditions = st.session_state.get("custom_conditions", [])
    all_conditions = list(dict.fromkeys(selected + custom_conditions))

    # Show final conditions summary
    st.markdown("---")
    if all_conditions:
        st.success(f"**{len(all_conditions)} condition(s) configured:** {', '.join(all_conditions)}")
    else:
        st.error("❌ No conditions defined. Please select or add at least one condition.")
        all_conditions = ["Condition A"]  # Fallback

    # ========================================
    # STEP 2: DESIGN STRUCTURE
    # ========================================
    st.markdown("---")
    st.markdown("### 2. Configure Design Structure")

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

    # ========================================
    # SAMPLE SIZE AND CONDITION ALLOCATION
    # ========================================
    st.markdown("---")
    st.markdown("### 3. Sample Size & Condition Allocation")

    # Sample size input (moved from Step 1)
    sample_size = st.number_input(
        "Target sample size (N) * — based on power analysis",
        min_value=10,
        max_value=MAX_SIMULATED_N,
        value=int(st.session_state.get("sample_size", 200)),
        step=10,
        help=(
            "Enter the sample size from your power analysis (a priori power calculation). "
            "This should be the N required to detect your expected effect size with adequate power (typically 80%). "
            f"Maximum: {MAX_SIMULATED_N:,}."
        ),
        key="sample_size_input_step3",
    )
    st.session_state["sample_size"] = int(sample_size)
    st.caption("💡 Use your power analysis result (e.g., from G*Power) to determine the appropriate N.")

    # Condition allocation (rebalancing) with sliders
    if all_conditions and len(all_conditions) > 1:
        st.markdown("#### Condition Allocation")
        st.caption(
            "By default, participants are divided equally across conditions. "
            "Use the sliders below to adjust the allocation if needed (must sum to 100%)."
        )

        # Initialize allocation if not set
        if "condition_allocation" not in st.session_state or len(st.session_state["condition_allocation"]) != len(all_conditions):
            # Equal allocation by default
            equal_pct = 100 // len(all_conditions)
            remainder = 100 - (equal_pct * len(all_conditions))
            st.session_state["condition_allocation"] = {
                cond: equal_pct + (1 if i < remainder else 0)
                for i, cond in enumerate(all_conditions)
            }

        allocation = st.session_state["condition_allocation"]

        # Use columns for compact slider layout
        slider_cols = st.columns(min(len(all_conditions), 3))

        new_allocation = {}
        for i, cond in enumerate(all_conditions):
            col_idx = i % len(slider_cols)
            with slider_cols[col_idx]:
                # Clean condition name for display
                display_name = cond[:25] + "..." if len(cond) > 25 else cond
                current_val = allocation.get(cond, 100 // len(all_conditions))
                new_val = st.slider(
                    display_name,
                    min_value=5,
                    max_value=95,
                    value=current_val,
                    step=5,
                    key=f"alloc_slider_{i}",
                    help=f"Percentage of participants in '{cond}'"
                )
                new_allocation[cond] = new_val

        # Calculate total and show warning if not 100%
        total_pct = sum(new_allocation.values())

        if total_pct != 100:
            st.warning(f"⚠️ Allocations sum to **{total_pct}%** (should be 100%). Adjust sliders to balance.")
            # Auto-normalize button
            if st.button("Auto-balance to 100%", key="auto_balance_btn"):
                scale_factor = 100 / total_pct
                normalized = {}
                running_total = 0
                sorted_conds = sorted(new_allocation.keys())
                for j, c in enumerate(sorted_conds[:-1]):
                    normalized[c] = max(5, min(95, round(new_allocation[c] * scale_factor)))
                    running_total += normalized[c]
                # Last condition gets the remainder
                normalized[sorted_conds[-1]] = 100 - running_total
                st.session_state["condition_allocation"] = normalized
                st.rerun()
        else:
            st.success(f"✓ Allocations sum to 100%")
            st.session_state["condition_allocation"] = new_allocation

        # Show expected N per condition
        st.markdown("**Expected participants per condition:**")
        n_per_cond = []
        for cond in all_conditions:
            pct = new_allocation.get(cond, 100 // len(all_conditions))
            n = round(sample_size * pct / 100)
            n_per_cond.append(f"{cond}: **{n}**")
        st.markdown(" | ".join(n_per_cond))

    elif all_conditions:
        # Single condition - show simple info
        st.info(f"Single condition design: all {sample_size} participants in '{all_conditions[0]}'")
        st.session_state["condition_allocation"] = {all_conditions[0]: 100}

    # Factor configuration with clear explanation
    st.markdown("---")
    st.markdown("#### Factor Structure")
    st.caption(
        "**Factors** are the independent variables you manipulate. "
        "A 2×2 design has 2 factors (e.g., AI presence × Product type). "
        "The tool auto-detects factors from condition names like 'AI x Hedonic' or 'Control | Treatment'."
    )

    # Let user override auto-detected factors
    use_auto_factors = auto_num_factors > 1 and st.checkbox(
        f"Use auto-detected {auto_num_factors}-factor structure",
        value=True,
        key="use_auto_factors",
        help="Uncheck to manually configure factors"
    )

    factors = []
    if use_auto_factors and auto_num_factors > 1:
        # Show auto-detected structure with ability to rename
        factors = auto_detected_factors
        st.markdown("**Detected factor structure:**")
        for i, f in enumerate(auto_detected_factors):
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                new_name = st.text_input(
                    f"Factor {i+1} name",
                    value=f.get("name", f"Factor_{i+1}"),
                    key=f"auto_factor_name_{i}",
                )
                factors[i]["name"] = new_name
            with col_f2:
                levels = f.get("levels", [])
                st.caption(f"Levels: {', '.join(levels)}")
    else:
        # Manual factor configuration
        if len(all_conditions) >= 4:
            num_factors = st.selectbox(
                "Number of factors",
                options=[1, 2, 3],
                index=0,
                key="num_factors_select",
                help=(
                    "• **1 factor**: Simple design (Control vs Treatment)\n"
                    "• **2 factors**: 2×2, 2×3, etc. (e.g., AI × Product Type)\n"
                    "• **3 factors**: Complex designs (e.g., AI × Price × Brand)"
                ),
            )

            if num_factors == 1:
                factors = [{"name": "Condition", "levels": all_conditions}]
            else:
                st.info(f"Define your {num_factors} factors by selecting which conditions belong to each level.")
                for f_idx in range(num_factors):
                    st.markdown(f"**Factor {f_idx + 1}**")
                    factor_name = st.text_input(
                        "Name",
                        value=f"Factor {f_idx + 1}",
                        key=f"factor_name_{f_idx}",
                        label_visibility="collapsed",
                        placeholder=f"e.g., {'AI Presence' if f_idx == 0 else 'Product Type'}"
                    )
                    factor_levels = st.multiselect(
                        f"Levels for {factor_name}",
                        options=all_conditions,
                        key=f"factor_levels_{f_idx}",
                        help=f"Select the levels/conditions that belong to {factor_name}",
                    )
                    if factor_levels:
                        factors.append({"name": factor_name, "levels": factor_levels})
        else:
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


    # ========================================
    # FINAL SUMMARY & LOCK DESIGN
    # ========================================
    st.markdown("---")
    st.markdown("### Design Summary")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Conditions", len(all_conditions))
    summary_cols[1].metric("Factors", len(factors))
    summary_cols[2].metric("Scales", len(scales))
    summary_cols[3].metric("Design", design_type.split("(")[0].strip())

    # Show condition list and detected scales
    st.markdown(f"**Conditions:** {', '.join(all_conditions)}")
    scale_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]
    st.markdown(f"**Scales:** {', '.join(scale_names) if scale_names else 'Main_DV (default)'}")

    # Validate and lock design - only require conditions and scales from QSF
    design_valid = len(all_conditions) >= 1 and len(scales) >= 1

    if design_valid:
        # Save to session state
        final_conditions = all_conditions
        final_factors = _normalize_factor_specs(factors, all_conditions)
        final_scales = _normalize_scale_specs(scales)
        final_open_ended = inferred.get("open_ended_questions", [])

        # Determine randomization level string
        rand_mapping = {
            "Participant-level (standard)": "Participant-level",
            "Group/Cluster-level": "Group/Cluster-level",
            "Not randomized / observational": "Not randomized",
        }
        final_rand_level = rand_mapping.get(rand_level, "Participant-level")

        st.session_state["inferred_design"] = {
            "conditions": final_conditions,
            "factors": final_factors,
            "scales": final_scales,
            "open_ended_questions": final_open_ended,
            "randomization_level": final_rand_level,
        }
        st.session_state["randomization_level"] = final_rand_level

        # Initialize variable review rows if not already set
        if not st.session_state.get("variable_review_rows"):
            prereg_outcomes = st.session_state.get("prereg_outcomes", "")
            prereg_iv = st.session_state.get("prereg_iv", "")
            st.session_state["variable_review_rows"] = _build_variable_review_rows(
                st.session_state["inferred_design"], prereg_outcomes, prereg_iv, enhanced_analysis
            )

        st.success("Design configuration complete. Proceed to the **Generate** step to run the simulation.")
    else:
        missing_bits = []
        if not all_conditions:
            missing_bits.append("conditions")
        if not scales:
            missing_bits.append("scales")
        st.error("Please complete all required fields before proceeding: " + ", ".join(missing_bits))

    # ========================================
    # ADVANCED: Variable Review (collapsed)
    # ========================================
    with st.expander("Advanced: Review All Variables"):
        st.caption("View and edit all detected survey variables and their roles.")

        prereg_outcomes = st.session_state.get("prereg_outcomes", "")
        prereg_iv = st.session_state.get("prereg_iv", "")
        default_rows = _build_variable_review_rows(inferred, prereg_outcomes, prereg_iv, enhanced_analysis)
        current_rows = st.session_state.get("variable_review_rows")

        if not current_rows:
            current_rows = default_rows

        # Simple filter
        show_timing = st.checkbox("Include timing/meta variables", value=False, key="adv_filter_timing")

        if not show_timing:
            filtered_rows = [r for r in current_rows if r.get("Type") != "Timing/Meta"]
        else:
            filtered_rows = current_rows

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

    # Navigation at bottom of Step 3 (outside expander)
    _render_step_navigation(2, design_valid, "Generate")


# -----------------------------
# Step 4: Generate (standard defaults; advanced controls optional)
# -----------------------------
if active_step == 3:
    inferred = st.session_state.get("inferred_design", None)
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    st.markdown("### Step 4: Generate Simulation")

    # Comprehensive pre-validation
    completion = _get_step_completion()
    all_required_complete = (
        completion["study_title"] and
        completion["study_description"] and
        completion["sample_size"] and
        completion["qsf_uploaded"] and
        completion["conditions_set"]
    )

    if not inferred:
        st.error("Please complete the previous steps first to configure your design.")
        if st.button("← Go to Step 3: Design Setup", key="go_step3_from_step4", type="primary"):
            _go_to_step(2)
        st.stop()  # Don't render rest of Step 4 if design not configured

    # If we reach here, inferred exists
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    # Check for missing fields
    required_fields = {
        "Study title": bool(st.session_state.get("study_title", "").strip()),
        "Study description": bool(st.session_state.get("study_description", "").strip()),
        "Pre-registered sample size": int(st.session_state.get("sample_size", 0)) >= 10,
        "QSF uploaded": bool(preview and preview.success),
        "Design configured": bool(st.session_state.get("inferred_design")),
    }
    completed = sum(required_fields.values())
    total_required = len(required_fields)

    missing_fields = [label for label, ok in required_fields.items() if not ok]
    if missing_fields:
        st.warning(f"**Missing required fields:** {', '.join(missing_fields)}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Go to Step 3: Design Setup", key="fix_missing_from_generate", use_container_width=True):
                _go_to_step(2)
    else:
        st.success("All required fields are complete. You can generate your simulation.")

    config_col1, config_col2 = st.columns(2)
    conditions = inferred.get('conditions', [])
    scales = inferred.get('scales', [])
    scale_names = [s.get('name', 'Unknown') for s in scales if s.get('name')]

    with config_col1:
        st.markdown(f"**Study:** {st.session_state.get('study_title', 'Untitled')}")
        st.markdown(f"**Sample Size:** {st.session_state.get('sample_size', 0)} participants")

    with config_col2:
        st.markdown(f"**Conditions:** {', '.join(conditions) if conditions else 'Not detected'}")
        st.markdown(f"**Scales:** {', '.join(scale_names[:3])}{' ...' if len(scale_names) > 3 else ''}" if scale_names else "**Scales:** Default (Main_DV)")

    st.markdown("---")

    if not st.session_state.get("advanced_mode", False):
        demographics = STANDARD_DEFAULTS["demographics"].copy()
        attention_rate = STANDARD_DEFAULTS["attention_rate"]
        random_responder_rate = STANDARD_DEFAULTS["random_responder_rate"]
        exclusion = ExclusionCriteria(**STANDARD_DEFAULTS["exclusion_criteria"])
        effect_sizes: List[EffectSizeSpec] = []
        custom_persona_weights = None

        with st.expander("View standardized settings (locked for comparability)"):
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
                st.markdown("- No directional effects (null hypothesis)")
                st.caption("Enable Advanced mode to specify expected effect sizes")
    else:
        st.markdown("### Advanced settings")

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

        st.markdown("### Expected Effect Sizes (Optional)")
        st.caption(
            "Specify the expected effect size for your main hypothesis. "
            "This makes the simulated data reflect a directional hypothesis rather than a null effect."
        )

        effect_sizes = []

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
                factor_levels = selected_factor_data.get("levels", []) if selected_factor_data else all_conditions

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

        custom_persona_weights = None

    # ========================================
    # GENERATE BUTTON - with proper state management
    # ========================================
    st.markdown("---")

    if "generation_requested" not in st.session_state:
        st.session_state["generation_requested"] = False

    is_generating = st.session_state.get("is_generating", False)
    has_generated = st.session_state.get("has_generated", False)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    if is_generating:
        status_placeholder.info("Simulation is running. Please wait...")

    if has_generated:
        st.success("Simulation complete! Download your files below.")

    # Button: disabled if not ready, generating, or already generated
    can_generate = all_required_complete and not is_generating and not has_generated

    # Create button row
    btn_col1, btn_col2 = st.columns([2, 1])

    with btn_col1:
        if is_generating:
            st.button("Generating simulated dataset...", type="primary", disabled=True, use_container_width=True)
        elif has_generated:
            st.button("Simulation generated", type="primary", disabled=True, use_container_width=True)
        else:
            if st.button("Generate simulated dataset", type="primary", disabled=not can_generate, use_container_width=True):
                st.session_state["generation_requested"] = True
                st.rerun()

    with btn_col2:
        # Reset button - allows user to restart if stuck
        if has_generated or is_generating:
            if st.button("Reset & Generate New", use_container_width=True):
                st.session_state["is_generating"] = False
                st.session_state["has_generated"] = False
                st.session_state["generation_requested"] = False
                st.session_state["last_df"] = None
                st.session_state["last_zip"] = None
                st.session_state["last_metadata"] = None
                st.rerun()

    if st.session_state.get("generation_requested") and not is_generating:
        st.session_state["generation_requested"] = False
        st.session_state["is_generating"] = True
        progress_bar = progress_placeholder.progress(5, text="Preparing simulation inputs...")
        status_placeholder.info("Preparing simulation inputs...")
        title = st.session_state.get("study_title", "") or "Untitled Study"
        desc = st.session_state.get("study_description", "") or ""
        requested_n = int(st.session_state.get("sample_size", 200))
        if requested_n > MAX_SIMULATED_N:
            st.info(
                f"Requested N ({requested_n}) exceeds the cap ({MAX_SIMULATED_N}). "
                "Using the capped value for standardization."
            )
        N = min(requested_n, MAX_SIMULATED_N)

        if missing_fields:
            st.error("Generation blocked until all required fields are completed.")
            st.session_state["is_generating"] = False
            progress_placeholder.empty()
            st.stop()

        prereg_text = st.session_state.get("prereg_text_sanitized", "")

        clean_scales = _normalize_scale_specs(inferred.get("scales", []))
        clean_factors = _normalize_factor_specs(inferred.get("factors", []), inferred.get("conditions", []))

        # Get condition allocation from session state
        condition_allocation = st.session_state.get("condition_allocation", None)

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
            open_ended_questions=inferred.get("open_ended_questions", []),
            condition_allocation=condition_allocation,
            seed=None,
            mode="pilot" if not st.session_state.get("advanced_mode", False) else "final",
        )

        try:
            progress_bar.progress(30, text="Generating simulated responses...")
            status_placeholder.info("Generating simulated responses...")
            df, metadata = engine.generate()
            progress_bar.progress(60, text="Packaging downloads...")
            status_placeholder.info("Packaging downloads and reports...")
            explainer = engine.generate_explainer()
            r_script = engine.generate_r_export(df)

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

            schema_results = validate_schema(
                df=df,
                expected_conditions=inferred.get("conditions", []),
                expected_scales=clean_scales,
                expected_n=N,
            )

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            meta_bytes = _safe_json(metadata).encode("utf-8")
            explainer_bytes = explainer.encode("utf-8")
            r_bytes = r_script.encode("utf-8")
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

            files = {
                "Simulated_Data.csv": csv_bytes,
                "Metadata.json": meta_bytes,
                "Data_Codebook_Handbook.txt": explainer_bytes,  # Explains all variable coding
                "R_Prepare_Data.R": r_bytes,
                "Schema_Validation.json": _safe_json(schema_results).encode("utf-8"),
                "Study_Summary.md": instructor_bytes,  # Summary report (not the comprehensive instructor analysis)
            }

            # Include uploaded source files in "Source_Files" subfolder
            qsf_content = st.session_state.get("qsf_raw_content")
            if qsf_content:
                qsf_name = st.session_state.get("qsf_file_name", "survey.qsf")
                files[f"Source_Files/{qsf_name}"] = qsf_content if isinstance(qsf_content, bytes) else qsf_content.encode("utf-8")

            prereg_pdf = st.session_state.get("prereg_pdf_content")
            if prereg_pdf:
                prereg_name = st.session_state.get("prereg_pdf_name", "preregistration.pdf")
                files[f"Source_Files/{prereg_name}"] = prereg_pdf

            survey_pdf = st.session_state.get("survey_pdf_content")
            if survey_pdf:
                survey_pdf_name = st.session_state.get("survey_pdf_name", "survey_export.pdf")
                files[f"Source_Files/{survey_pdf_name}"] = survey_pdf

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

            progress_bar.progress(85, text="Finalizing notifications...")
            status_placeholder.info("Finalizing notifications...")
            st.success("Simulation generated.")
            st.markdown("[Jump to download](#download)")

            if not schema_results.get("valid", True):
                st.error("Schema validation failed. Review Schema_Validation.json in the download.")
            elif schema_results.get("warnings"):
                st.info("Schema validation warnings found. Review Schema_Validation.json in the download.")
            else:
                st.info("Schema validation passed.")

            instructor_email = st.secrets.get("INSTRUCTOR_NOTIFICATION_EMAIL", "edimant@sas.upenn.edu")
            subject = f"[Behavioral Simulation] Output ({metadata.get('simulation_mode', 'pilot')}) - {title}"
            body = (
                "COMPREHENSIVE INSTRUCTOR ANALYSIS ATTACHED\n"
                "=========================================\n\n"
                "This email includes detailed statistical analysis that students do NOT receive.\n"
                "Students get only Study_Summary.md in their download ZIP.\n\n"
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
                "- Study_Summary.md (basic summary)\n"
                "- R_Prepare_Data.R (R script)\n"
                "- Metadata.json, Schema_Validation.json\n"
            )
            ok, msg = _send_email_with_sendgrid(
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
                st.info(f"Instructor auto-email sent to {instructor_email}.")
            else:
                st.error(f"Instructor auto-email failed: {msg}")

            progress_bar.progress(100, text="Simulation ready.")
            status_placeholder.success("Simulation complete.")
            st.session_state["has_generated"] = True
            st.session_state["is_generating"] = False
            st.rerun()  # Refresh to show download section
        except Exception as e:
            progress_bar.progress(100, text="Simulation failed.")
            status_placeholder.error("Simulation failed.")
            st.error(f"Simulation failed: {e}")
            st.session_state["is_generating"] = False
            st.session_state["generation_requested"] = False
            # Don't rerun on error - show error message to user

    zip_bytes = st.session_state.get("last_zip", None)
    df = st.session_state.get("last_df", None)
    if zip_bytes and df is not None:
        st.divider()
        st.markdown('<div id="download"></div>', unsafe_allow_html=True)
        st.subheader("Download")

        st.download_button(
            "Download ZIP (CSV + metadata + R script)",
            data=zip_bytes,
            file_name=f"behavioral_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )

        with st.expander("Preview (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)

        st.divider()
        st.subheader("Email (optional)")
        _render_email_setup_diagnostics()

        to_email = st.text_input("Send to email", value=st.session_state.get("send_to_email", ""))
        st.session_state["send_to_email"] = to_email

        colE1, colE2 = st.columns([1, 1])
        with colE1:
            if st.button("Send ZIP via email"):
                if not to_email or "@" not in to_email:
                    st.error("Please enter a valid email address.")
                else:
                    subject = f"[Behavioral Simulation] Output: {st.session_state.get('study_title','Untitled Study')}"
                    body = (
                        "Attached is the simulation output ZIP (Simulated.csv, metadata, R prep script).\n\n"
                        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
                    )
                    ok, msg = _send_email_with_sendgrid(
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
                if st.button("Send to instructor too"):
                    subject = f"[Behavioral Simulation] Output (team: {st.session_state.get('team_name','') or 'N/A'})"
                    body = (
                        f"Team: {st.session_state.get('team_name','')}\n"
                        f"Members:\n{st.session_state.get('team_members_raw','')}\n\n"
                        f"Study: {st.session_state.get('study_title','')}\n"
                        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
                    )
                    ok, msg = _send_email_with_sendgrid(
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

    _render_step_navigation(3, True, "Finish")
