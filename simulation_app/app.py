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
import io
import json
import re
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.group_management import GroupManager, APIKeyManager
from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult
from utils.schema_validator import validate_schema
from utils.instructor_report import InstructorReportGenerator
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


# -----------------------------
# App constants
# -----------------------------
APP_TITLE = "Behavioral Experiment Simulation Tool"
APP_SUBTITLE = "Fast, standardized pilot simulations from your Qualtrics QSF"
APP_VERSION = "2.1.2"  # Fixed QSF nested flow parsing, synced all module versions
APP_BUILD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")

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
    normalized: List[Dict[str, Any]] = []
    for scale in scales or []:
        if isinstance(scale, str):
            name = scale.strip()
            if name:
                normalized.append({"name": name, "num_items": 5, "scale_points": 7, "reverse_items": []})
            continue
        if isinstance(scale, dict):
            name = str(scale.get("name", "")).strip()
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "variable_name": str(scale.get("variable_name", name)),
                    "num_items": int(scale.get("num_items", 5)),
                    "scale_points": int(scale.get("scale_points", 7)),
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


def _infer_factors_from_conditions(conditions: List[str]) -> List[Dict[str, Any]]:
    """
    Heuristic inference:
    - If condition names look like 'A x B' (or 'A | B'), split into multiple factors.
    - Otherwise represent as single factor called 'Condition'.
    """
    if not conditions:
        return [{"name": "Condition", "levels": ["Condition A"]}]

    seps = [" x ", " X ", " | ", " + ", " * "]

    chosen_sep = None
    for sep in seps:
        if any(sep in c for c in conditions):
            chosen_sep = sep
            break

    if not chosen_sep:
        return [{"name": "Condition", "levels": conditions}]

    split_rows = [c.split(chosen_sep) for c in conditions]
    max_parts = max(len(r) for r in split_rows)

    if any(len(r) != max_parts for r in split_rows):
        return [{"name": "Condition", "levels": conditions}]

    factors: List[Dict[str, Any]] = []
    for j in range(max_parts):
        levels = sorted(list({r[j].strip() for r in split_rows}))
        factors.append({"name": f"Factor_{j+1}", "levels": levels})
    return factors


def _preview_to_engine_inputs(preview: QSFPreviewResult) -> Dict[str, Any]:
    """
    Convert QSFPreviewResult to engine-ready inputs with minimal assumptions.
    """
    conditions = (preview.detected_conditions or [])[:]
    if not conditions:
        conditions = ["Condition A"]

    factors = _infer_factors_from_conditions(conditions)

    scales: List[Dict[str, Any]] = []
    for s in (preview.detected_scales or []):
        name = str(s.get("name", "Scale")).strip() or "Scale"
        num_items = int(s.get("items", 5) or 5)
        scale_points = int(s.get("scale_points", 7) or 7)
        scales.append(
            {
                "name": name,
                "num_items": max(1, num_items),
                "scale_points": max(2, scale_points),
                "reverse_items": [],
            }
        )

    if not scales:
        scales = [{"name": "Main_DV", "num_items": 5, "scale_points": 7, "reverse_items": []}]

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
        st.download_button(
            "📄 Download Methods Summary (PDF)",
            data=methods_pdf_path.read_bytes(),
            file_name=methods_pdf_path.name,
            mime="application/pdf",
        )
    elif methods_md_path.exists():
        methods_updated = datetime.utcfromtimestamp(methods_md_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Methods summary updated: {methods_updated}")
        st.download_button(
            "Download full methods summary (Markdown)",
            data=methods_md_path.read_bytes(),
            file_name=methods_md_path.name,
            mime="text/markdown",
        )

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
    snapshot_conditions = st.session_state.get("current_conditions") or st.session_state.get("selected_conditions") or []
    snapshot_conditions = snapshot_conditions + st.session_state.get("custom_conditions", [])
    snapshot_conditions = list(dict.fromkeys([c for c in snapshot_conditions if str(c).strip()]))
    snapshot_scales = st.session_state.get("inferred_design", {}).get("scales", [])
    st.caption(f"Study title: {st.session_state.get('study_title', '—') or '—'}")
    st.caption(f"Sample size: {st.session_state.get('sample_size', '—')}")
    st.caption(f"Conditions: {len(snapshot_conditions)}")
    if snapshot_conditions:
        st.caption(", ".join(snapshot_conditions[:6]) + ("..." if len(snapshot_conditions) > 6 else ""))
    st.caption(f"Scales: {len(snapshot_scales) if snapshot_scales else 0}")

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


if "active_step" not in st.session_state:
    st.session_state["active_step"] = 0

active_step = st.session_state["active_step"]

# Get step completion status
completion = _get_step_completion()
step_complete = [
    completion["study_title"] and completion["study_description"] and completion["sample_size"],
    completion["qsf_uploaded"],
    completion["conditions_set"] and completion["primary_outcome"] and completion["independent_var"],
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
        missing.append(("Sample size (minimum 10)", "Step 1"))
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
    step1_done = completion["study_title"] and completion["study_description"] and completion["sample_size"]

    st.markdown("### Step 1: Study Information")

    if step1_done:
        st.success("All required fields complete. Click **Continue to Upload Files** below to proceed.")
    else:
        missing = []
        if not completion["study_title"]:
            missing.append("Study title")
        if not completion["study_description"]:
            missing.append("Study description")
        if not completion["sample_size"]:
            missing.append("Sample size (minimum 10)")
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
        )
        st.caption("💡 Use your power analysis result (e.g., from G*Power) to determine the appropriate N.")

        st.session_state["study_title"] = study_title
        st.session_state["study_description"] = study_description
        st.session_state["sample_size"] = int(sample_size)

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
    step1_done = completion["study_title"] and completion["study_description"] and completion["sample_size"]
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

    col_qsf, col_help = st.columns([2, 1])

    with col_qsf:
        qsf_file = st.file_uploader(
            "QSF file",
            type=["qsf", "zip", "json"],
            help=(
                "Export from Qualtrics via Tools → Import/Export → Export Survey. "
                "Upload the .qsf (or .zip) here."
            ),
        )

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

    survey_pdf = st.file_uploader(
        "Survey PDF",
        type=["pdf"],
        help="PDF export of your Qualtrics survey (optional but recommended)",
        key="survey_pdf_uploader",
    )

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

        # Number of factors
        num_factors = st.selectbox(
            "Number of factors (independent variables)",
            options=[1, 2, 3],
            index=0,
            key="num_factors_select",
            help="Most studies have 1-2 factors. A 2x2 design has 2 factors.",
        )

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

        # Sample size info
        sample_size = st.session_state.get("sample_size", 100)
        st.metric("Pre-registered sample size", sample_size)

    # Build factors automatically from conditions
    factors = []
    if num_factors == 1:
        factors = [{"name": "Condition", "levels": all_conditions}]
    elif num_factors >= 2 and len(all_conditions) >= 4:
        # Try to split conditions into factors
        st.info("With multiple factors, organize your conditions into groups:")
        for f_idx in range(num_factors):
            factor_name = st.text_input(
                f"Factor {f_idx + 1} name",
                value=f"Factor {f_idx + 1}",
                key=f"factor_name_{f_idx}",
            )
            factor_levels = st.multiselect(
                f"Levels for {factor_name}",
                options=all_conditions,
                key=f"factor_levels_{f_idx}",
                help=f"Select which conditions belong to {factor_name}",
            )
            if factor_levels:
                factors.append({"name": factor_name, "levels": factor_levels})
    else:
        # num_factors >= 2 but not enough conditions for multi-factor design
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

        with st.expander("Standardized settings (locked)"):
            st.json(
                {
                    "demographics": demographics,
                    "attention_rate": attention_rate,
                    "random_responder_rate": random_responder_rate,
                    "exclusion_criteria": asdict(exclusion),
                    "effect_sizes": [],
                }
            )
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

        st.markdown("### Optional: expected effect sizes")
        st.caption("Only add this if you want the simulated data to reflect a directional hypothesis.")
        effects_json = st.text_area(
            "Effect sizes (JSON list) - optional",
            value="[]",
            height=140,
            help='Example: [{"variable":"Main_DV","factor":"Condition","level_high":"AI","level_low":"No AI","cohens_d":0.3,"direction":"positive"}]',
        )
        effect_sizes = []
        try:
            raw = json.loads(effects_json)
            if isinstance(raw, list):
                for e in raw:
                    if not isinstance(e, dict):
                        continue  # Skip non-dict entries
                    effect_sizes.append(
                        EffectSizeSpec(
                            variable=str(e.get("variable", "")),
                            factor=str(e.get("factor", "")),
                            level_high=str(e.get("level_high", "")),
                            level_low=str(e.get("level_low", "")),
                            cohens_d=float(e.get("cohens_d", 0.0)),
                            direction=str(e.get("direction", "positive")),
                        )
                    )
        except Exception as e:
            st.error(f"Effect sizes JSON invalid; ignoring. ({e})")
            effect_sizes = []

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

            files = {
                "Simulated.csv": csv_bytes,
                "Metadata.json": meta_bytes,
                "Column_Explainer.txt": explainer_bytes,
                "R_Prepare_Data.R": r_bytes,
                "Schema_Validation.json": _safe_json(schema_results).encode("utf-8"),
                "Instructor_Report.md": instructor_bytes,
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
                "Automatic instructor notification with simulation output.\n\n"
                f"Team: {st.session_state.get('team_name','')}\n"
                f"Members:\n{st.session_state.get('team_members_raw','')}\n\n"
                f"Study: {title}\n"
                f"Generated: {metadata.get('generation_timestamp','')}\n"
                f"Run ID: {metadata.get('run_id','')}\n"
            )
            ok, msg = _send_email_with_sendgrid(
                to_email=instructor_email,
                subject=subject,
                body_text=body,
                attachments=[
                    ("simulation_output.zip", zip_bytes),
                    ("Instructor_Report.md", instructor_bytes),
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
