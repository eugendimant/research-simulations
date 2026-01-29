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
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
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
    EnhancedConditionIdentifier,
    DesignAnalysisResult,
    IdentifiedVariable,
    VariableRole,
    RandomizationLevel,
    analyze_qsf_design,
)
from utils.text_generator import (
    OpenEndedTextGenerator,
    PersonaTextTraits,
    create_text_generator,
)


# -----------------------------
# App constants
# -----------------------------
APP_TITLE = "Behavioral Experiment Simulation Tool"
APP_SUBTITLE = "Fast, standardized pilot simulations from your Qualtrics QSF"
APP_VERSION = "2.0.0"
APP_BUILD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")

BASE_STORAGE = Path("data")
BASE_STORAGE.mkdir(parents=True, exist_ok=True)

MAX_SIMULATED_N = 2000

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


def _extract_conditions_from_text(text: str) -> List[str]:
    if not text:
        return []

    conditions: List[str] = []
    keywords = ("condition", "conditions", "treatment", "group", "arm", "variant", "manipulation", "scenario")
    vs_pattern = r"\b(?:vs\.?|versus|v\.|compared to|compared with)\b"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.search(vs_pattern, line, flags=re.IGNORECASE):
            parts = re.split(vs_pattern, line, flags=re.IGNORECASE)
            for part in parts:
                normalized = _normalize_condition_label(part)
                if normalized:
                    conditions.append(normalized)
            continue

        if any(keyword in line.lower() for keyword in keywords):
            _, tail = line.split(":", 1) if ":" in line else ("", line)
            tail = re.sub(
                r"\b(?:conditions?|treatments?|groups?|arms?|variants?|manipulations?|scenarios?)\b",
                "",
                tail,
                flags=re.IGNORECASE,
            )
            for part in re.split(r"[;/|,]", tail):
                normalized = _normalize_condition_label(part)
                if normalized:
                    conditions.append(normalized)

    return conditions


def _extract_conditions_from_prereg(prereg_iv: str, prereg_notes: str, prereg_pdf_text: str) -> List[str]:
    conditions: List[str] = []

    conditions.extend(_extract_conditions_from_text(prereg_iv))
    conditions.extend(_extract_conditions_from_text(prereg_notes))

    if prereg_pdf_text:
        for line in prereg_pdf_text.splitlines():
            if re.search(r"\b(condition|treatment|group|arm|variant|manipulation|scenario)\b", line, flags=re.IGNORECASE):
                conditions.extend(_extract_conditions_from_text(line))

    return conditions


def _merge_condition_sources(qsf_conditions: List[str], prereg_conditions: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    sources_by_key: Dict[str, List[str]] = {}
    display_by_key: Dict[str, str] = {}
    order: List[str] = []

    def register(cond: str, source: str) -> None:
        normalized = _normalize_condition_label(cond)
        if not normalized:
            return
        key = normalized.lower()
        if key not in display_by_key:
            display_by_key[key] = normalized
            sources_by_key[key] = []
            order.append(key)
        if source not in sources_by_key[key]:
            sources_by_key[key].append(source)

    for cond in qsf_conditions:
        register(cond, "QSF/Survey flow")
    for cond in prereg_conditions:
        register(cond, "Preregistration")

    merged_conditions = [display_by_key[key] for key in order]
    source_rows = [
        {"Condition": display_by_key[key], "Source": ", ".join(sources_by_key[key])}
        for key in order
    ]
    return merged_conditions, source_rows


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


def _build_variable_review_rows(
    inferred: Dict[str, Any],
    prereg_outcomes: str,
    prereg_iv: str,
    design_analysis: Optional[DesignAnalysisResult] = None,
) -> List[Dict[str, Any]]:
    """
    Build variable review rows from inferred design and preregistration info.

    This creates a comprehensive list of variables for user review and correction.
    Each row includes:
    - Variable: The Qualtrics variable name
    - Display Name: Human-readable name
    - Role: The assigned role (editable)
    - Source: Where the variable was detected
    - Confidence: How confident the system is in the role assignment
    """
    rows: List[Dict[str, Any]] = []
    seen_vars: set = set()

    # If we have enhanced design analysis, use it
    if design_analysis and design_analysis.variables:
        for var in design_analysis.variables:
            if var.variable_id in seen_vars:
                continue
            seen_vars.add(var.variable_id)

            # Map VariableRole enum to display string
            role_map = {
                VariableRole.CONDITION: "Condition",
                VariableRole.INDEPENDENT_VARIABLE: "Independent variable",
                VariableRole.PRIMARY_OUTCOME: "Primary outcome",
                VariableRole.SECONDARY_OUTCOME: "Secondary outcome",
                VariableRole.MEDIATOR: "Secondary outcome",  # Map to available option
                VariableRole.MODERATOR: "Secondary outcome",
                VariableRole.COVARIATE: "Other",
                VariableRole.DEMOGRAPHICS: "Other",
                VariableRole.ATTENTION_CHECK: "Other",
                VariableRole.MANIPULATION_CHECK: "Other",
                VariableRole.OPEN_ENDED: "Open-ended",
                VariableRole.FILLER: "Other",
                VariableRole.OTHER: "Other",
            }

            rows.append({
                "Variable": var.variable_id,
                "Display Name": var.display_name,
                "Role": role_map.get(var.role, "Other"),
                "Source": var.source,
                "Question Text": var.question_text[:80] + "..." if len(var.question_text) > 80 else var.question_text,
            })
        return rows

    # Fallback: Build from inferred design
    # Add conditions
    for cond in inferred.get("conditions", []):
        if cond and cond not in seen_vars:
            seen_vars.add(cond)
            rows.append({
                "Variable": cond,
                "Display Name": cond,
                "Role": "Condition",
                "Source": "QSF/Preregistration",
                "Question Text": "",
            })

    # Add scales
    for scale in inferred.get("scales", []):
        name = scale.get("name", "")
        if name and name not in seen_vars:
            seen_vars.add(name)
            # Check if this matches preregistration outcomes
            role = "Other"
            if prereg_outcomes:
                name_lower = name.lower()
                outcomes_lower = prereg_outcomes.lower()
                if name_lower in outcomes_lower or any(
                    word in outcomes_lower for word in name_lower.split() if len(word) > 3
                ):
                    role = "Primary outcome"

            rows.append({
                "Variable": name,
                "Display Name": name.replace("_", " ").title(),
                "Role": role,
                "Source": "QSF Scale Detection",
                "Question Text": f"{scale.get('num_items', 0)} items, {scale.get('scale_points', 7)}-point scale",
            })

    # Add open-ended questions
    for q in inferred.get("open_ended_questions", []):
        if q and q not in seen_vars:
            seen_vars.add(q)
            rows.append({
                "Variable": q,
                "Display Name": q.replace("_", " ").title(),
                "Role": "Open-ended",
                "Source": "QSF Question Type",
                "Question Text": "",
            })

    # If no rows, add a placeholder
    if not rows:
        rows.append({
            "Variable": "Main_DV",
            "Display Name": "Main DV",
            "Role": "Primary outcome",
            "Source": "Default",
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
    except Exception as e:
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
        if 200 <= int(getattr(resp, "status_code", 0)) < 300:
            return True, f"Sent (status {resp.status_code})."
        return False, f"SendGrid error (status {resp.status_code})."
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


tabs = st.tabs(["1) Quick setup", "2) Upload QSF", "3) Review", "4) Generate"])


# -----------------------------
# Tab 1: Quick setup (minimal, aspredicted-like)
# -----------------------------
with tabs[0]:
    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.subheader("Team")
        gm = _get_group_manager()

        team_name = st.text_input("Team name", value=st.session_state.get("team_name", ""))
        members = st.text_area(
            "Team members (one per line)",
            value=st.session_state.get("team_members_raw", ""),
            height=120,
            help="Optional but recommended so the instructor can identify teams.",
        )
        st.session_state["team_name"] = team_name
        st.session_state["team_members_raw"] = members

    with colB:
        st.subheader("Study (aspredicted-style minimum)")
        study_title = st.text_input("Study title", value=st.session_state.get("study_title", ""))
        study_description = st.text_area(
            "One-paragraph study description",
            value=st.session_state.get("study_description", ""),
            height=140,
            help="Purpose, manipulation, main outcomes. The app uses this for domain detection and persona selection.",
        )

        sample_size = st.number_input(
            "Pre-registered target sample size (N)",
            min_value=10,
            max_value=MAX_SIMULATED_N,
            value=int(st.session_state.get("sample_size", 200)),
            step=10,
            help="Use the pre-registered N from your power calculation (capped for standardization).",
        )

        st.session_state["study_title"] = study_title
        st.session_state["study_description"] = study_description
        st.session_state["sample_size"] = int(sample_size)

    st.info(
        "Next: Upload your Qualtrics QSF. The app will infer conditions, factors, and scales automatically."
    )


# -----------------------------
# Tab 2: Upload QSF and optional prereg info
# -----------------------------
with tabs[1]:
    st.subheader("Upload your Qualtrics QSF")
    parser = _get_qsf_preview_parser()

    col_qsf, col_pdf = st.columns([1, 1])

    with col_qsf:
        st.markdown("**Required: QSF file**")
        qsf_file = st.file_uploader("QSF file", type=["qsf", "zip", "json"])
        st.caption("Export from Qualtrics: Survey → Tools → Import/Export → Export Survey")

    with col_pdf:
        st.markdown("**Optional: Survey PDF export**")
        survey_pdf = st.file_uploader("Survey PDF (for better domain detection)", type=["pdf"])
        with st.expander("Why upload the PDF? How to export it"):
            st.markdown("""
**Why it helps:**
- Improves detection of question wording and context
- Enables better domain inference for persona selection
- Captures visual elements and formatting not in QSF

**How to export from Qualtrics:**
1. Open your survey in Qualtrics
2. Go to **Tools** → **Import/Export** → **Print Survey**
3. Select "Print to PDF" or save as PDF
4. Upload the PDF here

This is optional but recommended for better simulation quality.
""")

    # Process survey PDF if provided
    if survey_pdf is not None:
        survey_pdf_text = _extract_pdf_text(survey_pdf.read())
        if survey_pdf_text:
            st.session_state["survey_pdf_text"] = survey_pdf_text
            st.success("Survey PDF uploaded successfully. Text extracted for domain detection.")
        else:
            st.info("Could not extract text from survey PDF.")

    st.markdown("### Aspredicted-style checklist (used only for labeling, not for simulation logic)")
    st.write(
        "To avoid bias, **do not include hypotheses**. We only record design facts needed for comparability."
    )

    prereg_outcomes = st.text_area(
        "Primary outcome variable(s)",
        value=st.session_state.get("prereg_outcomes", ""),
        height=80,
        help="Example: Purchase intention (7-point scale).",
    )
    prereg_iv = st.text_area(
        "Independent variable(s) / conditions",
        value=st.session_state.get("prereg_iv", ""),
        height=80,
        help="Example: AI-generated ad vs. human-written ad.",
    )
    prereg_exclusions = st.text_area(
        "Exclusion criteria",
        value=st.session_state.get("prereg_exclusions", ""),
        height=80,
        help="Example: failed attention checks, unrealistically fast completion.",
    )
    prereg_analysis = st.text_area(
        "Planned analysis (high-level)",
        value=st.session_state.get("prereg_analysis", ""),
        height=80,
        help="Example: independent-samples t-test on the main DV.",
    )

    st.session_state["prereg_outcomes"] = prereg_outcomes
    st.session_state["prereg_iv"] = prereg_iv
    st.session_state["prereg_exclusions"] = prereg_exclusions
    st.session_state["prereg_analysis"] = prereg_analysis

    st.markdown("### Optional: paste preregistration notes (hypotheses will be removed)")
    prereg_text = st.text_area(
        "Preregistration notes (optional)",
        value=st.session_state.get("prereg_text_raw", ""),
        height=140,
        help="If pasted, any lines that look like hypotheses or predictions are removed.",
    )
    sanitized_text, removed_lines = _sanitize_prereg_text(prereg_text)
    st.session_state["prereg_text_raw"] = prereg_text
    st.session_state["prereg_text_sanitized"] = sanitized_text

    if prereg_text and removed_lines:
        with st.expander("Removed hypothesis-like lines (not used)"):
            for line in removed_lines:
                st.write(f"- {line}")

    prereg_pdf = st.file_uploader("Optional: upload aspredicted PDF", type=["pdf"])
    if prereg_pdf is not None:
        pdf_text = _extract_pdf_text(prereg_pdf.read())
        st.session_state["prereg_pdf_text"] = pdf_text
        if pdf_text:
            st.caption("Extracted text (read-only). Hypotheses are ignored by the app.")
            with st.expander("Extracted preregistration text"):
                st.text(pdf_text[:4000] + ("..." if len(pdf_text) > 4000 else ""))
        else:
            st.info("Could not extract text from the PDF. You can still use the checklist above.")

    if qsf_file is not None:
        try:
            content = qsf_file.read()
            payload, payload_name = _extract_qsf_payload(content)
            preview = parser.parse(payload)
            st.session_state["qsf_preview"] = preview
            st.session_state["qsf_raw_content"] = payload  # Store for enhanced analysis

            if preview.success:
                st.success(f"QSF parsed successfully ({payload_name}).")

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
                        if enhanced_analysis.warnings:
                            for warn in enhanced_analysis.warnings[:3]:
                                st.warning(warn)
                        if enhanced_analysis.suggestions:
                            with st.expander("Analysis suggestions"):
                                for sug in enhanced_analysis.suggestions:
                                    st.info(sug)
            else:
                st.error("QSF parsed but validation failed. See warnings below.")
        except Exception as e:
            st.session_state["qsf_preview"] = None
            st.session_state["enhanced_analysis"] = None
            st.error(f"QSF parsing failed: {e}")

    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    if preview:
        prereg_conditions = _extract_conditions_from_prereg(
            st.session_state.get("prereg_iv", ""),
            st.session_state.get("prereg_text_sanitized", ""),
            st.session_state.get("prereg_pdf_text", ""),
        )
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", int(preview.total_questions))
        c2.metric("Scales detected", int(len(preview.detected_scales or [])))
        c3.metric("Conditions detected", int(len(preview.detected_conditions or [])))
        warnings = getattr(preview, "validation_warnings", []) or []
        c4.metric("Warnings", int(len(warnings)))

        errors = getattr(preview, "validation_errors", []) or []

        if errors:
            with st.expander("Show QSF errors"):
                for err in errors:
                    st.error(err)

        if warnings:
            with st.expander("Show QSF warnings"):
                for w in warnings:
                    st.info(w)

        st.caption("Next: Review the auto-detected design and run your simulation.")


# -----------------------------
# Tab 3: Review (enhanced variable editing and design confirmation)
# -----------------------------
with tabs[2]:
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)
    enhanced_analysis: Optional[DesignAnalysisResult] = st.session_state.get("enhanced_analysis", None)

    if not preview:
        st.info("Upload a QSF first to populate the review summary.")
    else:
        # Use enhanced analysis if available, otherwise fall back to basic inference
        basic_inferred = _preview_to_engine_inputs(preview)
        if enhanced_analysis:
            inferred = _design_analysis_to_inferred(enhanced_analysis, basic_inferred)
        else:
            inferred = basic_inferred
        st.session_state["inferred_design"] = inferred

        st.subheader("Review & Edit Detected Design")
        st.caption("Review the auto-detected design elements. **Edit any misidentified items** before running the simulation.")

        # Summary metrics row
        summary_cols = st.columns(5)
        summary_cols[0].metric("Conditions", len(inferred["conditions"]))
        summary_cols[1].metric("Factors", len(inferred["factors"]))
        summary_cols[2].metric("Scales", len(inferred["scales"]))
        summary_cols[3].metric("Open-ended", len(inferred.get("open_ended_questions", [])))

        # Show randomization level if detected
        rand_level = inferred.get("randomization_level", "Participant-level")
        summary_cols[4].metric("Randomization", rand_level.split("-")[0] if "-" in rand_level else rand_level)

        # Show detection confidence if available
        if enhanced_analysis and enhanced_analysis.conditions:
            avg_confidence = sum(c.confidence for c in enhanced_analysis.conditions) / len(enhanced_analysis.conditions)
            if avg_confidence < 0.5:
                st.warning(
                    "Low detection confidence. Please review and correct the conditions below. "
                    "The system may have misidentified some elements."
                )
            elif avg_confidence < 0.7:
                st.info(
                    "Moderate detection confidence. Please verify the detected design is correct."
                )

        st.divider()

        # Two-column layout for conditions and factors
        col1, col2 = st.columns([1.1, 0.9], gap="large")

        with col1:
            st.markdown("### Conditions")
            st.caption("Edit conditions directly in the table. Use the dropdown to change roles.")

            # Build conditions table with editable roles
            condition_rows = []
            condition_sources = st.session_state.get("condition_sources") or []

            if enhanced_analysis and enhanced_analysis.conditions:
                for cond in enhanced_analysis.conditions:
                    condition_rows.append({
                        "Condition": cond.name,
                        "Factor": cond.factor,
                        "Source": cond.source,
                        "Confidence": f"{cond.confidence:.0%}",
                    })
            elif condition_sources:
                for cs in condition_sources:
                    condition_rows.append({
                        "Condition": cs.get("Condition", ""),
                        "Factor": "Treatment",
                        "Source": cs.get("Source", "Unknown"),
                        "Confidence": "N/A",
                    })
            elif inferred["conditions"]:
                for cond in inferred["conditions"]:
                    condition_rows.append({
                        "Condition": cond,
                        "Factor": "Treatment",
                        "Source": "Detected",
                        "Confidence": "N/A",
                    })

            if condition_rows:
                edited_conditions = st.data_editor(
                    pd.DataFrame(condition_rows),
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Condition": st.column_config.TextColumn("Condition Name", help="Edit the condition name"),
                        "Factor": st.column_config.TextColumn("Factor", help="Which factor this belongs to"),
                        "Source": st.column_config.TextColumn("Source", disabled=True),
                        "Confidence": st.column_config.TextColumn("Confidence", disabled=True),
                    },
                    key="conditions_editor",
                )
                st.session_state["edited_conditions"] = edited_conditions.to_dict(orient="records")
            else:
                st.info("No conditions detected. Add conditions manually below.")
                st.session_state["edited_conditions"] = []

            st.markdown("### Scales (Dependent Variables)")
            st.caption("Scales are groups of related Likert-type items. Edit names or add/remove items.")

            # Build scales table with variable names
            scale_rows = []
            if enhanced_analysis and enhanced_analysis.scales:
                for scale in enhanced_analysis.scales:
                    scale_rows.append({
                        "Scale Name": scale.name,
                        "Qualtrics Variable": scale.variable_name,
                        "Items": scale.num_items,
                        "Points": scale.scale_points,
                        "Role": "Primary outcome" if scale.role == VariableRole.PRIMARY_OUTCOME else "Secondary outcome",
                    })
            elif inferred["scales"]:
                for scale in inferred["scales"]:
                    scale_rows.append({
                        "Scale Name": scale.get("name", ""),
                        "Qualtrics Variable": scale.get("variable_name", scale.get("name", "")),
                        "Items": scale.get("num_items", 5),
                        "Points": scale.get("scale_points", 7),
                        "Role": "Primary outcome",
                    })

            if scale_rows:
                edited_scales_df = st.data_editor(
                    pd.DataFrame(scale_rows),
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Scale Name": st.column_config.TextColumn("Display Name", help="Human-readable scale name"),
                        "Qualtrics Variable": st.column_config.TextColumn("Variable Name", help="Qualtrics variable prefix"),
                        "Items": st.column_config.NumberColumn("# Items", min_value=1, max_value=50),
                        "Points": st.column_config.NumberColumn("Scale Points", min_value=2, max_value=11),
                        "Role": st.column_config.SelectboxColumn(
                            "Role",
                            options=["Primary outcome", "Secondary outcome", "Mediator", "Moderator", "Other"],
                            help="Variable role in your study"
                        ),
                    },
                    key="scales_editor",
                )
                st.session_state["edited_scales"] = edited_scales_df.to_dict(orient="records")
            else:
                st.info("No scales detected. Add scales manually using the + button.")
                st.session_state["edited_scales"] = []

        with col2:
            st.markdown("### Factors (Independent Variables)")
            st.caption("Factors define your experimental design structure.")

            # Build factors table
            factor_rows = []
            if enhanced_analysis and enhanced_analysis.factors:
                for factor in enhanced_analysis.factors:
                    factor_rows.append({
                        "Factor": factor.name,
                        "Levels": ", ".join(factor.levels),
                        "Type": "Between-subjects" if factor.is_between_subjects else "Within-subjects",
                        "Confidence": f"{factor.confidence:.0%}",
                    })
            elif inferred["factors"]:
                for factor in inferred["factors"]:
                    factor_rows.append({
                        "Factor": factor.get("name", "Condition"),
                        "Levels": ", ".join(factor.get("levels", [])),
                        "Type": "Between-subjects",
                        "Confidence": "N/A",
                    })

            if factor_rows:
                edited_factors = st.data_editor(
                    pd.DataFrame(factor_rows),
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Factor": st.column_config.TextColumn("Factor Name"),
                        "Levels": st.column_config.TextColumn("Levels (comma-separated)"),
                        "Type": st.column_config.SelectboxColumn(
                            "Design Type",
                            options=["Between-subjects", "Within-subjects", "Mixed"],
                        ),
                        "Confidence": st.column_config.TextColumn("Confidence", disabled=True),
                    },
                    key="factors_editor",
                )
                st.session_state["edited_factors"] = edited_factors.to_dict(orient="records")
            else:
                st.info("No factors detected. Conditions will be treated as a single factor.")

            st.markdown("### Randomization")
            st.caption("How participants are assigned to conditions.")

            rand_options = [
                "Participant-level",
                "Group/Cluster-level",
                "Within-subject (repeated measures)",
                "Multiple stages",
                "Not randomized / observational",
            ]
            detected_rand = inferred.get("randomization_level", "Participant-level")
            default_idx = 0
            for i, opt in enumerate(rand_options):
                if detected_rand.lower().startswith(opt.split("-")[0].lower()) or detected_rand.lower().startswith(opt.split("/")[0].lower()):
                    default_idx = i
                    break

            randomization_level = st.selectbox(
                "Randomization level",
                rand_options,
                index=default_idx,
                help="Where does randomization occur in your study?",
                key="randomization_selector",
            )
            st.session_state["randomization_level"] = randomization_level

            # Show attention and manipulation checks if detected
            if enhanced_analysis:
                if enhanced_analysis.attention_checks:
                    with st.expander(f"Attention Checks ({len(enhanced_analysis.attention_checks)} detected)"):
                        for ac in enhanced_analysis.attention_checks:
                            st.text(f"- {ac}")

                if enhanced_analysis.manipulation_checks:
                    with st.expander(f"Manipulation Checks ({len(enhanced_analysis.manipulation_checks)} detected)"):
                        for mc in enhanced_analysis.manipulation_checks:
                            st.text(f"- {mc}")

            st.markdown("### Detection Info")
            st.markdown(
                "**How detection works:**\n"
                "- **Conditions** are identified from Qualtrics randomizer blocks and preregistration text\n"
                "- **Factors** are inferred from condition naming patterns (e.g., `A x B`)\n"
                "- **Scales** are detected from matrix questions and numbered item patterns\n"
                "- **Randomization** is determined from the survey flow structure\n\n"
                "If anything is incorrect, edit it directly in the tables above."
            )

        st.divider()

        # Variable Review Section - Shows ALL variables with ability to reassign roles
        st.subheader("All Survey Variables")
        st.caption(
            "Review all detected variables and their assigned roles. "
            "Use the **Role** dropdown to correct any misidentified variables. "
            "The **Variable** column shows the exact Qualtrics variable name."
        )

        prereg_outcomes = st.session_state.get("prereg_outcomes", "")
        prereg_iv = st.session_state.get("prereg_iv", "")
        default_rows = _build_variable_review_rows(inferred, prereg_outcomes, prereg_iv, enhanced_analysis)
        current_rows = st.session_state.get("variable_review_rows")

        # Reset to defaults if empty or user requests
        if not current_rows:
            current_rows = default_rows

        variable_df = st.data_editor(
            pd.DataFrame(current_rows),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Variable": st.column_config.TextColumn(
                    "Qualtrics Variable",
                    help="The exact variable name from Qualtrics",
                    disabled=False,
                ),
                "Display Name": st.column_config.TextColumn(
                    "Display Name",
                    help="Human-readable name",
                ),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=[
                        "Condition",
                        "Independent variable",
                        "Primary outcome",
                        "Secondary outcome",
                        "Mediator",
                        "Moderator",
                        "Covariate",
                        "Demographics",
                        "Attention check",
                        "Manipulation check",
                        "Open-ended",
                        "Filler",
                        "Other",
                    ],
                    help="Select the role this variable plays in your study",
                    required=True,
                ),
                "Source": st.column_config.TextColumn(
                    "Detection Source",
                    help="How this variable was detected",
                    disabled=True,
                ),
                "Question Text": st.column_config.TextColumn(
                    "Question Text",
                    help="Preview of the question text",
                    disabled=True,
                ),
            },
            key="variable_review_editor",
            height=400,
        )
        st.session_state["variable_review_rows"] = variable_df.to_dict(orient="records")

        # Buttons for variable table actions
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Reset to Detected", help="Reset variable assignments to auto-detected values"):
                st.session_state["variable_review_rows"] = default_rows
                st.rerun()
        with col_btn2:
            if st.button("Mark All Unknown as Other"):
                rows = st.session_state.get("variable_review_rows", [])
                for row in rows:
                    if row.get("Role") == "Other" or not row.get("Role"):
                        row["Role"] = "Other"
                st.session_state["variable_review_rows"] = rows
                st.rerun()

        st.divider()
        st.subheader("Design Overrides (Advanced)")
        st.caption("Manually override the design if the auto-detection missed anything.")

        cond_text = st.text_area(
            "Treatment conditions (one per line)",
            value="\n".join(inferred["conditions"]),
            height=140,
            help="These are the randomized conditions used to simulate between-group differences.",
        )
        conditions = [c.strip() for c in cond_text.splitlines() if c.strip()]
        if not conditions:
            conditions = ["Condition A"]

        factors_json = st.text_area(
            "Factors & levels (JSON list) - optional",
            value=_safe_json(_infer_factors_from_conditions(conditions)),
            height=180,
            help='Example: [{"name":"AI","levels":["AI","No AI"]},{"name":"Product","levels":["Hedonic","Utilitarian"]}]',
        )
        try:
            factors = json.loads(factors_json)
            if not isinstance(factors, list):
                raise ValueError("Factors JSON must be a list.")
        except Exception as e:
            st.error(f"Factors JSON invalid: {e}")
            factors = _infer_factors_from_conditions(conditions)

        st.caption("Scales drive most analyses. Keep this short: 1-3 scales is typical for class pilots.")
        scales_df = pd.DataFrame(inferred["scales"])
        edited_scales = st.data_editor(
            scales_df,
            num_rows="dynamic",
            use_container_width=True,
        )
        scales: List[Dict[str, Any]] = []
        for _, row in edited_scales.iterrows():
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            scales.append(
                {
                    "name": name,
                    "num_items": int(row.get("num_items", 5) or 5),
                    "scale_points": int(row.get("scale_points", 7) or 7),
                    "reverse_items": row.get("reverse_items", []) or [],
                }
            )
        if not scales:
            scales = [{"name": "Main_DV", "num_items": 5, "scale_points": 7, "reverse_items": []}]

        open_ended_text = st.text_area(
            "Open-ended questions (optional, one per line)",
            value="\n".join(inferred.get("open_ended_questions", [])),
            height=120,
        )
        open_ended_questions = [q.strip() for q in open_ended_text.splitlines() if q.strip()]

        randomization_level = st.selectbox(
            "Randomization level",
            ["Participant", "Group/Cluster", "Multiple stages", "Not randomized / observational"],
            index=0,
            help="Used in reporting and metadata to capture where randomization occurs.",
        )
        st.session_state["randomization_level"] = randomization_level

        st.session_state["inferred_design"] = {
            "conditions": conditions,
            "factors": factors,
            "scales": scales,
            "open_ended_questions": open_ended_questions,
        }

        st.success("Design locked for generation (based on the settings above).")


# -----------------------------
# Tab 4: Generate (standard defaults; advanced controls optional)
# -----------------------------
with tabs[3]:
    inferred = st.session_state.get("inferred_design", None)
    if not inferred:
        st.info("Complete the previous steps first (upload QSF, then review).")
    else:
        st.subheader("Generate simulation")
        preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

        required_fields = {
            "Study title": bool(st.session_state.get("study_title", "").strip()),
            "Study description": bool(st.session_state.get("study_description", "").strip()),
            "Pre-registered sample size": int(st.session_state.get("sample_size", 0)) >= 10,
            "Primary outcome variable(s)": bool(st.session_state.get("prereg_outcomes", "").strip()),
            "Independent variable(s)": bool(st.session_state.get("prereg_iv", "").strip()),
            "QSF uploaded": bool(preview and preview.success),
        }
        completed = sum(required_fields.values())
        total_required = len(required_fields)
        st.progress(
            completed / total_required,
            text=f"Setup completion: {completed}/{total_required} required fields filled",
        )
        missing = [label for label, ok in required_fields.items() if not ok]
        if missing:
            st.info("Missing required fields: " + ", ".join(missing))

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

        if "generation_requested" not in st.session_state:
            st.session_state["generation_requested"] = False

        is_generating = st.session_state.get("is_generating", False)
        has_generated = st.session_state.get("has_generated", False)
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        if is_generating:
            status_placeholder.info("Simulation is running. Please wait for the download section to appear.")
        elif has_generated:
            status_placeholder.info("Simulation already generated for this session. Downloads are ready below.")
        can_generate = completed == total_required and not is_generating and not has_generated
        if st.button("Generate simulated dataset", type="primary", disabled=not can_generate):
            st.session_state["generation_requested"] = True

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

            engine = EnhancedSimulationEngine(
                study_title=title,
                study_description=desc,
                sample_size=N,
                conditions=inferred["conditions"],
                factors=inferred["factors"],
                scales=inferred["scales"],
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
                    expected_conditions=inferred["conditions"],
                    expected_scales=inferred["scales"],
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
                    prereg_text=prereg_text,
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
            except Exception as e:
                progress_bar.progress(100, text="Simulation failed.")
                status_placeholder.error("Simulation failed.")
                st.error(f"Simulation failed: {e}")
            finally:
                st.session_state["is_generating"] = False
                time.sleep(0.2)
                progress_placeholder.empty()

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
