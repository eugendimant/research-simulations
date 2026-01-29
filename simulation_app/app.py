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
            if preview.success:
                st.success(f"QSF parsed successfully ({payload_name}).")
            else:
                st.error("QSF parsed but validation failed. See warnings below.")
        except Exception as e:
            st.session_state["qsf_preview"] = None
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
# Tab 3: Review (simple confirmation; advanced editing optional)
# -----------------------------
with tabs[2]:
    preview: Optional[QSFPreviewResult] = st.session_state.get("qsf_preview", None)

    if not preview:
        st.info("Upload a QSF first to populate the review summary.")
    else:
        inferred = _preview_to_engine_inputs(preview)
        st.session_state["inferred_design"] = inferred

        st.subheader("Review summary")
        st.caption("Confirm the auto-detected design before running the simulation.")

        summary_cols = st.columns(4)
        summary_cols[0].metric("Conditions", len(inferred["conditions"]))
        summary_cols[1].metric("Factors", len(inferred["factors"]))
        summary_cols[2].metric("Scales", len(inferred["scales"]))
        summary_cols[3].metric("Open-ended Qs", len(inferred.get("open_ended_questions", [])))

        st.divider()
        col1, col2 = st.columns([1.1, 0.9], gap="large")

        with col1:
            st.markdown("**Conditions**")
            if inferred["conditions"]:
                condition_sources = st.session_state.get("condition_sources") or []
                if condition_sources:
                    st.dataframe(
                        pd.DataFrame(condition_sources),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.dataframe(
                        pd.DataFrame({"Condition": inferred["conditions"]}),
                        hide_index=True,
                        use_container_width=True,
                    )
            else:
                st.info("No conditions detected. Advanced mode can be used to define them manually.")

            st.markdown("**Scales (inferred)**")
            if inferred["scales"]:
                st.dataframe(
                    pd.DataFrame(inferred["scales"]),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No scales detected. Advanced mode can be used to add them manually.")

        with col2:
            st.markdown("**Factors (inferred)**")
            if inferred["factors"]:
                st.dataframe(
                    pd.DataFrame(inferred["factors"]),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No factors inferred yet. Conditions will be treated as a single factor.")

            st.markdown("**Why these were inferred**")
            st.markdown(
                "- **Conditions** come from the Qualtrics survey flow and preregistration text (if provided).\n"
                "- **Factors** are inferred from condition naming patterns (e.g., `A x B`).\n"
                "- **Scales** are inferred from question blocks that look like Likert items.\n"
                "\nIf anything looks off, enable **Advanced mode** in the sidebar to edit."
            )

        if st.session_state.get("advanced_mode", False):
            st.divider()
            st.subheader("Advanced: edit design")
            st.caption("Changes here override the auto-detected design for the simulation.")

        prereg_outcomes = st.session_state.get("prereg_outcomes", "")
        prereg_iv = st.session_state.get("prereg_iv", "")
        default_rows = _build_variable_review_rows(inferred, prereg_outcomes, prereg_iv)
        current_rows = st.session_state.get("variable_review_rows", default_rows)
        if not current_rows:
            current_rows = default_rows

        variable_df = st.data_editor(
            pd.DataFrame(current_rows),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=[
                        "Condition",
                        "Independent variable",
                        "Primary outcome",
                        "Secondary outcome",
                        "Open-ended",
                        "Other",
                    ],
                )
            },
        )
        st.session_state["variable_review_rows"] = variable_df.to_dict(orient="records")

        st.divider()
        st.subheader("Design overrides")
        st.caption("Update the key pieces of the design before simulation if the auto-detection missed anything.")

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
