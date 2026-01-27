# app.py
"""
A Streamlit application for generating synthetic behavioral experiment data.

Two Modes:
1) PILOTING MODE - Unlimited use, Python-based simulation
2) FINAL DATA COLLECTION MODE - One-time use per group, Claude API-powered

BDS5010 Behavioral Experiment Simulation Tool
Prof. Dr. Eugen Dimant - Spring 2026
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit config (must be the first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BDS5010 Simulation Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Writable storage on Streamlit Cloud
# -----------------------------------------------------------------------------
APP_DATA_DIR = Path(os.getenv("BDS5010_APP_DATA_DIR", "/tmp/bds5010_simulation_app")).resolve()
LOG_STORAGE = APP_DATA_DIR / "logs"
GROUP_STORAGE = APP_DATA_DIR / "groups"
for p in (APP_DATA_DIR, LOG_STORAGE, GROUP_STORAGE):
    p.mkdir(parents=True, exist_ok=True)

INSTRUCTOR_EMAIL = "edimant@sas.upenn.edu"

# -----------------------------------------------------------------------------
# Import your custom modules, but do not crash the whole app at import time
# -----------------------------------------------------------------------------
CUSTOM_IMPORT_ERROR: Optional[str] = None
try:
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec, ExclusionCriteria
    from utils.persona_library import PersonaLibrary
    from utils.pdf_generator import generate_audit_log_pdf
    from utils.qsf_preview import QSFPreviewParser
    from utils.instructor_report import InstructorReportGenerator
    from utils.group_management import GroupManager, APIKeyManager
except Exception as e:  # noqa: BLE001
    CUSTOM_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# -----------------------------------------------------------------------------
# Minimal fallbacks for group + API enablement (keeps UI usable if that module breaks)
# -----------------------------------------------------------------------------
@dataclass
class _Member:
    name: str


@dataclass
class _Group:
    group_number: int
    members: List[_Member]
    project_title: str = ""
    final_mode_used: bool = False


class _FallbackAPIKeyManager:
    def is_final_mode_enabled(self) -> bool:
        return bool(st.secrets.get("FINAL_MODE_ENABLED", False))


class _FallbackGroupManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.file = storage_dir / "groups.json"
        if not self.file.exists():
            self.file.write_text(json.dumps({"groups": {}, "pilot_runs": {}}, indent=2), encoding="utf-8")

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except Exception:
            return {"groups": {}, "pilot_runs": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        self.file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_all_groups(self) -> Dict[int, _Group]:
        raw = self._read().get("groups", {})
        out: Dict[int, _Group] = {}
        for k, v in raw.items():
            try:
                num = int(k)
            except Exception:
                continue
            members = [_Member(name=m) for m in (v.get("members") or [])]
            out[num] = _Group(
                group_number=num,
                members=members,
                project_title=v.get("project_title", "") or "",
                final_mode_used=bool(v.get("final_mode_used", False)),
            )
        return out

    def record_pilot_run(self, group_number: int) -> None:
        data = self._read()
        pilot = data.setdefault("pilot_runs", {})
        key = str(int(group_number))
        pilot[key] = int(pilot.get(key, 0)) + 1
        self._write(data)


@st.cache_resource
def get_group_manager():
    if "GroupManager" in globals():
        for args, kwargs in [
            ([], {"storage_dir": str(GROUP_STORAGE)}),
            ([], {"storage_path": str(GROUP_STORAGE)}),
            ([str(GROUP_STORAGE)], {}),
            ([], {}),
        ]:
            try:
                return GroupManager(*args, **kwargs)  # type: ignore[name-defined]
            except TypeError:
                continue
            except Exception:
                break
    return _FallbackGroupManager(GROUP_STORAGE)


@st.cache_resource
def get_api_manager():
    if "APIKeyManager" in globals():
        for args, kwargs in [
            ([], {"storage_dir": str(GROUP_STORAGE)}),
            ([], {"storage_path": str(GROUP_STORAGE)}),
            ([str(GROUP_STORAGE)], {}),
            ([], {}),
        ]:
            try:
                return APIKeyManager(*args, **kwargs)  # type: ignore[name-defined]
            except TypeError:
                continue
            except Exception:
                break
    return _FallbackAPIKeyManager()


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_qsf_log(log_content: str, group_number: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_STORAGE / f"QSF_Log_Group{group_number}_{ts}.txt"
    path.write_text(log_content or "", encoding="utf-8")
    return path


def send_instructor_email(zip_buffer: io.BytesIO, metadata: Dict[str, Any]) -> bool:
    sendgrid_api_key = st.secrets.get("SENDGRID_API_KEY")
    if not sendgrid_api_key:
        st.warning("⚠️ Email not configured (missing SENDGRID_API_KEY). Download and email manually.")
        return False

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Attachment, Disposition, FileContent, FileName, FileType, Mail
    except Exception as e:
        st.warning(f"⚠️ SendGrid package not available: {e}. Download and email manually.")
        return False

    try:
        group_num = metadata.get("group_number", "Unknown")
        team_members = ", ".join(metadata.get("team_members", []))
        project_title = metadata.get("project_title", "Untitled")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        factors_txt = "\n".join(
            [f"  - {f.get('name','?')}: {', '.join(f.get('levels') or [])}" for f in metadata.get("factors", [])]
        ) or "  - N/A"
        scales_txt = "\n".join(
            [
                f"  - {s.get('name','?')}: {s.get('num_items','?')} items, {s.get('scale_points','?')}-point scale"
                for s in metadata.get("scales", [])
            ]
        ) or "  - N/A"

        email_body = f"""BDS5010 Simulation Submission

Group Number: {group_num}
Team Members: {team_members}
Project Title: {project_title}
Submission Time: {timestamp}

Project Description:
{metadata.get('project_description', 'N/A')}

Experimental Design:
- Design Type: {metadata.get('design_type', 'N/A')}
- Sample Size: {metadata.get('sample_size', 'N/A')}
- Target Population: {metadata.get('target_population', 'N/A')}
- Number of Conditions: {len(metadata.get('conditions', []))}

Factors:
{factors_txt}

Measurement Scales:
{scales_txt}

File Integrity:
- QSF Hash: {metadata.get('file_hashes', {}).get('qsf', 'N/A')}
"""

        message = Mail(
            from_email=st.secrets.get("SENDGRID_FROM_EMAIL", "noreply@bds5010.streamlit.app"),
            to_emails=INSTRUCTOR_EMAIL,
            subject=f"BDS5010 Simulation - Group {group_num} - {project_title}",
            plain_text_content=email_body,
        )

        encoded_file = base64.b64encode(zip_buffer.getvalue()).decode("utf-8")
        attachment = Attachment(
            FileContent(encoded_file),
            FileName(f"Group{group_num}_Simulation_{datetime.now().strftime('%Y%m%d')}.zip"),
            FileType("application/zip"),
            Disposition("attachment"),
        )
        message.attachment = attachment

        resp = SendGridAPIClient(sendgrid_api_key).send(message)
        return resp.status_code in (200, 202)
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
.main-header { font-size: 2.5rem; font-weight: bold; color: #1f4e79; text-align: center; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
.mode-pilot { background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid #2196F3; margin: 1rem 0; }
.mode-final { background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid #ff9800; margin: 1rem 0; }
.success-box { padding: 1rem; background-color: #e8f5e9; border-radius: 5px; border-left: 4px solid #4caf50; margin: 1rem 0; }
.warning-box { padding: 1rem; background-color: #fff3e0; border-radius: 5px; border-left: 4px solid #ff9800; margin: 1rem 0; }
.info-box { padding: 1rem; background-color: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196F3; margin: 1rem 0; }
.error-box { padding: 1rem; background-color: #ffebee; border-radius: 5px; border-left: 4px solid #f44336; margin: 1rem 0; }
.step-header { font-size: 1.3rem; font-weight: bold; color: #1f4e79; margin-top: 1.5rem; margin-bottom: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "mode": None,  # None, "pilot", "final"
        "group_number": None,
        "team_members": [],
        "project_title": "",
        "project_description": "",
        "target_population": "US adults 18-65",
        "sample_size": 300,
        "qsf_preview_result": None,
        "conditions": [],
        "factors": [],
        "scales": [],
        "additional_vars": [],
        "effect_sizes": [],
        "checklist": {
            "team_info": False,
            "project_info": False,
            "qsf_uploaded": False,
            "qsf_validated": False,
            "conditions_defined": False,
            "variables_confirmed": False,
            "effect_sizes_set": False,
            "simulation_params": False,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def render_mode_selector() -> None:
    st.markdown("### Select Simulation Mode")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="mode-pilot">
        <h4>🧪 Piloting Mode</h4>
        <p><strong>Unlimited use</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Select Piloting Mode", use_container_width=True):
            st.session_state.mode = "pilot"
            st.rerun()

    with col2:
        api_manager = get_api_manager()
        group_manager = get_group_manager()
        final_enabled = bool(getattr(api_manager, "is_final_mode_enabled", lambda: False)())
        registered = {}
        try:
            registered = group_manager.get_all_groups()
        except Exception:
            registered = {}

        st.markdown(
            f"""
        <div class="mode-final">
        <h4>🎯 Final Data Collection Mode</h4>
        <p><strong>One-time use per group</strong></p>
        <p>Status: {'✅ Enabled' if final_enabled else '❌ Not enabled'}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if final_enabled and registered:
            if st.button("Select Final Mode", use_container_width=True):
                st.session_state.mode = "final"
                st.rerun()
        elif not final_enabled:
            st.warning("Final mode not enabled.")
        else:
            st.warning("No groups registered yet.")

    if st.session_state.mode:
        st.info("Mode: **Piloting**" if st.session_state.mode == "pilot" else "Mode: **Final**")


def render_sidebar() -> None:
    st.sidebar.markdown("## 🔬 BDS5010 Simulation")

    if st.session_state.mode:
        st.sidebar.markdown("**Mode:** 🧪 Piloting" if st.session_state.mode == "pilot" else "**Mode:** 🎯 Final")
    else:
        st.sidebar.markdown("**Mode:** not selected")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Checklist")
    labels = [
        ("team_info", "Team Information"),
        ("project_info", "Project Details"),
        ("qsf_uploaded", "QSF File Uploaded"),
        ("qsf_validated", "QSF Validated"),
        ("conditions_defined", "Conditions Defined"),
        ("variables_confirmed", "Variables Confirmed"),
        ("effect_sizes_set", "Effect Sizes Set"),
        ("simulation_params", "Parameters Set"),
    ]
    for k, lab in labels:
        st.sidebar.markdown(f"✅ {lab}" if st.session_state.checklist.get(k) else f"⬜ {lab}")

    done = sum(bool(v) for v in st.session_state.checklist.values())
    total = len(st.session_state.checklist)
    st.sidebar.progress(done / total if total else 0.0)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Form", use_container_width=True):
        preserved_mode = st.session_state.get("mode")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_session_state()
        st.session_state.mode = preserved_mode
        st.rerun()


def step_team_info() -> None:
    st.markdown('<p class="step-header">Step 1: Team Information</p>', unsafe_allow_html=True)

    mode = st.session_state.mode
    group_manager = get_group_manager()

    if mode == "final":
        try:
            registered = group_manager.get_all_groups()
        except Exception:
            registered = {}

        if not registered:
            st.error("No groups registered.")
            st.session_state.checklist["team_info"] = False
            return

        available = {n: g for n, g in registered.items() if not getattr(g, "final_mode_used", False)}
        if not available:
            st.error("All groups already used Final Mode.")
            st.session_state.checklist["team_info"] = False
            return

        options = {f"Group {n} - {getattr(g, 'project_title', '') or 'No title'}": n for n, g in available.items()}
        selected = st.selectbox("Select Your Group *", options=list(options.keys()))
        if selected:
            num = int(options[selected])
            grp = available[num]
            st.session_state.group_number = num
            members = getattr(grp, "members", [])
            st.session_state.team_members = [getattr(m, "name", str(m)) for m in members]
            st.session_state.checklist["team_info"] = True
            st.markdown('<div class="success-box">✅ Team verified!</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.group_number = int(
                st.number_input("Group Number *", min_value=1, max_value=50, value=int(st.session_state.get("group_number") or 1))
            )
        with col2:
            num_members = int(st.number_input("Number of Team Members *", min_value=1, max_value=6, value=max(1, len(st.session_state.team_members) or 3)))

        st.markdown("**Team Members:**")
        team: List[str] = []
        cols = st.columns(min(num_members, 3))
        for i in range(num_members):
            with cols[i % 3]:
                existing = st.session_state.team_members[i] if i < len(st.session_state.team_members) else ""
                name = st.text_input(f"Member {i+1} *", value=existing, key=f"member_{i}").strip()
                if name:
                    team.append(name)

        st.session_state.team_members = team

        ok = bool(st.session_state.group_number) and len(team) == num_members and all(team)
        st.session_state.checklist["team_info"] = ok
        if ok:
            try:
                group_manager.record_pilot_run(st.session_state.group_number)
            except Exception:
                pass
            st.markdown('<div class="success-box">✅ Team information complete!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Fill all team member names</div>', unsafe_allow_html=True)


def step_project_info() -> None:
    st.markdown('<p class="step-header">Step 2: Project Details</p>', unsafe_allow_html=True)

    st.session_state.project_title = st.text_input("Project Title *", value=st.session_state.project_title)
    st.session_state.project_description = st.text_area("Project Description *", value=st.session_state.project_description, height=120)

    if st.session_state.project_description and "PersonaLibrary" in globals():
        try:
            domains = PersonaLibrary().detect_domains(st.session_state.project_description, st.session_state.project_title)  # type: ignore[name-defined]
            if domains:
                st.info(f"Detected domains: {', '.join(domains[:5])}")
        except Exception:
            pass

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.target_population = st.text_input("Target Population *", value=st.session_state.target_population)
    with c2:
        st.session_state.sample_size = int(st.number_input("Sample Size *", min_value=50, max_value=1000, value=int(st.session_state.sample_size), step=50))

    ok = bool(st.session_state.project_title.strip()) and bool(st.session_state.project_description.strip()) and st.session_state.sample_size >= 50
    st.session_state.checklist["project_info"] = ok
    if ok:
        st.markdown('<div class="success-box">✅ Project details complete!</div>', unsafe_allow_html=True)


def step_upload_qsf() -> None:
    st.markdown('<p class="step-header">Step 3: Upload Survey Files</p>', unsafe_allow_html=True)

    qsf = st.file_uploader("QSF File *", type=["qsf"])
    if qsf:
        content = qsf.read()
        st.session_state.qsf_content = content
        st.session_state.qsf_hash = compute_file_hash(content)
        st.session_state.checklist["qsf_uploaded"] = True

        if "QSFPreviewParser" in globals():
            try:
                parser = QSFPreviewParser()  # type: ignore[name-defined]
                st.session_state.qsf_preview_result = parser.parse(content)
            except Exception as e:
                st.session_state.qsf_preview_result = None
                st.warning(f"QSF preview failed: {e}")

        st.success(f"Uploaded: {qsf.name}")
        st.caption(f"Hash: {st.session_state.qsf_hash[:16]}...")

    res = st.session_state.get("qsf_preview_result")
    if res:
        st.markdown("### Survey Preview")
        st.write(res)

    if st.button("Confirm QSF Validation"):
        log = ""
        try:
            log = getattr(parser, "generate_log_report", lambda: "")()
        except Exception:
            log = ""
        if st.session_state.get("group_number"):
            save_qsf_log(log, int(st.session_state.group_number))
        st.session_state.checklist["qsf_validated"] = True
        st.success("QSF validated.")


def step_define_conditions() -> None:
    st.markdown('<p class="step-header">Step 4: Experimental Conditions</p>', unsafe_allow_html=True)

    st.session_state.design_type = st.selectbox("Experimental Design *", ["Between-subjects", "Within-subjects", "Mixed design"], index=0)
    num_factors = int(st.number_input("Number of Factors *", min_value=1, max_value=4, value=2))

    factors: List[Dict[str, Any]] = []
    for i in range(num_factors):
        st.markdown(f"**Factor {i+1}:**")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input(f"Factor {i+1} Name *", key=f"factor_name_{i}").strip()
        with c2:
            levels_txt = st.text_input(f"Factor {i+1} Levels (comma-separated) *", key=f"factor_levels_{i}").strip()

        if name and levels_txt:
            levels = [x.strip() for x in levels_txt.split(",") if x.strip()]
            if len(levels) >= 2:
                factors.append({"name": name, "levels": levels})

    st.session_state.factors = factors

    if factors:
        from itertools import product
        combos = list(product(*[f["levels"] for f in factors]))
        st.session_state.conditions = [" x ".join(c) for c in combos]
        st.session_state.checklist["conditions_defined"] = True
        st.markdown('<div class="success-box">✅ Conditions defined!</div>', unsafe_allow_html=True)
        for cond in st.session_state.conditions:
            st.markdown(f"- {cond}")
    else:
        st.session_state.conditions = []
        st.session_state.checklist["conditions_defined"] = False


def step_define_variables() -> None:
    st.markdown('<p class="step-header">Step 5: Variables & Scales</p>', unsafe_allow_html=True)

    num_scales = int(st.number_input("Number of Multi-Item Scales *", min_value=1, max_value=10, value=2))
    scales: List[Dict[str, Any]] = []

    for i in range(num_scales):
        st.markdown(f"**Scale {i+1}:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name *", key=f"scale_name_{i}").strip()
        with c2:
            items = int(st.number_input("Items *", min_value=1, max_value=20, value=5, key=f"scale_items_{i}"))
        with c3:
            points = int(st.selectbox("Points *", [5, 6, 7, 9, 10, 11], index=1, key=f"scale_points_{i}"))

        reverse = st.text_input("Reverse-coded items (comma-separated, optional)", key=f"scale_reverse_{i}").strip()
        rev_items = [int(x.strip()) for x in reverse.split(",") if x.strip().isdigit()]

        if name:
            scales.append({"name": name, "num_items": items, "scale_points": points, "reverse_items": rev_items})

    st.session_state.scales = scales

    add_vars_text = st.text_area("Additional single-item measures (name, min, max per line)", value="WTP, 0, 10")
    additional_vars: List[Dict[str, Any]] = []
    for line in add_vars_text.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                additional_vars.append({"name": parts[0], "min": int(parts[1]), "max": int(parts[2])})
            except Exception:
                pass
    st.session_state.additional_vars = additional_vars

    ok = bool(scales) and all(s["name"] for s in scales)
    st.session_state.checklist["variables_confirmed"] = ok
    if ok:
        st.markdown('<div class="success-box">✅ Variables defined!</div>', unsafe_allow_html=True)


def step_effect_sizes() -> None:
    st.markdown('<p class="step-header">Step 6: Expected Effect Sizes</p>', unsafe_allow_html=True)

    if not st.session_state.conditions or not st.session_state.scales:
        st.warning("Define conditions and scales first.")
        st.session_state.checklist["effect_sizes_set"] = False
        return

    num_effects = int(st.number_input("Number of expected effects", min_value=0, max_value=10, value=1))
    effects: List[Any] = []

    if "EffectSizeSpec" not in globals():
        st.info("Effect size UI requires utils.enhanced_simulation_engine. After you apply the utils patch below, this will work.")
        st.session_state.effect_sizes = []
        st.session_state.checklist["effect_sizes_set"] = True
        return

    scale_names = [s["name"] for s in st.session_state.scales]
    factor_names = [f["name"] for f in st.session_state.factors]

    for i in range(num_effects):
        c1, c2 = st.columns(2)
        with c1:
            dv = st.selectbox("Dependent Variable", scale_names, key=f"dv_{i}")
            fac = st.selectbox("Factor", factor_names, key=f"fac_{i}")
        with c2:
            factor_data = next((f for f in st.session_state.factors if f["name"] == fac), None)
            levels = (factor_data or {}).get("levels", [])
            hi = st.selectbox("Higher level", levels, key=f"hi_{i}") if levels else ""
            lo = st.selectbox("Lower level", [x for x in levels if x != hi], key=f"lo_{i}") if levels else ""
            d = float(st.slider("Cohen's d", 0.0, 1.5, 0.5, 0.1, key=f"d_{i}"))

        if dv and fac and hi and lo:
            effects.append(EffectSizeSpec(variable=dv, factor=fac, level_high=hi, level_low=lo, cohens_d=d))  # type: ignore[name-defined]

    st.session_state.effect_sizes = effects
    st.session_state.checklist["effect_sizes_set"] = True
    st.markdown('<div class="success-box">✅ Effect sizes set!</div>', unsafe_allow_html=True)


def step_simulation_params() -> None:
    st.markdown('<p class="step-header">Step 7: Simulation Parameters</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.gender_quota = int(st.slider("Male %", 0, 100, 50))
        st.session_state.age_mean = int(st.number_input("Mean Age", 18, 65, 35))
        st.session_state.age_sd = int(st.number_input("Age SD", 5, 20, 12))
    with c2:
        st.session_state.attention_rate = int(st.slider("Attention Check Pass %", 70, 100, 95))
        st.session_state.min_completion_time = int(st.number_input("Min completion (sec)", 30, 300, 60))
        st.session_state.max_completion_time = int(st.number_input("Max completion (sec)", 300, 3600, 1800))

    st.session_state.checklist["simulation_params"] = True
    st.markdown('<div class="success-box">✅ Parameters set!</div>', unsafe_allow_html=True)


def generate_simulation() -> None:
    st.markdown('<p class="step-header">Generate Simulation</p>', unsafe_allow_html=True)

    missing = [k for k, v in st.session_state.checklist.items() if not v]
    if missing:
        st.error("Complete checklist items: " + ", ".join(missing))
        return

    if CUSTOM_IMPORT_ERROR:
        st.error("Custom modules failed to import. Apply the utils patch below, then redeploy.")
        st.code(CUSTOM_IMPORT_ERROR)
        return

    if st.button("🚀 Generate & Submit Simulation", type="primary", use_container_width=True):
        try:
            exclusion = ExclusionCriteria(  # type: ignore[name-defined]
                completion_time_min_seconds=int(st.session_state.min_completion_time),
                completion_time_max_seconds=int(st.session_state.max_completion_time),
                attention_check_threshold=0.5,
                straight_line_threshold=10,
            )

            engine = EnhancedSimulationEngine(  # type: ignore[name-defined]
                study_title=st.session_state.project_title,
                study_description=st.session_state.project_description,
                sample_size=int(st.session_state.sample_size),
                conditions=st.session_state.conditions,
                factors=st.session_state.factors,
                scales=st.session_state.scales,
                additional_vars=st.session_state.additional_vars,
                demographics={
                    "gender_quota": int(st.session_state.gender_quota),
                    "age_mean": int(st.session_state.age_mean),
                    "age_sd": int(st.session_state.age_sd),
                },
                attention_rate=float(st.session_state.attention_rate) / 100.0,
                effect_sizes=st.session_state.effect_sizes,
                exclusion_criteria=exclusion,
                mode=st.session_state.mode,
            )

            df, metadata = engine.generate()

            metadata.update(
                {
                    "group_number": st.session_state.group_number,
                    "team_members": st.session_state.team_members,
                    "project_title": st.session_state.project_title,
                    "project_description": st.session_state.project_description,
                    "design_type": st.session_state.design_type,
                    "target_population": st.session_state.target_population,
                    "sample_size": st.session_state.sample_size,
                    "conditions": st.session_state.conditions,
                    "factors": st.session_state.factors,
                    "scales": st.session_state.scales,
                    "file_hashes": {"qsf": st.session_state.get("qsf_hash", "")},
                }
            )

            explainer = engine.generate_explainer()
            r_script = engine.generate_r_export(df)
            pdf_buffer = generate_audit_log_pdf(metadata, df)  # type: ignore[name-defined]

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Simulated.csv", df.to_csv(index=False))
                zf.writestr("Column_Explainer.txt", explainer)
                zf.writestr("Audit_Log.pdf", pdf_buffer.getvalue())
                zf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
                zf.writestr("R_Data_Prep.R", r_script)

            zip_buffer.seek(0)

            st.success("Simulation generated.")
            email_sent = send_instructor_email(zip_buffer, metadata)
            zip_buffer.seek(0)

            st.dataframe(df.head(10), use_container_width=True)

            mode_label = "FINAL" if st.session_state.mode == "final" else "PILOT"
            filename = f"{mode_label}_Group{st.session_state.group_number}_{datetime.now().strftime('%Y%m%d')}.zip"
            st.download_button("📥 Download ZIP", data=zip_buffer.getvalue(), file_name=filename, mime="application/zip", use_container_width=True)

            if email_sent:
                st.info(f"Sent to {INSTRUCTOR_EMAIL}.")
            else:
                st.warning("Email not sent. Download ZIP and email manually.")

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.exception(e)


def main() -> None:
    st.markdown('<p class="main-header">🔬 BDS5010 Simulation Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Behavioral Experiment Data Simulation | Spring 2026</p>', unsafe_allow_html=True)

    render_sidebar()

    if not st.session_state.mode:
        render_mode_selector()
        return

    tabs = st.tabs(["1️⃣ Team", "2️⃣ Project", "3️⃣ QSF Upload", "4️⃣ Conditions", "5️⃣ Variables", "6️⃣ Effects", "7️⃣ Parameters", "🚀 Generate"])
    with tabs[0]:
        step_team_info()
    with tabs[1]:
        step_project_info()
    with tabs[2]:
        step_upload_qsf()
    with tabs[3]:
        step_define_conditions()
    with tabs[4]:
        step_define_variables()
    with tabs[5]:
        step_effect_sizes()
    with tabs[6]:
        step_simulation_params()
    with tabs[7]:
        generate_simulation()

    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
BDS5010 Behavioral Data Science | Prof. Dr. Eugen Dimant | Spring 2026<br>
Email submission works when SendGrid is configured
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
