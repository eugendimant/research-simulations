"""
A Streamlit application for generating synthetic behavioral experiment data.

Two Modes:
1. PILOTING MODE - Unlimited use, Python-based simulation
2. FINAL DATA COLLECTION MODE - One-time use per group, Claude API-powered
BDS5010 Behavioral Experiment Simulation Tool
A Streamlit application for generating synthetic behavioral experiment data
following the methodology from "Simulating Behavioral Experiments with LLMs"

Prof. Dr. Eugen Dimant - Spring 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
import zipfile
import io
from datetime import datetime
from pathlib import Path
import base64
import re

# Import custom modules
from utils.qsf_parser import parse_qsf_file, extract_survey_structure
from utils.simulation_engine import SimulationEngine
from utils.enhanced_simulation_engine import (
    EnhancedSimulationEngine,
    EffectSizeSpec,
    ExclusionCriteria
)
from utils.persona_library import PersonaLibrary
from utils.pdf_generator import generate_audit_log_pdf
from utils.schema_validator import validate_schema, generate_schema_summary
from utils.qsf_preview import QSFPreviewParser, QSFCorrections
from utils.instructor_report import InstructorReportGenerator, generate_instructor_package
from utils.group_management import GroupManager, APIKeyManager, create_sample_groups_file

# Email imports
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# Page configuration
st.set_page_config(
    page_title="BDS5010 Simulation Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-pilot {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .mode-final {
        background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #e8f5e9;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3e0;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #ffebee;
        border-radius: 5px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .qsf-preview {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f4e79;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'step': 1,
        'mode': 'pilot',  # 'pilot' or 'final'
        'group_number': None,
        'team_members': [],
        'project_title': '',
        'project_description': '',
        'sample_size': 300,
        'qsf_preview_result': None,
        'qsf_corrections': None,
        'conditions': [],
        'factors': [],
        'scales': [],
        'additional_vars': [],
        'effect_sizes': [],
        'simulation_result': None,
        'checklist': {
            'team_info': False,
            'project_info': False,
            'qsf_uploaded': False,
            'qsf_validated': False,
            'conditions_defined': False,
            'variables_confirmed': False,
            'effect_sizes_set': False,
            'simulation_params': False
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Instructor email configuration
INSTRUCTOR_EMAIL = "edimant@sas.upenn.edu"

def send_instructor_email(zip_buffer, metadata):
    """
    Send simulation package to instructor via email
    """
    try:
        # Get SendGrid API key from Streamlit secrets
        sendgrid_api_key = st.secrets.get("SENDGRID_API_KEY")
        
        if not sendgrid_api_key:
            st.warning("⚠️ Email notification not configured. Download manually and send to instructor.")
            return False
        
        # Prepare email content
        group_num = metadata.get('group_number', 'Unknown')
        team_members = ', '.join(metadata.get('team_members', []))
        project_title = metadata.get('project_title', 'Untitled')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        email_body = f"""
BDS5010 Simulation Submission

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
{chr(10).join([f"  - {f['name']}: {', '.join(f['levels'])}" for f in metadata.get('factors', [])])}

Measurement Scales:
{chr(10).join([f"  - {s['name']}: {s['num_items']} items, {s['scale_points']}-point scale" for s in metadata.get('scales', [])])}

File Integrity:
- QSF Hash: {metadata.get('file_hashes', {}).get('qsf', 'N/A')}

This is an automated submission from the BDS5010 Simulation Tool.
The attached ZIP file contains:
  - Simulated.csv (synthetic dataset)
  - Column_Explainer.txt (variable descriptions)
  - Audit_Log.pdf (tamper-proof record)
  - metadata.json (verification data)
"""

        # Create email
        message = Mail(
            from_email='noreply@bds5010.streamlit.app',  # Sender (will be overridden by SendGrid)
            to_emails=INSTRUCTOR_EMAIL,
            subject=f'BDS5010 Simulation - Group {group_num} - {project_title}',
            plain_text_content=email_body
        )
        
        # Attach ZIP file
        zip_data = zip_buffer.getvalue()
        encoded_file = base64.b64encode(zip_data).decode()
        
        attachment = Attachment(
            FileContent(encoded_file),
            FileName(f'Group{group_num}_Simulation_{datetime.now().strftime("%Y%m%d")}.zip'),
            FileType('application/zip'),
            Disposition('attachment')
        )
        message.attachment = attachment
        
        # Send email
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        
        if response.status_code in [200, 202]:
            return True
        else:
            st.warning(f"Email sent with status {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Failed to send email notification: {str(e)}")
        st.info("Please download the file manually and email it to the instructor.")
        return False


def save_qsf_log(log_content: str, group_number: int) -> Path:
    """Save QSF parsing log for instructor review."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"QSF_Log_Group{group_number}_{timestamp}.txt"
    filepath = LOG_STORAGE / filename

    with open(filepath, 'w') as f:
        f.write(log_content)

    return filepath


def render_mode_selector():
    """Render the mode selection interface."""
    st.markdown("### Select Simulation Mode")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="mode-pilot">
        <h4>🧪 Piloting Mode</h4>
        <p><strong>Unlimited use</strong> - Test your survey design</p>
        <ul>
            <li>Python-based simulation engine</li>
            <li>High-quality persona-driven data</li>
            <li>Run as many times as needed</li>
            <li>Perfect for testing and iteration</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select Piloting Mode", key="select_pilot", use_container_width=True):
            st.session_state.mode = 'pilot'
            st.rerun()

    with col2:
        api_manager = get_api_manager()
        group_manager = get_group_manager()

        final_enabled = api_manager.is_final_mode_enabled()
        registered_groups = group_manager.get_all_groups()

        st.markdown(f"""
        <div class="mode-final">
        <h4>🎯 Final Data Collection Mode</h4>
        <p><strong>One-time use per group</strong> - Claude API-powered</p>
        <ul>
            <li>Claude Opus 4.5 generates optimal data</li>
            <li>Highest quality simulation</li>
            <li>Each group can use only ONCE</li>
            <li>Status: {'✅ Enabled' if final_enabled else '❌ Not enabled'}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        if final_enabled and len(registered_groups) > 0:
            if st.button("Select Final Mode", key="select_final", use_container_width=True):
                st.session_state.mode = 'final'
                st.rerun()
        elif not final_enabled:
            st.warning("Final mode not yet enabled by instructor")
        else:
            st.warning("No groups registered yet")

    # Show current mode
    if st.session_state.mode:
        mode_name = "Piloting" if st.session_state.mode == 'pilot' else "Final Data Collection"
        st.info(f"Currently in **{mode_name} Mode**")


def render_sidebar():
    """Render the sidebar with checklist and info."""
    st.sidebar.markdown("## 🔬 BDS5010 Simulation")

    # Mode indicator
    mode = st.session_state.mode
    if mode == 'pilot':
        st.sidebar.markdown("**Mode:** 🧪 Piloting (Unlimited)")
    else:
        st.sidebar.markdown("**Mode:** 🎯 Final (One-time)")

    st.sidebar.markdown("---")

    # Checklist
    st.sidebar.markdown("### 📋 Checklist")

    checklist_items = [
        ('team_info', 'Team Information'),
        ('project_info', 'Project Details'),
        ('qsf_uploaded', 'QSF File Uploaded'),
        ('qsf_validated', 'QSF Validated'),
        ('conditions_defined', 'Conditions Defined'),
        ('variables_confirmed', 'Variables Confirmed'),
        ('effect_sizes_set', 'Effect Sizes Set'),
        ('simulation_params', 'Parameters Set')
    ]

    for key, label in checklist_items:
        if st.session_state.checklist.get(key, False):
            st.sidebar.markdown(f"✅ {label}")
        else:
            st.sidebar.markdown(f"⬜ {label}")

    # Progress
    completed = sum(st.session_state.checklist.values())
    total = len(st.session_state.checklist)
    st.sidebar.progress(completed / total)
    st.sidebar.caption(f"{completed}/{total} completed")

    st.sidebar.markdown("---")

    # Reset button
    if st.sidebar.button("🔄 Reset Form", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['mode']:
                del st.session_state[key]
        init_session_state()
        st.rerun()


def step_team_info():
    """Step 1: Team Information."""
    st.markdown('<p class="step-header">Step 1: Team Information</p>', unsafe_allow_html=True)

    mode = st.session_state.mode
    group_manager = get_group_manager()

    if mode == 'final':
        # Final mode: Select from registered groups
        registered = group_manager.get_all_groups()

        if not registered:
            st.error("No groups registered. Contact instructor.")
            return

        # Filter out groups that already used final mode
        available_groups = {
            num: grp for num, grp in registered.items()
            if not grp.final_mode_used
        }

        if not available_groups:
            st.error("All registered groups have already used Final Data Collection Mode.")
            return

        group_options = {f"Group {num} - {grp.project_title or 'No title'}": num
                        for num, grp in available_groups.items()}

        selected = st.selectbox(
            "Select Your Group *",
            options=list(group_options.keys()),
            help="Select your registered group. Each group can only use Final Mode once."
        )

        if selected:
            group_num = group_options[selected]
            group = available_groups[group_num]
            st.session_state.group_number = group_num
            st.session_state.team_members = [m.name for m in group.members]

            st.markdown("**Team Members:**")
            for member in group.members:
                st.markdown(f"- {member.name}")

            st.session_state.checklist['team_info'] = True
            st.markdown('<div class="success-box">✅ Team verified!</div>', unsafe_allow_html=True)
    else:
        # Pilot mode: Manual entry
        col1, col2 = st.columns(2)

        with col1:
            group_number = st.number_input(
                "Group Number *",
                min_value=1,
                max_value=50,
                value=st.session_state.get('group_number') or 1,
                help="Your assigned group number"
            )
            st.session_state.group_number = group_number

        with col2:
            num_members = st.number_input(
                "Number of Team Members *",
                min_value=1,
                max_value=6,
                value=len(st.session_state.team_members) or 3
            )

        st.markdown("**Team Members:**")
        team_members = []
        cols = st.columns(min(num_members, 3))

        for i in range(num_members):
            col_idx = i % 3
            with cols[col_idx]:
                existing = st.session_state.team_members[i] if i < len(st.session_state.team_members) else ''
                name = st.text_input(f"Member {i+1} *", value=existing, key=f'member_{i}')
                if name:
                    team_members.append(name)

        st.session_state.team_members = team_members

        if group_number and len(team_members) == num_members and all(team_members):
            st.session_state.checklist['team_info'] = True
            st.markdown('<div class="success-box">✅ Team information complete!</div>', unsafe_allow_html=True)

            # Record pilot run
            group_manager.record_pilot_run(group_number)
        else:
            st.session_state.checklist['team_info'] = False
            st.markdown('<div class="warning-box">⚠️ Please fill in all team member names</div>', unsafe_allow_html=True)


def step_project_info():
    """Step 2: Project Information."""
    st.markdown('<p class="step-header">Step 2: Project Details</p>', unsafe_allow_html=True)

    project_title = st.text_input(
        "Project Title *",
        value=st.session_state.project_title,
        placeholder="e.g., AI Product Recommendations and Consumer Behavior"
    )
    st.session_state.project_title = project_title

    project_description = st.text_area(
        "Project Description *",
        value=st.session_state.project_description,
        placeholder="Describe your research question and experimental design in 2-3 sentences. "
                   "This helps the system select appropriate behavioral personas.",
        height=120
    )
    st.session_state.project_description = project_description

    # Show detected domains
    if project_description:
        library = PersonaLibrary()
        domains = library.detect_domains(project_description, project_title)
        if domains:
            st.info(f"**Detected research domains:** {', '.join(domains[:5])}")

    col1, col2 = st.columns(2)

    with col1:
        target_pop = st.text_input(
            "Target Population *",
            value=st.session_state.get('target_population', 'US adults 18-65'),
            help="Who are your participants?"
        )
        st.session_state.target_population = target_pop

    with col2:
        sample_size = st.number_input(
            "Sample Size *",
            min_value=50,
            max_value=1000,
            value=st.session_state.sample_size,
            step=50
        )
        st.session_state.sample_size = sample_size

    if project_title and project_description and sample_size:
        st.session_state.checklist['project_info'] = True
        st.markdown('<div class="success-box">✅ Project details complete!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['project_info'] = False


def step_upload_qsf():
    """Step 3: Upload and Preview QSF."""
    st.markdown('<p class="step-header">Step 3: Upload Survey Files</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Upload your Qualtrics survey file (.qsf)</strong><br>
    Export from Qualtrics: Tools → Import/Export → Export Survey
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        qsf_file = st.file_uploader(
            "QSF File *",
            type=['qsf'],
            help="Export your survey from Qualtrics"
        )

        if qsf_file:
            content = qsf_file.read()
            st.session_state.qsf_content = content
            st.session_state.qsf_hash = compute_file_hash(content)
            st.session_state.checklist['qsf_uploaded'] = True

            # Parse and preview
            parser = QSFPreviewParser()
            result = parser.parse(content)
            st.session_state.qsf_preview_result = result

            st.success(f"✅ Uploaded: {qsf_file.name}")
            st.caption(f"Hash: {st.session_state.qsf_hash[:16]}...")

    with col2:
        screenshots = st.file_uploader(
            "Survey Screenshots (optional)",
            type=['pdf', 'png', 'jpg'],
            accept_multiple_files=True
        )
        if screenshots:
            st.session_state.screenshots = screenshots

    # Show QSF Preview
    if st.session_state.qsf_preview_result:
        result = st.session_state.qsf_preview_result

        st.markdown("### 📋 Survey Preview")

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Questions", result.total_questions)
        col2.metric("Blocks", result.total_blocks)
        col3.metric("Detected Scales", len(result.detected_scales))

        # Validation status
        if result.validation_errors:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("**Validation Errors:**")
            for error in result.validation_errors:
                st.markdown(f"- {error}")
            st.markdown('</div>', unsafe_allow_html=True)

        if result.validation_warnings:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**Warnings:**")
            for warning in result.validation_warnings[:5]:
                st.markdown(f"- {warning}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Expandable details
        with st.expander("View Survey Structure"):
            for block in result.blocks[:10]:  # Limit display
                st.markdown(f"**{block.block_name}** ({len(block.questions)} questions)")
                for q in block.questions[:5]:
                    st.markdown(f"  - [{q.question_type}] {q.question_text[:80]}...")

        with st.expander("View Detected Conditions"):
            if result.detected_conditions:
                for cond in result.detected_conditions:
                    st.markdown(f"- {cond}")
            else:
                st.info("No conditions auto-detected. Please define manually in Step 4.")

        with st.expander("View Detected Scales"):
            if result.detected_scales:
                for scale in result.detected_scales:
                    st.markdown(f"- **{scale['name']}**: {scale.get('items', '?')} items, "
                               f"{scale.get('scale_points', '?')}-point scale")
            else:
                st.info("No scales auto-detected. Please define manually in Step 5.")

        # Correction interface
        st.markdown("### 🔧 Corrections")
        st.markdown("If the automatic detection is incorrect, you can override below:")

        corrections_needed = st.checkbox("I need to make corrections to the parsed survey")

        if corrections_needed:
            correction_notes = st.text_area(
                "Describe what needs correction",
                placeholder="e.g., 'The ownership scale has 5 items, not 3' or 'Missing condition: Control'",
                help="These notes will be logged for instructor review"
            )
            st.session_state.correction_notes = correction_notes

        # Save log for instructor
        if st.button("Confirm QSF Validation"):
            # Generate and save log
            log_content = parser.generate_log_report()
            if st.session_state.group_number:
                save_qsf_log(log_content, st.session_state.group_number)

            st.session_state.checklist['qsf_validated'] = True
            st.success("✅ QSF validated! Log saved for instructor review.")


def step_define_conditions():
    """Step 4: Define Experimental Conditions."""
    st.markdown('<p class="step-header">Step 4: Experimental Conditions</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Define your experimental conditions. For a 2x2 design, you have 4 conditions.
    </div>
    """, unsafe_allow_html=True)

    # Auto-detected conditions
    if st.session_state.qsf_preview_result:
        detected = st.session_state.qsf_preview_result.detected_conditions
        if detected:
            st.info(f"Auto-detected conditions: {detected}")

    # Design type
    design_type = st.selectbox(
        "Experimental Design *",
        ["Between-subjects", "Within-subjects", "Mixed design"],
        index=0
    )
    st.session_state.design_type = design_type

    # Factors
    num_factors = st.number_input("Number of Factors *", min_value=1, max_value=4, value=2)

    factors = []
    for i in range(num_factors):
        st.markdown(f"**Factor {i+1}:**")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                f"Factor {i+1} Name *",
                value=st.session_state.factors[i]['name'] if i < len(st.session_state.factors) else '',
                placeholder="e.g., AI Mention",
                key=f'factor_name_{i}'
            )

        with col2:
            levels = st.text_input(
                f"Factor {i+1} Levels (comma-separated) *",
                value=', '.join(st.session_state.factors[i]['levels']) if i < len(st.session_state.factors) else '',
                placeholder="e.g., AI, No AI",
                key=f'factor_levels_{i}'
            )

        if name and levels:
            level_list = [l.strip() for l in levels.split(',') if l.strip()]
            factors.append({'name': name, 'levels': level_list})

    st.session_state.factors = factors

    # Generate conditions
    if factors and all(f['levels'] for f in factors):
        from itertools import product

        level_combos = list(product(*[f['levels'] for f in factors]))
        conditions = [' x '.join(combo) for combo in level_combos]
        st.session_state.conditions = conditions

        st.markdown("**Generated Conditions:**")
        n_per = st.session_state.sample_size // len(conditions)
        for cond in conditions:
            st.markdown(f"- **{cond}** (n = {n_per})")

        st.session_state.checklist['conditions_defined'] = True
        st.markdown('<div class="success-box">✅ Conditions defined!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['conditions_defined'] = False


def step_define_variables():
    """Step 5: Define Variables and Scales."""
    st.markdown('<p class="step-header">Step 5: Variables & Scales</p>', unsafe_allow_html=True)

    # Number of scales
    num_scales = st.number_input("Number of Multi-Item Scales *", min_value=1, max_value=10, value=2)

    scales = []
    for i in range(num_scales):
        st.markdown(f"---\n**Scale {i+1}:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input(
                f"Scale {i+1} Name *",
                value=st.session_state.scales[i]['name'] if i < len(st.session_state.scales) else '',
                placeholder="e.g., Psychological Ownership",
                key=f'scale_name_{i}'
            )

        with col2:
            items = st.number_input(
                f"Number of Items *",
                min_value=1, max_value=20, value=5,
                key=f'scale_items_{i}'
            )

        with col3:
            points = st.selectbox(
                f"Scale Points *",
                [5, 6, 7, 9, 10, 11],
                index=1,
                key=f'scale_points_{i}'
            )

        reverse = st.text_input(
            f"Reverse-coded items (comma-separated, or blank)",
            placeholder="e.g., 3, 5",
            key=f'scale_reverse_{i}'
        )

        if name:
            scales.append({
                'name': name,
                'num_items': items,
                'scale_points': points,
                'reverse_items': [int(x.strip()) for x in reverse.split(',') if x.strip().isdigit()]
            })

    st.session_state.scales = scales

    # Additional single-item variables
    st.markdown("---\n**Additional Single-Item Measures:**")
    add_vars_text = st.text_area(
        "Format: name, min, max (one per line)",
        value=st.session_state.get('add_vars_text', 'WTP, 0, 10'),
        height=100
    )
    st.session_state.add_vars_text = add_vars_text

    additional_vars = []
    for line in add_vars_text.strip().split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                additional_vars.append({
                    'name': parts[0],
                    'min': int(parts[1]),
                    'max': int(parts[2])
                })

    st.session_state.additional_vars = additional_vars

    if scales and all(s['name'] for s in scales):
        st.session_state.checklist['variables_confirmed'] = True
        st.markdown('<div class="success-box">✅ Variables defined!</div>', unsafe_allow_html=True)


def step_effect_sizes():
    """Step 6: Expected Effect Sizes."""
    st.markdown('<p class="step-header">Step 6: Expected Effect Sizes</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Define your expected effects.</strong><br>
    This helps generate hypothesis-informed (but not deterministic) data.
    Effect sizes use Cohen's d: small (~0.2), medium (~0.5), large (~0.8).
    </div>
    """, unsafe_allow_html=True)

    # Check if we have conditions and variables
    if not st.session_state.conditions or not st.session_state.scales:
        st.warning("Please define conditions and scales first.")
        return

    num_effects = st.number_input(
        "Number of expected effects",
        min_value=0, max_value=10,
        value=len(st.session_state.effect_sizes) or 1
    )

    effect_sizes = []

    for i in range(num_effects):
        st.markdown(f"**Effect {i+1}:**")
        col1, col2 = st.columns(2)

        with col1:
            # Variable selection
            scale_names = [s['name'] for s in st.session_state.scales]
            var = st.selectbox(
                f"Dependent Variable *",
                options=scale_names,
                key=f'effect_var_{i}'
            )

            # Factor selection
            factor_names = [f['name'] for f in st.session_state.factors]
            factor = st.selectbox(
                f"Factor *",
                options=factor_names,
                key=f'effect_factor_{i}'
            )

        with col2:
            # Get levels for selected factor
            factor_data = next((f for f in st.session_state.factors if f['name'] == factor), None)
            levels = factor_data['levels'] if factor_data else []

            if len(levels) >= 2:
                high_level = st.selectbox(
                    f"Higher condition",
                    options=levels,
                    key=f'effect_high_{i}'
                )
                low_level = st.selectbox(
                    f"Lower condition",
                    options=[l for l in levels if l != high_level],
                    key=f'effect_low_{i}'
                )
            else:
                high_level, low_level = '', ''

            cohens_d = st.slider(
                f"Cohen's d",
                min_value=0.0, max_value=1.5, value=0.5, step=0.1,
                key=f'effect_d_{i}'
            )

        if var and factor and high_level and low_level:
            effect_sizes.append(EffectSizeSpec(
                variable=var,
                factor=factor,
                level_high=high_level,
                level_low=low_level,
                cohens_d=cohens_d
            ))

    st.session_state.effect_sizes = effect_sizes

    st.session_state.checklist['effect_sizes_set'] = True
    st.markdown('<div class="success-box">✅ Effect sizes set!</div>', unsafe_allow_html=True)


def step_simulation_params():
    """Step 7: Simulation Parameters."""
    st.markdown('<p class="step-header">Step 7: Simulation Parameters</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demographics:**")
        gender_quota = st.slider("Male %", 0, 100, 50)
        st.session_state.gender_quota = gender_quota

        age_mean = st.number_input("Mean Age", 18, 65, 35)
        st.session_state.age_mean = age_mean

        age_sd = st.number_input("Age SD", 5, 20, 12)
        st.session_state.age_sd = age_sd

    with col2:
        st.markdown("**Response Quality:**")
        attention_rate = st.slider("Attention Check Pass %", 70, 100, 95)
        st.session_state.attention_rate = attention_rate

        st.markdown("**Exclusion Criteria:**")
        min_time = st.number_input("Min completion (sec)", 30, 300, 60)
        st.session_state.min_completion_time = min_time

        max_time = st.number_input("Max completion (sec)", 300, 3600, 1800)
        st.session_state.max_completion_time = max_time

    st.markdown("**Persona Weights:**")
    st.caption("Adjust the distribution of response styles")

    col1, col2, col3 = st.columns(3)
    with col1:
        engaged_pct = st.slider("Engaged %", 0, 50, 30)
    with col2:
        satisficer_pct = st.slider("Satisficer %", 0, 50, 20)
    with col3:
        extreme_pct = st.slider("Extreme %", 0, 30, 8)

    remaining = 100 - engaged_pct - satisficer_pct - extreme_pct
    st.info(f"Neutral responders: {max(0, remaining)}%")

    st.session_state.persona_weights = {
        'engaged': engaged_pct / 100,
        'satisficer': satisficer_pct / 100,
        'extreme': extreme_pct / 100,
        'neutral': max(0, remaining) / 100
    }

    st.session_state.checklist['simulation_params'] = True
    st.markdown('<div class="success-box">✅ Parameters set!</div>', unsafe_allow_html=True)


def generate_simulation():
    """Generate and download simulation."""
    st.markdown('<p class="step-header">Generate Simulation</p>', unsafe_allow_html=True)

    mode = st.session_state.mode

    # Check prerequisites
    incomplete = [k for k, v in st.session_state.checklist.items() if not v]
    if incomplete:
        st.error(f"Please complete: {', '.join(incomplete)}")
        return

    st.markdown("""
    <div class="info-box">
    Ready to generate your simulation! This will:
    <ul>
        <li>📊 Create your synthetic dataset</li>
        <li>📖 Generate variable descriptions</li>
        <li>📋 Produce tamper-proof audit log</li>
        <li>📧 <strong>Automatically email everything to Prof. Dimant</strong></li>
    </ul>
    You'll also get a download link for your records.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate & Submit Simulation", type="primary", use_container_width=True):
        with st.spinner("Generating simulation... This may take a moment."):
            try:
                # Build effect sizes
                effect_specs = st.session_state.effect_sizes

                # Build exclusion criteria
                exclusion = ExclusionCriteria(
                    completion_time_min_seconds=st.session_state.get('min_completion_time', 60),
                    completion_time_max_seconds=st.session_state.get('max_completion_time', 1800),
                    attention_check_threshold=0.5,
                    straight_line_threshold=10
                )

                # Create engine
                engine = EnhancedSimulationEngine(
                    study_title=st.session_state.project_title,
                    study_description=st.session_state.project_description,
                    sample_size=st.session_state.sample_size,
                    conditions=st.session_state.conditions,
                    factors=st.session_state.factors,
                    scales=st.session_state.scales,
                    additional_vars=st.session_state.additional_vars,
                    demographics={
                        'gender_quota': st.session_state.gender_quota,
                        'age_mean': st.session_state.age_mean,
                        'age_sd': st.session_state.age_sd
                    },
                    attention_rate=st.session_state.attention_rate / 100,
                    effect_sizes=effect_specs,
                    exclusion_criteria=exclusion,
                    mode=mode
                )

                # Generate data
                df, metadata = engine.generate()

                # Add team info to metadata
                metadata['group_number'] = st.session_state.group_number
                metadata['team_members'] = st.session_state.team_members
                metadata['project_title'] = st.session_state.project_title

                st.session_state.simulation_result = df

                # Generate explainer
                explainer = engine.generate_explainer()

                # Generate R script
                r_script = engine.generate_r_export(df)

                # Generate instructor report
                report_gen = InstructorReportGenerator(df, metadata, effect_specs)
                instructor_report = report_gen.generate_full_report()
                instructor_json = report_gen.generate_json_report()

                # Generate PDF audit log
                pdf_buffer = generate_audit_log_pdf(metadata, df)

                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Student files
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr('Simulated.csv', csv_buffer.getvalue())
                    zf.writestr('Column_Explainer.txt', explainer)
                    zf.writestr('Audit_Log.pdf', pdf_buffer.getvalue())
                    zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=str))

                    # R export (for students to use if they want)
                    zf.writestr('R_Data_Prep.R', r_script)

                st.success("✅ Simulation generated successfully!")

                # Send email to instructor
                with st.spinner("📧 Sending to instructor..."):
                    email_sent = send_instructor_email(zip_buffer, metadata)
                    zip_buffer.seek(0)

                if email_sent:
                    st.success(f"✅ Email sent to {INSTRUCTOR_EMAIL}!")
                else:
                    st.warning("⚠️ Automatic email failed. Please download and email manually.")

                # Display preview
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown(f"**Total rows:** {len(df)} | **Columns:** {len(df.columns)}")

                # Show exclusion summary
                if 'Exclude_Recommended' in df.columns:
                    excluded = df['Exclude_Recommended'].sum()
                    st.info(f"**Exclusion flags:** {excluded} participants ({excluded/len(df)*100:.1f}%) flagged for potential exclusion")

                # Download
                mode_label = "FINAL" if mode == 'final' else "PILOT"
                filename = f"{mode_label}_Group{st.session_state.group_number}_{datetime.now().strftime('%Y%m%d')}.zip"

                st.download_button(
                    label="📥 Download Your Copy (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=filename,
                    mime="application/zip",
                    use_container_width=True
                )

                if email_sent:
                    st.markdown("""
                    <div class="success-box">
                    <strong>✅ Submission Complete!</strong><br>
                    Your simulation has been automatically sent to Prof. Dimant.<br>
                    Download your copy above for your records.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                    <strong>⚠️ Manual Submission Required</strong><br>
                    1. Download the ZIP file above<br>
                    2. Email it to edimant@sas.upenn.edu<br>
                    3. Subject: BDS5010 Simulation - Group [Your Group Number]
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating simulation: {str(e)}")
                st.exception(e)


def main():
    """Main application flow."""
    st.markdown('<p class="main-header">🔬 BDS5010 Simulation Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Behavioral Experiment Data Simulation | Spring 2026</p>', unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Mode selection (if not set)
    if not st.session_state.mode:
        render_mode_selector()
        return

    # Main content tabs
    tabs = st.tabs([
        "1️⃣ Team",
        "2️⃣ Project",
        "3️⃣ QSF Upload",
        "4️⃣ Conditions",
        "5️⃣ Variables",
        "6️⃣ Effects",
        "7️⃣ Parameters",
        "🚀 Generate"
    ])

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

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    BDS5010 Behavioral Data Science | Prof. Dr. Eugen Dimant | Spring 2026<br>
    Submissions automatically sent to edimant@sas.upenn.edu
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
