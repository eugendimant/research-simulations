"""
BDS5010 Behavioral Experiment Simulation Tool
==============================================
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
from utils.pdf_generator import generate_audit_log_pdf
from utils.schema_validator import validate_schema, generate_schema_summary

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

# Custom CSS for better UI
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
    .checklist-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .checklist-complete {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .checklist-incomplete {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f4e79;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3e0;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #e8f5e9;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'checklist' not in st.session_state:
    st.session_state.checklist = {
        'team_info': False,
        'project_info': False,
        'qsf_uploaded': False,
        'screenshots_uploaded': False,
        'conditions_defined': False,
        'variables_confirmed': False,
        'simulation_params': False
    }
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = None
if 'simulation_result' not in st.session_state:
    st.session_state.simulation_result = None

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
================================

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

================================
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

def compute_file_hash(file_content):
    """Compute SHA-256 hash for file integrity verification"""
    return hashlib.sha256(file_content).hexdigest()

def render_progress_bar():
    """Render the progress indicator"""
    total_items = len(st.session_state.checklist)
    completed = sum(st.session_state.checklist.values())
    progress = completed / total_items

    st.progress(progress)
    st.caption(f"Checklist Progress: {completed}/{total_items} items completed")

def render_checklist_sidebar():
    """Render the checklist in the sidebar"""
    st.sidebar.markdown("## 📋 Submission Checklist")

    checklist_items = [
        ('team_info', '👥 Team Information'),
        ('project_info', '📝 Project Details'),
        ('qsf_uploaded', '📄 QSF File Uploaded'),
        ('screenshots_uploaded', '🖼️ Screenshots Uploaded'),
        ('conditions_defined', '🔀 Conditions Defined'),
        ('variables_confirmed', '📊 Variables Confirmed'),
        ('simulation_params', '⚙️ Simulation Parameters')
    ]

    for key, label in checklist_items:
        if st.session_state.checklist[key]:
            st.sidebar.markdown(f"✅ {label}")
        else:
            st.sidebar.markdown(f"⬜ {label}")

    st.sidebar.markdown("---")
    completed = sum(st.session_state.checklist.values())
    total = len(st.session_state.checklist)

    if completed == total:
        st.sidebar.success("🎉 Ready to generate simulation!")
    else:
        st.sidebar.warning(f"Complete {total - completed} more item(s)")

def step1_team_info():
    """Step 1: Collect team information"""
    st.markdown('<p class="step-header">Step 1: Team Information</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        group_number = st.number_input(
            "Group Number *",
            min_value=1,
            max_value=50,
            value=st.session_state.get('group_number', 1),
            help="Your assigned group number for BDS5010"
        )
        st.session_state.group_number = group_number

    with col2:
        num_members = st.number_input(
            "Number of Team Members *",
            min_value=1,
            max_value=6,
            value=st.session_state.get('num_members', 3),
            help="How many students are in your team?"
        )
        st.session_state.num_members = num_members

    st.markdown("**Team Members:**")
    team_members = []
    cols = st.columns(min(num_members, 3))

    for i in range(num_members):
        col_idx = i % 3
        with cols[col_idx]:
            name = st.text_input(
                f"Member {i+1} Name *",
                value=st.session_state.get(f'member_{i}', ''),
                key=f'member_input_{i}'
            )
            st.session_state[f'member_{i}'] = name
            if name:
                team_members.append(name)

    st.session_state.team_members = team_members

    # Check completion
    if group_number and len(team_members) == num_members and all(team_members):
        st.session_state.checklist['team_info'] = True
        st.markdown('<div class="success-box">✅ Team information complete!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['team_info'] = False
        st.markdown('<div class="warning-box">⚠️ Please fill in all team member names</div>', unsafe_allow_html=True)

def step2_project_info():
    """Step 2: Collect project information"""
    st.markdown('<p class="step-header">Step 2: Project Details</p>', unsafe_allow_html=True)

    project_title = st.text_input(
        "Project Title *",
        value=st.session_state.get('project_title', ''),
        placeholder="e.g., AI Product Recommendations and Consumer Behavior",
        help="A descriptive title for your experiment"
    )
    st.session_state.project_title = project_title

    project_description = st.text_area(
        "Brief Project Description *",
        value=st.session_state.get('project_description', ''),
        placeholder="Describe your research question and experimental design in 2-3 sentences...",
        height=100,
        help="What is your study about? What are you testing?"
    )
    st.session_state.project_description = project_description

    col1, col2 = st.columns(2)

    with col1:
        target_population = st.text_input(
            "Target Population *",
            value=st.session_state.get('target_population', 'US adults 18-65, balanced by gender'),
            help="Who are your participants? (e.g., 'US adults 18-65, balanced by gender')"
        )
        st.session_state.target_population = target_population

    with col2:
        sample_size = st.number_input(
            "Desired Sample Size *",
            min_value=50,
            max_value=1000,
            value=st.session_state.get('sample_size', 300),
            step=50,
            help="Recommended: 300-500 for factorial designs"
        )
        st.session_state.sample_size = sample_size

    # Check completion
    if project_title and project_description and target_population and sample_size:
        st.session_state.checklist['project_info'] = True
        st.markdown('<div class="success-box">✅ Project details complete!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['project_info'] = False
        st.markdown('<div class="warning-box">⚠️ Please fill in all required fields</div>', unsafe_allow_html=True)

def step3_upload_files():
    """Step 3: Upload QSF and screenshots"""
    st.markdown('<p class="step-header">Step 3: Upload Survey Files</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Required Files:</strong><br>
    1. <strong>QSF File:</strong> Export from Qualtrics via Tools → Import/Export → Export Survey<br>
    2. <strong>Survey Screenshots/PDF:</strong> Print survey to PDF showing all questions and flow
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📄 QSF File (Qualtrics Survey Format)**")
        qsf_file = st.file_uploader(
            "Upload QSF file",
            type=['qsf'],
            help="Export your survey from Qualtrics as a .qsf file",
            key='qsf_uploader'
        )

        if qsf_file:
            st.session_state.qsf_file = qsf_file
            st.session_state.qsf_content = qsf_file.read()
            qsf_file.seek(0)  # Reset file pointer
            st.session_state.qsf_hash = compute_file_hash(st.session_state.qsf_content)
            st.success(f"✅ Uploaded: {qsf_file.name}")
            st.caption(f"File hash: {st.session_state.qsf_hash[:16]}...")
            st.session_state.checklist['qsf_uploaded'] = True
        else:
            st.session_state.checklist['qsf_uploaded'] = False

    with col2:
        st.markdown("**🖼️ Survey Screenshots/PDF**")
        screenshot_files = st.file_uploader(
            "Upload screenshots or PDF",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload the printed survey PDF or screenshots of each page",
            key='screenshot_uploader'
        )

        if screenshot_files:
            st.session_state.screenshot_files = screenshot_files
            st.session_state.screenshot_hashes = []
            for f in screenshot_files:
                content = f.read()
                f.seek(0)
                st.session_state.screenshot_hashes.append({
                    'name': f.name,
                    'hash': compute_file_hash(content)[:16]
                })
                st.success(f"✅ Uploaded: {f.name}")
            st.session_state.checklist['screenshots_uploaded'] = True
        else:
            st.session_state.checklist['screenshots_uploaded'] = False

def step4_define_conditions():
    """Step 4: Define experimental conditions"""
    st.markdown('<p class="step-header">Step 4: Define Experimental Conditions</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Define your experimental conditions (treatment groups). For a 2x2 design, you would have 4 conditions.
    </div>
    """, unsafe_allow_html=True)

    # Design type selection
    design_type = st.selectbox(
        "Experimental Design Type *",
        options=[
            "Between-subjects (each participant sees one condition)",
            "Within-subjects (each participant sees multiple conditions)",
            "Mixed design"
        ],
        index=0,
        help="Most behavioral experiments use between-subjects designs"
    )
    st.session_state.design_type = design_type

    col1, col2 = st.columns(2)

    with col1:
        num_factors = st.number_input(
            "Number of Factors *",
            min_value=1,
            max_value=4,
            value=st.session_state.get('num_factors', 2),
            help="How many independent variables are you manipulating?"
        )
        st.session_state.num_factors = num_factors

    # Define factors
    factors = []
    for i in range(num_factors):
        st.markdown(f"**Factor {i+1}:**")
        col_a, col_b = st.columns(2)

        with col_a:
            factor_name = st.text_input(
                f"Factor {i+1} Name *",
                value=st.session_state.get(f'factor_{i}_name', ''),
                placeholder=f"e.g., AI Mention" if i == 0 else "e.g., Product Type",
                key=f'factor_{i}_name_input'
            )
            st.session_state[f'factor_{i}_name'] = factor_name

        with col_b:
            levels = st.text_input(
                f"Factor {i+1} Levels (comma-separated) *",
                value=st.session_state.get(f'factor_{i}_levels', ''),
                placeholder="e.g., AI, No AI" if i == 0 else "e.g., Hedonic, Utilitarian",
                key=f'factor_{i}_levels_input'
            )
            st.session_state[f'factor_{i}_levels'] = levels

        if factor_name and levels:
            level_list = [l.strip() for l in levels.split(',') if l.strip()]
            factors.append({
                'name': factor_name,
                'levels': level_list
            })

    st.session_state.factors = factors

    # Generate conditions from factors
    if factors and all(f['levels'] for f in factors):
        from itertools import product

        level_combinations = list(product(*[f['levels'] for f in factors]))
        conditions = []
        for combo in level_combinations:
            condition_name = ' x '.join(combo)
            conditions.append(condition_name)

        st.markdown("**Generated Conditions:**")

        # Display conditions with allocation
        total_n = st.session_state.get('sample_size', 300)
        n_per_condition = total_n // len(conditions)

        for i, cond in enumerate(conditions):
            st.markdown(f"- **{cond}** (n = {n_per_condition})")

        st.session_state.conditions = conditions
        st.session_state.checklist['conditions_defined'] = True
        st.markdown('<div class="success-box">✅ Conditions defined!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['conditions_defined'] = False
        st.markdown('<div class="warning-box">⚠️ Please define all factors and levels</div>', unsafe_allow_html=True)

def step5_define_variables():
    """Step 5: Define dependent variables and scales"""
    st.markdown('<p class="step-header">Step 5: Define Variables & Scales</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Define your dependent variables and any scale items. The simulation will generate realistic responses.
    </div>
    """, unsafe_allow_html=True)

    # Number of DV scales
    num_scales = st.number_input(
        "Number of Measurement Scales *",
        min_value=1,
        max_value=10,
        value=st.session_state.get('num_scales', 2),
        help="How many multi-item scales are you using?"
    )
    st.session_state.num_scales = num_scales

    scales = []
    for i in range(num_scales):
        st.markdown(f"---\n**Scale {i+1}:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            scale_name = st.text_input(
                f"Scale {i+1} Name *",
                value=st.session_state.get(f'scale_{i}_name', ''),
                placeholder="e.g., Psychological Ownership",
                key=f'scale_{i}_name_input'
            )
            st.session_state[f'scale_{i}_name'] = scale_name

        with col2:
            num_items = st.number_input(
                f"Number of Items *",
                min_value=1,
                max_value=20,
                value=st.session_state.get(f'scale_{i}_items', 5),
                key=f'scale_{i}_items_input'
            )
            st.session_state[f'scale_{i}_items'] = num_items

        with col3:
            scale_points = st.selectbox(
                f"Scale Points *",
                options=[5, 6, 7, 9, 10, 11],
                index=1,  # Default to 6-point
                key=f'scale_{i}_points_input'
            )
            st.session_state[f'scale_{i}_points'] = scale_points

        # Reverse-coded items
        reverse_items = st.text_input(
            f"Reverse-coded item numbers (comma-separated, or leave blank)",
            value=st.session_state.get(f'scale_{i}_reverse', ''),
            placeholder="e.g., 3, 5",
            key=f'scale_{i}_reverse_input'
        )
        st.session_state[f'scale_{i}_reverse'] = reverse_items

        if scale_name:
            scales.append({
                'name': scale_name,
                'num_items': num_items,
                'scale_points': scale_points,
                'reverse_items': [int(x.strip()) for x in reverse_items.split(',') if x.strip().isdigit()]
            })

    st.session_state.scales = scales

    # Additional single-item measures
    st.markdown("---\n**Additional Single-Item Measures:**")

    additional_vars = st.text_area(
        "List additional variables (one per line, format: name, min, max)",
        value=st.session_state.get('additional_vars', 'WTP, 0, 10'),
        placeholder="WTP, 0, 10\nSatisfaction, 1, 7",
        height=100
    )
    st.session_state.additional_vars = additional_vars

    # Parse additional variables
    add_vars_list = []
    for line in additional_vars.strip().split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                add_vars_list.append({
                    'name': parts[0],
                    'min': int(parts[1]),
                    'max': int(parts[2])
                })
    st.session_state.additional_vars_list = add_vars_list

    # Attention checks
    st.markdown("---\n**Attention/Manipulation Checks:**")

    has_attention_check = st.checkbox(
        "Survey includes attention/manipulation checks",
        value=st.session_state.get('has_attention_check', True)
    )
    st.session_state.has_attention_check = has_attention_check

    if has_attention_check:
        attention_check_desc = st.text_area(
            "Describe the attention checks (helps simulation generate realistic failure rates)",
            value=st.session_state.get('attention_check_desc', ''),
            placeholder="e.g., 'Was AI mentioned on the page?' - correct answer depends on condition",
            height=80
        )
        st.session_state.attention_check_desc = attention_check_desc

    # Check completion
    if scales and all(s['name'] for s in scales):
        st.session_state.checklist['variables_confirmed'] = True
        st.markdown('<div class="success-box">✅ Variables defined!</div>', unsafe_allow_html=True)
    else:
        st.session_state.checklist['variables_confirmed'] = False
        st.markdown('<div class="warning-box">⚠️ Please define at least one scale</div>', unsafe_allow_html=True)

def step6_simulation_params():
    """Step 6: Set simulation parameters"""
    st.markdown('<p class="step-header">Step 6: Simulation Parameters</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Configure how the simulation should generate data. These settings affect the realism and variance of responses.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demographic Quotas:**")

        gender_quota = st.slider(
            "Male/Female ratio (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.get('gender_quota', 50),
            help="Percentage of male participants (remainder female, with ~2% non-binary)"
        )
        st.session_state.gender_quota = gender_quota

        age_mean = st.number_input(
            "Mean Age",
            min_value=18,
            max_value=65,
            value=st.session_state.get('age_mean', 35)
        )
        st.session_state.age_mean = age_mean

        age_sd = st.number_input(
            "Age Standard Deviation",
            min_value=5,
            max_value=20,
            value=st.session_state.get('age_sd', 12)
        )
        st.session_state.age_sd = age_sd

    with col2:
        st.markdown("**Response Quality:**")

        attention_rate = st.slider(
            "Expected Attention Check Pass Rate (%)",
            min_value=70,
            max_value=100,
            value=st.session_state.get('attention_rate', 95),
            help="What % of participants should pass attention checks?"
        )
        st.session_state.attention_rate = attention_rate

        random_responder_rate = st.slider(
            "Random Responder Rate (%)",
            min_value=0,
            max_value=20,
            value=st.session_state.get('random_responder_rate', 5),
            help="What % of participants respond randomly (adds noise)?"
        )
        st.session_state.random_responder_rate = random_responder_rate

    st.markdown("---\n**Persona Library (Theory-Grounded Heterogeneity):**")
    st.markdown("""
    Following Manning & Horton (2025), simulations use behavioral personas to create realistic variance.
    You can customize the persona weights:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        persona_1_weight = st.slider("Engaged Responder %", 0, 50, 30)
    with col2:
        persona_2_weight = st.slider("Satisficer %", 0, 50, 25)
    with col3:
        persona_3_weight = st.slider("Extreme Responder %", 0, 30, 10)

    # Normalize weights
    total_weight = persona_1_weight + persona_2_weight + persona_3_weight
    remaining = 100 - total_weight

    st.session_state.persona_weights = {
        'engaged': persona_1_weight / 100,
        'satisficer': persona_2_weight / 100,
        'extreme': persona_3_weight / 100,
        'neutral': max(0, remaining) / 100
    }

    if remaining < 0:
        st.warning(f"Weights sum to {total_weight}%, reducing proportionally...")

    st.session_state.checklist['simulation_params'] = True
    st.markdown('<div class="success-box">✅ Simulation parameters set!</div>', unsafe_allow_html=True)

def generate_simulation():
    """Generate the simulation based on all collected parameters"""
    st.markdown('<p class="step-header">Generate Simulation</p>', unsafe_allow_html=True)

    # Check if all checklist items are complete
    if not all(st.session_state.checklist.values()):
        incomplete = [k for k, v in st.session_state.checklist.items() if not v]
        st.error(f"Please complete all checklist items before generating: {', '.join(incomplete)}")
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
                # Create simulation engine
                engine = SimulationEngine(
                    sample_size=st.session_state.sample_size,
                    conditions=st.session_state.conditions,
                    factors=st.session_state.factors,
                    scales=st.session_state.scales,
                    additional_vars=st.session_state.additional_vars_list,
                    demographics={
                        'gender_quota': st.session_state.gender_quota,
                        'age_mean': st.session_state.age_mean,
                        'age_sd': st.session_state.age_sd
                    },
                    attention_rate=st.session_state.attention_rate / 100,
                    random_responder_rate=st.session_state.random_responder_rate / 100,
                    persona_weights=st.session_state.persona_weights
                )

                # Generate data
                df = engine.generate()
                st.session_state.simulation_result = df

                # Create column explainer
                explainer = engine.generate_explainer()

                # Create metadata for audit log
                metadata = {
                    'generation_timestamp': datetime.now().isoformat(),
                    'group_number': st.session_state.group_number,
                    'team_members': st.session_state.team_members,
                    'project_title': st.session_state.project_title,
                    'project_description': st.session_state.project_description,
                    'target_population': st.session_state.target_population,
                    'sample_size': st.session_state.sample_size,
                    'design_type': st.session_state.design_type,
                    'factors': st.session_state.factors,
                    'conditions': st.session_state.conditions,
                    'scales': st.session_state.scales,
                    'additional_variables': st.session_state.additional_vars_list,
                    'simulation_parameters': {
                        'gender_quota': st.session_state.gender_quota,
                        'age_mean': st.session_state.age_mean,
                        'age_sd': st.session_state.age_sd,
                        'attention_rate': st.session_state.attention_rate,
                        'random_responder_rate': st.session_state.random_responder_rate,
                        'persona_weights': st.session_state.persona_weights
                    },
                    'file_hashes': {
                        'qsf': st.session_state.get('qsf_hash', 'N/A'),
                        'screenshots': st.session_state.get('screenshot_hashes', [])
                    }
                }

                # Generate PDF audit log
                pdf_buffer = generate_audit_log_pdf(metadata, df)

                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Add CSV
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr('Simulated.csv', csv_buffer.getvalue())

                    # Add explainer
                    zf.writestr('Column_Explainer.txt', explainer)

                    # Add PDF audit log
                    zf.writestr('Audit_Log.pdf', pdf_buffer.getvalue())

                    # Add metadata JSON (for verification)
                    zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=str))

                zip_buffer.seek(0)

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

                st.markdown(f"**Total rows:** {len(df)} | **Total columns:** {len(df.columns)}")

                # Download button
                st.download_button(
                    label="📥 Download Your Copy (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"Group{st.session_state.group_number}_Simulation_{datetime.now().strftime('%Y%m%d')}.zip",
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
    """Main application flow"""
    # Header
    st.markdown('<p class="main-header">🔬 BDS5010 Simulation Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Behavioral Experiment Data Simulation | Spring 2026</p>', unsafe_allow_html=True)

    # Render sidebar checklist
    render_checklist_sidebar()

    # Progress bar
    render_progress_bar()

    # Create tabs for each step
    tabs = st.tabs([
        "1️⃣ Team Info",
        "2️⃣ Project",
        "3️⃣ Upload Files",
        "4️⃣ Conditions",
        "5️⃣ Variables",
        "6️⃣ Parameters",
        "🚀 Generate"
    ])

    with tabs[0]:
        step1_team_info()

    with tabs[1]:
        step2_project_info()

    with tabs[2]:
        step3_upload_files()

    with tabs[3]:
        step4_define_conditions()

    with tabs[4]:
        step5_define_variables()

    with tabs[5]:
        step6_simulation_params()

    with tabs[6]:
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
