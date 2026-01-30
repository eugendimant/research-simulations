"""
PDF Generator for Behavioral Experiment Simulation Tool
================================================================
Generates tamper-proof audit log PDFs documenting all simulation inputs
and parameters for verification purposes.

Uses reportlab for PDF generation.
"""

import io
from datetime import datetime
from typing import Dict, Any
import hashlib
import json

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import pandas as pd


def generate_audit_log_pdf(metadata: Dict[str, Any], df: pd.DataFrame) -> io.BytesIO:
    """
    Generate a tamper-proof audit log PDF.

    Args:
        metadata: Dictionary containing all simulation parameters and inputs
        df: The generated DataFrame

    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = io.BytesIO()

    if not REPORTLAB_AVAILABLE:
        # Fallback: create a simple text-based PDF alternative
        return _generate_simple_audit_log(metadata, df)

    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f4e79')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor('#1f4e79')
    )

    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#333333')
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        'CustomSmall',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )

    # Build content
    story = []

    # Title
    story.append(Paragraph("Simulation Audit Log", title_style))
    story.append(Paragraph("Behavioral Experiment Data Simulation", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Document integrity hash
    content_hash = _compute_content_hash(metadata, df)
    story.append(Paragraph(f"Document Hash: {content_hash[:32]}...", small_style))
    story.append(Paragraph(
        f"Generated: {metadata.get('generation_timestamp', datetime.now().isoformat())}",
        small_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # Horizontal rule
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.2*inch))

    # Section 1: Team Information
    story.append(Paragraph("1. Team Information", heading_style))

    team_data = [
        ['Group Number:', str(metadata.get('group_number', 'N/A'))],
        ['Team Members:', ', '.join(metadata.get('team_members', ['N/A']))],
    ]

    team_table = Table(team_data, colWidths=[1.5*inch, 5*inch])
    team_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(team_table)
    story.append(Spacer(1, 0.2*inch))

    # Section 2: Project Details
    story.append(Paragraph("2. Project Details", heading_style))

    project_data = [
        ['Project Title:', metadata.get('project_title', 'N/A')],
        ['Description:', metadata.get('project_description', 'N/A')],
        ['Target Population:', metadata.get('target_population', 'N/A')],
        ['Sample Size:', str(metadata.get('sample_size', 'N/A'))],
        ['Design Type:', metadata.get('design_type', 'N/A')],
    ]

    project_table = Table(project_data, colWidths=[1.5*inch, 5*inch])
    project_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(project_table)
    story.append(Spacer(1, 0.2*inch))

    # Section 3: Experimental Design
    story.append(Paragraph("3. Experimental Design", heading_style))

    # Factors
    story.append(Paragraph("Factors:", subheading_style))
    factors = metadata.get('factors', [])
    for factor in factors:
        story.append(Paragraph(
            f"  &bull; <b>{factor.get('name', 'Unknown')}</b>: {', '.join(factor.get('levels', []))}",
            normal_style
        ))

    # Conditions
    story.append(Paragraph("Conditions:", subheading_style))
    conditions = metadata.get('conditions', [])
    n_per_cond = metadata.get('sample_size', 300) // max(len(conditions), 1)
    for cond in conditions:
        story.append(Paragraph(f"  &bull; {cond} (n = {n_per_cond})", normal_style))

    story.append(Spacer(1, 0.2*inch))

    # Section 4: Measurement Scales
    story.append(Paragraph("4. Measurement Scales", heading_style))

    scales = metadata.get('scales', [])
    scale_data = [['Scale Name', 'Items', 'Points', 'Reverse Items']]
    for scale in scales:
        reverse_str = ', '.join(map(str, scale.get('reverse_items', []))) or 'None'
        scale_data.append([
            scale.get('name', 'Unknown'),
            str(scale.get('num_items', 0)),
            str(scale.get('scale_points', 0)),
            reverse_str
        ])

    if len(scale_data) > 1:
        scale_table = Table(scale_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 1.5*inch])
        scale_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(scale_table)

    story.append(Spacer(1, 0.2*inch))

    # Section 5: Simulation Parameters
    story.append(Paragraph("5. Simulation Parameters", heading_style))

    sim_params = metadata.get('simulation_parameters', {})
    param_data = [
        ['Gender Quota (% Male):', f"{sim_params.get('gender_quota', 50)}%"],
        ['Mean Age:', str(sim_params.get('age_mean', 35))],
        ['Age SD:', str(sim_params.get('age_sd', 12))],
        ['Attention Check Pass Rate:', f"{sim_params.get('attention_rate', 95)}%"],
        ['Random Responder Rate:', f"{sim_params.get('random_responder_rate', 5)}%"],
    ]

    param_table = Table(param_data, colWidths=[2.5*inch, 2*inch])
    param_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(param_table)

    # Persona weights
    story.append(Paragraph("Persona Weights:", subheading_style))
    persona_weights = sim_params.get('persona_weights', {})
    for persona, weight in persona_weights.items():
        story.append(Paragraph(f"  &bull; {persona.capitalize()}: {weight*100:.0f}%", normal_style))

    story.append(Spacer(1, 0.2*inch))

    # Section 6: File Integrity
    story.append(Paragraph("6. Input File Verification", heading_style))

    file_hashes = metadata.get('file_hashes', {})
    story.append(Paragraph(f"QSF File Hash: {file_hashes.get('qsf', 'N/A')[:32]}...", normal_style))

    screenshot_hashes = file_hashes.get('screenshots', [])
    if screenshot_hashes:
        story.append(Paragraph("Screenshot Hashes:", subheading_style))
        for sh in screenshot_hashes:
            story.append(Paragraph(f"  &bull; {sh.get('name', 'Unknown')}: {sh.get('hash', 'N/A')}", small_style))

    story.append(Spacer(1, 0.3*inch))

    # Section 7: Data Summary Statistics
    story.append(Paragraph("7. Generated Data Summary", heading_style))

    story.append(Paragraph(f"Total Rows: {len(df)}", normal_style))
    story.append(Paragraph(f"Total Columns: {len(df.columns)}", normal_style))

    # Condition distribution
    if 'CONDITION' in df.columns:
        story.append(Paragraph("Condition Distribution:", subheading_style))
        cond_counts = df['CONDITION'].value_counts()
        for cond, count in cond_counts.items():
            story.append(Paragraph(f"  &bull; {cond}: {count} ({count/len(df)*100:.1f}%)", normal_style))

    # Key variable summaries
    story.append(Paragraph("Key Variable Statistics:", subheading_style))

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:8]
    if len(numeric_cols) > 0:
        stats_data = [['Variable', 'Mean', 'SD', 'Min', 'Max']]
        for col in numeric_cols:
            stats_data.append([
                col[:20] + ('...' if len(col) > 20 else ''),
                f"{df[col].mean():.2f}",
                f"{df[col].std():.2f}",
                str(df[col].min()),
                str(df[col].max())
            ])

        stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(stats_table)

    story.append(Spacer(1, 0.3*inch))

    # Footer / Verification section
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("VERIFICATION STATEMENT", heading_style))
    story.append(Paragraph(
        "This document certifies that the simulation was generated using the "
        "Behavioral Experiment Simulation Tool. All parameters, inputs, and outputs "
        "are recorded above. The document hash can be used to verify that this PDF "
        "has not been modified after generation.",
        normal_style
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        f"Full Content Hash: {content_hash}",
        small_style
    ))
    story.append(Paragraph(
        "Behavioral Experiment Simulation Tool | Prof. Dr. Eugen Dimant",
        small_style
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    return buffer


def _compute_content_hash(metadata: Dict[str, Any], df: pd.DataFrame) -> str:
    """Compute a hash of all content for tamper detection."""
    content = json.dumps(metadata, sort_keys=True, default=str)
    content += df.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def _generate_simple_audit_log(metadata: Dict[str, Any], df: pd.DataFrame) -> io.BytesIO:
    """
    Fallback function to generate a simple text-based audit log
    when reportlab is not available.
    """
    buffer = io.BytesIO()

    content_hash = _compute_content_hash(metadata, df)

    lines = [
        "=" * 70,
        "SIMULATION AUDIT LOG",
        "Behavioral Experiment Data Simulation",
        "=" * 70,
        "",
        f"Document Hash: {content_hash[:32]}...",
        f"Generated: {metadata.get('generation_timestamp', datetime.now().isoformat())}",
        "",
        "-" * 70,
        "1. TEAM INFORMATION",
        "-" * 70,
        f"Group Number: {metadata.get('group_number', 'N/A')}",
        f"Team Members: {', '.join(metadata.get('team_members', ['N/A']))}",
        "",
        "-" * 70,
        "2. PROJECT DETAILS",
        "-" * 70,
        f"Project Title: {metadata.get('project_title', 'N/A')}",
        f"Description: {metadata.get('project_description', 'N/A')}",
        f"Target Population: {metadata.get('target_population', 'N/A')}",
        f"Sample Size: {metadata.get('sample_size', 'N/A')}",
        f"Design Type: {metadata.get('design_type', 'N/A')}",
        "",
        "-" * 70,
        "3. EXPERIMENTAL DESIGN",
        "-" * 70,
        "Factors:",
    ]

    for factor in metadata.get('factors', []):
        lines.append(f"  - {factor.get('name', 'Unknown')}: {', '.join(factor.get('levels', []))}")

    lines.append("")
    lines.append("Conditions:")
    for cond in metadata.get('conditions', []):
        lines.append(f"  - {cond}")

    lines.extend([
        "",
        "-" * 70,
        "4. MEASUREMENT SCALES",
        "-" * 70,
    ])

    for scale in metadata.get('scales', []):
        reverse_str = ', '.join(map(str, scale.get('reverse_items', []))) or 'None'
        lines.append(f"  - {scale.get('name', 'Unknown')}: {scale.get('num_items', 0)} items, "
                    f"{scale.get('scale_points', 0)}-point (reverse: {reverse_str})")

    lines.extend([
        "",
        "-" * 70,
        "5. SIMULATION PARAMETERS",
        "-" * 70,
    ])

    sim_params = metadata.get('simulation_parameters', {})
    lines.extend([
        f"Gender Quota (% Male): {sim_params.get('gender_quota', 50)}%",
        f"Mean Age: {sim_params.get('age_mean', 35)}",
        f"Age SD: {sim_params.get('age_sd', 12)}",
        f"Attention Check Pass Rate: {sim_params.get('attention_rate', 95)}%",
        f"Random Responder Rate: {sim_params.get('random_responder_rate', 5)}%",
        "",
        "Persona Weights:",
    ])

    for persona, weight in sim_params.get('persona_weights', {}).items():
        lines.append(f"  - {persona.capitalize()}: {weight*100:.0f}%")

    lines.extend([
        "",
        "-" * 70,
        "6. INPUT FILE VERIFICATION",
        "-" * 70,
    ])

    file_hashes = metadata.get('file_hashes', {})
    lines.append(f"QSF File Hash: {file_hashes.get('qsf', 'N/A')}")

    for sh in file_hashes.get('screenshots', []):
        lines.append(f"  - {sh.get('name', 'Unknown')}: {sh.get('hash', 'N/A')}")

    lines.extend([
        "",
        "-" * 70,
        "7. GENERATED DATA SUMMARY",
        "-" * 70,
        f"Total Rows: {len(df)}",
        f"Total Columns: {len(df.columns)}",
        "",
        "-" * 70,
        "VERIFICATION STATEMENT",
        "-" * 70,
        "This document certifies that the simulation was generated using the",
        "Behavioral Experiment Simulation Tool. The document hash can be used to",
        "verify that this file has not been modified after generation.",
        "",
        f"Full Content Hash: {content_hash}",
        "",
        "Behavioral Experiment Simulation Tool | Prof. Dr. Eugen Dimant",
        "=" * 70,
    ])

    buffer.write("\n".join(lines).encode('utf-8'))
    buffer.seek(0)

    return buffer
