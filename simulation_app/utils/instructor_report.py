# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "2.1.6"  # Added HTML report with visualizations and statistical tests

from dataclasses import dataclass
from datetime import datetime
import json
import base64
import io
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports for statistics and visualization
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def _safe_to_markdown(df: pd.DataFrame, **kwargs) -> str:
    """
    Safely convert DataFrame to markdown, falling back to string representation
    if tabulate is not available (required by pandas.to_markdown).
    """
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        return df.to_string(**kwargs)


@dataclass
class InstructorReportConfig:
    include_metadata: bool = True
    include_schema_validation: bool = True
    include_preview_stats: bool = True
    include_design_summary: bool = True
    include_attention_checks: bool = True
    include_exclusions: bool = True
    include_persona_summary: bool = True
    include_variables: bool = True
    include_r_script: bool = True


class InstructorReportGenerator:
    """
    Generates instructor-facing documentation for a simulation run.
    This is intended to help instructors quickly assess whether a team's
    simulation is coherent, standardized, and internally consistent.
    """

    def __init__(self, config: Optional[InstructorReportConfig] = None):
        self.config = config or InstructorReportConfig()

    def _get_persona_impact(self, persona: str) -> str:
        """Get expected impact description for a persona type."""
        impacts = {
            "engaged responder": "Reliable data, realistic variance",
            "engaged": "Reliable data, realistic variance",
            "satisficer": "May inflate midpoint responses",
            "extreme responder": "Increases overall variance",
            "extreme": "Increases overall variance",
            "acquiescent": "May inflate agreement/positive responses",
            "skeptic": "May deflate agreement/positive responses",
            "random": "Adds noise, likely flagged for exclusion",
            "careless": "Straight-line patterns, likely excluded",
            "careful responder": "High quality responses, low exclusion rate",
            "moderate responder": "Reduces variance, avoids extremes",
        }
        return impacts.get(persona, "Standard response patterns")

    def generate_markdown_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = []

        lines.append(f"# Instructor Report: {metadata.get('study_title', 'Untitled Study')}")
        lines.append("")
        lines.append("## Generation Information")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Generated** | {metadata.get('generation_timestamp', datetime.now().isoformat())} |")
        lines.append(f"| **Run ID** | `{metadata.get('run_id', 'N/A')}` |")
        lines.append(f"| **Mode** | {metadata.get('simulation_mode', 'pilot').title()} |")
        lines.append(f"| **Tool Version** | {metadata.get('app_version', 'N/A')} |")
        lines.append(f"| **Random Seed** | {metadata.get('random_seed', 'Auto')} |")
        lines.append("")

        if team_info:
            lines.append("## Team Information")
            lines.append("")
            team_name = team_info.get("team_name", "")
            team_members = team_info.get("team_members", "")
            if team_name:
                lines.append(f"- **Team Name:** {team_name}")
            if team_members:
                lines.append("- **Members:**")
                for member in team_members.strip().split('\n'):
                    if member.strip():
                        lines.append(f"  - {member.strip()}")
            lines.append("")

        # DATA QUALITY SUMMARY (Executive Overview)
        lines.append("## Data Quality Summary")
        lines.append("")
        lines.append("### Quick Assessment")
        lines.append("")

        n_total = len(df)
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
        exclusion_rate = (n_excluded / n_total * 100) if n_total > 0 else 0

        # Calculate balance (coefficient of variation of condition sizes)
        balance_status = "N/A"
        if "CONDITION" in df.columns:
            cond_counts = df["CONDITION"].value_counts()
            if len(cond_counts) > 1:
                cv = cond_counts.std() / cond_counts.mean() * 100 if cond_counts.mean() > 0 else 0
                balance_status = "Balanced" if cv < 10 else ("Slightly unbalanced" if cv < 20 else "Unbalanced")

        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| Sample Size | N = {n_total} | {'Adequate' if n_total >= 30 else 'Small'} |")
        lines.append(f"| Exclusion Rate | {exclusion_rate:.1f}% ({n_excluded}/{n_total}) | {'Normal' if exclusion_rate < 20 else 'High'} |")
        lines.append(f"| Condition Balance | {balance_status} | - |")
        lines.append(f"| Missing Data | {int(df.isna().sum().sum())} cells | {'Clean' if df.isna().sum().sum() == 0 else 'Some missing'} |")
        lines.append("")

        lines.append("### Interpretation Guide")
        lines.append("")
        lines.append("- **Exclusion Rate**: Simulated exclusion rates typically range 5-15%. Higher rates may indicate stricter criteria or more careless responders.")
        lines.append("- **Condition Balance**: Slight imbalances (<10%) are normal and won't affect most analyses.")
        lines.append("- **This is simulated data**: Results demonstrate what your analysis pipeline will produce with realistic-looking data structures.")
        lines.append("")

        # SIMULATION SETTINGS TRANSPARENCY SECTION
        lines.append("## Simulation Settings (Transparency)")
        lines.append("")
        lines.append("These settings were used to generate the simulated data:")
        lines.append("")

        # Demographics settings
        demo = metadata.get("demographics", {})
        lines.append("### Demographics Configuration")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Gender quota (% male) | {demo.get('gender_quota', 50)}% |")
        lines.append(f"| Age mean | {demo.get('age_mean', 35)} |")
        lines.append(f"| Age standard deviation | {demo.get('age_sd', 12)} |")
        lines.append("")

        # Response quality settings
        lines.append("### Response Quality Settings")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Attention check pass rate | {metadata.get('attention_rate', 0.85):.0%} |")
        lines.append(f"| Random responder rate | {metadata.get('random_responder_rate', 0.05):.0%} |")
        lines.append("")

        # Exclusion criteria
        exclusion = metadata.get("exclusion_criteria", {})
        if exclusion:
            lines.append("### Exclusion Criteria")
            lines.append("")
            lines.append("| Criterion | Threshold |")
            lines.append("|-----------|-----------|")
            lines.append(f"| Min completion time | {exclusion.get('completion_time_min_seconds', 60)} seconds |")
            lines.append(f"| Max completion time | {exclusion.get('completion_time_max_seconds', 1800)} seconds |")
            lines.append(f"| Straight-line threshold | {exclusion.get('straight_line_threshold', 10)} items |")
            lines.append(f"| Duplicate IP check | {'Yes' if exclusion.get('duplicate_ip_check', True) else 'No'} |")
            lines.append("")

        if prereg_text:
            lines.append("## Preregistration Notes (as provided)")
            lines.append("")
            lines.append("```")
            lines.append(prereg_text.strip()[:2000])
            if len(prereg_text) > 2000:
                lines.append("... [truncated]")
            lines.append("```")
            lines.append("")

        if self.config.include_design_summary:
            lines.append("## Experimental Design Summary")
            lines.append("")
            lines.append("| Element | Details |")
            lines.append("|---------|---------|")
            lines.append(f"| **Sample Size (N)** | {metadata.get('sample_size', 'N/A')} |")

            conditions = metadata.get("conditions", []) or []
            lines.append(f"| **Conditions** | {len(conditions)}: {', '.join(conditions) if conditions else 'N/A'} |")

            factors = metadata.get("factors", []) or []
            if factors:
                for i, f in enumerate(factors):
                    fname = f.get("name", "Factor")
                    levels = f.get("levels", [])
                    lines.append(f"| **Factor {i+1}** | {fname}: {', '.join(levels)} |")
            else:
                lines.append("| **Factors** | Single factor (Condition) |")

            # Randomization level
            design_review = metadata.get("design_review", {})
            rand_level = design_review.get("randomization_level", "Participant-level")
            lines.append(f"| **Randomization** | {rand_level} |")

            domains = metadata.get("detected_domains", []) or []
            if domains:
                lines.append(f"| **Detected Domains** | {', '.join(domains[:5])} |")
            lines.append("")

        if self.config.include_preview_stats:
            lines.append("## Quick data checks")
            lines.append("")
            lines.append(f"- Rows: {df.shape[0]}")
            lines.append(f"- Columns: {df.shape[1]}")
            lines.append(f"- Missing values (total): {int(df.isna().sum().sum())}")
            lines.append("")

            if "CONDITION" in df.columns:
                vc = df["CONDITION"].value_counts(dropna=False)
                lines.append("### Condition counts")
                lines.append("")
                lines.append(_safe_to_markdown(vc.to_frame("n")))
                lines.append("")

        if self.config.include_attention_checks:
            lines.append("## Attention checks")
            lines.append("")
            if "AI_Mentioned_Check" in df.columns:
                lines.append("- **AI_Mentioned_Check** distribution:")
                lines.append(_safe_to_markdown(df["AI_Mentioned_Check"].value_counts(dropna=False).to_frame("n")))
                lines.append("")
            if "Attention_Pass_Rate" in df.columns:
                lines.append("- **Attention_Pass_Rate** summary:")
                lines.append(_safe_to_markdown(df["Attention_Pass_Rate"].describe().to_frame()))
                lines.append("")

        if self.config.include_exclusions:
            lines.append("## Exclusions")
            lines.append("")
            if "Exclude_Recommended" in df.columns:
                excl = df["Exclude_Recommended"].value_counts(dropna=False).to_frame("n")
                lines.append(_safe_to_markdown(excl))
                lines.append("")
            for col in ["Completion_Time_Seconds", "Max_Straight_Line"]:
                if col in df.columns:
                    lines.append(f"### {col} summary")
                    lines.append("")
                    lines.append(_safe_to_markdown(df[col].describe().to_frame()))
                    lines.append("")

        if self.config.include_persona_summary:
            lines.append("## Persona Distribution (Simulated Response Styles)")
            lines.append("")

            # Comprehensive theory section
            lines.append("### Theoretical Background")
            lines.append("")
            lines.append("Personas in this simulation are based on well-established survey methodology research on response styles and participant behaviors. Real survey data naturally contains participants who respond in systematically different ways - some are highly engaged and thoughtful, while others may satisfice (provide minimally acceptable responses), exhibit response biases, or respond carelessly.")
            lines.append("")
            lines.append("By incorporating these response patterns into simulated data, we create more realistic datasets that mirror what researchers encounter in actual studies. This allows students to practice identifying and handling data quality issues before collecting real data.")
            lines.append("")

            lines.append("### How Personas Were Selected")
            lines.append("")
            lines.append("Personas are automatically assigned to simulated participants based on:")
            lines.append("1. **Study domain detection**: The simulation analyzes your study title and description to identify relevant behavioral domains (e.g., consumer behavior, attitudes, decision-making)")
            lines.append("2. **Weighted random assignment**: Each persona has a base probability, adjusted by study characteristics")
            lines.append("3. **Realistic proportions**: The distribution aims to match what researchers typically observe in online panel data (e.g., ~5-15% satisficers, ~5% careless responders)")
            lines.append("")

            dist = metadata.get("persona_distribution", {}) or {}
            if dist:
                lines.append("### Persona Breakdown for This Simulation")
                lines.append("")
                lines.append("| Persona Type | Description | Behavioral Characteristics | Share |")
                lines.append("|--------------|-------------|---------------------------|-------|")

                # Comprehensive persona descriptions with behavioral characteristics
                persona_info = {
                    "engaged responder": {
                        "desc": "Thoughtful, attentive participant",
                        "chars": "Reads questions carefully, uses full scale range appropriately, consistent with their attitudes"
                    },
                    "engaged": {
                        "desc": "Thoughtful, attentive participant",
                        "chars": "Reads questions carefully, uses full scale range appropriately, consistent with their attitudes"
                    },
                    "satisficer": {
                        "desc": "Minimally effortful responder",
                        "chars": "Gravitates to middle options, may skip reading full questions, faster completion times"
                    },
                    "extreme responder": {
                        "desc": "Uses scale endpoints frequently",
                        "chars": "Strong opinions, uses 1s and 7s (or max/min) more than average, high variance"
                    },
                    "extreme": {
                        "desc": "Uses scale endpoints frequently",
                        "chars": "Strong opinions, uses 1s and 7s (or max/min) more than average, high variance"
                    },
                    "acquiescent": {
                        "desc": "Agreement bias responder",
                        "chars": "Tendency to agree with statements regardless of content, inflated positive responses"
                    },
                    "skeptic": {
                        "desc": "Disagreement bias responder",
                        "chars": "Tendency to disagree or rate negatively, lower mean responses"
                    },
                    "random": {
                        "desc": "Inconsistent, inattentive responder",
                        "chars": "High variance, fails attention checks, no clear pattern"
                    },
                    "careless": {
                        "desc": "Pattern-based responder",
                        "chars": "Straight-lining, very fast completion, likely to be flagged for exclusion"
                    },
                    "careful responder": {
                        "desc": "Highly attentive, methodical",
                        "chars": "Longer completion times, passes all attention checks, low variance within scales"
                    },
                    "moderate responder": {
                        "desc": "Avoids extreme responses",
                        "chars": "Uses middle portion of scale, rarely selects endpoints"
                    },
                }

                for persona, share in sorted(dist.items(), key=lambda x: -float(x[1])):
                    persona_key = persona.lower()
                    info = persona_info.get(persona_key, {"desc": "Standard response pattern", "chars": "Typical survey behavior"})
                    pct = float(share) * 100 if float(share) <= 1 else float(share)
                    lines.append(f"| **{persona.title()}** | {info['desc']} | {info['chars']} | {pct:.1f}% |")
                lines.append("")

                # Show total participants by persona
                lines.append("### Participant Counts by Persona")
                lines.append("")
                n_total = metadata.get("sample_size", len(df))
                lines.append("| Persona | Approximate Count | Expected Impact |")
                lines.append("|---------|-------------------|-----------------|")
                for persona, share in sorted(dist.items(), key=lambda x: -float(x[1])):
                    share_val = float(share) if float(share) <= 1 else float(share) / 100
                    count = int(round(n_total * share_val))
                    # Describe expected impact
                    impact = self._get_persona_impact(persona.lower())
                    lines.append(f"| {persona.title()} | ~{count} | {impact} |")
                lines.append("")

                # Interpretation guidance
                lines.append("### Interpreting Simulated Data with Personas")
                lines.append("")
                lines.append("Understanding the persona distribution helps interpret your simulated results:")
                lines.append("")
                lines.append("1. **Scale means may be slightly inflated or deflated** depending on the balance of acquiescent vs. skeptic personas")
                lines.append("2. **Exclusion rates reflect realistic data cleaning** - participants flagged for exclusion are often those with 'careless' or 'random' personas")
                lines.append("3. **Effect sizes remain interpretable** - condition effects are applied consistently across personas, so treatment differences reflect your specified effect sizes")
                lines.append("4. **Variance patterns mirror real data** - some participants show more extreme responses (higher variance) while others cluster around the mean")
                lines.append("")
                lines.append("**Important**: The persona assigned to each simulated participant is NOT included in the exported data. This is intentional - in real research, you wouldn't know each participant's response style. The personas are used only to generate realistic response patterns.")
                lines.append("")
            else:
                lines.append("_Persona distribution data not available in metadata. This may indicate an older simulation or configuration issue._")
                lines.append("")

        if self.config.include_variables:
            lines.append("## Dependent Variables (Scales)")
            lines.append("")
            lines.append("The following scales/DVs were simulated based on your QSF and configuration:")
            lines.append("")

            scales = metadata.get("scales", []) or []
            if scales:
                lines.append("| Scale Name | Items | Scale Points | Source | Reverse Items |")
                lines.append("|------------|-------|--------------|--------|---------------|")

                for s in scales:
                    name = s.get("name", "Scale")
                    num_items = s.get("num_items", s.get("items", "?"))
                    scale_points = s.get("scale_points", "?")

                    # Determine source of scale points
                    detected = s.get("detected_from_qsf", None)
                    if detected is True:
                        source = "QSF (detected)"
                    elif detected is False:
                        source = "Default"
                    else:
                        source = "Config"

                    reverse = s.get("reverse_items", [])
                    reverse_str = ", ".join(str(r) for r in reverse) if reverse else "None"

                    lines.append(f"| {name} | {num_items} | {scale_points}-point | {source} | {reverse_str} |")

                lines.append("")

                # Add note about scale points
                lines.append("### Scale Points Note")
                lines.append("")
                lines.append("- **QSF (detected)**: Scale points were automatically detected from your Qualtrics survey file")
                lines.append("- **Default**: Scale points defaulted to 7 because they couldn't be detected from QSF")
                lines.append("- **Config**: Scale points were set in the simulation configuration")
                lines.append("")
                lines.append("If the scale points don't match your preregistration, please verify your QSF has the correct response options defined, or manually specify scale points in the tool's configuration.")
                lines.append("")
            else:
                lines.append("_No scales/DVs listed in metadata. This may indicate a configuration issue._")
                lines.append("")

        if self.config.include_schema_validation and schema_validation is not None:
            lines.append("## Schema validation")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(schema_validation, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")

        if self.config.include_r_script:
            lines.append("## R script (basic preparation)")
            lines.append("")
            lines.append("```r")
            lines.append(self._generate_basic_r_script(metadata))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def _generate_basic_r_script(self, metadata: Dict[str, Any]) -> str:
        conditions = metadata.get("conditions", [])
        scales = metadata.get('scales', [])

        def _r_quote(x: str) -> str:
            x = str(x).replace('\\', '\\\\').replace('"', '\\"')
            return f'"{x}"'

        condition_levels = ", ".join([_r_quote(c) for c in conditions])

        r_lines = [
            "# ============================================================",
            f"# Basic R script for: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {metadata.get('run_id', 'N/A')}",
            "# ============================================================",
            "",
            "library(readr)",
            "library(dplyr)",
            "",
            "data <- read_csv('Simulated.csv', show_col_types = FALSE)",
            "",
            "# Set up factors",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({condition_levels}))",
            "data$Gender <- factor(data$Gender, levels = 1:4, labels = c('Male','Female','Non-binary','Prefer not to say'))",
            "",
        ]

        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            num_items = int(s.get("num_items", 5) or 5)
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            items_quoted = ", ".join([f"\"{x}\"" for x in items])
            r_lines.append(f"# Composite for {name}")
            r_lines.append(
                f"data${name}_composite <- rowMeans(data[, c({items_quoted})], na.rm = TRUE)"
            )
            r_lines.append("")

        r_lines.append("# Optional: remove recommended exclusions")
        r_lines.append("data_clean <- data %>% filter(Exclude_Recommended == 0)")
        r_lines.append("")
        r_lines.append("summary(data_clean)")
        return "\n".join(r_lines)


class ComprehensiveInstructorReport:
    """
    Generates a detailed, comprehensive report for instructors ONLY.
    This report includes statistical analyses, visualizations (as text/tables),
    hypothesis testing based on preregistration, and data quality diagnostics.

    This is NOT shared with students - they should practice these analyses themselves.
    """

    def __init__(self):
        pass

    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the comprehensive instructor-only report."""
        lines: List[str] = []

        # Header
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE INSTRUCTOR REPORT (CONFIDENTIAL)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("**This report is for instructor review only. Students receive a simpler version.**")
        lines.append("")

        # Basic info
        lines.append(f"**Study:** {metadata.get('study_title', 'Untitled')}")
        lines.append(f"**Generated:** {metadata.get('generation_timestamp', datetime.now().isoformat())}")
        lines.append(f"**Run ID:** `{metadata.get('run_id', 'N/A')}`")
        lines.append(f"**Mode:** {metadata.get('simulation_mode', 'pilot').title()}")
        lines.append("")

        if team_info:
            lines.append(f"**Team:** {team_info.get('team_name', 'N/A')}")
            members = team_info.get('team_members', '')
            if members:
                lines.append(f"**Members:** {members.replace(chr(10), ', ')}")
            lines.append("")

        # =============================================================
        # SECTION 1: DATA QUALITY SUMMARY
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 1. DATA QUALITY SUMMARY")
        lines.append("-" * 80)
        lines.append("")

        n_total = len(df)
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
        n_clean = n_total - n_excluded
        exclusion_rate = (n_excluded / n_total * 100) if n_total > 0 else 0

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total N | {n_total} |")
        lines.append(f"| Excluded | {n_excluded} ({exclusion_rate:.1f}%) |")
        lines.append(f"| Clean N | {n_clean} |")
        lines.append(f"| Missing cells | {int(df.isna().sum().sum())} |")
        lines.append("")

        # Attention check analysis
        if "Attention_Pass_Rate" in df.columns:
            lines.append("### Attention Check Analysis")
            lines.append("")
            attention_stats = df["Attention_Pass_Rate"].describe()
            lines.append(f"- Mean pass rate: {attention_stats['mean']:.2%}")
            lines.append(f"- Median pass rate: {attention_stats['50%']:.2%}")
            lines.append(f"- Failed all checks (0%): {(df['Attention_Pass_Rate'] == 0).sum()} participants")
            lines.append(f"- Passed all checks (100%): {(df['Attention_Pass_Rate'] == 1).sum()} participants")
            lines.append("")

        # Completion time analysis
        if "Completion_Time_Seconds" in df.columns:
            lines.append("### Completion Time Analysis")
            lines.append("")
            time_stats = df["Completion_Time_Seconds"].describe()
            lines.append(f"- Mean: {time_stats['mean']:.1f} seconds ({time_stats['mean']/60:.1f} min)")
            lines.append(f"- Median: {time_stats['50%']:.1f} seconds")
            lines.append(f"- Min: {time_stats['min']:.1f} seconds")
            lines.append(f"- Max: {time_stats['max']:.1f} seconds")
            lines.append(f"- Suspiciously fast (<60s): {(df['Completion_Time_Seconds'] < 60).sum()}")
            lines.append(f"- Very slow (>30min): {(df['Completion_Time_Seconds'] > 1800).sum()}")
            lines.append("")

        # =============================================================
        # SECTION 2: EXPERIMENTAL DESIGN CHECK
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 2. EXPERIMENTAL DESIGN VERIFICATION")
        lines.append("-" * 80)
        lines.append("")

        conditions = metadata.get("conditions", [])
        factors = metadata.get("factors", [])
        scales = metadata.get("scales", [])

        lines.append(f"**Design type:** {metadata.get('design_type', 'Between-subjects')}")
        lines.append(f"**Number of conditions:** {len(conditions)}")
        lines.append(f"**Number of factors:** {len(factors)}")
        lines.append(f"**Number of DVs:** {len(scales)}")
        lines.append("")

        # Condition distribution
        if "CONDITION" in df.columns:
            lines.append("### Condition Distribution")
            lines.append("")
            cond_counts = df["CONDITION"].value_counts()
            lines.append("| Condition | N | % |")
            lines.append("|-----------|---|---|")
            for cond, count in cond_counts.items():
                pct = count / n_total * 100
                lines.append(f"| {cond} | {count} | {pct:.1f}% |")
            lines.append("")

            # Balance check
            cv = cond_counts.std() / cond_counts.mean() * 100 if cond_counts.mean() > 0 else 0
            if cv < 5:
                lines.append("✅ **Excellent balance** (CV < 5%)")
            elif cv < 10:
                lines.append("✅ **Good balance** (CV < 10%)")
            elif cv < 20:
                lines.append("⚠️ **Slight imbalance** (CV 10-20%)")
            else:
                lines.append("❌ **Notable imbalance** (CV > 20%)")
            lines.append("")

        # Factor structure
        if factors:
            lines.append("### Factor Structure")
            lines.append("")
            for f in factors:
                fname = f.get("name", "Factor")
                levels = f.get("levels", [])
                lines.append(f"- **{fname}**: {', '.join(levels)} ({len(levels)} levels)")
            lines.append("")

        # =============================================================
        # SECTION 3: DV ANALYSIS & STATISTICS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 3. DEPENDENT VARIABLE ANALYSIS")
        lines.append("-" * 80)
        lines.append("")

        # Clean data for analysis
        df_clean = df[df["Exclude_Recommended"] == 0] if "Exclude_Recommended" in df.columns else df

        for scale in scales:
            scale_name = scale.get("name", "Scale")
            num_items = scale.get("num_items", 5)
            scale_points = scale.get("scale_points", 7)

            lines.append(f"### {scale_name}")
            lines.append("")
            lines.append(f"Configuration: {num_items} items, {scale_points}-point scale")
            lines.append("")

            # Find scale columns
            scale_cols = [c for c in df_clean.columns if c.startswith(f"{scale_name.replace(' ', '_')}_") and c[-1].isdigit()]

            if scale_cols:
                # Item-level statistics
                lines.append("#### Item-level Statistics")
                lines.append("")
                lines.append("| Item | Mean | SD | Min | Max |")
                lines.append("|------|------|----|----|-----|")
                for col in scale_cols[:10]:  # Limit to first 10 items
                    if col in df_clean.columns:
                        stats = df_clean[col].describe()
                        lines.append(f"| {col} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.0f} | {stats['max']:.0f} |")
                lines.append("")

                # Composite calculation
                if len(scale_cols) >= 2:
                    composite = df_clean[scale_cols].mean(axis=1)
                    lines.append("#### Composite Score (Mean)")
                    lines.append("")
                    comp_stats = composite.describe()
                    lines.append(f"- Mean: {comp_stats['mean']:.3f}")
                    lines.append(f"- SD: {comp_stats['std']:.3f}")
                    lines.append(f"- Range: [{comp_stats['min']:.2f}, {comp_stats['max']:.2f}]")
                    lines.append("")

                    # By condition
                    if "CONDITION" in df_clean.columns:
                        lines.append("#### By Condition")
                        lines.append("")
                        lines.append("| Condition | N | Mean | SD | 95% CI |")
                        lines.append("|-----------|---|------|----|---------| ")

                        df_clean_copy = df_clean.copy()
                        df_clean_copy["_composite"] = composite

                        for cond in conditions:
                            cond_data = df_clean_copy[df_clean_copy["CONDITION"] == cond]["_composite"]
                            if len(cond_data) > 0:
                                mean = cond_data.mean()
                                sd = cond_data.std()
                                n = len(cond_data)
                                se = sd / (n ** 0.5) if n > 0 else 0
                                ci_low = mean - 1.96 * se
                                ci_high = mean + 1.96 * se
                                lines.append(f"| {cond} | {n} | {mean:.3f} | {sd:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |")
                        lines.append("")

                        # Effect size (Cohen's d for first two conditions)
                        if len(conditions) >= 2:
                            cond1_data = df_clean_copy[df_clean_copy["CONDITION"] == conditions[0]]["_composite"]
                            cond2_data = df_clean_copy[df_clean_copy["CONDITION"] == conditions[1]]["_composite"]
                            if len(cond1_data) > 1 and len(cond2_data) > 1:
                                pooled_std = ((cond1_data.std()**2 + cond2_data.std()**2) / 2) ** 0.5
                                if pooled_std > 0:
                                    cohens_d = (cond1_data.mean() - cond2_data.mean()) / pooled_std
                                    lines.append(f"**Effect size (Cohen's d, {conditions[0]} vs {conditions[1]}):** {cohens_d:.3f}")
                                    if abs(cohens_d) < 0.2:
                                        lines.append("  → Negligible effect")
                                    elif abs(cohens_d) < 0.5:
                                        lines.append("  → Small effect")
                                    elif abs(cohens_d) < 0.8:
                                        lines.append("  → Medium effect")
                                    else:
                                        lines.append("  → Large effect")
                                    lines.append("")

            lines.append("")

        # =============================================================
        # SECTION 4: PREREGISTRATION CHECK
        # =============================================================
        if prereg_text:
            lines.append("-" * 80)
            lines.append("## 4. PREREGISTRATION ALIGNMENT CHECK")
            lines.append("-" * 80)
            lines.append("")

            # Sample size check
            prereg_lower = prereg_text.lower()
            lines.append("### Key Checks")
            lines.append("")

            # Look for sample size mentions
            import re
            size_matches = re.findall(r'n\s*=\s*(\d+)|sample.*?(\d+)|(\d+)\s*participants', prereg_lower)
            if size_matches:
                lines.append(f"**Sample size in preregistration:** Patterns found - check matches actual N={n_total}")

            # Look for scale mentions
            for scale in scales:
                scale_name = scale.get("name", "").lower()
                scale_points = scale.get("scale_points", 7)
                if scale_name in prereg_lower:
                    lines.append(f"**{scale_name}:** Mentioned in preregistration. Simulated with {scale_points}-point scale.")
                    # Check for scale point mentions
                    point_patterns = [f"{scale_points}-point", f"{scale_points} point", f"1-{scale_points}"]
                    if any(p in prereg_lower for p in point_patterns):
                        lines.append(f"  ✅ Scale points match preregistration")
                    else:
                        lines.append(f"  ⚠️ Verify scale points match preregistration")

            lines.append("")
            lines.append("### Preregistration Text (excerpt)")
            lines.append("```")
            lines.append(prereg_text[:1500] + ("..." if len(prereg_text) > 1500 else ""))
            lines.append("```")
            lines.append("")

        # =============================================================
        # SECTION 5: PERSONA DISTRIBUTION ANALYSIS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 5. PERSONA DISTRIBUTION & IMPACT")
        lines.append("-" * 80)
        lines.append("")

        dist = metadata.get("persona_distribution", {}) or {}
        if dist:
            lines.append("| Persona | % | Expected N | Impact on Data |")
            lines.append("|---------|---|------------|----------------|")
            for persona, share in sorted(dist.items(), key=lambda x: -float(x[1])):
                share_val = float(share) if float(share) <= 1 else float(share) / 100
                pct = share_val * 100
                count = int(round(n_total * share_val))
                impact = self._get_detailed_impact(persona.lower())
                lines.append(f"| {persona.title()} | {pct:.1f}% | ~{count} | {impact} |")
            lines.append("")

            # Estimate impact on results
            lines.append("### Estimated Impact on Results")
            lines.append("")
            acquiescent_share = float(dist.get("acquiescent", 0))
            skeptic_share = float(dist.get("skeptic", 0))
            if acquiescent_share > 0.1:
                lines.append(f"⚠️ High acquiescence ({acquiescent_share:.0%}) may inflate positive responses")
            if skeptic_share > 0.1:
                lines.append(f"⚠️ High skepticism ({skeptic_share:.0%}) may deflate responses")
            careless_share = float(dist.get("careless", 0)) + float(dist.get("random", 0))
            if careless_share > 0.1:
                lines.append(f"⚠️ Notable careless/random ({careless_share:.0%}) - verify exclusion criteria are working")
            lines.append("")

        # =============================================================
        # SECTION 6: RECOMMENDATIONS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 6. INSTRUCTOR RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("")

        lines.append("### For Student Evaluation")
        lines.append("")
        lines.append("1. **Data cleaning**: Have students identify and justify exclusions")
        lines.append("2. **Descriptive statistics**: Verify they compute means/SDs by condition")
        lines.append("3. **Effect sizes**: Check if they interpret Cohen's d correctly")
        lines.append("4. **Scale reliability**: If they compute Cronbach's alpha, it should be reasonable")
        lines.append("5. **Visualization**: Check for appropriate choice of plots")
        lines.append("")

        lines.append("### Things to Watch For")
        lines.append("")
        if exclusion_rate > 15:
            lines.append(f"- Exclusion rate is {exclusion_rate:.1f}% - ask students to justify their criteria")
        if n_clean < 30:
            lines.append(f"- Clean N is only {n_clean} - discuss power implications")
        if len(conditions) > 4:
            lines.append(f"- {len(conditions)} conditions may be complex for analysis")
        lines.append("")

        # Footer
        lines.append("-" * 80)
        lines.append("END OF COMPREHENSIVE INSTRUCTOR REPORT")
        lines.append("-" * 80)

        return "\n".join(lines)

    def _get_detailed_impact(self, persona: str) -> str:
        """Get detailed impact description for instructor understanding."""
        impacts = {
            "engaged responder": "High quality data, typical variance patterns",
            "engaged": "High quality data, typical variance patterns",
            "satisficer": "Central tendency bias, may reduce effect detection",
            "extreme responder": "Inflated variance, potential outlier effects",
            "extreme": "Inflated variance, potential outlier effects",
            "acquiescent": "Positively skewed responses, inflated agreement",
            "skeptic": "Negatively skewed responses, deflated agreement",
            "random": "Noise injection, should be caught by exclusion criteria",
            "careless": "Straight-lining patterns, should trigger exclusion",
            "careful responder": "Consistent, low-variance responses",
            "moderate responder": "Restricted range, may reduce variance",
        }
        return impacts.get(persona, "Standard patterns")

    def _run_statistical_tests(
        self,
        df: pd.DataFrame,
        dv_column: str,
        condition_column: str = "CONDITION",
    ) -> Dict[str, Any]:
        """Run comprehensive statistical tests on the data."""
        results = {}

        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for statistical tests"}

        # Get unique conditions
        conditions = df[condition_column].unique().tolist()

        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for comparison"}

        # Get data by condition
        groups = {cond: df[df[condition_column] == cond][dv_column].dropna() for cond in conditions}

        # Basic descriptive stats
        results["descriptives"] = {}
        for cond, data in groups.items():
            results["descriptives"][cond] = {
                "n": len(data),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "median": float(data.median()),
                "se": float(data.std() / np.sqrt(len(data))) if len(data) > 0 else 0,
            }

        # Two-group comparisons
        if len(conditions) == 2:
            g1, g2 = groups[conditions[0]], groups[conditions[1]]

            # Independent samples t-test
            t_stat, t_p = scipy_stats.ttest_ind(g1, g2, equal_var=True)
            results["t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p),
                "significant": t_p < 0.05,
            }

            # Welch's t-test (unequal variances)
            welch_t, welch_p = scipy_stats.ttest_ind(g1, g2, equal_var=False)
            results["welch_t_test"] = {
                "statistic": float(welch_t),
                "p_value": float(welch_p),
                "significant": welch_p < 0.05,
            }

            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = scipy_stats.mannwhitneyu(g1, g2, alternative='two-sided')
            results["mann_whitney"] = {
                "statistic": float(u_stat),
                "p_value": float(u_p),
                "significant": u_p < 0.05,
            }

            # Cohen's d effect size
            pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
            results["cohens_d"] = {
                "value": float(cohens_d),
                "interpretation": self._interpret_cohens_d(cohens_d),
            }

        # Multi-group comparisons (3+ conditions)
        if len(conditions) >= 2:
            # One-way ANOVA
            f_stat, anova_p = scipy_stats.f_oneway(*[groups[c] for c in conditions])
            results["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(anova_p),
                "significant": anova_p < 0.05,
            }

            # Kruskal-Wallis (non-parametric)
            h_stat, kw_p = scipy_stats.kruskal(*[groups[c] for c in conditions])
            results["kruskal_wallis"] = {
                "h_statistic": float(h_stat),
                "p_value": float(kw_p),
                "significant": kw_p < 0.05,
            }

            # Eta-squared effect size for ANOVA
            grand_mean = df[dv_column].mean()
            ss_between = sum(len(groups[c]) * (groups[c].mean() - grand_mean)**2 for c in conditions)
            ss_total = sum((df[dv_column] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            results["eta_squared"] = {
                "value": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared),
            }

        # Levene's test for homogeneity of variances
        levene_stat, levene_p = scipy_stats.levene(*[groups[c] for c in conditions])
        results["levene_test"] = {
            "statistic": float(levene_stat),
            "p_value": float(levene_p),
            "homogeneous": levene_p > 0.05,
        }

        # Shapiro-Wilk normality test (on pooled data, limited to 5000)
        pooled = df[dv_column].dropna()
        if len(pooled) <= 5000:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(pooled)
            results["shapiro_wilk"] = {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "normal": shapiro_p > 0.05,
            }

        return results

    def _run_regression_analysis(
        self,
        df: pd.DataFrame,
        dv_column: str,
        condition_column: str = "CONDITION",
    ) -> Dict[str, Any]:
        """Run simple regression analysis with condition as predictor."""
        results = {}

        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        try:
            # Create dummy variables for conditions
            conditions = df[condition_column].unique().tolist()
            if len(conditions) < 2:
                return {"error": "Need at least 2 conditions"}

            # Reference category is first condition
            reference = conditions[0]
            X = pd.get_dummies(df[condition_column], drop_first=True)
            y = df[dv_column].values

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X.values])

            # OLS regression: beta = (X'X)^-1 X'y
            XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
            beta = XtX_inv @ X_with_const.T @ y

            # Predictions and residuals
            y_pred = X_with_const @ beta
            residuals = y - y_pred

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Standard errors
            n = len(y)
            k = X_with_const.shape[1]
            mse = ss_res / (n - k)
            se = np.sqrt(np.diag(XtX_inv) * mse)

            # t-statistics and p-values
            t_stats = beta / se
            p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), n - k))

            results["coefficients"] = {
                "intercept": {
                    "estimate": float(beta[0]),
                    "std_error": float(se[0]),
                    "t_stat": float(t_stats[0]),
                    "p_value": float(p_values[0]),
                }
            }

            # Condition coefficients
            predictor_names = X.columns.tolist()
            for i, name in enumerate(predictor_names):
                results["coefficients"][name] = {
                    "estimate": float(beta[i + 1]),
                    "std_error": float(se[i + 1]),
                    "t_stat": float(t_stats[i + 1]),
                    "p_value": float(p_values[i + 1]),
                    "significant": p_values[i + 1] < 0.05,
                }

            results["model_fit"] = {
                "r_squared": float(r_squared),
                "adj_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k - 1)),
                "n": n,
                "df_residual": n - k,
            }

            # F-test for overall model
            if k > 1:
                f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k)) if r_squared < 1 else float('inf')
                f_p = 1 - scipy_stats.f.cdf(f_stat, k - 1, n - k)
                results["f_test"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(f_p),
                    "significant": f_p < 0.05,
                }

        except Exception as e:
            results["error"] = str(e)

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"

    def _create_bar_chart(
        self,
        data: Dict[str, Tuple[float, float]],
        title: str,
        ylabel: str,
    ) -> Optional[str]:
        """Create a bar chart with error bars and return as base64-encoded PNG."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            conditions = list(data.keys())
            means = [data[c][0] for c in conditions]
            errors = [data[c][1] for c in conditions]

            colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
            bars = ax.bar(conditions, means, yerr=errors, capsize=5,
                         color=colors[:len(conditions)], edgecolor='black', alpha=0.8)

            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, mean, error in zip(bars, means, errors):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.05,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception:
            return None

    def _create_distribution_plot(
        self,
        df: pd.DataFrame,
        column: str,
        condition_column: str = "CONDITION",
        title: str = "Distribution by Condition",
    ) -> Optional[str]:
        """Create a distribution plot (box + strip) and return as base64 PNG."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            conditions = df[condition_column].unique().tolist()
            colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

            # Create box plots
            positions = range(len(conditions))
            box_data = [df[df[condition_column] == c][column].dropna() for c in conditions]

            bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.5)

            for patch, color in zip(bp['boxes'], colors[:len(conditions)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_ylabel(column, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception:
            return None

    def generate_html_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a comprehensive HTML report with visualizations and statistical tests."""

        # CSS styles for the report
        css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
            .report-container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }
            h3 { color: #7f8c8d; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .stat-box { background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #3498db; }
            .warning-box { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }
            .success-box { background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745; }
            .error-box { background: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #dc3545; }
            .chart-container { text-align: center; margin: 20px 0; }
            .chart-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
            .confidential { background: #dc3545; color: white; padding: 5px 15px; border-radius: 3px; display: inline-block; margin-bottom: 20px; }
            .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .metric-value { font-size: 2em; font-weight: bold; }
            .metric-label { font-size: 0.9em; opacity: 0.9; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
            .sig { color: #28a745; font-weight: bold; }
            .nonsig { color: #6c757d; }
        </style>
        """

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Instructor Report: {metadata.get('study_title', 'Study')}</title>",
            css,
            "</head>",
            "<body>",
            "<div class='report-container'>",
        ]

        # Header
        html_parts.append("<span class='confidential'>CONFIDENTIAL - INSTRUCTOR ONLY</span>")
        html_parts.append(f"<h1>Comprehensive Statistical Report</h1>")
        html_parts.append(f"<p><strong>Study:</strong> {metadata.get('study_title', 'Untitled')}</p>")
        html_parts.append(f"<p><strong>Generated:</strong> {metadata.get('generation_timestamp', datetime.now().isoformat())}</p>")
        html_parts.append(f"<p><strong>Run ID:</strong> <code>{metadata.get('run_id', 'N/A')}</code></p>")

        if team_info:
            html_parts.append(f"<p><strong>Team:</strong> {team_info.get('team_name', 'N/A')}</p>")

        # Summary metrics
        n_total = len(df)
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
        n_clean = n_total - n_excluded
        exclusion_rate = (n_excluded / n_total * 100) if n_total > 0 else 0
        conditions = metadata.get("conditions", [])

        html_parts.append("<h2>1. Sample Overview</h2>")
        html_parts.append("<div class='metric-grid'>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{n_total}</div><div class='metric-label'>Total N</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{n_clean}</div><div class='metric-label'>Clean N</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{exclusion_rate:.1f}%</div><div class='metric-label'>Exclusion Rate</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{len(conditions)}</div><div class='metric-label'>Conditions</div></div>")
        html_parts.append("</div>")

        # Condition distribution
        if "CONDITION" in df.columns:
            html_parts.append("<h3>Condition Distribution</h3>")
            html_parts.append("<table><tr><th>Condition</th><th>N</th><th>%</th></tr>")
            cond_counts = df["CONDITION"].value_counts()
            for cond, count in cond_counts.items():
                pct = count / n_total * 100
                html_parts.append(f"<tr><td>{cond}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
            html_parts.append("</table>")

        # Clean data for analysis
        df_clean = df[df["Exclude_Recommended"] == 0] if "Exclude_Recommended" in df.columns else df

        # DV Analysis with statistical tests
        scales = metadata.get("scales", [])
        html_parts.append("<h2>2. Statistical Analysis by DV</h2>")

        for scale in scales:
            scale_name = scale.get("name", "Scale")
            scale_cols = [c for c in df_clean.columns if c.startswith(f"{scale_name.replace(' ', '_')}_") and c[-1].isdigit()]

            if not scale_cols:
                continue

            html_parts.append(f"<h3>{scale_name}</h3>")

            # Calculate composite
            if len(scale_cols) >= 1:
                composite = df_clean[scale_cols].mean(axis=1)
                df_analysis = df_clean.copy()
                df_analysis["_composite"] = composite

                # Descriptive stats table
                html_parts.append("<h4>Descriptive Statistics</h4>")
                html_parts.append("<table><tr><th>Condition</th><th>N</th><th>Mean</th><th>SD</th><th>95% CI</th></tr>")

                chart_data = {}
                for cond in conditions:
                    cond_data = df_analysis[df_analysis["CONDITION"] == cond]["_composite"]
                    if len(cond_data) > 0:
                        mean = cond_data.mean()
                        sd = cond_data.std()
                        n = len(cond_data)
                        se = sd / np.sqrt(n) if n > 0 else 0
                        ci_low = mean - 1.96 * se
                        ci_high = mean + 1.96 * se
                        html_parts.append(f"<tr><td>{cond}</td><td>{n}</td><td>{mean:.3f}</td><td>{sd:.3f}</td><td>[{ci_low:.3f}, {ci_high:.3f}]</td></tr>")
                        chart_data[cond] = (mean, 1.96 * se)

                html_parts.append("</table>")

                # Create visualization
                chart_img = self._create_bar_chart(
                    chart_data,
                    f"{scale_name}: Means with 95% CI",
                    "Mean Score"
                )
                if chart_img:
                    html_parts.append("<div class='chart-container'>")
                    html_parts.append(f"<img src='data:image/png;base64,{chart_img}' alt='Bar chart'>")
                    html_parts.append("</div>")

                # Distribution plot
                dist_img = self._create_distribution_plot(
                    df_analysis, "_composite", "CONDITION",
                    f"{scale_name}: Distribution by Condition"
                )
                if dist_img:
                    html_parts.append("<div class='chart-container'>")
                    html_parts.append(f"<img src='data:image/png;base64,{dist_img}' alt='Distribution'>")
                    html_parts.append("</div>")

                # Statistical tests
                if len(conditions) >= 2 and "CONDITION" in df_analysis.columns:
                    stats_results = self._run_statistical_tests(df_analysis, "_composite", "CONDITION")

                    html_parts.append("<h4>Statistical Tests</h4>")

                    # Two-group tests
                    if "t_test" in stats_results:
                        t = stats_results["t_test"]
                        sig_class = "sig" if t["significant"] else "nonsig"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Independent Samples t-test:</strong> t = {t['statistic']:.3f}, ")
                        html_parts.append(f"<span class='{sig_class}'>p = {t['p_value']:.4f}</span>")
                        html_parts.append("</div>")

                    if "welch_t_test" in stats_results:
                        w = stats_results["welch_t_test"]
                        sig_class = "sig" if w["significant"] else "nonsig"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Welch's t-test:</strong> t = {w['statistic']:.3f}, ")
                        html_parts.append(f"<span class='{sig_class}'>p = {w['p_value']:.4f}</span>")
                        html_parts.append("</div>")

                    if "mann_whitney" in stats_results:
                        mw = stats_results["mann_whitney"]
                        sig_class = "sig" if mw["significant"] else "nonsig"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Mann-Whitney U test:</strong> U = {mw['statistic']:.1f}, ")
                        html_parts.append(f"<span class='{sig_class}'>p = {mw['p_value']:.4f}</span>")
                        html_parts.append("</div>")

                    if "cohens_d" in stats_results:
                        d = stats_results["cohens_d"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Cohen's d:</strong> {d['value']:.3f} ({d['interpretation']} effect)")
                        html_parts.append("</div>")

                    # ANOVA for any number of conditions
                    if "anova" in stats_results:
                        a = stats_results["anova"]
                        sig_class = "sig" if a["significant"] else "nonsig"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>One-way ANOVA:</strong> F = {a['f_statistic']:.3f}, ")
                        html_parts.append(f"<span class='{sig_class}'>p = {a['p_value']:.4f}</span>")
                        html_parts.append("</div>")

                    if "kruskal_wallis" in stats_results:
                        kw = stats_results["kruskal_wallis"]
                        sig_class = "sig" if kw["significant"] else "nonsig"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Kruskal-Wallis test:</strong> H = {kw['h_statistic']:.3f}, ")
                        html_parts.append(f"<span class='{sig_class}'>p = {kw['p_value']:.4f}</span>")
                        html_parts.append("</div>")

                    if "eta_squared" in stats_results:
                        e = stats_results["eta_squared"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>η² (eta-squared):</strong> {e['value']:.4f} ({e['interpretation']} effect)")
                        html_parts.append("</div>")

                    # Assumption checks
                    html_parts.append("<h4>Assumption Checks</h4>")

                    if "levene_test" in stats_results:
                        lev = stats_results["levene_test"]
                        if lev["homogeneous"]:
                            html_parts.append("<div class='success-box'>")
                            html_parts.append(f"<strong>Levene's test:</strong> Variances are homogeneous (p = {lev['p_value']:.4f})")
                        else:
                            html_parts.append("<div class='warning-box'>")
                            html_parts.append(f"<strong>Levene's test:</strong> Variances may be heterogeneous (p = {lev['p_value']:.4f}). Consider Welch's t-test.")
                        html_parts.append("</div>")

                    if "shapiro_wilk" in stats_results:
                        sw = stats_results["shapiro_wilk"]
                        if sw["normal"]:
                            html_parts.append("<div class='success-box'>")
                            html_parts.append(f"<strong>Shapiro-Wilk test:</strong> Data appears normally distributed (p = {sw['p_value']:.4f})")
                        else:
                            html_parts.append("<div class='warning-box'>")
                            html_parts.append(f"<strong>Shapiro-Wilk test:</strong> Data may not be normally distributed (p = {sw['p_value']:.4f}). Consider non-parametric tests.")
                        html_parts.append("</div>")

                    # Regression analysis
                    html_parts.append("<h4>Regression Analysis</h4>")
                    reg_results = self._run_regression_analysis(df_analysis, "_composite", "CONDITION")

                    if "error" not in reg_results:
                        html_parts.append("<div class='stat-box'>")
                        if "model_fit" in reg_results:
                            fit = reg_results["model_fit"]
                            html_parts.append(f"<strong>Model Fit:</strong> R² = {fit['r_squared']:.4f}, Adj. R² = {fit['adj_r_squared']:.4f}<br>")

                        if "f_test" in reg_results:
                            f = reg_results["f_test"]
                            sig_class = "sig" if f["significant"] else "nonsig"
                            html_parts.append(f"<strong>F-test:</strong> F = {f['f_statistic']:.3f}, <span class='{sig_class}'>p = {f['p_value']:.4f}</span><br>")

                        # Coefficients table
                        if "coefficients" in reg_results:
                            html_parts.append("<br><strong>Coefficients:</strong>")
                            html_parts.append("<table><tr><th>Predictor</th><th>B</th><th>SE</th><th>t</th><th>p</th></tr>")
                            for pred, coef in reg_results["coefficients"].items():
                                sig_class = "sig" if coef.get("significant", coef["p_value"] < 0.05) else "nonsig"
                                html_parts.append(f"<tr><td>{pred}</td><td>{coef['estimate']:.3f}</td><td>{coef['std_error']:.3f}</td><td>{coef['t_stat']:.3f}</td><td class='{sig_class}'>{coef['p_value']:.4f}</td></tr>")
                            html_parts.append("</table>")
                        html_parts.append("</div>")
                    else:
                        html_parts.append(f"<div class='warning-box'>Regression analysis unavailable: {reg_results['error']}</div>")

        # Chi-squared test for categorical associations
        if "CONDITION" in df_clean.columns and "Gender" in df_clean.columns:
            html_parts.append("<h2>3. Categorical Analysis</h2>")
            html_parts.append("<h3>Condition × Gender</h3>")

            try:
                contingency = pd.crosstab(df_clean["CONDITION"], df_clean["Gender"])
                chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency)

                sig_class = "sig" if p < 0.05 else "nonsig"
                html_parts.append("<div class='stat-box'>")
                html_parts.append(f"<strong>Chi-squared test:</strong> χ² = {chi2:.3f}, df = {dof}, ")
                html_parts.append(f"<span class='{sig_class}'>p = {p:.4f}</span>")
                if p >= 0.05:
                    html_parts.append("<br><em>No significant association between condition and gender (randomization appears successful)</em>")
                html_parts.append("</div>")

                # Contingency table
                html_parts.append("<table><tr><th>Condition</th>")
                for col in contingency.columns:
                    html_parts.append(f"<th>{col}</th>")
                html_parts.append("</tr>")
                for idx, row in contingency.iterrows():
                    html_parts.append(f"<tr><td>{idx}</td>")
                    for val in row:
                        html_parts.append(f"<td>{val}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</table>")
            except Exception:
                pass

        # Footer
        html_parts.append("<h2>Notes for Instructors</h2>")
        html_parts.append("<div class='warning-box'>")
        html_parts.append("<strong>This is simulated data.</strong> Results demonstrate what the analysis pipeline will produce. ")
        html_parts.append("Students should practice these analyses independently and may get similar (but not identical) results due to random variation.")
        html_parts.append("</div>")

        html_parts.append(f"<p style='color:#999;font-size:0.9em;margin-top:30px;'>Generated by Behavioral Experiment Simulation Tool v{__version__}</p>")
        html_parts.append("</div></body></html>")

        return "\n".join(html_parts)
