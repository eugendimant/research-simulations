# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "2.1.4"  # Added comprehensive persona transparency and scale source tracking

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import pandas as pd

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
