# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "2.1.1"  # Synced with app.py

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
    This is intended to help instructors quickly assess whether a team’s
    simulation is coherent, standardized, and internally consistent.
    """

    def __init__(self, config: Optional[InstructorReportConfig] = None):
        self.config = config or InstructorReportConfig()

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
            lines.append("Personas model different response patterns observed in survey research.")
            lines.append("Each simulated participant was assigned a persona that influenced their response style.")
            lines.append("")

            dist = metadata.get("persona_distribution", {}) or {}
            if dist:
                lines.append("### Persona Breakdown")
                lines.append("")
                lines.append("| Persona Type | Description | Share |")
                lines.append("|--------------|-------------|-------|")

                # Add descriptions for known personas
                persona_descriptions = {
                    "engaged": "Thoughtful, consistent responses with moderate variance",
                    "satisficer": "Tends toward middle options, lower effort",
                    "extreme": "Uses scale endpoints more frequently",
                    "acquiescent": "Agreement bias, tends toward positive responses",
                    "skeptic": "Disagreement bias, tends toward negative responses",
                    "random": "Inconsistent, possibly inattentive responses",
                    "careless": "Fast, pattern-based responses",
                }

                for persona, share in sorted(dist.items(), key=lambda x: -float(x[1])):
                    desc = persona_descriptions.get(persona.lower(), "Standard response pattern")
                    pct = float(share) * 100 if float(share) <= 1 else float(share)
                    lines.append(f"| **{persona.title()}** | {desc} | {pct:.1f}% |")
                lines.append("")

                # Show total participants by persona
                lines.append("### Persona Counts (N)")
                lines.append("")
                n_total = metadata.get("sample_size", len(df))
                lines.append("| Persona | Count |")
                lines.append("|---------|-------|")
                for persona, share in sorted(dist.items(), key=lambda x: -float(x[1])):
                    share_val = float(share) if float(share) <= 1 else float(share) / 100
                    count = int(round(n_total * share_val))
                    lines.append(f"| {persona.title()} | ~{count} |")
                lines.append("")
            else:
                lines.append("_Persona distribution data not available in metadata._")
                lines.append("")

        if self.config.include_variables:
            lines.append("## Variables")
            lines.append("")
            scales = metadata.get("scales", []) or []
            if scales:
                rows = []
                for s in scales:
                    rows.append(
                        {
                            "scale": s.get("name", "Scale"),
                            "num_items": s.get("num_items", s.get("items", "")),
                            "scale_points": s.get("scale_points", ""),
                            "reverse_items": s.get("reverse_items", []),
                        }
                    )
                lines.append(_safe_to_markdown(pd.DataFrame(rows), index=False))
                lines.append("")
            else:
                lines.append("_No scales listed in metadata._")
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
