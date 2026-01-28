# simulation_app/utils/instructor_report.py
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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
        lines.append(f"**Generated:** {metadata.get('generation_timestamp', datetime.now().isoformat())}")
        lines.append(f"**Run ID:** {metadata.get('run_id', 'N/A')}")
        lines.append(f"**Mode:** {metadata.get('simulation_mode', 'N/A')}")
        lines.append("")

        if team_info:
            lines.append("## Team")
            lines.append("")
            for k, v in team_info.items():
                if v is None:
                    continue
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if prereg_text:
            lines.append("## Preregistration notes (as provided)")
            lines.append("")
            lines.append("```")
            lines.append(prereg_text.strip())
            lines.append("```")
            lines.append("")

        if self.config.include_design_summary:
            lines.append("## Design summary")
            lines.append("")
            lines.append(f"- Sample size (N): {metadata.get('sample_size', 'N/A')}")
            conditions = metadata.get("conditions", []) or []
            lines.append(f"- Conditions ({len(conditions)}): {', '.join(conditions) if conditions else 'N/A'}")

            factors = metadata.get("factors", []) or []
            if factors:
                lines.append("- Factors:")
                for f in factors:
                    fname = f.get("name", "Factor")
                    levels = f.get("levels", [])
                    lines.append(f"  - {fname}: {', '.join(levels)}")
            else:
                lines.append("- Factors: (none provided; using CONDITION as the only factor)")

            domains = metadata.get("detected_domains", []) or []
            if domains:
                lines.append(f"- Detected topical domains: {', '.join(domains[:8])}")
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
                lines.append(vc.to_frame("n").to_markdown())
                lines.append("")

        if self.config.include_attention_checks:
            lines.append("## Attention checks")
            lines.append("")
            if "AI_Mentioned_Check" in df.columns:
                lines.append("- **AI_Mentioned_Check** distribution:")
                lines.append(df["AI_Mentioned_Check"].value_counts(dropna=False).to_frame("n").to_markdown())
                lines.append("")
            if "Attention_Pass_Rate" in df.columns:
                lines.append("- **Attention_Pass_Rate** summary:")
                lines.append(df["Attention_Pass_Rate"].describe().to_frame().to_markdown())
                lines.append("")

        if self.config.include_exclusions:
            lines.append("## Exclusions")
            lines.append("")
            if "Exclude_Recommended" in df.columns:
                excl = df["Exclude_Recommended"].value_counts(dropna=False).to_frame("n")
                lines.append(excl.to_markdown())
                lines.append("")
            for col in ["Completion_Time_Seconds", "Max_Straight_Line"]:
                if col in df.columns:
                    lines.append(f"### {col} summary")
                    lines.append("")
                    lines.append(df[col].describe().to_frame().to_markdown())
                    lines.append("")

        if self.config.include_persona_summary:
            lines.append("## Persona summary (metadata)")
            lines.append("")
            dist = metadata.get("persona_distribution", {}) or {}
            if dist:
                dist_df = pd.DataFrame(
                    [{"persona": k, "share": float(v)} for k, v in dist.items()]
                ).sort_values("share", ascending=False)
                lines.append(dist_df.to_markdown(index=False))
            else:
                lines.append("_No persona distribution in metadata._")
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
                lines.append(pd.DataFrame(rows).to_markdown(index=False))
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
