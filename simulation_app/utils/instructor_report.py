# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "2.1.12"  # Guaranteed visualizations, chart interpretations, executive summary

from dataclasses import dataclass
from datetime import datetime
import json
import base64
import io
import warnings
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try multiple import strategies for scipy
SCIPY_AVAILABLE = False
scipy_stats = None

try:
    from scipy import stats as _scipy_stats
    scipy_stats = _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    pass

if not SCIPY_AVAILABLE:
    try:
        import scipy.stats as _scipy_stats
        scipy_stats = _scipy_stats
        SCIPY_AVAILABLE = True
    except ImportError:
        pass

# Matplotlib imports
MATPLOTLIB_AVAILABLE = False
plt = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as _plt
    plt = _plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

# Import SVG chart generators (guaranteed fallback - no external dependencies)
try:
    from . import svg_charts
    SVG_CHARTS_AVAILABLE = True
except ImportError:
    try:
        import svg_charts
        SVG_CHARTS_AVAILABLE = True
    except ImportError:
        SVG_CHARTS_AVAILABLE = False


def _clean_condition_name(condition: str) -> str:
    """Remove common suffixes and clean up condition names for report display."""
    if not condition:
        return condition
    cleaned = re.sub(r'\s*\(new\)', '', str(condition), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(copy\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*- copy\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*_\d+$', '', cleaned)
    cleaned = re.sub(r'\s*\(\d+\)\s*$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


# ============================================================================
# NUMPY-BASED STATISTICAL FUNCTIONS (fallbacks when scipy unavailable)
# ============================================================================

def _numpy_ttest_ind(group1: np.ndarray, group2: np.ndarray, equal_var: bool = True) -> Tuple[float, float]:
    """Independent samples t-test using numpy only."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if equal_var:
        # Pooled variance
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        se = sp * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        se = np.sqrt(v1/n1 + v2/n2)
        df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    t_stat = (m1 - m2) / se if se > 0 else 0

    # Approximate p-value using normal distribution for large samples
    # For small samples, this is an approximation
    if df > 30:
        # Use normal approximation
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        # Use t-distribution approximation
        p_value = 2 * _t_sf(abs(t_stat), df)

    return float(t_stat), float(p_value)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_sf(t: float, df: float) -> float:
    """Survival function (1-CDF) for t-distribution, approximated."""
    # Use normal approximation for large df
    if df > 100:
        return 1 - _normal_cdf(t)
    # For smaller df, use a rough approximation
    x = df / (df + t**2)
    # Beta function approximation
    return 0.5 * _betainc(df/2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Incomplete beta function approximation (very rough)."""
    # For our purposes, a simple approximation
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use a simple numerical integration
    steps = 100
    dx = x / steps
    result = 0
    for i in range(steps):
        xi = (i + 0.5) * dx
        result += (xi ** (a-1)) * ((1-xi) ** (b-1)) * dx
    # Normalize (approximate)
    norm = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return min(1.0, result / norm) if norm > 0 else 0.5


def _numpy_f_oneway(*groups) -> Tuple[float, float]:
    """One-way ANOVA using numpy only."""
    groups = [np.asarray(g) for g in groups]
    k = len(groups)  # number of groups
    n_total = sum(len(g) for g in groups)

    # Grand mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # Within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1

    # F statistic
    f_stat = ms_between / ms_within if ms_within > 0 else 0

    # P-value approximation (using chi-square approximation for large samples)
    # This is a rough approximation
    if df_within > 30:
        p_value = 1 - _chi2_cdf(f_stat * df_between, df_between)
    else:
        p_value = _f_sf(f_stat, df_between, df_within)

    return float(f_stat), float(p_value)


def _chi2_cdf(x: float, df: float) -> float:
    """Chi-square CDF approximation."""
    if x <= 0:
        return 0.0
    # Wilson-Hilferty approximation
    if df > 0:
        z = (x/df)**(1/3) - (1 - 2/(9*df))
        z /= math.sqrt(2/(9*df))
        return _normal_cdf(z)
    return 0.5


def _f_sf(f: float, df1: float, df2: float) -> float:
    """F-distribution survival function approximation."""
    if f <= 0:
        return 1.0
    # Use normal approximation for large df
    if df1 > 30 and df2 > 30:
        z = (f**(1/3) * (1 - 2/(9*df2)) - (1 - 2/(9*df1))) / math.sqrt(2/(9*df1) + f**(2/3) * 2/(9*df2))
        return 1 - _normal_cdf(z)
    # Rough approximation
    return min(1.0, max(0.0, 1 - _chi2_cdf(f * df1, df1)))


def _numpy_mannwhitneyu(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney U test using numpy only."""
    n1, n2 = len(group1), len(group2)

    # Combine and rank
    combined = np.concatenate([group1, group2])
    ranks = np.argsort(np.argsort(combined)) + 1

    # Sum of ranks for group 1
    r1 = np.sum(ranks[:n1])

    # U statistics
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Normal approximation for large samples
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mu) / sigma if sigma > 0 else 0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return float(u_stat), float(p_value)


def _numpy_levene(*groups) -> Tuple[float, float]:
    """Levene's test for homogeneity of variances using numpy."""
    groups = [np.asarray(g) for g in groups]
    k = len(groups)

    # Use absolute deviations from median (Brown-Forsythe variant)
    z_groups = [np.abs(g - np.median(g)) for g in groups]

    # Apply one-way ANOVA on the transformed data
    return _numpy_f_oneway(*z_groups)


def _numpy_shapiro(data: np.ndarray) -> Tuple[float, float]:
    """Simplified normality test using numpy (skewness/kurtosis based)."""
    n = len(data)
    if n < 3:
        return 1.0, 1.0

    # Calculate skewness and kurtosis
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 1.0, 1.0

    skew = np.mean(((data - mean) / std) ** 3)
    kurt = np.mean(((data - mean) / std) ** 4) - 3

    # D'Agostino-Pearson test approximation
    # Combine skewness and kurtosis into a test statistic
    z_skew = skew * math.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
    z_kurt = kurt / math.sqrt(24 * n * (n - 2) * (n - 3) / ((n + 1)**2 * (n + 3) * (n + 5)))

    k2 = z_skew**2 + z_kurt**2
    p_value = 1 - _chi2_cdf(k2, 2)

    # Return a pseudo W statistic and p-value
    w_stat = 1 - min(abs(skew), 2) / 4  # Rough approximation

    return float(w_stat), float(p_value)

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
        """Run comprehensive statistical tests on the data.

        Uses scipy if available, falls back to numpy implementations otherwise.
        """
        results = {}
        results["scipy_used"] = SCIPY_AVAILABLE

        # Get unique conditions
        conditions = df[condition_column].unique().tolist()

        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for comparison"}

        # Get data by condition
        groups = {cond: df[df[condition_column] == cond][dv_column].dropna().values for cond in conditions}

        # Ensure all groups have data
        valid_groups = {c: g for c, g in groups.items() if len(g) >= 2}
        if len(valid_groups) < 2:
            return {"error": "Need at least 2 groups with 2+ observations each"}

        conditions = list(valid_groups.keys())
        groups = valid_groups

        # Basic descriptive stats
        results["descriptives"] = {}
        for cond, data in groups.items():
            clean_cond = _clean_condition_name(cond)
            results["descriptives"][clean_cond] = {
                "n": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
                "median": float(np.median(data)),
                "se": float(np.std(data, ddof=1) / np.sqrt(len(data))) if len(data) > 0 else 0,
            }

        # Define test functions (scipy or numpy fallback)
        if SCIPY_AVAILABLE and scipy_stats is not None:
            ttest_func = lambda g1, g2, eq_var: scipy_stats.ttest_ind(g1, g2, equal_var=eq_var)
            anova_func = scipy_stats.f_oneway
            mannwhitney_func = lambda g1, g2: scipy_stats.mannwhitneyu(g1, g2, alternative='two-sided')
            levene_func = scipy_stats.levene
            shapiro_func = scipy_stats.shapiro
            kruskal_func = scipy_stats.kruskal
        else:
            ttest_func = lambda g1, g2, eq_var: _numpy_ttest_ind(g1, g2, equal_var=eq_var)
            anova_func = _numpy_f_oneway
            mannwhitney_func = _numpy_mannwhitneyu
            levene_func = _numpy_levene
            shapiro_func = _numpy_shapiro
            kruskal_func = lambda *args: _numpy_f_oneway(*args)  # Approximate

        try:
            # Two-group comparisons
            if len(conditions) == 2:
                g1, g2 = groups[conditions[0]], groups[conditions[1]]

                # Independent samples t-test
                t_stat, t_p = ttest_func(g1, g2, True)
                results["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": t_p < 0.05,
                    "groups": [_clean_condition_name(conditions[0]), _clean_condition_name(conditions[1])],
                }

                # Welch's t-test (unequal variances)
                welch_t, welch_p = ttest_func(g1, g2, False)
                results["welch_t_test"] = {
                    "statistic": float(welch_t),
                    "p_value": float(welch_p),
                    "significant": welch_p < 0.05,
                }

                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = mannwhitney_func(g1, g2)
                results["mann_whitney"] = {
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "significant": u_p < 0.05,
                }

                # Cohen's d effect size
                pooled_std = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
                cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                results["cohens_d"] = {
                    "value": float(cohens_d),
                    "interpretation": self._interpret_cohens_d(cohens_d),
                }

            # Multi-group comparisons (2+ conditions)
            group_list = [groups[c] for c in conditions]

            # One-way ANOVA
            f_stat, anova_p = anova_func(*group_list)
            results["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(anova_p),
                "significant": anova_p < 0.05,
                "num_groups": len(conditions),
            }

            # Kruskal-Wallis (non-parametric) - only with scipy
            if SCIPY_AVAILABLE and scipy_stats is not None:
                try:
                    h_stat, kw_p = scipy_stats.kruskal(*group_list)
                    results["kruskal_wallis"] = {
                        "h_statistic": float(h_stat),
                        "p_value": float(kw_p),
                        "significant": kw_p < 0.05,
                    }
                except Exception:
                    pass

            # Eta-squared effect size for ANOVA
            all_data = np.concatenate(group_list)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_list)
            ss_total = np.sum((all_data - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            results["eta_squared"] = {
                "value": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared),
            }

            # Levene's test for homogeneity of variances
            levene_stat, levene_p = levene_func(*group_list)
            results["levene_test"] = {
                "statistic": float(levene_stat),
                "p_value": float(levene_p),
                "homogeneous": levene_p > 0.05,
            }

            # Normality test (on pooled data, limited to 5000)
            if len(all_data) <= 5000:
                shapiro_stat, shapiro_p = shapiro_func(all_data)
                results["normality_test"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "normal": shapiro_p > 0.05,
                    "test_name": "Shapiro-Wilk" if SCIPY_AVAILABLE else "D'Agostino-Pearson (approx)",
                }

            # Pairwise comparisons for 3+ groups
            if len(conditions) >= 3:
                pairwise = []
                for i in range(len(conditions)):
                    for j in range(i + 1, len(conditions)):
                        c1, c2 = conditions[i], conditions[j]
                        g1, g2 = groups[c1], groups[c2]
                        t_stat, t_p = ttest_func(g1, g2, True)
                        pooled_std = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
                        d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                        pairwise.append({
                            "comparison": f"{_clean_condition_name(c1)} vs {_clean_condition_name(c2)}",
                            "t_stat": float(t_stat),
                            "p_value": float(t_p),
                            "cohens_d": float(d),
                            "significant": t_p < 0.05,
                        })
                results["pairwise_comparisons"] = pairwise

        except Exception as e:
            results["error"] = f"Statistical test error: {str(e)}"

        return results

    def _parse_prereg_hypotheses(self, prereg_text: str) -> Dict[str, Any]:
        """Parse pre-registration text to extract hypotheses and analysis plans.

        Returns structured information about:
        - Hypotheses (H1, H2, etc.)
        - Mentioned DVs/outcomes
        - Control variables mentioned
        - Analysis methods mentioned
        """
        result = {
            "hypotheses": [],
            "mentioned_dvs": [],
            "control_variables": [],
            "analysis_methods": [],
            "interactions_mentioned": False,
        }

        if not prereg_text:
            return result

        prereg_lower = prereg_text.lower()

        # Extract hypotheses (H1, H2, Hypothesis 1, etc.)
        hyp_patterns = [
            r'h\d+[:\s]([^.!?\n]+[.!?]?)',
            r'hypothesis\s*\d*[:\s]([^.!?\n]+[.!?]?)',
            r'we\s+(?:hypothesize|predict|expect)\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
        ]
        for pattern in hyp_patterns:
            matches = re.findall(pattern, prereg_lower, re.IGNORECASE)
            result["hypotheses"].extend([m.strip() for m in matches if len(m.strip()) > 10])

        # Identify control variables mentioned
        control_indicators = [
            r'control(?:ling)?\s+(?:for\s+)?(\w+(?:\s+\w+)?)',
            r'covariat(?:e|es)[:\s]+([^.!?\n]+)',
            r'(?:age|gender|sex|education|income)\s+(?:as\s+)?(?:a\s+)?control',
        ]
        for pattern in control_indicators:
            matches = re.findall(pattern, prereg_lower)
            result["control_variables"].extend([m.strip() for m in matches])

        # Check for specific control variables
        if 'age' in prereg_lower:
            if 'age' not in result["control_variables"]:
                result["control_variables"].append('age')
        if 'gender' in prereg_lower or 'sex' in prereg_lower:
            if 'gender' not in result["control_variables"]:
                result["control_variables"].append('gender')

        # Check for interaction effects
        interaction_terms = ['interaction', 'moderat', 'x ', ' × ', 'cross-over', 'crossover']
        result["interactions_mentioned"] = any(term in prereg_lower for term in interaction_terms)

        # Analysis methods mentioned
        method_keywords = {
            't-test': ['t-test', 't test', 'ttest'],
            'ANOVA': ['anova', 'analysis of variance'],
            'regression': ['regression', 'ols', 'ordinary least squares'],
            'chi-squared': ['chi-squared', 'chi-square', 'χ²'],
            'mediation': ['mediation', 'mediator', 'indirect effect'],
            'moderation': ['moderation', 'moderator', 'interaction effect'],
        }
        for method, keywords in method_keywords.items():
            if any(kw in prereg_lower for kw in keywords):
                result["analysis_methods"].append(method)

        return result

    def _run_regression_analysis(
        self,
        df: pd.DataFrame,
        dv_column: str,
        condition_column: str = "CONDITION",
        include_controls: bool = True,
        prereg_controls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run regression analysis with condition as predictor and optional control variables.

        Args:
            df: DataFrame with data
            dv_column: Name of dependent variable column
            condition_column: Name of condition column
            include_controls: Whether to include Age/Gender controls if available
            prereg_controls: Additional control variables from pre-registration

        Works with or without scipy using numpy-based p-value approximations.
        """
        results = {}
        results["scipy_used"] = SCIPY_AVAILABLE
        results["controls_included"] = []

        try:
            # Create dummy variables for conditions
            conditions = df[condition_column].unique().tolist()
            if len(conditions) < 2:
                return {"error": "Need at least 2 conditions"}

            # Reference category is first condition
            reference = _clean_condition_name(conditions[0])

            # Start with condition dummies
            X = pd.get_dummies(df[condition_column], drop_first=True)

            # Add control variables if requested and available
            control_cols = []

            if include_controls:
                # Age control
                age_cols = [c for c in df.columns if 'age' in c.lower() and df[c].dtype in ['int64', 'float64']]
                if age_cols:
                    age_col = age_cols[0]
                    # Standardize age for regression
                    age_data = df[age_col].fillna(df[age_col].mean())
                    X['Age'] = (age_data - age_data.mean()) / (age_data.std() + 1e-10)
                    control_cols.append('Age')
                    results["controls_included"].append('Age')

                # Gender control (dummy coded)
                gender_cols = [c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
                if gender_cols:
                    gender_col = gender_cols[0]
                    gender_dummies = pd.get_dummies(df[gender_col], prefix='Gender', drop_first=True)
                    for col in gender_dummies.columns:
                        X[col] = gender_dummies[col]
                        control_cols.append(col)
                    results["controls_included"].append('Gender')

            # Add pre-registered control variables if specified
            if prereg_controls:
                for ctrl_name in prereg_controls:
                    ctrl_cols = [c for c in df.columns if ctrl_name.lower() in c.lower()]
                    for ctrl_col in ctrl_cols:
                        if ctrl_col not in X.columns and df[ctrl_col].dtype in ['int64', 'float64']:
                            ctrl_data = df[ctrl_col].fillna(df[ctrl_col].mean())
                            X[ctrl_col] = (ctrl_data - ctrl_data.mean()) / (ctrl_data.std() + 1e-10)
                            control_cols.append(ctrl_col)
                            results["controls_included"].append(ctrl_col)

            y = df[dv_column].dropna().values
            X = X.loc[df[dv_column].notna()]

            if len(y) < 3:
                return {"error": "Insufficient data for regression"}

            # CRITICAL: Convert X to numeric array to avoid dtype('O') error
            try:
                X_values = X.values.astype(np.float64)
            except (ValueError, TypeError):
                # If direct conversion fails, convert column by column
                X_values = np.zeros((len(X), len(X.columns)), dtype=np.float64)
                for i, col in enumerate(X.columns):
                    try:
                        X_values[:, i] = pd.to_numeric(X[col], errors='coerce').fillna(0).values
                    except Exception:
                        X_values[:, i] = 0

            # Ensure y is also numeric
            y = np.asarray(y, dtype=np.float64)

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X_values)), X_values])

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
            df_resid = n - k
            mse = ss_res / df_resid if df_resid > 0 else 1
            se = np.sqrt(np.maximum(np.diag(XtX_inv) * mse, 1e-10))

            # t-statistics and p-values
            t_stats = beta / se

            # Calculate p-values (scipy or fallback)
            if SCIPY_AVAILABLE and scipy_stats is not None:
                p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), df_resid))
            else:
                # Use numpy fallback
                p_values = np.array([2 * _t_sf(abs(t), df_resid) for t in t_stats])

            results["coefficients"] = {
                "intercept": {
                    "estimate": float(beta[0]),
                    "std_error": float(se[0]),
                    "t_stat": float(t_stats[0]),
                    "p_value": float(p_values[0]),
                }
            }

            # Condition coefficients (clean names)
            predictor_names = X.columns.tolist()
            results["reference_category"] = reference
            for i, name in enumerate(predictor_names):
                clean_name = _clean_condition_name(name)
                results["coefficients"][clean_name] = {
                    "estimate": float(beta[i + 1]),
                    "std_error": float(se[i + 1]),
                    "t_stat": float(t_stats[i + 1]),
                    "p_value": float(p_values[i + 1]),
                    "significant": p_values[i + 1] < 0.05,
                }

            results["model_fit"] = {
                "r_squared": float(r_squared),
                "adj_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k - 1)) if n > k + 1 else 0,
                "n": n,
                "df_residual": df_resid,
            }

            # F-test for overall model
            if k > 1 and df_resid > 0:
                f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / df_resid) if r_squared < 1 else float('inf')

                # F-test p-value (scipy or fallback)
                if SCIPY_AVAILABLE and scipy_stats is not None:
                    f_p = 1 - scipy_stats.f.cdf(f_stat, k - 1, df_resid)
                else:
                    f_p = _f_sf(f_stat, k - 1, df_resid)

                results["f_test"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(f_p),
                    "significant": f_p < 0.05,
                }

        except Exception as e:
            results["error"] = f"Regression error: {str(e)}"

        return results

    def _run_factorial_anova(
        self,
        df: pd.DataFrame,
        dv_column: str,
        factors: List[Dict[str, Any]],
        condition_column: str = "CONDITION",
    ) -> Dict[str, Any]:
        """Run factorial ANOVA for 2x2+ designs with interaction effects.

        Parses factorial structure from condition names and computes main effects
        and interactions using Type III SS (approximated with numpy when scipy unavailable).
        """
        results = {}
        results["scipy_used"] = SCIPY_AVAILABLE

        try:
            # Need at least 2 factors for factorial ANOVA
            if len(factors) < 2:
                return {"error": "Need at least 2 factors for factorial ANOVA", "single_factor": True}

            # Extract factor levels from condition names
            conditions = df[condition_column].dropna().unique().tolist()
            if len(conditions) < 4:
                return {"error": "Need at least 4 conditions for 2x2 factorial", "conditions": len(conditions)}

            # Try to parse factor structure from conditions
            # Common patterns: "Factor1_Level1 x Factor2_Level1", "Level1-Level2", etc.
            factor1_name = factors[0].get("name", "Factor1") if len(factors) > 0 else "Factor1"
            factor2_name = factors[1].get("name", "Factor2") if len(factors) > 1 else "Factor2"
            factor1_levels = factors[0].get("levels", []) if len(factors) > 0 else []
            factor2_levels = factors[1].get("levels", []) if len(factors) > 1 else []

            # Create factor columns by parsing condition names
            df_analysis = df.copy()

            def extract_factor_level(condition: str, factor_levels: List[str], factor_idx: int) -> Optional[str]:
                """Extract factor level from condition name."""
                condition_lower = str(condition).lower()
                for level in factor_levels:
                    if level.lower() in condition_lower:
                        return level
                # Try positional extraction (split by common delimiters)
                parts = re.split(r'[x×_\-\s]+', str(condition))
                if len(parts) > factor_idx:
                    return parts[factor_idx].strip()
                return None

            # Extract factor levels for each row
            factor1_col = f"_factor1_{factor1_name}"
            factor2_col = f"_factor2_{factor2_name}"

            df_analysis[factor1_col] = df_analysis[condition_column].apply(
                lambda x: extract_factor_level(x, factor1_levels, 0)
            )
            df_analysis[factor2_col] = df_analysis[condition_column].apply(
                lambda x: extract_factor_level(x, factor2_levels, 1)
            )

            # Drop rows with missing factor assignments
            df_analysis = df_analysis.dropna(subset=[factor1_col, factor2_col, dv_column])

            if len(df_analysis) < 10:
                return {"error": "Insufficient data after factor parsing"}

            # Get unique levels
            f1_levels = df_analysis[factor1_col].unique().tolist()
            f2_levels = df_analysis[factor2_col].unique().tolist()

            if len(f1_levels) < 2 or len(f2_levels) < 2:
                return {"error": f"Need at least 2 levels per factor. Found {len(f1_levels)} and {len(f2_levels)}"}

            results["factor1"] = {"name": factor1_name, "levels": f1_levels}
            results["factor2"] = {"name": factor2_name, "levels": f2_levels}

            # Calculate cell means and grand mean
            grand_mean = df_analysis[dv_column].mean()
            n_total = len(df_analysis)

            # Cell means
            cell_stats = {}
            for f1 in f1_levels:
                for f2 in f2_levels:
                    cell_data = df_analysis[(df_analysis[factor1_col] == f1) &
                                           (df_analysis[factor2_col] == f2)][dv_column]
                    if len(cell_data) > 0:
                        cell_key = f"{f1} × {f2}"
                        cell_stats[cell_key] = {
                            "n": len(cell_data),
                            "mean": float(cell_data.mean()),
                            "std": float(cell_data.std()) if len(cell_data) > 1 else 0,
                        }

            results["cell_statistics"] = cell_stats

            # Calculate marginal means
            f1_means = {lvl: df_analysis[df_analysis[factor1_col] == lvl][dv_column].mean() for lvl in f1_levels}
            f2_means = {lvl: df_analysis[df_analysis[factor2_col] == lvl][dv_column].mean() for lvl in f2_levels}

            results["marginal_means"] = {
                factor1_name: {_clean_condition_name(k): float(v) for k, v in f1_means.items()},
                factor2_name: {_clean_condition_name(k): float(v) for k, v in f2_means.items()},
            }

            # Calculate Sum of Squares
            # SS Total
            ss_total = np.sum((df_analysis[dv_column] - grand_mean) ** 2)

            # SS Factor 1 (main effect)
            ss_f1 = sum(
                len(df_analysis[df_analysis[factor1_col] == lvl]) * (f1_means[lvl] - grand_mean) ** 2
                for lvl in f1_levels
            )

            # SS Factor 2 (main effect)
            ss_f2 = sum(
                len(df_analysis[df_analysis[factor2_col] == lvl]) * (f2_means[lvl] - grand_mean) ** 2
                for lvl in f2_levels
            )

            # SS Within (Error)
            ss_within = 0
            for f1 in f1_levels:
                for f2 in f2_levels:
                    cell_data = df_analysis[(df_analysis[factor1_col] == f1) &
                                           (df_analysis[factor2_col] == f2)][dv_column]
                    if len(cell_data) > 0:
                        cell_mean = cell_data.mean()
                        ss_within += np.sum((cell_data - cell_mean) ** 2)

            # SS Interaction (by subtraction)
            ss_interaction = ss_total - ss_f1 - ss_f2 - ss_within

            # Degrees of freedom
            df_f1 = len(f1_levels) - 1
            df_f2 = len(f2_levels) - 1
            df_interaction = df_f1 * df_f2
            df_within = n_total - len(f1_levels) * len(f2_levels)

            if df_within <= 0:
                return {"error": "Insufficient degrees of freedom for error term"}

            # Mean squares
            ms_f1 = ss_f1 / df_f1 if df_f1 > 0 else 0
            ms_f2 = ss_f2 / df_f2 if df_f2 > 0 else 0
            ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
            ms_within = ss_within / df_within if df_within > 0 else 1

            # F statistics
            f_f1 = ms_f1 / ms_within if ms_within > 0 else 0
            f_f2 = ms_f2 / ms_within if ms_within > 0 else 0
            f_interaction = ms_interaction / ms_within if ms_within > 0 else 0

            # P-values
            if SCIPY_AVAILABLE and scipy_stats is not None:
                p_f1 = 1 - scipy_stats.f.cdf(f_f1, df_f1, df_within)
                p_f2 = 1 - scipy_stats.f.cdf(f_f2, df_f2, df_within)
                p_interaction = 1 - scipy_stats.f.cdf(f_interaction, df_interaction, df_within)
            else:
                p_f1 = _f_sf(f_f1, df_f1, df_within)
                p_f2 = _f_sf(f_f2, df_f2, df_within)
                p_interaction = _f_sf(f_interaction, df_interaction, df_within)

            # Effect sizes (partial eta-squared)
            eta2_f1 = ss_f1 / (ss_f1 + ss_within) if (ss_f1 + ss_within) > 0 else 0
            eta2_f2 = ss_f2 / (ss_f2 + ss_within) if (ss_f2 + ss_within) > 0 else 0
            eta2_interaction = ss_interaction / (ss_interaction + ss_within) if (ss_interaction + ss_within) > 0 else 0

            # Store results
            results["main_effect_1"] = {
                "factor": factor1_name,
                "ss": float(ss_f1),
                "df": df_f1,
                "ms": float(ms_f1),
                "f_statistic": float(f_f1),
                "p_value": float(p_f1),
                "partial_eta_squared": float(eta2_f1),
                "significant": p_f1 < 0.05,
                "interpretation": self._interpret_eta_squared(eta2_f1),
            }

            results["main_effect_2"] = {
                "factor": factor2_name,
                "ss": float(ss_f2),
                "df": df_f2,
                "ms": float(ms_f2),
                "f_statistic": float(f_f2),
                "p_value": float(p_f2),
                "partial_eta_squared": float(eta2_f2),
                "significant": p_f2 < 0.05,
                "interpretation": self._interpret_eta_squared(eta2_f2),
            }

            results["interaction"] = {
                "factors": f"{factor1_name} × {factor2_name}",
                "ss": float(ss_interaction),
                "df": df_interaction,
                "ms": float(ms_interaction),
                "f_statistic": float(f_interaction),
                "p_value": float(p_interaction),
                "partial_eta_squared": float(eta2_interaction),
                "significant": p_interaction < 0.05,
                "interpretation": self._interpret_eta_squared(eta2_interaction),
            }

            results["error"] = {
                "ss": float(ss_within),
                "df": df_within,
                "ms": float(ms_within),
            }

            results["total"] = {
                "ss": float(ss_total),
                "df": n_total - 1,
            }

        except Exception as e:
            results["error"] = f"Factorial ANOVA error: {str(e)}"

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

    def _generate_chart_interpretation(
        self,
        chart_data: Dict[str, Tuple[float, float]],
        stats_results: Dict[str, Any],
        scale_name: str
    ) -> str:
        """
        Generate a 1-2 sentence interpretation of the chart results.

        Args:
            chart_data: Dict mapping condition names to (mean, std_error) tuples
            stats_results: Statistical test results
            scale_name: Name of the scale being analyzed

        Returns:
            HTML string with interpretation
        """
        if not chart_data:
            return ""

        conditions = list(chart_data.keys())
        means = {c: chart_data[c][0] for c in conditions}
        n_conditions = len(conditions)

        interpretation_parts = []

        # Find highest and lowest scoring conditions
        sorted_conds = sorted(means.items(), key=lambda x: x[1], reverse=True)
        highest = sorted_conds[0]
        lowest = sorted_conds[-1]
        mean_diff = highest[1] - lowest[1]

        # Get significance and effect size info
        p_value = None
        effect_size = None
        effect_interpretation = None
        is_significant = False

        if "t_test" in stats_results:
            p_value = stats_results["t_test"]["p_value"]
            is_significant = stats_results["t_test"]["significant"]
        elif "anova" in stats_results:
            p_value = stats_results["anova"]["p_value"]
            is_significant = stats_results["anova"]["significant"]

        if "cohens_d" in stats_results:
            effect_size = stats_results["cohens_d"]["value"]
            effect_interpretation = stats_results["cohens_d"]["interpretation"]
        elif "eta_squared" in stats_results:
            effect_size = stats_results["eta_squared"]["value"]
            effect_interpretation = stats_results["eta_squared"]["interpretation"]

        # Generate interpretation
        if n_conditions == 2:
            # Two-group comparison
            if is_significant and p_value is not None:
                interpretation_parts.append(
                    f"<strong>{highest[0]}</strong> scored significantly higher (M = {highest[1]:.2f}) than "
                    f"<strong>{lowest[0]}</strong> (M = {lowest[1]:.2f}), with a difference of {mean_diff:.2f} points "
                    f"(p = {p_value:.4f})."
                )
                if effect_interpretation:
                    interpretation_parts.append(
                        f" This represents a <strong>{effect_interpretation} effect</strong>"
                        f"{f' (d = {effect_size:.2f})' if effect_size else ''}."
                    )
            else:
                interpretation_parts.append(
                    f"No statistically significant difference was found between <strong>{highest[0]}</strong> "
                    f"(M = {highest[1]:.2f}) and <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f})"
                    f"{f' (p = {p_value:.4f})' if p_value else ''}."
                )
                if effect_interpretation and effect_size:
                    interpretation_parts.append(
                        f" The effect size was {effect_interpretation} (d = {effect_size:.2f})."
                    )
        else:
            # Multi-group comparison
            if is_significant and p_value is not None:
                interpretation_parts.append(
                    f"Significant differences were found across conditions (p = {p_value:.4f}). "
                    f"<strong>{highest[0]}</strong> showed the highest mean (M = {highest[1]:.2f}), "
                    f"while <strong>{lowest[0]}</strong> showed the lowest (M = {lowest[1]:.2f})."
                )
                if effect_interpretation:
                    interpretation_parts.append(
                        f" The overall effect was <strong>{effect_interpretation}</strong>"
                        f"{f' (η² = {effect_size:.3f})' if effect_size else ''}."
                    )
            else:
                interpretation_parts.append(
                    f"No significant differences were found across the {n_conditions} conditions"
                    f"{f' (p = {p_value:.4f})' if p_value else ''}. "
                    f"Means ranged from {lowest[1]:.2f} to {highest[1]:.2f}."
                )

        return "".join(interpretation_parts)

    def _generate_executive_summary(
        self,
        all_scale_results: List[Dict[str, Any]],
        prereg_text: Optional[str],
        n_total: int,
        conditions: List[str]
    ) -> str:
        """
        Generate executive summary paragraph with key takeaways.

        Args:
            all_scale_results: List of dicts containing scale analysis results
            prereg_text: Pre-registration text if available
            n_total: Total sample size
            conditions: List of condition names

        Returns:
            HTML string with executive summary
        """
        html = ["<h2>Executive Summary</h2>"]
        html.append("<div class='summary-box' style='background:#f0f7ff;padding:20px;border-radius:8px;border-left:4px solid #3498db;margin:20px 0;'>")

        # Count significant findings
        sig_findings = []
        nonsig_findings = []
        largest_effect = None
        largest_effect_size = 0

        for result in all_scale_results:
            scale_name = result.get("scale_name", "Unknown Scale")
            stats = result.get("stats_results", {})
            is_sig = False
            effect_val = 0
            effect_type = None

            # Check significance
            if "t_test" in stats and stats["t_test"].get("significant"):
                is_sig = True
            elif "anova" in stats and stats["anova"].get("significant"):
                is_sig = True

            # Get effect size
            if "cohens_d" in stats:
                effect_val = abs(stats["cohens_d"]["value"])
                effect_type = "d"
            elif "eta_squared" in stats:
                effect_val = stats["eta_squared"]["value"]
                effect_type = "η²"

            if is_sig:
                sig_findings.append({
                    "scale": scale_name,
                    "effect_size": effect_val,
                    "effect_type": effect_type,
                    "stats": stats
                })
                if effect_val > largest_effect_size:
                    largest_effect_size = effect_val
                    largest_effect = {"scale": scale_name, "value": effect_val, "type": effect_type}
            else:
                nonsig_findings.append({"scale": scale_name, "stats": stats})

        # Generate summary text
        n_scales = len(all_scale_results)
        n_sig = len(sig_findings)
        n_conditions = len(conditions)

        html.append("<p style='font-size:14px;line-height:1.6;margin:0;'>")

        # Opening sentence
        html.append(
            f"This simulation included <strong>{n_total} participants</strong> randomly assigned to "
            f"<strong>{n_conditions} condition{'s' if n_conditions > 1 else ''}</strong> "
            f"({', '.join(conditions[:3])}{', ...' if n_conditions > 3 else ''}). "
        )

        # Main findings
        if n_sig > 0:
            html.append(
                f"Across {n_scales} dependent measure{'s' if n_scales > 1 else ''}, "
                f"<strong>{n_sig} showed statistically significant differences</strong> between conditions. "
            )

            if largest_effect:
                effect_desc = "large" if (largest_effect["type"] == "d" and largest_effect["value"] >= 0.8) or \
                                        (largest_effect["type"] == "η²" and largest_effect["value"] >= 0.14) else \
                             "medium" if (largest_effect["type"] == "d" and largest_effect["value"] >= 0.5) or \
                                        (largest_effect["type"] == "η²" and largest_effect["value"] >= 0.06) else "small"
                html.append(
                    f"The strongest effect was observed for <strong>{largest_effect['scale']}</strong> "
                    f"({largest_effect['type']} = {largest_effect['value']:.3f}, {effect_desc} effect). "
                )
        else:
            html.append(
                f"Across {n_scales} dependent measure{'s' if n_scales > 1 else ''}, "
                f"<strong>no statistically significant differences</strong> were found between conditions. "
            )

        # Pre-registration hypothesis comparison
        if prereg_text:
            prereg_info = self._parse_prereg_hypotheses(prereg_text)
            hypotheses = prereg_info.get("hypotheses", [])

            if hypotheses:
                html.append("<br><br><strong>Hypothesis Evaluation:</strong> ")

                # Simple heuristic: check if any hypothesis keywords match significant findings
                confirmed = []
                not_confirmed = []

                for h in hypotheses:
                    h_text = h.get("text", "").lower()
                    h_matched = False

                    for finding in sig_findings:
                        scale_lower = finding["scale"].lower()
                        if any(word in h_text for word in scale_lower.split()):
                            confirmed.append(h.get("text", "Unknown hypothesis"))
                            h_matched = True
                            break

                    if not h_matched:
                        not_confirmed.append(h.get("text", "Unknown hypothesis"))

                if confirmed:
                    html.append(f"Results appear consistent with {len(confirmed)} pre-registered hypothesis/hypotheses. ")
                if not_confirmed and len(not_confirmed) < len(hypotheses):
                    html.append(f"{len(not_confirmed)} hypothesis/hypotheses did not reach statistical significance. ")
                elif not confirmed and not_confirmed:
                    html.append("Pre-registered hypotheses were not supported by the simulated data. ")

        # Closing recommendation
        html.append("<br><br><em>Note: These are simulated results for pedagogical purposes. ")
        html.append("Actual experimental results may vary based on real participant responses.</em>")
        html.append("</p></div>")

        return "\n".join(html)

    def _create_bar_chart(
        self,
        data: Dict[str, Tuple[float, float]],
        title: str,
        ylabel: str,
        effect_size: Optional[float] = None,
        p_value: Optional[float] = None,
    ) -> Optional[str]:
        """Create an enhanced bar chart with error bars, annotations, and styling."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not data:
            return None

        try:
            # Reset matplotlib state and use default style as fallback
            plt.close('all')
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except Exception:
                try:
                    plt.style.use('seaborn-whitegrid')
                except Exception:
                    plt.style.use('default')

            fig, ax = plt.subplots(figsize=(10, 6))

            conditions = list(data.keys())
            means = [data[c][0] for c in conditions]
            errors = [data[c][1] for c in conditions]

            # Modern color palette (colorblind-friendly)
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
            bar_colors = colors[:len(conditions)]

            # Create bars with gradient effect
            bars = ax.bar(conditions, means, yerr=errors, capsize=8,
                         color=bar_colors, edgecolor='white', linewidth=2,
                         alpha=0.85, error_kw={'linewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50'})

            # Style improvements
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
            ax.tick_params(axis='x', rotation=30, labelsize=11)
            ax.tick_params(axis='y', labelsize=10)

            # Add value labels on bars with better formatting
            for bar, mean, error in zip(bars, means, errors):
                height = bar.get_height()
                ax.annotate(f'{mean:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + error),
                           xytext=(0, 8), textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold',
                           color='#2c3e50',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

            # Add significance and effect size annotation if provided
            annotation_text = []
            if p_value is not None:
                sig_symbol = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
                annotation_text.append(f"p = {p_value:.4f} {sig_symbol}")
            if effect_size is not None:
                annotation_text.append(f"d = {effect_size:.2f}")

            if annotation_text:
                ax.text(0.98, 0.98, "\n".join(annotation_text),
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9, edgecolor='#bdc3c7'))

            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            # Add subtle gridlines
            ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            # Save to base64 with higher DPI
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler chart
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                conditions = list(data.keys())
                means = [data[c][0] for c in conditions]
                ax.bar(conditions, means, color='steelblue', alpha=0.7)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

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
        """Create an enhanced violin/box plot with individual data points."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Clean condition names for display
            df_plot = df.copy()
            df_plot['_clean_condition'] = df_plot[condition_column].apply(_clean_condition_name)
            conditions = df_plot['_clean_condition'].unique().tolist()

            # Modern color palette
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

            positions = range(len(conditions))
            box_data = [df_plot[df_plot['_clean_condition'] == c][column].dropna().values for c in conditions]

            # Create violin plots for density visualization
            parts = ax.violinplot(box_data, positions=positions, showmeans=False,
                                  showmedians=False, showextrema=False)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_edgecolor('white')
                pc.set_alpha(0.3)

            # Overlay box plots
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.3,
                           showfliers=False)

            for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
                patch.set_edgecolor('white')
                patch.set_linewidth(2)
                median.set_color('white')
                median.set_linewidth(2)

            # Style whiskers and caps
            for whisker in bp['whiskers']:
                whisker.set_color('#7f8c8d')
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_color('#7f8c8d')
                cap.set_linewidth(1.5)

            # Add individual data points with jitter
            for i, (pos, data) in enumerate(zip(positions, box_data)):
                if len(data) > 0:
                    jitter = np.random.normal(0, 0.04, len(data))
                    ax.scatter(pos + jitter, data, alpha=0.4, s=20,
                              color=colors[i % len(colors)], edgecolor='white', linewidth=0.5,
                              zorder=3)

            # Add mean markers
            for i, (pos, data) in enumerate(zip(positions, box_data)):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    ax.scatter(pos, mean_val, marker='D', s=80, color='white',
                              edgecolor=colors[i % len(colors)], linewidth=2, zorder=4)

            # Styling
            ax.set_xticks(positions)
            ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=11)
            ax.set_ylabel("Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Add legend for mean marker
            ax.scatter([], [], marker='D', s=80, color='white', edgecolor='#2c3e50',
                      linewidth=2, label='Mean')
            ax.legend(loc='upper right', framealpha=0.9)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler box plot
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                df_plot = df.copy()
                df_plot['_clean_condition'] = df_plot[condition_column].apply(_clean_condition_name)
                conditions = df_plot['_clean_condition'].unique().tolist()
                box_data = [df_plot[df_plot['_clean_condition'] == c][column].dropna().values for c in conditions]

                ax.boxplot(box_data, labels=conditions)
                ax.set_title(title)
                ax.set_ylabel("Score")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_interaction_plot(
        self,
        df: pd.DataFrame,
        column: str,
        factor1_col: str,
        factor2_col: str,
        factor1_name: str,
        factor2_name: str,
        title: str = "Interaction Plot",
    ) -> Optional[str]:
        """Create an interaction plot for factorial designs."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get unique levels
            f1_levels = df[factor1_col].dropna().unique().tolist()
            f2_levels = df[factor2_col].dropna().unique().tolist()

            # Colors and markers
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
            markers = ['o', 's', '^', 'D']

            # Calculate means and SEs for each cell
            for i, f2 in enumerate(f2_levels):
                means = []
                errors = []
                for f1 in f1_levels:
                    cell_data = df[(df[factor1_col] == f1) & (df[factor2_col] == f2)][column].dropna()
                    if len(cell_data) > 0:
                        means.append(cell_data.mean())
                        errors.append(1.96 * cell_data.std() / np.sqrt(len(cell_data)) if len(cell_data) > 1 else 0)
                    else:
                        means.append(np.nan)
                        errors.append(0)

                # Plot line with error bars
                x_positions = range(len(f1_levels))
                ax.errorbar(x_positions, means, yerr=errors,
                           marker=markers[i % len(markers)], markersize=12,
                           color=colors[i % len(colors)], linewidth=2.5,
                           capsize=6, capthick=2, label=f"{factor2_name}: {f2}",
                           markeredgecolor='white', markeredgewidth=2)

            # Styling
            ax.set_xticks(range(len(f1_levels)))
            ax.set_xticklabels([str(l) for l in f1_levels], fontsize=11)
            ax.set_xlabel(factor1_name, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_ylabel("Mean Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Legend
            ax.legend(loc='best', framealpha=0.95, fontsize=10)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception:
            return None

    def _create_effect_size_forest_plot(
        self,
        comparisons: List[Dict[str, Any]],
        title: str = "Effect Sizes (Cohen's d) with 95% CI",
    ) -> Optional[str]:
        """Create a forest plot showing effect sizes for all pairwise comparisons."""
        if not MATPLOTLIB_AVAILABLE or not comparisons:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, max(4, len(comparisons) * 0.6 + 1)))

            y_positions = range(len(comparisons))
            effects = [c['cohens_d'] for c in comparisons]
            labels = [c['comparison'] for c in comparisons]
            significant = [c['significant'] for c in comparisons]

            # Approximate CI for Cohen's d (rough estimate)
            ci_widths = [0.4 for _ in comparisons]  # Simplified

            # Colors based on significance
            colors = ['#2ecc71' if sig else '#95a5a6' for sig in significant]

            # Plot effect sizes
            for i, (effect, label, sig, color) in enumerate(zip(effects, labels, significant, colors)):
                # Horizontal line for CI
                ax.hlines(i, effect - ci_widths[i], effect + ci_widths[i],
                         color=color, linewidth=3, alpha=0.7)
                # Diamond marker for point estimate
                ax.scatter(effect, i, marker='D', s=150, color=color,
                          edgecolor='white', linewidth=2, zorder=3)

            # Reference line at 0
            ax.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,
                      label='No effect')

            # Effect size interpretation zones
            ax.axvspan(-0.2, 0.2, alpha=0.1, color='#f39c12', label='Negligible')
            ax.axvspan(0.2, 0.5, alpha=0.1, color='#f1c40f')
            ax.axvspan(-0.5, -0.2, alpha=0.1, color='#f1c40f')
            ax.axvspan(0.5, 0.8, alpha=0.1, color='#e67e22')
            ax.axvspan(-0.8, -0.5, alpha=0.1, color='#e67e22')

            # Styling
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel("Cohen's d", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Add interpretation text
            xlim = ax.get_xlim()
            ax.text(xlim[1], -0.7, "Green = Significant (p < .05)\nGray = Non-significant",
                   fontsize=9, ha='right', va='top', style='italic', color='#7f8c8d')

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler line plot
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                for j, (ci, f2) in enumerate(pairwise_results):
                    ax.errorbar([0, 1], ci['means'], yerr=ci['ses'], marker='o', label=f2)
                ax.set_title(title)
                ax.legend()
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_histogram_by_condition(
        self,
        df: pd.DataFrame,
        column: str,
        condition_column: str = "CONDITION",
        title: str = "Distribution Histogram by Condition",
    ) -> Optional[str]:
        """Create overlapping histograms for each condition."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))

            # Clean condition names for display
            df_plot = df.copy()
            df_plot['_clean_condition'] = df_plot[condition_column].apply(_clean_condition_name)
            conditions = df_plot['_clean_condition'].unique().tolist()

            # Modern color palette with transparency
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

            # Create histograms for each condition
            for i, cond in enumerate(conditions):
                data = df_plot[df_plot['_clean_condition'] == cond][column].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=15, alpha=0.5, label=cond,
                           color=colors[i % len(colors)], edgecolor='white', linewidth=1)

            # Styling
            ax.set_xlabel("Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_ylabel("Frequency", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
            ax.legend(loc='upper right', framealpha=0.95, fontsize=10)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler histogram
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                df_plot = df.copy()
                data = df_plot[column].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='white')
                ax.set_title(title)
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_means_dot_plot(
        self,
        data: Dict[str, Tuple[float, float]],
        title: str,
        ylabel: str,
        grand_mean: Optional[float] = None,
    ) -> Optional[str]:
        """Create a dot plot with means and error bars - cleaner alternative to bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            conditions = list(data.keys())
            means = [data[c][0] for c in conditions]
            errors = [data[c][1] for c in conditions]

            # Modern color palette
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

            y_positions = range(len(conditions))

            # Plot dots with error bars (horizontal)
            for i, (cond, mean, error) in enumerate(zip(conditions, means, errors)):
                color = colors[i % len(colors)]
                # Error bar
                ax.errorbar(mean, i, xerr=error, fmt='o', markersize=15,
                           color=color, ecolor=color, capsize=8, capthick=2,
                           markeredgecolor='white', markeredgewidth=2, elinewidth=2)
                # Value label
                ax.text(mean + error + 0.05, i, f'{mean:.2f}', va='center', ha='left',
                       fontsize=11, fontweight='bold', color='#2c3e50')

            # Add grand mean line if provided
            if grand_mean is not None:
                ax.axvline(x=grand_mean, color='#e74c3c', linestyle='--', linewidth=2,
                          alpha=0.7, label=f'Grand Mean: {grand_mean:.2f}')
                ax.legend(loc='lower right', framealpha=0.95)

            # Styling
            ax.set_yticks(y_positions)
            ax.set_yticklabels(conditions, fontsize=11)
            ax.set_xlabel(ylabel, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            # Adjust x-axis to show full range
            ax.set_xlim(left=min(means) - max(errors) - 0.5)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
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

        # Condition distribution (with cleaned names)
        if "CONDITION" in df.columns:
            html_parts.append("<h3>Condition Distribution</h3>")
            html_parts.append("<table><tr><th>Condition</th><th>N</th><th>%</th></tr>")
            cond_counts = df["CONDITION"].value_counts()
            for cond, count in cond_counts.items():
                pct = count / n_total * 100
                clean_cond = _clean_condition_name(cond)
                html_parts.append(f"<tr><td>{clean_cond}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
            html_parts.append("</table>")

        # Clean data for analysis
        df_clean = df[df["Exclude_Recommended"] == 0] if "Exclude_Recommended" in df.columns else df

        # DV Analysis with statistical tests
        scales = metadata.get("scales", [])
        html_parts.append("<h2>2. Statistical Analysis by DV</h2>")

        # Track all scale analysis results for executive summary
        all_scale_results = []

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
                        clean_cond = _clean_condition_name(cond)
                        html_parts.append(f"<tr><td>{clean_cond}</td><td>{n}</td><td>{mean:.3f}</td><td>{sd:.3f}</td><td>[{ci_low:.3f}, {ci_high:.3f}]</td></tr>")
                        chart_data[clean_cond] = (mean, 1.96 * se)

                html_parts.append("</table>")

                # Run statistical tests first to get effect size and p-value for chart
                stats_results = {}
                if len(conditions) >= 2 and "CONDITION" in df_analysis.columns:
                    stats_results = self._run_statistical_tests(df_analysis, "_composite", "CONDITION")

                # Extract effect size and p-value for bar chart annotation
                effect_size_val = stats_results.get("cohens_d", {}).get("value") if "cohens_d" in stats_results else None
                p_value_val = stats_results.get("t_test", {}).get("p_value") if "t_test" in stats_results else (
                    stats_results.get("anova", {}).get("p_value") if "anova" in stats_results else None
                )

                # ========================================
                # VISUALIZATIONS - GUARANTEED to produce charts
                # Strategy: Try matplotlib first, always fall back to SVG
                # ========================================
                html_parts.append("<h4>Visualizations</h4>")
                viz_count = 0
                matplotlib_worked = False

                # Prepare data for SVG fallbacks (condition -> list of values)
                svg_dist_data = {}
                if "CONDITION" in df_analysis.columns:
                    for cond in conditions:
                        mask = df_analysis["CONDITION"].apply(
                            lambda x: _clean_condition_name(str(x)) == _clean_condition_name(str(cond))
                        )
                        svg_dist_data[_clean_condition_name(str(cond))] = df_analysis.loc[mask, "_composite"].dropna().tolist()

                # Clean chart_data keys for consistency
                clean_chart_data = {_clean_condition_name(str(k)): v for k, v in chart_data.items()}

                # === TRY MATPLOTLIB CHARTS FIRST ===
                if MATPLOTLIB_AVAILABLE:
                    # Visualization 1: Bar chart with error bars
                    try:
                        chart_img = self._create_bar_chart(
                            chart_data,
                            f"{scale_name}: Means with 95% CI",
                            "Mean Score",
                            effect_size=effect_size_val,
                            p_value=p_value_val
                        )
                        if chart_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{chart_img}' alt='Bar chart'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                    # Visualization 2: Distribution plot (violin + box)
                    try:
                        dist_img = self._create_distribution_plot(
                            df_analysis, "_composite", "CONDITION",
                            f"{scale_name}: Distribution by Condition"
                        )
                        if dist_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{dist_img}' alt='Distribution'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                    # Visualization 3: Histogram by condition
                    try:
                        hist_img = self._create_histogram_by_condition(
                            df_analysis, "_composite", "CONDITION",
                            f"{scale_name}: Score Distribution Histogram"
                        )
                        if hist_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{hist_img}' alt='Histogram'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                # === SVG FALLBACK - GUARANTEED TO WORK ===
                # If matplotlib failed or produced no charts, use SVG
                if viz_count == 0 and SVG_CHARTS_AVAILABLE:
                    try:
                        # SVG Bar Chart - ALWAYS works
                        svg_bar = svg_charts.create_bar_chart_svg(
                            clean_chart_data,
                            title=f"{scale_name}: Means with 95% CI",
                            ylabel="Mean Score",
                            effect_size=effect_size_val,
                            p_value=p_value_val
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_bar)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Distribution Plot - ALWAYS works
                        svg_dist = svg_charts.create_distribution_svg(
                            svg_dist_data,
                            title=f"{scale_name}: Distribution by Condition",
                            xlabel="Score"
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_dist)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Histogram - ALWAYS works
                        svg_hist = svg_charts.create_histogram_svg(
                            svg_dist_data,
                            title=f"{scale_name}: Score Distribution Histogram",
                            xlabel="Score"
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_hist)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Means Comparison - ALWAYS works
                        grand_mean = df_analysis["_composite"].mean() if "_composite" in df_analysis.columns else None
                        svg_means = svg_charts.create_means_comparison_svg(
                            clean_chart_data,
                            title=f"{scale_name}: Condition Means",
                            xlabel="Mean Score",
                            grand_mean=grand_mean
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_means)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                # === ULTIMATE FALLBACK: Generate inline SVG directly ===
                # If even SVG module failed, generate basic inline SVG
                if viz_count == 0 and clean_chart_data:
                    try:
                        # Create a simple inline SVG bar chart directly
                        conds = list(clean_chart_data.keys())
                        means = [clean_chart_data[c][0] for c in conds]
                        max_mean = max(means) if means else 1

                        svg_lines = [
                            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
                            '<rect width="500" height="300" fill="white"/>',
                            f'<text x="250" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">{scale_name}: Condition Means</text>',
                        ]

                        bar_width = 60
                        spacing = 400 / len(conds)
                        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

                        for i, (cond, mean) in enumerate(zip(conds, means)):
                            x = 50 + spacing * i + spacing/2 - bar_width/2
                            bar_height = (mean / max_mean) * 180
                            y = 250 - bar_height
                            color = colors[i % len(colors)]
                            label = cond[:10] if len(cond) > 10 else cond

                            svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" opacity="0.8"/>')
                            svg_lines.append(f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="11" fill="#2c3e50">{mean:.2f}</text>')
                            svg_lines.append(f'<text x="{x + bar_width/2}" y="270" text-anchor="middle" font-size="9" fill="#2c3e50">{label}</text>')

                        svg_lines.append('</svg>')

                        html_parts.append("<div class='chart-container'>")
                        html_parts.append('\n'.join(svg_lines))
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                # Show info message only if we have visualizations
                if viz_count > 0 and not matplotlib_worked:
                    html_parts.append("<p style='font-size:10px;color:#7f8c8d;margin-top:10px;'><em>Charts rendered using SVG visualization engine.</em></p>")

                # Add chart interpretation summary
                if viz_count > 0 and chart_data:
                    interpretation = self._generate_chart_interpretation(chart_data, stats_results, scale_name)
                    if interpretation:
                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                        html_parts.append(f"<strong>Key Finding:</strong> {interpretation}")
                        html_parts.append("</div>")

                # Statistical tests
                if len(conditions) >= 2 and "CONDITION" in df_analysis.columns:

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

                    if "normality_test" in stats_results:
                        sw = stats_results["normality_test"]
                        test_name = sw.get("test_name", "Normality test")
                        if sw["normal"]:
                            html_parts.append("<div class='success-box'>")
                            html_parts.append(f"<strong>{test_name}:</strong> Data appears normally distributed (p = {sw['p_value']:.4f})")
                        else:
                            html_parts.append("<div class='warning-box'>")
                            html_parts.append(f"<strong>{test_name}:</strong> Data may not be normally distributed (p = {sw['p_value']:.4f}). Consider non-parametric tests.")
                        html_parts.append("</div>")

                    # Pairwise comparisons for 3+ groups
                    if "pairwise_comparisons" in stats_results and len(stats_results["pairwise_comparisons"]) > 0:
                        html_parts.append("<h4>Pairwise Comparisons</h4>")
                        html_parts.append("<table><tr><th>Comparison</th><th>t</th><th>p</th><th>Cohen's d</th><th>Significant</th></tr>")
                        for comp in stats_results["pairwise_comparisons"]:
                            sig_class = "sig" if comp["significant"] else "nonsig"
                            sig_text = "Yes" if comp["significant"] else "No"
                            html_parts.append(f"<tr><td>{comp['comparison']}</td><td>{comp['t_stat']:.3f}</td><td class='{sig_class}'>{comp['p_value']:.4f}</td><td>{comp['cohens_d']:.3f}</td><td class='{sig_class}'>{sig_text}</td></tr>")
                        html_parts.append("</table>")
                        html_parts.append("<div class='warning-box'><em>Note: Multiple comparisons may inflate Type I error. Consider Bonferroni or other corrections.</em></div>")

                        # Forest plot for effect sizes
                        forest_img = self._create_effect_size_forest_plot(
                            stats_results["pairwise_comparisons"],
                            f"{scale_name}: Effect Sizes (Cohen's d)"
                        )
                        if forest_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{forest_img}' alt='Forest plot'>")
                            html_parts.append("</div>")

                    # Regression analysis with control variables
                    html_parts.append("<h4>Regression Analysis (with Controls)</h4>")

                    # Parse pre-registration for control variables
                    prereg_info = self._parse_prereg_hypotheses(prereg_text) if prereg_text else {}
                    prereg_controls = prereg_info.get("control_variables", [])

                    reg_results = self._run_regression_analysis(
                        df_analysis, "_composite", "CONDITION",
                        include_controls=True,
                        prereg_controls=prereg_controls
                    )

                    if "error" not in reg_results:
                        # Show which controls were included
                        controls_used = reg_results.get("controls_included", [])
                        if controls_used:
                            html_parts.append(f"<div class='stat-box'><strong>Control variables included:</strong> {', '.join(controls_used)}</div>")

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
                        # Provide a helpful fallback message instead of showing error details
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append("<strong>Regression Note:</strong> The standard regression analysis could not be computed for this data. ")
                        html_parts.append("This can happen when the data contains non-numeric values or insufficient variation. ")
                        html_parts.append("Please refer to the ANOVA results above for condition comparisons.")
                        html_parts.append("</div>")

                    # Factorial ANOVA for 2x2+ designs
                    factors = metadata.get("factors", [])
                    if len(factors) >= 2 and len(conditions) >= 4:
                        html_parts.append("<h4>Factorial ANOVA (Main Effects & Interaction)</h4>")
                        factorial_results = self._run_factorial_anova(df_analysis, "_composite", factors, "CONDITION")

                        if "error" not in factorial_results or factorial_results.get("single_factor"):
                            if "main_effect_1" in factorial_results and "main_effect_2" in factorial_results:
                                # ANOVA summary table
                                html_parts.append("<table><tr><th>Source</th><th>SS</th><th>df</th><th>MS</th><th>F</th><th>p</th><th>η²<sub>p</sub></th></tr>")

                                # Main effect 1
                                me1 = factorial_results["main_effect_1"]
                                sig_class = "sig" if me1["significant"] else "nonsig"
                                html_parts.append(f"<tr><td><strong>{me1['factor']}</strong></td><td>{me1['ss']:.2f}</td><td>{me1['df']}</td><td>{me1['ms']:.2f}</td><td>{me1['f_statistic']:.3f}</td><td class='{sig_class}'>{me1['p_value']:.4f}</td><td>{me1['partial_eta_squared']:.4f}</td></tr>")

                                # Main effect 2
                                me2 = factorial_results["main_effect_2"]
                                sig_class = "sig" if me2["significant"] else "nonsig"
                                html_parts.append(f"<tr><td><strong>{me2['factor']}</strong></td><td>{me2['ss']:.2f}</td><td>{me2['df']}</td><td>{me2['ms']:.2f}</td><td>{me2['f_statistic']:.3f}</td><td class='{sig_class}'>{me2['p_value']:.4f}</td><td>{me2['partial_eta_squared']:.4f}</td></tr>")

                                # Interaction
                                if "interaction" in factorial_results:
                                    inter = factorial_results["interaction"]
                                    sig_class = "sig" if inter["significant"] else "nonsig"
                                    html_parts.append(f"<tr><td><strong>{inter['factors']}</strong></td><td>{inter['ss']:.2f}</td><td>{inter['df']}</td><td>{inter['ms']:.2f}</td><td>{inter['f_statistic']:.3f}</td><td class='{sig_class}'>{inter['p_value']:.4f}</td><td>{inter['partial_eta_squared']:.4f}</td></tr>")

                                # Error
                                if "error" in factorial_results and isinstance(factorial_results["error"], dict):
                                    err = factorial_results["error"]
                                    html_parts.append(f"<tr><td>Residual</td><td>{err['ss']:.2f}</td><td>{err['df']}</td><td>{err['ms']:.2f}</td><td>-</td><td>-</td><td>-</td></tr>")

                                html_parts.append("</table>")

                                # Interpretation
                                html_parts.append("<div class='stat-box'>")
                                html_parts.append("<strong>Interpretation:</strong><br>")
                                if me1["significant"]:
                                    html_parts.append(f"• Main effect of <strong>{me1['factor']}</strong> is significant ({me1['interpretation']} effect size)<br>")
                                else:
                                    html_parts.append(f"• No significant main effect of {me1['factor']}<br>")

                                if me2["significant"]:
                                    html_parts.append(f"• Main effect of <strong>{me2['factor']}</strong> is significant ({me2['interpretation']} effect size)<br>")
                                else:
                                    html_parts.append(f"• No significant main effect of {me2['factor']}<br>")

                                if "interaction" in factorial_results:
                                    inter = factorial_results["interaction"]
                                    if inter["significant"]:
                                        html_parts.append(f"• <strong>Significant interaction</strong> between factors ({inter['interpretation']} effect size). Main effects should be interpreted with caution.<br>")
                                    else:
                                        html_parts.append("• No significant interaction between factors<br>")
                                html_parts.append("</div>")

                                # Cell means table
                                if "cell_statistics" in factorial_results:
                                    html_parts.append("<br><strong>Cell Means:</strong>")
                                    html_parts.append("<table><tr><th>Cell</th><th>N</th><th>Mean</th><th>SD</th></tr>")
                                    for cell, stats in factorial_results["cell_statistics"].items():
                                        html_parts.append(f"<tr><td>{cell}</td><td>{stats['n']}</td><td>{stats['mean']:.3f}</td><td>{stats['std']:.3f}</td></tr>")
                                    html_parts.append("</table>")

                                # Interaction plot for factorial design
                                if "factor1" in factorial_results and "factor2" in factorial_results:
                                    f1_name = factorial_results["factor1"]["name"]
                                    f2_name = factorial_results["factor2"]["name"]
                                    f1_levels = factorial_results["factor1"]["levels"]
                                    f2_levels = factorial_results["factor2"]["levels"]

                                    # Create factor columns for plotting
                                    df_plot = df_analysis.copy()
                                    df_plot["_f1"] = df_plot["CONDITION"].apply(
                                        lambda x: next((l for l in f1_levels if str(l).lower() in str(x).lower()), None)
                                    )
                                    df_plot["_f2"] = df_plot["CONDITION"].apply(
                                        lambda x: next((l for l in f2_levels if str(l).lower() in str(x).lower()), None)
                                    )

                                    interaction_img = self._create_interaction_plot(
                                        df_plot, "_composite", "_f1", "_f2",
                                        f1_name, f2_name,
                                        f"{scale_name}: Interaction Plot"
                                    )
                                    if interaction_img:
                                        html_parts.append("<div class='chart-container'>")
                                        html_parts.append(f"<img src='data:image/png;base64,{interaction_img}' alt='Interaction plot'>")
                                        html_parts.append("</div>")
                        else:
                            err_msg = factorial_results.get("error", "Unable to parse factorial structure")
                            html_parts.append(f"<div class='warning-box'>Factorial ANOVA: {err_msg}</div>")

                # Track this scale's results for executive summary
                all_scale_results.append({
                    "scale_name": scale_name,
                    "chart_data": chart_data,
                    "stats_results": stats_results
                })

        # Chi-squared test for categorical associations
        if "CONDITION" in df_clean.columns and "Gender" in df_clean.columns:
            html_parts.append("<h2>3. Categorical Analysis</h2>")
            html_parts.append("<h3>Condition × Gender</h3>")

            try:
                contingency = pd.crosstab(df_clean["CONDITION"].apply(_clean_condition_name), df_clean["Gender"])

                # Chi-squared test (scipy or numpy fallback)
                if SCIPY_AVAILABLE and scipy_stats is not None:
                    chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency)
                else:
                    # Numpy-based chi-squared calculation
                    observed = contingency.values
                    row_sums = observed.sum(axis=1, keepdims=True)
                    col_sums = observed.sum(axis=0, keepdims=True)
                    total = observed.sum()
                    expected = row_sums * col_sums / total
                    chi2 = np.sum((observed - expected) ** 2 / np.where(expected > 0, expected, 1))
                    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
                    p = 1 - _chi2_cdf(chi2, dof) if dof > 0 else 1.0

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

        # Executive Summary - Main takeaways from all analyses
        if all_scale_results:
            exec_summary = self._generate_executive_summary(
                all_scale_results,
                prereg_text,
                n_total,
                [_clean_condition_name(c) for c in conditions]
            )
            html_parts.append(exec_summary)

        # Footer
        html_parts.append("<h2>Notes for Instructors</h2>")
        html_parts.append("<div class='warning-box'>")
        html_parts.append("<strong>This is simulated data.</strong> Results demonstrate what the analysis pipeline will produce. ")
        html_parts.append("Students should practice these analyses independently and may get similar (but not identical) results due to random variation.")
        html_parts.append("</div>")

        html_parts.append(f"<p style='color:#999;font-size:0.9em;margin-top:30px;'>Generated by Behavioral Experiment Simulation Tool v{__version__}</p>")
        html_parts.append("</div></body></html>")

        return "\n".join(html_parts)
