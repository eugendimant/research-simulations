# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "1.3.8"  # v1.3.8: Builder improvements (sample size guidance, condition hints, duplicate detection, analysis recommendations)

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


def _extract_persona_proportions(persona_dist: Any) -> Dict[str, float]:
    """Extract a flat {persona_name: proportion} dict from persona_distribution metadata.

    v1.2.3: The persona_distribution metadata was changed to a nested dict with
    'counts', 'proportions', and 'total_participants' keys. This function handles
    both the new nested format and the old flat format for backwards compatibility.

    Args:
        persona_dist: Either a flat dict {name: proportion} or nested dict with
                      'proportions' key.

    Returns:
        Flat dict mapping persona names to float proportions (0-1 range).
    """
    if not persona_dist or not isinstance(persona_dist, dict):
        return {}

    # New nested format: {"counts": {...}, "proportions": {...}, "total_participants": N}
    if "proportions" in persona_dist and isinstance(persona_dist["proportions"], dict):
        raw = persona_dist["proportions"]
        result = {}
        for k, v in raw.items():
            try:
                result[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
        return result

    # Old flat format: {"Engaged Responder": 0.35, "Satisficer": 0.22, ...}
    # Check if values are numeric (not dicts/lists)
    result = {}
    for k, v in persona_dist.items():
        if isinstance(v, (int, float)):
            result[str(k)] = float(v)
        elif isinstance(v, str):
            try:
                result[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
        # Skip dict/list values (sub-keys of new format without 'proportions')
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling dicts, None, NaN, and other edge cases.

    v1.2.3: Added to prevent float() crashes on unexpected types.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return default
        return float(value)
    if isinstance(value, dict):
        # Try to extract a numeric value from common dict structures
        for key in ('value', 'proportion', 'mean', 'count'):
            if key in value:
                try:
                    return float(value[key])
                except (ValueError, TypeError):
                    pass
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


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
    # v1.0.0: Guard against division issues with small samples
    skew_denom = 6 * max(n - 2, 1)
    z_skew = skew * math.sqrt((n + 1) * (n + 3) / skew_denom)
    kurt_denom = (n + 1)**2 * (n + 3) * (n + 5)
    kurt_numer = 24 * n * max(n - 2, 1) * max(n - 3, 1)
    z_kurt = kurt / math.sqrt(kurt_numer / max(kurt_denom, 1)) if kurt_denom > 0 else 0

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
    include_python_script: bool = True  # v1.2.3: Added missing config field
    include_spss_syntax: bool = True  # v1.2.3: Added missing config field
    include_stata_script: bool = True  # v1.2.3: Added missing config field


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

        # v1.1.0: Add Study Overview section
        lines.append("## Study Overview")
        lines.append("")
        study_desc = metadata.get('study_description', '')
        if study_desc:
            # Clean and truncate description
            clean_desc = study_desc.strip()[:500]
            if len(study_desc) > 500:
                clean_desc += "..."
            lines.append(f"**Description:** {clean_desc}")
            lines.append("")

        # Design summary
        conditions = metadata.get('conditions', [])
        scales = metadata.get('scales', [])
        sample_size = metadata.get('sample_size', 0)

        lines.append("### Design at a Glance")
        lines.append("")
        lines.append("| Element | Details |")
        lines.append("|---------|---------|")
        lines.append(f"| **Sample Size** | N = {sample_size} |")
        lines.append(f"| **Conditions** | {len(conditions)} ({', '.join(conditions[:4])}{'...' if len(conditions) > 4 else ''}) |")
        lines.append(f"| **Outcome Measures** | {len(scales)} scale(s) detected |")

        # Detected research domain
        domains = metadata.get('detected_domains', [])
        if domains:
            lines.append(f"| **Research Domain** | {domains[0] if domains else 'General'} |")
        lines.append("")

        # =====================================================================
        # v1.2.0: COMPREHENSIVE SIMULATION INTELLIGENCE REPORT
        # =====================================================================
        lines.append("## Simulation Intelligence Report")
        lines.append("")
        lines.append("This section explains how the simulation system analyzed and approached your study, providing transparency into the data generation process.")
        lines.append("")

        # --- TOPIC/DOMAIN ANALYSIS ---
        lines.append("### Topic & Domain Analysis")
        lines.append("")

        detected_domains = metadata.get('detected_domains', [])
        study_context = metadata.get('study_context', {})
        if not isinstance(study_context, dict):
            study_context = {}
        domain_keywords = study_context.get('detected_keywords', [])

        if detected_domains:
            lines.append(f"**Primary Research Domain:** {detected_domains[0].replace('_', ' ').title()}")
            if len(detected_domains) > 1:
                secondary = [d.replace('_', ' ').title() for d in detected_domains[1:5]]
                lines.append(f"**Secondary Domains:** {', '.join(secondary)}")
            lines.append("")

            # Domain explanation
            domain_explanations = {
                'behavioral_economics': 'Study involves economic decision-making, incentives, or behavioral biases',
                'social_psychology': 'Study examines social influence, attitudes, or interpersonal dynamics',
                'consumer_behavior': 'Study focuses on purchasing decisions, brand perceptions, or marketing',
                'ai_attitudes': 'Study explores perceptions of AI, algorithms, or automation',
                'technology_adoption': 'Study examines technology use, acceptance, or digital behaviors',
                'organizational_behavior': 'Study involves workplace dynamics, leadership, or employee attitudes',
                'health_psychology': 'Study addresses health behaviors, medical decisions, or wellbeing',
                'political_psychology': 'Study examines political attitudes, voting, or civic engagement',
                'environmental_psychology': 'Study focuses on environmental attitudes or sustainable behaviors',
                'moral_psychology': 'Study explores ethical judgments, moral reasoning, or values',
            }
            primary_domain = detected_domains[0].lower()
            if primary_domain in domain_explanations:
                lines.append(f"**Domain Interpretation:** {domain_explanations[primary_domain]}")
                lines.append("")
        else:
            lines.append("**Research Domain:** General (no specific domain detected)")
            lines.append("")

        if domain_keywords:
            lines.append(f"**Keywords Detected:** {', '.join(domain_keywords[:10])}")
            lines.append("")

        # --- SIMULATION APPROACH ---
        lines.append("### How the Simulation Approached This Study")
        lines.append("")
        lines.append("Based on the analysis of your QSF file, study description, and condition structure, the simulation:")
        lines.append("")

        conditions = metadata.get('conditions', [])
        scales = metadata.get('scales', [])
        effect_sizes = metadata.get('effect_sizes_configured', []) or metadata.get('effect_sizes', [])

        # Describe the approach
        approach_points = []

        if len(conditions) > 1:
            approach_points.append(f"Assigned **{len(conditions)} experimental conditions** with balanced allocation")

        if effect_sizes:
            approach_points.append(f"Applied **{len(effect_sizes)} user-specified effect size(s)** to create systematic condition differences")
        else:
            approach_points.append("Applied **automatic semantic-based effects** derived from condition label analysis")

        if scales:
            scale_types = set(s.get('type', 'likert') for s in scales)
            approach_points.append(f"Generated responses for **{len(scales)} DV(s)** ({', '.join(scale_types)})")

        open_ended = metadata.get('open_ended_questions', [])
        if open_ended:
            approach_points.append(f"Created **{len(open_ended)} unique open-ended responses** per participant based on detected topic context")

        for point in approach_points:
            lines.append(f"- {point}")
        lines.append("")

        # --- CONDITION EFFECT STRATEGY ---
        lines.append("### Condition Effects Strategy")
        lines.append("")

        if effect_sizes:
            lines.append("**User-Specified Effects:**")
            lines.append("")
            lines.append("| Variable | Factor | High Level | Low Level | Cohen's d |")
            lines.append("|----------|--------|------------|-----------|-----------|")
            for es in effect_sizes[:10]:
                var = es.get('variable', 'DV')
                factor = es.get('factor', 'Condition')
                high = es.get('level_high', 'Treatment')
                low = es.get('level_low', 'Control')
                d = es.get('cohens_d', 0.5)
                lines.append(f"| {var} | {factor} | {high} | {low} | d = {d:.2f} |")
            lines.append("")
        else:
            lines.append("**Automatic Semantic Effects:** The system analyzed your condition labels to apply research-grounded effects.")
            lines.append("")

            # Analyze condition names for semantic content
            semantic_effects = []
            condition_keywords = {
                'ai': ('AI/Algorithm', -0.12, 'Algorithm aversion effect (Dietvorst et al., 2015)'),
                'human': ('Human agent', +0.08, 'Human preference in decision-making'),
                'control': ('Control condition', 0.0, 'Baseline comparison'),
                'treatment': ('Treatment', +0.15, 'Active intervention effect'),
                'gain': ('Gain frame', +0.12, 'Positive framing effect'),
                'loss': ('Loss frame', -0.18, 'Loss aversion (Kahneman & Tversky)'),
                'scarcity': ('Scarcity', +0.25, 'Scarcity principle (Cialdini)'),
                'social': ('Social proof', +0.20, 'Social influence effect'),
                'hedonic': ('Hedonic', +0.22, 'Hedonic consumption boost'),
                'utilitarian': ('Utilitarian', -0.08, 'Utilitarian discount'),
                'high': ('High condition', +0.15, 'Elevated manipulation'),
                'low': ('Low condition', -0.15, 'Reduced manipulation'),
            }

            for cond in conditions:
                cond_lower = cond.lower()
                for keyword, (label, effect, cite) in condition_keywords.items():
                    if keyword in cond_lower:
                        semantic_effects.append((cond, label, effect, cite))
                        break

            if semantic_effects:
                lines.append("| Condition | Detected Pattern | Effect Applied | Research Basis |")
                lines.append("|-----------|-----------------|----------------|----------------|")
                for cond, label, effect, cite in semantic_effects[:8]:
                    effect_str = f"+{effect:.2f}" if effect > 0 else f"{effect:.2f}"
                    lines.append(f"| {cond} | {label} | {effect_str} | {cite} |")
                lines.append("")
            else:
                lines.append("_No specific semantic patterns detected in condition labels. Equal baseline applied to all conditions._")
                lines.append("")

        # --- OBSERVED EFFECTS ---
        observed_effects = metadata.get('effect_sizes_observed', [])
        if observed_effects:
            lines.append("### Observed Effects in Generated Data")
            lines.append("")
            lines.append("| Variable | Condition 1 | Condition 2 | M₁ | M₂ | Cohen's d |")
            lines.append("|----------|-------------|-------------|-----|-----|-----------|")
            for obs in observed_effects[:10]:
                var = obs.get('variable', 'DV')[:20]
                c1 = obs.get('condition_1', 'C1')[:15]
                c2 = obs.get('condition_2', 'C2')[:15]
                m1 = obs.get('mean_1', 0)
                m2 = obs.get('mean_2', 0)
                d = obs.get('cohens_d', 0)
                lines.append(f"| {var} | {c1} | {c2} | {m1:.2f} | {m2:.2f} | {d:.2f} |")
            lines.append("")

            lines.append("**Effect Size Interpretation:**")
            lines.append("- d = 0.2: Small effect (subtle but potentially meaningful)")
            lines.append("- d = 0.5: Medium effect (moderate practical significance)")
            lines.append("- d = 0.8: Large effect (substantial difference)")
            lines.append("")

        # --- PERSONA RATIONALE ---
        lines.append("### Persona Selection Rationale")
        lines.append("")

        persona_dist_raw = metadata.get('persona_distribution', {})
        persona_dist = _extract_persona_proportions(persona_dist_raw)
        if persona_dist:
            # Explain why certain personas were chosen
            lines.append("The simulation assigned response style personas based on:")
            lines.append("")
            lines.append("1. **Base prevalence rates** from survey methodology research (Krosnick, 1991; Meade & Craig, 2012)")
            lines.append("2. **Domain-specific adjustments** based on detected research topic")
            lines.append("3. **Realistic online panel proportions** (~35% engaged, ~22% satisficers, ~5% careless)")
            lines.append("")

            # Show persona traits summary
            persona_traits = {
                'engaged responder': {'attention': 0.92, 'consistency': 0.78, 'extremity': 0.18},
                'satisficer': {'attention': 0.68, 'consistency': 0.55, 'extremity': 0.12},
                'extreme responder': {'attention': 0.80, 'consistency': 0.70, 'extremity': 0.88},
                'acquiescent': {'attention': 0.75, 'consistency': 0.65, 'extremity': 0.35},
                'careless': {'attention': 0.35, 'consistency': 0.28, 'extremity': 0.45},
            }

            lines.append("**Persona Trait Profiles:**")
            lines.append("")
            lines.append("| Persona | Attention Level | Response Consistency | Endpoint Use |")
            lines.append("|---------|-----------------|---------------------|--------------|")
            for persona, share in sorted(persona_dist.items(), key=lambda x: -_safe_float(x[1]))[:5]:
                traits = persona_traits.get(persona.lower(), {'attention': 0.7, 'consistency': 0.6, 'extremity': 0.3})
                att = traits['attention']
                con = traits['consistency']
                ext = traits['extremity']
                att_label = "High" if att > 0.8 else ("Medium" if att > 0.5 else "Low")
                con_label = "High" if con > 0.7 else ("Medium" if con > 0.5 else "Low")
                ext_label = "High" if ext > 0.6 else ("Medium" if ext > 0.3 else "Low")
                lines.append(f"| {persona.title()} | {att_label} ({att:.0%}) | {con_label} ({con:.0%}) | {ext_label} ({ext:.0%}) |")
            lines.append("")
        else:
            lines.append("_Persona distribution details not available._")
            lines.append("")

        # --- OPEN-ENDED RESPONSE GENERATION ---
        open_ended_details = metadata.get('open_ended_questions', [])
        if open_ended_details:
            lines.append("### Open-Ended Response Generation")
            lines.append("")
            lines.append(f"**{len(open_ended_details)} open-ended question(s)** were detected and populated with contextually appropriate text responses.")
            lines.append("")
            lines.append("**Generation approach:**")
            lines.append("- Responses are unique per participant (no duplicate sentences within the dataset)")
            lines.append("- Topic context is extracted from question text to ensure relevance")
            lines.append("- Response length varies based on simulated persona verbosity")
            lines.append("- Sentiment aligns with participant's scale responses for consistency")
            lines.append("")

            # Show sample question types
            lines.append("**Questions detected:**")
            lines.append("")
            for i, q in enumerate(open_ended_details[:5], 1):
                q_name = q.get('variable_name', q.get('name', f'Q{i}'))
                q_text = q.get('question_text', '')[:60]
                if q_text:
                    lines.append(f"- **{q_name}**: \"{q_text}...\"")
                else:
                    lines.append(f"- **{q_name}**")
            if len(open_ended_details) > 5:
                lines.append(f"- _...and {len(open_ended_details) - 5} more_")
            lines.append("")

        lines.append("---")
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
        lines.append(f"| Attention check pass rate | {_safe_float(metadata.get('attention_rate', 0.85)):.0%} |")
        lines.append(f"| Random responder rate | {_safe_float(metadata.get('random_responder_rate', 0.05)):.0%} |")
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
            lines.append(f"| **Conditions** | {len(conditions)}: {', '.join(str(c) for c in conditions) if conditions else 'N/A'} |")

            factors = metadata.get("factors", []) or []
            if factors:
                for i, f in enumerate(factors):
                    fname = f.get("name", "Factor")
                    levels = f.get("levels", [])
                    lines.append(f"| **Factor {i+1}** | {fname}: {', '.join(str(l) for l in levels)} |")
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

            dist_raw = metadata.get("persona_distribution", {}) or {}
            dist = _extract_persona_proportions(dist_raw)
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

                for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                    persona_key = str(persona).lower() if persona else "unknown"
                    info = persona_info.get(persona_key, {"desc": "Standard response pattern", "chars": "Typical survey behavior"})
                    share_f = _safe_float(share)
                    pct = share_f * 100 if share_f <= 1 else share_f
                    lines.append(f"| **{str(persona).title()}** | {info['desc']} | {info['chars']} | {pct:.1f}% |")
                lines.append("")

                # Show total participants by persona
                lines.append("### Participant Counts by Persona")
                lines.append("")
                n_total = metadata.get("sample_size", len(df))
                lines.append("| Persona | Approximate Count | Expected Impact |")
                lines.append("|---------|-------------------|-----------------|")
                for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                    share_f = _safe_float(share)
                    share_val = share_f if share_f <= 1 else share_f / 100
                    count = int(round(n_total * share_val))
                    # Describe expected impact
                    impact = self._get_persona_impact(str(persona).lower() if persona else "unknown")
                    lines.append(f"| {str(persona).title()} | ~{count} | {impact} |")
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

        # === v1.2.4: PRACTICAL DATA INTERPRETATION GUIDE ===
        lines.append("## How to Work With This Data")
        lines.append("")
        lines.append("### Quick Start Checklist")
        lines.append("")
        lines.append("1. **Load the CSV** into your preferred analysis tool (R, Python, SPSS, Stata, Julia)")
        lines.append("2. **Check the codebook** (Data_Codebook_Handbook.txt) for variable definitions")
        lines.append("3. **Examine exclusion flags** — filter out flagged participants before analysis")
        lines.append("4. **Verify your conditions** — check that CONDITION matches your expected groups")
        lines.append("5. **Run descriptive statistics** before hypothesis testing")
        lines.append("")

        conditions = metadata.get("conditions", [])
        if conditions and len(conditions) >= 2:
            lines.append("### Suggested Analysis Approach")
            lines.append("")
            if len(conditions) == 2:
                lines.append(f"With **2 conditions** ({', '.join(str(c) for c in conditions)}), consider:")
                lines.append("- **Independent samples t-test** for comparing condition means")
                lines.append("- **Mann-Whitney U test** as a non-parametric alternative")
                lines.append("- **Cohen's d** for effect size estimation")
            elif len(conditions) <= 4:
                lines.append(f"With **{len(conditions)} conditions** ({', '.join(str(c) for c in conditions)}), consider:")
                lines.append("- **One-way ANOVA** for comparing condition means")
                lines.append("- **Kruskal-Wallis test** as a non-parametric alternative")
                lines.append("- **Post-hoc pairwise comparisons** (Tukey HSD or Bonferroni)")
                lines.append("- **Eta-squared (η²)** for effect size")
            else:
                lines.append(f"With **{len(conditions)} conditions**, consider a structured ANOVA approach")
            lines.append("")

            factors = metadata.get("factors", [])
            if factors and len(factors) >= 2:
                factor_names = [f.get("name", "Factor") for f in factors if isinstance(f, dict)]
                if len(factor_names) >= 2:
                    lines.append(f"Your **factorial design** ({' × '.join(factor_names)}) allows testing:")
                    lines.append(f"- Main effect of {factor_names[0]}")
                    lines.append(f"- Main effect of {factor_names[1]}")
                    lines.append(f"- **Interaction** between {factor_names[0]} and {factor_names[1]}")
                    lines.append("- Use **two-way ANOVA** (or factorial ANOVA) for this analysis")
                    lines.append("")

        exclusion_summary = metadata.get("exclusion_summary", {})
        if exclusion_summary:
            lines.append("### Data Cleaning Guide")
            lines.append("")
            lines.append("Your dataset includes these quality flags (use `Exclude_Recommended` for filtering):")
            lines.append("")
            n = metadata.get("sample_size", "N/A")
            flagged_speed = exclusion_summary.get("flagged_speed", 0)
            flagged_attention = exclusion_summary.get("flagged_attention", 0)
            flagged_straight = exclusion_summary.get("flagged_straightline", 0)
            total_excluded = exclusion_summary.get("total_excluded", 0)
            lines.append(f"| Flag | Count | % of Sample | Action |")
            lines.append(f"|------|-------|-------------|--------|")
            if isinstance(n, (int, float)) and n > 0:
                lines.append(f"| Speed flags | {flagged_speed} | {flagged_speed/n*100:.1f}% | Too fast/slow completion |")
                lines.append(f"| Attention flags | {flagged_attention} | {flagged_attention/n*100:.1f}% | Failed attention checks |")
                lines.append(f"| Straight-lining | {flagged_straight} | {flagged_straight/n*100:.1f}% | Identical responses in sequence |")
                lines.append(f"| **Total excluded** | **{total_excluded}** | **{total_excluded/n*100:.1f}%** | **Recommended for removal** |")
            lines.append("")
            lines.append("**Tip:** Filter with `Exclude_Recommended == 0` to keep only clean responses.")
            lines.append("")

        # v1.2.0: Enhanced Analysis Recommendations with statistical test recommendations and power analysis
        lines.append("## Analysis Recommendations")
        lines.append("")

        # Get design information
        conditions = metadata.get('conditions', [])
        factors = metadata.get('factors', [])
        factors = [f for f in factors if isinstance(f, dict)]
        scales = metadata.get('scales', [])
        scales = [s for s in scales if isinstance(s, dict)]
        sample_size = metadata.get('sample_size', len(df) if df is not None else 100)
        effect_sizes_cfg = metadata.get('effect_sizes_configured', [])
        is_factorial = len(factors) >= 2
        num_conditions = len(conditions)

        lines.append("### Suggested Analysis Steps")
        lines.append("")
        lines.append("1. **Data Cleaning**")
        lines.append("   - Review `Exclude_Recommended` column for data quality issues")
        lines.append("   - Check `Completion_Time_Seconds` for speedy responders (< 60s suspicious)")
        lines.append("   - Examine `Max_Straight_Line` for response patterns (> 5 suggests inattention)")
        lines.append("   - Check `Attention_Pass_Rate` (< 50% warrants exclusion)")
        lines.append("")
        lines.append("2. **Descriptive Statistics**")
        lines.append("   - Calculate means and SDs by condition")
        lines.append("   - Check for outliers (beyond 3 SD from mean)")
        lines.append("   - Verify condition balance (N per group)")
        lines.append("   - Assess normality (Shapiro-Wilk test)")
        lines.append("")

        # Statistical Test Recommendations
        lines.append("### Statistical Test Recommendations")
        lines.append("")
        has_ordinal = any(s.get('scale_points', 7) <= 5 for s in scales)

        if is_factorial:
            factor_str = " x ".join([str(len(f.get('levels', []))) for f in factors])
            lines.append(f"**Design:** {factor_str} Factorial")
            lines.append("")
            lines.append("| Analysis | Purpose |")
            lines.append("|----------|---------|")
            lines.append("| **Factorial ANOVA** | Main effects + interaction |")
            lines.append("| **Simple effects** | If interaction significant |")
            lines.append("")
            if has_ordinal:
                lines.append("**Non-parametric:** Aligned Rank Transform (ART) ANOVA")
                lines.append("")
        elif num_conditions == 2:
            lines.append("**Design:** Two-Group Comparison")
            lines.append("")
            lines.append("| Analysis | When to Use |")
            lines.append("|----------|-------------|")
            lines.append("| **Independent t-test** | Normal data, equal variances |")
            lines.append("| **Welch's t-test** | Unequal variances |")
            lines.append("| **Mann-Whitney U** | Non-normal or ordinal data |")
            lines.append("")
            lines.append("**Report:** t, df, p, Cohen's d")
            lines.append("")
        elif num_conditions > 2:
            lines.append(f"**Design:** {num_conditions}-Group Comparison")
            lines.append("")
            lines.append("| Analysis | When to Use |")
            lines.append("|----------|-------------|")
            lines.append("| **One-way ANOVA** | Normal data |")
            lines.append("| **Kruskal-Wallis** | Non-normal or ordinal |")
            lines.append("")
            lines.append("**Post-hoc:** Tukey HSD or Dunn's test")
            lines.append("")
            lines.append("**Report:** F (or H), df, p, eta-squared")
            lines.append("")

        # Power Analysis
        lines.append("### Power Analysis Estimates")
        lines.append("")
        n_per_group = sample_size // max(num_conditions, 1) if num_conditions > 0 else sample_size
        lines.append(f"**Sample:** N = {sample_size} (~{n_per_group} per condition)")
        lines.append("")
        lines.append("| Effect Size | d | Estimated Power |")
        lines.append("|-------------|---|-----------------|")
        z_alpha = 1.96
        for label, d in [("Small", 0.2), ("Medium", 0.5), ("Large", 0.8)]:
            ncp = abs(d) * math.sqrt(n_per_group / 2)
            z_power = ncp - z_alpha
            power = 0.5 * (1 + math.erf(z_power / math.sqrt(2)))
            power = max(0.05, min(0.99, power))
            lines.append(f"| {label} | {d:.2f} | {power:.0%} |")
        lines.append("")
        lines.append("*80% power is generally considered adequate*")
        lines.append("")

        # Effect Size Guide
        lines.append("### Effect Size Interpretation")
        lines.append("")
        lines.append("| Measure | Small | Medium | Large |")
        lines.append("|---------|-------|--------|-------|")
        lines.append("| Cohen's d | 0.20 | 0.50 | 0.80 |")
        lines.append("| Eta-squared | 0.01 | 0.06 | 0.14 |")
        lines.append("")

        if self.config.include_schema_validation and schema_validation is not None:
            lines.append("## Schema validation")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(schema_validation, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")

        # Analysis Scripts Section
        lines.append("## Analysis Scripts")
        lines.append("")
        lines.append("Auto-generated scripts with explanatory comments for your statistical software:")
        lines.append("")

        if self.config.include_r_script:
            lines.append("### R Script")
            lines.append("")
            lines.append("```r")
            lines.append(self._generate_comprehensive_r_script(metadata))
            lines.append("```")
            lines.append("")

        if self.config.include_python_script:
            lines.append("### Python Script")
            lines.append("")
            lines.append("```python")
            lines.append(self._generate_python_script(metadata))
            lines.append("```")
            lines.append("")

        if self.config.include_spss_syntax:
            lines.append("### SPSS Syntax")
            lines.append("")
            lines.append("```spss")
            lines.append(self._generate_spss_syntax(metadata))
            lines.append("```")
            lines.append("")

        if self.config.include_stata_script:
            lines.append("### Stata Script")
            lines.append("")
            lines.append("```stata")
            lines.append(self._generate_stata_script(metadata))
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

    def _generate_comprehensive_r_script(self, metadata: Dict[str, Any]) -> str:
        """Generate comprehensive R script with explanatory comments."""
        conditions = metadata.get("conditions", [])
        factors = metadata.get("factors", [])
        scales = metadata.get('scales', [])
        is_factorial = len(factors) >= 2
        num_conditions = len(conditions)

        def _r_quote(x: str) -> str:
            return f'"{str(x).replace(chr(92), chr(92)+chr(92)).replace(chr(34), chr(92)+chr(34))}"'

        condition_levels = ", ".join([_r_quote(c) for c in conditions])

        r_lines = [
            "# ============================================================================",
            f"# COMPREHENSIVE R ANALYSIS SCRIPT",
            f"# Study: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# ============================================================================",
            "",
            "# --- SECTION 1: SETUP ---",
            "# Load required packages (install if needed)",
            "library(readr)      # CSV reading",
            "library(dplyr)      # Data manipulation",
            "# library(effectsize) # Effect sizes (optional)",
            "# library(car)        # Levene's test (optional)",
            "",
            "# --- SECTION 2: DATA LOADING ---",
            "data <- read_csv('Simulated.csv', show_col_types = FALSE)",
            "head(data)  # Verify data loaded correctly",
            "",
            "# --- SECTION 3: DATA PREPARATION ---",
            "# Set condition as factor with proper ordering",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({condition_levels}))",
            "",
            "# --- SECTION 4: DATA CLEANING ---",
            "# Apply exclusion criteria (document in your methods!)",
            "data_clean <- data %>% filter(Exclude_Recommended == 0)",
            "cat('Total N:', nrow(data), '| Clean N:', nrow(data_clean), '\\n')",
            "",
            "# --- SECTION 5: COMPUTE COMPOSITES ---",
        ]

        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            num_items = int(s.get("num_items", 5) or 5)
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            items_quoted = ", ".join([f'"{x}"' for x in items])
            r_lines.append(f"# {name}: {num_items} items")
            r_lines.append(f"data_clean${name}_composite <- rowMeans(data_clean[, c({items_quoted})], na.rm = TRUE)")

        r_lines.append("")
        r_lines.append("# --- SECTION 6: DESCRIPTIVES BY CONDITION ---")
        r_lines.append("data_clean %>% group_by(CONDITION) %>%")
        r_lines.append("  summarise(n = n(),")
        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            r_lines.append(f"            {name}_M = mean({name}_composite, na.rm=TRUE),")
            r_lines.append(f"            {name}_SD = sd({name}_composite, na.rm=TRUE),")
        r_lines.append("  )")
        r_lines.append("")
        r_lines.append("# --- SECTION 7: STATISTICAL TESTS ---")

        if is_factorial:
            r_lines.append("# FACTORIAL ANOVA - tests main effects and interaction")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                fnames = [f.get('name', 'F').replace(' ', '_') for f in factors]
                formula = " * ".join(fnames)
                r_lines.append(f"model_{name} <- aov({name}_composite ~ {formula}, data = data_clean)")
                r_lines.append(f"summary(model_{name})")
                r_lines.append("# Effect sizes: effectsize::eta_squared(model, partial = TRUE)")
        elif num_conditions == 2:
            r_lines.append("# TWO-GROUP T-TEST")
            r_lines.append("# Welch's t-test (var.equal=FALSE) is more robust")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                r_lines.append(f"t.test({name}_composite ~ CONDITION, data = data_clean, var.equal = FALSE)")
                r_lines.append(f"# Effect size: effectsize::cohens_d({name}_composite ~ CONDITION, data = data_clean)")
            r_lines.append("")
            r_lines.append("# NON-PARAMETRIC ALTERNATIVE (if non-normal):")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                r_lines.append(f"# wilcox.test({name}_composite ~ CONDITION, data = data_clean)")
        elif num_conditions > 2:
            r_lines.append("# ONE-WAY ANOVA")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                r_lines.append(f"model_{name} <- aov({name}_composite ~ CONDITION, data = data_clean)")
                r_lines.append(f"summary(model_{name})")
                r_lines.append(f"# Post-hoc: TukeyHSD(model_{name})")
            r_lines.append("")
            r_lines.append("# NON-PARAMETRIC ALTERNATIVE:")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                r_lines.append(f"# kruskal.test({name}_composite ~ CONDITION, data = data_clean)")

        r_lines.append("")
        r_lines.append("# ============================================================================")
        return "\n".join(r_lines)

    def _generate_python_script(self, metadata: Dict[str, Any]) -> str:
        """Generate Python analysis script with explanatory comments."""
        conditions = metadata.get("conditions", [])
        factors = metadata.get("factors", [])
        scales = metadata.get('scales', [])
        is_factorial = len(factors) >= 2
        num_conditions = len(conditions)

        py_lines = [
            "# ============================================================================",
            f"# COMPREHENSIVE PYTHON ANALYSIS SCRIPT",
            f"# Study: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# ============================================================================",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "",
            "# --- SECTION 1: DATA LOADING ---",
            "df = pd.read_csv('Simulated.csv')",
            "print('Dataset shape:', df.shape)",
            "",
            "# --- SECTION 2: DATA PREPARATION ---",
            f"condition_order = {conditions}",
            "df['CONDITION'] = pd.Categorical(df['CONDITION'], categories=condition_order, ordered=True)",
            "",
            "# --- SECTION 3: DATA CLEANING ---",
            "# Apply exclusion criteria (document in methods!)",
            "df_clean = df[df['Exclude_Recommended'] == 0].copy()",
            "print(f'Total N: {len(df)} | Clean N: {len(df_clean)}')",
            "",
            "# --- SECTION 4: COMPUTE COMPOSITES ---",
        ]

        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            num_items = int(s.get("num_items", 5) or 5)
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            py_lines.append(f"# {name}: {num_items} items")
            py_lines.append(f"df_clean['{name}_composite'] = df_clean[{items}].mean(axis=1)")

        py_lines.append("")
        py_lines.append("# --- SECTION 5: DESCRIPTIVES ---")
        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            py_lines.append(f"print(df_clean.groupby('CONDITION')['{name}_composite'].agg(['count', 'mean', 'std']))")

        py_lines.append("")
        py_lines.append("# --- SECTION 6: STATISTICAL TESTS ---")

        if num_conditions == 2 and conditions:
            py_lines.append("# TWO-GROUP T-TEST")
            py_lines.append(f"cond1, cond2 = '{conditions[0]}', '{conditions[1]}'")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                py_lines.append(f"g1 = df_clean[df_clean['CONDITION'] == cond1]['{name}_composite']")
                py_lines.append(f"g2 = df_clean[df_clean['CONDITION'] == cond2]['{name}_composite']")
                py_lines.append("t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test")
                py_lines.append(f"print(f'{name}: t={{t_stat:.3f}}, p={{p_val:.4f}}')")
                py_lines.append("# Cohen's d")
                py_lines.append("d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)")
                py_lines.append("print(f\"Cohen's d = {d:.3f}\")")
        elif num_conditions > 2:
            py_lines.append("# ONE-WAY ANOVA")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                py_lines.append(f"groups = [df_clean[df_clean['CONDITION']==c]['{name}_composite'] for c in condition_order]")
                py_lines.append("f_stat, p_val = stats.f_oneway(*groups)")
                py_lines.append(f"print(f'{name} ANOVA: F={{f_stat:.3f}}, p={{p_val:.4f}}')")

        py_lines.append("")
        py_lines.append("# ============================================================================")
        return "\n".join(py_lines)

    def _generate_spss_syntax(self, metadata: Dict[str, Any]) -> str:
        """Generate SPSS syntax with explanatory comments."""
        conditions = metadata.get("conditions", [])
        scales = metadata.get('scales', [])
        num_conditions = len(conditions)

        spss_lines = [
            "* ============================================================================.",
            f"* COMPREHENSIVE SPSS ANALYSIS SYNTAX.",
            f"* Study: {metadata.get('study_title', 'Untitled Study')}.",
            f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            "* ============================================================================.",
            "",
            "* --- SECTION 1: DATA LOADING ---.",
            "* Use File > Open > Data to import Simulated.csv.",
            "* Or use GET DATA /TYPE=TXT command.",
            "",
            "* --- SECTION 2: DATA CLEANING ---.",
            "* Filter to clean data (exclude flagged participants).",
            "USE ALL.",
            "COMPUTE filter_clean = (Exclude_Recommended = 0).",
            "FILTER BY filter_clean.",
            "EXECUTE.",
            "",
            "* Check sample size.",
            "FREQUENCIES VARIABLES=CONDITION.",
            "",
            "* --- SECTION 3: COMPUTE COMPOSITES ---.",
        ]

        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_")
            num_items = int(s.get("num_items", 5) or 5)
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            items_str = " ".join(items)
            spss_lines.append(f"* {name}: {num_items} items.")
            spss_lines.append(f"COMPUTE {name}_composite = MEAN({items_str}).")
            spss_lines.append("EXECUTE.")

        spss_lines.append("")
        spss_lines.append("* --- SECTION 4: DESCRIPTIVES ---.")
        dv_vars = " ".join([f"{str(s.get('name', 'Scale')).replace(' ', '_')}_composite" for s in scales])
        spss_lines.append(f"MEANS TABLES={dv_vars} BY CONDITION")
        spss_lines.append("  /CELLS=MEAN STDDEV COUNT.")
        spss_lines.append("")
        spss_lines.append("* --- SECTION 5: STATISTICAL TESTS ---.")

        if num_conditions == 2:
            spss_lines.append("* TWO-GROUP T-TEST.")
            spss_lines.append("* Levene's test included in output - check for equal variances.")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                spss_lines.append(f"T-TEST GROUPS=CONDITION")
                spss_lines.append(f"  /VARIABLES={name}_composite")
                spss_lines.append("  /MISSING=ANALYSIS.")
            spss_lines.append("")
            spss_lines.append("* NON-PARAMETRIC: Mann-Whitney U.")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                spss_lines.append(f"NPAR TESTS /M-W={name}_composite BY CONDITION(1 2).")
        elif num_conditions > 2:
            spss_lines.append("* ONE-WAY ANOVA with post-hoc tests.")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                spss_lines.append(f"ONEWAY {name}_composite BY CONDITION")
                spss_lines.append("  /STATISTICS DESCRIPTIVES HOMOGENEITY")
                spss_lines.append("  /POSTHOC=TUKEY ALPHA(0.05).")
            spss_lines.append("")
            spss_lines.append("* For effect size (eta-squared), use GLM.")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_")
                spss_lines.append(f"UNIANOVA {name}_composite BY CONDITION")
                spss_lines.append("  /PRINT=ETASQ.")

        spss_lines.append("")
        spss_lines.append("* ============================================================================.")
        return "\n".join(spss_lines)

    def _generate_stata_script(self, metadata: Dict[str, Any]) -> str:
        """Generate Stata script with explanatory comments."""
        conditions = metadata.get("conditions", [])
        scales = metadata.get('scales', [])
        num_conditions = len(conditions)

        stata_lines = [
            "/* ============================================================================",
            f"   COMPREHENSIVE STATA ANALYSIS SCRIPT",
            f"   Study: {metadata.get('study_title', 'Untitled Study')}",
            f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "   ============================================================================ */",
            "",
            "// --- SECTION 1: DATA LOADING ---",
            "clear all",
            'import delimited "Simulated.csv", clear',
            "describe",
            "",
            "// --- SECTION 2: DATA PREPARATION ---",
            "// Encode condition as numeric factor",
            "encode condition, generate(condition_num)",
            "",
            "// --- SECTION 3: DATA CLEANING ---",
            "// Keep only clean data (document criteria in methods!)",
            "keep if exclude_recommended == 0",
            "count",
            "tabulate condition",
            "",
            "// --- SECTION 4: COMPUTE COMPOSITES ---",
        ]

        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_").lower()
            num_items = int(s.get("num_items", 5) or 5)
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            items_str = " ".join(items)
            stata_lines.append(f"// {name}: {num_items} items")
            stata_lines.append(f"egen {name}_composite = rowmean({items_str})")

        stata_lines.append("")
        stata_lines.append("// --- SECTION 5: DESCRIPTIVES ---")
        for s in scales:
            name = str(s.get("name", "Scale")).replace(" ", "_").lower()
            stata_lines.append(f"tabstat {name}_composite, by(condition) statistics(n mean sd)")

        stata_lines.append("")
        stata_lines.append("// --- SECTION 6: STATISTICAL TESTS ---")

        if num_conditions == 2:
            stata_lines.append("// TWO-GROUP T-TEST")
            stata_lines.append("// Use 'unequal' for Welch's t-test (more robust)")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_").lower()
                stata_lines.append(f"ttest {name}_composite, by(condition) unequal")
            stata_lines.append("")
            stata_lines.append("// NON-PARAMETRIC: Wilcoxon rank-sum (Mann-Whitney)")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_").lower()
                stata_lines.append(f"ranksum {name}_composite, by(condition)")
        elif num_conditions > 2:
            stata_lines.append("// ONE-WAY ANOVA")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_").lower()
                stata_lines.append(f"oneway {name}_composite condition, tabulate")
            stata_lines.append("")
            stata_lines.append("// Post-hoc tests")
            stata_lines.append("// Run after anova command:")
            stata_lines.append("// pwcompare condition, mcompare(tukey) effects")
            stata_lines.append("")
            stata_lines.append("// NON-PARAMETRIC: Kruskal-Wallis")
            for s in scales:
                name = str(s.get("name", "Scale")).replace(" ", "_").lower()
                stata_lines.append(f"kwallis {name}_composite, by(condition)")

        stata_lines.append("")
        stata_lines.append("/* ============================================================================ */")
        return "\n".join(stata_lines)


class ComprehensiveInstructorReport:
    """
    Generates a detailed, comprehensive report for instructors ONLY.

    This report includes:
    - Statistical analyses with full interpretation (t-tests, ANOVA, effect sizes)
    - Data quality diagnostics (attention checks, completion times, exclusions)
    - Visualizations with automatic chart interpretation
    - Hypothesis testing based on preregistration information
    - Descriptive statistics by condition
    - Recommendations for student grading

    This is NOT shared with students - they should practice these analyses themselves.

    Report Sections:
    1. Data Quality Summary - Exclusions, attention, completion times
    2. Experimental Design Check - Condition balance, randomization
    3. Descriptive Statistics - By condition and overall
    4. Inferential Statistics - t-tests, ANOVA, effect sizes
    5. Visualization Gallery - Charts with interpretation
    6. Recommendations - Grading guidance for instructors

    Version: 2.2.1 - Enhanced interpretations and practical significance
    """

    # Report formatting constants
    SECTION_SEPARATOR = "=" * 80
    SUBSECTION_SEPARATOR = "-" * 80

    def __init__(self):
        self._warnings: List[str] = []
        self._insights: List[str] = []

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

        # =============================================================
        # STUDY OVERVIEW SECTION (NEW - All first page info)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## STUDY OVERVIEW")
        lines.append("-" * 80)
        lines.append("")

        # Study Title
        study_title = metadata.get('study_title', 'Untitled Study')
        lines.append(f"### {study_title}")
        lines.append("")

        # Team Information
        if team_info:
            team_name = team_info.get('team_name', '')
            team_members = team_info.get('team_members', '')
            if team_name:
                lines.append(f"**Team:** {team_name}")
            if team_members:
                # Format members nicely (handle newlines)
                members_formatted = team_members.replace('\n', ', ').replace(',,', ',').strip(', ')
                lines.append(f"**Team Members:** {members_formatted}")
            lines.append("")

        # Study Description / Abstract
        study_description = metadata.get('study_description', '')
        if study_description:
            lines.append("### Abstract / Study Description")
            lines.append("")
            lines.append(study_description)
            lines.append("")

        # Experimental Design
        lines.append("### Experimental Design")
        lines.append("")

        # Conditions
        conditions = metadata.get('conditions', [])
        if conditions:
            lines.append(f"**Conditions ({len(conditions)}):**")
            for i, cond in enumerate(conditions, 1):
                lines.append(f"  {i}. {cond}")
            lines.append("")

        # Factors (if factorial design)
        factors = metadata.get('factors', [])
        if factors:
            lines.append(f"**Factors ({len(factors)}):**")
            for factor in factors:
                factor_name = factor.get('name', 'Factor')
                levels = factor.get('levels', [])
                lines.append(f"  - {factor_name}: {', '.join(str(l) for l in levels)}")
            lines.append("")

        # Scales / DVs
        scales = metadata.get('scales', [])
        if scales:
            lines.append(f"**Dependent Variables / Scales ({len(scales)}):**")
            for scale in scales:
                scale_name = scale.get('name', 'Scale')
                scale_points = scale.get('scale_points', 7)
                num_items = scale.get('num_items', 1)
                lines.append(f"  - {scale_name} ({num_items} item{'s' if num_items > 1 else ''}, {scale_points}-point scale)")
            lines.append("")

        # Effect Sizes (hypotheses)
        effect_sizes = metadata.get('effect_sizes_configured', [])
        if effect_sizes:
            lines.append("**Hypothesized Effects:**")
            for effect in effect_sizes:
                var = effect.get('variable', '')
                factor = effect.get('factor', '')
                d = effect.get('cohens_d', 0)
                direction = effect.get('direction', 'higher')
                if d > 0:
                    lines.append(f"  - {var}: d = {d:.2f} ({direction} in treatment)")
            lines.append("")

        # Sample Size
        sample_size = metadata.get('sample_size', 0)
        lines.append(f"**Sample Size:** N = {sample_size}")
        lines.append("")

        # Generation Info
        lines.append("### Generation Details")
        lines.append("")
        lines.append(f"**Generated:** {metadata.get('generation_timestamp', datetime.now().isoformat())}")
        lines.append(f"**Run ID:** `{metadata.get('run_id', 'N/A')}`")
        lines.append(f"**Mode:** {metadata.get('simulation_mode', 'pilot').title()}")

        # Internal usage counter (for instructor tracking)
        usage_stats = metadata.get('usage_stats', {})
        total_simulations = usage_stats.get('total_simulations', 'N/A')
        lines.append(f"**Total Simulations Run (all time):** {total_simulations}")
        lines.append("")

        # === v1.2.5: DATA QUALITY ASSURANCE SECTION ===
        lines.append("")
        lines.append("-" * 80)
        lines.append("## DATA QUALITY ASSURANCE")
        lines.append("-" * 80)
        lines.append("")
        lines.append("### Automated Quality Checks")
        lines.append("")

        # Scale range verification
        scales = metadata.get("scales", [])
        if scales:
            lines.append("| Scale | Items | Expected Range | Actual Range | Status |")
            lines.append("|-------|-------|---------------|--------------|--------|")
            for scale in scales:
                s_name = str(scale.get("name", "Unknown")).strip().replace(" ", "_")
                n_items = scale.get("num_items", 5)
                s_min = scale.get("scale_min", 1)
                s_max = scale.get("scale_max", 7)
                cols = [c for c in df.columns if c.startswith(f"{s_name}_") and c[len(s_name)+1:].isdigit()]
                if cols:
                    actual_min = df[cols].min().min()
                    actual_max = df[cols].max().max()
                    status = "✅ Pass" if actual_min >= s_min and actual_max <= s_max else "⚠️ Review"
                    lines.append(f"| {s_name} | {len(cols)} | [{s_min}-{s_max}] | [{actual_min}-{actual_max}] | {status} |")
            lines.append("")

        # Response uniqueness check for open-ended
        oe_cols = [c for c in df.columns if df[c].dtype == object and c not in ['CONDITION', 'PARTICIPANT_ID', 'RUN_ID', 'SIMULATION_MODE', 'SIMULATION_SEED', 'Gender']]
        if oe_cols:
            lines.append("### Open-Ended Response Uniqueness")
            lines.append("")
            for col in oe_cols:
                responses = df[col].dropna().tolist()
                unique_responses = len(set(responses))
                total = len(responses)
                pct = (unique_responses / total * 100) if total > 0 else 0
                status = "✅" if pct >= 95 else ("⚠️" if pct >= 80 else "❌")
                lines.append(f"- {col}: {unique_responses}/{total} unique ({pct:.1f}%) {status}")
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
                lines.append(f"- **{fname}**: {', '.join(str(l) for l in levels)} ({len(levels)} levels)")
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

        dist_raw = metadata.get("persona_distribution", {}) or {}
        dist = _extract_persona_proportions(dist_raw)
        if dist:
            lines.append("| Persona | % | Expected N | Impact on Data |")
            lines.append("|---------|---|------------|----------------|")
            for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                share_f = _safe_float(share)
                share_val = share_f if share_f <= 1 else share_f / 100
                pct = share_val * 100
                count = int(round(n_total * share_val))
                impact = self._get_detailed_impact(persona.lower())
                lines.append(f"| {persona.title()} | {pct:.1f}% | ~{count} | {impact} |")
            lines.append("")

            # Estimate impact on results
            lines.append("### Estimated Impact on Results")
            lines.append("")
            acquiescent_share = _safe_float(dist.get("acquiescent", 0))
            skeptic_share = _safe_float(dist.get("skeptic", 0))
            if acquiescent_share > 0.1:
                lines.append(f"⚠️ High acquiescence ({acquiescent_share:.0%}) may inflate positive responses")
            if skeptic_share > 0.1:
                lines.append(f"⚠️ High skepticism ({skeptic_share:.0%}) may deflate responses")
            careless_share = _safe_float(dist.get("careless", 0)) + _safe_float(dist.get("random", 0))
            if careless_share > 0.1:
                lines.append(f"⚠️ Notable careless/random ({careless_share:.0%}) - verify exclusion criteria are working")
            lines.append("")

        # =============================================================
        # SECTION 6: OPEN-ENDED QUESTIONS SUMMARY (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 6. OPEN-ENDED QUESTIONS SUMMARY")
        lines.append("-" * 80)
        lines.append("")

        # Find open-ended columns (text columns that aren't standard)
        standard_cols = {
            'PARTICIPANT_ID', 'RUN_ID', 'SIMULATION_ID', 'CONDITION', 'Age', 'Gender',
            'Completion_Time_Seconds', 'Attention_Pass_Rate', 'Max_Straight_Line',
            'Flag_Speed', 'Flag_Attention', 'Flag_StraightLine', 'Exclude_Recommended'
        }
        open_ended_cols = []
        for col in df.columns:
            # Check if column contains string data and is not a standard column
            if col not in standard_cols and df[col].dtype == 'object':
                sample_val = df[col].iloc[0] if len(df) > 0 else ""
                if isinstance(sample_val, str) and len(sample_val) > 10:
                    open_ended_cols.append(col)

        if open_ended_cols:
            lines.append(f"**{len(open_ended_cols)} open-ended question(s) simulated:**")
            lines.append("")
            for col in open_ended_cols[:10]:  # Limit to first 10
                lines.append(f"### {col}")
                lines.append("")
                # Show sample responses by condition if possible
                if "CONDITION" in df.columns and len(conditions) > 0:
                    for cond in conditions[:3]:  # First 3 conditions
                        cond_data = df[df["CONDITION"] == cond][col]
                        if len(cond_data) > 0:
                            sample = cond_data.iloc[0]
                            if isinstance(sample, str) and len(sample) > 0:
                                truncated = sample[:200] + "..." if len(sample) > 200 else sample
                                lines.append(f"**{cond}:** *\"{truncated}\"*")
                                lines.append("")
                else:
                    # Just show first response
                    sample = df[col].iloc[0] if len(df) > 0 else ""
                    if isinstance(sample, str) and len(sample) > 0:
                        truncated = sample[:300] + "..." if len(sample) > 300 else sample
                        lines.append(f"*Sample:* \"{truncated}\"")
                        lines.append("")

                # Response length statistics
                lengths = df[col].apply(lambda x: len(str(x)) if x else 0)
                lines.append(f"- **Response length:** Mean={lengths.mean():.0f} chars, Range=[{lengths.min()}-{lengths.max()}]")
                lines.append("")
        else:
            lines.append("No open-ended questions were simulated in this dataset.")
            lines.append("")

        # =============================================================
        # SECTION 7: EFFECT SIZE QUALITY ASSESSMENT (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 7. EFFECT SIZE QUALITY ASSESSMENT")
        lines.append("-" * 80)
        lines.append("")

        configured_effects = metadata.get("effect_sizes_configured", [])
        observed_effects = metadata.get("effect_sizes_observed", [])

        if configured_effects:
            lines.append("### Configured vs Observed Effect Sizes")
            lines.append("")
            lines.append("| Variable | Configured d | Observed d | Match Quality |")
            lines.append("|----------|--------------|------------|---------------|")

            for cfg in configured_effects:
                var = cfg.get("variable", "Unknown")
                cfg_d = cfg.get("cohens_d", 0)
                # Find matching observed effect
                obs_d = None
                for obs in observed_effects:
                    if obs.get("variable") == var:
                        obs_d = obs.get("cohens_d", obs.get("d_observed"))
                        break

                if obs_d is not None:
                    diff = abs(cfg_d - obs_d)
                    if diff < 0.1:
                        quality = "✅ Excellent"
                    elif diff < 0.2:
                        quality = "✅ Good"
                    elif diff < 0.3:
                        quality = "⚠️ Acceptable"
                    else:
                        quality = "❌ Poor match"
                    lines.append(f"| {var} | {cfg_d:.3f} | {obs_d:.3f} | {quality} |")
                else:
                    lines.append(f"| {var} | {cfg_d:.3f} | N/A | ⚠️ Not computed |")
            lines.append("")

            # Overall assessment
            lines.append("### Quality Interpretation")
            lines.append("")
            lines.append("Effect sizes are calibrated from published meta-analyses and research findings.")
            lines.append("Good matches indicate the simulation faithfully reproduces expected effect magnitudes.")
            lines.append("")
        else:
            lines.append("No effect sizes were explicitly configured for this simulation.")
            lines.append("The simulation used domain-inferred defaults based on study context.")
            lines.append("")

        # =============================================================
        # SECTION 8: CONDITION BALANCE ANALYSIS (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 8. CONDITION BALANCE ANALYSIS")
        lines.append("-" * 80)
        lines.append("")

        if "CONDITION" in df.columns:
            cond_counts = df["CONDITION"].value_counts()
            total = len(df)
            expected_per_cond = total / len(conditions) if conditions else total

            lines.append("### Participant Distribution by Condition")
            lines.append("")
            lines.append("| Condition | N | % | Deviation from Expected |")
            lines.append("|-----------|---|---|------------------------|")

            max_deviation = 0
            for cond in conditions:
                count = cond_counts.get(cond, 0)
                pct = (count / total * 100) if total > 0 else 0
                deviation = abs(count - expected_per_cond)
                deviation_pct = (deviation / expected_per_cond * 100) if expected_per_cond > 0 else 0
                max_deviation = max(max_deviation, deviation_pct)

                if deviation_pct < 5:
                    dev_indicator = "✅"
                elif deviation_pct < 10:
                    dev_indicator = "⚠️"
                else:
                    dev_indicator = "❌"

                lines.append(f"| {cond} | {count} | {pct:.1f}% | {dev_indicator} {deviation_pct:.1f}% |")

            lines.append("")

            # Balance assessment
            lines.append("### Balance Assessment")
            lines.append("")
            if max_deviation < 5:
                lines.append("✅ **Excellent balance** - Conditions are evenly distributed.")
            elif max_deviation < 10:
                lines.append("✅ **Good balance** - Minor deviations within acceptable range.")
            elif max_deviation < 15:
                lines.append("⚠️ **Acceptable balance** - Some imbalance, but unlikely to affect analysis.")
            else:
                lines.append("❌ **Imbalanced** - Consider adjusting condition allocation for future runs.")
            lines.append("")

            # Check for empty conditions
            empty_conds = [c for c in conditions if cond_counts.get(c, 0) == 0]
            if empty_conds:
                lines.append(f"⚠️ **Warning:** {len(empty_conds)} condition(s) have no participants: {', '.join(empty_conds)}")
                lines.append("")
        else:
            lines.append("No CONDITION column found in data.")
            lines.append("")

        # =============================================================
        # SECTION 9: RECOMMENDATIONS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 9. INSTRUCTOR RECOMMENDATIONS")
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

        Enhanced pattern matching for common pre-registration formats.
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
        prereg_original = prereg_text  # Keep original case for better extraction

        # ========================================
        # ENHANCED HYPOTHESIS EXTRACTION
        # ========================================

        # Pattern 1: Explicit hypothesis labels (H1:, H2:, Hypothesis 1, etc.)
        explicit_hyp_patterns = [
            r'h\s*(\d+)[:\.\s-]+([^.!?\n]+[.!?]?)',  # H1: text, H1. text, H1 - text
            r'hypothesis\s*(\d*)[:\.\s-]+([^.!?\n]+[.!?]?)',  # Hypothesis 1: text
            r'prediction\s*(\d*)[:\.\s-]+([^.!?\n]+[.!?]?)',  # Prediction 1: text
        ]
        for pattern in explicit_hyp_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE)
            for match in matches:
                # match is (number, text) or just (text,) depending on pattern
                text = match[1] if len(match) > 1 else match[0]
                text = text.strip()
                if len(text) > 10 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 2: Prediction statements (we hypothesize/predict/expect/anticipate/propose)
        prediction_patterns = [
            r'we\s+(?:hypothesize|predict|expect|anticipate|propose)\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
            r'it\s+is\s+(?:hypothesized|predicted|expected|anticipated)\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
            r'our\s+(?:hypothesis|prediction)\s+is\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
        ]
        for pattern in prediction_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE)
            for m in matches:
                text = m.strip()
                if len(text) > 10 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 3: Effect direction statements (common in pre-regs)
        effect_patterns = [
            r'(?:participants|those|individuals)\s+(?:in|assigned\s+to|who\s+receive)\s+(?:the\s+)?(\w+\s+)?(?:condition|group)\s+will\s+([^.!?\n]+[.!?]?)',
            r'(?:the\s+)?(\w+\s+)?(?:condition|group|treatment)\s+will\s+(?:result\s+in|lead\s+to|show|demonstrate|have)\s+([^.!?\n]+[.!?]?)',
            r'(?:there\s+will\s+be\s+)?(?:a\s+)?(?:significant|positive|negative)?\s*(?:difference|effect|relationship|correlation)\s+(?:between|in)\s+([^.!?\n]+[.!?]?)',
            r'(\w+)\s+will\s+be\s+(?:higher|lower|greater|less|more|stronger|weaker)\s+(?:in|for|among)\s+([^.!?\n]+[.!?]?)',
        ]
        for pattern in effect_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE)
            for match in matches:
                # Combine match groups into a hypothesis statement
                if isinstance(match, tuple):
                    text = " ".join([m for m in match if m]).strip()
                else:
                    text = match.strip()
                if len(text) > 15 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 4: Look for hypotheses in sections/headers
        # Find text after "Hypotheses:" or "Predictions:" headers
        section_patterns = [
            r'hypothes[ie]s?[:\s]*\n+([^\n]+(?:\n[^\n#]+)*)',  # After "Hypotheses:" header
            r'predictions?[:\s]*\n+([^\n]+(?:\n[^\n#]+)*)',  # After "Predictions:" header
        ]
        for pattern in section_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE | re.MULTILINE)
            for section_text in matches:
                # Split by numbered items or bullet points
                items = re.split(r'\n\s*(?:\d+[\.\)]\s*|\*\s*|-\s*|•\s*)', section_text)
                for item in items:
                    item = item.strip()
                    if len(item) > 15 and item not in result["hypotheses"]:
                        result["hypotheses"].append(item)

        # Pattern 5: Bullet points or numbered lists that look like hypotheses
        list_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|\*\s*|-\s*|•\s*)([^.!?\n]*(?:will|should|expect|predict|hypothesize|higher|lower|greater|less|more|significant)[^.!?\n]*[.!?]?)', prereg_original, re.IGNORECASE)
        for item in list_items:
            item = item.strip()
            if len(item) > 15 and item not in result["hypotheses"]:
                result["hypotheses"].append(item)

        # Deduplicate and clean hypotheses
        seen = set()
        unique_hypotheses = []
        for hyp in result["hypotheses"]:
            hyp_clean = hyp.strip().lower()
            if hyp_clean not in seen and len(hyp) > 10:
                seen.add(hyp_clean)
                unique_hypotheses.append(hyp.strip())
        result["hypotheses"] = unique_hypotheses[:10]  # Cap at 10 hypotheses

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
        """Interpret Cohen's d effect size with detailed thresholds."""
        abs_d = abs(d)
        if abs_d < 0.1:
            return "negligible"
        elif abs_d < 0.2:
            return "very small"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        elif abs_d < 1.2:
            return "large"
        else:
            return "very large"

    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared effect size with detailed thresholds."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.02:
            return "very small"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        elif eta2 < 0.26:
            return "large"
        else:
            return "very large"

    def _interpret_omega_squared(self, omega2: float) -> str:
        """Interpret omega-squared effect size (less biased than eta-squared)."""
        if omega2 < 0.01:
            return "negligible"
        elif omega2 < 0.06:
            return "small"
        elif omega2 < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_r_squared(self, r2: float) -> str:
        """Interpret R-squared (coefficient of determination)."""
        if r2 < 0.02:
            return "negligible"
        elif r2 < 0.13:
            return "small"
        elif r2 < 0.26:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret Pearson correlation coefficient."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"

    def _get_practical_significance(self, effect_size: float, effect_type: str) -> str:
        """Generate practical significance statement based on effect size."""
        if effect_type == "cohens_d":
            interpretation = self._interpret_cohens_d(effect_size)
        elif effect_type == "eta_squared":
            interpretation = self._interpret_eta_squared(effect_size)
        elif effect_type == "r":
            interpretation = self._interpret_correlation(effect_size)
        else:
            interpretation = "unknown"

        significance_statements = {
            "negligible": "This effect is too small to have practical importance.",
            "very small": "This effect is minimal and unlikely to be noticeable in practice.",
            "small": "This effect is small but may be meaningful in some contexts.",
            "weak": "This relationship is weak and has limited practical value.",
            "moderate": "This effect is moderate and likely meaningful in practice.",
            "medium": "This effect is of medium magnitude and practically meaningful.",
            "strong": "This effect is strong and has clear practical implications.",
            "large": "This effect is large and has substantial practical importance.",
            "very strong": "This effect is very strong with major practical implications.",
            "very large": "This effect is very large and highly practically significant.",
        }
        return significance_statements.get(interpretation, "")

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
        Generate detailed executive summary with key takeaways and hypothesis evaluation.

        Args:
            all_scale_results: List of dicts containing scale analysis results
            prereg_text: Pre-registration text if available
            n_total: Total sample size
            conditions: List of condition names

        Returns:
            HTML string with executive summary
        """
        html = ["<h2>2. Executive Summary</h2>"]
        html.append("<div class='section-block' style='background:#f0f7ff;border-left:4px solid #3498db;'>")

        # Collect detailed findings
        sig_findings = []
        nonsig_findings = []
        largest_effect = None
        largest_effect_size = 0
        all_effects = []

        for result in all_scale_results:
            scale_name = result.get("scale_name", "Unknown Scale")
            stats = result.get("stats_results", {})
            chart_data = result.get("chart_data", {})
            is_sig = False
            effect_val = 0
            effect_type = None
            p_val = None

            # Check significance and get p-value
            if "t_test" in stats:
                p_val = stats["t_test"]["p_value"]
                is_sig = stats["t_test"]["significant"]
            elif "anova" in stats:
                p_val = stats["anova"]["p_value"]
                is_sig = stats["anova"]["significant"]

            # Get effect size
            if "cohens_d" in stats:
                effect_val = abs(stats["cohens_d"]["value"])
                effect_type = "d"
            elif "eta_squared" in stats:
                effect_val = stats["eta_squared"]["value"]
                effect_type = "η²"

            # Get means for condition comparison
            if chart_data:
                sorted_conds = sorted(chart_data.items(), key=lambda x: x[1][0], reverse=True)
                highest_cond = sorted_conds[0][0] if sorted_conds else None
                highest_mean = sorted_conds[0][1][0] if sorted_conds else None
                lowest_cond = sorted_conds[-1][0] if sorted_conds else None
                lowest_mean = sorted_conds[-1][1][0] if sorted_conds else None
            else:
                highest_cond = lowest_cond = highest_mean = lowest_mean = None

            finding_info = {
                "scale": scale_name,
                "effect_size": effect_val,
                "effect_type": effect_type,
                "p_value": p_val,
                "stats": stats,
                "highest_cond": highest_cond,
                "highest_mean": highest_mean,
                "lowest_cond": lowest_cond,
                "lowest_mean": lowest_mean,
                "significant": is_sig
            }

            if is_sig:
                sig_findings.append(finding_info)
                if effect_val > largest_effect_size:
                    largest_effect_size = effect_val
                    largest_effect = finding_info
            else:
                nonsig_findings.append(finding_info)

            all_effects.append(finding_info)

        # Generate summary text
        n_scales = len(all_scale_results)
        n_sig = len(sig_findings)
        n_conditions = len(conditions)

        html.append("<p style='font-size:14px;line-height:1.8;margin:0;'>")

        # Opening - Study Overview
        html.append("<strong style='color:#2c3e50;font-size:15px;'>Study Overview:</strong><br>")
        html.append(
            f"This simulation generated data for <strong>{n_total} participants</strong> randomly assigned to "
            f"<strong>{n_conditions} experimental condition{'s' if n_conditions > 1 else ''}</strong>: "
            f"{', '.join(str(c) for c in conditions)}. "
        )
        html.append(f"The analysis examined {n_scales} dependent variable{'s' if n_scales > 1 else ''}.")

        # Main Findings
        html.append("<br><br><strong style='color:#2c3e50;font-size:15px;'>Key Results:</strong><br>")

        if n_sig > 0:
            html.append(
                f"<span style='color:#27ae60;'>✓</span> <strong>{n_sig} of {n_scales} dependent variable{'s' if n_sig > 1 else ''} showed statistically significant differences</strong> between conditions:<br>"
            )

            for i, finding in enumerate(sig_findings):
                effect_desc = "large" if (finding["effect_type"] == "d" and finding["effect_size"] >= 0.8) or \
                                        (finding["effect_type"] == "η²" and finding["effect_size"] >= 0.14) else \
                             "medium" if (finding["effect_type"] == "d" and finding["effect_size"] >= 0.5) or \
                                        (finding["effect_type"] == "η²" and finding["effect_size"] >= 0.06) else "small"

                html.append(f"&nbsp;&nbsp;• <strong>{finding['scale']}</strong>: ")
                if finding['highest_cond'] and finding['lowest_cond']:
                    html.append(
                        f"{finding['highest_cond']} (M = {finding['highest_mean']:.2f}) > {finding['lowest_cond']} (M = {finding['lowest_mean']:.2f}), "
                    )
                html.append(f"p = {finding['p_value']:.4f}, {effect_desc} effect ({finding['effect_type']} = {finding['effect_size']:.2f})<br>")

            if largest_effect:
                html.append(
                    f"<br>The <strong>strongest effect</strong> was observed for <strong>{largest_effect['scale']}</strong>. "
                )
        else:
            html.append(
                f"<span style='color:#e74c3c;'>✗</span> <strong>No statistically significant differences</strong> were found between conditions "
                f"on any of the {n_scales} dependent variable{'s' if n_scales > 1 else ''}. "
            )
            # Show the closest to significance
            if all_effects:
                closest = min(all_effects, key=lambda x: x['p_value'] if x['p_value'] else 1)
                if closest['p_value']:
                    html.append(f"The closest to significance was <strong>{closest['scale']}</strong> (p = {closest['p_value']:.4f}).")

        # Pre-registration Hypothesis Evaluation
        if prereg_text:
            prereg_info = self._parse_prereg_hypotheses(prereg_text)
            hypotheses = prereg_info.get("hypotheses", [])

            html.append("<br><br><strong style='color:#2c3e50;font-size:15px;'>Pre-Registration Evaluation:</strong><br>")

            if hypotheses:
                html.append(f"The pre-registration document specified {len(hypotheses)} hypothesis/hypotheses. Based on the simulated results:<br>")

                # More sophisticated matching
                for i, h in enumerate(hypotheses):
                    h_text = h.get("text", "")
                    h_matched = False

                    # Check if any significant finding relates to this hypothesis
                    for finding in sig_findings:
                        scale_lower = finding["scale"].lower()
                        h_lower = h_text.lower()
                        if any(word in h_lower for word in scale_lower.split() if len(word) > 3):
                            html.append(f"&nbsp;&nbsp;<span style='color:#27ae60;'>✓</span> <em>\"{h_text[:80]}{'...' if len(h_text) > 80 else ''}\"</em> — <strong>Supported</strong><br>")
                            h_matched = True
                            break

                    if not h_matched:
                        html.append(f"&nbsp;&nbsp;<span style='color:#e74c3c;'>✗</span> <em>\"{h_text[:80]}{'...' if len(h_text) > 80 else ''}\"</em> — Not supported<br>")
            else:
                html.append("No specific hypotheses were extracted from the pre-registration document. Review the document manually to compare predictions with results.")

        # Practical Implications
        html.append("<br><strong style='color:#2c3e50;font-size:15px;'>Interpretation:</strong><br>")
        if n_sig > 0:
            html.append(
                f"These results suggest that the experimental manipulation had a measurable effect on participant responses. "
                f"Students should examine the pattern of means to understand the direction of effects and consider whether "
                f"these findings align with theoretical predictions."
            )
        else:
            html.append(
                f"The lack of significant findings could indicate that: (1) the manipulation was not strong enough, "
                f"(2) the sample size was insufficient to detect small effects, or (3) there is genuinely no effect of the "
                f"experimental conditions on the measured outcomes. Students should consider these possibilities in their discussion."
            )

        # Note about simulation
        html.append("<br><br><em style='color:#7f8c8d;font-size:12px;'>")
        html.append("Note: These are simulated results generated for pedagogical purposes. The patterns reflect the simulation parameters chosen, ")
        html.append("and actual experimental results will depend on real participant responses.</em>")
        html.append("</p></div>")

        return "\n".join(html)

    def _generate_stat_test_interpretation(
        self,
        test_type: str,
        stats_results: Dict[str, Any],
        chart_data: Dict[str, Tuple[float, float]],
        scale_name: str
    ) -> str:
        """Generate plain-language interpretation for a statistical test."""
        conditions = list(chart_data.keys())
        means = {c: chart_data[c][0] for c in conditions}
        sorted_conds = sorted(means.items(), key=lambda x: x[1], reverse=True)
        highest = sorted_conds[0]
        lowest = sorted_conds[-1]

        if test_type == "t_test" and "t_test" in stats_results:
            t = stats_results["t_test"]
            if t["significant"]:
                return f"The t-test indicates a statistically significant difference between conditions. Participants in the <strong>{highest[0]}</strong> condition scored higher (M = {highest[1]:.2f}) than those in <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f})."
            else:
                return f"The t-test did not find a statistically significant difference between the two conditions, suggesting that {scale_name} scores were similar regardless of experimental condition."

        elif test_type == "anova" and "anova" in stats_results:
            a = stats_results["anova"]
            if a["significant"]:
                return f"The ANOVA reveals significant variation in {scale_name} across conditions. <strong>{highest[0]}</strong> showed the highest scores while <strong>{lowest[0]}</strong> showed the lowest. Post-hoc comparisons (below) identify which specific pairs differ."
            else:
                return f"The ANOVA did not detect significant differences in {scale_name} across the {len(conditions)} conditions, suggesting the experimental manipulation may not have affected this outcome."

        elif test_type == "effect_size":
            if "cohens_d" in stats_results:
                d = stats_results["cohens_d"]
                interpretation = d["interpretation"]
                if abs(d["value"]) >= 0.8:
                    return f"The effect size is <strong>large</strong> (d = {d['value']:.2f}), indicating a substantial and practically meaningful difference between conditions."
                elif abs(d["value"]) >= 0.5:
                    return f"The effect size is <strong>medium</strong> (d = {d['value']:.2f}), suggesting a moderate and potentially meaningful difference."
                elif abs(d["value"]) >= 0.2:
                    return f"The effect size is <strong>small</strong> (d = {d['value']:.2f}), indicating a modest difference that may have limited practical significance."
                else:
                    return f"The effect size is <strong>negligible</strong> (d = {d['value']:.2f}), suggesting minimal practical difference between conditions."
            elif "eta_squared" in stats_results:
                e = stats_results["eta_squared"]
                return f"The effect size (η² = {e['value']:.3f}) indicates that {e['value']*100:.1f}% of variance in {scale_name} is explained by condition assignment ({e['interpretation']} effect)."

        elif test_type == "regression":
            return f"The regression analysis examines condition effects while controlling for demographic variables, providing a more precise estimate of the experimental effect."

        return ""

    def _get_prereg_requested_analyses(self, prereg_text: Optional[str]) -> Dict[str, bool]:
        """Parse pre-registration to determine which analyses were requested."""
        requested = {
            "t_test": False,
            "anova": False,
            "regression": False,
            "factorial": False,
            "chi_squared": False,
            "mann_whitney": False,
            "correlation": False
        }

        if not prereg_text:
            return requested

        text_lower = prereg_text.lower()

        # Check for specific analysis mentions
        if any(term in text_lower for term in ["t-test", "t test", "independent samples", "two-sample"]):
            requested["t_test"] = True
        if any(term in text_lower for term in ["anova", "analysis of variance", "f-test", "between-subjects"]):
            requested["anova"] = True
        if any(term in text_lower for term in ["regression", "linear model", "glm", "control variable", "covariate"]):
            requested["regression"] = True
        if any(term in text_lower for term in ["factorial", "interaction", "2x2", "2x3", "3x3", "two-way", "main effect"]):
            requested["factorial"] = True
        if any(term in text_lower for term in ["chi-square", "chi square", "χ²", "contingency"]):
            requested["chi_squared"] = True
        if any(term in text_lower for term in ["mann-whitney", "wilcoxon", "non-parametric", "nonparametric"]):
            requested["mann_whitney"] = True
        if any(term in text_lower for term in ["correlation", "pearson", "spearman"]):
            requested["correlation"] = True

        return requested

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

        # CSS styles for the report (v1.3.4: improved layout with TOC and sections)
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            * { box-sizing: border-box; }
            body { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; color: #1a1a2e; line-height: 1.6; }
            .page-wrapper { max-width: 1200px; margin: 0 auto; display: flex; gap: 30px; align-items: flex-start; }
            .toc-sidebar { position: sticky; top: 20px; width: 220px; min-width: 220px; background: white; border-radius: 10px; padding: 20px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); font-size: 0.82em; max-height: calc(100vh - 40px); overflow-y: auto; }
            .toc-sidebar h3 { color: #1e3a5f; font-size: 0.95em; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #2563eb; }
            .toc-sidebar a { display: block; color: #475569; text-decoration: none; padding: 5px 8px; border-radius: 4px; margin-bottom: 2px; transition: all 0.15s; }
            .toc-sidebar a:hover { background: #eef2ff; color: #2563eb; }
            .toc-sidebar a.toc-h2 { font-weight: 600; color: #1e3a5f; }
            .toc-sidebar a.toc-h3 { padding-left: 18px; font-size: 0.92em; }
            .report-container { flex: 1; min-width: 0; background: white; padding: 40px 45px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            h1 { color: #1a1a2e; border-bottom: 3px solid #2563eb; padding-bottom: 12px; font-weight: 700; font-size: 1.8em; }
            h2 { color: #1e3a5f; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 40px; font-weight: 600; font-size: 1.4em; scroll-margin-top: 20px; }
            h3 { color: #475569; font-weight: 600; font-size: 1.15em; margin-top: 24px; scroll-margin-top: 20px; }
            h4 { color: #64748b; font-weight: 500; font-size: 1.05em; margin-top: 18px; }
            p { margin: 8px 0; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }
            th, td { border: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; }
            th { background-color: #1e3a5f; color: white; font-weight: 500; letter-spacing: 0.02em; }
            tr:nth-child(even) { background-color: #f8fafc; }
            tr:hover { background-color: #eef2ff; }
            .section-block { background: #fafbfc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 24px 28px; margin: 20px 0; }
            .stat-box { background: #f0f7ff; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #2563eb; }
            .warning-box { background: #fffbeb; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #f59e0b; }
            .success-box { background: #f0fdf4; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #22c55e; }
            .error-box { background: #fef2f2; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #ef4444; }
            .chart-container { text-align: center; margin: 20px 0; }
            .chart-container img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
            .confidential { background: #dc2626; color: white; padding: 6px 18px; border-radius: 4px; display: inline-block; margin-bottom: 20px; font-weight: 600; letter-spacing: 0.05em; font-size: 0.85em; }
            .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric-card { background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); color: white; padding: 22px 16px; border-radius: 10px; text-align: center; box-shadow: 0 3px 12px rgba(37,99,235,0.15); }
            .metric-value { font-size: 2em; font-weight: 700; letter-spacing: -0.02em; }
            .metric-label { font-size: 0.85em; opacity: 0.9; margin-top: 4px; font-weight: 400; }
            .back-to-top { display: inline-block; margin-top: 10px; font-size: 0.8em; color: #94a3b8; text-decoration: none; }
            .back-to-top:hover { color: #2563eb; }
            code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #475569; }
            .sig { color: #16a34a; font-weight: 600; }
            .nonsig { color: #94a3b8; }
            ol, ul { padding-left: 24px; }
            li { margin-bottom: 4px; }
            em { color: #64748b; }
            @media print {
                body { background: white; padding: 0; }
                .page-wrapper { display: block; }
                .toc-sidebar { display: none; }
                .report-container { box-shadow: none; padding: 20px; }
                .metric-card { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
                .confidential { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
                th { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            }
            @media (max-width: 900px) {
                .page-wrapper { flex-direction: column; }
                .toc-sidebar { position: static; width: 100%; min-width: 0; }
            }
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
            "<div class='page-wrapper'>",
            # Table of Contents sidebar
            "<nav class='toc-sidebar'>",
            "<h3>Contents</h3>",
            "<a href='#top' class='toc-h2'>Report Header</a>",
            "<a href='#study-overview' class='toc-h2'>Study Overview</a>",
            "<a href='#sample-overview' class='toc-h2'>1. Sample Overview</a>",
            "<a href='#exec-summary' class='toc-h2'>2. Executive Summary</a>",
            "<a href='#statistical-analysis' class='toc-h2'>3. Statistical Analysis</a>",
            "<a href='#persona-analysis' class='toc-h2'>4. Persona &amp; Response</a>",
            "<a href='#categorical-analysis' class='toc-h2'>5. Categorical Analysis</a>",
            "<a href='#effect-verification' class='toc-h2'>6. Effect Verification</a>",
            "<a href='#data-quality' class='toc-h2'>7. Data Quality</a>",
            "<a href='#methodology' class='toc-h2'>8. Methodology</a>",
            "</nav>",
            # Main report content
            "<div class='report-container'>",
            "<a id='top'></a>",
        ]

        # Header
        html_parts.append("<span class='confidential'>CONFIDENTIAL &mdash; INSTRUCTOR ONLY</span>")
        html_parts.append("<h1>Comprehensive Simulation &amp; Statistical Report</h1>")
        html_parts.append(f"<p style='color:#64748b;margin-top:-8px;font-size:1.05em;'>Behavioral Experiment Simulation Tool v{__version__}</p>")

        # =============================================================
        # STUDY OVERVIEW SECTION (All first page info)
        # =============================================================
        html_parts.append("<a id='study-overview'></a>")
        html_parts.append("<div class='section-block'>")
        html_parts.append("<h2>Study Overview</h2>")

        # Study Title
        study_title = metadata.get('study_title', 'Untitled Study')
        html_parts.append(f"<h3>{study_title}</h3>")

        # Team Information
        if team_info:
            team_name = team_info.get('team_name', '')
            team_members = team_info.get('team_members', '')
            if team_name:
                html_parts.append(f"<p><strong>Team:</strong> {team_name}</p>")
            if team_members:
                members_formatted = team_members.replace('\n', ', ').replace(',,', ',').strip(', ')
                html_parts.append(f"<p><strong>Team Members:</strong> {members_formatted}</p>")

        # Study Description / Abstract
        study_description = metadata.get('study_description', '')
        if study_description:
            html_parts.append(f"<p><strong>Abstract:</strong> {study_description}</p>")

        # Experimental Design (inside same section-block)
        html_parts.append("<h3>Experimental Design</h3>")

        # Conditions
        conditions = metadata.get('conditions', [])
        if conditions:
            html_parts.append(f"<p><strong>Conditions ({len(conditions)}):</strong></p>")
            html_parts.append("<ul>")
            for cond in conditions:
                html_parts.append(f"<li>{cond}</li>")
            html_parts.append("</ul>")

        # Factors
        factors = metadata.get('factors', [])
        if factors:
            html_parts.append(f"<p><strong>Factors ({len(factors)}):</strong></p>")
            html_parts.append("<ul>")
            for factor in factors:
                factor_name = factor.get('name', 'Factor')
                levels = factor.get('levels', [])
                html_parts.append(f"<li>{factor_name}: {', '.join(str(l) for l in levels)}</li>")
            html_parts.append("</ul>")

        # Scales / DVs
        scales_meta = metadata.get('scales', [])
        if scales_meta:
            html_parts.append(f"<p><strong>Dependent Variables ({len(scales_meta)}):</strong></p>")
            html_parts.append("<ul>")
            for scale in scales_meta:
                scale_name = scale.get('name', 'Scale')
                scale_points = scale.get('scale_points', 7)
                num_items = scale.get('num_items', 1)
                html_parts.append(f"<li>{scale_name} ({num_items} item{'s' if num_items > 1 else ''}, {scale_points}-point)</li>")
            html_parts.append("</ul>")

        # Effect Sizes
        effect_sizes = metadata.get('effect_sizes_configured', [])
        if effect_sizes and any(e.get('cohens_d', 0) > 0 for e in effect_sizes):
            html_parts.append("<p><strong>Hypothesized Effects:</strong></p>")
            html_parts.append("<ul>")
            for effect in effect_sizes:
                d = effect.get('cohens_d', 0)
                if d > 0:
                    var = effect.get('variable', '')
                    direction = effect.get('direction', 'higher')
                    html_parts.append(f"<li>{var}: d = {d:.2f} ({direction} in treatment)</li>")
            html_parts.append("</ul>")

        # ── Study Context / Domain ─────────────────────────────────────
        study_context = metadata.get("study_context", {})
        detected_domains = metadata.get("detected_domains", [])
        if study_context or detected_domains:
            html_parts.append("<h3>Research Context</h3>")
            _domain = study_context.get("study_domain", study_context.get("domain", ""))
            if _domain:
                html_parts.append(f"<p><strong>Research Domain:</strong> {_domain.title()}</p>")
            if detected_domains:
                html_parts.append(f"<p><strong>Detected Topic Areas:</strong> {', '.join(detected_domains[:10])}</p>")
            _source = study_context.get("source", "")
            if _source:
                _source_label = "Conversational Builder" if "builder" in _source else "QSF Upload"
                html_parts.append(f"<p><strong>Input Method:</strong> {_source_label}</p>")
            _participant_chars = study_context.get("participant_characteristics", "")
            if _participant_chars:
                html_parts.append(f"<p><strong>Target Participants:</strong> {_participant_chars}</p>")
            _persona_domains = study_context.get("persona_domains", [])
            if _persona_domains:
                html_parts.append(f"<p><strong>Persona Domains Activated:</strong> {', '.join(d.replace('_', ' ').title() for d in _persona_domains)}</p>")
        # Open-ended questions summary
        oe_questions = metadata.get("open_ended_questions", [])
        if oe_questions:
            html_parts.append(f"<h3>Open-Ended Questions ({len(oe_questions)})</h3>")
            html_parts.append("<ul>")
            for oe in oe_questions:
                q_text = oe.get("question_text", oe.get("name", "")) if isinstance(oe, dict) else str(oe)
                var_name = oe.get("variable_name", "") if isinstance(oe, dict) else ""
                if q_text:
                    _var_tag = f" <code>({var_name})</code>" if var_name else ""
                    html_parts.append(f"<li>{q_text[:120]}{_var_tag}</li>")
            html_parts.append("</ul>")

        # Generation Details (still inside section-block)
        html_parts.append("<h3 style='margin-top:28px;padding-top:16px;border-top:1px solid #e2e8f0;'>Generation Details</h3>")
        html_parts.append(f"<p><strong>Generated:</strong> {metadata.get('generation_timestamp', datetime.now().isoformat())}</p>")
        html_parts.append(f"<p><strong>Run ID:</strong> <code>{metadata.get('run_id', 'N/A')}</code></p>")
        html_parts.append(f"<p><strong>Mode:</strong> {metadata.get('simulation_mode', 'pilot').title()}</p>")
        html_parts.append(f"<p><strong>Seed:</strong> <code>{metadata.get('seed', 'N/A')}</code></p>")
        html_parts.append(f"<p><strong>App Version:</strong> {__version__}</p>")

        # Internal usage counter (for instructor tracking)
        usage_stats = metadata.get('usage_stats', {})
        total_simulations = usage_stats.get('total_simulations', 'N/A')
        html_parts.append(f"<p><strong>Total Simulations Run (all time):</strong> {total_simulations}</p>")
        html_parts.append("</div>")  # close Study Overview section-block
        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

        # Summary metrics
        n_total = len(df)
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
        n_clean = n_total - n_excluded
        exclusion_rate = (n_excluded / n_total * 100) if n_total > 0 else 0
        conditions = metadata.get("conditions", [])

        html_parts.append("<a id='sample-overview'></a>")
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

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

        # Clean data for analysis
        df_clean = df[df["Exclude_Recommended"] == 0] if "Exclude_Recommended" in df.columns else df

        # v1.3.4: Executive Summary placeholder — will be computed after DV analysis and inserted here
        html_parts.append("<a id='exec-summary'></a>")
        exec_summary_index = len(html_parts)
        html_parts.append("<!-- EXEC_SUMMARY_PLACEHOLDER -->")

        # DV Analysis with statistical tests
        scales = metadata.get("scales", [])
        html_parts.append("<a id='statistical-analysis'></a>")
        html_parts.append("<h2>3. Statistical Analysis by DV</h2>")

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

                # Add interpretation for descriptive statistics
                if chart_data:
                    means = [(c, m[0]) for c, m in chart_data.items()]
                    if len(means) >= 2:
                        sorted_means = sorted(means, key=lambda x: x[1], reverse=True)
                        highest = sorted_means[0]
                        lowest = sorted_means[-1]
                        diff = highest[1] - lowest[1]
                        grand_mean = sum(m[1] for m in means) / len(means)

                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:12px;border-radius:6px;margin:10px 0;border-left:3px solid #3498db;'>")
                        html_parts.append(f"<strong>Summary:</strong> The <em>{highest[0]}</em> condition showed the highest mean ({highest[1]:.2f}), while <em>{lowest[0]}</em> showed the lowest ({lowest[1]:.2f}). ")
                        html_parts.append(f"The difference between highest and lowest conditions is {diff:.2f} scale points. ")
                        html_parts.append(f"The grand mean across conditions is {grand_mean:.2f}.")
                        html_parts.append("</div>")

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

                    # Determine which analyses were requested in pre-registration
                    prereg_analyses = self._get_prereg_requested_analyses(prereg_text)

                    html_parts.append("<h4>Statistical Tests</h4>")

                    # Two-group tests (t-test)
                    if "t_test" in stats_results:
                        t = stats_results["t_test"]
                        sig_class = "sig" if t["significant"] else "nonsig"
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("t_test") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Independent Samples t-test:</strong>{prereg_badge}<br>")
                        html_parts.append(f"t = {t['statistic']:.3f}, <span class='{sig_class}'>p = {t['p_value']:.4f}</span>")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("t_test", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Effect size for two groups
                    if "cohens_d" in stats_results:
                        d = stats_results["cohens_d"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Effect Size (Cohen's d):</strong> {d['value']:.3f}")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("effect_size", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # ANOVA for 3+ groups
                    if "anova" in stats_results:
                        a = stats_results["anova"]
                        sig_class = "sig" if a["significant"] else "nonsig"
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("anova") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>One-way ANOVA:</strong>{prereg_badge}<br>")
                        html_parts.append(f"F = {a['f_statistic']:.3f}, <span class='{sig_class}'>p = {a['p_value']:.4f}</span>")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("anova", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Effect size for ANOVA
                    if "eta_squared" in stats_results:
                        e = stats_results["eta_squared"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Effect Size (η²):</strong> {e['value']:.4f}")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("effect_size", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Assumption checks (simplified, no warnings)
                    assumption_notes = []
                    if "levene_test" in stats_results:
                        lev = stats_results["levene_test"]
                        if lev["homogeneous"]:
                            assumption_notes.append(f"Variance homogeneity: ✓ Met (Levene's p = {lev['p_value']:.3f})")
                        else:
                            assumption_notes.append(f"Variance homogeneity: Welch's correction applied (Levene's p = {lev['p_value']:.3f})")

                    if "normality_test" in stats_results:
                        sw = stats_results["normality_test"]
                        if sw["normal"]:
                            assumption_notes.append(f"Normality: ✓ Met (p = {sw['p_value']:.3f})")
                        else:
                            assumption_notes.append(f"Normality: Non-parametric tests also reported (p = {sw['p_value']:.3f})")

                    if assumption_notes:
                        html_parts.append("<div class='stat-box' style='background:#f8f9fa;'>")
                        html_parts.append("<strong>Assumption Checks:</strong> " + " | ".join(assumption_notes))
                        html_parts.append("</div>")

                    # Pairwise comparisons for 3+ groups
                    if "pairwise_comparisons" in stats_results and len(stats_results["pairwise_comparisons"]) > 0:
                        html_parts.append("<h4>Pairwise Comparisons</h4>")
                        html_parts.append("<table><tr><th>Comparison</th><th>t</th><th>p</th><th>Cohen's d</th><th>Significant</th></tr>")

                        sig_pairs = []
                        largest_effect_pair = None
                        largest_d = 0

                        for comp in stats_results["pairwise_comparisons"]:
                            sig_class = "sig" if comp["significant"] else "nonsig"
                            sig_text = "Yes" if comp["significant"] else "No"
                            html_parts.append(f"<tr><td>{comp['comparison']}</td><td>{comp['t_stat']:.3f}</td><td class='{sig_class}'>{comp['p_value']:.4f}</td><td>{comp['cohens_d']:.3f}</td><td class='{sig_class}'>{sig_text}</td></tr>")

                            if comp["significant"]:
                                sig_pairs.append(comp['comparison'])
                            if abs(comp['cohens_d']) > largest_d:
                                largest_d = abs(comp['cohens_d'])
                                largest_effect_pair = comp

                        html_parts.append("</table>")

                        # Interpretation instead of warning
                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                        html_parts.append("<strong>Key Finding:</strong> ")
                        if sig_pairs:
                            html_parts.append(f"{len(sig_pairs)} of {len(stats_results['pairwise_comparisons'])} pairwise comparisons reached statistical significance. ")
                            if largest_effect_pair:
                                html_parts.append(f"The largest effect was between {largest_effect_pair['comparison']} (d = {largest_effect_pair['cohens_d']:.2f}).")
                        else:
                            html_parts.append("No pairwise comparisons reached statistical significance, suggesting the overall ANOVA effect may be driven by subtle differences across multiple groups rather than any single pair.")
                        html_parts.append("</div>")

                        # Forest plot for effect sizes
                        forest_img = self._create_effect_size_forest_plot(
                            stats_results["pairwise_comparisons"],
                            f"{scale_name}: Effect Sizes (Cohen's d)"
                        )
                        if forest_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{forest_img}' alt='Forest plot'>")
                            html_parts.append("</div>")

                    # Regression analysis with control variables (only show if successful)
                    prereg_info = self._parse_prereg_hypotheses(prereg_text) if prereg_text else {}
                    prereg_controls = prereg_info.get("control_variables", [])

                    reg_results = self._run_regression_analysis(
                        df_analysis, "_composite", "CONDITION",
                        include_controls=True,
                        prereg_controls=prereg_controls
                    )

                    # Only show regression if it worked (no warnings for failures)
                    if "error" not in reg_results and "model_fit" in reg_results:
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("regression") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append(f"<h4>Regression Analysis (with Controls){prereg_badge}</h4>")

                        controls_used = reg_results.get("controls_included", [])
                        if controls_used:
                            html_parts.append(f"<div class='stat-box'><strong>Control variables included:</strong> {', '.join(controls_used)}</div>")

                        html_parts.append("<div class='stat-box'>")
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

                        # Regression interpretation
                        html_parts.append("<br><em style='color:#666;'>")
                        html_parts.append(f"The regression model explains {fit['r_squared']*100:.1f}% of variance in {scale_name}. ")
                        if "f_test" in reg_results and reg_results["f_test"]["significant"]:
                            html_parts.append("The overall model is statistically significant.")
                        html_parts.append("</em>")
                        html_parts.append("</div>")

                    # Factorial ANOVA for 2x2+ designs (only show if successful)
                    factors = metadata.get("factors", [])
                    if len(factors) >= 2 and len(conditions) >= 4:
                        factorial_results = self._run_factorial_anova(df_analysis, "_composite", factors, "CONDITION")

                        # Only show factorial ANOVA if it worked (no warnings for failures)
                        if ("error" not in factorial_results or factorial_results.get("single_factor")) and \
                           "main_effect_1" in factorial_results and "main_effect_2" in factorial_results:
                            prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("factorial") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                            html_parts.append(f"<h4>Factorial ANOVA (Main Effects & Interaction){prereg_badge}</h4>")
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

                            # Interpretation with key finding box
                            html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                            html_parts.append("<strong>Key Finding:</strong> ")
                            findings = []
                            if me1["significant"]:
                                findings.append(f"significant main effect of <strong>{me1['factor']}</strong> ({me1['interpretation']} effect)")
                            if me2["significant"]:
                                findings.append(f"significant main effect of <strong>{me2['factor']}</strong> ({me2['interpretation']} effect)")
                            if "interaction" in factorial_results and factorial_results["interaction"]["significant"]:
                                findings.append(f"<strong>significant interaction</strong> between factors ({factorial_results['interaction']['interpretation']} effect)")

                            if findings:
                                html_parts.append("The factorial analysis revealed " + ", ".join(findings) + ".")
                            else:
                                html_parts.append("No significant main effects or interaction were detected in this factorial design.")
                            html_parts.append("</div>")

                            # Cell means table
                            if "cell_statistics" in factorial_results:
                                html_parts.append("<strong>Cell Means:</strong>")
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

                # Track this scale's results for executive summary
                all_scale_results.append({
                    "scale_name": scale_name,
                    "chart_data": chart_data,
                    "stats_results": stats_results
                })

        # Chi-squared test for categorical associations
        if "CONDITION" in df_clean.columns and "Gender" in df_clean.columns:
            html_parts.append("<a id='persona-analysis'></a>")
            html_parts.append("<h2>4. Persona Analysis &amp; Response Styles</h2>")

            # ── Persona Distribution ──────────────────────────────────────
            persona_dist_raw = metadata.get("persona_distribution", {}) or {}
            persona_dist = _extract_persona_proportions(persona_dist_raw)
            personas_used = metadata.get("personas_used", [])

            if persona_dist:
                html_parts.append("<div class='stat-box'>")
                html_parts.append("<h3>Simulated Participant Personas</h3>")
                html_parts.append("<p>Each simulated participant was assigned a response style persona based on decades of survey methodology research. "
                                  "These personas reflect patterns observed in real survey respondents, producing data with realistic "
                                  "statistical properties (varying attention, response styles, and biases).</p>")
                html_parts.append("</div>")

                # Build persona info lookup
                _persona_info = {
                    "engaged responder": ("Thoughtful, attentive participant", "High attention, full scale range, consistent with attitudes"),
                    "engaged": ("Thoughtful, attentive participant", "High attention, full scale range, consistent with attitudes"),
                    "satisficer": ("Minimally effortful responder", "Gravitates to middle options, faster completion, may skip reading"),
                    "extreme responder": ("Uses scale endpoints frequently", "Strong opinions, uses 1s and 7s, high within-scale variance"),
                    "extreme": ("Uses scale endpoints frequently", "Strong opinions, uses 1s and 7s, high within-scale variance"),
                    "acquiescent": ("Agreement bias responder", "Tends to agree regardless of content, inflated positive responses"),
                    "skeptic": ("Disagreement bias responder", "Tends to disagree or rate negatively, lower mean responses"),
                    "random": ("Inconsistent, inattentive responder", "High variance, fails attention checks, no clear pattern"),
                    "careless": ("Pattern-based responder", "Straight-lining, very fast completion, flagged for exclusion"),
                    "careful responder": ("Highly attentive, methodical", "Longer completion, passes all attention checks, low variance"),
                    "moderate responder": ("Avoids extreme responses", "Uses middle portion of scale, rarely selects endpoints"),
                }

                html_parts.append("<h4>Persona Distribution</h4>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Persona Type</th><th>Description</th><th>Behavioral Pattern</th><th>Share</th><th>~Count</th></tr>")
                _n_total_pers = metadata.get("sample_size", n_total)
                for persona, share in sorted(persona_dist.items(), key=lambda x: -_safe_float(x[1])):
                    share_f = _safe_float(share)
                    pct = share_f * 100 if share_f <= 1 else share_f
                    share_val = share_f if share_f <= 1 else share_f / 100
                    count = int(round(_n_total_pers * share_val))
                    pkey = persona.lower()
                    info = _persona_info.get(pkey, ("Standard response pattern", "Typical survey behavior"))
                    html_parts.append(
                        f"<tr><td><strong>{persona.title()}</strong></td>"
                        f"<td>{info[0]}</td>"
                        f"<td><em>{info[1]}</em></td>"
                        f"<td>{pct:.1f}%</td>"
                        f"<td>{count}</td></tr>"
                    )
                html_parts.append("</table>")

                # Scientific references for personas
                html_parts.append("<div class='stat-box' style='font-size:0.9em;'>")
                html_parts.append("<strong>Scientific Basis:</strong> Engaged responders based on Krosnick's (1991) 'optimizers'; "
                                  "satisficers per Krosnick (1991); extreme responders per Greenleaf (1992); "
                                  "acquiescent responders per Billiet &amp; McClendon (2000); "
                                  "careless responders per Meade &amp; Craig (2012).")
                html_parts.append("</div>")

            # ── Persona by Condition ──────────────────────────────────────
            persona_by_cond = metadata.get("persona_by_condition", {})
            pbc_counts = persona_by_cond.get("counts", {}) if isinstance(persona_by_cond, dict) else {}
            if pbc_counts and conditions:
                html_parts.append("<h4>Persona Distribution by Condition</h4>")
                html_parts.append("<p>This table shows how personas were distributed across experimental conditions, "
                                  "verifying that response style composition is balanced across groups.</p>")

                # Collect all persona types across conditions
                all_ptypes = set()
                for cond_dict in pbc_counts.values():
                    if isinstance(cond_dict, dict):
                        all_ptypes.update(cond_dict.keys())
                all_ptypes_sorted = sorted(all_ptypes)

                if all_ptypes_sorted:
                    html_parts.append("<table>")
                    html_parts.append("<tr><th>Persona</th>")
                    for cond in conditions:
                        clean_c = _clean_condition_name(cond)
                        html_parts.append(f"<th>{clean_c}</th>")
                    html_parts.append("</tr>")

                    for ptype in all_ptypes_sorted:
                        html_parts.append(f"<tr><td><strong>{ptype.title()}</strong></td>")
                        for cond in conditions:
                            cond_dict = pbc_counts.get(cond, {})
                            count = cond_dict.get(ptype, 0) if isinstance(cond_dict, dict) else 0
                            html_parts.append(f"<td>{count}</td>")
                        html_parts.append("</tr>")
                    html_parts.append("</table>")

            # ── Trait Averages (Overall) ──────────────────────────────────
            trait_avg_overall = metadata.get("trait_averages_overall", {})
            if trait_avg_overall and isinstance(trait_avg_overall, dict):
                html_parts.append("<h4>Simulated Personality Profile (Sample Averages)</h4>")
                html_parts.append("<p>Average trait values across all simulated participants, derived from persona assignments.</p>")
                html_parts.append("<div class='metric-grid'>")
                for trait, val in sorted(trait_avg_overall.items()):
                    trait_display = trait.replace("_", " ").title()
                    html_parts.append(
                        f"<div class='metric-card' style='background:linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #1a1a2e;'>"
                        f"<div class='metric-value'>{val:.2f}</div>"
                        f"<div class='metric-label'>{trait_display}</div></div>"
                    )
                html_parts.append("</div>")

            # ── Trait Averages by Condition ────────────────────────────────
            trait_avg_by_cond = metadata.get("trait_averages_by_condition", {})
            if trait_avg_by_cond and isinstance(trait_avg_by_cond, dict) and conditions:
                # Check if any condition has trait data
                has_traits = any(
                    isinstance(v, dict) and len(v) > 0
                    for v in trait_avg_by_cond.values()
                )
                if has_traits:
                    html_parts.append("<h4>Personality Profiles by Condition</h4>")
                    html_parts.append("<p>Average trait values per condition. Balanced profiles across conditions indicates "
                                      "that persona assignment did not confound the experimental manipulation.</p>")

                    # Collect all traits
                    all_traits = set()
                    for cond_traits in trait_avg_by_cond.values():
                        if isinstance(cond_traits, dict):
                            all_traits.update(cond_traits.keys())
                    all_traits_sorted = sorted(all_traits)

                    if all_traits_sorted:
                        html_parts.append("<table>")
                        html_parts.append("<tr><th>Trait</th>")
                        for cond in conditions:
                            clean_c = _clean_condition_name(cond)
                            html_parts.append(f"<th>{clean_c}</th>")
                        html_parts.append("</tr>")
                        for trait in all_traits_sorted:
                            trait_display = trait.replace("_", " ").title()
                            html_parts.append(f"<tr><td>{trait_display}</td>")
                            for cond in conditions:
                                cond_traits = trait_avg_by_cond.get(cond, {})
                                val = cond_traits.get(trait, 0) if isinstance(cond_traits, dict) else 0
                                html_parts.append(f"<td>{val:.3f}</td>")
                            html_parts.append("</tr>")
                        html_parts.append("</table>")

            # ── Validation Issues Corrected ────────────────────────────────
            validation_corrected = metadata.get("validation_issues_corrected", 0)
            if validation_corrected and validation_corrected > 0:
                html_parts.append("<div class='stat-box'>")
                html_parts.append(f"<strong>Data Validation:</strong> {validation_corrected} response value(s) were "
                                  f"automatically corrected during generation to stay within valid scale ranges.")
                html_parts.append("</div>")

            html_parts.append("<a id='categorical-analysis'></a>")
            html_parts.append("<h2>5. Categorical Analysis</h2>")
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

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

        # v1.3.4: Insert executive summary into its placeholder position (before statistical analysis)
        if all_scale_results:
            exec_summary = self._generate_executive_summary(
                all_scale_results,
                prereg_text,
                n_total,
                [_clean_condition_name(c) for c in conditions]
            )
            html_parts[exec_summary_index] = exec_summary
        else:
            html_parts[exec_summary_index] = ""  # Remove placeholder if no results

        # ── Observed vs Configured Effect Sizes ─────────────────────────
        obs_effects = metadata.get("effect_sizes_observed", [])
        cfg_effects = metadata.get("effect_sizes_configured", [])
        if obs_effects or cfg_effects:
            html_parts.append("<a id='effect-verification'></a>")
            html_parts.append("<h2>6. Effect Size Verification</h2>")
            if cfg_effects and any(e.get("cohens_d", 0) > 0 for e in cfg_effects):
                html_parts.append("<h3>Configured Effects</h3>")
                html_parts.append("<table><tr><th>DV</th><th>Target d</th><th>Direction</th><th>Comparison</th></tr>")
                for eff in cfg_effects:
                    d_val = eff.get("cohens_d", 0)
                    if d_val > 0:
                        html_parts.append(
                            f"<tr><td>{eff.get('variable', '')}</td>"
                            f"<td>{d_val:.2f}</td>"
                            f"<td>{eff.get('direction', '')}</td>"
                            f"<td>{eff.get('level_high', '')} vs {eff.get('level_low', '')}</td></tr>"
                        )
                html_parts.append("</table>")
            if obs_effects:
                html_parts.append("<h3>Observed Effects in Generated Data</h3>")
                html_parts.append("<table><tr><th>DV</th><th>Observed d</th><th>Comparison</th><th>Interpretation</th></tr>")
                for eff in obs_effects:
                    d_val = abs(_safe_float(eff.get("cohens_d", eff.get("d", 0))))
                    if d_val < 0.2:
                        interp = "Negligible"
                    elif d_val < 0.5:
                        interp = "Small"
                    elif d_val < 0.8:
                        interp = "Medium"
                    else:
                        interp = "Large"
                    var_name = eff.get("variable", eff.get("scale", ""))
                    comp = f"{eff.get('condition_high', '')} vs {eff.get('condition_low', '')}"
                    html_parts.append(
                        f"<tr><td>{var_name}</td><td>{d_val:.3f}</td><td>{comp}</td><td>{interp}</td></tr>"
                    )
                html_parts.append("</table>")

        # ── Exclusion Summary ─────────────────────────────────────────
        excl = metadata.get("exclusion_summary", {})
        if excl:
            html_parts.append("<a id='data-quality'></a>")
            html_parts.append("<h2>7. Data Quality &amp; Exclusions</h2>")
            html_parts.append("<div class='metric-grid'>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_speed', 0)}</div>"
                              f"<div class='metric-label'>Speed Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_attention', 0)}</div>"
                              f"<div class='metric-label'>Attention Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_straightline', 0)}</div>"
                              f"<div class='metric-label'>Straight-line Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('total_excluded', 0)}</div>"
                              f"<div class='metric-label'>Total Excluded</div></div>")
            html_parts.append("</div>")

        # ── Generation Warnings ────────────────────────────────────────
        gen_warnings = metadata.get("generation_warnings", [])
        if gen_warnings:
            html_parts.append("<div class='warning-box' style='border-left:4px solid #f59e0b;background:#fffbeb;padding:15px;margin:20px 0;'>")
            html_parts.append("<strong>Generation Warnings:</strong><ul>")
            for gw in gen_warnings:
                html_parts.append(f"<li>{gw}</li>")
            html_parts.append("</ul></div>")

        # Footer - Notes for Instructors
        html_parts.append("<a id='methodology'></a>")
        html_parts.append("<h2>8. Instructor Notes &amp; Methodology</h2>")
        html_parts.append("<div class='warning-box'>")
        html_parts.append("<strong>Important: This is simulated data.</strong> Results demonstrate what the analysis pipeline will produce. "
                          "Students should practice these analyses independently and may get similar (but not identical) results due to random variation.")
        html_parts.append("</div>")
        html_parts.append("<div class='stat-box'>")
        html_parts.append("<h3>About This Simulation</h3>")
        html_parts.append("<ul>")
        html_parts.append("<li><strong>Persona-based generation:</strong> Each participant is assigned a response style persona "
                          "(engaged, satisficer, extreme, acquiescent, careless, etc.) based on survey methodology research.</li>")
        html_parts.append("<li><strong>Domain-specific knowledge:</strong> Responses draw on 225+ research domains across 33 categories, "
                          "ensuring contextually appropriate language and attitudes.</li>")
        html_parts.append("<li><strong>Effect size calibration:</strong> Treatment effects are calibrated to target Cohen's d values, "
                          "applied at the individual response level with validation checks.</li>")
        html_parts.append("<li><strong>Scale reliability:</strong> Multi-item scales use a factor model (Response = λ·Factor + √(1-λ²)·Error) "
                          "producing realistic Cronbach's alpha values (typically 0.75-0.90).</li>")
        html_parts.append("<li><strong>Reproducibility:</strong> Simulations are seeded for exact reproducibility. "
                          "The same seed + parameters produce identical datasets.</li>")
        html_parts.append("</ul>")
        html_parts.append("<h3>Scientific References</h3>")
        html_parts.append("<ol style='font-size:0.9em;'>")
        html_parts.append("<li>Krosnick, J. A. (1991). Response strategies for coping with the cognitive demands of attitude measures. <em>Applied Cognitive Psychology, 5</em>, 213-236.</li>")
        html_parts.append("<li>Greenleaf, E. A. (1992). Measuring extreme response style. <em>Public Opinion Quarterly, 56</em>, 328-351.</li>")
        html_parts.append("<li>Billiet, J. B., &amp; McClendon, M. J. (2000). Modeling acquiescence in measurement models. <em>Structural Equation Modeling, 7</em>, 608-628.</li>")
        html_parts.append("<li>Meade, A. W., &amp; Craig, S. B. (2012). Identifying careless responses in survey data. <em>Psychological Methods, 17</em>, 437-455.</li>")
        html_parts.append("<li>Cohen, J. (1988). <em>Statistical power analysis for the behavioral sciences</em>. Lawrence Erlbaum.</li>")
        html_parts.append("<li>Richard, F. D., Bond, C. F., &amp; Stokes-Zoota, J. J. (2003). One hundred years of social psychology quantitatively described. <em>Review of General Psychology, 7</em>, 331-363.</li>")
        html_parts.append("</ol>")
        html_parts.append("</div>")

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")
        html_parts.append(f"<p style='color:#999;font-size:0.9em;margin-top:30px;text-align:center;'>"
                          f"Generated by Behavioral Experiment Simulation Tool v{__version__} "
                          f"&middot; Proprietary Software by Dr. Eugen Dimant</p>")
        html_parts.append("</div></div></body></html>")  # close report-container + page-wrapper

        return "\n".join(html_parts)
