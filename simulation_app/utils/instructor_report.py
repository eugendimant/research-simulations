"""
Instructor Report Generator
===========================
Generates comprehensive visual and statistical reports for instructor review.
Analyzes simulated data for quality, expected effects, and potential issues.

Reports are generated for INSTRUCTOR EYES ONLY - not shared with students.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    interpretation: str
    details: Dict[str, Any]


class InstructorReportGenerator:
    """
    Generates comprehensive instructor reports for simulated data.

    Reports include:
    1. Data quality checks
    2. Descriptive statistics
    3. Condition comparisons
    4. Scale reliability estimates
    5. Effect size analysis
    6. Visualization plots (as base64 or description)
    7. Comparison with expected effects
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        expected_effects: List[Dict[str, Any]] = None
    ):
        """
        Initialize the report generator.

        Args:
            df: The simulated dataset
            metadata: Simulation metadata
            expected_effects: List of expected effect specifications
        """
        self.df = df
        self.metadata = metadata
        self.expected_effects = expected_effects or []
        self.report_sections = []

    def _calculate_cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _calculate_cronbach_alpha(self, items: pd.DataFrame) -> float:
        """Calculate Cronbach's alpha for a set of items."""
        n_items = items.shape[1]
        if n_items < 2:
            return np.nan

        item_variances = items.var(axis=0, ddof=1)
        total_var = items.sum(axis=1).var(ddof=1)

        if total_var == 0:
            return np.nan

        alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_var)
        return alpha

    def _interpret_effect_size(self, d: float) -> str:
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

    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze overall data quality."""
        quality = {
            'sample_size': len(self.df),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': round(
                self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100, 2
            ),
            'duplicate_rows': self.df.duplicated().sum(),
        }

        # Exclusion summary
        if 'Exclude_Recommended' in self.df.columns:
            quality['excluded_count'] = int(self.df['Exclude_Recommended'].sum())
            quality['excluded_percentage'] = round(
                quality['excluded_count'] / len(self.df) * 100, 2
            )

        # Check for suspicious patterns
        quality['issues'] = []

        # Zero variance columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].std() == 0:
                quality['issues'].append(f"Zero variance in column: {col}")

        # Check condition balance
        if 'CONDITION' in self.df.columns:
            condition_counts = self.df['CONDITION'].value_counts()
            max_imbalance = condition_counts.max() - condition_counts.min()
            if max_imbalance > len(self.df) * 0.1:  # >10% imbalance
                quality['issues'].append(
                    f"Condition imbalance detected: {dict(condition_counts)}"
                )
            quality['condition_distribution'] = condition_counts.to_dict()

        return quality

    def analyze_descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics for all numeric variables."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['PARTICIPANT_ID', 'SIMULATION_SEED', 'Flag_Speed',
                       'Flag_Attention', 'Flag_StraightLine', 'Exclude_Recommended']

        stats = {}
        for col in numeric_cols:
            if col not in exclude_cols:
                col_data = self.df[col].dropna()
                stats[col] = {
                    'mean': round(col_data.mean(), 3),
                    'sd': round(col_data.std(), 3),
                    'median': round(col_data.median(), 3),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': round(col_data.skew(), 3),
                    'kurtosis': round(col_data.kurtosis(), 3),
                    'n': len(col_data)
                }

        return stats

    def analyze_condition_effects(self) -> List[Dict[str, Any]]:
        """Analyze effects across experimental conditions."""
        if 'CONDITION' not in self.df.columns:
            return []

        conditions = self.df['CONDITION'].unique()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['PARTICIPANT_ID', 'SIMULATION_SEED', 'Flag_Speed',
                       'Flag_Attention', 'Flag_StraightLine', 'Exclude_Recommended',
                       'Completion_Time_Seconds', 'Max_Straight_Line']

        results = []

        for col in numeric_cols:
            if col not in exclude_cols:
                # Get means by condition
                condition_means = {}
                for cond in conditions:
                    cond_data = self.df[self.df['CONDITION'] == cond][col].dropna()
                    condition_means[cond] = {
                        'mean': round(cond_data.mean(), 3),
                        'sd': round(cond_data.std(), 3),
                        'n': len(cond_data)
                    }

                # Calculate pairwise effect sizes
                effect_sizes = {}
                condition_list = list(conditions)
                for i in range(len(condition_list)):
                    for j in range(i + 1, len(condition_list)):
                        c1, c2 = condition_list[i], condition_list[j]
                        g1 = self.df[self.df['CONDITION'] == c1][col].dropna().values
                        g2 = self.df[self.df['CONDITION'] == c2][col].dropna().values
                        d = self._calculate_cohens_d(g1, g2)
                        effect_sizes[f"{c1} vs {c2}"] = {
                            'd': round(d, 3),
                            'interpretation': self._interpret_effect_size(d)
                        }

                results.append({
                    'variable': col,
                    'condition_means': condition_means,
                    'effect_sizes': effect_sizes
                })

        return results

    def analyze_scale_reliability(self) -> Dict[str, Any]:
        """Analyze reliability of multi-item scales."""
        scales = self.metadata.get('scales', [])
        reliability = {}

        for scale in scales:
            scale_name = scale['name'].replace(' ', '_')
            num_items = scale.get('num_items', 0)

            if num_items >= 2:
                # Find scale items
                item_cols = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]
                available_cols = [c for c in item_cols if c in self.df.columns]

                if len(available_cols) >= 2:
                    items_df = self.df[available_cols]
                    alpha = self._calculate_cronbach_alpha(items_df)

                    # Item-total correlations
                    total_score = items_df.sum(axis=1)
                    item_total_corrs = {}
                    for col in available_cols:
                        corr = items_df[col].corr(total_score - items_df[col])
                        item_total_corrs[col] = round(corr, 3)

                    reliability[scale_name] = {
                        'cronbach_alpha': round(alpha, 3) if not np.isnan(alpha) else None,
                        'n_items': len(available_cols),
                        'item_total_correlations': item_total_corrs,
                        'interpretation': self._interpret_alpha(alpha)
                    }

        return reliability

    def _interpret_alpha(self, alpha: float) -> str:
        """Interpret Cronbach's alpha."""
        if np.isnan(alpha):
            return "Cannot calculate"
        elif alpha >= 0.9:
            return "Excellent"
        elif alpha >= 0.8:
            return "Good"
        elif alpha >= 0.7:
            return "Acceptable"
        elif alpha >= 0.6:
            return "Questionable"
        elif alpha >= 0.5:
            return "Poor"
        else:
            return "Unacceptable"

    def compare_to_expected(self) -> List[Dict[str, Any]]:
        """Compare observed effects to expected effect sizes."""
        if not self.expected_effects:
            return []

        comparisons = []

        for expected in self.expected_effects:
            var_name = expected.get('variable', '')
            level_high = expected.get('level_high', '')
            level_low = expected.get('level_low', '')
            expected_d = expected.get('cohens_d', 0)

            # Find matching data
            high_mask = self.df['CONDITION'].str.contains(level_high, case=False, na=False)
            low_mask = self.df['CONDITION'].str.contains(level_low, case=False, na=False)

            # Find matching variable columns
            matching_cols = [c for c in self.df.columns if var_name.lower() in c.lower()]

            for col in matching_cols:
                if self.df[col].dtype in [np.float64, np.int64]:
                    high_data = self.df[high_mask][col].dropna().values
                    low_data = self.df[low_mask][col].dropna().values

                    if len(high_data) > 0 and len(low_data) > 0:
                        observed_d = self._calculate_cohens_d(high_data, low_data)

                        comparisons.append({
                            'variable': col,
                            'expected_effect': f"{level_high} > {level_low}",
                            'expected_d': expected_d,
                            'observed_d': round(observed_d, 3),
                            'difference': round(observed_d - expected_d, 3),
                            'direction_match': (expected_d > 0 and observed_d > 0) or
                                              (expected_d < 0 and observed_d < 0),
                            'high_group_mean': round(np.mean(high_data), 3),
                            'low_group_mean': round(np.mean(low_data), 3)
                        })

        return comparisons

    def generate_text_visualizations(self) -> Dict[str, str]:
        """Generate ASCII-based visualizations for the report."""
        viz = {}

        # Condition distribution bar chart (ASCII)
        if 'CONDITION' in self.df.columns:
            counts = self.df['CONDITION'].value_counts()
            max_count = counts.max()
            bar_width = 40

            lines = ["Condition Distribution:", "-" * 60]
            for cond, count in counts.items():
                bar_len = int((count / max_count) * bar_width)
                bar = "=" * bar_len
                lines.append(f"{cond[:20]:20} |{bar:40}| {count}")
            viz['condition_distribution'] = "\n".join(lines)

        # Numeric variable distributions (summary)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        scale_cols = [c for c in numeric_cols if any(
            c.startswith(s['name'].replace(' ', '_'))
            for s in self.metadata.get('scales', [])
        )]

        if scale_cols:
            lines = ["Scale Response Distributions:", "-" * 60]
            for col in scale_cols[:5]:  # Limit to first 5
                data = self.df[col].dropna()
                lines.append(f"\n{col}:")
                lines.append(f"  Mean: {data.mean():.2f} (SD: {data.std():.2f})")
                lines.append(f"  Range: {data.min():.0f} - {data.max():.0f}")
                # Simple histogram
                hist, bins = np.histogram(data, bins=5)
                max_h = max(hist)
                for i, h in enumerate(hist):
                    bar = "*" * int((h / max_h) * 20) if max_h > 0 else ""
                    lines.append(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({h})")
            viz['scale_distributions'] = "\n".join(lines)

        return viz

    def generate_full_report(self) -> str:
        """Generate complete instructor report as text."""
        lines = [
            "=" * 70,
            "INSTRUCTOR REPORT - CONFIDENTIAL",
            "=" * 70,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Study: {self.metadata.get('study_title', 'Unknown')}",
            f"Run ID: {self.metadata.get('run_id', 'Unknown')}",
            f"Mode: {self.metadata.get('simulation_mode', 'Unknown').upper()}",
            "",
        ]

        # Team Info (if available)
        if 'group_number' in self.metadata:
            lines.extend([
                "TEAM INFORMATION",
                "-" * 70,
                f"  Group Number: {self.metadata.get('group_number', 'N/A')}",
                f"  Team Members: {', '.join(self.metadata.get('team_members', []))}",
                "",
            ])

        # Data Quality
        quality = self.analyze_data_quality()
        lines.extend([
            "1. DATA QUALITY SUMMARY",
            "-" * 70,
            f"  Sample Size: {quality['sample_size']}",
            f"  Missing Values: {quality['missing_values']} ({quality['missing_percentage']}%)",
            f"  Duplicate Rows: {quality['duplicate_rows']}",
        ])
        if 'excluded_count' in quality:
            lines.append(
                f"  Recommended Exclusions: {quality['excluded_count']} "
                f"({quality['excluded_percentage']}%)"
            )
        if quality['issues']:
            lines.append("  Issues Detected:")
            for issue in quality['issues']:
                lines.append(f"    - {issue}")
        if 'condition_distribution' in quality:
            lines.append("  Condition Distribution:")
            for cond, count in quality['condition_distribution'].items():
                lines.append(f"    - {cond}: n={count}")
        lines.append("")

        # Descriptive Statistics
        stats = self.analyze_descriptive_statistics()
        lines.extend([
            "2. DESCRIPTIVE STATISTICS (Key Variables)",
            "-" * 70,
        ])
        # Show key variables only
        key_vars = ['Age', 'Gender'] + [
            c for c in stats.keys()
            if any(s['name'].replace(' ', '_') in c for s in self.metadata.get('scales', []))
        ][:10]
        for var in key_vars:
            if var in stats:
                s = stats[var]
                lines.append(f"  {var}:")
                lines.append(f"    M = {s['mean']}, SD = {s['sd']}, Range = [{s['min']}, {s['max']}]")
        lines.append("")

        # Scale Reliability
        reliability = self.analyze_scale_reliability()
        if reliability:
            lines.extend([
                "3. SCALE RELIABILITY (Cronbach's Alpha)",
                "-" * 70,
            ])
            for scale_name, rel in reliability.items():
                alpha = rel['cronbach_alpha']
                interp = rel['interpretation']
                lines.append(f"  {scale_name}: alpha = {alpha} ({interp})")
            lines.append("")

        # Condition Effects
        effects = self.analyze_condition_effects()
        if effects:
            lines.extend([
                "4. CONDITION EFFECTS (Cohen's d)",
                "-" * 70,
            ])
            for effect in effects[:5]:  # Limit to first 5 variables
                var = effect['variable']
                lines.append(f"\n  {var}:")
                lines.append("  Means by condition:")
                for cond, means in effect['condition_means'].items():
                    lines.append(f"    {cond}: M={means['mean']}, SD={means['sd']}")
                lines.append("  Pairwise effect sizes:")
                for comparison, es in effect['effect_sizes'].items():
                    lines.append(f"    {comparison}: d={es['d']} ({es['interpretation']})")
            lines.append("")

        # Expected vs Observed Effects
        comparisons = self.compare_to_expected()
        if comparisons:
            lines.extend([
                "5. EXPECTED VS OBSERVED EFFECTS",
                "-" * 70,
            ])
            for comp in comparisons:
                match_str = "MATCH" if comp['direction_match'] else "MISMATCH"
                lines.append(f"  {comp['variable']}:")
                lines.append(f"    Expected: d = {comp['expected_d']} ({comp['expected_effect']})")
                lines.append(f"    Observed: d = {comp['observed_d']} [{match_str}]")
                lines.append(f"    Group means: {comp['high_group_mean']} vs {comp['low_group_mean']}")
            lines.append("")

        # Visualizations
        viz = self.generate_text_visualizations()
        lines.extend([
            "6. VISUALIZATIONS",
            "-" * 70,
        ])
        for name, content in viz.items():
            lines.append(f"\n{content}")
        lines.append("")

        # Simulation Parameters
        lines.extend([
            "7. SIMULATION PARAMETERS USED",
            "-" * 70,
            f"  Random Seed: {self.metadata.get('seed', 'N/A')}",
            f"  Detected Domains: {', '.join(self.metadata.get('detected_domains', [])[:5])}",
        ])
        if 'persona_distribution' in self.metadata:
            lines.append("  Persona Distribution:")
            for persona, pct in self.metadata['persona_distribution'].items():
                lines.append(f"    - {persona}: {pct*100:.1f}%")
        lines.append("")

        # Footer
        lines.extend([
            "=" * 70,
            "END OF INSTRUCTOR REPORT",
            "=" * 70,
            "",
            "NOTE: This report is for instructor review only.",
            "Compare student-submitted analyses against this report to verify accuracy.",
        ])

        return "\n".join(lines)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate report as JSON for programmatic access."""
        return {
            'metadata': self.metadata,
            'data_quality': self.analyze_data_quality(),
            'descriptive_statistics': self.analyze_descriptive_statistics(),
            'scale_reliability': self.analyze_scale_reliability(),
            'condition_effects': self.analyze_condition_effects(),
            'expected_vs_observed': self.compare_to_expected(),
            'generated_at': datetime.now().isoformat()
        }


def generate_instructor_package(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    expected_effects: List[Dict[str, Any]] = None
) -> Tuple[str, str, Dict]:
    """
    Generate complete instructor package.

    Returns:
        Tuple of (text_report, r_script, json_data)
    """
    generator = InstructorReportGenerator(df, metadata, expected_effects)

    text_report = generator.generate_full_report()
    json_data = generator.generate_json_report()

    # Generate R script
    from .enhanced_simulation_engine import EnhancedSimulationEngine
    # Create minimal engine just for R export
    r_script = _generate_basic_r_script(df, metadata)

    return text_report, r_script, json_data


def _generate_basic_r_script(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """Generate basic R script for data analysis."""
    conditions = metadata.get('conditions', [])
    scales = metadata.get('scales', [])

    lines = [
        "# ============================================================",
        f"# R Analysis Script - {metadata.get('study_title', 'Study')}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# FOR INSTRUCTOR USE ONLY",
        "# ============================================================",
        "",
        "# Load data",
        'data <- read.csv("Simulated.csv", stringsAsFactors = FALSE)',
        "",
        "# Set up factors",
        f'data$CONDITION <- factor(data$CONDITION, levels = c({", ".join([f\'"{c}"\' for c in conditions])}))',
        'data$Gender <- factor(data$Gender, levels = 1:4,',
        '                       labels = c("Male", "Female", "Non-binary", "Prefer not to say"))',
        "",
        "# Filter excluded participants",
        'data_clean <- data[data$Exclude_Recommended == 0, ]',
        f'cat("Clean N:", nrow(data_clean), "\\n")',
        "",
        "# Quick descriptives",
        'summary(data_clean)',
        "",
    ]

    # Add scale composites and reliability
    for scale in scales:
        scale_name = scale['name'].replace(' ', '_')
        num_items = scale.get('num_items', 1)
        items = [f'data_clean${scale_name}_{i}' for i in range(1, num_items + 1)]

        if num_items >= 2:
            lines.extend([
                f"# {scale['name']} composite and reliability",
                f'items_{scale_name} <- cbind({", ".join(items)})',
                f'data_clean${scale_name}_composite <- rowMeans(items_{scale_name}, na.rm=TRUE)',
                f'# psych::alpha(items_{scale_name})  # Uncomment if psych package installed',
                "",
            ])

    # Add basic analysis
    lines.extend([
        "# Condition comparisons",
        'table(data_clean$CONDITION)',
        "",
        "# Key variable means by condition",
        'aggregate(. ~ CONDITION, data = data_clean[, c("CONDITION", "Age")], mean)',
        "",
    ])

    return "\n".join(lines)


# Export
__all__ = ['InstructorReportGenerator', 'generate_instructor_package']
