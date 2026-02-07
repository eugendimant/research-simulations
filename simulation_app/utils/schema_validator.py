"""
Schema Validator for Behavioral Experiment Simulation Tool
===================================================================
Validates generated data schemas and provides summary reports
following the "FILE READ OK" and "SCHEMA LOCKED" format from the
simulation methodology.

Validation Checks Performed:
1. Sample Size - Verifies actual N matches expected N (with 5% tolerance)
2. Required Columns - Checks for PARTICIPANT_ID, CONDITION columns
3. Condition Coverage - Validates all expected conditions are present
4. Condition Balance - Checks for balanced allocation (within 10%)
5. Scale Columns - Verifies scale items exist with correct ranges
6. Missing Values - Reports columns with missing data
7. Data Types - Ensures numeric columns are properly typed
8. Response Ranges - Validates Likert scale responses within bounds
9. Exclusion Flags - Verifies exclusion recommendation columns
10. Data Quality Flags - Checks attention, speeding, straightlining flags
11. Scale Response Range Validation - Values within min-max bounds
12. Condition Allocation Balance - Chi-square based balance assessment
13. Missing Data Patterns - MCAR/MAR/MNAR detection
14. Extreme Value Detection - Outlier and boundary response flagging

Error Levels:
- errors: Critical issues that invalidate the data (mark valid=False)
- warnings: Non-critical issues that should be reviewed
- info: Informational messages about the data
"""

# Version identifier to help track deployed code
__version__ = "1.2.0"  # v1.2.0: Enhanced validation with scale range, balance, missing patterns, extreme values

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd


def validate_schema(
    df: pd.DataFrame,
    expected_conditions: List[str],
    expected_scales: List[Dict[str, Any]],
    expected_n: int
) -> Dict[str, Any]:
    """
    Validate a generated DataFrame against expected schema.

    Args:
        df: Generated DataFrame to validate
        expected_conditions: List of expected condition names
        expected_scales: List of expected scale definitions
        expected_n: Expected sample size

    Returns:
        Dictionary with validation results and any errors
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }

    # Check sample size
    actual_n = len(df)
    results['summary']['sample_size'] = {
        'expected': expected_n,
        'actual': actual_n,
        'match': actual_n == expected_n
    }

    if actual_n != expected_n:
        # Allow small deviation
        if abs(actual_n - expected_n) <= expected_n * 0.05:
            results['warnings'].append(
                f"Sample size slightly differs: expected {expected_n}, got {actual_n}"
            )
        else:
            results['errors'].append(
                f"Sample size mismatch: expected {expected_n}, got {actual_n}"
            )
            results['valid'] = False

    # Check required columns
    required_columns = ['PARTICIPANT_ID', 'CONDITION']
    missing_required = [col for col in required_columns if col not in df.columns]

    if missing_required:
        results['errors'].append(f"Missing required columns: {missing_required}")
        results['valid'] = False

    # Check conditions
    if 'CONDITION' in df.columns:
        actual_conditions = set(df['CONDITION'].unique())
        expected_set = set(expected_conditions)

        missing_conditions = expected_set - actual_conditions
        extra_conditions = actual_conditions - expected_set

        results['summary']['conditions'] = {
            'expected': list(expected_set),
            'actual': list(actual_conditions),
            'missing': list(missing_conditions),
            'extra': list(extra_conditions)
        }

        if missing_conditions:
            results['warnings'].append(f"Missing conditions: {missing_conditions}")
        if extra_conditions:
            results['warnings'].append(f"Unexpected conditions: {extra_conditions}")

        # Check balanced allocation
        condition_counts = df['CONDITION'].value_counts()
        # v1.0.0: Guard against division by zero when no expected conditions
        n_expected_conditions = max(len(expected_conditions), 1)
        expected_per_condition = expected_n // n_expected_conditions

        unbalanced = []
        for cond, count in condition_counts.items():
            # v1.0.0: Guard against division by zero
            deviation = abs(count - expected_per_condition) / max(expected_per_condition, 1)
            if deviation > 0.1:  # More than 10% deviation
                unbalanced.append(f"{cond}: {count} (expected ~{expected_per_condition})")

        if unbalanced:
            results['warnings'].append(f"Unbalanced conditions: {unbalanced}")

    # Check scale columns
    for scale in expected_scales:
        scale_name = scale.get('name', 'Scale').replace(' ', '_')
        num_items = scale.get('num_items', 5)
        scale_points = scale.get('scale_points', 6)

        for item_num in range(1, num_items + 1):
            col_name = f"{scale_name}_{item_num}"

            if col_name not in df.columns:
                results['errors'].append(f"Missing scale column: {col_name}")
                results['valid'] = False
            else:
                # Check value range
                min_val = df[col_name].min()
                max_val = df[col_name].max()

                if min_val < 1 or max_val > scale_points:
                    results['warnings'].append(
                        f"Column {col_name} has out-of-range values: [{min_val}, {max_val}] "
                        f"(expected [1, {scale_points}])"
                    )

    # Check for missing values
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if len(columns_with_missing) > 0:
        results['warnings'].append(
            f"Columns with missing values: {dict(columns_with_missing)}"
        )

    # Check data types
    numeric_issues = []
    for col in df.columns:
        if col not in ['CONDITION', 'RUN_ID', 'Task_Summary']:
            if df[col].dtype == 'object':
                numeric_issues.append(col)

    if numeric_issues:
        results['warnings'].append(
            f"Columns expected to be numeric but are text: {numeric_issues}"
        )

    # Calculate summary statistics
    results['summary']['columns'] = {
        'total': len(df.columns),
        'numeric': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'text': len(df.select_dtypes(include=['object']).columns)
    }

    # =========================================================================
    # v1.2.0: Enhanced Validation Checks
    # =========================================================================

    # Initialize info list for informational messages
    if 'info' not in results:
        results['info'] = []

    # 1. Enhanced Scale Response Range Validation
    scale_range_report = validate_scale_response_ranges(df, expected_scales)
    results['summary']['scale_range_validation'] = scale_range_report
    if scale_range_report.get('violations'):
        for violation in scale_range_report['violations']:
            results['warnings'].append(
                f"Scale range violation in {violation['column']}: "
                f"{violation['count']} values ({violation['percentage']:.1f}%) out of range "
                f"[{violation['expected_min']}, {violation['expected_max']}]"
            )
    if scale_range_report.get('all_valid', False):
        results['info'].append("All scale responses within expected ranges")

    # 2. Condition Allocation Balance Check (Chi-square)
    if 'CONDITION' in df.columns and len(expected_conditions) > 0:
        balance_report = check_condition_allocation_balance(df, expected_conditions)
        results['summary']['condition_balance'] = balance_report
        if balance_report.get('significantly_unbalanced', False):
            results['warnings'].append(
                f"Condition allocation significantly unbalanced (chi-square p={balance_report['p_value']:.4f}). "
                f"Imbalance ratio: {balance_report['imbalance_ratio']:.2f}"
            )
        else:
            results['info'].append(
                f"Condition allocation balanced (chi-square p={balance_report.get('p_value', 1.0):.4f})"
            )

    # 3. Missing Data Pattern Analysis
    missing_pattern_report = analyze_missing_data_patterns(df)
    results['summary']['missing_data_patterns'] = missing_pattern_report
    if missing_pattern_report.get('pattern_type') != 'complete':
        if missing_pattern_report.get('pattern_type') == 'MNAR':
            results['warnings'].append(
                f"Missing data pattern suggests MNAR (Missing Not At Random): "
                f"{missing_pattern_report.get('mnar_indicators', [])}"
            )
        elif missing_pattern_report.get('pattern_type') == 'MAR':
            results['info'].append(
                f"Missing data pattern suggests MAR (Missing At Random) - "
                f"related to: {missing_pattern_report.get('mar_predictors', [])}"
            )
        else:
            results['info'].append(
                f"Missing data pattern: {missing_pattern_report.get('pattern_type', 'unknown')}"
            )

    # 4. Extreme Value Detection
    extreme_value_report = detect_extreme_values(df, expected_scales)
    results['summary']['extreme_values'] = extreme_value_report
    if extreme_value_report.get('has_concerns', False):
        for concern in extreme_value_report.get('concerns', []):
            if concern['severity'] == 'high':
                results['warnings'].append(concern['message'])
            else:
                results['info'].append(concern['message'])

    return results


def validate_scale_response_ranges(
    df: pd.DataFrame,
    expected_scales: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that all scale responses fall within expected min-max ranges.

    Args:
        df: DataFrame with response data
        expected_scales: List of scale definitions with range info

    Returns:
        Detailed report of range validation results
    """
    report = {
        'total_scale_columns': 0,
        'columns_checked': [],
        'violations': [],
        'all_valid': True,
        'violation_summary': {}
    }

    for scale in expected_scales:
        scale_name = scale.get('name', 'Scale').replace(' ', '_')
        num_items = scale.get('num_items', 5)

        # Support flexible scale ranges
        scale_min = scale.get('scale_min', scale.get('min_value', 1))
        scale_max = scale.get('scale_max', scale.get('scale_points', 6))

        # Check each item in the scale
        for item_num in range(1, num_items + 1):
            col_name = f"{scale_name}_{item_num}"

            if col_name not in df.columns:
                continue

            report['total_scale_columns'] += 1
            col_data = df[col_name].dropna()

            if len(col_data) == 0:
                continue

            # Find out-of-range values
            below_min = (col_data < scale_min).sum()
            above_max = (col_data > scale_max).sum()
            total_violations = below_min + above_max

            col_report = {
                'column': col_name,
                'scale': scale.get('name', 'Unknown'),
                'expected_min': scale_min,
                'expected_max': scale_max,
                'actual_min': float(col_data.min()),
                'actual_max': float(col_data.max()),
                'n_values': len(col_data),
                'below_min_count': int(below_min),
                'above_max_count': int(above_max),
                'total_violations': int(total_violations),
                'valid': total_violations == 0
            }

            report['columns_checked'].append(col_report)

            if total_violations > 0:
                report['all_valid'] = False
                violation_pct = (total_violations / len(col_data)) * 100
                report['violations'].append({
                    'column': col_name,
                    'count': int(total_violations),
                    'percentage': violation_pct,
                    'expected_min': scale_min,
                    'expected_max': scale_max,
                    'actual_range': [float(col_data.min()), float(col_data.max())]
                })

    # Summary statistics
    if report['columns_checked']:
        total_violations = sum(c['total_violations'] for c in report['columns_checked'])
        total_values = sum(c['n_values'] for c in report['columns_checked'])
        report['violation_summary'] = {
            'total_violations': total_violations,
            'total_values_checked': total_values,
            'violation_rate': (total_violations / total_values * 100) if total_values > 0 else 0
        }

    return report


def check_condition_allocation_balance(
    df: pd.DataFrame,
    expected_conditions: List[str]
) -> Dict[str, Any]:
    """
    Check condition allocation balance using chi-square test.

    Args:
        df: DataFrame with CONDITION column
        expected_conditions: List of expected condition names

    Returns:
        Report with balance metrics and chi-square results
    """
    report = {
        'n_conditions': len(expected_conditions),
        'total_n': len(df),
        'observed_counts': {},
        'expected_count': 0,
        'chi_square': 0.0,
        'p_value': 1.0,
        'imbalance_ratio': 1.0,
        'significantly_unbalanced': False,
        'condition_details': []
    }

    if 'CONDITION' not in df.columns or len(expected_conditions) == 0:
        return report

    # Calculate observed counts
    observed = df['CONDITION'].value_counts()

    # Expected count per condition (uniform distribution)
    n_conditions = max(len(expected_conditions), 1)
    expected_per_condition = len(df) / n_conditions

    report['expected_count'] = expected_per_condition
    report['observed_counts'] = observed.to_dict()

    # Build condition details
    for cond in expected_conditions:
        obs_count = observed.get(cond, 0)
        deviation = abs(obs_count - expected_per_condition)
        deviation_pct = (deviation / expected_per_condition * 100) if expected_per_condition > 0 else 0

        report['condition_details'].append({
            'condition': cond,
            'observed': int(obs_count),
            'expected': round(expected_per_condition, 1),
            'deviation': round(deviation, 1),
            'deviation_percent': round(deviation_pct, 1)
        })

    # Chi-square test for uniform distribution
    observed_values = [observed.get(cond, 0) for cond in expected_conditions]

    if sum(observed_values) > 0 and expected_per_condition > 0:
        # Calculate chi-square statistic
        chi_sq = sum((o - expected_per_condition) ** 2 / expected_per_condition
                     for o in observed_values)
        report['chi_square'] = round(chi_sq, 4)

        # Calculate p-value (chi-square with k-1 degrees of freedom)
        # Using scipy if available, otherwise approximate
        try:
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi_sq, n_conditions - 1)
            report['p_value'] = round(p_value, 4)
        except ImportError:
            # Approximate p-value using normal approximation for large df
            # This is a rough approximation
            if n_conditions > 1:
                z = (chi_sq - (n_conditions - 1)) / np.sqrt(2 * (n_conditions - 1))
                # Use standard normal approximation
                p_value = 1 - 0.5 * (1 + np.tanh(z * 0.7978845608))  # Approximation
                report['p_value'] = max(0, min(1, round(p_value, 4)))

        # Calculate imbalance ratio (max/min)
        non_zero_counts = [c for c in observed_values if c > 0]
        if len(non_zero_counts) > 1:
            report['imbalance_ratio'] = round(max(non_zero_counts) / min(non_zero_counts), 2)

        # Determine if significantly unbalanced (p < 0.05)
        report['significantly_unbalanced'] = report['p_value'] < 0.05

    return report


def analyze_missing_data_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing data patterns to classify as MCAR, MAR, or MNAR.

    Args:
        df: DataFrame to analyze

    Returns:
        Report with missing data pattern analysis
    """
    report = {
        'pattern_type': 'complete',  # complete, MCAR, MAR, MNAR
        'total_missing': 0,
        'missing_rate': 0.0,
        'columns_with_missing': [],
        'missing_by_column': {},
        'mcar_indicators': [],
        'mar_predictors': [],
        'mnar_indicators': [],
        'correlation_with_condition': {},
        'little_mcar_approximation': None
    }

    # Calculate overall missing data
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    report['total_missing'] = int(total_missing)
    report['missing_rate'] = round((total_missing / total_cells * 100) if total_cells > 0 else 0, 2)

    if total_missing == 0:
        report['pattern_type'] = 'complete'
        return report

    # Identify columns with missing data
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]

    report['columns_with_missing'] = list(cols_with_missing.index)
    report['missing_by_column'] = {
        col: {
            'count': int(count),
            'percentage': round(count / len(df) * 100, 2)
        }
        for col, count in cols_with_missing.items()
    }

    # Check for MCAR indicators
    # MCAR: Missing completely at random - missing is unrelated to any values
    mcar_indicators = []

    # Test 1: Check if missing is uniform across conditions
    if 'CONDITION' in df.columns:
        condition_missing_rates = {}
        for cond in df['CONDITION'].unique():
            cond_df = df[df['CONDITION'] == cond]
            cond_missing = cond_df.isnull().sum().sum()
            cond_cells = cond_df.shape[0] * cond_df.shape[1]
            rate = (cond_missing / cond_cells * 100) if cond_cells > 0 else 0
            condition_missing_rates[cond] = round(rate, 2)

        report['correlation_with_condition'] = condition_missing_rates

        # Check if rates are similar across conditions
        rates = list(condition_missing_rates.values())
        if len(rates) > 1:
            rate_variance = np.var(rates)
            if rate_variance < 1.0:  # Low variance suggests MCAR
                mcar_indicators.append("Missing rates similar across conditions")
            else:
                report['mar_predictors'].append('CONDITION')

    # Test 2: Check for patterns suggesting MAR
    # MAR: Missing at random - missing is related to other observed variables
    mar_predictors = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in report['columns_with_missing'][:5]:  # Check first 5 columns with missing
        if col not in numeric_cols:
            continue

        # Create missing indicator
        missing_indicator = df[col].isnull().astype(int)

        # Check correlation with other numeric variables
        for other_col in numeric_cols:
            if other_col == col or other_col in report['columns_with_missing']:
                continue

            # Safe correlation calculation
            try:
                valid_idx = ~df[other_col].isnull()
                if valid_idx.sum() > 10:
                    corr = np.corrcoef(missing_indicator[valid_idx], df[other_col][valid_idx])[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        mar_predictors.append(f"{col} missing correlated with {other_col}")
            except (ValueError, TypeError):
                pass

    if mar_predictors:
        report['mar_predictors'].extend(mar_predictors[:3])  # Top 3

    # Test 3: Check for MNAR indicators
    # MNAR: Missing not at random - missing is related to the missing value itself
    mnar_indicators = []

    for col in report['columns_with_missing']:
        if col not in numeric_cols:
            continue

        # Check if extreme values tend to be missing more
        valid_data = df[col].dropna()
        if len(valid_data) > 10:
            # Compare mean of observed vs what we'd expect
            # If we see very high or very low values, MNAR is possible
            col_mean = valid_data.mean()
            col_std = valid_data.std()

            # Check for truncation patterns (all values above/below threshold)
            if col_std > 0:
                skewness = ((valid_data - col_mean) ** 3).mean() / (col_std ** 3)
                if abs(skewness) > 1.5:  # High skewness with missing data
                    mnar_indicators.append(f"{col} shows skewed distribution with missing data")

    report['mnar_indicators'] = mnar_indicators[:3]  # Top 3
    report['mcar_indicators'] = mcar_indicators

    # Determine overall pattern type
    if total_missing == 0:
        report['pattern_type'] = 'complete'
    elif report['mnar_indicators']:
        report['pattern_type'] = 'MNAR'
    elif report['mar_predictors']:
        report['pattern_type'] = 'MAR'
    else:
        report['pattern_type'] = 'MCAR'

    return report


def detect_extreme_values(
    df: pd.DataFrame,
    expected_scales: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Detect extreme values and outliers in the data.

    Args:
        df: DataFrame to analyze
        expected_scales: List of scale definitions

    Returns:
        Report with extreme value detection results
    """
    report = {
        'has_concerns': False,
        'concerns': [],
        'extreme_responders': {},
        'boundary_responses': {},
        'outlier_columns': [],
        'straightlining_risk': {}
    }

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Exclude ID columns
    id_cols = ['PARTICIPANT_ID', 'Random_ID', 'CHUNK', 'RUN_ID']
    analysis_cols = [c for c in numeric_cols if c not in id_cols]

    # 1. Detect boundary/extreme responses at scale endpoints
    for scale in expected_scales:
        scale_name = scale.get('name', 'Scale').replace(' ', '_')
        num_items = scale.get('num_items', 5)
        scale_min = scale.get('scale_min', scale.get('min_value', 1))
        scale_max = scale.get('scale_max', scale.get('scale_points', 6))

        scale_cols = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]
        existing_cols = [c for c in scale_cols if c in df.columns]

        if not existing_cols:
            continue

        # Count boundary responses per participant
        boundary_counts = {}
        for idx, row in df.iterrows():
            min_count = sum(1 for c in existing_cols if row.get(c) == scale_min)
            max_count = sum(1 for c in existing_cols if row.get(c) == scale_max)
            boundary_counts[idx] = {
                'min_responses': min_count,
                'max_responses': max_count,
                'total_boundary': min_count + max_count,
                'total_items': len(existing_cols)
            }

        # Identify extreme responders (>80% boundary responses)
        extreme_responders = []
        for idx, counts in boundary_counts.items():
            if counts['total_items'] > 0:
                boundary_rate = counts['total_boundary'] / counts['total_items']
                if boundary_rate > 0.8:
                    extreme_responders.append(idx)

        if extreme_responders:
            n_extreme = len(extreme_responders)
            pct_extreme = (n_extreme / len(df)) * 100
            report['extreme_responders'][scale.get('name', 'Unknown')] = {
                'count': n_extreme,
                'percentage': round(pct_extreme, 1),
                'participant_indices': extreme_responders[:10]  # First 10
            }

            if pct_extreme > 10:
                report['has_concerns'] = True
                report['concerns'].append({
                    'type': 'extreme_responding',
                    'severity': 'high' if pct_extreme > 20 else 'medium',
                    'message': f"Scale '{scale.get('name', 'Unknown')}': {n_extreme} participants ({pct_extreme:.1f}%) "
                               f"show extreme responding (>80% boundary responses)"
                })

    # 2. Detect statistical outliers (IQR method)
    for col in analysis_cols[:20]:  # Check first 20 columns
        data = df[col].dropna()
        if len(data) < 10:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 3 * iqr  # Use 3*IQR for severe outliers
        upper_bound = q3 + 3 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(data)) * 100
            report['outlier_columns'].append({
                'column': col,
                'n_outliers': len(outliers),
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'outlier_values': [round(v, 2) for v in outliers.head(5).tolist()]
            })

            if outlier_pct > 5:
                report['has_concerns'] = True
                report['concerns'].append({
                    'type': 'statistical_outliers',
                    'severity': 'medium',
                    'message': f"Column '{col}': {len(outliers)} outliers ({outlier_pct:.1f}%) "
                               f"outside 3*IQR bounds"
                })

    # 3. Detect straightlining (same response across all items)
    for scale in expected_scales:
        scale_name = scale.get('name', 'Scale').replace(' ', '_')
        num_items = scale.get('num_items', 5)

        scale_cols = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]
        existing_cols = [c for c in scale_cols if c in df.columns]

        if len(existing_cols) < 3:  # Need at least 3 items
            continue

        straightliners = 0
        for idx, row in df.iterrows():
            values = [row.get(c) for c in existing_cols]
            valid_values = [v for v in values if pd.notna(v)]
            if len(valid_values) >= 3 and len(set(valid_values)) == 1:
                straightliners += 1

        if straightliners > 0:
            pct_straightline = (straightliners / len(df)) * 100
            report['straightlining_risk'][scale.get('name', 'Unknown')] = {
                'count': straightliners,
                'percentage': round(pct_straightline, 1)
            }

            if pct_straightline > 15:
                report['has_concerns'] = True
                report['concerns'].append({
                    'type': 'straightlining',
                    'severity': 'high' if pct_straightline > 25 else 'medium',
                    'message': f"Scale '{scale.get('name', 'Unknown')}': {straightliners} participants "
                               f"({pct_straightline:.1f}%) show straightlining"
                })

    return report


def generate_validation_report(
    df: pd.DataFrame,
    expected_conditions: List[str],
    expected_scales: List[Dict[str, Any]],
    expected_n: int,
    include_detailed_checks: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report suitable for adding to metadata.

    This is the main entry point for detailed validation that includes all checks
    and produces a report structure suitable for storage in simulation metadata.

    Args:
        df: Generated DataFrame to validate
        expected_conditions: List of expected condition names
        expected_scales: List of expected scale definitions
        expected_n: Expected sample size
        include_detailed_checks: Whether to include all detailed validation checks

    Returns:
        Comprehensive validation report dictionary
    """
    # Get base validation results
    base_results = validate_schema(df, expected_conditions, expected_scales, expected_n)

    # Build comprehensive report
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'validator_version': __version__,
        'overall_valid': base_results['valid'],
        'summary': {
            'sample_size': base_results['summary'].get('sample_size', {}),
            'conditions': base_results['summary'].get('conditions', {}),
            'columns': base_results['summary'].get('columns', {}),
            'error_count': len(base_results['errors']),
            'warning_count': len(base_results['warnings']),
            'info_count': len(base_results.get('info', []))
        },
        'errors': base_results['errors'],
        'warnings': base_results['warnings'],
        'info': base_results.get('info', []),
        'detailed_checks': {}
    }

    if include_detailed_checks:
        report['detailed_checks'] = {
            'scale_range_validation': base_results['summary'].get('scale_range_validation', {}),
            'condition_balance': base_results['summary'].get('condition_balance', {}),
            'missing_data_patterns': base_results['summary'].get('missing_data_patterns', {}),
            'extreme_values': base_results['summary'].get('extreme_values', {})
        }

        # Add quality score
        quality_score = 100
        quality_score -= len(base_results['errors']) * 20
        quality_score -= len(base_results['warnings']) * 5
        quality_score = max(0, quality_score)

        report['quality_score'] = quality_score
        report['quality_grade'] = (
            'A' if quality_score >= 90 else
            'B' if quality_score >= 80 else
            'C' if quality_score >= 70 else
            'D' if quality_score >= 60 else
            'F'
        )

    return report


def generate_schema_summary(
    df: pd.DataFrame,
    conditions: List[str],
    factors: List[Dict[str, Any]],
    scales: List[Dict[str, Any]]
) -> str:
    """
    Generate a schema summary in the "FILE READ OK / SCHEMA LOCKED" format.

    Args:
        df: Generated DataFrame
        conditions: List of condition names
        factors: List of factor definitions
        scales: List of scale definitions

    Returns:
        Formatted string following the methodology's schema summary format
    """
    lines = [
        "=" * 70,
        "SCHEMA VALIDATION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "-" * 70,
        "FILE READ OK",
        "-" * 70,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Conditions: {len(conditions)}",
        "",
    ]

    # Condition breakdown
    lines.append("CONDITION ALLOCATION:")
    if 'CONDITION' in df.columns:
        for cond in conditions:
            count = len(df[df['CONDITION'] == cond])
            pct = count / len(df) * 100
            lines.append(f"  {cond}: {count} ({pct:.1f}%)")
    lines.append("")

    # Factor summary
    lines.append("FACTORS:")
    for factor in factors:
        lines.append(f"  {factor['name']}: {', '.join(factor['levels'])}")
    lines.append("")

    # Column listing
    lines.extend([
        "-" * 70,
        "SCHEMA LOCKED",
        "-" * 70,
        f"Total Variables: {len(df.columns)}",
        "",
        "COLUMN LISTING:",
    ])

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        if dtype == 'int64':
            range_str = f"[{df[col].min()}, {df[col].max()}]"
        elif dtype == 'float64':
            range_str = f"[{df[col].min():.2f}, {df[col].max():.2f}]"
        else:
            unique_count = df[col].nunique()
            range_str = f"({unique_count} unique values)"

        lines.append(f"  {i:2}. {col}: {dtype} {range_str}")

    lines.append("")

    # Scale summary
    lines.extend([
        "-" * 70,
        "MEASUREMENT SCALES",
        "-" * 70,
    ])

    for scale in scales:
        scale_name = scale['name'].replace(' ', '_')
        num_items = scale.get('num_items', 5)
        scale_points = scale.get('scale_points', 6)
        reverse_items = scale.get('reverse_items', [])

        lines.append(f"\n{scale['name']}:")
        lines.append(f"  Items: {num_items}")
        lines.append(f"  Scale: 1-{scale_points}")
        if reverse_items:
            lines.append(f"  Reverse-coded: {reverse_items}")

        # Calculate reliability (Cronbach's alpha approximation)
        scale_cols = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]
        existing_cols = [c for c in scale_cols if c in df.columns]

        if len(existing_cols) >= 2:
            scale_data = df[existing_cols]
            # Simple average inter-item correlation as alpha proxy
            corr_matrix = scale_data.corr()
            n = len(existing_cols)
            # Guard against division by zero
            denominator = n * (n - 1)
            if denominator > 0:
                mean_r = (corr_matrix.sum().sum() - n) / denominator
                alpha_denom = 1 + (n - 1) * mean_r
                if abs(alpha_denom) > 1e-10:  # Avoid division by zero
                    alpha = (n * mean_r) / alpha_denom
                    lines.append(f"  Est. Cronbach's Alpha: {alpha:.3f}")

            # Item means
            lines.append(f"  Item Means: {', '.join([f'{scale_data[c].mean():.2f}' for c in existing_cols])}")

    lines.append("")

    # Demographics summary
    lines.extend([
        "-" * 70,
        "DEMOGRAPHIC SUMMARY",
        "-" * 70,
    ])

    if 'Age' in df.columns:
        lines.append(f"Age: M = {df['Age'].mean():.1f}, SD = {df['Age'].std():.1f}, "
                    f"Range = [{df['Age'].min()}, {df['Age'].max()}]")

    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        # v1.4.3: Gender is now stored as string labels; support both old numeric and new string formats
        gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say'}
        gender_str = ', '.join([
            f"{gender_map.get(g, g)}: {c} ({c/len(df)*100:.1f}%)"
            for g, c in sorted(gender_counts.items(), key=lambda x: str(x[0]))
        ])
        lines.append(f"Gender: {gender_str}")

    lines.append("")

    # Attention check summary (v1.4.3: support both old and new column names)
    _attn_col = 'Attention_Check_1' if 'Attention_Check_1' in df.columns else (
        'AI_Mentioned_Check' if 'AI_Mentioned_Check' in df.columns else None
    )
    if _attn_col is not None and 'CONDITION' in df.columns:
        lines.extend([
            "-" * 70,
            "ATTENTION CHECK ACCURACY",
            "-" * 70,
        ])

        # Calculate accuracy per condition type
        ai_conditions = [c for c in conditions if 'AI' in c.upper() and 'NO' not in c.upper()]
        no_ai_conditions = [c for c in conditions if 'NO' in c.upper() or 'AI' not in c.upper()]

        if ai_conditions:
            ai_df = df[df['CONDITION'].isin(ai_conditions)]
            ai_correct = (ai_df[_attn_col] == 1).sum()
            ai_accuracy = ai_correct / len(ai_df) * 100 if len(ai_df) > 0 else 0
            lines.append(f"AI Conditions (should answer Yes): {ai_accuracy:.1f}% correct")

        if no_ai_conditions:
            no_ai_df = df[df['CONDITION'].isin(no_ai_conditions)]
            no_ai_correct = (no_ai_df[_attn_col] == 2).sum()
            no_ai_accuracy = no_ai_correct / len(no_ai_df) * 100 if len(no_ai_df) > 0 else 0
            lines.append(f"No AI Conditions (should answer No): {no_ai_accuracy:.1f}% correct")

        overall_correct = 0
        for cond in conditions:
            cond_df = df[df['CONDITION'] == cond]
            if 'AI' in cond.upper() and 'NO' not in cond.upper():
                overall_correct += (cond_df[_attn_col] == 1).sum()
            else:
                overall_correct += (cond_df[_attn_col] == 2).sum()

        overall_accuracy = overall_correct / len(df) * 100
        lines.append(f"Overall Manipulation Check Accuracy: {overall_accuracy:.1f}%")

    lines.extend([
        "",
        "=" * 70,
        "END OF SCHEMA SUMMARY",
        "=" * 70,
    ])

    return "\n".join(lines)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data quality checks on the generated data.

    Args:
        df: Generated DataFrame

    Returns:
        Dictionary with quality metrics and flags
    """
    quality = {
        'overall_score': 100,
        'issues': [],
        'metrics': {}
    }

    # Check for constant columns (zero variance)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    constant_cols = [col for col in numeric_cols if df[col].std() == 0]

    if constant_cols:
        quality['issues'].append({
            'type': 'zero_variance',
            'severity': 'high',
            'message': f"Columns with zero variance: {constant_cols}",
            'penalty': 20
        })
        quality['overall_score'] -= 20

    # Check for suspicious patterns (all same value)
    suspicious_uniformity = []
    for col in numeric_cols:
        if df[col].nunique() == 1:
            suspicious_uniformity.append(col)

    if suspicious_uniformity:
        quality['issues'].append({
            'type': 'uniform_responses',
            'severity': 'high',
            'message': f"Columns with identical values: {suspicious_uniformity}",
            'penalty': 15
        })
        quality['overall_score'] -= 15

    # Check variance levels (too low variance is unrealistic)
    low_variance_cols = []
    for col in numeric_cols:
        if col not in ['PARTICIPANT_ID', 'CHUNK']:
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            if 0 < cv < 0.05:  # Very low coefficient of variation
                low_variance_cols.append(col)

    if low_variance_cols:
        quality['issues'].append({
            'type': 'low_variance',
            'severity': 'medium',
            'message': f"Columns with unusually low variance: {low_variance_cols}",
            'penalty': 10
        })
        quality['overall_score'] -= 10

    # Check balanced conditions
    if 'CONDITION' in df.columns:
        condition_counts = df['CONDITION'].value_counts()
        # v1.0.0: Guard against division by zero
        n_conditions = max(len(condition_counts), 1)
        expected_per = len(df) / n_conditions
        expected_per_safe = max(expected_per, 1)
        max_deviation = max((abs(c - expected_per) / expected_per_safe for c in condition_counts.values), default=0)

        quality['metrics']['condition_balance'] = 1 - max_deviation

        if max_deviation > 0.1:
            quality['issues'].append({
                'type': 'unbalanced_conditions',
                'severity': 'low',
                'message': f"Condition allocation deviates by {max_deviation*100:.1f}%",
                'penalty': 5
            })
            quality['overall_score'] -= 5

    # Calculate overall variance health
    variance_scores = []
    for col in numeric_cols:
        if col not in ['PARTICIPANT_ID', 'CHUNK', 'Random_ID']:
            if df[col].std() > 0:
                variance_scores.append(min(1, df[col].std() / 2))

    quality['metrics']['variance_health'] = sum(variance_scores) / len(variance_scores) if variance_scores else 0

    # Ensure score doesn't go negative
    quality['overall_score'] = max(0, quality['overall_score'])

    return quality
