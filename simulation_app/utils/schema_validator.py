"""
Schema Validator for Behavioral Experiment Simulation Tool
===================================================================
Validates generated data schemas and provides summary reports
following the "FILE READ OK" and "SCHEMA LOCKED" format from the
simulation methodology.
"""

# Version identifier to help track deployed code
__version__ = "2.1.1"  # Synced with app.py

from datetime import datetime
from typing import Any, Dict, List

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
        expected_per_condition = expected_n // len(expected_conditions)

        unbalanced = []
        for cond, count in condition_counts.items():
            deviation = abs(count - expected_per_condition) / expected_per_condition
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

    return results


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
        gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say'}
        gender_str = ', '.join([
            f"{gender_map.get(g, g)}: {c} ({c/len(df)*100:.1f}%)"
            for g, c in sorted(gender_counts.items())
        ])
        lines.append(f"Gender: {gender_str}")

    lines.append("")

    # Attention check summary
    if 'AI_Mentioned_Check' in df.columns and 'CONDITION' in df.columns:
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
            ai_correct = (ai_df['AI_Mentioned_Check'] == 1).sum()
            ai_accuracy = ai_correct / len(ai_df) * 100 if len(ai_df) > 0 else 0
            lines.append(f"AI Conditions (should answer Yes): {ai_accuracy:.1f}% correct")

        if no_ai_conditions:
            no_ai_df = df[df['CONDITION'].isin(no_ai_conditions)]
            no_ai_correct = (no_ai_df['AI_Mentioned_Check'] == 2).sum()
            no_ai_accuracy = no_ai_correct / len(no_ai_df) * 100 if len(no_ai_df) > 0 else 0
            lines.append(f"No AI Conditions (should answer No): {no_ai_accuracy:.1f}% correct")

        overall_correct = 0
        for cond in conditions:
            cond_df = df[df['CONDITION'] == cond]
            if 'AI' in cond.upper() and 'NO' not in cond.upper():
                overall_correct += (cond_df['AI_Mentioned_Check'] == 1).sum()
            else:
                overall_correct += (cond_df['AI_Mentioned_Check'] == 2).sum()

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
        expected_per = len(df) / len(condition_counts)
        max_deviation = max(abs(c - expected_per) / expected_per for c in condition_counts.values)

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
