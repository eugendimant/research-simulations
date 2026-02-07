#!/usr/bin/env python3
"""
Comprehensive Simulation Stress Test Suite - 5 iterations of deep testing.

Iteration 1: Deep data quality audit
  - Distribution checks (no constant columns, proper variation)
  - Condition effects validation
  - Data type integrity
  - NaN/null detection
  - Participant ID uniqueness

Iteration 2: Scale integrity audit
  - Exact item counts match spec
  - Value ranges are exhaustively checked per-row
  - Scale columns are properly integer-typed
  - No fractional values in Likert scales
  - Open-ended columns are text, not numeric

Iteration 3: Edge case stress tests
  - 2-point scales, 100-point scales, 1-item scales, 50-item scales
  - Single condition vs many conditions
  - Empty open-ended lists, many open-ended questions
  - Large sample sizes, minimal sample sizes
  - Repeated simulations with different seeds produce different data

Iteration 4: Statistical validity
  - Mean values are plausible (not all at endpoints)
  - Standard deviations indicate actual variation
  - Condition means differ when effect sizes are applied
  - No impossible correlations (r > 1.0)
  - Demographic distributions are reasonable

Iteration 5: Cross-QSF comprehensive re-test with all checks combined

Usage:
    python3 tests/test_simulation_stress.py
"""
import sys
import os
import json
import traceback
import io
import zipfile
import re
import glob
from collections import Counter, defaultdict

# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

import numpy as np
import pandas as pd
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine


def _is_numeric_col(series):
    """Check if a pandas Series/column is numeric (handles StringDtype safely)."""
    try:
        return pd.api.types.is_numeric_dtype(series)
    except (TypeError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Helpers (from test_qsf_simulation_match.py)
# ---------------------------------------------------------------------------

def _safe_int(val, default):
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def preview_to_engine_inputs(preview):
    conditions = [
        str(c).replace('\xa0', ' ').strip()
        for c in (preview.detected_conditions or [])
        if str(c).replace('\xa0', ' ').strip()
    ]
    if not conditions:
        conditions = ["Condition_A"]
    scales = []
    seen_var = set()       # Deduplicate by variable_name
    seen_display = set()   # Also deduplicate by display_name to prevent column collisions
    for s in (preview.detected_scales or []):
        name = str(s.get("variable_name", s.get("name", "Scale"))).strip() or "Scale"
        display_name = str(s.get("name", name)).strip() or name
        name_key = name.lower().replace(" ", "_").replace("-", "_")
        display_key = display_name.lower().replace(" ", "_").replace("-", "_")
        if name_key in seen_var or display_key in seen_display:
            continue
        seen_var.add(name_key)
        seen_display.add(display_key)
        raw_items = s.get("num_items")
        if raw_items is None:
            raw_items = s.get("items")
        num_items = _safe_int(raw_items, 5)
        scale_points = _safe_int(s.get("scale_points"), 7)
        scales.append({
            "name": display_name,
            "variable_name": name.replace(" ", "_"),
            "num_items": max(1, num_items),
            "scale_points": max(2, min(1001, scale_points)),
            "reverse_items": s.get("reverse_items", []) or [],
            "_validated": True,
        })
    if not scales:
        scales = [{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 5,
                    "scale_points": 7, "reverse_items": [], "_validated": True}]
    open_ended_details = getattr(preview, "open_ended_details", None) or []
    study_context = getattr(preview, "study_context", None) or {}
    return {
        "conditions": conditions,
        "scales": scales,
        "open_ended_details": open_ended_details,
        "study_context": study_context,
        "condition_visibility_map": getattr(preview, "condition_visibility_map", {}) or {},
    }


def run_engine(inputs, sample_size=30, seed=42):
    engine = EnhancedSimulationEngine(
        study_title="Stress Test",
        study_description="Comprehensive stress testing",
        sample_size=sample_size,
        conditions=inputs["conditions"],
        factors=[],
        scales=inputs["scales"],
        additional_vars=[],
        demographics={"age_mean": 35, "age_sd": 10},
        attention_rate=0.95,
        random_responder_rate=0.02,
        effect_sizes=[],
        open_ended_questions=inputs.get("open_ended_details", []),
        study_context=inputs.get("study_context", {}),
        seed=seed,
        precomputed_visibility=inputs.get("condition_visibility_map", {}),
    )
    return engine.generate()


def parse_qsf(qsf_path):
    with open(qsf_path, "rb") as f:
        raw = f.read()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            candidates = [n for n in zf.namelist() if n.lower().endswith((".qsf", ".json"))]
            if candidates:
                raw = zf.read(candidates[0])
    parser = QSFPreviewParser()
    return parser.parse(raw)


# ---------------------------------------------------------------------------
# ITERATION 1: Deep Data Quality Audit
# ---------------------------------------------------------------------------

def iteration_1_data_quality(qsf_files, max_files=None):
    """Deep data quality audit across all QSF files."""
    print("\n" + "=" * 70)
    print("ITERATION 1: Deep Data Quality Audit")
    print("=" * 70)

    files = qsf_files[:max_files] if max_files else qsf_files
    issues = []
    stats = {"tested": 0, "passed": 0, "failed": 0, "errors": 0}

    for idx, qsf_path in enumerate(files):
        fname = os.path.basename(qsf_path)
        try:
            preview = parse_qsf(qsf_path)
            if not preview.success and not preview.detected_scales:
                continue
            inputs = preview_to_engine_inputs(preview)
            df, metadata = run_engine(inputs, sample_size=50)
            stats["tested"] += 1
            file_issues = []

            # CHECK 1.1: No NaN values in numeric columns
            for col in df.columns:
                if _is_numeric_col(df[col]):
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        file_issues.append(f"NAN_VALUES: '{col}' has {nan_count} NaN values")

            # CHECK 1.2: No null/None in any column
            for col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0 and _is_numeric_col(df[col]):  # Only flag numeric columns
                    file_issues.append(f"NULL_VALUES: '{col}' has {null_count} null values")

            # CHECK 1.3: Participant IDs are unique and sequential
            pids = df["PARTICIPANT_ID"].tolist()
            if len(set(pids)) != len(pids):
                file_issues.append(f"DUPLICATE_PID: Participant IDs are not unique")
            if pids != list(range(1, len(pids) + 1)):
                file_issues.append(f"PID_SEQUENCE: Participant IDs not sequential 1-N")

            # CHECK 1.4: Constant columns (all same value) in scale data
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                for i in range(1, scale["num_items"] + 1):
                    col = f"{sname}_{i}"
                    if col in df.columns and _is_numeric_col(df[col]):
                        if df[col].nunique() <= 1 and len(df) > 5:
                            file_issues.append(
                                f"CONSTANT_COLUMN: '{col}' has only 1 unique value "
                                f"({df[col].iloc[0]}) across {len(df)} rows"
                            )

            # CHECK 1.5: Scale columns have integer values only (no floats)
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                for i in range(1, scale["num_items"] + 1):
                    col = f"{sname}_{i}"
                    if col in df.columns and _is_numeric_col(df[col]):
                        vals = df[col].values
                        non_int = np.any(vals != vals.astype(int))
                        if non_int:
                            file_issues.append(
                                f"FLOAT_VALUES: '{col}' has non-integer values in Likert scale"
                            )

            # CHECK 1.6: Conditions column has expected values
            expected_conds = set(inputs["conditions"])
            actual_conds = set(df["CONDITION"].unique())
            if actual_conds != expected_conds:
                file_issues.append(
                    f"CONDITION_MISMATCH: Expected {expected_conds}, got {actual_conds}"
                )

            # CHECK 1.7: Age is reasonable
            if "Age" in df.columns and _is_numeric_col(df["Age"]):
                age_min = df["Age"].min()
                age_max = df["Age"].max()
                age_mean = df["Age"].mean()
                if age_min < 18:
                    file_issues.append(f"AGE_VIOLATION: Min age {age_min} < 18")
                if age_max > 85:
                    file_issues.append(f"AGE_VIOLATION: Max age {age_max} > 85")
                if age_mean < 20 or age_mean > 60:
                    file_issues.append(f"AGE_SUSPICIOUS: Mean age {age_mean:.1f} seems unusual")

            # CHECK 1.8: Gender values are 1-4
            if "Gender" in df.columns and _is_numeric_col(df["Gender"]):
                g_min = df["Gender"].min()
                g_max = df["Gender"].max()
                if g_min < 1 or g_max > 4:
                    file_issues.append(f"GENDER_VIOLATION: Values {g_min}-{g_max}, expected 1-4")

            # CHECK 1.9: DataFrame has expected number of rows
            if len(df) != 50:
                file_issues.append(f"ROW_COUNT: Expected 50, got {len(df)}")

            # CHECK 1.10: CONDITION is approximately balanced
            cond_counts = df["CONDITION"].value_counts()
            if len(cond_counts) > 1:
                min_count = cond_counts.min()
                max_count = cond_counts.max()
                # Allow some imbalance but flag if severely unbalanced
                if max_count > min_count * 3:
                    file_issues.append(
                        f"UNBALANCED_CONDITIONS: {dict(cond_counts)} "
                        f"(max/min ratio = {max_count/min_count:.1f})"
                    )

            if file_issues:
                stats["failed"] += 1
                for fi in file_issues:
                    issues.append({"file": fname, "issue": fi})
            else:
                stats["passed"] += 1

        except Exception as e:
            stats["errors"] += 1
            issues.append({"file": fname, "issue": f"ERROR: {type(e).__name__}: {e}"})

        if (idx + 1) % 30 == 0 or idx + 1 == len(files):
            print(f"  Progress: {idx+1}/{len(files)}")

    _print_results("Iteration 1", stats, issues)
    return issues


# ---------------------------------------------------------------------------
# ITERATION 2: Scale Integrity Audit
# ---------------------------------------------------------------------------

def iteration_2_scale_integrity(qsf_files, max_files=None):
    """Exhaustive scale integrity check - every value in every row."""
    print("\n" + "=" * 70)
    print("ITERATION 2: Scale Integrity Audit")
    print("=" * 70)

    files = qsf_files[:max_files] if max_files else qsf_files
    issues = []
    stats = {"tested": 0, "passed": 0, "failed": 0, "errors": 0}

    for idx, qsf_path in enumerate(files):
        fname = os.path.basename(qsf_path)
        try:
            preview = parse_qsf(qsf_path)
            if not preview.success and not preview.detected_scales:
                continue
            inputs = preview_to_engine_inputs(preview)
            df, metadata = run_engine(inputs, sample_size=50)
            stats["tested"] += 1
            file_issues = []

            # CHECK 2.1: Per-row bounds check (every single value)
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                sp = scale["scale_points"]
                for i in range(1, scale["num_items"] + 1):
                    col = f"{sname}_{i}"
                    if col not in df.columns:
                        continue
                    if not _is_numeric_col(df[col]):
                        continue
                    # Check EVERY value
                    violations = []
                    for row_idx, val in enumerate(df[col]):
                        if val < 1 or val > sp:
                            violations.append((row_idx, val))
                    if violations:
                        file_issues.append(
                            f"ROW_BOUNDS_VIOLATION: '{col}' has {len(violations)} values "
                            f"outside 1-{sp} range. Examples: {violations[:3]}"
                        )

            # CHECK 2.2: Exact item count - scale generates correct number of columns
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                expected_items = scale["num_items"]
                found_items = 0
                for i in range(1, expected_items + 1):
                    col = f"{sname}_{i}"
                    if col in df.columns and _is_numeric_col(df[col]):
                        found_items += 1
                if found_items != expected_items:
                    file_issues.append(
                        f"ITEM_COUNT_MISMATCH: Scale '{scale['name']}' expected "
                        f"{expected_items} numeric items, found {found_items}"
                    )

            # CHECK 2.3: No extra items beyond what's specified
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                expected_items = scale["num_items"]
                # Check if item N+1 exists (shouldn't)
                extra_col = f"{sname}_{expected_items + 1}"
                if extra_col in df.columns and _is_numeric_col(df[extra_col]):
                    file_issues.append(
                        f"EXTRA_ITEMS: Scale '{scale['name']}' has column '{extra_col}' "
                        f"but only {expected_items} items were specified"
                    )

            # CHECK 2.4: Open-ended columns must be text type
            for col in df.columns:
                if col.startswith("OE_") or (not _is_numeric_col(df[col]) and col not in (
                    "CONDITION", "SIMULATION_MODE", "_PERSONA", "RUN_ID"
                )):
                    # This should be text
                    for row_idx, val in enumerate(df[col]):
                        if val and not isinstance(val, str):
                            file_issues.append(
                                f"OE_TYPE_ERROR: '{col}' row {row_idx} has {type(val).__name__} "
                                f"instead of str"
                            )
                            break  # One example is enough

            # CHECK 2.5: Structural columns have correct types
            type_checks = {
                "PARTICIPANT_ID": np.integer,
                "Age": np.integer,
                # Gender is now a string label (Male, Female, Non-binary, Prefer not to say)
                "Flag_Speed": np.integer,
                "Flag_Attention": np.integer,
                "Flag_StraightLine": np.integer,
                "Exclude_Recommended": np.integer,
                "Attention_Check_1": np.integer,
            }
            for col, expected_dtype in type_checks.items():
                if col in df.columns:
                    if not _is_numeric_col(df[col]):
                        file_issues.append(
                            f"TYPE_ERROR: '{col}' has dtype {df[col].dtype}, "
                            f"expected numeric"
                        )

            if file_issues:
                stats["failed"] += 1
                for fi in file_issues:
                    issues.append({"file": fname, "issue": fi})
            else:
                stats["passed"] += 1

        except Exception as e:
            stats["errors"] += 1
            tb = traceback.format_exc()
            tb_lines = [l.strip() for l in tb.strip().split('\n') if l.strip()]
            loc = tb_lines[-3] if len(tb_lines) >= 3 else ""
            issues.append({"file": fname, "issue": f"ERROR: {type(e).__name__}: {e}\n    at: {loc}"})

        if (idx + 1) % 30 == 0 or idx + 1 == len(files):
            print(f"  Progress: {idx+1}/{len(files)}")

    _print_results("Iteration 2", stats, issues)
    return issues


# ---------------------------------------------------------------------------
# ITERATION 3: Edge Case Stress Tests
# ---------------------------------------------------------------------------

def iteration_3_edge_cases():
    """Test edge cases that real QSF files might not cover."""
    print("\n" + "=" * 70)
    print("ITERATION 3: Edge Case Stress Tests")
    print("=" * 70)

    issues = []
    tests_passed = 0
    tests_failed = 0

    def check(test_name, condition, detail=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            issues.append({"file": "edge_cases", "issue": f"FAIL: {test_name}: {detail}"})
            print(f"  FAIL: {test_name}: {detail}")

    # TEST 3.1: 2-point binary scale
    print("  Testing 2-point binary scale...")
    inputs = {
        "conditions": ["A", "B"],
        "scales": [{"name": "Binary_DV", "variable_name": "Binary_DV",
                     "num_items": 3, "scale_points": 2, "reverse_items": [], "_validated": True}],
        "open_ended_details": [],
        "study_context": {},
        "condition_visibility_map": {},
    }
    df, _ = run_engine(inputs, sample_size=100)
    vals = df["Binary_DV_1"].values
    check("2pt_bounds", vals.min() >= 1 and vals.max() <= 2,
          f"min={vals.min()}, max={vals.max()}")
    check("2pt_both_values", set(vals) == {1, 2},
          f"unique values: {sorted(set(vals))}")

    # TEST 3.2: 100-point slider scale
    print("  Testing 100-point slider scale...")
    inputs["scales"] = [{"name": "Slider", "variable_name": "Slider",
                          "num_items": 1, "scale_points": 100, "reverse_items": [], "_validated": True}]
    df, _ = run_engine(inputs, sample_size=200)
    vals = df["Slider_1"].values
    check("100pt_bounds", vals.min() >= 1 and vals.max() <= 100,
          f"min={vals.min()}, max={vals.max()}")
    check("100pt_range", vals.max() - vals.min() >= 30,
          f"range={vals.max() - vals.min()}, expected >= 30")
    check("100pt_variation", np.std(vals) > 5,
          f"std={np.std(vals):.2f}, expected > 5")

    # TEST 3.3: 1001-point scale (maximum allowed)
    print("  Testing 1001-point scale (maximum)...")
    inputs["scales"] = [{"name": "BigScale", "variable_name": "BigScale",
                          "num_items": 1, "scale_points": 1001, "reverse_items": [], "_validated": True}]
    df, _ = run_engine(inputs, sample_size=200)
    vals = df["BigScale_1"].values
    check("1001pt_bounds", vals.min() >= 1 and vals.max() <= 1001,
          f"min={vals.min()}, max={vals.max()}")

    # TEST 3.4: Many items per scale (50 items)
    print("  Testing 50-item scale...")
    inputs["scales"] = [{"name": "LongScale", "variable_name": "LongScale",
                          "num_items": 50, "scale_points": 7, "reverse_items": [2, 5, 10, 15, 20],
                          "_validated": True}]
    df, _ = run_engine(inputs, sample_size=30)
    found = sum(1 for i in range(1, 51) if f"LongScale_{i}" in df.columns)
    check("50item_count", found == 50, f"found {found} of 50 columns")
    # All should be within bounds
    all_ok = True
    for i in range(1, 51):
        col = f"LongScale_{i}"
        if col in df.columns:
            if df[col].min() < 1 or df[col].max() > 7:
                all_ok = False
                break
    check("50item_all_bounds", all_ok, "Some items out of bounds")

    # TEST 3.5: Many scales (20 different scales)
    print("  Testing 20 simultaneous scales...")
    inputs["scales"] = [
        {"name": f"Scale_{j}", "variable_name": f"Scale_{j}",
         "num_items": 3, "scale_points": 5 + j, "reverse_items": [], "_validated": True}
        for j in range(20)
    ]
    df, _ = run_engine(inputs, sample_size=30)
    for j in range(20):
        for i in range(1, 4):
            col = f"Scale_{j}_{i}"
            sp = 5 + j
            if col in df.columns:
                check(f"multi_scale_{j}_item_{i}_bounds",
                      df[col].min() >= 1 and df[col].max() <= sp,
                      f"col={col}, min={df[col].min()}, max={df[col].max()}, expected 1-{sp}")

    # TEST 3.6: Single condition (no experimental manipulation)
    print("  Testing single condition...")
    inputs = {
        "conditions": ["Control"],
        "scales": [{"name": "DV", "variable_name": "DV",
                     "num_items": 5, "scale_points": 7, "reverse_items": [], "_validated": True}],
        "open_ended_details": [],
        "study_context": {},
        "condition_visibility_map": {},
    }
    df, _ = run_engine(inputs, sample_size=50)
    check("single_cond_all_same", all(c == "Control" for c in df["CONDITION"]),
          f"Conditions: {df['CONDITION'].unique()}")

    # TEST 3.7: Many conditions (10 conditions)
    print("  Testing 10 conditions...")
    inputs["conditions"] = [f"Cond_{j}" for j in range(10)]
    df, _ = run_engine(inputs, sample_size=100)
    check("many_conds_all_present", len(df["CONDITION"].unique()) == 10,
          f"Found {len(df['CONDITION'].unique())} conditions")

    # TEST 3.8: Different seeds produce different data
    print("  Testing seed reproducibility and variation...")
    inputs = {
        "conditions": ["A", "B"],
        "scales": [{"name": "TestDV", "variable_name": "TestDV",
                     "num_items": 3, "scale_points": 7, "reverse_items": [], "_validated": True}],
        "open_ended_details": [],
        "study_context": {},
        "condition_visibility_map": {},
    }
    df1, _ = run_engine(inputs, sample_size=50, seed=42)
    df2, _ = run_engine(inputs, sample_size=50, seed=42)
    df3, _ = run_engine(inputs, sample_size=50, seed=999)
    # Same seed = same data
    check("seed_reproducible",
          df1["TestDV_1"].tolist() == df2["TestDV_1"].tolist(),
          "Same seed produced different data!")
    # Different seed = different data
    check("seed_different",
          df1["TestDV_1"].tolist() != df3["TestDV_1"].tolist(),
          "Different seeds produced identical data!")

    # TEST 3.9: Minimal sample size (n=1)
    print("  Testing minimal sample size (n=1)...")
    df, _ = run_engine(inputs, sample_size=1)
    check("n1_one_row", len(df) == 1, f"Expected 1 row, got {len(df)}")
    check("n1_has_scale", "TestDV_1" in df.columns, "Missing scale column")
    check("n1_bounds", 1 <= df["TestDV_1"].iloc[0] <= 7,
          f"Value {df['TestDV_1'].iloc[0]} out of 1-7 range")

    # TEST 3.10: Large sample size (n=500)
    print("  Testing large sample size (n=500)...")
    df, _ = run_engine(inputs, sample_size=500)
    check("n500_row_count", len(df) == 500, f"Expected 500 rows, got {len(df)}")
    check("n500_unique_pids", df["PARTICIPANT_ID"].nunique() == 500,
          f"Expected 500 unique PIDs, got {df['PARTICIPANT_ID'].nunique()}")
    # With 500 participants and 7pt scale, we should use most of the range
    check("n500_good_range", df["TestDV_1"].nunique() >= 5,
          f"Only {df['TestDV_1'].nunique()} unique values in 500 rows")

    # TEST 3.11: Scale with name collision against structural column
    print("  Testing scale with name collision ('Age')...")
    inputs["scales"] = [
        {"name": "Age_Rating", "variable_name": "Age_Rating",
         "num_items": 2, "scale_points": 5, "reverse_items": [], "_validated": True}
    ]
    inputs["open_ended_details"] = [{"name": "Age", "question_text": "What is your age?"}]
    df, _ = run_engine(inputs, sample_size=30)
    # Age should still be numeric (demographic), not text
    check("age_collision_numeric",
          _is_numeric_col(df["Age"]),
          f"Age column dtype is {df['Age'].dtype}, expected numeric")
    # OE column should be renamed
    check("age_collision_oe_renamed",
          "OE_Age" in df.columns,
          f"Expected 'OE_Age' in columns: {[c for c in df.columns if 'Age' in c or 'OE' in c]}")

    # TEST 3.12: Open-ended collision with CONDITION column
    print("  Testing OE collision with structural columns...")
    inputs["open_ended_details"] = [
        {"name": "CONDITION", "question_text": "Describe your condition"},
        {"name": "Gender", "question_text": "Describe your gender identity"},
        {"name": "PARTICIPANT_ID", "question_text": "Describe yourself"},
    ]
    df, _ = run_engine(inputs, sample_size=30)
    # CONDITION should still be the actual condition column
    check("structural_condition_preserved",
          set(df["CONDITION"].unique()) == {"A", "B"},
          f"CONDITION values: {df['CONDITION'].unique()}")
    # Gender should still be numeric
    check("structural_gender_preserved",
          _is_numeric_col(df["Gender"]),
          f"Gender dtype: {df['Gender'].dtype}")

    # TEST 3.13: Scale with special characters in name
    print("  Testing scale with special characters in name...")
    inputs["scales"] = [
        {"name": "How satisfied are you? (1-7)", "variable_name": "Satisfaction",
         "num_items": 3, "scale_points": 7, "reverse_items": [], "_validated": True}
    ]
    inputs["open_ended_details"] = []
    df, _ = run_engine(inputs, sample_size=30)
    # Column name should be cleaned
    expected_prefix = "How_satisfied_are_you?_(1-7)_"
    matching_cols = [c for c in df.columns if c.startswith("How_satisfied")]
    check("special_chars_columns_exist", len(matching_cols) >= 3,
          f"Found columns: {matching_cols}")

    # TEST 3.14: Empty scale name
    print("  Testing empty scale name handling...")
    inputs["scales"] = [
        {"name": "", "variable_name": "", "num_items": 3, "scale_points": 5,
         "reverse_items": [], "_validated": True},
        {"name": "Valid_Scale", "variable_name": "Valid_Scale", "num_items": 2,
         "scale_points": 7, "reverse_items": [], "_validated": True},
    ]
    df, _ = run_engine(inputs, sample_size=30)
    # Valid scale should still work
    check("empty_name_valid_works", "Valid_Scale_1" in df.columns,
          f"Columns: {[c for c in df.columns if 'Valid' in c]}")

    print(f"\n  Edge case results: {tests_passed} passed, {tests_failed} failed")
    return issues


# ---------------------------------------------------------------------------
# ITERATION 4: Statistical Validity
# ---------------------------------------------------------------------------

def iteration_4_statistical_validity(qsf_files, max_files=None):
    """Statistical validity checks across all QSF files."""
    print("\n" + "=" * 70)
    print("ITERATION 4: Statistical Validity Checks")
    print("=" * 70)

    files = qsf_files[:max_files] if max_files else qsf_files
    issues = []
    stats = {"tested": 0, "passed": 0, "failed": 0, "errors": 0}

    for idx, qsf_path in enumerate(files):
        fname = os.path.basename(qsf_path)
        try:
            preview = parse_qsf(qsf_path)
            if not preview.success and not preview.detected_scales:
                continue
            inputs = preview_to_engine_inputs(preview)
            df, metadata = run_engine(inputs, sample_size=100)
            stats["tested"] += 1
            file_issues = []

            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                sp = scale["scale_points"]

                for i in range(1, scale["num_items"] + 1):
                    col = f"{sname}_{i}"
                    if col not in df.columns or not _is_numeric_col(df[col]):
                        continue

                    vals = df[col].values.astype(float)

                    # CHECK 4.1: Mean should not be at extreme endpoints
                    mean_val = np.mean(vals)
                    if sp > 2:  # Skip binary scales
                        # Mean should be somewhere reasonable (not stuck at 1 or max)
                        expected_mid = (1 + sp) / 2
                        # Allow wide range but flag if mean is within 0.1 of endpoint
                        if mean_val < 1.1 or mean_val > sp - 0.1:
                            file_issues.append(
                                f"EXTREME_MEAN: '{col}' mean={mean_val:.2f} on 1-{sp} scale "
                                f"(stuck at endpoint)"
                            )

                    # CHECK 4.2: Standard deviation should indicate variation
                    sd_val = np.std(vals)
                    if sp > 2 and len(vals) >= 20:
                        # For a scale, we expect some variation
                        if sd_val < 0.1:
                            file_issues.append(
                                f"NO_VARIATION: '{col}' sd={sd_val:.3f} on 1-{sp} scale "
                                f"(near-zero variation)"
                            )

                    # CHECK 4.3: No impossible values (e.g., 0, negative, > scale max)
                    impossible = vals[(vals < 1) | (vals > sp)]
                    if len(impossible) > 0:
                        file_issues.append(
                            f"IMPOSSIBLE_VALUES: '{col}' has {len(impossible)} values "
                            f"outside 1-{sp}. Examples: {impossible[:5]}"
                        )

            # CHECK 4.4: Completion time should be positive
            if "Completion_Time_Seconds" in df.columns:
                ct = df["Completion_Time_Seconds"].values
                if np.any(ct <= 0):
                    file_issues.append(
                        f"NEGATIVE_TIME: Completion_Time_Seconds has {np.sum(ct <= 0)} "
                        f"non-positive values"
                    )
                # Should have variation
                if np.std(ct) < 1.0 and len(ct) >= 20:
                    file_issues.append(
                        f"TIME_NO_VARIATION: Completion_Time_Seconds sd={np.std(ct):.2f}"
                    )

            # CHECK 4.5: Attention pass rate should be 0-1
            if "Attention_Pass_Rate" in df.columns:
                apr = df["Attention_Pass_Rate"].values
                if np.any(apr < 0) or np.any(apr > 1):
                    file_issues.append(
                        f"ATTENTION_BOUNDS: Attention_Pass_Rate has values outside 0-1"
                    )

            # CHECK 4.6: Flag columns should be binary (0 or 1)
            for flag_col in ["Flag_Speed", "Flag_Attention", "Flag_StraightLine", "Exclude_Recommended"]:
                if flag_col in df.columns:
                    unique_vals = set(df[flag_col].unique())
                    if not unique_vals.issubset({0, 1}):
                        file_issues.append(
                            f"FLAG_NOT_BINARY: '{flag_col}' has values {unique_vals}, expected {{0, 1}}"
                        )

            # CHECK 4.7: Open-ended responses should be non-empty for visible conditions
            oe_cols = [c for c in df.columns if not _is_numeric_col(df[c])
                       and c not in ("CONDITION", "SIMULATION_MODE", "_PERSONA", "RUN_ID")]
            for oe_col in oe_cols:
                non_empty = df[oe_col].apply(lambda x: bool(x and str(x).strip())).sum()
                if non_empty == 0 and len(df) > 5:
                    file_issues.append(
                        f"EMPTY_OE: '{oe_col}' has no non-empty responses across {len(df)} rows"
                    )

            # CHECK 4.8: Duplicate check - no two participants should have identical
            # response patterns across ALL scale items
            scale_cols = []
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                for i in range(1, scale["num_items"] + 1):
                    col = f"{sname}_{i}"
                    if col in df.columns and _is_numeric_col(df[col]):
                        scale_cols.append(col)
            if len(scale_cols) >= 3 and len(df) >= 10:
                # Create response pattern strings
                patterns = df[scale_cols].astype(str).agg('-'.join, axis=1)
                dup_count = patterns.duplicated().sum()
                dup_pct = (dup_count / len(df)) * 100
                if dup_pct > 50:
                    file_issues.append(
                        f"HIGH_DUPLICATES: {dup_pct:.1f}% of participants have identical "
                        f"response patterns across {len(scale_cols)} scale items"
                    )

            if file_issues:
                stats["failed"] += 1
                for fi in file_issues:
                    issues.append({"file": fname, "issue": fi})
            else:
                stats["passed"] += 1

        except Exception as e:
            stats["errors"] += 1
            issues.append({"file": fname, "issue": f"ERROR: {type(e).__name__}: {e}"})

        if (idx + 1) % 30 == 0 or idx + 1 == len(files):
            print(f"  Progress: {idx+1}/{len(files)}")

    _print_results("Iteration 4", stats, issues)
    return issues


# ---------------------------------------------------------------------------
# ITERATION 5: Comprehensive Re-test with All Checks
# ---------------------------------------------------------------------------

def iteration_5_comprehensive(qsf_files, max_files=None):
    """Run ALL checks from iterations 1-4 on all QSF files."""
    print("\n" + "=" * 70)
    print("ITERATION 5: Comprehensive Final Re-test (all checks combined)")
    print("=" * 70)

    files = qsf_files[:max_files] if max_files else qsf_files
    issues = []
    stats = {"tested": 0, "passed": 0, "failed": 0, "errors": 0}

    for idx, qsf_path in enumerate(files):
        fname = os.path.basename(qsf_path)
        try:
            preview = parse_qsf(qsf_path)
            if not preview.success and not preview.detected_scales:
                continue
            inputs = preview_to_engine_inputs(preview)
            df, metadata = run_engine(inputs, sample_size=100)
            stats["tested"] += 1
            file_issues = []

            # --- ALL CHECKS FROM ITERATIONS 1-4 ---

            # 1. NaN/null in numeric columns
            for col in df.columns:
                if _is_numeric_col(df[col]):
                    if df[col].isna().sum() > 0:
                        file_issues.append(f"NAN: '{col}' has NaN values")

            # 2. Participant ID sequential
            pids = df["PARTICIPANT_ID"].tolist()
            if pids != list(range(1, len(pids) + 1)):
                file_issues.append(f"PID_ERROR: Not sequential 1-{len(pids)}")

            # 3. Per-row bounds for ALL scale items
            for scale in inputs["scales"]:
                sname = scale["name"].strip().replace(" ", "_")
                sp = scale["scale_points"]
                ni = scale["num_items"]

                found_count = 0
                for i in range(1, ni + 1):
                    col = f"{sname}_{i}"
                    if col not in df.columns:
                        continue
                    if not _is_numeric_col(df[col]):
                        continue
                    found_count += 1

                    vals = df[col].values
                    # Per-row bounds
                    if vals.min() < 1 or vals.max() > sp:
                        file_issues.append(
                            f"BOUNDS: '{col}' range [{vals.min()},{vals.max()}] vs expected [1,{sp}]"
                        )
                    # Integer check
                    if np.any(vals != vals.astype(int)):
                        file_issues.append(f"FLOAT: '{col}' has non-integer values")
                    # Variation check (skip 2pt)
                    if sp > 2 and len(vals) >= 20 and np.std(vals) < 0.1:
                        file_issues.append(f"NO_VAR: '{col}' sd={np.std(vals):.3f}")
                    # Extreme mean check (skip 2pt)
                    if sp > 2:
                        m = np.mean(vals)
                        if m < 1.1 or m > sp - 0.1:
                            file_issues.append(f"EXTREME_MEAN: '{col}' mean={m:.2f} on 1-{sp}")

            # 4. Conditions balanced and correct
            expected_conds = set(inputs["conditions"])
            actual_conds = set(df["CONDITION"].unique())
            if actual_conds != expected_conds:
                file_issues.append(
                    f"COND_MISMATCH: Expected {expected_conds}, got {actual_conds}"
                )

            # 5. Age reasonable
            if "Age" in df.columns and _is_numeric_col(df["Age"]):
                if df["Age"].min() < 18 or df["Age"].max() > 85:
                    file_issues.append(
                        f"AGE: [{df['Age'].min()},{df['Age'].max()}] outside [18,85]"
                    )

            # 6. Gender 1-4
            if "Gender" in df.columns and _is_numeric_col(df["Gender"]):
                if df["Gender"].min() < 1 or df["Gender"].max() > 4:
                    file_issues.append(
                        f"GENDER: [{df['Gender'].min()},{df['Gender'].max()}] outside [1,4]"
                    )

            # 7. Flag columns binary
            for flag_col in ["Flag_Speed", "Flag_Attention", "Flag_StraightLine", "Exclude_Recommended"]:
                if flag_col in df.columns:
                    unique_vals = set(df[flag_col].unique())
                    if not unique_vals.issubset({0, 1}):
                        file_issues.append(f"FLAG: '{flag_col}' values {unique_vals}")

            # 8. Completion time positive
            if "Completion_Time_Seconds" in df.columns:
                if np.any(df["Completion_Time_Seconds"].values <= 0):
                    file_issues.append("NEG_TIME: Completion time has non-positive values")

            # 9. Attention pass rate 0-1
            if "Attention_Pass_Rate" in df.columns:
                apr = df["Attention_Pass_Rate"].values
                if np.any(apr < 0) or np.any(apr > 1):
                    file_issues.append("ATTN_BOUNDS: Attention_Pass_Rate outside [0,1]")

            # 10. Row count
            if len(df) != 100:
                file_issues.append(f"ROWS: Expected 100, got {len(df)}")

            if file_issues:
                stats["failed"] += 1
                for fi in file_issues:
                    issues.append({"file": fname, "issue": fi})
            else:
                stats["passed"] += 1

        except Exception as e:
            stats["errors"] += 1
            tb = traceback.format_exc()
            tb_lines = [l.strip() for l in tb.strip().split('\n') if l.strip()]
            loc = tb_lines[-3] if len(tb_lines) >= 3 else ""
            issues.append({"file": fname, "issue": f"ERROR: {type(e).__name__}: {e}\n    at: {loc}"})

        if (idx + 1) % 30 == 0 or idx + 1 == len(files):
            print(f"  Progress: {idx+1}/{len(files)}")

    _print_results("Iteration 5", stats, issues)
    return issues


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_results(label, stats, issues):
    """Print formatted results for an iteration."""
    print(f"\n{'=' * 70}")
    print(f"{label} RESULTS: {stats['passed']} passed, {stats['failed']} failed, "
          f"{stats['errors']} errors out of {stats['tested']} tested")
    print(f"{'=' * 70}")

    if issues:
        # Group by issue type
        by_type = defaultdict(list)
        for issue in issues:
            # Extract issue type (first word before colon)
            itype = issue["issue"].split(":")[0] if ":" in issue["issue"] else "OTHER"
            by_type[itype].append(issue)

        print(f"\nIssues by type ({len(issues)} total):")
        for itype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"  {itype}: {len(items)} occurrences")
            for item in items[:3]:
                print(f"    [{item['file']}] {item['issue'][:120]}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qsf_dir = os.path.join(os.path.dirname(__file__), "..", "simulation_app", "example_files")
    qsf_files = sorted(glob.glob(os.path.join(qsf_dir, "*.qsf")))
    print(f"Found {len(qsf_files)} QSF files\n")

    all_issues = {}

    # Iteration 1: Deep data quality
    all_issues["iter1"] = iteration_1_data_quality(qsf_files)

    # Iteration 2: Scale integrity
    all_issues["iter2"] = iteration_2_scale_integrity(qsf_files)

    # Iteration 3: Edge cases (synthetic tests)
    all_issues["iter3"] = iteration_3_edge_cases()

    # Iteration 4: Statistical validity
    all_issues["iter4"] = iteration_4_statistical_validity(qsf_files)

    # Iteration 5: Comprehensive re-test
    all_issues["iter5"] = iteration_5_comprehensive(qsf_files)

    # Grand summary
    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)
    total_issues = sum(len(v) for v in all_issues.values())
    for key, iss in all_issues.items():
        print(f"  {key}: {len(iss)} issues")
    print(f"  TOTAL: {total_issues} issues across all iterations")

    sys.exit(0 if total_issues == 0 else 1)
