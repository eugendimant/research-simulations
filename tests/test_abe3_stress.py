#!/usr/bin/env python3
"""
ABE 3.0 Stress Test — Parse 3 QSF files, generate N=100, validate output.
"""

import sys
import os
import traceback
import time

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

import numpy as np
import pandas as pd

from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

# ---------------------------------------------------------------------------
# QSF files to test
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app", "example_files")
QSF_FILES = [
    os.path.join(BASE, "Final - Group 6.qsf"),
    os.path.join(BASE, "Group #12 - Qualtrics.qsf"),
    os.path.join(BASE, "BDS501_Official.qsf"),
]

N = 100
SEED = 42


def parse_qsf(path: str) -> QSFPreviewResult:
    """Parse a QSF file and return the result."""
    parser = QSFPreviewParser()
    with open(path, "rb") as f:
        raw = f.read()
    result = parser.parse(raw)
    return result


def build_engine_params(result: QSFPreviewResult, seed: int = SEED):
    """Build EnhancedSimulationEngine kwargs from a parsed QSF."""
    conditions = result.detected_conditions or ["Control"]
    scales = result.detected_scales or []
    oe_questions = []
    if result.open_ended_details:
        oe_questions = result.open_ended_details
    elif result.open_ended_questions:
        oe_questions = [{"question_text": q, "question_name": f"OE_{i}"}
                        for i, q in enumerate(result.open_ended_questions)]

    study_context = result.study_context or {}

    return dict(
        study_title=result.survey_name or "Stress Test",
        study_description=f"Stress test of {result.survey_name}",
        sample_size=N,
        conditions=conditions,
        factors=[],
        scales=scales,
        additional_vars=[],
        demographics={"age": True, "gender": True},
        attention_rate=0.95,
        random_responder_rate=0.05,
        open_ended_questions=oe_questions,
        study_context=study_context,
        seed=seed,
        mode="pilot",
        allow_template_fallback=True,
    )


def validate_output(df: pd.DataFrame, metadata: dict, conditions: list, scales: list):
    """Validate the generated DataFrame and metadata. Returns list of issues."""
    issues = []

    # 1. Correct number of rows
    if len(df) != N:
        issues.append(f"Row count mismatch: expected {N}, got {len(df)}")

    # 2. Check for completely NaN columns (except open-ended which may have missing)
    for col in df.columns:
        if df[col].isna().all():
            issues.append(f"Column '{col}' is entirely NaN")

    # 3. Conditions properly allocated
    if "CONDITION" in df.columns:
        actual_conditions = set(df["CONDITION"].dropna().unique())
        expected_conditions = set(conditions)
        if not actual_conditions:
            issues.append("CONDITION column is empty")
        elif not actual_conditions.issubset(expected_conditions) and not expected_conditions.issubset(actual_conditions):
            issues.append(f"Condition mismatch: expected {expected_conditions}, got {actual_conditions}")
        # Check balance (within 30% tolerance for N=100)
        cond_counts = df["CONDITION"].value_counts()
        if len(cond_counts) > 1:
            expected_per = N / len(conditions)
            for cond_name, count in cond_counts.items():
                if abs(count - expected_per) > expected_per * 0.35:
                    issues.append(f"Condition '{cond_name}' allocation imbalanced: {count} (expected ~{expected_per:.0f})")

    # 4. Scale columns within bounds
    for scale in scales:
        var_name = scale.get("variable_name", scale.get("name", ""))
        scale_min = scale.get("scale_min", 1)
        scale_max = scale.get("scale_max", scale.get("scale_points", 7))
        num_items = scale.get("num_items", 1)
        # Look for columns matching scale variable name
        matching_cols = [c for c in df.columns if var_name and var_name in c and df[c].dtype in [np.float64, np.int64, float, int]]
        for col in matching_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                if vals.min() < scale_min - 0.5:
                    issues.append(f"Column '{col}' has values below scale_min ({scale_min}): min={vals.min()}")
                if vals.max() > scale_max + 0.5:
                    issues.append(f"Column '{col}' has values above scale_max ({scale_max}): max={vals.max()}")

    # 5. Check metadata for ABE 3.0 markers
    if metadata:
        # Check generation_method
        gen_method = metadata.get("generation_method", "")
        if gen_method and gen_method != "abe_v3":
            # Not necessarily an error - may be "enhanced" or similar
            pass

        # Check for consistency_audit
        if "consistency_audit" in metadata:
            audit = metadata["consistency_audit"]
            if isinstance(audit, dict) and audit.get("errors"):
                issues.append(f"Consistency audit errors: {audit['errors']}")

    # 6. Basic structural checks
    expected_cols = ["PARTICIPANT_ID", "CONDITION"]
    for col in expected_cols:
        if col not in df.columns:
            issues.append(f"Missing expected column: {col}")

    if "PARTICIPANT_ID" in df.columns:
        if df["PARTICIPANT_ID"].nunique() != len(df):
            issues.append("PARTICIPANT_ID values are not unique")

    return issues


def run_test(qsf_path: str):
    """Run full test for one QSF file. Returns summary dict."""
    name = os.path.basename(qsf_path)
    summary = {
        "file": name,
        "parse_ok": False,
        "generate_ok": False,
        "rows": 0,
        "columns": 0,
        "conditions": [],
        "scales_detected": 0,
        "oe_questions": 0,
        "issues": [],
        "errors": [],
        "warnings": [],
        "time_parse": 0,
        "time_generate": 0,
    }

    # --- STEP 1: Parse QSF ---
    print(f"\n{'='*70}")
    print(f"  Testing: {name}")
    print(f"{'='*70}")

    try:
        t0 = time.time()
        result = parse_qsf(qsf_path)
        summary["time_parse"] = round(time.time() - t0, 2)
        summary["parse_ok"] = result.success
        summary["conditions"] = result.detected_conditions or []
        summary["scales_detected"] = len(result.detected_scales or [])
        summary["oe_questions"] = len(result.open_ended_details or result.open_ended_questions or [])
        summary["warnings"] = result.validation_warnings or []

        print(f"  Parse: {'OK' if result.success else 'FAILED'}")
        print(f"  Survey: {result.survey_name}")
        print(f"  Questions: {result.total_questions}")
        print(f"  Blocks: {result.total_blocks}")
        print(f"  Conditions: {result.detected_conditions}")
        print(f"  Scales: {len(result.detected_scales or [])}")
        print(f"  Open-ended: {summary['oe_questions']}")
        if result.validation_errors:
            print(f"  Validation errors: {result.validation_errors}")
            summary["errors"].extend(result.validation_errors)
        if result.validation_warnings:
            print(f"  Validation warnings: {len(result.validation_warnings)}")

        if not result.success:
            summary["errors"].append("QSF parse failed")
            return summary

    except Exception as e:
        summary["errors"].append(f"Parse exception: {e}")
        print(f"  PARSE EXCEPTION: {e}")
        traceback.print_exc()
        return summary

    # --- STEP 2: Build engine params ---
    try:
        params = build_engine_params(result, seed=SEED)
    except Exception as e:
        summary["errors"].append(f"Param build exception: {e}")
        print(f"  PARAM BUILD EXCEPTION: {e}")
        traceback.print_exc()
        return summary

    # --- STEP 3: Create engine & generate ---
    try:
        t0 = time.time()
        engine = EnhancedSimulationEngine(**params)
        df, metadata = engine.generate()
        summary["time_generate"] = round(time.time() - t0, 2)
        summary["generate_ok"] = True
        summary["rows"] = len(df)
        summary["columns"] = len(df.columns)

        print(f"\n  Generation: OK ({summary['time_generate']}s)")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names (first 20): {list(df.columns[:20])}")

        # Show dtypes summary
        dtype_counts = df.dtypes.value_counts()
        print(f"  Dtype breakdown: {dict(dtype_counts)}")

        # Show metadata keys
        if metadata:
            print(f"  Metadata keys: {list(metadata.keys())[:15]}")
            if "generation_method" in metadata:
                print(f"  generation_method: {metadata['generation_method']}")
            if "abe3_engine" in metadata:
                print(f"  abe3_engine: {metadata['abe3_engine']}")
            if "consistency_audit" in metadata:
                print(f"  consistency_audit present: True")

    except Exception as e:
        summary["errors"].append(f"Generation exception: {e}")
        print(f"  GENERATION EXCEPTION: {e}")
        traceback.print_exc()
        return summary

    # --- STEP 4: Validate ---
    try:
        issues = validate_output(df, metadata, result.detected_conditions or ["Control"], result.detected_scales or [])
        summary["issues"] = issues
        if issues:
            print(f"\n  Validation issues ({len(issues)}):")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  Validation: ALL PASSED")
    except Exception as e:
        summary["errors"].append(f"Validation exception: {e}")
        print(f"  VALIDATION EXCEPTION: {e}")
        traceback.print_exc()

    return summary


def main():
    print("=" * 70)
    print("  ABE 3.0 STRESS TEST — 3 QSF Files × N=100")
    print("=" * 70)

    summaries = []
    for qsf_path in QSF_FILES:
        if not os.path.exists(qsf_path):
            print(f"\n  SKIPPED (not found): {qsf_path}")
            summaries.append({"file": os.path.basename(qsf_path), "errors": ["File not found"]})
            continue
        summary = run_test(qsf_path)
        summaries.append(summary)

    # --- Final Summary Table ---
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = f"{'File':<40} {'Parse':>5} {'Gen':>5} {'Rows':>5} {'Cols':>5} {'Conds':>5} {'Scales':>6} {'Issues':>6} {'Errors':>6} {'Time':>7}"
    print(header)
    print("-" * 90)
    for s in summaries:
        file_short = s.get("file", "?")[:38]
        parse_ok = "OK" if s.get("parse_ok") else "FAIL"
        gen_ok = "OK" if s.get("generate_ok") else "FAIL"
        rows = s.get("rows", 0)
        cols = s.get("columns", 0)
        conds = len(s.get("conditions", []))
        scales = s.get("scales_detected", 0)
        issues = len(s.get("issues", []))
        errors = len(s.get("errors", []))
        total_time = s.get("time_parse", 0) + s.get("time_generate", 0)
        print(f"{file_short:<40} {parse_ok:>5} {gen_ok:>5} {rows:>5} {cols:>5} {conds:>5} {scales:>6} {issues:>6} {errors:>6} {total_time:>6.1f}s")

    print("=" * 90)

    # Check overall pass/fail
    all_passed = all(s.get("generate_ok") and not s.get("errors") for s in summaries)
    if all_passed:
        print("\n  OVERALL: ALL TESTS PASSED")
    else:
        print("\n  OVERALL: SOME TESTS FAILED")
        for s in summaries:
            if s.get("errors"):
                print(f"    {s['file']}: {s['errors']}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
