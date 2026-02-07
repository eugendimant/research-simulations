"""
End-to-end tests for scale data generation.

Tests that the simulation engine correctly generates data matching
user-specified scale configurations across a wide range of scenarios.

Run with: python3 -m pytest tests/test_scale_generation.py -v
Or directly: python3 tests/test_scale_generation.py
"""
import sys
import os

# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

import numpy as np
import pandas as pd
from utils.enhanced_simulation_engine import EnhancedSimulationEngine


def _make_engine(scales, sample_size=50, conditions=None, seed=42):
    """Helper to create engine with given scales."""
    if conditions is None:
        conditions = ["Control", "Treatment"]
    return EnhancedSimulationEngine(
        study_title="Test Study",
        study_description="Testing scale generation",
        sample_size=sample_size,
        conditions=conditions,
        factors=[],
        scales=scales,
        additional_vars=[],
        demographics={"age_mean": 35, "age_sd": 10},
        attention_rate=0.95,
        random_responder_rate=0.02,
        effect_sizes=[],
        open_ended_questions=[],
        seed=seed,
    )


def test_7_point_likert():
    """Standard 7-point Likert scale should produce values 1-7."""
    scales = [{"name": "Satisfaction", "num_items": 5, "scale_points": 7, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales)
    df, metadata = engine.generate()

    for i in range(1, 6):
        col = f"Satisfaction_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
        assert df[col].max() <= 7, f"{col} max={df[col].max()} > 7"
        assert df[col].dtype in [np.int64, np.int32, int], f"{col} not integer type"

    print("PASS: test_7_point_likert")


def test_100_point_scale():
    """100-point scale (e.g., VAS slider) should produce values 1-100."""
    scales = [{"name": "Decision", "num_items": 1, "scale_points": 100, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales, sample_size=200)
    df, metadata = engine.generate()

    col = "Decision_1"
    assert col in df.columns, f"Missing column {col}"
    assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
    assert df[col].max() <= 100, f"{col} max={df[col].max()} > 100"

    # Critical: values should USE the full range, not be capped at 11
    value_range = df[col].max() - df[col].min()
    assert value_range >= 30, (
        f"CRITICAL: {col} range is only {value_range} (min={df[col].min()}, max={df[col].max()}). "
        f"Expected range should span at least 30 for a 100-point scale with 200 participants."
    )

    # Check verification report
    verification = metadata.get("scale_verification", [])
    assert len(verification) >= 1, "No scale verification report"
    assert verification[0]["specified_scale_points"] == 100
    assert verification[0]["status"] == "OK", f"Verification status: {verification[0]['status']}"

    print(f"PASS: test_100_point_scale (range: {df[col].min()}-{df[col].max()})")


def test_101_point_slider():
    """101-point slider (0-100 style, mapped to 1-101) should use full range."""
    scales = [{"name": "VAS_Pain", "num_items": 1, "scale_points": 101, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales, sample_size=200)
    df, metadata = engine.generate()

    col = "VAS_Pain_1"
    assert col in df.columns, f"Missing column {col}"
    assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
    assert df[col].max() <= 101, f"{col} max={df[col].max()} > 101"

    value_range = df[col].max() - df[col].min()
    assert value_range >= 30, (
        f"CRITICAL: {col} range is only {value_range}. Expected wider distribution for 101-point scale."
    )

    print(f"PASS: test_101_point_slider (range: {df[col].min()}-{df[col].max()})")


def test_2_point_binary():
    """Binary scale (2 points) should produce only 1 or 2."""
    scales = [{"name": "YesNo", "num_items": 3, "scale_points": 2, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales)
    df, metadata = engine.generate()

    for i in range(1, 4):
        col = f"YesNo_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
        assert df[col].max() <= 2, f"{col} max={df[col].max()} > 2"
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({1, 2}), f"{col} has unexpected values: {unique_vals}"

    print("PASS: test_2_point_binary")


def test_5_point_likert():
    """5-point Likert scale should produce values 1-5."""
    scales = [{"name": "Agreement", "num_items": 4, "scale_points": 5, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales)
    df, metadata = engine.generate()

    for i in range(1, 5):
        col = f"Agreement_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
        assert df[col].max() <= 5, f"{col} max={df[col].max()} > 5"

    print("PASS: test_5_point_likert")


def test_10_point_scale():
    """10-point scale should produce values 1-10."""
    scales = [{"name": "NPS", "num_items": 1, "scale_points": 10, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales, sample_size=100)
    df, metadata = engine.generate()

    col = "NPS_1"
    assert col in df.columns, f"Missing column {col}"
    assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
    assert df[col].max() <= 10, f"{col} max={df[col].max()} > 10"

    print(f"PASS: test_10_point_scale (range: {df[col].min()}-{df[col].max()})")


def test_11_point_scale():
    """11-point scale (the old cap!) should produce values 1-11."""
    scales = [{"name": "Trust", "num_items": 3, "scale_points": 11, "reverse_items": [], "_validated": True}]
    engine = _make_engine(scales, sample_size=100)
    df, metadata = engine.generate()

    for i in range(1, 4):
        col = f"Trust_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1, f"{col} min={df[col].min()} < 1"
        assert df[col].max() <= 11, f"{col} max={df[col].max()} > 11"

    print("PASS: test_11_point_scale")


def test_multiple_scales_different_ranges():
    """Multiple scales with different ranges should each respect their own range."""
    scales = [
        {"name": "Likert7", "num_items": 3, "scale_points": 7, "reverse_items": [], "_validated": True},
        {"name": "VAS100", "num_items": 1, "scale_points": 100, "reverse_items": [], "_validated": True},
        {"name": "Binary", "num_items": 2, "scale_points": 2, "reverse_items": [], "_validated": True},
        {"name": "Likert5", "num_items": 4, "scale_points": 5, "reverse_items": [], "_validated": True},
    ]
    engine = _make_engine(scales, sample_size=200)
    df, metadata = engine.generate()

    # Likert7: 1-7
    for i in range(1, 4):
        col = f"Likert7_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1 and df[col].max() <= 7, f"{col} out of 1-7 range"

    # VAS100: 1-100, should actually USE the range
    col = "VAS100_1"
    assert col in df.columns, f"Missing column {col}"
    assert df[col].min() >= 1 and df[col].max() <= 100, f"{col} out of 1-100 range"
    value_range = df[col].max() - df[col].min()
    assert value_range >= 30, f"CRITICAL: VAS100 range only {value_range}, expected wider"

    # Binary: 1-2
    for i in range(1, 3):
        col = f"Binary_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1 and df[col].max() <= 2, f"{col} out of 1-2 range"

    # Likert5: 1-5
    for i in range(1, 5):
        col = f"Likert5_{i}"
        assert col in df.columns, f"Missing column {col}"
        assert df[col].min() >= 1 and df[col].max() <= 5, f"{col} out of 1-5 range"

    print("PASS: test_multiple_scales_different_ranges")


def test_validated_flag_preserved():
    """_validated flag should flow from app.py normalization to engine."""
    scales = [
        {"name": "TestScale", "num_items": 3, "scale_points": 100, "reverse_items": [], "_validated": True},
    ]
    engine = _make_engine(scales)

    # Check that the engine preserved the validated scale
    assert len(engine.scales) == 1, f"Expected 1 scale, got {len(engine.scales)}"
    assert engine.scales[0]["scale_points"] == 100, (
        f"CRITICAL: scale_points={engine.scales[0]['scale_points']}, expected 100. "
        f"The _validated flag was not respected!"
    )
    assert engine.scales[0]["num_items"] == 3, (
        f"num_items={engine.scales[0]['num_items']}, expected 3"
    )

    print("PASS: test_validated_flag_preserved")


def test_scale_verification_report():
    """Verification report should accurately reflect generated data."""
    scales = [
        {"name": "Scale50", "num_items": 2, "scale_points": 50, "reverse_items": [], "_validated": True},
    ]
    engine = _make_engine(scales, sample_size=100)
    df, metadata = engine.generate()

    verification = metadata.get("scale_verification", [])
    assert len(verification) >= 1, "No verification report"

    report = verification[0]
    assert report["name"] == "Scale50"
    assert report["specified_scale_points"] == 50
    assert report["specified_num_items"] == 2
    assert len(report["columns_found"]) == 2
    assert report["columns_missing"] == []
    assert report["all_values_in_bounds"] is True
    assert report["range_utilization_pct"] > 0

    print(f"PASS: test_scale_verification_report (utilization: {report['range_utilization_pct']}%)")


def test_no_default_override():
    """Explicitly set scale_points should never be overridden to 7."""
    for pts in [2, 3, 5, 9, 10, 11, 50, 100, 101]:
        scales = [{"name": "TestDV", "num_items": 1, "scale_points": pts, "reverse_items": [], "_validated": True}]
        engine = _make_engine(scales, sample_size=50)
        df, metadata = engine.generate()

        col = "TestDV_1"
        assert col in df.columns, f"Missing column for scale_points={pts}"
        assert df[col].max() <= pts, (
            f"CRITICAL: scale_points={pts} but max value={df[col].max()}. "
            f"Values exceeded specified scale range!"
        )
        assert df[col].min() >= 1, f"Min value={df[col].min()} < 1 for scale_points={pts}"

        # For scales > 11 points, ensure we're NOT capped at old limit of 11
        if pts > 15 and len(df) >= 30:
            assert df[col].max() > 11, (
                f"CRITICAL: scale_points={pts} but max={df[col].max()} <= 11. "
                f"Old 11-point cap may still be in effect!"
            )

    print("PASS: test_no_default_override (tested scale_points: 2,3,5,9,10,11,50,100,101)")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_7_point_likert,
        test_5_point_likert,
        test_10_point_scale,
        test_11_point_scale,
        test_2_point_binary,
        test_100_point_scale,
        test_101_point_slider,
        test_multiple_scales_different_ranges,
        test_validated_flag_preserved,
        test_scale_verification_report,
        test_no_default_override,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"FAIL: {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print("\nAll tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
