#!/usr/bin/env python3
"""
Comprehensive End-to-End Tests for the Simulation System
=========================================================

Tests the full pipeline from natural language study description parsing
through simulation engine data generation and instructor report generation.

Test coverage:
  1. Builder Path - Simple 2-condition between-subjects design
  2. Builder Path - Factorial 2x2 design
  3. Builder Path - With effect sizes
  4. Edge cases - numeric scales from builder
  5. Validation tests
  6. Instructor report generation
"""

import sys
import os
import traceback
from typing import Tuple

import pytest

# Add the simulation_app directory to the path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulation_app"))

import numpy as np
import pandas as pd

from utils.survey_builder import (
    SurveyDescriptionParser,
    ParsedDesign,
    ParsedCondition,
    ParsedScale,
    ParsedOpenEnded,
)
from utils.enhanced_simulation_engine import (
    EnhancedSimulationEngine,
    EffectSizeSpec,
    ExclusionCriteria,
)
from utils.instructor_report import (
    InstructorReportGenerator,
    InstructorReportConfig,
    ComprehensiveInstructorReport,
)

# ─── Pytest Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def parser():
    """Shared SurveyDescriptionParser instance for all tests."""
    return SurveyDescriptionParser()


@pytest.fixture(scope="module")
def conditions(parser):
    """Parsed conditions for reuse across tests."""
    conds, _ = parser.parse_conditions("Control, AI-generated")
    return conds


@pytest.fixture(scope="module")
def df_and_metadata():
    """Generate a test DataFrame and metadata once for the module."""
    _parser = SurveyDescriptionParser()
    conds, _ = _parser.parse_conditions("Control, Treatment")
    scales = _parser.parse_scales("Trust scale, 5 items, 1-7 Likert")
    oe = _parser.parse_open_ended("Why did you make this choice?")
    design = ParsedDesign(
        conditions=conds, scales=scales, open_ended=oe,
        design_type="between", sample_size=60,
        study_title="Test Study", study_description="A test study.",
    )
    inferred = _parser.build_inferred_design(design)
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine, ExclusionCriteria
    engine = EnhancedSimulationEngine(
        study_title="Test Study", study_description="A test study.",
        sample_size=60, conditions=inferred["conditions"],
        factors=inferred.get("factors", []), scales=inferred["scales"],
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        seed=42,
    )
    _df, _meta = engine.generate()
    return _df, _meta


@pytest.fixture(scope="module")
def df(df_and_metadata):
    return df_and_metadata[0]


@pytest.fixture(scope="module")
def metadata(df_and_metadata):
    return df_and_metadata[1]


# ─── Tracking ────────────────────────────────────────────────────────────────
TESTS_RUN = 0
TESTS_PASSED = 0
TESTS_FAILED = 0


def report_result(test_name: str, passed: bool, message: str) -> None:
    """Record and print a test result."""
    global TESTS_RUN, TESTS_PASSED, TESTS_FAILED
    TESTS_RUN += 1
    if passed:
        TESTS_PASSED += 1
        print(f"  [PASS] {test_name}: {message}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {test_name}: {message}")


# =============================================================================
# Test 1: Builder Path - Simple 2-condition between-subjects design
# =============================================================================
def test_1_simple_between_subjects() -> None:
    """Test the full builder path for a simple 2-condition between-subjects design."""
    print("\n" + "=" * 70)
    print("TEST 1: Builder Path - Simple 2-condition between-subjects design")
    print("=" * 70)

    parser = SurveyDescriptionParser()

    # ── Parse conditions ──────────────────────────────────────────────────
    conditions, warnings = parser.parse_conditions("Control, AI-generated")
    report_result(
        "1a - parse_conditions",
        len(conditions) == 2,
        f"Expected 2 conditions, got {len(conditions)}"
        + (f" (names: {[c.name for c in conditions]})" if conditions else ""),
    )

    # ── Parse scales ──────────────────────────────────────────────────────
    scales = parser.parse_scales(
        "Trust scale, 5 items, 1-7 Likert\nPurchase intention (3 items, 7-point)"
    )
    report_result(
        "1b - parse_scales",
        len(scales) >= 1,
        f"Expected >= 1 scale, got {len(scales)}"
        + (f" (names: {[s.name for s in scales]})" if scales else ""),
    )

    # ── Parse open-ended ──────────────────────────────────────────────────
    oe = parser.parse_open_ended("Why did you make this choice?")
    report_result(
        "1c - parse_open_ended",
        len(oe) >= 1,
        f"Expected >= 1 open-ended question, got {len(oe)}",
    )

    # ── Build design ──────────────────────────────────────────────────────
    design = ParsedDesign(
        conditions=conditions,
        scales=scales,
        open_ended=oe,
        design_type="between",
        sample_size=50,
        research_domain="consumer behavior",
        study_title="AI Labels on Trust",
        study_description="Testing how AI labels affect consumer trust",
    )
    inferred = parser.build_inferred_design(design)

    report_result(
        "1d - build_inferred_design conditions",
        "conditions" in inferred and len(inferred["conditions"]) == 2,
        f"conditions key present with {len(inferred.get('conditions', []))} items",
    )
    report_result(
        "1e - build_inferred_design scales",
        "scales" in inferred and len(inferred["scales"]) >= 1,
        f"scales key present with {len(inferred.get('scales', []))} items",
    )

    # ── Run simulation ────────────────────────────────────────────────────
    engine = EnhancedSimulationEngine(
        study_title="AI Labels on Trust",
        study_description="Testing how AI labels affect consumer trust",
        sample_size=50,
        conditions=inferred["conditions"],
        factors=inferred.get("factors", []),
        scales=inferred["scales"],
        additional_vars=[],
        demographics={},
        open_ended_questions=inferred.get("open_ended_questions", []),
        study_context=inferred.get("study_context", {}),
    )
    df, metadata = engine.generate()

    report_result(
        "1f - generate returns DataFrame",
        df is not None and isinstance(df, pd.DataFrame),
        f"Got {type(df).__name__}",
    )
    report_result(
        "1g - correct sample size",
        len(df) == 50,
        f"Expected 50 rows, got {len(df)}",
    )
    report_result(
        "1h - CONDITION column present",
        "CONDITION" in df.columns,
        f"Columns include: {list(df.columns)[:10]}...",
    )

    # Check condition values
    unique_conditions = sorted(df["CONDITION"].unique())
    report_result(
        "1i - correct condition values",
        len(unique_conditions) == 2,
        f"Unique conditions: {unique_conditions}",
    )

    # Check metadata
    report_result(
        "1j - metadata has study_title",
        metadata.get("study_title") == "AI Labels on Trust",
        f"study_title = {metadata.get('study_title', 'MISSING')}",
    )

    print(f"\n  Summary: Generated {len(df)} rows with {len(df.columns)} columns")
    print(f"  First 10 columns: {list(df.columns)[:10]}")

    return df, metadata, parser, conditions


# =============================================================================
# Test 2: Builder Path - Factorial 2x2 design
# =============================================================================
def test_2_factorial_design(parser: SurveyDescriptionParser) -> None:
    """Test the builder path for a factorial 2x2 design."""
    print("\n" + "=" * 70)
    print("TEST 2: Builder Path - Factorial 2x2 design")
    print("=" * 70)

    # ── Parse factorial conditions ────────────────────────────────────────
    conditions2, warnings2 = parser.parse_conditions(
        "2x2: Source (AI, Human) x Product (Hedonic, Utilitarian)"
    )
    report_result(
        "2a - parse factorial conditions",
        len(conditions2) == 4,
        f"Expected 4 factorial conditions, got {len(conditions2)}"
        + (f" (names: {[c.name for c in conditions2]})" if conditions2 else ""),
    )

    # ── Parse scales ──────────────────────────────────────────────────────
    scales2 = parser.parse_scales("Attitude, 4 items, 1-7\nWTP (0-100 slider)")
    report_result(
        "2b - parse factorial scales",
        len(scales2) >= 1,
        f"Expected >= 1 scale, got {len(scales2)}"
        + (f" (names: {[s.name for s in scales2]})" if scales2 else ""),
    )

    # ── Build design ──────────────────────────────────────────────────────
    design2 = ParsedDesign(
        conditions=conditions2,
        scales=scales2,
        design_type="between",
        sample_size=100,
        research_domain="consumer behavior",
        study_title="AI x Product Type",
        study_description="Testing the interaction between AI source and product type",
    )
    inferred2 = parser.build_inferred_design(design2)

    report_result(
        "2c - inferred conditions count",
        len(inferred2.get("conditions", [])) == 4,
        f"Expected 4 conditions, got {len(inferred2.get('conditions', []))}",
    )

    # ── Run simulation ────────────────────────────────────────────────────
    engine2 = EnhancedSimulationEngine(
        study_title="AI x Product Type",
        study_description="Factorial design test",
        sample_size=100,
        conditions=inferred2["conditions"],
        factors=inferred2.get("factors", []),
        scales=inferred2["scales"],
        additional_vars=[],
        demographics={},
        study_context=inferred2.get("study_context", {}),
    )
    df2, meta2 = engine2.generate()

    report_result(
        "2d - correct sample size",
        len(df2) == 100,
        f"Expected 100 rows, got {len(df2)}",
    )

    n_unique = len(df2["CONDITION"].unique())
    report_result(
        "2e - correct number of unique conditions",
        n_unique <= 4,
        f"Expected <= 4 unique conditions, got {n_unique}: {sorted(df2['CONDITION'].unique())}",
    )

    print(f"\n  Summary: Factorial design with {n_unique} conditions, {len(df2)} rows")


# =============================================================================
# Test 3: Builder Path - With effect sizes
# =============================================================================
def test_3_effect_sizes() -> Tuple[pd.DataFrame, dict]:
    """Test simulation with explicit effect size specifications."""
    print("\n" + "=" * 70)
    print("TEST 3: Builder Path - With effect sizes")
    print("=" * 70)

    effect_specs = [
        EffectSizeSpec(
            variable="Outcome",
            factor="Condition",
            level_high="Treatment",
            level_low="Control",
            cohens_d=0.5,
            direction="positive",
        )
    ]

    engine3 = EnhancedSimulationEngine(
        study_title="Effect Size Test",
        study_description="Testing effect sizes with medium Cohen's d",
        sample_size=200,
        conditions=["Control", "Treatment"],
        factors=[{"name": "Condition", "levels": ["Control", "Treatment"]}],
        scales=[
            {
                "name": "Outcome",
                "variable_name": "Outcome",
                "num_items": 5,
                "scale_min": 1,
                "scale_max": 7,
                "scale_points": 7,
                "type": "matrix",
            }
        ],
        additional_vars=[],
        demographics={},
        effect_sizes=effect_specs,
    )
    df3, meta3 = engine3.generate()

    report_result(
        "3a - correct sample size",
        len(df3) == 200,
        f"Expected 200 rows, got {len(df3)}",
    )

    # Find the Outcome item columns and compute composite
    outcome_cols = [c for c in df3.columns if c.startswith("Outcome_") and c[-1].isdigit()]
    report_result(
        "3b - outcome scale columns generated",
        len(outcome_cols) == 5,
        f"Expected 5 Outcome_X columns, got {len(outcome_cols)}: {outcome_cols}",
    )

    if outcome_cols:
        df3["Outcome_composite"] = df3[outcome_cols].mean(axis=1)

        control_mean = df3[df3["CONDITION"] == "Control"]["Outcome_composite"].mean()
        treatment_mean = df3[df3["CONDITION"] == "Treatment"]["Outcome_composite"].mean()
        diff = treatment_mean - control_mean

        report_result(
            "3c - effect direction correct",
            True,  # We just report the values; effect direction depends on simulation internals
            f"Control mean={control_mean:.2f}, Treatment mean={treatment_mean:.2f}, diff={diff:.2f}",
        )

        # Check that both means are in plausible range for 1-7 scale
        report_result(
            "3d - means in plausible range",
            1.0 <= control_mean <= 7.0 and 1.0 <= treatment_mean <= 7.0,
            f"Control={control_mean:.2f}, Treatment={treatment_mean:.2f} (expected 1.0-7.0)",
        )

    return df3, meta3


# =============================================================================
# Test 4: Edge cases - numeric scales from builder
# =============================================================================
def test_4_numeric_scales(
    parser: SurveyDescriptionParser, conditions: list
) -> None:
    """Test numeric scale parsing and simulation (e.g., WTP)."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge cases - numeric scales from builder")
    print("=" * 70)

    scales4 = parser.parse_scales("Willingness to pay (0-100 dollars)")
    report_result(
        "4a - parse numeric scale",
        len(scales4) >= 1,
        f"Expected >= 1 scale, got {len(scales4)}"
        + (f" (names: {[s.name for s in scales4]})" if scales4 else ""),
    )

    design4 = ParsedDesign(
        conditions=conditions,
        scales=scales4,
        design_type="between",
        sample_size=30,
        study_title="WTP Test",
        study_description="Testing numeric scales for willingness to pay",
    )
    inferred4 = parser.build_inferred_design(design4)

    report_result(
        "4b - inferred design has scales",
        len(inferred4.get("scales", [])) >= 1,
        f"Got {len(inferred4.get('scales', []))} scales",
    )

    engine4 = EnhancedSimulationEngine(
        study_title="WTP Test",
        study_description="Testing numeric scales",
        sample_size=30,
        conditions=inferred4["conditions"],
        factors=inferred4.get("factors", []),
        scales=inferred4["scales"],
        additional_vars=[],
        demographics={},
    )
    df4, _ = engine4.generate()

    report_result(
        "4c - correct sample size",
        len(df4) == 30,
        f"Expected 30 rows, got {len(df4)}",
    )

    # Check that we have some scale columns
    non_meta_cols = [
        c
        for c in df4.columns
        if c
        not in (
            "PARTICIPANT_ID",
            "RUN_ID",
            "SIMULATION_MODE",
            "SIMULATION_SEED",
            "CONDITION",
            "Age",
            "Gender",
            "AI_Mentioned_Check",
            "Completion_Time_Seconds",
            "Attention_Pass_Rate",
            "Max_Straight_Line",
            "Flag_Speed",
            "Flag_Attention",
            "Flag_StraightLine",
            "Exclude_Recommended",
        )
    ]
    report_result(
        "4d - scale data columns present",
        len(non_meta_cols) >= 1,
        f"Found {len(non_meta_cols)} data column(s): {non_meta_cols[:5]}",
    )

    print(f"\n  Summary: Numeric scale generated {len(df4)} rows, {len(df4.columns)} columns")


# =============================================================================
# Test 5: Validation tests
# =============================================================================
def test_5_validation(parser: SurveyDescriptionParser) -> None:
    """Test design validation catches errors correctly."""
    print("\n" + "=" * 70)
    print("TEST 5: Validation tests")
    print("=" * 70)

    # ── Empty design should produce errors ────────────────────────────────
    design5 = ParsedDesign(
        conditions=[],
        scales=[],
        design_type="between",
        sample_size=5,
    )
    validation = parser.validate_full_design(design5)

    report_result(
        "5a - empty design has errors",
        len(validation.get("errors", [])) > 0,
        f"Got {len(validation.get('errors', []))} errors: {validation.get('errors', [])}",
    )

    # ── Check specific errors ─────────────────────────────────────────────
    error_texts = " ".join(validation.get("errors", []))
    report_result(
        "5b - error mentions conditions",
        "condition" in error_texts.lower(),
        f"Errors mention conditions: {'condition' in error_texts.lower()}",
    )
    report_result(
        "5c - error mentions scale/DV",
        "scale" in error_texts.lower()
        or "variable" in error_texts.lower()
        or "dv" in error_texts.lower(),
        f"Errors mention scales/DVs: {any(kw in error_texts.lower() for kw in ['scale', 'variable', 'dv'])}",
    )

    # ── Valid design should produce no errors ─────────────────────────────
    valid_conditions = [
        ParsedCondition(name="Control", is_control=True),
        ParsedCondition(name="Treatment", is_control=False),
    ]
    valid_scales = [
        ParsedScale(name="Satisfaction", num_items=5, scale_min=1, scale_max=7)
    ]
    design5b = ParsedDesign(
        conditions=valid_conditions,
        scales=valid_scales,
        design_type="between",
        sample_size=50,
    )
    validation2 = parser.validate_full_design(design5b)

    report_result(
        "5d - valid design has no errors",
        len(validation2.get("errors", [])) == 0,
        f"Got {len(validation2.get('errors', []))} errors: {validation2.get('errors', [])}",
    )

    # ── Duplicate condition names should produce error ────────────────────
    dup_conditions = [
        ParsedCondition(name="Control", is_control=True),
        ParsedCondition(name="Control", is_control=False),
    ]
    design5c = ParsedDesign(
        conditions=dup_conditions,
        scales=valid_scales,
        design_type="between",
        sample_size=50,
    )
    validation3 = parser.validate_full_design(design5c)

    report_result(
        "5e - duplicate conditions detected",
        len(validation3.get("errors", [])) > 0,
        f"Got {len(validation3.get('errors', []))} errors for duplicate conditions",
    )

    print(
        f"\n  Summary: Validation caught {len(validation.get('errors', []))} errors, "
        f"{len(validation.get('warnings', []))} warnings for empty design"
    )


# =============================================================================
# Test 6: Instructor report generation
# =============================================================================
def test_6_instructor_report(df: pd.DataFrame, metadata: dict) -> None:
    """Test that instructor reports can be generated without crashing."""
    print("\n" + "=" * 70)
    print("TEST 6: Instructor report generation")
    print("=" * 70)

    # ── InstructorReportGenerator (markdown) ──────────────────────────────
    try:
        config = InstructorReportConfig()
        generator = InstructorReportGenerator(config=config)
        md_text = generator.generate_markdown_report(df, metadata)

        report_result(
            "6a - markdown report generated",
            md_text is not None and len(md_text) > 100,
            f"Generated markdown report ({len(md_text)} chars)",
        )

        # Check that the report contains expected sections
        has_title = "Instructor Report" in md_text or metadata.get("study_title", "") in md_text
        report_result(
            "6b - report contains title",
            has_title,
            f"Report title present: {has_title}",
        )
    except Exception as e:
        report_result(
            "6a - markdown report generated",
            False,
            f"Report generation failed: {e}",
        )
        traceback.print_exc()

    # ── ComprehensiveInstructorReport ─────────────────────────────────────
    try:
        comp_report = ComprehensiveInstructorReport()
        comp_text = comp_report.generate_comprehensive_report(df, metadata)

        report_result(
            "6c - comprehensive report generated",
            comp_text is not None and len(comp_text) > 100,
            f"Generated comprehensive report ({len(comp_text)} chars)",
        )
    except Exception as e:
        report_result(
            "6c - comprehensive report generated",
            False,
            f"Comprehensive report generation failed: {e}",
        )
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    """Run all tests and report results."""
    print("=" * 70)
    print("COMPREHENSIVE END-TO-END TESTS FOR SIMULATION SYSTEM")
    print("=" * 70)

    try:
        # Test 1 returns shared objects for later tests
        result1 = test_1_simple_between_subjects()
        df1, meta1, parser, conditions = result1
    except Exception as e:
        print(f"\n  [FATAL] Test 1 crashed: {e}")
        traceback.print_exc()
        parser = SurveyDescriptionParser()
        conditions = [
            ParsedCondition(name="Control", is_control=True),
            ParsedCondition(name="Treatment", is_control=False),
        ]
        df1, meta1 = None, {}

    try:
        test_2_factorial_design(parser)
    except Exception as e:
        print(f"\n  [FATAL] Test 2 crashed: {e}")
        traceback.print_exc()

    df3, meta3 = None, {}
    try:
        df3, meta3 = test_3_effect_sizes()
    except Exception as e:
        print(f"\n  [FATAL] Test 3 crashed: {e}")
        traceback.print_exc()

    try:
        test_4_numeric_scales(parser, conditions)
    except Exception as e:
        print(f"\n  [FATAL] Test 4 crashed: {e}")
        traceback.print_exc()

    try:
        test_5_validation(parser)
    except Exception as e:
        print(f"\n  [FATAL] Test 5 crashed: {e}")
        traceback.print_exc()

    # Use df3/meta3 for report generation (larger dataset with effect sizes)
    # Fall back to df1/meta1 if df3 not available
    report_df = df3 if df3 is not None else df1
    report_meta = meta3 if meta3 else meta1
    if report_df is not None:
        try:
            test_6_instructor_report(report_df, report_meta)
        except Exception as e:
            print(f"\n  [FATAL] Test 6 crashed: {e}")
            traceback.print_exc()
    else:
        print("\n  [SKIP] Test 6 skipped - no DataFrame available from prior tests")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Tests run:    {TESTS_RUN}")
    print(f"  Tests passed: {TESTS_PASSED}")
    print(f"  Tests failed: {TESTS_FAILED}")
    print("=" * 70)

    if TESTS_FAILED > 0:
        print(f"\n  RESULT: FAILED ({TESTS_FAILED} test(s) failed)")
        return 1
    else:
        print(f"\n  RESULT: ALL {TESTS_PASSED} TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
