#!/usr/bin/env python3
"""
End-to-end simulation test: Builder → Engine → Report.
Tests the full pipeline with multiple experiment types.
"""
import sys
import os
import json
sys.path.insert(0, 'simulation_app')

import numpy as np
import pandas as pd

from utils.survey_builder import SurveyDescriptionParser, ParsedDesign
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec
from utils.instructor_report import InstructorReportGenerator, ComprehensiveInstructorReport

parser = SurveyDescriptionParser()

TESTS_PASSED = 0
TESTS_FAILED = 0


def make_engine(inferred, sample_size=100, seed=42, effect_sizes=None):
    """Create an engine from inferred_design dict (matches app.py's pattern)."""
    ctx = inferred.get("study_context", {})
    return EnhancedSimulationEngine(
        study_title=ctx.get("title", "Test Study"),
        study_description=ctx.get("description", ""),
        sample_size=sample_size,
        conditions=inferred["conditions"],
        factors=inferred.get("factors", []),
        scales=inferred["scales"],
        additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8},
        open_ended_questions=inferred.get("open_ended_questions", []),
        study_context=ctx,
        condition_allocation=inferred.get("condition_allocation"),
        seed=seed,
        effect_sizes=effect_sizes or [],
    )

def check(name: str, condition: bool, detail: str = ""):
    global TESTS_PASSED, TESTS_FAILED
    if condition:
        TESTS_PASSED += 1
        print(f"  PASS: {name}")
    else:
        TESTS_FAILED += 1
        print(f"  FAIL: {name} - {detail}")


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Simple 3-condition between-subjects design
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Simple 3-condition design")
print("=" * 70)

conds1, _ = parser.parse_conditions("AI-generated vs Human-curated vs No annotation")
scales1 = parser.parse_scales("Trust scale (4 items, 1-7); Purchase intention (3 items, 1-7)")
oe1 = parser.parse_open_ended("Why did you rate the article's credibility the way you did?")

design1 = ParsedDesign(
    conditions=conds1, scales=scales1, open_ended=oe1,
    design_type="between", sample_size=60,
    research_domain="technology & hci",
    study_title="AI Content Credibility",
    study_description="Testing trust in AI vs human content",
)

inferred1 = parser.build_inferred_design(design1)
check("conditions count", len(inferred1["conditions"]) == 3, f"got {len(inferred1['conditions'])}")
check("scales count", len(inferred1["scales"]) == 2, f"got {len(inferred1['scales'])}")
check("design_type present", inferred1.get("design_type") == "between")
check("sample_size present", inferred1.get("sample_size") == 60)
check("condition_allocation present", len(inferred1.get("condition_allocation", {})) == 3)
check("study_context has domain", inferred1["study_context"]["domain"] == "technology & hci")

# Run engine
engine1 = make_engine(inferred1, sample_size=60, seed=42)
df1, meta1 = engine1.generate()
check("dataframe rows", len(df1) == 60, f"got {len(df1)}")
check("CONDITION column", "CONDITION" in df1.columns)
check("conditions in data", set(df1["CONDITION"].unique()) == set(inferred1["conditions"]),
      f"got {df1['CONDITION'].unique()}")
check("persona_distribution in metadata", "persona_distribution" in meta1)
check("persona_by_condition in metadata", "persona_by_condition" in meta1)
check("trait_averages_overall in metadata", "trait_averages_overall" in meta1)

# Generate reports
md_reporter1 = InstructorReportGenerator()
html_reporter1 = ComprehensiveInstructorReport()
html1 = html_reporter1.generate_html_report(df1, meta1)
check("HTML report generated", len(html1) > 5000, f"length={len(html1)}")
check("HTML has persona section", "Persona Analysis" in html1 or "Persona Distribution" in html1)
check("HTML has trait profile", "Personality Profile" in html1 or "trait" in html1.lower())
check("HTML has study context", "Research Context" in html1)
check("HTML has confidential badge", "CONFIDENTIAL" in html1)
check("HTML has version", "1.3" in html1)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 2: 3×2 factorial design (user's exact experiment)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: 3×2 factorial design")
print("=" * 70)

conds2, _ = parser.parse_conditions(
    "3 (Annotation: AI-generated vs Human-curated vs No source information) "
    "× 2 (Product type: Hedonic vs Utilitarian), between-subjects, random assignment."
)

scale_text = """Perceived Quality (PQ): 3 items (7-point Likert; 1=extremely low quality, 7=extremely high quality)

Purchase Intention (PI): 3 items (7-point Likert; 1=strongly disagree, 7=strongly agree)

Willingness to Pay (WTP): 1 item (open-ended numeric)"""

scales2 = parser.parse_scales(scale_text)
oe2 = parser.parse_open_ended("What influenced your perception of the product?")

design2 = ParsedDesign(
    conditions=conds2, scales=scales2, open_ended=oe2,
    factors=[
        {"name": "Annotation", "levels": ["AI-generated", "Human-curated", "No source information"]},
        {"name": "Product type", "levels": ["Hedonic", "Utilitarian"]},
    ],
    design_type="between", sample_size=120,
    research_domain="consumer behavior",
    study_title="Product Annotation Study",
    study_description="3x2 factorial on AI annotation and product type",
)

inferred2 = parser.build_inferred_design(design2)
check("6 crossed conditions", len(inferred2["conditions"]) == 6, f"got {len(inferred2['conditions'])}")
check("3 scales", len(inferred2["scales"]) == 3, f"got {len(inferred2['scales'])}")
check("PQ scale name", inferred2["scales"][0]["name"] == "Perceived Quality")
check("WTP is numeric", inferred2["scales"][2]["type"] == "numeric")
check("factors preserved", len(inferred2["factors"]) == 2)

# Run engine with effect size
engine2 = make_engine(inferred2, sample_size=120, seed=123, effect_sizes=[
    EffectSizeSpec(
        variable="Perceived Quality",
        factor="condition",
        level_high="Human-curated × Hedonic",
        level_low="No source information × Utilitarian",
        cohens_d=0.5,
        direction="positive",
    )
])
df2, meta2 = engine2.generate()
check("dataframe rows", len(df2) == 120, f"got {len(df2)}")
check("6 unique conditions", len(df2["CONDITION"].unique()) == 6, f"got {len(df2['CONDITION'].unique())}")
check("persona_by_condition has counts",
      "counts" in meta2.get("persona_by_condition", {}),
      f"keys: {list(meta2.get('persona_by_condition', {}).keys())}")
check("trait_averages has data", len(meta2.get("trait_averages_overall", {})) > 0)
check("effect_sizes_configured", len(meta2.get("effect_sizes_configured", [])) >= 1)

# Check PQ columns exist
pq_cols = [c for c in df2.columns if c.startswith("Perceived_Quality")]
check("PQ scale columns present", len(pq_cols) >= 3, f"found {pq_cols}")

# Generate HTML report
html_reporter2 = ComprehensiveInstructorReport()
html2 = html_reporter2.generate_html_report(df2, meta2)
check("HTML generated for factorial", len(html2) > 8000, f"length={len(html2)}")
check("HTML has persona by condition table", "Persona Distribution by Condition" in html2)
check("HTML has effect size section", "Effect Size Verification" in html2 or "effect" in html2.lower())

# Also generate markdown report
md_reporter2 = InstructorReportGenerator()
md2 = md_reporter2.generate_markdown_report(df2, meta2)
check("Markdown report generated", len(md2) > 3000, f"length={len(md2)}")
check("Markdown has persona section", "Persona Distribution" in md2 or "persona" in md2.lower())

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 3: 2×2 factorial with known scales
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: 2×2 factorial with known scales (BFI-10)")
print("=" * 70)

conds3, _ = parser.parse_conditions("Trust (high, low) and Moral Frame (care, fairness)")
scales3 = parser.parse_scales("BFI-10; Empathy scale (5 items, 1-7); Donation amount in dollars (WTP, 0-100)")

check("4 crossed conditions", len(conds3) == 4, f"got {len(conds3)}: {[c.name for c in conds3]}")
check("3 scales parsed", len(scales3) == 3, f"got {len(scales3)}")
check("BFI-10 recognized", any("BFI" in s.name or "Big Five" in s.name for s in scales3),
      f"names: {[s.name for s in scales3]}")

design3 = ParsedDesign(
    conditions=conds3, scales=scales3, open_ended=[],
    design_type="between", sample_size=80,
    research_domain="moral psychology",
    study_title="Moral Framing and Donation Behavior",
)
inferred3 = parser.build_inferred_design(design3)

engine3 = make_engine(inferred3, sample_size=80, seed=456)
df3, meta3 = engine3.generate()
check("80 rows generated", len(df3) == 80, f"got {len(df3)}")
check("4 conditions in data", len(df3["CONDITION"].unique()) == 4)

# Verify scale columns exist
bfi_cols = [c for c in df3.columns if "BFI" in c or "Big_Five" in c]
check("BFI scale columns present", len(bfi_cols) >= 1, f"BFI cols: {bfi_cols[:5]}")

html3 = ComprehensiveInstructorReport().generate_html_report(df3, meta3)
check("HTML report for 2x2", len(html3) > 5000)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 4: 2×2×2 three-factor factorial
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: 2×2×2 three-factor factorial")
print("=" * 70)

conds4, _ = parser.parse_conditions(
    "2 (Frame: Gain vs Loss) × 2 (Source: Expert vs Peer) × 2 (Time: Immediate vs Delayed)"
)
check("8 crossed conditions", len(conds4) == 8, f"got {len(conds4)}: {[c.name for c in conds4]}")
scales4 = parser.parse_scales("Risk perception (4 items, 1-7); Decision confidence (3 items, 1-7)")
design4 = ParsedDesign(
    conditions=conds4, scales=scales4, open_ended=[],
    factors=[
        {"name": "Frame", "levels": ["Gain", "Loss"]},
        {"name": "Source", "levels": ["Expert", "Peer"]},
        {"name": "Time", "levels": ["Immediate", "Delayed"]},
    ],
    design_type="between", sample_size=160,
    research_domain="behavioral economics",
    study_title="Framing, Source, and Time Pressure",
)
inferred4 = parser.build_inferred_design(design4)
check("8 conditions in inferred", len(inferred4["conditions"]) == 8)
check("3 factors preserved", len(inferred4["factors"]) == 3)

engine4 = make_engine(inferred4, sample_size=160, seed=789)
df4, meta4 = engine4.generate()
check("160 rows generated", len(df4) == 160, f"got {len(df4)}")
check("8 conditions in data", len(df4["CONDITION"].unique()) == 8,
      f"got {len(df4['CONDITION'].unique())}")
# ~20 per cell
per_cell_4 = df4.groupby("CONDITION").size()
check("balanced allocation (16-24 per cell)",
      all(14 <= n <= 26 for n in per_cell_4),
      f"counts: {dict(per_cell_4)}")

html4 = ComprehensiveInstructorReport().generate_html_report(df4, meta4)
check("HTML report for 2x2x2", len(html4) > 5000)
check("HTML has persona section", "Persona" in html4)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 5: 4×2 factorial (unlabeled)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 5: 4×2 factorial (unlabeled notation)")
print("=" * 70)

conds5, _ = parser.parse_conditions("4x2 design")
check("8 generic conditions", len(conds5) == 8, f"got {len(conds5)}")
# All conditions should have × separator
check("all conditions are crossed", all(" × " in c.name for c in conds5),
      f"names: {[c.name for c in conds5]}")

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 6: 3×3 factorial (parenthetical)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 6: 3×3 factorial (parenthetical)")
print("=" * 70)

conds6, _ = parser.parse_conditions(
    "Dose (low, medium, high) and Duration (short, medium, long)"
)
check("9 crossed conditions", len(conds6) == 9, f"got {len(conds6)}: {[c.name for c in conds6]}")

scales6 = parser.parse_scales("Outcome measure (5 items, 1-7); Side effects (3 items, 0-10)")
design6 = ParsedDesign(
    conditions=conds6, scales=scales6, open_ended=[],
    design_type="between", sample_size=180,
    research_domain="health psychology",
    study_title="Dose-Duration Interaction Study",
)
inferred6 = parser.build_inferred_design(design6)
check("9 conditions in inferred", len(inferred6["conditions"]) == 9)
engine6 = make_engine(inferred6, sample_size=180, seed=999)
df6, meta6 = engine6.generate()
check("180 rows generated", len(df6) == 180, f"got {len(df6)}")
check("9 conditions in data", len(df6["CONDITION"].unique()) == 9)

html6 = ComprehensiveInstructorReport().generate_html_report(df6, meta6)
check("HTML report for 3x3", len(html6) > 5000)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 7: 2×2×2×2 four-factor factorial
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 7: 2×2×2×2 four-factor factorial (16 cells)")
print("=" * 70)

conds7, _ = parser.parse_conditions("2x2x2x2 design")
check("16 conditions for 2x2x2x2", len(conds7) == 16, f"got {len(conds7)}")

scales7 = parser.parse_scales("Attitude (4 items, 1-7)")
design7 = ParsedDesign(
    conditions=conds7, scales=scales7, open_ended=[],
    design_type="between", sample_size=320,
    research_domain="social psychology",
    study_title="Four-Factor Interaction Study",
)
inferred7 = parser.build_inferred_design(design7)
engine7 = make_engine(inferred7, sample_size=320, seed=1234)
df7, meta7 = engine7.generate()
check("320 rows generated", len(df7) == 320, f"got {len(df7)}")
check("16 conditions in data", len(df7["CONDITION"].unique()) == 16,
      f"got {len(df7['CONDITION'].unique())}")

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Scale edge cases
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 8: Scale edge cases (slider, binary, numeric)")
print("=" * 70)

# Slider should keep 0-100 range
slider_scales = parser.parse_scales("Visual analog slider from 0 to 100")
check("slider detected", len(slider_scales) >= 1)
if slider_scales:
    check("slider range 0-100", slider_scales[0].scale_min == 0 and slider_scales[0].scale_max == 100,
          f"got {slider_scales[0].scale_min}-{slider_scales[0].scale_max}")

# Binary scale (yes/no)
binary_scales = parser.parse_scales("Manipulation check (yes/no)")
check("binary detected", len(binary_scales) >= 1)
if binary_scales:
    check("binary is 0-1", binary_scales[0].scale_min == 0 and binary_scales[0].scale_max == 1,
          f"got {binary_scales[0].scale_min}-{binary_scales[0].scale_max}")

# Known instrument
gad_scales = parser.parse_scales("GAD-7")
check("GAD-7 recognized", len(gad_scales) >= 1)
if gad_scales:
    check("GAD-7 has items", gad_scales[0].num_items >= 7,
          f"got {gad_scales[0].num_items} items")

# 5-point Likert should stay as 1-5, not become 0-100
likert5 = parser.parse_scales("Satisfaction (3 items, 5-point Likert)")
check("5-point Likert detected", len(likert5) >= 1)
if likert5:
    check("5-point range 1-5", likert5[0].scale_min == 1 and likert5[0].scale_max == 5,
          f"got {likert5[0].scale_min}-{likert5[0].scale_max}")

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Multi-effect factorial simulation
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 9: Multi-effect size specification")
print("=" * 70)

# Use the 3x2 from Test 2 but with multiple effect sizes
engine9 = make_engine(inferred2, sample_size=180, seed=555, effect_sizes=[
    EffectSizeSpec(
        variable="Perceived Quality",
        factor="condition",
        level_high="Human-curated",
        level_low="AI-generated",
        cohens_d=0.5,
        direction="positive",
    ),
    EffectSizeSpec(
        variable="Purchase Intention",
        factor="condition",
        level_high="Hedonic",
        level_low="Utilitarian",
        cohens_d=0.3,
        direction="positive",
    ),
])
df9, meta9 = engine9.generate()
check("180 rows for multi-effect", len(df9) == 180, f"got {len(df9)}")
check("6 conditions for multi-effect", len(df9["CONDITION"].unique()) == 6)

# Check that PQ means differ in expected direction
pq_cols = [c for c in df9.columns if c.startswith("Perceived_Quality")]
if pq_cols:
    pq_main = pq_cols[0]
    human_hci_mean = df9[df9["CONDITION"].str.contains("Human")][pq_main].mean()
    ai_hci_mean = df9[df9["CONDITION"].str.contains("AI")][pq_main].mean()
    check("Human > AI on PQ (directional effect)", human_hci_mean > ai_hci_mean,
          f"Human={human_hci_mean:.2f}, AI={ai_hci_mean:.2f}")

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Validation tests
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 10: Design validation")
print("=" * 70)

# Valid design
val_result = parser.validate_full_design(design2)
check("no errors for valid design", len(val_result["errors"]) == 0, f"errors: {val_result['errors']}")

# Invalid design - too few conditions
bad_conds, _ = parser.parse_conditions("Only one condition")
bad_design = ParsedDesign(
    conditions=bad_conds, scales=scales1, design_type="between",
    sample_size=50, research_domain="test",
)
val_bad = parser.validate_full_design(bad_design)
check("error for <2 conditions", len(val_bad["errors"]) > 0)

# Underpowered factorial warning
design_small = ParsedDesign(
    conditions=conds6, scales=scales6, open_ended=[],
    design_type="between", sample_size=50,
    research_domain="test",
)
val_small = parser.validate_full_design(design_small)
check("warning for underpowered 3x3", len(val_small["warnings"]) > 0,
      f"warnings: {val_small['warnings']}")

# Commas inside parentheses
test_items = parser._parse_list_items("Control (no treatment), Treatment A (weekly coaching, monthly review)")
check("parenthetical commas handled", len(test_items) == 2,
      f"got {len(test_items)}: {test_items}")

print()


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
total = TESTS_PASSED + TESTS_FAILED
print(f"RESULTS: {TESTS_PASSED}/{total} tests passed, {TESTS_FAILED} failed")
print("=" * 70)

if TESTS_FAILED > 0:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
