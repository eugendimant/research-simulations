import sys, os, traceback
# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedScale, ParsedCondition
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec

parser = SurveyDescriptionParser()
results = {"passed": 0, "failed": 0, "errors": []}

def test(name, condition, details=""):
    if condition:
        results["passed"] += 1
        print(f"  [PASS] {name}")
    else:
        results["failed"] += 1
        results["errors"].append(name)
        print(f"  [FAIL] {name} -- {details}")

print("=" * 70)
print("FINAL COMPREHENSIVE E2E TESTS")
print("=" * 70)

# ===== TEST SUITE 1: Health Psychology - 2-condition =====
print("\n--- Test Suite 1: Health Psychology (2-condition between-subjects) ---")
try:
    conds, _ = parser.parse_conditions("Mindfulness training, Waitlist control")
    test("1a: Parse 2 conditions", len(conds) == 2, f"got {len(conds)}")

    scales = parser.parse_scales("Perceived Stress Scale, 10 items, 1-5\nAnxiety (GAD-7), 7 items, 0-3")
    test("1b: Parse 2 scales", len(scales) >= 2, f"got {len(scales)}")

    oe = parser.parse_open_ended("Describe your experience with the intervention")

    design = ParsedDesign(conditions=conds, scales=scales, open_ended=oe,
        design_type="between", sample_size=80, research_domain="health psychology",
        study_title="Mindfulness and Stress")
    inferred = parser.build_inferred_design(design)
    test("1c: Build inferred design", bool(inferred.get("conditions")))

    engine = EnhancedSimulationEngine(
        study_title="Mindfulness and Stress", study_description="Testing mindfulness effects on stress",
        sample_size=80, conditions=inferred["conditions"], factors=inferred.get("factors", []),
        scales=inferred["scales"], additional_vars=[], demographics={"age_mean": 30, "age_sd": 10, "gender_quota": 50},
        open_ended_questions=inferred.get("open_ended_questions", []),
        study_context=inferred.get("study_context", {}))
    df, meta = engine.generate()
    test("1d: Generate 80 rows", len(df) == 80, f"got {len(df)}")
    test("1e: 2 unique conditions", len(df["CONDITION"].unique()) == 2, f"got {df['CONDITION'].unique()}")
    test("1f: Has metadata", bool(meta.get("study_title")))
    test("1g: Has column_descriptions", "column_descriptions" in meta, f"keys: {list(meta.keys())[:10]}")
    # Check for _mean columns (new feature)
    mean_cols = [c for c in df.columns if "_mean" in c.lower()]
    test("1h: Has composite mean columns", len(mean_cols) >= 1, f"found: {mean_cols}")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 1 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 2: Marketing - 2x2 Factorial =====
print("\n--- Test Suite 2: Marketing (2x2 factorial) ---")
try:
    conds2, _ = parser.parse_conditions("2x2: Brand (Premium, Economy) x Ad Type (Emotional, Rational)")
    test("2a: Parse 4 factorial conditions", len(conds2) == 4, f"got {len(conds2)}: {[c.name for c in conds2]}")

    scales2 = parser.parse_scales("Purchase intention, 3 items, 1-7\nBrand attitude, 4 items, 1-7\nWTP slider (0-100)")
    test("2b: Parse 3 scales", len(scales2) >= 3, f"got {len(scales2)}")

    design2 = ParsedDesign(conditions=conds2, scales=scales2, design_type="between",
        sample_size=200, research_domain="consumer behavior", study_title="Brand x Ad Type")
    inferred2 = parser.build_inferred_design(design2)

    factors = inferred2.get("factors", [])
    test("2c: Detect 2 factors", len(factors) >= 2, f"got {len(factors)}")

    engine2 = EnhancedSimulationEngine(
        study_title="Brand x Ad Type", study_description="Factorial marketing experiment",
        sample_size=200, conditions=inferred2["conditions"], factors=factors,
        scales=inferred2["scales"], additional_vars=[], demographics={},
        study_context=inferred2.get("study_context", {}))
    df2, meta2 = engine2.generate()
    test("2d: Generate 200 rows", len(df2) == 200)
    test("2e: 4 conditions in data", len(df2["CONDITION"].unique()) == 4, f"got {df2['CONDITION'].unique()}")
    test("2f: Balanced allocation", all(abs(v - 50) < 15 for v in df2["CONDITION"].value_counts()),
         f"counts: {dict(df2['CONDITION'].value_counts())}")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 2 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 3: AI Tech - 3-condition with effect sizes =====
print("\n--- Test Suite 3: AI Technology (3-condition with effect sizes) ---")
try:
    conds3, _ = parser.parse_conditions("AI-generated, Human-written, No label (control)")
    test("3a: Parse 3 conditions", len(conds3) == 3, f"got {len(conds3)}: {[c.name for c in conds3]}")

    scales3 = parser.parse_scales("Trust scale, 5 items, 1-7\nCredibility, 4 items, 1-7")

    design3 = ParsedDesign(conditions=conds3, scales=scales3, design_type="between",
        sample_size=150, research_domain="ai_technology", study_title="AI Labels on Trust")
    inferred3 = parser.build_inferred_design(design3)

    engine3 = EnhancedSimulationEngine(
        study_title="AI Labels on Trust", study_description="Testing AI labeling effects",
        sample_size=150, conditions=inferred3["conditions"], factors=inferred3.get("factors", []),
        scales=inferred3["scales"], additional_vars=[], demographics={},
        effect_sizes=[
            EffectSizeSpec(variable="Trust", factor="condition", level_high="Human-written", level_low="AI-generated", cohens_d=0.5),
            EffectSizeSpec(variable="Credibility", factor="condition", level_high="Human-written", level_low="AI-generated", cohens_d=0.3),
        ],
        study_context=inferred3.get("study_context", {}))
    df3, meta3 = engine3.generate()
    test("3b: Generate 150 rows", len(df3) == 150)
    test("3c: 3 conditions", len(df3["CONDITION"].unique()) == 3)

    # Check effect sizes are working
    trust_cols = [c for c in df3.columns if "Trust" in c and c != "Trust_mean"]
    test("3d: Has Trust columns", len(trust_cols) >= 5, f"found: {trust_cols}")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 3 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 4: Large factorial 2x2x2 =====
print("\n--- Test Suite 4: Complex Factorial (2x2x2) ---")
try:
    conds4, _ = parser.parse_conditions("2x2x2: Frame (Gain, Loss) x Source (Expert, Novice) x Time (Now, Later)")
    test("4a: Parse 8 conditions", len(conds4) == 8, f"got {len(conds4)}: {[c.name for c in conds4]}")

    scales4 = parser.parse_scales("Persuasion, 4 items, 1-7\nBehavioral intention, 3 items, 1-7")
    design4 = ParsedDesign(conditions=conds4, scales=scales4, design_type="between",
        sample_size=320, research_domain="communication", study_title="3-way Factorial")
    inferred4 = parser.build_inferred_design(design4)

    engine4 = EnhancedSimulationEngine(
        study_title="3-way Factorial", study_description="Complex factorial experiment",
        sample_size=320, conditions=inferred4["conditions"], factors=inferred4.get("factors", []),
        scales=inferred4["scales"], additional_vars=[], demographics={},
        study_context=inferred4.get("study_context", {}))
    df4, meta4 = engine4.generate()
    test("4b: Generate 320 rows", len(df4) == 320)
    test("4c: 8 conditions", len(df4["CONDITION"].unique()) == 8, f"got {df4['CONDITION'].unique()}")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 4 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 5: Numeric scales (WTP/slider) =====
print("\n--- Test Suite 5: Numeric Scales (WTP, Slider) ---")
try:
    conds5, _ = parser.parse_conditions("Luxury brand, Budget brand")
    scales5 = parser.parse_scales("Willingness to pay (0-200 dollars)\nSatisfaction slider (0-100)")

    design5 = ParsedDesign(conditions=conds5, scales=scales5, design_type="between",
        sample_size=60, research_domain="consumer behavior")
    inferred5 = parser.build_inferred_design(design5)

    # Check scale_points handling for numeric scales
    for s in inferred5["scales"]:
        name = s.get("name", "")
        sp = s.get("scale_points")
        test(f"5a: {name} scale_points", sp is None or isinstance(sp, (int, float)), f"type={type(sp)}, val={sp}")

    engine5 = EnhancedSimulationEngine(
        study_title="WTP Test", study_description="Testing numeric scales",
        sample_size=60, conditions=inferred5["conditions"], factors=[],
        scales=inferred5["scales"], additional_vars=[], demographics={},
        study_context=inferred5.get("study_context", {}))
    df5, meta5 = engine5.generate()
    test("5b: Generate 60 rows", len(df5) == 60)
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 5 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 6: Domain detection accuracy =====
print("\n--- Test Suite 6: Domain Detection ---")
try:
    domains = [
        ("Effect of AI labels on trust", "AI labeling study", "technology"),
        ("Brand positioning strategy", "Marketing experiment", "consumer"),
        ("Vaccination attitudes", "Health behavior study", "health"),
        ("Workplace diversity training", "Org behavior study", "organizational"),
        ("Political messaging effects", "Political communication", "politic"),
        ("Moral decision making", "Trolley problem study", "moral"),
    ]
    for title, desc, expected_substr in domains:
        detected = parser.detect_research_domain(title, desc)
        match = expected_substr.lower() in detected.lower()
        test(f"6: '{title}' -> '{detected}'", match, f"expected contains '{expected_substr}'")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 6 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 7: Report generation =====
print("\n--- Test Suite 7: Report Generation ---")
try:
    from utils.instructor_report import InstructorReportGenerator
    report_gen = InstructorReportGenerator()
    md = report_gen.generate_markdown_report(df3, meta3)
    test("7a: Markdown report not empty", len(md) > 1000, f"got {len(md)} chars")
    test("7b: Has data dictionary", "Data Dictionary" in md or "data dictionary" in md.lower() or "Column" in md)
    test("7c: Has analysis recommendations", "Statistical" in md or "Analysis" in md or "analysis" in md)

    from utils.instructor_report import ComprehensiveInstructorReport
    comp = ComprehensiveInstructorReport()
    comp_md = comp.generate_comprehensive_report(df3, meta3)
    test("7d: Comprehensive report not empty", len(comp_md) > 500, f"got {len(comp_md)} chars")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 7 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 8: Validation edge cases =====
print("\n--- Test Suite 8: Validation Edge Cases ---")
try:
    # Empty conditions
    val = parser.validate_full_design(ParsedDesign(conditions=[], scales=[], sample_size=5))
    test("8a: Catches empty design", len(val["errors"]) >= 2)

    # Duplicate conditions
    dup_conds = [ParsedCondition(name="A"), ParsedCondition(name="A"), ParsedCondition(name="B")]
    val2 = parser.validate_full_design(ParsedDesign(conditions=dup_conds, scales=[ParsedScale(name="X")], sample_size=50))
    test("8b: Catches duplicates", any("duplicate" in e.lower() or "Duplicate" in e for e in val2["errors"]))

    # Valid design
    good_conds, _ = parser.parse_conditions("A, B, C")
    good_scales = parser.parse_scales("DV, 5 items, 1-7")
    val3 = parser.validate_full_design(ParsedDesign(conditions=good_conds, scales=good_scales, sample_size=100))
    test("8c: Valid design passes", len(val3["errors"]) == 0, f"errors: {val3['errors']}")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 8 crashed: {e}")
    traceback.print_exc()

# ===== TEST SUITE 9: Auto-fill suggestions =====
print("\n--- Test Suite 9: Auto-fill Suggestions ---")
try:
    # Test suggest_scales_for_domain
    for domain in ["consumer behavior", "health psychology", "technology & hci", "social psychology", "educational psychology"]:
        sugs = parser.suggest_scales_for_domain(domain)
        test(f"9: {domain} has suggestions", len(sugs) >= 2, f"got {len(sugs)}")

    # Test default fallback
    sugs_default = parser.suggest_scales_for_domain("unknown_domain_xyz")
    test("9f: Unknown domain has default", len(sugs_default) >= 1)
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Suite 9 crashed: {e}")
    traceback.print_exc()

# ===== FINAL RESULTS =====
print("\n" + "=" * 70)
print(f"FINAL RESULTS: {results['passed']} passed, {results['failed']} failed")
if results["errors"]:
    print("FAILURES:")
    for e in results["errors"]:
        print(f"  - {e}")
print("=" * 70)
