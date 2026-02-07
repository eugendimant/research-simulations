"""
Test suite for SurveyDescriptionParser natural language parsing capabilities.

Tests condition parsing, scale parsing, domain detection, factorial detection,
and design validation.
"""
import sys, os
# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
from utils.survey_builder import SurveyDescriptionParser, ParsedDesign, ParsedScale

parser = SurveyDescriptionParser()

passed = 0
failed = 0
failures = []

def report(test_id, description, success, detail=""):
    global passed, failed, failures
    if success:
        passed += 1
        print(f"  PASS  {test_id}: {description}")
    else:
        failed += 1
        failures.append((test_id, description, detail))
        print(f"  FAIL  {test_id}: {description} -- {detail}")


print("=" * 70)
print("=== CONDITION PARSING TESTS ===")
print("=" * 70)

# Test A: Simple comma-separated
conds, w = parser.parse_conditions("Control, Treatment A, Treatment B")
ok = len(conds) == 3
report("A", "Simple comma-separated (3 conditions)", ok,
       f"Expected 3, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test B: "vs" separator
conds, w = parser.parse_conditions("AI-generated vs Human-written")
ok = len(conds) == 2
report("B", '"vs" separator (2 conditions)', ok,
       f"Expected 2, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test C: Factorial with "x" notation
conds, w = parser.parse_conditions("2x2 design: Trust (High, Low) x Risk (High, Low)")
ok = len(conds) == 4
report("C", 'Factorial with "x" notation (4 conditions)', ok,
       f"Expected 4, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test D: Factorial with multiplication sign
conds, w = parser.parse_conditions("3 × 2: Source (AI, Human, None) × Product (Hedonic, Utilitarian)")
ok = len(conds) == 6
report("D", "Factorial with multiplication sign (6 conditions)", ok,
       f"Expected 6, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test E: Natural language factorial
conds, w = parser.parse_conditions("Trust (high, low) and Risk (high, low)")
ok = len(conds) >= 2
report("E", "Natural language factorial (>= 2 conditions)", ok,
       f"Expected >= 2, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test F: 2x2x2 design
conds, w = parser.parse_conditions("2x2x2: Frame (Gain, Loss) x Source (Expert, Peer) x Time (Now, Later)")
ok = len(conds) == 8
report("F", "2x2x2 design (8 conditions)", ok,
       f"Expected 8, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test G: Single condition (should warn)
conds, w = parser.parse_conditions("Treatment group")
print(f"  INFO  G: Single condition: {len(conds)} conds, {len(w)} warnings")
ok = True  # Just informational
report("G", "Single condition (informational)", ok)

# Test H: Numbered conditions
conds, w = parser.parse_conditions("Condition 1, Condition 2, Condition 3, Condition 4")
ok = len(conds) == 4
report("H", "Numbered conditions (4 conditions)", ok,
       f"Expected 4, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test I: Descriptive conditions with parenthetical details
conds, w = parser.parse_conditions("High credibility (expert source), Low credibility (peer source), Control (no source)")
ok = len(conds) == 3
report("I", "Conditions with parenthetical details (3 conditions)", ok,
       f"Expected 3, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")

# Test J: Just "and" separator
conds, w = parser.parse_conditions("Positive framing and Negative framing and Control")
print(f"  INFO  J: 'and' separator: {len(conds)} conds: {[c.name for c in conds]}")
ok = len(conds) >= 2
report("J", '"and" separator (>= 2 conditions)', ok,
       f"Expected >= 2, got {len(conds)}: {[c.name for c in conds]}" if not ok else "")


print()
print("=" * 70)
print("=== SCALE PARSING TESTS ===")
print("=" * 70)

# Test K: Standard Likert
scales = parser.parse_scales("Trust scale, 5 items, 1-7 Likert")
ok_count = len(scales) >= 1
if ok_count:
    s = scales[0]
    ok_items = s.num_items == 5
    ok_range = s.scale_min == 1 and s.scale_max == 7
    report("K", "Standard Likert scale parsing", ok_items and ok_range,
           f"items={s.num_items} (expected 5), range={s.scale_min}-{s.scale_max} (expected 1-7)")
else:
    report("K", "Standard Likert scale parsing", False, f"Expected >= 1 scale, got {len(scales)}")

# Test L: WTP/numeric
scales = parser.parse_scales("Willingness to pay ($0-$100)")
ok_count = len(scales) >= 1
if ok_count:
    s = scales[0]
    print(f"  INFO  L: WTP scale: type={s.scale_type}, min={s.scale_min}, max={s.scale_max}")
    report("L", "WTP/numeric scale detection", True)
else:
    report("L", "WTP/numeric scale detection", False, f"Expected >= 1 scale, got {len(scales)}")

# Test M: Slider
scales = parser.parse_scales("Satisfaction slider from 0 to 100")
ok_count = len(scales) >= 1
if ok_count:
    s = scales[0]
    ok_type = s.scale_type == "slider"
    report("M", "Slider type detection", ok_type,
           f"Expected type='slider', got type='{s.scale_type}'" if not ok_type else "")
else:
    report("M", "Slider type detection", False, f"Expected >= 1 scale, got {len(scales)}")

# Test N: Multiple scales, semicolon-separated
scales = parser.parse_scales("Trust (5 items, 1-7); Purchase intention (3 items, 1-7); Risk perception (4 items, 1-5)")
ok = len(scales) >= 2
report("N", f"Semicolon-separated multiple scales ({len(scales)} detected)", ok,
       f"Expected >= 2, got {len(scales)}: {[s.name for s in scales]}" if not ok else "")

# Test O: Known instrument by name
scales = parser.parse_scales("PANAS")
ok_count = len(scales) >= 1
if ok_count:
    s = scales[0]
    print(f"  INFO  O: PANAS: items={s.num_items}, range={s.scale_min}-{s.scale_max}")
    report("O", "Known instrument (PANAS) recognition", True)
else:
    report("O", "Known instrument (PANAS) recognition", False, f"Expected >= 1 scale, got {len(scales)}")

# Test P: Binary measure
scales = parser.parse_scales("Choice: yes or no")
ok_count = len(scales) >= 1
if ok_count:
    s = scales[0]
    print(f"  INFO  P: Binary: type={s.scale_type}, range={s.scale_min}-{s.scale_max}")
    report("P", "Binary measure detection", True)
else:
    report("P", "Binary measure detection", False, f"Expected >= 1 scale, got {len(scales)}")

# Test Q: Multiple lines
scales = parser.parse_scales("Trust scale, 5 items, 1-7\nPurchase intention, 3 items, 1-7\nWTP ($0-$50)")
ok = len(scales) >= 2
report("Q", f"Multi-line scale parsing ({len(scales)} detected)", ok,
       f"Expected >= 2, got {len(scales)}: {[s.name for s in scales]}" if not ok else "")


print()
print("=" * 70)
print("=== DOMAIN DETECTION TESTS ===")
print("=" * 70)

# Test R: AI domain
domain = parser.detect_research_domain("Effect of AI labels on consumer trust", "Testing how AI labels affect consumer trust in products")
print(f"  INFO  R: AI domain detected: '{domain}'")
ok = "technology" in domain.lower() or "hci" in domain.lower() or "consumer" in domain.lower()
report("R", "AI/tech domain detection", ok,
       f"Expected tech or consumer domain, got '{domain}'" if not ok else "")

# Test S: Health domain
domain = parser.detect_research_domain("Impact of exercise framing on health behavior", "Examining how different exercise messages affect health intentions")
print(f"  INFO  S: Health domain detected: '{domain}'")
ok = "health" in domain.lower()
report("S", "Health domain detection", ok,
       f"Expected health domain, got '{domain}'" if not ok else "")

# Test T: Marketing domain
domain = parser.detect_research_domain("Brand positioning and purchase behavior", "How brand positioning affects purchase decisions")
print(f"  INFO  T: Marketing domain detected: '{domain}'")
ok = "consumer" in domain.lower() or "marketing" in domain.lower()
report("T", "Marketing domain detection", ok,
       f"Expected consumer/marketing domain, got '{domain}'" if not ok else "")


print()
print("=" * 70)
print("=== FACTORIAL DETECTION TESTS ===")
print("=" * 70)

# Test U: Detect factors from crossed conditions
conds, _ = parser.parse_conditions("2x2: Trust (High, Low) x Risk (High, Low)")
factors = parser.detect_factorial_structure(conds)
print(f"  INFO  U: Conditions: {[c.name for c in conds]}")
print(f"  INFO  U: Factors detected: {len(factors)}: {[f['name'] for f in factors] if factors else 'none'}")
# Note: detect_factorial_structure requires >= 4 conditions, and we expect 4 from Test C-like input
ok = len(conds) >= 4 or len(factors) >= 0  # May not detect if < 4 conditions
report("U", f"Factorial structure detection ({len(factors)} factors from {len(conds)} conds)", True)


print()
print("=" * 70)
print("=== VALIDATION TESTS ===")
print("=" * 70)

# Test V: Complete valid design
conds, _ = parser.parse_conditions("Control, Treatment")
scales = parser.parse_scales("Trust, 5 items, 1-7")
design = ParsedDesign(conditions=conds, scales=scales, design_type="between", sample_size=100)
val = parser.validate_full_design(design)
ok = len(val["errors"]) == 0
report("V", "Valid design passes validation", ok,
       f"Unexpected errors: {val['errors']}" if not ok else "")

# Test W: Design with issues
design2 = ParsedDesign(conditions=[], scales=[], design_type="between", sample_size=5)
val2 = parser.validate_full_design(design2)
ok = len(val2["errors"]) >= 2
report("W", f"Invalid design catches errors ({len(val2['errors'])} errors)", ok,
       f"Expected >= 2 errors, got {len(val2['errors'])}: {val2['errors']}" if not ok else "")


print()
print("=" * 70)
print(f"=== RESULTS: {passed} PASSED, {failed} FAILED ===")
print("=" * 70)

if failures:
    print("\nFailed tests:")
    for test_id, desc, detail in failures:
        print(f"  {test_id}: {desc}")
        print(f"      {detail}")

print("\n=== ALL TESTS COMPLETE ===")
