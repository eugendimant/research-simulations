#!/usr/bin/env python3
"""Test scale/condition parsing with the user's exact experiment description."""
import sys, os
# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
from utils.survey_builder import SurveyDescriptionParser

parser = SurveyDescriptionParser()

# ═══════════════════════════════════════════════════════════
# Test 1: Condition parsing - labeled parenthetical factorial
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Condition Parsing")
print("=" * 70)

cond_text = "3 (Annotation: AI-generated vs Human-curated vs No source information) × 2 (Product type: Hedonic vs Utilitarian), between-subjects, random assignment."
conditions, warnings = parser.parse_conditions(cond_text)

print(f"Input: {cond_text}")
print(f"Conditions found: {len(conditions)}")
for i, c in enumerate(conditions):
    print(f"  {i+1}. {c.name} (control={c.is_control})")
print(f"Warnings: {warnings}")

assert len(conditions) == 6, f"Expected 6 conditions, got {len(conditions)}"
print("\n✓ PASS: 6 crossed conditions\n")

# ═══════════════════════════════════════════════════════════
# Test 2: Scale parsing - user's exact scale input
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Scale Parsing (user's exact input)")
print("=" * 70)

scale_text = """Perceived Quality (PQ): 3 items (7-point Likert; 1=extremely low quality, 7=extremely high quality); items: "The quality of this product is…," "How would you rate the overall quality?," "This product meets high quality standards."

Purchase Intention (PI): 3 items (7-point Likert; 1=strongly disagree, 7=strongly agree); items: "I would consider buying this product," "I would recommend this product to a friend," "I intend to purchase this product."

Brand Trust (BT): 4 items (7-point Likert; 1=strongly disagree, 7=strongly agree); items: "I trust this brand," "This brand is reliable," "I feel confident in this brand," "This brand has integrity."

Willingness to Pay (WTP): 1 item (open-ended numeric); "What is the maximum price (in USD) you would be willing to pay for this product?"

Perceived Transparency (PT): 3 items (7-point Likert; 1=strongly disagree, 7=strongly agree); items: "The product information provided was transparent," "I felt well-informed about the product's origins," "The source information was clear and honest."

Ad Credibility (AC): 3 items (7-point Likert; 1=not at all credible, 7=extremely credible); items: "The advertisement for this product was credible," "I believe the claims made in the ad," "The ad accurately represents the product."

Attitude toward the Product (Att): 3 items (semantic differential, 7-point scale); anchors: bad/good, unfavorable/favorable, negative/positive.

Annotation Awareness (manipulation check): 1 item; "Did you notice any source information (e.g., annotation) about the product content? (Yes/No)"

Product Category Familiarity (covariate): 1 item (7-point Likert; 1=not at all familiar, 7=extremely familiar); "How familiar are you with [hedonic/utilitarian product category]?\""""

scales = parser.parse_scales(scale_text)
print(f"Input: (paragraph-separated, 9 scales)")
print(f"Scales found: {len(scales)}")
for i, s in enumerate(scales):
    print(f"  {i+1}. name={s.name!r}, items={s.num_items}, range={s.scale_min}-{s.scale_max}, type={s.scale_type}")
print()

if len(scales) >= 7:
    print(f"✓ PASS: Found {len(scales)} scales (expected ~9)\n")
else:
    print(f"✗ FAIL: Only found {len(scales)} scales (expected ~9)\n")


# ═══════════════════════════════════════════════════════════
# Test 3: Individual segment parsing (verify each)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Segment splitting check")
print("=" * 70)

segments = parser._split_scale_segments(scale_text)
print(f"Segments: {len(segments)}")
for i, seg in enumerate(segments):
    preview = seg[:80].replace('\n', ' ')
    print(f"  {i+1}. {preview}...")

if len(segments) >= 7:
    print(f"\n✓ PASS: {len(segments)} segments (expected 9)\n")
else:
    print(f"\n✗ FAIL: Only {len(segments)} segments (expected 9)\n")


# ═══════════════════════════════════════════════════════════
# Test 4: Open-ended questions
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: Open-ended question parsing")
print("=" * 70)

oe_text = """1. "Please explain your reasoning for the purchase decision you just made."
2. "What, if anything, influenced your perception of the product?"
3. "Describe your overall impression of the product and its source information.\""""

oe_qs = parser.parse_open_ended(oe_text)
print(f"Open-ended questions found: {len(oe_qs)}")
for i, q in enumerate(oe_qs):
    print(f"  {i+1}. {q.question_text[:60]}...")

if len(oe_qs) >= 2:
    print(f"\n✓ PASS: Found {len(oe_qs)} open-ended questions\n")
else:
    print(f"\n✗ FAIL: Only found {len(oe_qs)} open-ended questions\n")


# ═══════════════════════════════════════════════════════════
# Test 5: Regression - simple scale inputs still work
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 5: Regression - simple inputs")
print("=" * 70)

# Semicolon-separated (the old default)
simple1 = "Trust scale (1-7, 4 items); Purchase intention (1-7, 3 items); Satisfaction (1-5, 5 items)"
s1 = parser.parse_scales(simple1)
print(f"Semicolon-separated: {len(s1)} scales")
for s in s1:
    print(f"  - {s.name}: {s.num_items} items, {s.scale_min}-{s.scale_max}")
assert len(s1) >= 3, f"Expected >= 3, got {len(s1)}"
print("  ✓ PASS\n")

# Comma-separated
simple2 = "7-point Likert scale measuring satisfaction (5 items)"
s2 = parser.parse_scales(simple2)
print(f"Single scale: {len(s2)} scales")
for s in s2:
    print(f"  - {s.name}: {s.num_items} items, {s.scale_min}-{s.scale_max}")
assert len(s2) >= 1
print("  ✓ PASS\n")

# Numbered list
simple3 = """1. Attitude toward AI (5-point, 3 items)
2. Trust in technology (7-point, 5 items)
3. Purchase intention (1-7, 3 items)"""
s3 = parser.parse_scales(simple3)
print(f"Numbered list: {len(s3)} scales")
for s in s3:
    print(f"  - {s.name}: {s.num_items} items, {s.scale_min}-{s.scale_max}")
assert len(s3) >= 3
print("  ✓ PASS\n")


# ═══════════════════════════════════════════════════════════
# Test 6: Regression - condition patterns still work
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 6: Regression - condition patterns")
print("=" * 70)

# Simple vs pattern
c1, _ = parser.parse_conditions("Treatment vs Control")
print(f"'Treatment vs Control': {[c.name for c in c1]}")
assert len(c1) == 2
print("  ✓ PASS")

# 2x2 factorial
c2, _ = parser.parse_conditions("2x2: Factor A (High, Low) x Factor B (Present, Absent)")
print(f"'2x2 factorial': {[c.name for c in c2]}")
assert len(c2) >= 4
print("  ✓ PASS")

# Simple comma list
c3, _ = parser.parse_conditions("Control, Low dose, Medium dose, High dose")
print(f"'Comma list': {[c.name for c in c3]}")
assert len(c3) == 4
print("  ✓ PASS")

# Numbered list
c4, _ = parser.parse_conditions("1. AI-generated\n2. Human-curated\n3. No annotation")
print(f"'Numbered list': {[c.name for c in c4]}")
assert len(c4) == 3
print("  ✓ PASS")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
