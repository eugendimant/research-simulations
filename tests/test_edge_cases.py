#!/usr/bin/env python3
"""Edge case stress tests for all 4 generation methods."""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation_app"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "simulation_app"))
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec

os.environ.pop("LLM_API_KEY", None)

def run_test(name, **kwargs):
    print(f"\n  [{name}]", end=" ")
    try:
        engine = EnhancedSimulationEngine(**kwargs)
        df, meta = engine.generate()
        if df is None or len(df) == 0:
            print(f"FAIL — empty output")
            return False
        print(f"PASS — {df.shape[0]} rows, {df.shape[1]} cols")
        return True
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

results = []

# ============ EDGE CASE 1: Single condition ============
print("\n=== Edge Case 1: Single condition ===")
r = run_test("template_single_cond",
    study_title="Simple Study", study_description="Test",
    sample_size=20, conditions=["Control"], factors=[], scales=[
        {"name": "DV", "variable_name": "DV", "num_items": 3, "scale_points": 5,
         "scale_min": 1, "scale_max": 5, "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={}, attention_rate=0.05,
    random_responder_rate=0.03, effect_sizes=[], open_ended_questions=[],
    study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("Single condition, no OE", r))

# ============ EDGE CASE 2: No scales at all, only OE ============
print("\n=== Edge Case 2: No scales, only OE ===")
r = run_test("template_no_scales",
    study_title="Qual Study", study_description="Qualitative only",
    sample_size=15, conditions=["Treatment", "Control"], factors=[], scales=[],
    additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0, effect_sizes=[], open_ended_questions=[
        {"name": "Thoughts", "variable_name": "Thoughts",
         "question_text": "What are your thoughts on climate change?",
         "question_purpose": "DV Response", "type": "text"}
    ], study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("No scales, only OE", r))

# ============ EDGE CASE 3: N=1 ============
print("\n=== Edge Case 3: N=1 ===")
r = run_test("template_n1",
    study_title="Tiny Study", study_description="1 participant",
    sample_size=1, conditions=["A", "B"], factors=[], scales=[
        {"name": "Rating", "variable_name": "Rating", "num_items": 1,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0, effect_sizes=[], open_ended_questions=[
        {"name": "Why", "variable_name": "Why",
         "question_text": "Why did you give that rating?",
         "question_purpose": "DV Response", "type": "text"}
    ], study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("N=1", r))

# ============ EDGE CASE 4: Many conditions (6) ============
print("\n=== Edge Case 4: Many conditions (6) ===")
r = run_test("template_many_conds",
    study_title="Complex Design", study_description="6 conditions",
    sample_size=60, conditions=["A", "B", "C", "D", "E", "F"], factors=[],
    scales=[
        {"name": "Likert", "variable_name": "Likert", "num_items": 10,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [3, 7], "type": "likert"}
    ], additional_vars=[], demographics={"age_mean": 25, "age_sd": 5, "gender_quota": 50},
    attention_rate=0.1, random_responder_rate=0.05, effect_sizes=[],
    open_ended_questions=[
        {"name": "Feedback", "variable_name": "Feedback",
         "question_text": "Describe your experience.",
         "question_purpose": "DV Response", "type": "text"},
        {"name": "Gender_OE", "variable_name": "Gender_OE",
         "question_text": "What is your gender?",
         "question_purpose": "Demographic", "type": "text"},
    ], study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("6 conditions, 10-item scale", r))

# ============ EDGE CASE 5: Slider scales (0-100) ============
print("\n=== Edge Case 5: Slider scales 0-100 ===")
r = run_test("template_slider",
    study_title="Slider Study", study_description="Visual analog scales",
    sample_size=30, conditions=["Low", "High"], factors=[], scales=[
        {"name": "VAS_Pain", "variable_name": "VAS_Pain", "num_items": 1,
         "scale_points": 101, "scale_min": 0, "scale_max": 100,
         "reverse_items": [], "type": "slider"},
        {"name": "VAS_Mood", "variable_name": "VAS_Mood", "num_items": 1,
         "scale_points": 101, "scale_min": 0, "scale_max": 100,
         "reverse_items": [], "type": "slider"},
    ], additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0, effect_sizes=[], open_ended_questions=[],
    study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("Slider scales 0-100", r))

# ============ EDGE CASE 6: Empty conditions list → engine defaults ============
print("\n=== Edge Case 6: Empty conditions ===")
r = run_test("template_empty_conds",
    study_title="No Conditions", study_description="Should auto-create",
    sample_size=10, conditions=[], factors=[], scales=[
        {"name": "Scale1", "variable_name": "Scale1", "num_items": 2,
         "scale_points": 5, "scale_min": 1, "scale_max": 5,
         "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0, effect_sizes=[], open_ended_questions=[],
    study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("Empty conditions", r))

# ============ EDGE CASE 7: Effect sizes as dicts (not EffectSizeSpec) ============
print("\n=== Edge Case 7: Effect sizes as raw dicts ===")
r = run_test("template_dict_effects",
    study_title="Dict Effects", study_description="Raw dict effect sizes",
    sample_size=20, conditions=["High", "Low"], factors=[], scales=[
        {"name": "DV1", "variable_name": "DV1", "num_items": 3,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0,
    effect_sizes=[{"variable": "DV1", "factor": "condition",
                   "level_high": "High", "level_low": "Low",
                   "cohens_d": 0.8, "direction": "positive"}],
    open_ended_questions=[], study_context={}, seed=42, mode="pilot",
    allow_template_fallback=True)
results.append(("Dict effect sizes", r))

# ============ EDGE CASE 8: Experimental with OE (no LLM) ============
print("\n=== Edge Case 8: Experimental + OE ===")
r = run_test("experimental_oe",
    study_title="Trust Game", study_description="Economic game study",
    sample_size=20, conditions=["Partner", "Stranger"], factors=[], scales=[
        {"name": "Trust", "variable_name": "Trust", "num_items": 5,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={"age_mean": 30, "age_sd": 10, "gender_quota": 50},
    attention_rate=0.05, random_responder_rate=0.02, effect_sizes=[],
    open_ended_questions=[
        {"name": "Decision", "variable_name": "Decision",
         "question_text": "Explain your decision in the trust game.",
         "question_purpose": "DV Response", "type": "text"}
    ], study_context={}, seed=42, mode="pilot",
    allow_template_fallback=True, use_socsim_experimental=True)
results.append(("Experimental + OE", r))

# ============ EDGE CASE 9: N=200 (stress test) ============
print("\n=== Edge Case 9: N=200 stress test ===")
r = run_test("template_n200",
    study_title="Large Study", study_description="200 participants",
    sample_size=200, conditions=["Control", "Treatment_A", "Treatment_B"], factors=[],
    scales=[
        {"name": "Attitudes", "variable_name": "Attitudes", "num_items": 8,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [2, 5], "type": "likert"},
        {"name": "Satisfaction", "variable_name": "Satisfaction", "num_items": 4,
         "scale_points": 5, "scale_min": 1, "scale_max": 5,
         "reverse_items": [], "type": "likert"},
    ], additional_vars=[], demographics={"age_mean": 35, "age_sd": 12, "gender_quota": 50},
    attention_rate=0.08, random_responder_rate=0.04,
    effect_sizes=[EffectSizeSpec(variable="Attitudes", factor="condition",
        level_high="Treatment_A", level_low="Control", cohens_d=0.6, direction="positive")],
    open_ended_questions=[
        {"name": "Comments", "variable_name": "Comments",
         "question_text": "Any additional comments about the study?",
         "question_purpose": "DV Response", "type": "text"},
    ], study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("N=200 stress test", r))

# ============ EDGE CASE 10: Only demographic OE questions ============
print("\n=== Edge Case 10: Only demographic OE questions ===")
r = run_test("template_demo_only",
    study_title="Demo Study", study_description="Demographics only OE",
    sample_size=15, conditions=["Group_A"], factors=[], scales=[
        {"name": "Rating", "variable_name": "Rating", "num_items": 1,
         "scale_points": 7, "scale_min": 1, "scale_max": 7,
         "reverse_items": [], "type": "likert"}
    ], additional_vars=[], demographics={}, attention_rate=0.0,
    random_responder_rate=0.0, effect_sizes=[], open_ended_questions=[
        {"name": "Age", "variable_name": "Age", "question_text": "What is your age?",
         "question_purpose": "Demographic", "type": "text"},
        {"name": "Education", "variable_name": "Education",
         "question_text": "What is your highest level of education?",
         "question_purpose": "Demographic", "type": "text"},
    ], study_context={}, seed=42, mode="pilot", allow_template_fallback=True)
results.append(("Only demographic OE", r))

print(f"\n{'='*70}")
print("EDGE CASE SUMMARY")
print(f"{'='*70}")
total = len(results)
passed = sum(1 for _, r in results if r)
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n{passed}/{total} passed, {total - passed} failed")
sys.exit(0 if total == passed else 1)
