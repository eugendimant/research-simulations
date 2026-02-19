#!/usr/bin/env python3
"""
Real runtime test of ALL 4 generation methods.
Exercises the actual code paths that users trigger.
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulation_app"))
os.chdir(os.path.join(os.path.dirname(__file__), "simulation_app"))

from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec

STUDY_TITLE = "Effect of Political Identity on Trust Game Behavior"
STUDY_DESC = "Participants play a trust game with ingroup vs outgroup members"
CONDITIONS = ["Ingroup", "Outgroup", "Control"]
SCALES = [
    {"name": "Trust_Scale", "variable_name": "Trust_Scale", "num_items": 5,
     "scale_points": 7, "scale_min": 1, "scale_max": 7,
     "reverse_items": [2, 4], "type": "likert"},
    {"name": "Single_Item_DV", "variable_name": "Single_Item_DV", "num_items": 1,
     "scale_points": 7, "scale_min": 1, "scale_max": 7,
     "reverse_items": [], "type": "single_item"},
    {"name": "Slider_DV", "variable_name": "Slider_DV", "num_items": 1,
     "scale_points": 101, "scale_min": 0, "scale_max": 100,
     "reverse_items": [], "type": "slider"},
]
OPEN_ENDED = [
    {"name": "OE_Feelings", "variable_name": "OE_Feelings",
     "question_text": "How did you feel about interacting with your partner?",
     "question_context": "Trust game partner interaction",
     "question_purpose": "DV Response", "context_type": "general",
     "type": "text", "force_response": False, "min_chars": None, "block_name": "Main"},
    {"name": "Age", "variable_name": "Age",
     "question_text": "What is your age?", "question_context": "",
     "question_purpose": "Demographic", "context_type": "general",
     "type": "text", "force_response": False, "min_chars": None, "block_name": "Demographics"},
    {"name": "OE_Explain", "variable_name": "OE_Explain",
     "question_text": "Please explain your decision in the trust game.",
     "question_context": "Trust game decision explanation",
     "question_purpose": "DV Response", "context_type": "general",
     "type": "text", "force_response": False, "min_chars": None, "block_name": "Main"},
]

EFFECT_SIZES = [EffectSizeSpec(
    variable="Trust_Scale", factor="condition",
    level_high="Ingroup", level_low="Outgroup",
    cohens_d=0.5, direction="positive",
)]

progress_log = []
def progress_callback(phase, current, total):
    progress_log.append((phase, current, total))


def run_method(method_name, allow_template_fallback, use_socsim, sample_size):
    """Run one generation method and return (success, error_msg, df, metadata)."""
    progress_log.clear()
    os.environ.pop("LLM_API_KEY", None)

    try:
        engine = EnhancedSimulationEngine(
            study_title=STUDY_TITLE, study_description=STUDY_DESC,
            sample_size=sample_size, conditions=CONDITIONS, factors=[],
            scales=SCALES, additional_vars=[],
            demographics={"age_mean": 35, "age_sd": 12, "gender_quota": 50},
            attention_rate=0.05, random_responder_rate=0.03,
            effect_sizes=EFFECT_SIZES, open_ended_questions=OPEN_ENDED,
            study_context={}, condition_allocation=None, seed=42, mode="pilot",
            allow_template_fallback=allow_template_fallback,
            progress_callback=progress_callback,
            use_socsim_experimental=use_socsim,
        )
    except Exception as e:
        return False, f"Engine construction failed: {e}\n{traceback.format_exc()}", None, None

    try:
        df, metadata = engine.generate()
    except Exception as e:
        return False, f"engine.generate() failed: {e}\n{traceback.format_exc()}", None, None

    errors = []
    if df is None or len(df) == 0:
        return False, "df is None or empty!", None, None

    print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")

    if len(df) != sample_size:
        errors.append(f"Expected {sample_size} rows, got {len(df)}")

    # Check required columns (using actual names)
    for col in ["CONDITION", "Age", "Gender"]:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    # Check condition values
    if "CONDITION" in df.columns:
        actual = set(df["CONDITION"].unique())
        if not actual.issubset(set(CONDITIONS)):
            errors.append(f"Unexpected conditions: {actual - set(CONDITIONS)}")

    # Check scale columns — engine appends _N suffix
    for s in SCALES:
        var = s["variable_name"]
        n_items = s["num_items"]
        for item_i in range(1, n_items + 1):
            col = f"{var}_{item_i}"
            if col not in df.columns:
                errors.append(f"Missing scale column: {col}")
            else:
                vals = df[col].dropna()
                if len(vals) > 0:
                    s_min, s_max = s["scale_min"], s["scale_max"]
                    if vals.min() < s_min - 0.5:
                        errors.append(f"{col}: min={vals.min()} < scale_min={s_min}")
                    if vals.max() > s_max + 0.5:
                        errors.append(f"{col}: max={vals.max()} > scale_max={s_max}")

    # Check OE columns — engine uses _clean_column_name which may prefix OE_
    for oe in OPEN_ENDED:
        var = oe["variable_name"]
        purpose = oe.get("question_purpose", "DV Response")
        # Engine may prefix with OE_ for some columns
        possible = [var, f"OE_{var}"]
        found = None
        for p in possible:
            if p in df.columns:
                found = p
                break
        if found is None:
            errors.append(f"Missing OE column (tried {possible})")
        else:
            non_empty = df[found].apply(lambda x: bool(str(x).strip()) if x is not None else False)
            pct = non_empty.sum() / len(df) * 100
            print(f"    OE '{found}': {non_empty.sum()}/{len(df)} non-empty ({pct:.0f}%)")
            if non_empty.sum() == 0:
                errors.append(f"OE column {found} has ALL empty responses!")

            # Demographic check
            if purpose == "Demographic" and "age" in var.lower():
                sample = df[found].dropna().head(5).tolist()
                for v in sample:
                    try:
                        val = float(str(v))
                        if val < 10 or val > 120:
                            errors.append(f"Age value out of range: {v}")
                    except (ValueError, TypeError):
                        if len(str(v)) > 10:
                            errors.append(f"Age has text: '{str(v)[:50]}'")

    if metadata is None:
        errors.append("metadata is None!")
    elif not isinstance(metadata, dict):
        errors.append(f"metadata type: {type(metadata)}")
    else:
        # Check key metadata fields
        for key in ["conditions", "run_id"]:
            if key not in metadata:
                errors.append(f"Missing metadata key: {key}")

    # Check progress was reported
    phases = set(p[0] for p in progress_log)
    for phase in ["personas", "scales", "generating", "complete"]:
        if phase not in phases:
            errors.append(f"Missing progress phase: {phase}")

    if errors:
        return False, "\n".join(errors), df, metadata
    return True, "OK", df, metadata


def main():
    methods = [
        ("Template Engine",             True,  False),
        ("Adaptive Behavioral Engine",  True,  True),
        ("Built-in AI (free_llm)",      False, False),
        ("Own API Key",                 False, False),
    ]

    all_results = []
    for method_name, allow_fallback, use_socsim in methods:
        for n in [10, 50]:
            print(f"\n{'='*70}")
            print(f"Testing: {method_name} | N={n}")
            print(f"{'='*70}")
            success, msg, df, meta = run_method(method_name, allow_fallback, use_socsim, n)
            if success:
                print(f"  PASS")
            else:
                print(f"  FAIL:")
                for line in msg.split("\n")[:5]:
                    print(f"    {line}")
            all_results.append((method_name, n, success, msg))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total = len(all_results)
    passed = sum(1 for r in all_results if r[2])
    for name, n, ok, msg in all_results:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name} N={n}")
        if not ok:
            for line in msg.split("\n")[:3]:
                print(f"         {line}")
    print(f"\n{passed}/{total} passed, {total-passed} failed")
    return 0 if total - passed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
