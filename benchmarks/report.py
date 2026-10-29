"""Render a benchmark JSON as Markdown tables.   python -m benchmarks.report results/<study>.json > results/<study>.md"""
from __future__ import annotations

import json
import sys

ARM_ORDER = ["ENGINE_DEFAULT", "ENGINE_DECLARED", "ENGINE_NULL", "ENGINE_PRIOR", "BASELINE_NULL", "BASELINE_PRIOR"]
LABEL = {
    "ENGINE_DEFAULT": "Engine, nothing configured",
    "ENGINE_DECLARED": "Engine, direction declared",
    "ENGINE_NULL": "Engine, effects off",
    "ENGINE_PRIOR": "Engine, analyst d",
    "BASELINE_NULL": "Naive simulation, no effect",
    "BASELINE_PRIOR": "Naive simulation, analyst d",
}


def f(x: float, k: int = 2) -> str:
    return "n/a" if x != x else f"{x:.{k}f}"


def render(r: dict) -> str:
    dvs = list(r["human"]["dv"].keys())
    L = []
    L.append(f"# Retrospective benchmark: {r['study']}\n")
    L.append(f"{r['citation']}\n")
    L.append(f"Seeds per arm: {r['seeds']} (first seed {r['seed0']}). Analyst-stated d: {r['prior_d']}. Code commit: `{r['git_sha'][:10]}`. "
             f"Human data SHA-256: `{r['human_data_sha256'][:16]}...`. Python {r['python']}, numpy {r['numpy']}, pandas {r['pandas']}.\n")
    L.append("## Human data (n = %d, focal cell n = %d)\n" % (r["human"]["n"], r["human"]["n_focal"]))
    L.append("| DV | mean | SD | focal g [95% CI] | focal p |\n|---|---|---|---|---|")
    for k in dvs:
        h = r["human"]["dv"][k]
        L.append(f"| {k} | {f(h['mean'])} | {f(h['sd'])} | {f(h['focal_g'])} [{f(h['focal_lo'])}, {f(h['focal_hi'])}] | {f(h['focal_p'], 4)} |")
    L.append("\nFocal contrast: negative tone x Latino immigrant against the other three cells.\n")
    L.append("## Behavioral forecasting: mean over seeds (SD over seeds in parentheses)\n")
    for k in dvs:
        L.append(f"### {k}\n")
        L.append("| Arm | mean error (human SD) | SD ratio | Wasserstein | TV | focal g | sign agrees | CI covers human g | focal decision agrees |\n|---|---|---|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            a = r["arms"][arm]["dv"][k]
            g = r["synthetic_focal_g_over_seeds"][arm][k]
            L.append(f"| {LABEL[arm]} | {f(a['mean_err_sd']['mean'])} ({f(a['mean_err_sd']['sd'])}) | {f(a['sd_ratio']['mean'])} | {f(a['wasserstein']['mean'], 3)} | {f(a['tv']['mean'], 3)} | "
                     f"{f(g['mean'])} ({f(g['sd'])}) | {f(a['focal_sign_agree']['mean'], 2)} | {f(a['focal_ci_covers_human']['mean'], 2)} | {f(a['focal_decision_agree']['mean'], 2)} |")
        L.append("")
    L.append("## Correlation structure: mean absolute difference in off-diagonal Spearman correlations\n")
    L.append("| Arm | difference |\n|---|---|")
    for arm in ARM_ORDER:
        c = r["arms"][arm]["corr_mad"]
        L.append(f"| {LABEL[arm]} | {f(c['mean'], 3)} ({f(c['sd'], 3)}) |")
    L.append("\n## Design sensitivity: implied raw minimum detectable effect, synthetic divided by human\n")
    L.append("| Arm | " + " | ".join(dvs) + " |\n|---|" + "---|" * len(dvs))
    for arm in ARM_ORDER:
        L.append(f"| {LABEL[arm]} | " + " | ".join(f(r["arms"][arm]["dv"][k]["mde_ratio"]["mean"]) for k in dvs) + " |")
    L.append("\n## Null calibration: share of seeds with a significant focal contrast when no effect was configured\n")
    L.append("| DV | Engine, effects off | Naive simulation |\n|---|---|---|")
    for k in dvs:
        n = r["null_calibration"][k]
        L.append(f"| {k} | {f(n['engine_null_focal_reject_rate'])} | {f(n['baseline_null_focal_reject_rate'])} |")
    L.append("\nExpected value under a true null: 0.05.\n")
    L.append("## Pipeline rehearsal (one engine run, nothing configured)\n")
    pr = r["pipeline_rehearsal"]
    L.append("| DV | column present | finite | inside human range | integer coded | levels observed |\n|---|---|---|---|---|---|")
    for k in dvs:
        p = pr[k]
        L.append(f"| {k} | {p['column_present']} | {p['all_finite']} | {p['within_human_range']} | {p['integer_valued']} | {p['levels_observed']} |")
    L.append(f"\nProvenance columns present: {pr['_provenance_columns_present']}. Engine-detected domains: {r['engine_detected_domains']}.\n")
    L.append("## What the adapter had to do\n")
    for n in r["adapter_notes"]:
        L.append(f"- {n}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(render(json.load(open(sys.argv[1]))))
