"""Worker: run full pipeline on ONE qsf, print a single JSON result line to stdout.
All validation checks performed here. Designed to be run as an isolated subprocess
with an external timeout so a hang on one file does not block the corpus."""
import os, sys, json, traceback, warnings
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')

path = sys.argv[1]
R = {"file": os.path.basename(path), "stage": "start"}

def emit(extra=None):
    if extra:
        R.update(extra)
    sys.stdout.write("@@RESULT@@" + json.dumps(R, default=str) + "\n")
    sys.stdout.flush()

try:
    import numpy as np
    import pandas as pd
    import app
    from utils.qsf_preview import QSFPreviewParser
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine

    R["stage"] = "parse"
    res = QSFPreviewParser().parse(open(path, 'rb').read())
    R["survey_name"] = (res.survey_name or "")[:80]

    R["stage"] = "bridge"
    inp = app._preview_to_engine_inputs(res)
    scales = inp.get('scales') or []
    R["n_scales"] = len(scales)
    R["n_conditions"] = len(inp.get('conditions') or [])
    R["n_factors"] = len(inp.get('factors') or [])
    oeq = inp.get('open_ended_questions')
    R["n_oeq"] = len(oeq) if oeq else 0

    # Capture scale metadata for validation. Key by BOTH display name and
    # variable_name because the engine emits DataFrame columns using
    # variable_name (e.g. "Foo_1") while QSF detection may set a different
    # human-readable "name". Keying by both guarantees spec_for() can map a
    # column back to its scale regardless of which key the column derives from.
    scale_specs = {}
    for s in scales:
        if not isinstance(s, dict):
            continue
        spec = {
            "dv_type": s.get('dv_type') or s.get('type') or s.get('response_type'),
            "scale_min": s.get('scale_min'),
            "scale_max": s.get('scale_max'),
            # For constant_sum the engine uses int(round(scale_max)) as the row
            # total (falling back to 100 when scale_max < n_items). There is no
            # separate "total" field in the bridge output, so derive from these.
            "total": s.get('total') if s.get('total') is not None else s.get('sum_total'),
            "n_items": s.get('n_items') if s.get('n_items') is not None else (
                s.get('num_items') if s.get('num_items') is not None else s.get('items')),
            "scale_points": s.get('scale_points'),
            "items": s.get('items'),
        }
        for k in (s.get('name'), s.get('variable_name'), s.get('variable'), s.get('label')):
            if k:
                scale_specs[str(k)] = spec
    R["scale_specs"] = scale_specs

    R["stage"] = "ctor"
    eng = EnhancedSimulationEngine(
        study_title=res.survey_name,
        study_description=(res.study_context or {}).get('description', ''),
        sample_size=30,
        conditions=inp['conditions'],
        factors=inp['factors'],
        scales=inp['scales'],
        additional_vars=[],
        demographics={'gender_quota': 50, 'age_mean': 35, 'age_sd': 12},
        open_ended_questions=inp.get('open_ended_questions'),
        seed=7,
    )
    if eng.llm_generator is not None:
        try:
            eng.llm_generator.disable_permanently('test')
        except Exception:
            pass

    R["stage"] = "generate"
    df, meta = eng.generate()

    R["stage"] = "validate"
    R["shape"] = list(df.shape)
    cols = list(df.columns)
    R["columns"] = cols[:200]

    # --- Check 5a: duplicate column names ---
    seen = {}
    dups = []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
    dups = [c for c, n in seen.items() if n > 1]
    R["dup_columns"] = dups

    # --- Check 5b: condition column present? ---
    cond_candidates = [c for c in cols if str(c).strip().lower() in
                       ('condition', 'conditions', 'group', 'arm', 'treatment', 'cond')]
    # only flag if there ARE conditions to assign
    R["has_condition_col"] = bool(cond_candidates)
    R["missing_condition_col"] = (R["n_conditions"] > 1 and not cond_candidates)
    R["cond_col_name"] = cond_candidates[0] if cond_candidates else None

    # Identify numeric columns
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    # --- Check 2: NaN/inf ---
    nan_all = []          # 100% NaN columns
    inf_cols = []         # columns containing +/-inf
    for c in num_cols:
        col = df[c]
        n = len(col)
        nn = int(col.isna().sum())
        if n > 0 and nn == n:
            nan_all.append(c)
        try:
            if np.isinf(col.to_numpy(dtype='float64', na_value=np.nan)).any():
                inf_cols.append(c)
        except Exception:
            pass
    R["nan_all_columns"] = nan_all
    R["inf_columns"] = inf_cols

    # --- Map DataFrame columns to scale specs (handle multi-item: name, name_1, name_2 ...) ---
    def spec_for(colname):
        cn = str(colname)
        if cn in scale_specs:
            return cn, scale_specs[cn]
        # strip trailing _<int>
        import re as _re
        m = _re.match(r'^(.*)_(\d+)$', cn)
        if m and m.group(1) in scale_specs:
            return m.group(1), scale_specs[m.group(1)]
        return None, None

    # --- Check 4: numeric DV out of [scale_min, scale_max] (likert/slider/single etc.) ---
    out_of_range = []   # list of [col, min_seen, max_seen, smin, smax, n_bad]
    for c in num_cols:
        base, spec = spec_for(c)
        if not spec:
            continue
        dvt = (str(spec.get('dv_type') or '')).lower()
        if dvt in ('constant_sum', 'rank_order', 'rank'):
            continue  # handled separately
        smin, smax = spec.get('scale_min'), spec.get('scale_max')
        if smin is None or smax is None:
            continue
        col = df[c].dropna()
        if col.empty:
            continue
        try:
            lo, hi = float(col.min()), float(col.max())
            nbad = int(((col < float(smin) - 1e-9) | (col > float(smax) + 1e-9)).sum())
            if nbad > 0:
                out_of_range.append([c, lo, hi, smin, smax, nbad])
        except Exception:
            pass
    R["out_of_range"] = out_of_range[:40]
    R["n_out_of_range"] = len(out_of_range)

    # --- Check 3a: constant_sum columns must sum to total per row ---
    cs_bad = []  # [base, total, n_rows_bad, example_sum]
    cs_groups = {}
    for c in num_cols:
        base, spec = spec_for(c)
        if spec and (str(spec.get('dv_type') or '')).lower() == 'constant_sum':
            cs_groups.setdefault(base, []).append(c)
    for base, gcols in cs_groups.items():
        spec = scale_specs.get(base, {})
        # Derive expected per-row total exactly as the engine does:
        # _total = int(round(scale_max)) if scale_max >= n_items else 100.
        # Honor an explicit 'total'/'sum_total' if one was ever provided.
        k = len(gcols)
        total = spec.get('total')
        if total is None:
            smax = spec.get('scale_max')
            try:
                smax_f = float(smax) if smax is not None else None
            except (TypeError, ValueError):
                smax_f = None
            if smax_f is not None and smax_f >= k:
                total = int(round(smax_f))
            else:
                total = 100
        if k < 2:
            # single column constant_sum: each value should equal total (degenerate) -> skip strict
            continue
        sub = df[gcols].dropna(how='all')
        if sub.empty:
            continue
        rowsums = sub.sum(axis=1, skipna=True)
        nbad = int((np.abs(rowsums - float(total)) > 1e-6).sum())
        if nbad > 0:
            cs_bad.append([base, total, nbad, float(rowsums.iloc[0]) if len(rowsums) else None, k])
    R["constant_sum_bad"] = cs_bad
    R["constant_sum_groups"] = {k: len(v) for k, v in cs_groups.items()}

    # --- Check 3b: rank_order columns must be valid permutations of 1..n_items ---
    ro_bad = []
    ro_groups = {}
    for c in num_cols:
        base, spec = spec_for(c)
        if spec and (str(spec.get('dv_type') or '')).lower() in ('rank_order', 'rank'):
            ro_groups.setdefault(base, []).append(c)
    for base, gcols in ro_groups.items():
        spec = scale_specs.get(base, {})
        n_items = spec.get('n_items') or len(gcols)
        if len(gcols) < 2:
            continue  # need multiple columns to form a ranking
        expected = set(range(1, int(n_items) + 1))
        sub = df[gcols].dropna(how='all')
        nbad = 0
        for _, row in sub.iterrows():
            vals = [v for v in row.tolist() if v == v]  # drop nan
            try:
                ivals = [int(round(float(v))) for v in vals]
            except Exception:
                nbad += 1
                continue
            if set(ivals) != expected or len(ivals) != len(expected):
                nbad += 1
        if nbad > 0:
            ro_bad.append([base, int(n_items), len(gcols), nbad, len(sub)])
    R["rank_order_bad"] = ro_bad
    R["rank_order_groups"] = {k: len(v) for k, v in ro_groups.items()}

    # --- Check 5c: zero-variance DV columns (numeric, mapped to a scale, >1 non-nan, not joint-DV component expected-constant) ---
    zero_var = []
    for c in num_cols:
        base, spec = spec_for(c)
        if not spec:
            continue
        dvt = (str(spec.get('dv_type') or '')).lower()
        col = df[c].dropna()
        if len(col) < 5:
            continue
        try:
            if float(col.std()) == 0.0:
                zero_var.append([c, float(col.iloc[0]), dvt])
        except Exception:
            pass
    R["zero_variance"] = zero_var[:40]
    R["n_zero_variance"] = len(zero_var)

    R["stage"] = "done"
    R["ok"] = True
    emit()

except Exception as e:
    tb = traceback.format_exc().strip().splitlines()
    R["ok"] = False
    R["exc_type"] = type(e).__name__
    R["exc_msg"] = str(e)[:300]
    R["tb_last"] = tb[-1] if tb else ""
    # find last frame referencing our source
    src_line = ""
    for ln in reversed(tb):
        if 'simulation_app' in ln or '.py' in ln:
            src_line = ln.strip()
            break
    R["tb_src"] = src_line
    emit()
