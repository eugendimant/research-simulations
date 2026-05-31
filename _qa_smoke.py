import os, sys, traceback
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')

import app
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

path = sys.argv[1]
print("FILE:", os.path.basename(path))
res = QSFPreviewParser().parse(open(path, 'rb').read())
print("survey_name:", repr(res.survey_name))
inp = app._preview_to_engine_inputs(res)
print("inp keys:", list(inp.keys()))
print("n conditions:", len(inp.get('conditions') or []))
print("n factors:", len(inp.get('factors') or []))
scales = inp.get('scales') or []
print("n scales:", len(scales))
for s in scales[:8]:
    if isinstance(s, dict):
        print("  scale:", {k: s.get(k) for k in ('name','type','dv_type','response_type','scale_min','scale_max','total','n_items','items','sum_total')})
    else:
        print("  scale(non-dict):", type(s), repr(s)[:120])
oeq = inp.get('open_ended_questions')
print("n oeq:", len(oeq or []) if oeq else 0)

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
    eng.llm_generator.disable_permanently('test')
df, meta = eng.generate()
print("df shape:", df.shape)
print("columns:", list(df.columns))
print("dtypes:")
print(df.dtypes)
print(df.head(3).to_string())
