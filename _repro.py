import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')
import numpy as np, pandas as pd
import app
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

path = sys.argv[1]
target_cols = sys.argv[2].split(',') if len(sys.argv) > 2 else None
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200

res = QSFPreviewParser().parse(open(path, 'rb').read())
inp = app._preview_to_engine_inputs(res)
specs = {}
for s in inp['scales']:
    if isinstance(s, dict):
        nm = s.get('variable_name') or s.get('name')
        specs[nm] = dict(type=s.get('type'), smin=s.get('scale_min'), smax=s.get('scale_max'),
                         ni=s.get('num_items'), sp=s.get('scale_points'),
                         qtext=(s.get('question_text') or '')[:70])
eng = EnhancedSimulationEngine(
    study_title=res.survey_name, study_description=(res.study_context or {}).get('description', ''),
    sample_size=N, conditions=inp['conditions'], factors=inp['factors'], scales=inp['scales'],
    additional_vars=[], demographics={'gender_quota':50,'age_mean':35,'age_sd':12},
    open_ended_questions=None, seed=7)
if eng.llm_generator is not None:
    eng.llm_generator.disable_permanently('test')
df, meta = eng.generate()

out = ["FILE %s  N=%d" % (os.path.basename(path), N)]
# show spec for targeted base
def base_of(c):
    return c.rsplit('_',1)[0] if '_' in c and c.rsplit('_',1)[1].isdigit() else c
cols = target_cols or list(df.columns)
for c in cols:
    if c not in df.columns:
        out.append("  [missing col %s]" % c); continue
    if not pd.api.types.is_numeric_dtype(df[c]):
        out.append("  %s: non-numeric" % c); continue
    b = base_of(c)
    sp = specs.get(b) or specs.get(c) or {}
    s = df[c]
    vc = s.value_counts().head(6).to_dict()
    out.append("  %s base=%s spec(type=%s min=%s max=%s ni=%s sp=%s)" % (
        c, b, sp.get('type'), sp.get('smin'), sp.get('smax'), sp.get('ni'), sp.get('sp')))
    out.append("    seen=[%s..%s] mean=%.2f std=%.2f n_unique=%d top=%s qtext=%r" % (
        s.min(), s.max(), float(s.mean()), float(s.std()), s.nunique(), vc, sp.get('qtext')))
open('_repro_out.txt','w').write("\n".join(out))
print("done")
