import os, sys, time
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')
import app
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
path = sys.argv[1]
t0=time.time()
res = QSFPreviewParser().parse(open(path,'rb').read())
print("PARSE_OK %.1fs" % (time.time()-t0)); sys.stdout.flush()
t0=time.time()
inp = app._preview_to_engine_inputs(res)
print("BRIDGE_OK %.1fs  nscales=%d nconds=%d noeq=%d" % (time.time()-t0, len(inp.get('scales') or []), len(inp.get('conditions') or []), len(inp.get('open_ended_questions') or []) if inp.get('open_ended_questions') else 0)); sys.stdout.flush()
for s in (inp.get('scales') or []):
    if isinstance(s,dict): print("  S:", {k:s.get(k) for k in ('name','type','dv_type','response_type','scale_min','scale_max','total','sum_total','n_items')})
t0=time.time()
eng = EnhancedSimulationEngine(study_title=res.survey_name, study_description=(res.study_context or {}).get('description',''), sample_size=30, conditions=inp['conditions'], factors=inp['factors'], scales=inp['scales'], additional_vars=[], demographics={'gender_quota':50,'age_mean':35,'age_sd':12}, open_ended_questions=inp.get('open_ended_questions'), seed=7)
print("ENGINE_CTOR_OK %.1fs llm=%s" % (time.time()-t0, eng.llm_generator is not None)); sys.stdout.flush()
if eng.llm_generator is not None: eng.llm_generator.disable_permanently('test')
t0=time.time()
df, meta = eng.generate()
print("GEN_OK %.1fs shape=%s" % (time.time()-t0, df.shape)); sys.stdout.flush()
