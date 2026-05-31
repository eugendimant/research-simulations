import os, sys, time
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')
import app
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
path = sys.argv[1]
no_oe = len(sys.argv) > 2 and sys.argv[2] == 'nooe'
res = QSFPreviewParser().parse(open(path,'rb').read())
inp = app._preview_to_engine_inputs(res)
oeq = None if no_oe else inp.get('open_ended_questions')
print("noeq passed:", len(oeq) if oeq else 0); sys.stdout.flush()
eng = EnhancedSimulationEngine(study_title=res.survey_name, study_description='', sample_size=30, conditions=inp['conditions'], factors=inp['factors'], scales=inp['scales'], additional_vars=[], demographics={'gender_quota':50,'age_mean':35,'age_sd':12}, open_ended_questions=oeq, seed=7)
print("llm_generator is None?", eng.llm_generator is None); sys.stdout.flush()
if eng.llm_generator is not None:
    r = eng.llm_generator.disable_permanently('test')
    print("disabled, _force_disabled=", getattr(eng.llm_generator,'_force_disabled', '??'), " _api_available=", getattr(eng.llm_generator,'_api_available','??')); sys.stdout.flush()
t0=time.time()
df, meta = eng.generate()
print("GEN_OK %.1fs shape=%s" % (time.time()-t0, df.shape)); sys.stdout.flush()
