"""Fast scan: parse + bridge only (no simulation). Inventory scale types and
flag any parse/bridge crashes. One JSON line per file to stdout."""
import os, sys, json, glob, traceback, warnings
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, 'simulation_app')

import app
from utils.qsf_preview import QSFPreviewParser

files = sorted(glob.glob('simulation_app/example_files/*.qsf'))
type_counter = {}
joint_files = []
crashes = []
for path in files:
    base = os.path.basename(path)
    rec = {"file": base}
    try:
        res = QSFPreviewParser().parse(open(path, 'rb').read())
        inp = app._preview_to_engine_inputs(res)
        types = []
        for s in (inp.get('scales') or []):
            if isinstance(s, dict):
                t = str(s.get('type') or s.get('dv_type') or 'likert').lower()
                types.append(t)
                type_counter[t] = type_counter.get(t, 0) + 1
        rec["types"] = types
        rec["n_conditions"] = len(inp.get('conditions') or [])
        if any(t in ('constant_sum', 'rank_order', 'ranking', 'rank order') for t in types):
            joint_files.append((base, [t for t in types if t in
                                ('constant_sum', 'rank_order', 'ranking', 'rank order')]))
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        crashes.append((base, rec["error"]))
    sys.stdout.write("@@SCAN@@" + json.dumps(rec, default=str) + "\n")

sys.stdout.write("@@TYPES@@" + json.dumps(type_counter) + "\n")
sys.stdout.write("@@JOINT@@" + json.dumps(joint_files) + "\n")
sys.stdout.write("@@PARSEBRIDGE_CRASHES@@" + json.dumps(crashes) + "\n")
sys.stdout.flush()
