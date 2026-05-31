import os, sys, inspect, traceback
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
sys.path.insert(0, '/home/user/research-simulations/simulation_app')
out = []
try:
    import app
    out.append("app imported OK")
    out.append("has _preview_to_engine_inputs: %s" % hasattr(app, '_preview_to_engine_inputs'))
    # show source line of bridge
    fn = getattr(app, '_preview_to_engine_inputs', None)
    if fn:
        try:
            src = inspect.getsource(fn)
            out.append("=== BRIDGE SOURCE (first 4000 chars) ===")
            out.append(src[:4000])
        except Exception as e:
            out.append("could not get source: %r" % e)
except Exception:
    out.append("APP IMPORT FAILED:")
    out.append(traceback.format_exc())

with open('/home/user/research-simulations/_introspect_out.txt', 'w') as f:
    f.write("\n".join(str(x) for x in out))
print("DONE")
