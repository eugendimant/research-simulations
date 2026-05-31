import sys, inspect
sys.path.insert(0, 'simulation_app')
from utils.enhanced_simulation_engine import EnhancedSimulationEngine as E

out = []
def dump(name):
    obj = getattr(E, name, None)
    out.append("\n\n##### %s #####" % name)
    if obj is None:
        out.append("(not found)"); return
    try:
        out.append(inspect.getsource(obj))
    except Exception as e:
        out.append("err: %r" % e)

for m in ['_build_behavioral_profile', '_assign_persona', '_initialize_personas',
          '_looks_like_economic_game', '_infer_game_type']:
    dump(m)

with open('_audit_out3.txt', 'w') as f:
    f.write("\n".join(out))
print("len", len("\n".join(out)))
