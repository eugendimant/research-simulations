import json

recs = []
bad_json = 0
with open('/home/user/research-simulations/_qa_results.jsonl') as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except Exception:
            bad_json += 1

total = len(recs)
ok = [r for r in recs if r.get('ok') is True]
crashed = [r for r in recs if r.get('ok') is not True]

nan_inf = []
invalid_joint = []
out_of_range = []
zero_var = []
dup_cols = []
missing_cond = []

for r in ok:
    f = r.get('file')
    nanall = r.get('nan_all_columns') or []
    infc = r.get('inf_columns') or []
    if nanall or infc:
        nan_inf.append((f, {'nan_all': nanall, 'inf': infc}))
    csb = r.get('constant_sum_bad') or []
    rob = r.get('rank_order_bad') or []
    if csb or rob:
        invalid_joint.append((f, {'constant_sum_bad': csb, 'rank_order_bad': rob}))
    if (r.get('n_out_of_range') or 0) > 0:
        out_of_range.append((f, r.get('out_of_range') or []))
    if (r.get('n_zero_variance') or 0) > 0:
        zero_var.append((f, r.get('zero_variance') or []))
    if r.get('dup_columns'):
        dup_cols.append((f, r.get('dup_columns')))
    if r.get('missing_condition_col'):
        missing_cond.append((f, r.get('n_conditions')))

def w(o, title, items, fmt, cap=15):
    o.write("\n### %s (%d)\n" % (title, len(items)))
    for it in items[:cap]:
        o.write("  " + fmt(it) + "\n")
    if len(items) > cap:
        o.write("  ... and %d more\n" % (len(items) - cap))

with open('/home/user/research-simulations/_QA_REPORT.txt', 'w') as o:
    o.write("=" * 64 + "\n")
    o.write("QA CORPUS REPORT (N=30/file, seed=7, LLM disabled)\n")
    o.write("=" * 64 + "\n")
    o.write("Total result records:      %d\n" % total)
    o.write("Bad JSON lines (skipped):  %d\n" % bad_json)
    o.write("Simulated OK:              %d\n" % len(ok))
    o.write("Crashes/exceptions:        %d\n" % len(crashed))
    o.write("NaN-all / inf columns:     %d\n" % len(nan_inf))
    o.write("Invalid joint-DV:          %d\n" % len(invalid_joint))
    o.write("Out-of-range numeric DV:   %d\n" % len(out_of_range))
    o.write("Zero-variance DV cols:     %d\n" % len(zero_var))
    o.write("Duplicate column names:    %d\n" % len(dup_cols))
    o.write("Missing CONDITION column:  %d\n" % len(missing_cond))

    # crash type tally
    from collections import Counter
    ct = Counter(str(r.get('exc_type')) for r in crashed)
    o.write("\nCrash type tally: %s\n" % dict(ct))
    # stage tally for crashes
    cs = Counter(str(r.get('stage')) for r in crashed)
    o.write("Crash stage tally: %s\n" % dict(cs))

    w(o, "CRASHES", crashed,
      lambda r: "%-46s | %s @%s: %s | %s" % (
          str(r.get('file'))[:46], r.get('exc_type'), r.get('stage'),
          str(r.get('exc_msg'))[:80], str(r.get('tb_src'))[:80]), cap=40)

    w(o, "INVALID JOINT-DV (constant_sum/rank_order)", invalid_joint,
      lambda t: "%-46s | %s" % (str(t[0])[:46], json.dumps(t[1])[:170]))

    w(o, "OUT-OF-RANGE numeric DV [col, seenmin, seenmax, smin, smax, nbad]", out_of_range,
      lambda t: "%-40s | %s" % (str(t[0])[:40], json.dumps(t[1])[:180]))

    w(o, "NaN-all / inf columns", nan_inf,
      lambda t: "%-46s | %s" % (str(t[0])[:46], json.dumps(t[1])[:170]))

    w(o, "ZERO-VARIANCE DV columns [col, value, type]", zero_var,
      lambda t: "%-40s | %s" % (str(t[0])[:40], json.dumps(t[1])[:180]))

    w(o, "DUPLICATE column names", dup_cols,
      lambda t: "%-46s | %s" % (str(t[0])[:46], json.dumps(t[1])[:120]))

    w(o, "MISSING CONDITION column (n_conditions>1)", missing_cond,
      lambda t: "%-46s | n_conditions=%s" % (str(t[0])[:46], t[1]))

    # Corpus-wide stats
    shapes = [r.get('shape') for r in ok if r.get('shape')]
    if shapes:
        rows = [s[0] for s in shapes]
        cols = [s[1] for s in shapes]
        o.write("\n### Corpus stats\n")
        o.write("  rows: all == 30? %s (min=%s max=%s)\n" % (all(x == 30 for x in rows), min(rows), max(rows)))
        o.write("  cols: min=%s max=%s\n" % (min(cols), max(cols)))
        slow = sorted([(r.get('_elapsed_s', 0), r.get('file')) for r in recs], reverse=True)[:8]
        o.write("  slowest files: %s\n" % [(s, f[:35]) for s, f in slow])

print("ANALYSIS_DONE total=%d ok=%d crashed=%d nan=%d joint=%d oor=%d zv=%d dup=%d miscond=%d" % (
    total, len(ok), len(crashed), len(nan_inf), len(invalid_joint),
    len(out_of_range), len(zero_var), len(dup_cols), len(missing_cond)))
