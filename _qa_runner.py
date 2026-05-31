"""Corpus runner: invoke _qa_worker.py as an isolated subprocess per QSF with a
per-file timeout so a hang on one file cannot block the corpus."""
import os, sys, json, glob, subprocess, time

ROOT = '/home/user/research-simulations'
os.chdir(ROOT)
files = sorted(glob.glob('simulation_app/example_files/*.qsf'))
PER_FILE_TIMEOUT = 300
OUT = os.path.join(ROOT, '_qa_results.jsonl')

done = 0
with open(OUT, 'w') as out:
    for idx, path in enumerate(files, 1):
        base = os.path.basename(path)
        t0 = time.time()
        rec = None
        try:
            proc = subprocess.run(
                [sys.executable, '_qa_worker.py', path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=PER_FILE_TIMEOUT,
            )
            txt = proc.stdout.decode('utf-8', 'replace')
            line = None
            for ln in txt.splitlines():
                if ln.startswith('@@RESULT@@'):
                    line = ln[len('@@RESULT@@'):]
            if line is not None:
                rec = json.loads(line)
            else:
                rec = {"file": base, "ok": False, "exc_type": "NoResultLine",
                       "exc_msg": txt.strip()[-300:], "tb_src": "", "stage": "subprocess",
                       "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            rec = {"file": base, "ok": False, "exc_type": "TIMEOUT",
                   "exc_msg": "exceeded %ds" % PER_FILE_TIMEOUT, "stage": "timeout"}
        except Exception as e:
            rec = {"file": base, "ok": False, "exc_type": "RunnerError",
                   "exc_msg": repr(e)[:300], "stage": "runner"}
        rec["_elapsed_s"] = round(time.time() - t0, 1)
        rec["_idx"] = idx
        out.write(json.dumps(rec, default=str) + "\n")
        out.flush()
        os.fsync(out.fileno())
        done += 1
        flag = "OK" if rec.get("ok") else ("ERR:" + str(rec.get("exc_type")))
        print("PROGRESS %d/%d %s [%ss] %s" % (idx, len(files), base[:40], rec["_elapsed_s"], flag), flush=True)

print("@@RUN_DONE@@ %d files" % done, flush=True)
