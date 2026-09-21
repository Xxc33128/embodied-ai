"""Disposable eight-device capacity probe; synthetic inputs, no task scores."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
workers = []
report = {"kind": "synthetic_capacity_probe", "cards": list(range(8)),
          "started_unix": time.time(), "rounds": [], "status": "running"}
script = Path('/workspace/repo/scripts/lp_a2_policy_server.py')
report['server_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()

def save():
    tmp = root / 'report.tmp'
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(root / 'report.json')

def interrupted(signum, frame):
    raise RuntimeError('probe interrupted by signal %s' % signum)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
save()
try:
    for card in range(8):
        folder = root / ('card%d' % card)
        folder.mkdir()
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(card),
                   ASCEND_VISIBLE_DEVICES=str(card),
                   LP_POLICY_EXCHANGE_DIR=str(folder),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        log = (folder / 'server.log').open('w')
        proc = subprocess.Popen([sys.executable, str(script), 'npu'],
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        workers.append((card, folder, proc, log))
    report['worker_pids'] = {str(c): p.pid for c, f, p, l in workers}
    save()
    deadline = time.monotonic() + 300
    while not all((f / 'server_ready_npu').exists() for c, f, p, l in workers):
        for c, f, p, l in workers:
            if p.poll() is not None:
                raise RuntimeError('card %s server exited %s' % (c, p.returncode))
        if time.monotonic() > deadline:
            raise TimeoutError('model loading exceeded 300 seconds')
        time.sleep(0.5)
    report['all_ready_unix'] = time.time()
    rng = np.random.default_rng(20260921)
    payload = dict(agentview=rng.integers(0, 256, (224,224,3), dtype=np.uint8),
                   wrist=rng.integers(0, 256, (224,224,3), dtype=np.uint8),
                   state=np.array([0.1,0.1,0.3,0,0,0,0.04,0.04], dtype=np.float32),
                   prompt=np.array('put the bowl on the plate'))
    np.savez(root / 'synthetic_input.npz', **payload)
    for round_id in range(4):
        row = {'round': round_id, 'warmup': round_id == 0, 'responses': []}
        starts = {}
        for c, f, p, l in workers:
            temp = f / ('round%d.tmp.npz' % round_id)
            np.savez(temp, **payload)
        began = time.monotonic_ns()
        for c, f, p, l in workers:
            starts[c] = time.monotonic_ns()
            (f / ('round%d.tmp.npz' % round_id)).rename(f / 'req' / ('round%d.npz' % round_id))
        deadline = time.monotonic() + 180
        pending = set(range(8))
        while pending:
            for c, f, p, l in workers:
                if c not in pending:
                    continue
                error = f / 'resp' / ('round%d.err' % round_id)
                if error.exists():
                    raise RuntimeError('card %d inference failed; see response error' % c)
                response = f / 'resp' / ('round%d.npz' % round_id)
                if response.exists():
                    received = time.monotonic_ns()
                    with np.load(response, allow_pickle=False) as data:
                        actions = data['actions']
                        meta = json.loads(str(data['meta']))
                        valid = actions.shape == (10,7) and bool(np.isfinite(actions).all())
                    row['responses'].append(dict(card=c, shape=list(actions.shape),
                        valid=valid, client_roundtrip_s=(received-starts[c])/1e9, meta=meta))
                    if not valid:
                        raise RuntimeError('invalid actions from card %d' % c)
                    pending.remove(c)
                elif p.poll() is not None:
                    raise RuntimeError('card %d server died' % c)
            if time.monotonic() > deadline:
                raise TimeoutError('response timeout; pending %s' % sorted(pending))
            if pending:
                time.sleep(0.005)
        row['batch_wall_s'] = (time.monotonic_ns()-began)/1e9
        report['rounds'].append(row)
        save()
    report['status'] = 'passed'
except BaseException as error:
    report['status'] = 'failed'
    report['error'] = '%s: %s' % (type(error).__name__, error)
finally:
    for c, f, p, l in workers:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for c, f, p, l in workers:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
        l.close()
    report['finished_unix'] = time.time()
    report['cleanup'] = 'only probe-owned process groups terminated'
    save()
