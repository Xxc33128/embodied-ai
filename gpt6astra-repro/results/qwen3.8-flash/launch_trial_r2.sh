#!/bin/zsh
# R2 launcher (2-slot rolling pool, added 2026-09-15 17:32 after two provider 429 pauses).
# Protocol-relevant things are UNCHANGED: same runner, same frozen bank, same prompt template,
# same budgets. Only the mailbox-path hygiene and the concurrency window differ from launch_trial.sh:
#  1) refuse to start when the attempt dir already exists (root cause of the earlier FileExistsError spam);
#  2) public (model-visible) mailbox path is deterministic per seed AND under a fresh r2 root, so a stale
#     mailbox from the 5-slot window can never be mistaken for a new runner's request (the 17:11 resume had
#     attached a NEW policy worker to trial-06/07 live runners because launch_trial.sh polled a shared
#     status.json owned by a still-alive runner). Ownership of a seed is enforced by guard (1): only one
#     attempt dir per seed may exist, so the deterministic mailbox is never shared by two live runners.
PUB_ROOT="/tmp/bowl-qwen38flash-r2-20260915-1730"
set -e
INDEX=$1
B="logs-qwen3.8flash-subagent20-20260915-1545"
TRIAL=$(printf 'trial-%02d' $((INDEX+1)))
cd "/Users/xerxes3/Documents/huawei实习/gpt6astra-repro"
# Pin the batch-local frozen standard copy: another session relocated the repo package at ~19:0x and the
# old hard-coded path killed runners with FileNotFoundError. Content is byte-identical (diff -r) and the
# runner still verifies prompt/system/environment-source SHA256 on every load -> protocol cannot drift.
export BOWL_EVAL_STANDARD="$PWD/$B/standard"
if [ -d "$B/$TRIAL" ]; then echo "ATTEMPT_DIR_EXISTS $B/$TRIAL (this seed is owned by another attempt; not starting a runner)"; exit 4; fi
PUB="$PUB_ROOT/$TRIAL"
mkdir -p "$PUB_ROOT"
sed "s|__PUBLIC__|$PUB|g" "$B/dispatch-prompt-template.txt" > "$B/prompt-$TRIAL.txt"
if ! grep -q "$PUB" "$B/prompt-$TRIAL.txt"; then echo "PROMPT RENDER FAILED"; exit 3; fi
if grep -q "__PUBLIC__" "$B/prompt-$TRIAL.txt"; then echo "UNSUBSTITUTED PLACEHOLDER"; exit 3; fi
nohup .venv-robosuite/bin/python run_subagent_trials_qwen.py --private "$B/$TRIAL" --public "$PUB" --bank "$B/initializations" --index $INDEX > "$B/$TRIAL.stdout.log" 2> "$B/$TRIAL.stderr.log" &
RUNNER=$!
python3 - "$RUNNER" "$PUB/status.json" <<'EOF'
import json, os, sys, time
runner, pub = int(sys.argv[1]), sys.argv[2]
t0 = time.time()
while time.time() - t0 < 300:
    try:
        st = json.load(open(pub))
        if st.get('state') == 'awaiting_decision':
            print('READY', json.dumps(st)); sys.exit(0)
    except Exception: pass
    try: os.kill(runner, 0)
    except OSError: print('RUNNER DIED EARLY'); sys.exit(1)
    time.sleep(2)
print('TIMEOUT waiting for first request'); sys.exit(2)
EOF
python3 - "$INDEX" "$TRIAL" "$RUNNER" "$PUB" <<'EOF'
import json, sys, time
from pathlib import Path
index, trial, runner, pub = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
b = Path('logs-qwen3.8flash-subagent20-20260915-1545')
ev = {'trial': index+1, 'index': index, 'event': 'started', 'model': 'qwen3.8-flash-subagent',
      'access_mode': 'persistent-subagent', 'scheduling_window': 'r2-2-slot', 'launcher': 'launch_trial_r2.sh',
      'runner_pid': runner,
      'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'prompt_file': f'prompt-{trial}.txt',
      'public': pub}
with (b/'dispatch.jsonl').open('a') as f: f.write(json.dumps(ev, ensure_ascii=False)+'\n')
print('dispatch logged')
EOF
