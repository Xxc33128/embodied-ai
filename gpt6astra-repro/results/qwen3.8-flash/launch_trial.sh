#!/bin/zsh
set -e
INDEX=$1
B="logs-qwen3.8flash-subagent20-20260915-1545"
TRIAL=$(printf 'trial-%02d' $((INDEX+1)))
PUB="/tmp/bowl-qwen38flash-20260915-1545/$TRIAL"
cd "/Users/xerxes3/Documents/huawei实习/gpt6astra-repro"
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
python3 - "$INDEX" "$TRIAL" "$RUNNER" <<'EOF'
import json, sys, time
from pathlib import Path
index, trial, runner = int(sys.argv[1]), sys.argv[2], sys.argv[3]
b = Path('logs-qwen3.8flash-subagent20-20260915-1545')
ev = {'trial': index+1, 'index': index, 'event': 'started', 'model': 'qwen3.8-flash-subagent',
      'access_mode': 'persistent-subagent', 'runner_pid': runner,
      'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'prompt_file': f'prompt-{trial}.txt',
      'public': f'/tmp/bowl-qwen38flash-20260915-1545/{trial}'}
with (b/'dispatch.jsonl').open('a') as f: f.write(json.dumps(ev, ensure_ascii=False)+'\n')
print('dispatch logged')
EOF
