"""Read-only export of safe item-clock intervals; never export message payloads."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/data/rpent2/deep-timing-20261008')

def extract(root):
    records = json.loads((root / 'batch/state.json').read_text())['records']
    episodes = []
    for model in ('gpt-6-astra', 'gpt-6-sol'):
        for task in (0, 9):
            for seed in (0, 1):
                key = f'{model}/libero_10_swap/t{task}_s{seed}'
                record = records[key]
                assert record['state'] in ('success', 'task_failure')
                assert record['measurement_invalid'] is False
                attempt = record.get('attempt', 'attempt-001')
                assert re.fullmatch(r'attempt-\d+', attempt)
                files = list((root / 'formal-results' / key / attempt).glob('*.stream.jsonl'))
                assert len(files) == 1
                raw = files[0].read_bytes()
                starts, completed, tools, bounds = {}, set(), [], []
                for line, entry in enumerate(raw.splitlines(), 1):
                    event = json.loads(entry)
                    payload = event.get('payload') or {}
                    item = payload.get('item') or {}
                    ident = item.get('id')
                    method = event.get('method')
                    if method == 'item/started':
                        start = payload.get('started_at_ms')
                        starts[ident] = start
                        if isinstance(start, (int, float)): bounds.append(start)
                    if method != 'item/completed': continue
                    end = payload.get('completed_at_ms')
                    if isinstance(end, (int, float)): bounds.append(end)
                    if item.get('type') != 'mcpToolCall' or ident in completed: continue
                    completed.add(ident)
                    start = starts.get(ident)
                    if start is None and isinstance(item.get('duration_ms'), (int, float)):
                        start = end - item['duration_ms']
                    assert isinstance(start, (int, float)) and end >= start
                    name = item['tool']
                    assert re.fullmatch(r'[A-Za-z0-9_.-]+', name)
                    bounds.append(start)
                    tools.append(dict(name=name, start_ms=start, end_ms=end, source_line=line))
                assert tools and bounds
                origin, finish = min(bounds), max(bounds)
                tools.sort(key=lambda x: (x['start_ms'], x['end_ms']))
                for tool in tools:
                    tool['start_s'] = (tool.pop('start_ms') - origin) / 1000
                    tool['end_s'] = (tool.pop('end_ms') - origin) / 1000
                episodes.append(dict(model=model, task=task, seed=seed, attempt=attempt,
                    origin_ms=origin, window_s=(finish-origin)/1000, tools=tools,
                    stream_sha256=hashlib.sha256(raw).hexdigest()))
    return dict(schema_version=1, evidence_level='E2',
        scope='item event wall clock, first observed item boundary to last; not episode/process boundaries',
        episodes=episodes)

if __name__ == '__main__':
    print(json.dumps(extract(ROOT), ensure_ascii=False, indent=2))
