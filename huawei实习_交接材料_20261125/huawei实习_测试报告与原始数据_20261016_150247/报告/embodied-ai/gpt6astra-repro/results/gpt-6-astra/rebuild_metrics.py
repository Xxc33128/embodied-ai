"""Export report metrics from raw robot logs and numeric Codex usage events.

Does not export session messages, private reasoning, credentials, or full rollouts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

VALID = {
    '/root/persistent20_manager/policy_01': 'trial-01',
    '/root/resume_persistent20/policy_02': 'trial-02-attempt-02',
    '/root/resume_persistent20/policy_03': 'trial-03',
    '/root/policy_04': 'trial-04',
    '/root/policy_05': 'trial-05',
}
INTERRUPTED = {
    '/root/persistent20_manager/policy_02': 'trial-02',
    '/root/policy_06': 'trial-06',
}
MANAGERS = {'/root/standard20_coordinator', '/root/persistent20_manager', '/root/resume_persistent20'}
REVIEWERS = {'/root/blind_video_review', '/root/blind_review_second', '/root/blind_review_remaining'}
ROOT_ID = '01a09f0e-afac-7be0-94f8-5145bc8cc6a4'


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def csv_dump(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def extract(root, sessions, output):
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in sorted(sessions.glob('*.jsonl')):
        with path.open() as stream:
            meta = json.loads(next(stream)).get('payload', {})
        source = meta.get('source', {})
        if not isinstance(source, dict):
            continue
        spawn = source.get('subagent', {}).get('thread_spawn', {})
        if spawn:
            candidates.append((path, meta, spawn))
    related = {ROOT_ID}
    while True:
        expanded = related | {m['id'] for _, m, s in candidates if s.get('parent_thread_id') in related}
        if expanded == related:
            break
        related = expanded
    usage = []
    for path, meta, spawn in candidates:
        if meta['id'] not in related:
            continue
        agent = spawn['agent_path']
        category = ('valid_policy' if agent in VALID else 'interrupted_policy' if agent in INTERRUPTED else
                    'management' if agent in MANAGERS else 'review' if agent in REVIEWERS else
                    'superseded_per_decision' if '/standard20_coordinator/decision_' in agent else 'unclassified')
        events, models, efforts = [], set(), set()
        tool_calls = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                event = json.loads(line)
                payload = event.get('payload', {})
                if event['type'] == 'event_msg' and payload.get('type') == 'token_count' and payload.get('info'):
                    events.append((line_number, event.get('timestamp'), payload['info']))
                elif event['type'] == 'turn_context':
                    models.add(payload.get('model'))
                    efforts.add(payload.get('effort'))
                elif event['type'] == 'response_item' and payload.get('type') in ('function_call', 'custom_tool_call'):
                    tool_calls += 1
        if not events:
            continue
        last_line, last_time, info = events[-1]
        total = info['total_token_usage']
        unique = []
        previous = 0
        for _, _, item in events:
            value = item['total_token_usage']['total_tokens']
            assert value >= previous, f'Non-monotonic usage: {agent}'
            if value > previous:
                unique.append(item)
            previous = value
        sum_matches = all(sum(x['last_token_usage'].get(k, 0) for x in unique) == v for k, v in total.items())
        assert sum_matches, f'Usage increments do not reconcile: {agent}'
        assert total['total_tokens'] == total['input_tokens'] + total['output_tokens']
        row = {'agent_path': agent, 'category': category, 'attempt_path': VALID.get(agent, INTERRUPTED.get(agent)),
               'session_id': meta['id'], 'model': sorted(models), 'effort': sorted(efforts),
               'usage_source_file': path.name, 'usage_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'final_usage_line': last_line, 'first_usage_timestamp': events[0][1], 'last_usage_timestamp': last_time,
               'usage_events': len(events), 'unique_usage_increments': len(unique), 'tool_calls': tool_calls,
               **total, 'noncached_input_tokens': total['input_tokens'] - total['cached_input_tokens'],
               'increment_sum_matches_cumulative': sum_matches, 'provider_cost_usd': None}
        usage.append(row)
    by_attempt = {x['attempt_path']: x for x in usage if x['category'] == 'valid_policy'}
    assert len(by_attempt) == 5
    timing, decisions = [], []
    for attempt in VALID.values():
        folder = root / attempt
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'success'
        rows = [json.loads(x) for x in (folder / 'physics.jsonl').read_text().splitlines()]
        starts, ends = [], []
        n = result['calls']
        for call in range(1, n + 1):
            starts.append(json.loads((folder / f'call-{call:02d}-started.json').read_text()))
            ends.append(json.loads((folder / f'call-{call:02d}-ended.json').read_text()))
        for call in range(1, n + 1):
            start, end = starts[call - 1], ends[call - 1]
            step0 = start['physics_steps']
            step1 = starts[call]['physics_steps'] if call < n else result['physics_steps']
            assert step0 == end['physics_steps']
            response = json.loads((folder / f'incoming-{call:02d}.json').read_text())
            function = response['choices'][0]['message']['tool_calls'][0]['function']
            args = json.loads(function['arguments'])
            decisions.append({'trial': result['trial_index'] + 1, 'call': call, 'start_step': step0,
                              'end_step': step1, 'control_steps': step1 - step0, 'decision_wait_seconds': round(end['wait_seconds'], 4),
                              'tool': function['name'], 'targets_json': json.dumps(args.get('targets'), ensure_ascii=False),
                              'recorded_note': args.get('note', args.get('reason', '')),
                              'eef_before': json.dumps(rows[step0]['state']['eef_pos']),
                              'eef_after': json.dumps(rows[step1]['state']['eef_pos']),
                              'grasped_after_private_audit': rows[step1]['grasped'],
                              'cube_bottom_after_private_audit': rows[step1]['cube_bottom_z']})
        wait = sum(x['wait_seconds'] for x in ends)
        u = by_attempt[attempt]
        timing.append({'trial': result['trial_index'] + 1, 'scene_seed': 2000 + result['trial_index'],
                       'attempt_path': attempt, 'env_success': result['metrics']['success_at_end'],
                       'policy_decisions': n, 'physics_steps': result['physics_steps'], 'sim_seconds': result['physics_steps'] / 20,
                       'runner_wall_seconds': result['wall_seconds'], 'decision_wait_seconds': wait,
                       'other_runner_seconds': result['wall_seconds'] - wait,
                       'wait_pct': 100 * wait / result['wall_seconds'], 'mean_wait_seconds': wait / n,
                       'max_wait_seconds': max(x['wait_seconds'] for x in ends),
                       'first_request_unix': starts[0]['wall_time'], 'last_response_unix': ends[-1]['wall_time'],
                       'session_token_events': u['unique_usage_increments'],
                       **{key: u[key] for key in ['input_tokens', 'cached_input_tokens', 'noncached_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens']},
                       'provider_cost_usd': None})
    groups = {}
    for category in sorted({u['category'] for u in usage}):
        selected = [u for u in usage if u['category'] == category]
        groups[category] = {'sessions': len(selected), **{key: sum(x[key] for x in selected) for key in
                           ['input_tokens', 'cached_input_tokens', 'noncached_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens']}}
    summary = {'completed_valid_trials': len(timing), 'auto_successes': sum(t['env_success'] for t in timing),
               'policy_decisions_total': sum(t['policy_decisions'] for t in timing),
               'runner_wall_seconds_total': sum(t['runner_wall_seconds'] for t in timing),
               'decision_wait_seconds_total': sum(t['decision_wait_seconds'] for t in timing),
               'sim_seconds_total': sum(t['sim_seconds'] for t in timing),
               'mean_policy_decisions': statistics.mean(t['policy_decisions'] for t in timing),
               'mean_runner_wall_seconds': statistics.mean(t['runner_wall_seconds'] for t in timing),
               'mean_sim_seconds': statistics.mean(t['sim_seconds'] for t in timing),
               'first_to_last_request_span_seconds': max(t['last_response_unix'] for t in timing) - min(t['first_request_unix'] for t in timing),
               'usage_by_category': groups, 'root_conversation_tokens_attributed': None,
               'account_or_provider_cost_usd': None, 'notes': 'Cache is a subset of input; reasoning token count is not added again to output. No private reasoning content exported. Main results include exactly five valid policy sessions; other categories are separate.'}
    dump(output / 'subagent-usage.json', usage)
    csv_dump(output / 'subagent-usage.csv', [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in row.items()} for row in usage])
    csv_dump(output / 'tokens-and-timing.csv', timing)
    csv_dump(output / 'decision-trace.csv', decisions)
    dump(output / 'metrics.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--sessions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.logs, args.sessions, args.output)
