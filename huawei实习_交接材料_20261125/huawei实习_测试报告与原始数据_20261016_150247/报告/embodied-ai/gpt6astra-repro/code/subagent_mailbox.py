"""Policy-side file transport. Does not import the simulator or expose scoring."""
import argparse,json,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('public');p.add_argument('operation',choices=['next','submit'])
p.add_argument('--call',type=int);p.add_argument('--json');args=p.parse_args()
root=Path(args.public).resolve()
if args.operation=='submit':
    choice=json.loads(args.json)
    if set(choice)!={'name','arguments'} or not isinstance(choice['arguments'],dict):
        raise ValueError('Expected exactly {name, arguments}, a single tool call')
    status=json.loads((root/'status.json').read_text())
    if status.get('state')!='awaiting_decision' or status['call']!=args.call:
        raise ValueError('No matching pending request; do not resubmit')
    target=root/f'response-{args.call:02d}.json'
    if target.exists():raise ValueError('Response already submitted')
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(choice,ensure_ascii=False));temp.replace(target)
for _ in range(300):
    if (root/'status.json').exists():
        status=json.loads((root/'status.json').read_text())
        if status['state']=='finished':print('TRIAL_FINISHED');break
        if status['state']=='awaiting_decision' and (args.operation=='next' or status['call']!=args.call):
            request=json.loads(Path(status['request']).read_text())
            print(json.dumps({'call':status['call'],'request':request},ensure_ascii=False));break
    time.sleep(.1)
else:print('PENDING: call next to wait for the request; do not submit again')
