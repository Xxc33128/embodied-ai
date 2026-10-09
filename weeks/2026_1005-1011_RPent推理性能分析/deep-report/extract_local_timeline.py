import json,hashlib,re
from pathlib import Path
R=Path('/data/rpent2/deep-timing-20261008')
records=json.loads((R/'batch/state.json').read_text())['records']
NAMES={'primitive.'+n for n in ('pi0_pick','pi0_doubled','move_to','move_pose','release','set_gripper','rotate_wrist','rotate_pitch')}
output=[]
for model in ('gpt-6-astra','gpt-6-sol'):
 for task in (0,9):
  for seed in (0,1):
   key=f'{model}/libero_10_swap/t{task}_s{seed}';rec=records[key];attempt=rec.get('attempt','attempt-001')
   assert rec['state'] in ('success','task_failure') and rec['measurement_invalid'] is False
   base=R/'formal-results'/key/attempt;streams=list(base.glob('*.stream.jsonl'));assert len(streams)==1
   raw=streams[0].read_bytes();tools=[];seen=set()
   for line,entry in enumerate(raw.splitlines(),1):
    ev=json.loads(entry);p=ev.get('payload') or {};i=p.get('item') or {}
    if ev.get('method')!='item/completed' or i.get('type')!='mcpToolCall' or i['id'] in seen:continue
    seen.add(i['id']);assert re.fullmatch('[A-Za-z0-9_.-]+',i['tool'])
    tools.append({'item_sha256':hashlib.sha256(i['id'].encode()).hexdigest(),'name':i['tool'],'source_line':line})
   stages=[];hashes={}
   for path in sorted((R/'measurements'/key/attempt).glob('spans-*.jsonl')):
    blob=path.read_bytes();hashes[path.name]=hashlib.sha256(blob).hexdigest()
    for line in blob.splitlines():
     row=json.loads(line)
     if row['name'] in NAMES|{'pi.handler','sim.env.step','sim.render'}:stages.append(row)
   roots=sorted([s for s in stages if s['name'] in NAMES],key=lambda x:x['start_ns']);assert roots
   domain=roots[0]['clock_domain'];assert all(x['clock_domain']==domain and x['clock_domain_kind']=='kernel_boot_id_sha256' for x in roots)
   assert all(a['end_ns']<=b['start_ns'] for a,b in zip(roots,roots[1:]))
   origin=roots[0]['start_ns'];actions=[]
   for a in roots:
    lanes={}
    for lane,name in [('pi','pi.handler'),('sim','sim.env.step'),('render','sim.render')]:
     included=sorted([s for s in stages if s['name']==name and s['clock_domain']==domain and a['start_ns']<=s['start_ns']<=s['end_ns']<=a['end_ns']],key=lambda x:x['start_ns'])
     lanes[lane]=[dict(start_s=(s['start_ns']-a['start_ns'])/1e9,end_s=(s['end_ns']-a['start_ns'])/1e9,name=name) for s in included]
    actions.append(dict(name=a['name'].removeprefix('primitive.'),start_s=(a['start_ns']-origin)/1e9,end_s=(a['end_ns']-origin)/1e9,detail=dict(window_s=(a['end_ns']-a['start_ns'])/1e9,lanes=lanes)))
   output.append(dict(model=model,task=task,seed=seed,attempt=attempt,tools=tools,stream_sha256=hashlib.sha256(raw).hexdigest(),measurement_sha256=hashes,local=dict(window_s=(roots[-1]['end_ns']-origin)/1e9,actions=actions),scope='local same-kernel monotonic; service spans fully contained in primitive window; not RPC causal match'))
print(json.dumps({'episodes':output},ensure_ascii=False,separators=(',',':')))
