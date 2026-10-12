import json,hashlib
from pathlib import Path
B=Path('/data/rpent2/ds-adapter-20261009/measurements')
out={}
for p in sorted(B.glob('shim-deepseek*/libero_10_swap/*/attempt-001')):
 rows=[];domains=set();sources=[]; clocks=set()
 for f in sorted(p.glob('spans-*.jsonl')):
  sources.append({'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  for l in f.open():
   r=json.loads(l)
   if r.get('event')=='start' or not r.get('end_ns'):continue
   clocks.add(r.get('clock_domain'));n=r.get('name','');domains.add(r.get('clock_domain_kind','unknown'))
   if n=='llm.response' or n in ['sim.render','pi.client.predict','sam.client.total'] or (n.startswith('primitive.') and n.split('.')[-1] in ['move_to','move_pose','pi0_pick','pi0_doubled','release','set_gripper','rotate_pitch','rotate_wrist']):
    rows.append({'name':n,'start':r['start_ns'],'end':r['end_ns'],'lid':r.get('logical_request_id')})
 if not rows:continue
 start=min(r['start'] for r in rows)
 for r in rows:r['start']=round((r['start']-start)/1e9,6);r['end']=round((r['end']-start)/1e9,6)
 out[str(p.relative_to(B)).removesuffix('/attempt-001')]={'origin':'first selected span; not episode start','clock_kinds':sorted(domains),'clock_domain_count':len(clocks),'rows':sorted(rows,key=lambda x:x['start']),'sources':sources}
print(json.dumps(out))
