import json,hashlib,pathlib
P=pathlib.Path('embodied-ai/weeks/2026_1005-1011_RPent推理性能分析/deep-report');R=pathlib.Path('rpent-repo/deep-timing-20261008')
sha=lambda x:hashlib.sha256(x).hexdigest()
d=json.loads((P/'report-data.json').read_text());mapping=json.loads((P/'evidence/map-eight-final-001.json').read_text())['episodes'];oldcloud=json.loads((P/'evidence/cloud-eight-final-001.json').read_text())['sessions'];loc=json.loads((R/'probes/timeline-local-v2.json').read_text())['episodes'];audit=[]
for e in d['episodes']:
 key=f"{e['model']}/libero_10_swap/t{e['task']}_s{e['seed']}";m=next(x for x in mapping if x['episode_key']==key);assert m['attempt']==e['attempt']
 x=next(x for x in loc if all(x[k]==e[k] for k in ['model','task','seed','attempt']));assert x['stream_sha256']==e['timeline']['stream_sha256']
 session=next(x for x in (R/'sessions').iterdir() if sha(x.name.encode())==m['session_sha256'])
 blob=(session/'client-events.jsonl').read_bytes();events={}
 for line in blob.splitlines():
  r=json.loads(line)
  if r.get('direction')=='receive' and r.get('item_id') and r.get('method') in ('item/started','item/completed'):
   k=(sha(r['item_id'].encode()),r['method']);events.setdefault(k,[]).append(r['wall_ns'])
 tools=[]
 for t in x['tools']:
  a=events[(t['item_sha256'],'item/started')];b=events[(t['item_sha256'],'item/completed')];assert len(a)==len(b)==1 and b[0]>=a[0]
  tools.append(dict(name=t['name'],start_ns=a[0],end_ns=b[0],item_sha256=t['item_sha256']))
 old=next(s for s in oldcloud if s['session_sha256']==m['session_sha256']);required={s['request_span_sha256']:s for s in old['requests'] if s['warmup'] is False}
 telemetry=R/'telemetry'/(session.name+'.jsonl');raw=telemetry.read_bytes();spans={}
 for line in raw.splitlines():
  payload=json.loads(line);payload=payload.get('data',payload)
  for resource in payload.get('resourceSpans',[]):
   for scope in resource.get('scopeSpans',[]):
    for s in scope.get('spans',[]):
     h=sha((s['traceId']+'|'+s['spanId']).encode())
     if h in required:spans[h]=s
 assert set(spans)==set(required)
 gpt=[]
 for h,s in spans.items():
  a=int(s['startTimeUnixNano']);b=int(s['endTimeUnixNano']);assert b>=a
  assert abs((b-a)/1e6-required[h]['stream_lifetime_ms'])<.001
  gpt.append(dict(start_ns=a,end_ns=b,request_span_sha256=h))
 gpt.sort(key=lambda s:s['start_ns'])
 for i,s in enumerate(gpt):s['name']=f'GPT 请求尝试 {i+1}'
 allrows=tools+gpt;origin=min(s['start_ns'] for s in allrows);finish=max(s['end_ns'] for s in allrows)
 lanes={};ordered=[]
 for lane,rows in [('gpt',gpt),('tools',tools)]:
  lanes[lane]=[]
  for row in sorted(rows,key=lambda s:(s['start_ns'],s['end_ns'])):
   result=dict(name=row['name'],start_s=(row['start_ns']-origin)/1e9,end_s=(row['end_ns']-origin)/1e9)
   lanes[lane].append(result);ordered.append(dict(result,lane=lane))
 merged=[]
 for r in lanes['gpt']:
  a,b=r['start_s'],r['end_s']
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 union=sum(b-a for a,b in merged);assert abs(union-e['cloud']['transport_union_s'])<.001
 # Local action sequence validates existing primitive accounting, without a cross-machine offset.
 actions=x['local']['actions'];assert len(actions)==len(e['trajectory'])
 for action,prev in zip(actions,e['trajectory']):
  assert action['name']==prev['action'];assert abs(action['detail']['window_s']-prev['seconds'])<.011
 detail_sums={lane:sum(r['end_s']-r['start_s'] for a in actions for r in a['detail']['lanes'][lane]) for lane in ['pi','sim','render']}
 e['timeline_v2']=dict(client=dict(window_s=(finish-origin)/1e9,lanes=lanes,events=sorted(ordered,key=lambda s:(s['start_s'],s['end_s'])),origin='首个非预热 GPT 请求或工具开始事件；Mac 墙钟',gpt_union_s=union),local=x['local'])
 audit.append(dict(id=e['id'],attempt=e['attempt'],session_sha256=m['session_sha256'],stream_sha256=x['stream_sha256'],client_events_sha256=sha(blob),telemetry_sha256=sha(raw),measurement_sha256=x['measurement_sha256'],tool_count=len(tools),request_attempt_count=len(gpt),gpt_union_s=union,local_action_count=len(actions),local_stage_sums=detail_sums,association='exact session + exact item ID hashes; request span hashes; local action sequence/duration checked, no cross-host clock alignment'))
(P/'report-data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(P/'evidence/timeline-v2-provenance.json').write_text(json.dumps(dict(episodes=audit,clock_policy='Mac client wall for GPT and MCP; separate Linux monotonic per-primitive detail. No wall offset estimated; no gap substituted for GPT. Service intervals use same-kernel primitive containment, not RPC IDs.'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:a[k] for k in ['id','tool_count','request_attempt_count','local_action_count','local_stage_sums']} for a in audit],indent=2))
