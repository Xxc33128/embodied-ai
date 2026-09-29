# -*- coding: utf-8 -*-
import json,base64,hashlib,statistics,re,math,collections
from pathlib import Path
import argparse,sys
parser=argparse.ArgumentParser(description='Build a standalone per-decision LIBERO report from recorded episodes')
parser.add_argument('campaign',type=Path)
parser.add_argument('--knowledge-md',type=Path,help='Authoritative Markdown companion; figures and local references are embedded')
parser.add_argument('--include', type=Path, action='append', default=[], help='Additional campaigns to include in the same report')
args=parser.parse_args()
root=args.campaign.resolve()
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_knowledge import find_companion, render_knowledge
knowledge_html, knowledge_manifest = render_knowledge(find_companion(args.knowledge_md))

def readlines(path):
 return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def uri(p,mime='image/png'):
 return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
def duration(span):return (int(span['end_ns'])-int(span['start_ns']))/1e9

def tracking_error(row):
 target=row['response'].get('target')
 if row['response']['mode']!='eef' or not target:return None
 def errors(t):
  q=t['eef_quaternion_xyzw'];q=[q[3],*q[:3]];v=target['quaternion_wxyz']
  dot=abs(sum(a*b for a,b in zip(q,v)))/(math.sqrt(sum(a*a for a in q))*math.sqrt(sum(a*a for a in v)))
  return math.sqrt(sum((a-b)**2 for a,b in zip(t['state'][:3],target['position']))),2*math.acos(min(1,dot))
 first,last=row['trajectory'][0],row['trajectory'][-1];p0,r0=errors(first);p1,r1=errors(last)
 return dict(position_initial_m=p0,position_final_m=p1,rotation_initial_rad=r0,rotation_final_rad=r1,target_gripper_closed=target['gripper_closed'],finger_initial=first['state'][6:],finger_final=last['state'][6:],executed=row['end']>row['start'])

def build_episode(ep):
 events=readlines(ep/'events.jsonl');obs={e['step']:e for e in events if e.get('event')=='observation'}
 spans=[e for e in events if e.get('event')=='span']
 props={e['request_id']:e for e in events if e.get('event')=='pi_proposal'}
 reqs={e['request_id']:e for e in events if e.get('event')=='pi_request'}
 rows=[x for x in readlines(ep/'decisions.jsonl') if x['event']=='decision']
 cfg=json.loads((ep/'config.json').read_text()); result=json.loads((ep/'result.json').read_text()) if (ep/'result.json').exists() else {'status':'running'}
 method=cfg['arguments']['method'];out=[]
 if method=='student':
  for i, (rid,p) in enumerate(props.items()):
   start=reqs[rid]['input_observation']['observation_index']
   actions=[o for o in obs.values() if o.get('pi_request_id')==rid and o.get('action') is not None]
   if not actions:continue
   arc=[s for s in spans if s['name'] in ('pi.input_archive','pi.preprocess') and s.get('request_id')==rid]
   begin=min(s['start_ns'] for s in arc) if arc else reqs[rid]['mono_ns']
   end=max(s['end_ns'] for s in spans if s['name']=='observation.save' and s.get('step')==actions[-1]['step'])
   rows.append(dict(decision_id='decision-%04d'%i,step_start=start,step_end=actions[-1]['step'],cycle_start_ns=str(begin),cycle_end_ns=str(end),accepted=True,pi_request_id=rid,response={'mode':'student','steps':len(actions),'reason':'π0.5 produces a continuous action chunk; the host executes the first 5 steps and replans.'},input=[{'type':'text','text':json.dumps({'instruction':reqs[rid]['prompt'],'state':reqs[rid]['input_state']})}],outcome={'native_done':actions[-1].get('native_done')}))
 for r in rows:
  a,b=r['step_start'],r['step_end'];lo,hi=int(r['cycle_start_ns']),int(r['cycle_end_ns'])
  subset=[s for s in spans if s['start_ns']>=lo and s['end_ns']<=hi]
  sums={name:sum(s['duration_s'] for s in subset if s['name']==name) for name in {s['name'] for s in subset}}
  rid=r.get('pi_request_id')
  if rid is None:
   rid=next((x.get('pi_request_id') for x in obs.values() if x.get('tool_call_id')==r['decision_id'] and x.get('pi_request_id')),None)
  if rid is None and method=='hybrid':
   rid=next((rid for rid,q in reqs.items() if q['input_observation']['observation_index']==a),None)
  p=props.get(rid);server=(p.get('server_timing') or p.get('meta')) if p else None
  piinfer=sum(server['timings'][n]['duration_ns']/1e9 for n in ('policy_infer','sync_before','sync_after') if n in server['timings']) if server else 0
  piround=sums.get('pi.request',0)
  if server: assert server['clock_domain']==p['clock_domain']
  if piround: assert piinfer<=piround+0.002
  piinfer=min(piinfer,piround) if piround else 0
  gpt=(int(r['turn_end_ns'])-int(r['request_start_ns']))/1e9 if r.get('model') else 0
  physics=sums.get('env.physics_step',0);render=sums.get('env.render_pixels',0);env=sums.get('env.step',0)
  parts={'π inference + sync':piinfer,'π file transport / queue / serialization':max(0,piround-piinfer),'GPT turn round trip':gpt,'Physics mj_step':physics,'Render + pixel readback':render,'Other environment work':max(0,env-physics-render),'Save observations':sums.get('observation.save',0),'Encode GPT input':sums.get('decision.encode_input',0)}
  total=(hi-lo)/1e9;parts['Other host work']=max(0,total-sum(parts.values()))
  assert abs(sum(parts.values())-total)<0.02,(ep.name,r['decision_id'],total,parts)
  trajectory=[{k:v for k,v in obs[i].items() if k in ('step','state','action','eef_quaternion_xyzw','robot_joint_positions','robot_joint_velocities','sim_qpos','sim_qvel','sim_time_s','native_done','tool_call_id','source')} for i in range(a,b+1)]
  sim_start=obs[a]['sim_time_s'];sim_end=obs[b]['sim_time_s']
  sim_duration=sim_end-sim_start
  assert abs(sim_duration-(b-a)*0.05)<1e-6
  inp=[x for x in r['input'] if x['type']=='text']
  imgs=[x['url'] for x in r['input'] if x['type']=='image'] or [uri(ep/obs[a]['frames'][name]) for name in ('agentview','wrist')]
  for j,name in enumerate(('agentview','wrist')):
   assert hashlib.sha256(base64.b64decode(imgs[j].split(',',1)[1])).digest()==hashlib.sha256((ep/obs[a]['frames'][name]).read_bytes()).digest()
  m=r.get('model')
  if m:
   wire=[x for x in events if x.get('event')=='rpc_receive' and x.get('message',{}).get('method')=='item/agentMessage/delta' and x['message']['params'].get('turnId')==r['turn_id'] and x['message']['params'].get('delta')]
   all_wire=wire
   if m.get('output_item_id'):wire=[x for x in wire if x['message']['params']['itemId']==m['output_item_id']]
   assert ''.join(x['message']['params']['delta'] for x in wire)==m['text']
   m=dict(m, online_consumer_ttft_s=m['ttft_client_s'], online_consumer_tpot_s=m['tpot_visible_estimate_s'])
   first,last=all_wire[0]['mono_ns'],wire[-1]['mono_ns']
   if m.get('stream_restarted'):m['abandoned_stream_chunks']=[{'received_ns':str(x['mono_ns']),'text':x['message']['params']['delta'],'item_id':x['message']['params']['itemId']} for x in all_wire if x['message']['params']['itemId']!=m['output_item_id']]
   m.update(ttft_client_s=(first-int(r['request_start_ns']))/1e9,tpot_visible_estimate_s=(last-first)/1e9/(m['visible_output_tokens']-1),first_chunk_ns=str(first),last_chunk_ns=str(last),timestamp_source='rpc_receive: transport reader timestamp before queue consumption',chunks=[{'received_ns':str(x['mono_ns']),'text':x['message']['params']['delta']} for x in wire])
  if m:
   m.update(request_wall_s=gpt,decode_client_s=(int(m["last_chunk_ns"])-int(m["first_chunk_ns"]))/1e9,server_inference_s=None)
  if m:
   total_output=m['usage']['outputTokens']
   m['tpot_all_output_folded_s']=m['decode_client_s']/(total_output-1) if total_output>1 else None
  publicmodel={k:v for k,v in m.items() if k not in ('raw_usage_updates',)} if m else None
  timeline=[{'stage':s['name'],'start_s':(s['start_ns']-lo)/1e9,'duration_s':s['duration_s']} for s in sorted(subset,key=lambda s:s['start_ns'])]
  if m:
   timeline.extend([{'stage':'GPT turn round trip','start_s':(int(r['request_start_ns'])-lo)/1e9,'duration_s':gpt},{'stage':'GPT first visible JSON chunk','start_s':(int(m['first_chunk_ns'])-lo)/1e9,'duration_s':0},{'stage':'GPT visible output stream','start_s':(int(m['first_chunk_ns'])-lo)/1e9,'duration_s':(int(m['last_chunk_ns'])-int(m['first_chunk_ns']))/1e9}])
  if server and piround:
   timeline.extend({'stage':'π server '+k,'start_s':(v['start_ns']-lo)/1e9,'duration_s':v['duration_ns']/1e9} for k,v in server['timings'].items() if v['start_ns']>=lo and v['end_ns']<=hi)
  pi_public=None
  if p:
   q=reqs[rid];archive=ep/q['input_archive'];acts=p['actions'];executed=[t['action'] for t in trajectory[1:]]
   candidate=next((json.loads(x['text']).get('student_proposal',{}).get('candidate_actions',[]) for x in inp if x.get('type')=='text'),[])
   matches=len(executed)<=len(acts) and all(abs(v-w)<1e-6 for row,ref in zip(executed,acts) for v,w in zip(row,ref))
   analysis={'predicted_steps':len(acts),'gpt_candidate_steps':len(candidate),'executed_steps':len(executed),'execution_matches_prefix':matches,'mode':r['response']['mode'],'candidate_translation_control_sum':[sum(row[i] for row in acts[:5]) for i in range(3)],'gripper_signs':['close' if row[6]>0 else 'open' if row[6]<0 else 'zero' for row in acts],'actual_eef_delta_m':[trajectory[-1]['state'][i]-trajectory[0]['state'][i] for i in range(3)]}
   pi_public={'request_id':rid,'actions':acts,'server':{k:v for k,v in server.items() if k not in ('host','pid')},'input':{k:v for k,v in q.items() if k in ('prompt','input_state','input_observation')},'input_images':[uri(ep/q['input_observation']['frames'][name]) for name in ('agentview','wrist')],'input_archive':q['input_archive'],'input_archive_data':uri(archive,'application/octet-stream'),'input_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'analysis':analysis}
  out.append({'simulation_start_s':sim_start,'simulation_end_s':sim_end,'simulation_duration_s':sim_duration,'environment_execution_wall_s':env,'timeline':sorted(timeline,key=lambda x:x['start_s']),'id':r['decision_id'],'start':a,'end':b,'accepted':r['accepted'],'input':inp,'images':imgs,'after_images':[uri(ep/obs[b]['frames'][name]) for name in ('agentview','wrist')],'response':r['response'],'output_raw':m['text'] if m else None,'model':publicmodel,'outcome':r['outcome'],'trajectory':trajectory,'pi':pi_public,'parts':parts,'total_s':total,'turn_id':r.get('turn_id'),'cycle_start_ns':r['cycle_start_ns'],'cycle_end_ns':r['cycle_end_ns']})
 for row in out:
  row['response_source']='gpt' if row['model'] else 'host_summary'
  row['tracking']=tracking_error(row)
 sums={k:sum(r['parts'][k] for r in out) for k in out[0]['parts']} if out else {}
 g=[r['model'] for r in out if r['model']]
 summary={'decisions':len(out),'accepted':sum(r['accepted'] for r in out),'actions':sum(r['end']-r['start'] for r in out),'mean_cycle_s':statistics.mean(r['total_s'] for r in out) if out else None,'mean_parts_s':{k:v/len(out) for k,v in sums.items()},'input_tokens':sum(m['usage']['inputTokens'] for m in g),'output_tokens':sum(m['usage']['outputTokens'] for m in g),'reasoning_tokens':sum(m['usage'].get('reasoningOutputTokens',0) for m in g),'mean_ttft_s':statistics.mean(m['ttft_client_s'] for m in g) if g else None,'mean_tpot_estimate_s':statistics.mean(m['tpot_visible_estimate_s'] for m in g if m['tpot_visible_estimate_s'] is not None) if g else None}
 summary.update(mean_simulation_duration_s=statistics.mean(x['simulation_duration_s'] for x in out),total_simulation_duration_s=sum(x['simulation_duration_s'] for x in out),mean_environment_execution_wall_s=statistics.mean(x['environment_execution_wall_s'] for x in out))
 summary.update(mean_gpt_request_s=statistics.mean(m['request_wall_s'] for m in g) if g else None,mean_gpt_decode_s=statistics.mean(m['decode_client_s'] for m in g) if g else None)
 summary.update(pi_calls=len(props), pi_text_tokens=sum((p.get('server_timing') or p['meta']).get('input_metrics',{}).get('text_tokens_valid',0) for p in props.values()), pi_action_scalars=70*len(props))
 cycles=sorted(x['total_s'] for x in out);pos=(len(cycles)-1)*.95;low=int(pos)
 summary.update(loop_total_s=sum(cycles),episode_wall_s=result.get('wall_seconds'),p95_cycle_s=cycles[low]+(cycles[min(low+1,len(cycles)-1)]-cycles[low])*(pos-low),max_cycle_s=max(cycles),gpt_calls=len(g),mode_counts=dict(collections.Counter(x['response']['mode'] for x in out)))
 if g:
  assert all(m['tpot_visible_estimate_s'] is not None for m in g)
  assert summary['input_tokens']==g[-1]['cumulative_usage']['inputTokens']
  assert summary['output_tokens']==g[-1]['cumulative_usage']['outputTokens']
 assert summary['actions']==max(obs) if result['status']!='running' else True
 protocol=json.loads((ep/'decision_protocol.json').read_text()) if (ep/'decision_protocol.json').exists() else {'protocol':'student_5_step_replan'}
 video=ep/'video/actions.mp4'
 return {'source_campaign':ep.parent.parent.name,'source_episode':ep.name,'method':method,'case':cfg['case'],'status':result['status'],'result':result.get('record',{}).get('success'),'summary':summary,'terminal_drawer_qpos':{'middle':out[-1]['trajectory'][-1]['sim_qpos'][38],'bottom':out[-1]['trajectory'][-1]['sim_qpos'][39]} if cfg['case']['benchmark']=='libero_goal_task' else None,'decisions':out,'protocol':protocol,'video':uri(video,'video/mp4') if video.exists() else None,'source_commit':cfg['source'].get('git_commit'),'deployment':cfg.get('deployment',{}).get('campaign')}

sources=root/'report_sources.json'
include=args.include or ([root/p for p in json.loads(sources.read_text())] if sources.exists() else [])
campaigns=[root]+[p.resolve() for p in include]
data={'campaign':root.name,'campaigns':[p.name for p in campaigns],'episodes':[build_episode(p) for c in campaigns for p in sorted((c/'episodes').glob('*')) if (p/'config.json').exists()]}
assert len({e['case']['episode_id'] for e in data['episodes']})==len(data['episodes']), 'duplicate episode identity'
notes=root/'report_notes.json'
if notes.exists():data['notes']=json.loads(notes.read_text())
(root/'report_data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
# Keep report_data.json expanded for offline consumers; deduplicate only the HTML payload.
assets={}
def pack_images(value):
 if isinstance(value,str) and value.startswith('data:image/'):
  key=hashlib.sha256(value.encode()).hexdigest();assets[key]=value;return 'asset:'+key
 if isinstance(value,list):return [pack_images(x) for x in value]
 if isinstance(value,dict):return {k:pack_images(v) for k,v in value.items()}
 return value
packed=pack_images(data);packed['image_assets']=assets
template=(Path(__file__).resolve().parent/'templates/libero_decision_report.html').read_text().replace('__KNOWLEDGE__',knowledge_html)
(root/'knowledge_manifest.json').write_text(json.dumps(knowledge_manifest,ensure_ascii=False,indent=2))
(root/'index.html').write_text(template.replace('__REPORT_DATA__',json.dumps(packed,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
(root/'report_summary.json').write_text(json.dumps({e['case']['episode_id']:{'status':e['status'],**e['summary']} for e in data['episodes']},indent=2))
print(json.dumps({'episodes':len(data['episodes']),'decisions':sum(len(e['decisions']) for e in data['episodes']),'html_bytes':(root/'index.html').stat().st_size}))
