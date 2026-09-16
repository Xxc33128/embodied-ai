import json,csv,math,statistics,hashlib
from pathlib import Path
from collections import Counter
import argparse
parser=argparse.ArgumentParser(description='Read-only post-hoc audit; never feed private fields to a policy.')
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args()
ROOT=options.workspace.resolve();OUT=options.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
MODELS={'glm-5.3-flash':'logs-glm-subagent20-20260915-120407','deepseek-v4.1-flash':'logs-deepseekv4.1-subagent20-20260915-1435','qwen3.8-flash':'logs-qwen3.8flash-subagent20-20260915-1545'}
def dist(a,b):return math.dist(a,b)
def dumpcsv(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
trials=[];calls=[];sources=[]
for model,batch in MODELS.items():
 mirror=ROOT/'embodied-ai/gpt6astra-repro/results'/model
 for s in csv.DictReader((mirror/'summary.csv').open()):
  i=int(s['trial']);p=ROOT/'gpt6astra-repro'/batch/f'trial-{i:02d}'
  def read(n):return json.loads((p/n).read_text())
  physics=[json.loads(l) for l in (p/'physics.jsonl').open()];by={r['step']:r for r in physics};result=read('result.json'); review=json.loads((mirror/f'trials/trial-{i:02d}/review.json').read_text())
  assert len(physics)==result['physics_steps']+1
  firstlift=next((r['step'] for r in physics if r['grasped'] and r['cube_bottom_z']>.82),None)
  firstgrasp=next((r['step'] for r in physics if r['grasped']),None)
  g=[r for r in physics if r['grasped']]
  tc=[]
  for k in range(1,result['calls']+1):
   start=read(f'call-{k:02d}-started.json');end=read(f'call-{k:02d}-ended.json');assert start['physics_steps']==end['physics_steps']
   st=start['physics_steps'];en=read(f'call-{k+1:02d}-started.json')['physics_steps'] if k<result['calls'] else result['physics_steps']
   b=by[st];a=by[en];f=read(f'incoming-{k:02d}.json')['choices'][0]['message']['tool_calls'][0]['function'];args=json.loads(f['arguments']);t=args.get('targets',{});ee=b['state']['eef_pos'];ae=a['state']['eef_pos'];tar=[t.get(key,ee[j]) for j,key in enumerate(['x','y','z'])]
   if k==result['calls'] and f['name']=='move_to' and result['termination_reasons']==['give_up']:
    en-=1  # Budget exhaustion adds a synthetic stop step, not a model action.
    a=by[en];ae=a['state']['eef_pos']
   cs=physics[st+1:en+1]
   row=dict(model=model,trial=i,call=k,name=f['name'],start_step=st,end_step=en,steps=en-st,targets=json.dumps(t),note=args.get('note',args.get('reason','')),hindsight=args.get('hindsight',''),eef_before=json.dumps(ee),eef_after=json.dumps(ae),gripper_before=b['state']['gripper'][0],gripper_after=a['state']['gripper'][0],command_xyz_distance_m=dist(ee,tar),actual_xyz_distance_m=dist(ee,ae),remaining_xyz_error_m=dist(tar,ae),cube_xy_before=json.dumps(b['cube_pos'][:2]),cube_xy_after=json.dumps(a['cube_pos'][:2]),cube_bottom_before=b['cube_bottom_z'],cube_bottom_after=a['cube_bottom_z'],grasped_before=b['grasped'],grasped_after=a['grasped'],any_grasp_during=any(x['grasped'] for x in cs),any_contact_during=any(x['any_gripper_contact'] for x in cs),bowl_distance_before=dist(b['cube_pos'][:2],b['bowl_xy']),bowl_distance_after=dist(a['cube_pos'][:2],a['bowl_xy']),wait_s=end['wait_seconds'])
   calls.append(row);tc.append(row)
  liftcall=next((r['call'] for r in tc if firstlift is not None and r['start_step']<firstlift<=r['end_step']),None)
  trials.append(dict(model=model,trial=i,seed=int(s['scene_seed']),stage=int(s['reviewed_stage']),auto_success=float(s['env_success']),calls=result['calls'],steps=result['physics_steps'],termination=';'.join(result['termination_reasons']),any_contact=any(r['any_gripper_contact'] for r in physics),any_grasp=bool(g),first_grasp_step=firstgrasp,first_lift_step=firstlift,first_lift_call=liftcall,max_bottom_z=max(r['cube_bottom_z'] for r in physics[1:]),held_min_bowl_distance=min((dist(r['cube_pos'][:2],r['bowl_xy']) for r in g),default=None),final_bowl_distance=dist(physics[-1]['cube_pos'][:2],physics[-1]['bowl_xy']),review_basis=review.get('basis','')))
  for n in ['physics.jsonl','result.json']:
   sources.append(dict(model=model,file=f'{batch}/trial-{i:02d}/{n}',sha256=hashlib.sha256((p/n).read_bytes()).hexdigest()))
dumpcsv('trial-audit.csv',trials);dumpcsv('call-audit.csv',calls)
(OUT/'source-hashes.json').write_text(json.dumps(sources,indent=2))
for m in MODELS:
 ts=[r for r in trials if r['model']==m];cs=[r for r in calls if r['model']==m];mov=[r for r in cs if r['name']=='move_to'];xyz=[r for r in mov if 'gripper' not in json.loads(r['targets']) and r['command_xyz_distance_m']>.02];gr=[r for r in mov if 'gripper' in json.loads(r['targets']) and r['command_xyz_distance_m']>.02]
 print(m,'STAGES',dict(Counter(r['stage'] for r in ts)),'END',dict(Counter(r['termination'] for r in ts)))
 print('contact/grasp/lift',sum(r['any_contact'] for r in ts),sum(r['any_grasp'] for r in ts),sum(r['first_lift_step'] is not None for r in ts))
 print('translations>2cm no gripper',len(xyz),'mediansteps',statistics.median(r['steps'] for r in xyz),'<=4',sum(r['steps']<=4 for r in xyz),'median residual',statistics.median(r['remaining_xyz_error_m'] for r in xyz))
 print('translations>2cm withgripper',len(gr),'mediansteps',statistics.median(r['steps'] for r in gr))
 print('LIFTED',[(r['trial'],r['stage'],r['first_lift_call']) for r in ts if r['first_lift_step'] is not None])
 print('DONE',[(r['trial'],r['stage']) for r in ts if r['termination']=='done'])
print('TOTAL',len(trials),len(calls))
