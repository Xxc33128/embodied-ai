# -*- coding: utf-8 -*-
import json,hashlib,base64,subprocess,re,argparse,collections
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);args=ap.parse_args();r=args.campaign.resolve()
data=json.loads((r/'report_data.json').read_text());checks=[]
for e in data['episodes']:
 ep=r.parent/e['source_campaign']/'episodes'/e['source_episode'];assert e['status'] in ('valid_success','valid_failure')
 hashes=json.loads((ep/'checksums.json').read_text())
 for name,digest in hashes.items():
  assert hashlib.sha256((ep/name).read_bytes()).hexdigest()==digest,(e['method'],name)
 rows=[json.loads(x) for x in (ep/'actions.jsonl').read_text().splitlines()]
 assert len(rows)==e['summary']['actions']
 assert 'p95_cycle_s' in e['summary'], 'missing long-tail summary'
 covered=[]
 for d in e['decisions']:
  assert d['response_source']==('gpt' if d['model'] else 'host_summary')
  if d['response']['mode']=='eef':
   assert d['tracking']['position_final_m']>=0
   assert 0<=d['tracking']['rotation_final_rad']<=3.141592654
  assert len(d['images'])==2 and len(d['after_images'])==2
  assert len(d['trajectory'])==d['end']-d['start']+1
  for t in d['trajectory']:
   assert len(t['robot_joint_positions'])==7
   assert t['sim_qpos'] and t['sim_qvel']
  assert abs(sum(d['parts'].values())-d['total_s'])<0.02
  covered.extend(range(d['start']+1,d['end']+1))
  if d['model']:
   m=d['model'];assert 'decode_client_s' in m, 'missing decode measurement'
   assert abs(m['decode_client_s']-(int(m['last_chunk_ns'])-int(m['first_chunk_ns']))/1e9)<1e-9
   assert abs(m['request_wall_s']-d['parts']['GPT turn round trip'])<1e-9
   assert m['server_inference_s'] is None
   assert m['usage']['inputTokens']>0 and m['usage']['outputTokens']>0
   assert m['ttft_client_s']>0 and m['tpot_visible_estimate_s']>0
   assert ''.join(c['text'] for c in m['chunks'])==d['output_raw']
   assert json.loads(d['output_raw'])==d['response']
   assert m['timestamp_source'].startswith('rpc_receive')
  if d['pi']:
   p=d['pi'];archive=base64.b64decode(p['input_archive_data'].split(',',1)[1])
   assert archive==(ep/p['input_archive']).read_bytes()
   assert hashlib.sha256(archive).hexdigest()==p['input_archive_sha256']
   assert p['analysis']['executed_steps']==d['end']-d['start']
   assert p['analysis']['gpt_candidate_steps']==(5 if e['method']=='hybrid' else 0)
   if e['method']=='hybrid': assert json.loads(d['input'][0]['text'])['student_proposal']['candidate_actions']==p['actions'][:5]
   assert p['analysis']['actual_eef_delta_m']==[d['trajectory'][-1]['state'][i]-d['trajectory'][0]['state'][i] for i in range(3)]
   for name,img in zip(('agentview','wrist'),p['input_images']):
    assert base64.b64decode(img.split(',',1)[1])==(ep/p['input']['input_observation']['frames'][name]).read_bytes()
   if d['response']['mode']=='student': assert p['analysis']['execution_matches_prefix']
   im=d['pi']['server']['input_metrics'];assert im['text_tokens_valid']==sum(im['tokenized_prompt_mask'])>0
   assert len(d['pi']['actions'])==10 and all(len(x)==7 for x in d['pi']['actions'])
 assert covered==list(range(1,len(rows)+1))
 video=ep/'video/actions.mp4'
 assert hashlib.sha256(base64.b64decode(e['video'].split(',',1)[1])).digest()==hashlib.sha256(video.read_bytes()).digest()
 probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','stream=codec_name,pix_fmt,nb_frames,width,height','-of','json',str(video)]))
 stream=probe['streams'][0];assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p';assert int(stream['nb_frames'])==len(rows)+1
 checks.append({'episode_id':e['case']['episode_id'],'method':e['method'],'status':e['status'],'decisions':len(e['decisions']),'control_steps':len(rows),'sha256_files_verified':len(hashes),'video':stream})
groups=collections.defaultdict(list)
for e in data['episodes']:groups[(e['case']['benchmark'],e['case']['task_index'],e['case']['init_index'])].append(e['method'])
assert all(sorted(v)==['direct','hybrid','student'] for v in groups.values())
paired={}
for key in groups:
 entries=[e for e in data['episodes'] if (e['case']['benchmark'],e['case']['task_index'],e['case']['init_index'])==key]
 fingerprints=[json.loads((r.parent/e['source_campaign']/'episodes'/e['source_episode']/'initial_state.json').read_text()) for e in entries]
 paired[str(key)]={k:len({x[k] for x in fingerprints})==1 for k in ('state','full','init_state_sha256')}
 assert all(paired[str(key)].values())
h=(r/'index.html').read_text();assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="(?:https?:|file:)',h)
assert '人工判定' not in h
packed=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',h,re.S).group(1))
assets=packed['image_assets'];assert len(assets)==len(set(assets.values()))
def expand(x):
 if isinstance(x,str) and x.startswith('asset:'):return assets[x[6:]]
 if isinstance(x,list):return [expand(v) for v in x]
 if isinstance(x,dict):return {k:expand(v) for k,v in x.items()}
 return x
assert expand(packed['episodes'])==data['episodes'], 'packed HTML must preserve every evidence value'
assert (r/'index.html').stat().st_size<200_000_000, 'image deduplication regression'

ui=subprocess.check_output(['node',str(Path(__file__).resolve().parent/'check_decision_report.cjs'),str(r/'index.html')],text=True).strip()
from verify_report_knowledge import verify as verify_knowledge
knowledge_checks=verify_knowledge(r)
state={'knowledge':knowledge_checks,'complete':True,'paired_initial_states':paired,'checks':checks,'ui':json.loads(ui),'html_sha256':hashlib.sha256((r/'index.html').read_bytes()).hexdigest(),'html_bytes':(r/'index.html').stat().st_size}
(r/'verification.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
