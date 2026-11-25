import argparse,collections,hashlib,json,math,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
raw=Path(a.source).read_bytes();D=json.loads(raw)
def mean(x): return statistics.mean(x) if x else None
def pct(x,p):
 x=sorted(x);i=(len(x)-1)*p;l=int(i);return x[l]+(x[min(l+1,len(x)-1)]-x[l])*(i-l)
def dist(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
result={'source_sha256':hashlib.sha256(raw).hexdigest(),'scope':'nine episodes; descriptive, no causal or success-rate estimate','episodes':[]}
for e in D['episodes']:
 rs=e['decisions'];s=e['summary'];ms=[r['model'] for r in rs if r['model']];parts=s['mean_parts_s'];cy=s['mean_cycle_s'];render=parts['Render + pixel readback'];pi=parts['π inference + sync'];gpt=parts['GPT turn round trip']
 o={'episode_id':e['case']['episode_id'],'result':e['result'],'rounds':len(rs),'steps':s['actions'],'cycle_mean_s':cy,'cycle_p95_s':pct([r['total_s'] for r in rs],.95),'loop_total_s':sum(r['total_s'] for r in rs),'sim_s':s['total_simulation_duration_s'],'wall_per_sim_second':sum(r['total_s'] for r in rs)/s['total_simulation_duration_s'],'gpt_share':gpt/cy,'pi_share':pi/cy,'render_share':render/cy,'render_save_share':(render+parts['Save observations'])/cy,'pi_2x_loop_speedup_if_other_times_fixed':cy/(cy-pi/2),'render_2x_loop_speedup_if_other_times_fixed':cy/(cy-render/2),'modes':dict(collections.Counter(r['response'].get('mode') for r in rs))}
 if ms:
  spans=[(int(m['last_chunk_ns'])-int(m['first_chunk_ns']))/1e9 for m in ms]
  o.update(reasoning_zero=sum(m['usage']['reasoningOutputTokens']==0 for m in ms),reasoning_total=sum(m['usage']['reasoningOutputTokens'] for m in ms),visible_mean=mean([m['visible_output_tokens'] for m in ms]),visible_span_mean_s=mean(spans),ttft_mean_s=mean([m['ttft_client_s'] for m in ms]),ttft_p95_s=pct([m['ttft_client_s'] for m in ms],.95),completion_after_last_chunk_mean_s=gpt-mean([m['ttft_client_s'] for m in ms])-mean(spans),cache_fraction=sum(m['usage']['cachedInputTokens'] for m in ms)/sum(m['usage']['inputTokens'] for m in ms))
 if e['method']=='hybrid':
  passthrough=[r for r in rs if r['response']['mode']=='student'];tails=[]
  for r in passthrough:
   txt=''
   for c in r['model']['chunks']:
    txt+=c['text']
    if re.search(r'"steps"\s*:\s*\d+\s*,',txt):
     tails.append((int(r['model']['last_chunk_ns'])-int(c['received_ns']))/1e9);break
  o.update(pass_gpt_total_s=sum(r['parts']['GPT turn round trip'] for r in passthrough),pass_output_tail_after_mode_steps_mean_s=mean(tails),pass_prefix_samples=len(tails))
 if e['method']=='hybrid' and 'goal' in e['case']['benchmark']:
  take=[(i,r) for i,r in enumerate(rs) if r['response']['mode']=='eef'];first=take[0][0]
  cross=next(t['step'] for r in rs for t in r['trajectory'] if t['sim_qpos'][38]<-.14)
  o['drawer']={'first_takeover_round_zero_based':first,'first_takeover_step':rs[first]['start'],'loop_wait_before_first_takeover_s':sum(r['total_s'] for r in rs[:first]),'middle_qpos_before_first_takeover_m':rs[first]['trajectory'][0]['sim_qpos'][38],'middle_open_threshold_first_step':cross,'bottom_max_abs_qpos_m':max(abs(t['sim_qpos'][39]) for r in rs for t in r['trajectory']),'status_mode_counts':dict(collections.Counter(r['response']['assessment']['execution_status']+'/'+r['response']['mode'] for r in rs)),'takeover_rounds_zero_based':[i for i,r in take],'takeover_target_error_mm_mean':mean([dist(r['trajectory'][-1]['state'][:3],r['response']['target']['position'])*1000 for i,r in take]),'takeover_target_error_mm_median':statistics.median([dist(r['trajectory'][-1]['state'][:3],r['response']['target']['position'])*1000 for i,r in take]),'takeover_eef_displacement_mm_mean':mean([dist(r['trajectory'][-1]['state'][:3],r['trajectory'][0]['state'][:3])*1000 for i,r in take]),'takeovers_gripper_closed':sum(r['response']['target']['gripper_closed'] for i,r in take)}
 result['episodes'].append(o)
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
