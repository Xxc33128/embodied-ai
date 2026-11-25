#!/usr/bin/env python3
"""Import selected original media and minimal audit records from the handoff.

No raw model requests, credentials, image payloads or training weights are bundled.
Run before build_report.py when the handoff changes; normal builds use the bundle.
"""
import csv, hashlib, io, json, shutil, tarfile, zipfile
from pathlib import Path
from media_utils import mp4_metadata
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HANDOFF = ROOT/'huawei实习_交接材料_20261125'
DATA = OUT/'data'
ASSETS = OUT/'assets'

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def sha_file(p):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def unique(pattern):
    paths = list(HANDOFF.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return paths[0]

provenance = []
media = []

def file_ref(path):
    ref = dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path))
    provenance.append(ref)
    return ref

archive = unique('*/原始数据/本机/GPT-as-Policy_实验批次与代码.tar.gz')
archive_ref = file_ref(archive)
prefixes = {
    'hybrid-pass': 'GPT-as-Policy-repro/data/decision_v2_20260922/episodes/libero_10-hybrid/',
    'hybrid-takeover': 'GPT-as-Policy-repro/data/decision_extra_20260922/episodes/libero_goal_task-task0-hybrid/',
}
selected = {}
wanted = {prefix+name for prefix in prefixes.values() for name in ('video/actions.mp4','decisions.jsonl','frame_index.json','result.json')}
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        if member.name in wanted:
            assert member.isfile()
            selected[member.name] = tar.extractfile(member).read()
assert set(selected) == wanted

def add(mid, label, raw, source, summary, note, timeline=None, key_calls=None, kind='round'):
    p = ASSETS/(mid+'.mp4')
    p.write_bytes(raw)
    meta = mp4_metadata(p)
    assert meta['avc1_present'] and meta['constant_frame_rate']
    m = dict(id=mid,label=label,file='assets/'+p.name,source=source,sha256=sha_bytes(raw),size=len(raw),
             metadata=meta,nframes=meta['sample_count'],duration_s=meta['duration_s'],fps=meta['fps'],
             status='included',summary=summary,playback_note=note,timeline_kind=kind,key_calls=key_calls or [])
    if timeline is None:
        # Navigation only. These are equal-duration quarters, NOT inferred task steps.
        duration=meta['duration_s']
        timeline=[dict(call=i+1,label=f'视频第{i+1}/4段（按时长）',start_s=duration*i/4,end_s=duration*(i+1)/4,
                       tool='按视频时长等分；无动作语义') for i in range(4)]
        m['timeline_kind']='navigation'
        m['key_calls']=[1,2,3,4]
    for r in timeline:
        assert 0 <= r['start_s'] < r['end_s'] <= meta['duration_s']+1e-7, (mid,r)
    assert all(abs(a['end_s']-b['start_s'])<1e-7 for a,b in zip(timeline,timeline[1:])),mid
    m['timeline_file']='data/'+mid+'-timeline.json'
    dump(OUT/m['timeline_file'],timeline)
    media.append(m)
    return m

for mid, prefix in prefixes.items():
    def read(name):
        raw=selected[prefix+name]
        provenance.append(dict(archive=archive_ref['path'],member=prefix+name,sha256=sha_bytes(raw)))
        return raw
    frames=json.loads(read('frame_index.json'))
    times={r['step']:r['action_video_time_s'] for r in frames}
    decisions=[json.loads(l) for l in read('decisions.jsonl').splitlines() if json.loads(l)['event']=='decision']
    result=json.loads(read('result.json'))
    timeline=[]
    for i,r in enumerate(decisions):
        mode=r['response']['mode']
        assert r['accepted']
        timeline.append(dict(call=i+1,label=f'第{i+1}轮 · '+('eef接管' if mode=='eef' else '放行π'),
            start_step=r['step_start'],end_step=r['step_end'],start_s=times[r['step_start']],end_s=times[r['step_end']],
            tool=mode,decision_wait_seconds=(int(r['turn_end_ns'])-int(r['request_start_ns']))/1e9))
    assert len(decisions)==(51 if mid=='hybrid-pass' else 60)
    assert sum(r['tool']=='eef' for r in timeline)==(0 if mid=='hybrid-pass' else 13)
    raw=read('video/actions.mp4')
    m=add(mid,'V04 · Hybrid · 两罐入篮 · init0' if mid=='hybrid-pass' else 'V05 · Hybrid · 底层抽屉目标 · init0',
        raw,dict(archive=archive_ref['path'],member=prefix+'video/actions.mp4'),
        '51轮全部放行π；原生任务成功，253控制步。' if mid=='hybrid-pass' else '60轮中13轮eef接管、47轮放行π；300步预算耗尽，原生任务失败。接管次数不等于目标纠正成功。',
        '原始双视角录像，20 fps。按钮时间直接取frame_index.json的action_video_time_s；模型等待不占视频长度。下表等待为请求发出至轮次返回的墙钟时间。',timeline,
        [1,13,26,39,51] if mid=='hybrid-pass' else [r['call'] for r in timeline if r['tool']=='eef'])
    assert m['nframes']==len(frames)==decisions[-1]['step_end']+1
    m['result']=result['status']

for model,vid in [('astra','V06'),('sol','V07')]:
    p=unique(f'RPent_deep*/data/rpent2/deep*/formal-results/gpt-6-{model}/libero_10_swap/t0_s1/attempt-001/episode.mp4')
    ref=file_ref(p)
    add('rpent-'+model,f'{vid} · RPent {model.title()} · t0 / seed 1 / attempt-001',p.read_bytes(),ref,
        '独立8集深测：原生成功，episode墙钟1049.1秒。' if model=='astra' else '独立8集深测：原生失败，episode墙钟2176.5秒。',
        '原始录像20 fps；视频秒数不等于episode墙钟。归档有工具记录，但未建立工具步至视频帧的可靠逐项映射。下方仅按视频时长四等分导航，不表示模型轮次或任务阶段。')

for mid,vid,name,summary in [
    ('homebody-corner','V08','cartons_to_corner_counter__fail.mp4','Qwen角柜任务失败案例：目标落点邻段偏差，原报告记录9步；仍沿用原批次判分。'),
    ('homebody-juice','V09','conditional_juice__fail.mp4','Qwen条件指令失败案例：条件成立后搬运掉落；仍沿用原批次判分。')]:
    p=HANDOFF/'homebody-repo/videos'/name
    add(mid,f'{vid} · HomeBody · {name.removesuffix(".mp4")}',p.read_bytes(),file_ref(p),summary,
        '交接原片为1 fps第三人称采样回放，逐秒换帧是原文件特征。没有配套逐轮时间线；按钮仅按视频时长四等分导航，不代表第几步。视频时长不用于重算控制周期或模型等待。')

halo=unique('HALO_qual*/data/halo/data/renders/qual02_mj/episodes/episode_10000/tpv.mp4')
hmeta=unique('HALO_qual*/data/halo/data/motion/qual02/episode_10000/meta.json')
file_ref(hmeta)
hd=json.loads(hmeta.read_text())
phases=[dict(call=i+1,label=phase,start_s=a/30,end_s=b/30,start_step=a,end_step=b,tool='生成参考阶段') for i,(phase,(a,b)) in enumerate(hd['phases'].items())]
add('halo-reference','V10 · HALO · qual02 / ep10000 · 生成参考',halo.read_bytes(),file_ref(halo),
    '198帧合格生成轨迹的第三人称渲染；用于查看WALK → TURN → SIT拼接，不是Tracker物理执行或承重坐稳证据。',
    '30 fps；按生成meta.json中的阶段帧区间映射。各区间左闭右开；SIT片段在整片结束时停止。',phases,[1,2,3],'phase')

archive=HANDOFF/'DreamLab-repo/bottle_experiment_report.zip'
ref=file_ref(archive)
with zipfile.ZipFile(archive) as z:
    def read(name):
        member='bottle_experiment_report/'+name
        raw=z.read(member)
        provenance.append(dict(archive=ref['path'],member=member,sha256=sha_bytes(raw)))
        return raw
    result=json.loads(read('result.json'))
    rows=list(csv.DictReader(io.StringIO(read('gpt_decisions.csv').decode())))
    actions=[r for r in rows if r['tool']=='robodojo_act']
    timeline=[];no_motion=[]
    for i,r in enumerate(actions):
        a=int(r['step_before']); b=int(actions[i+1]['step_before']) if i+1<len(actions) else result['step_id']
        if a==b:
            no_motion.append(int(r['call']));continue
        timeline.append(dict(call=int(r['call']),label=f'调用{r["call"]} · 执行动作',start_s=a/25,end_s=b/25,
            start_step=a,end_step=b,tool=r['tool'],decision_wait_seconds=float(r['decision_wall_seconds'])))
    m=add('dreamlab-direct','V11 · DreamLab · Astra×Codex · 四瓶入桶',read('rollout.mp4'),dict(archive=ref['path'],member='bottle_experiment_report/rollout.mp4'),
        '原生成功；653/700控制步，654帧。132次动作请求中131次推进物理，另有1次启动调用。',
        '25 fps，与control_dt=0.04秒一致。按钮沿用CSV调用编号，不重新编号；调用1未推进物理，不生成零长度片段。等待为decision_wall_seconds。',timeline,[2,34,67,100,132],'call')
    assert result['success'] and result['video_frames']==m['nframes']==654 and len(timeline)==131 and no_motion==[1]

r15=[]
for p in sorted(HANDOFF.glob('HALO_R15*/data/halo/diag/play_ab/*.json')):
    ref=file_ref(p)
    d=json.loads(p.read_text())
    records=d['records']
    r15.append(dict(source=ref,record_count=len(records),first_record=records[0],last_record=records[-1]))
dump(DATA/'halo-tracker-record-summary.json',dict(video_status='not_recorded',notes='交接README明确说明这八次配对回放未录制视频；保留首末记录供核对，不据此重判成功。',runs=r15))
file_ref(HANDOFF/'README.md')
dump(DATA/'handoff-media-provenance.json',dict(date='2026-11-25',sources=provenance,
    recording_note='HomeBody 1fps; RPent navigation quarters are not semantic steps; HALO reference is not physical tracking.'))
dump(DATA/'handoff-media.json',dict(available=media,unrecorded=[dict(id='halo-r15',status='not_recorded',source='huawei实习_交接材料_20261125/README.md',evidence='data/halo-tracker-record-summary.json')]))
print(json.dumps([dict(id=m['id'],frames=m['nframes'],seconds=m['duration_s'],fps=m['fps']) for m in media],ensure_ascii=False))
