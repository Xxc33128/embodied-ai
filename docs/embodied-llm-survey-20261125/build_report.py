#!/usr/bin/env python3
"""Build the offline report using only the Python standard library."""
import base64, csv, hashlib, html, json, re, shutil, struct
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ASSETS = OUT / 'assets'
DATA = OUT / 'data'
ESC = html.escape
COLORS = ['#205f71','#669b9b','#c79553','#6c719b','#b8c2cc']

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(pattern):
    paths = list(ROOT.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return paths[0]

SPECS = [
 ('S01','Astra首批五场：结果、耗时与逐步决策','weeks/2026_0914*/*Astra五次任务总结与用量.md'),
 ('S02','Astra入碗续测五场及匿名审核','weeks/2026_0921*/*Astra-bowl续测五次.md'),
 ('S03','具身智能测试项目与数据汇总 v1.3','weeks/2026_1012*/*具身智能测试项目与数据汇总.md'),
 ('S04','LIBERO三任务逐轮决策实录 v1.8','weeks/2026_0921*/*LIBERO10-task0-Ori-init0-逐轮决策实录.md'),
 ('S05','LIBERO init0 18格运行记录','weeks/2026_0921*/*init0-18格运行记录.md'),
 ('S06','RPent Astra／Sol 120集正式结果','weeks/2026_0928*/*双模型120集结果.md'),
 ('S07','RPent独立8集深测：机器可读结果','weeks/2026_1005*/deep-report/report-data.json'),
 ('S08','DeepSeek两场景计时审计说明','weeks/2026_1012*/ds-report/README.md'),
 ('S09','DreamLab四瓶入桶：批次与接口对照','weeks/2026_1026*/*DreamLab四瓶入桶Dream1与Astra对比分析总结.md'),
 ('S10','HomeBody仿真复现报告（含批次口径提示）','weeks/2026_1026*/*HomeBody复现报告.md'),
 ('S11','HomeBody Qwen换模型与关思考补测','weeks/2026_1109*/*HomeBody-Qwen3.8换脑报告.md'),
 ('S12','HALO架构与输入输出核查摘要','weeks/2026_1109*/*HALO架构总结摘要.md'),
 ('S13','HALO复现归因与首条合格轨迹','weeks/2026_1116*/*HALO复现归因与首条合格总结.md'),
 ('S14','HALO 11月23日结项与盲评进展','weeks/2026_1123*/*HALO结项与盲评进展.md'),
 ('S15','HomeBody VLM×SONIC扩展与验收边界','weeks/2026_1102*/*HomeBody-VLMxSONIC扩展与ARDY诊断执行总结.md'),
 ('S16','DreamLab评测与审计总结','weeks/2026_1019*/*DreamLab-Dream1评测与审计总结.md'),
 ('S17','RPent120集逐episode统计CSV','weeks/2026_1012*/2026-10-16-实验结果与改进分析/data/rpent120_episodes.csv'),
 ('S18','RPent Flash、goal与失败重跑','weeks/2026_0921*/*RPent扩展-FlashMode与libero_goal.md'),
 ('S19','COAP分析与RPent程序化执行实验建议','weeks/2026_1012*/*COAP论文分析与RPent实验建议.md'),
 ('S20','EmbodiedEvo分析与经验复用实验建议','weeks/2026_1012*/*EmbodiedEvo分析与RPent经验复用实验建议.md'),
 ('S21','Motus冻结VLM训练阶段报告','weeks/2026_1026*/*Motus架构与LIBERO-Long阶段训练报告.md'),
 ('S22','Motus续训至10000步及停止记录','weeks/2026_1026*/*Motus-LIBERO-Long续训执行记录.md'),
 ('S23','Hybrid门控与观测白名单源码','gpt-as-policy-repro/src/gap_repro/agent/gate.py'),
 ('S24','LIBERO动作接口及模式约束源码','gpt-as-policy-repro/src/gap_repro/libero/agent_contract.py'),
]
SOURCES = []
for sid, title, pat in SPECS:
    p=source(pat)
    evidence=OUT/'evidence'/f'{sid}{p.suffix}'
    shutil.copy2(p,evidence)
    SOURCES.append(dict(id=sid,title=title,path=str(p.relative_to(ROOT)),bundled=str(evidence.relative_to(OUT)),sha256=sha(p),size=p.stat().st_size))
SRC = {x['id']:ROOT/x['path'] for x in SOURCES}
RELATED = [
 dict(id='R01',title='SayCan: Grounding Language in Robotic Affordances',url='https://say-can.github.io/',accessed='2026-11-25',usage='技能规划与可执行性结合；仅方法背景'),
 dict(id='R02',title='RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control',url='https://robotics-transformer2.github.io/',accessed='2026-11-25',usage='动作token与联合训练；仅方法背景'),
 dict(id='R03',title='π0.5: a VLA with Open-World Generalization',url='https://www.pi.website/blog/pi05',accessed='2026-11-25',usage='异构数据联合训练；仅方法背景'),
 dict(id='R04',title='Huang et al. Inner Monologue: Embodied Reasoning through Planning with Language Models. CoRL 2022, PMLR 205 (2023)',url='https://proceedings.mlr.press/v205/huang23c.html',accessed='2026-11-25',usage='执行反馈与在线重规划；方法背景'),
 dict(id='R05',title='Galbot Team. Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies. arXiv:2609.38537v1 (2026)',url='https://arxiv.org/html/2609.38537v1',accessed='2026-11-25',usage='Direct／Hybrid上游方法与作者结果；不并入本地统计'),
 dict(id='R06',title='Zhang et al. Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents. arXiv:2607.08448v5 (2026)',url='https://arxiv.org/abs/2607.08448v5',accessed='2026-11-25',usage='RPent技术来源；与本地实现和评测条件分别说明'),
 dict(id='R07',title='Hu et al. Embodied Turing Machines: Stateful Code for Robot Recursive Self-Improvement. arXiv:2610.12369v1 (2026)',url='https://arxiv.org/html/2610.12369v1',accessed='2026-11-25',usage='COAP方法与作者结果；未本地复现实验成绩'),
 dict(id='R08',title='Li et al. EmbodiedEvo: Self-Evolving Coding Agent For Physical Intelligence. arXiv:2610.18867v2 (2026)',url='https://arxiv.org/html/2610.18867v2',accessed='2026-11-25',usage='经验冻结机制、表8与附录E；作者配对开发集结果'),
 dict(id='R09',title='Huh et al. HomeBody: A Humanoid That Explores, Remembers, and Acts on Its Own. 官方项目页 (2026)',url='https://tml.stanford.edu/homebody/',accessed='2026-11-25',usage='上游技能与空间记忆设计；本地成绩来自S10／S11'),
 dict(id='R10',title='Hu et al. HALO: Perceptive Humanoid Motion from Simulation. Apple官方项目页 (2026)',url='https://apple-aiml-research.github.io/ml-halo/',accessed='2026-11-25',usage='编排—生成—跟踪架构；页面仍标Paper Soon，不作为已发表论文列录'),
]

rows=list(csv.DictReader(SRC['S17'].open()))
deep=json.loads(SRC['S07'].read_text())
fields=['elapsed_s','manual_primitive_s','vla_primitive_s','tool_other_s','sdk_non_mcp_s','outside_sdk_s']
models={}
for m in ['gpt-6-astra','gpt-6-sol']:
    rr=[r for r in rows if r['model']==m]
    models[m]={'n':len(rr),'successes':sum(r['environment_success'].lower() in ('true','1','1.0') for r in rr),
        **{k:sum(float(r[k]) for r in rr) for k in fields+['request_count']}}
    models[m]['suites']={}
    for s in ['libero_spatial_task','libero_spatial_swap','libero_10_swap']:
        sr=[r for r in rr if r['suite']==s]
        models[m]['suites'][s]={'n':len(sr),'successes':sum(r['environment_success'].lower() in ('true','1','1.0') for r in sr),'elapsed_mean_s':sum(float(r['elapsed_s']) for r in sr)/len(sr)}
episodes=[{k:e[k] for k in ['id','model','task','seed','attempt','environment_success','request_count','vla_chunks']+fields} for e in deep['episodes']]
totals={k:sum(e[k] for e in episodes) for k in fields}
quant=json.loads(source('weeks/2026_1005*/deep-report/evidence/quantitative-conclusion.json').read_text())
derived={'rpent120':models,'rpent8':episodes,'rpent8_totals':totals,'rpent8_nonrender':quant,'source_date':'2026-11-23','report_date':'2026-11-25'}
dump(DATA/'derived-metrics.json',derived)
shutil.copy2(SRC['S17'],DATA/'rpent120-episodes.csv')
with (DATA/'rpent8-episodes.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)

# Exact vector diagrams and charts; no remote image or script dependencies.
def svg_begin(w,h,title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{ESC(title)}</title>',
      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#6d8190"/></marker></defs>',
      f'<rect width="{w}" height="{h}" rx="16" fill="#f5f8f8"/>',
      '<style>text{font-family:PingFang SC,Microsoft YaHei,Arial,sans-serif;fill:#203642;font-size:16px}.sub{fill:#637785;font-size:13px}.label{font-weight:600}</style>']
def txt(a,x,y,s,cl='',anchor='start',size=None):
    a.append(f'<text x="{x}" y="{y}" class="{cl}" text-anchor="{anchor}"'+(f' style="font-size:{size}px"' if size else '')+f'>{ESC(str(s))}</text>')
def box(a,x,y,w,h,label,subs=(),accent=False):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{"#e1eeee" if accent else "white"}" stroke="{"#7ca7ad" if accent else "#cedadc"}"/>')
    txt(a,x+w/2,y+28,label,'label','middle')
    for i,s in enumerate(subs):txt(a,x+w/2,y+51+i*19,s,'sub','middle')
def edge(a,points,label='',lx=None,ly=None,dash=False):
    a.append(f'<polyline points="{points}" fill="none" stroke="#6d8190" stroke-width="1.6" marker-end="url(#arrow)"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
    if label:txt(a,lx,ly,label,'sub','middle')
def save_svg(name,a):
    (ASSETS/name).write_text('\n'.join(a+['</svg>']))

a=svg_begin(940,360,'Direct：模型选择动作，控制器执行，观测反馈与事后评分分开')
for x,label,subs,accent in [(25,'观测与任务',['三视角 RGB','本体状态／轮内历史'],False),(265,'通用模型',['末端位姿＋夹爪','每次机器人决策'],True),(505,'适配与控制器',['Schema／边界校验','OSC_POSE／伺服'],False),(745,'Panda 仿真',['推进控制步','渲染新观测'],False)]:box(a,x,65,170,100,label,subs,accent)
for x in [195,435,675]:edge(a,f'{x},115 {x+70},115')
edge(a,'830,165 830,215 110,215 110,165','动作后观测返回；等待模型时物理暂停',470,204)
box(a,340,260,260,72,'独立结果核验',['自动入碗／录像阶段／私有轨迹'])
edge(a,'830,215 830,294 600,294','仅评测使用',735,282,True)
txt(a,25,30,'执行闭环', 'label');save_svg('architecture-direct.svg',a)

a=svg_begin(940,405,'Hybrid：π候选动作由模型审核，执行与目标评分分开')
box(a,25,60,190,100,'观测与任务',['双视角 RGB＋状态','当前指令／剩余预算'])
box(a,300,35,230,85,'π0.5 候选策略',['10×7 动作块；可见前缀'])
box(a,300,165,230,105,'通用模型审核',['student／edit／eef','放行、修改或接管'],True)
box(a,650,165,255,105,'校验与仿真执行',['最多执行前 5 步','保留实际动作来源'])
edge(a,'215,100 255,100 255,77 300,77');edge(a,'415,120 415,165','候选',452,145)
edge(a,'215,130 250,130 250,215 300,215');edge(a,'530,215 650,215','实际选定动作',590,199)
edge(a,'775,270 775,320 115,320 115,160','新观测／执行结果',460,310)
txt(a,470,370,'审核发生 ≠ 接管发生 ≠ 目标纠正成功','label','middle')
save_svg('architecture-hybrid.svg',a)

a=svg_begin(940,455,'RPent：高层模型编排感知、VLA和解析原语')
box(a,25,65,200,110,'任务上下文',['指令／冻结记忆','观测／工具返回'])
box(a,345,65,250,110,'LLM Planner',['选择工具、目标与参数','按反馈推进或恢复'],True)
edge(a,'225,120 345,120')
for x,label,subs in [(40,'感知工具',['SAM3／反投影']),(350,'π0.5 技能',['抓取等接触动作']),(660,'解析原语',['移动／夹爪／释放'])]:box(a,x,245,235,75,label,subs)
for x in [157,467,777]:edge(a,f'470,175 470,210 {x},210 {x},245')
box(a,345,365,250,65,'LIBERO 执行与目标检查',[])
for x in [157,467,777]:edge(a,f'{x},320 {x},340 470,340 470,365')
edge(a,'345,400 15,400 15,200 125,200 125,175',dash=True)
txt(a,176,385,'结果反馈到下一轮','sub','middle');save_svg('architecture-plan.svg',a)

a=svg_begin(940,460,'HomeBody：VLM直接选择技能，几何、AMO和PD分别执行')
box(a,30,35,260,90,'观测／任务／记忆',['RGB、深度、地图、历史'])
box(a,380,35,260,90,'VLM 决策',['当前技能＋目标参数'],True)
edge(a,'290,80 380,80')
box(a,380,170,260,75,'技能与结果契约',['navigate／pick／place／done'])
edge(a,'510,125 510,170')
for x,l,s in [(35,'行走',['几何导航 → AMO 50 Hz']),(350,'手臂',['几何／IK → 伺服 25 Hz']),(665,'手指',['闭合度 → 标定位置'])]:box(a,x,290,240,75,l,s)
for x in [155,470,785]:edge(a,f'510,245 510,265 {x},265 {x},290')
txt(a,470,407,'控制参考 250 Hz · PD 与 MuJoCo 积分 1000 Hz','label','middle')
txt(a,470,435,'以上为仿真配置周期；模型等待期间 heartbeat 维持物理与站立','sub','middle')
edge(a,'640,205 910,205 910,80 640,80','技能反馈',777,193,True)
save_svg('architecture-homebody.svg',a)

a=svg_begin(940,440,'HALO：在线编排生成跟踪与离线训练数据分开')
txt(a,30,30,'在线目标架构｜本地完整闭环尚未验收','label')
for x,l,s,ac in [(30,'任务编排器',['语言／状态 → 技能条件','通用模型'],True),(350,'运动生成器',['图像＋条件＋运动前缀','150×36 参考轨迹'],False),(670,'运动跟踪器',['参考＋深度＋本体状态','关节目标／50 Hz'],False)]:box(a,x,65,240,110,l,s,ac)
edge(a,'270,115 350,115');edge(a,'590,115 670,115')
edge(a,'790,175 790,210 470,210 470,175','已执行前缀／运动反馈',630,201)
edge(a,'910,120 925,120 925,240 150,240 150,175','任务状态反馈',310,232,True)
txt(a,30,285,'离线数据与训练｜已有部分分层证据','label')
box(a,30,315,240,85,'场景＋Kimodo轨迹',['WALK／TURN／SIT质量门禁'])
box(a,350,315,240,85,'渲染与数据集',['图像／parquet／轨迹'])
box(a,670,315,240,85,'训练与分层评估',['Generator／Tracker分别验收'])
edge(a,'270,355 350,355');edge(a,'590,355 670,355');save_svg('architecture-halo.svg',a)

a=svg_begin(940,405,'时间分解：Panda等待与LIBERO三路线每轮时间')
txt(a,30,32,'Panda Astra首批｜累计 1,755.02 s','label')
x=30
for v,c,l in [(1659.37,COLORS[0],'决策与工具等待 94.55%'),(95.65,COLORS[4],'其他 5.45%')]:
    w=v/1755.02*860;a.append(f'<rect x="{x}" y="53" width="{w}" height="30" fill="{c}"/>');x+=w
txt(a,30,111,'等待 1,659.37 s；其他 95.65 s。仿真 108.05 s 属于另一时钟。','sub')
txt(a,30,158,'LIBERO 两罐｜平均每决策轮墙钟，秒','label')
for i,(name,total,vals) in enumerate([('Student',4.203,[0,.606,2.774,.823]),('Direct',14.413,[10.819,0,2.709,.885]),('Hybrid',18.818,[14.374,.607,2.938,.899])]):
    y=188+i*50;txt(a,30,y+20,name);x=145
    for v,c in zip(vals,[COLORS[0],COLORS[1],COLORS[2],COLORS[4]]):
        w=v/20*670;a.append(f'<rect x="{x}" y="{y}" width="{w}" height="27" fill="{c}"/>');x+=w
    txt(a,x+10,y+20,f'{total:.3f}')
for i,(label,c) in enumerate(zip(['GPT往返','π推理与交换','渲染','其他'],[COLORS[0],COLORS[1],COLORS[2],COLORS[4]])):
    x=30+i*220;a.append(f'<rect x="{x}" y="367" width="12" height="12" fill="{c}"/>');txt(a,x+20,378,label,'sub')
save_svg('timing-direct-hybrid.svg',a)

a=svg_begin(940,430,'RPent：分套件成功数和平均episode耗时')
txt(a,30,32,'同套件对照｜每模型每套件 n=20','label')
txt(a,280,64,'原生成功数 / 20','sub','middle');txt(a,705,64,'episode均值 / s（含失败）','sub','middle')
for i,(s,n) in enumerate([('libero_spatial_task','Spatial Task'),('libero_spatial_swap','Spatial Swap'),('libero_10_swap','Long Swap')]):
    y=90+i*100;txt(a,30,y+24,n)
    for j,(m,c) in enumerate(zip(models,[COLORS[0],COLORS[2]])):
        d=models[m]['suites'][s]; yy=y+j*33
        a.append(f'<rect x="170" y="{yy}" width="{d["successes"]/20*230}" height="22" rx="3" fill="{c}"/>');txt(a,410,yy+17,f'{d["successes"]}/20','sub')
        a.append(f'<rect x="535" y="{yy}" width="{d["elapsed_mean_s"]/1800*290}" height="22" rx="3" fill="{c}"/>');txt(a,835,yy+17,f'{d["elapsed_mean_s"]:.1f}','sub')
txt(a,170,405,'深青：Astra / medium','sub');txt(a,535,405,'赭黄：Sol / medium','sub');save_svg('results-rpent.svg',a)

a=svg_begin(940,335,'HomeBody：思考开关的成功率与单步中位时延')
txt(a,30,32,'同一厨房 · 17任务×3重复 · 每组51场','label')
txt(a,270,65,'整任务成功率','sub','middle');txt(a,710,65,'单步决策中位数 / s','sub','middle')
for i,(l,n,t,c) in enumerate([('Qwen 思考＋降级',36,78,COLORS[0]),('DeepSeek 思考',32,43,COLORS[2]),('Qwen 关思考',4,4.2,COLORS[0]),('DeepSeek 关思考',6,2.2,COLORS[2])]):
    y=87+i*51;txt(a,24,y+19,l,'sub')
    a.append(f'<rect x="180" y="{y}" width="{n/51*290}" height="26" fill="{c}" rx="3"/>');txt(a,485,y+19,f'{n}/51 · {n/51*100:.1f}%','sub')
    a.append(f'<rect x="630" y="{y}" width="{t/80*230}" height="26" fill="{c}" rx="3"/>');txt(a,875,y+19,str(t),'sub')
txt(a,30,315,'首token快不等于完整决策快；关闭思考的速度收益伴随明显任务退化。','sub');save_svg('results-homebody.svg',a)

# RSI: offline development, freeze boundary, and two different online executors.
a=svg_begin(940,540,'RSI：开发反馈更新程序或经验，冻结后在独立留出场景验证')
txt(a,30,32,'开发阶段｜只使用开发场景', 'label')
box(a,30,65,240,90,'执行与收集反馈',['成功、失败、状态证据'])
box(a,350,65,240,90,'编程代理修订',['代码／Skills／Lessons'],True)
box(a,670,65,240,90,'开发验证与回归',['确认收益，检查旧任务退化'])
edge(a,'270,110 350,110');edge(a,'590,110 670,110')
edge(a,'790,155 790,185 150,185 150,155','未通过则继续修订',470,176,True)
box(a,350,220,240,70,'冻结版本与证据',['测试集不得回流'])
edge(a,'790,185 790,255 590,255')
box(a,60,335,350,95,'COAP：冻结程序执行',['根据显式状态分支','严格路线无在线VLM／VLA'])
box(a,530,335,350,95,'EmbodiedEvo：在线代理',['读取冻结Skills／Lessons','仍可修正当前回合程序'],True)
edge(a,'470,290 470,310 235,310 235,335');edge(a,'470,310 705,310 705,335')
txt(a,470,473,'独立留出初态 → 原生任务结果、退化、在线成本与开发摊销','label','middle')
txt(a,470,510,'本项目状态：已调研与设计；尚无正式RSI留出评测成绩','sub','middle')
save_svg('architecture-rsi.svg',a)

# Copy actual videos; no synthetic or upstream substitute clips.
MEDIA=[
 dict(id='direct-astra',label='V01 · Astra trial-03 · seed 2002',source='gpt6astra-repro/results/gpt-6-astra/trials/trial-03/video.mp4',poster='gpt6astra-repro/results/gpt-6-astra/trials/trial-03/initial-side.png',nframes=448,duration_s=22.4,steps=447,summary='自动入碗成功；13次决策。视频为20 fps仿真时钟，省略模型等待。',result='environment_success'),
 dict(id='direct-qwen',label='V02 · Qwen trial-03 · seed 2002',source='gpt6astra-repro/results/qwen3.8-flash/trials/trial-03/video.mp4',poster='gpt6astra-repro/results/qwen3.8-flash/initializations/trial-03/side.png',nframes=622,duration_s=31.1,steps=621,summary='自动失败；19次决策，归档阶段评分2。与V01初态相同；接入条件不同。',result='environment_failure'),
 dict(id='direct-boundary',label='V03 · Astra trial-01 · seed 2000',source='gpt6astra-repro/results/gpt-6-astra/trials/trial-01/video.mp4',poster='gpt6astra-repro/results/gpt-6-astra/trials/trial-01/initial-side.png',nframes=430,duration_s=21.5,steps=429,summary='自动入碗成功；10次决策。终帧在着底后反弹阶段，稳定放置尚未证实。',result='environment_success_stability_uncertain'),
]
from media_utils import mp4_metadata
for m in MEDIA:
    p=ROOT/m['source'];target=ASSETS/(m['id']+'.mp4');shutil.copy2(p,target)
    shutil.copy2(ROOT/m['poster'],ASSETS/(m['id']+'-poster.png'))
    m.update(sha256=sha(p),size=p.stat().st_size,file='assets/'+target.name,status='included',metadata=mp4_metadata(p))
    assert m['metadata']['sample_count']==m['nframes'],m
    assert abs(m['metadata']['duration_s']-m['duration_s'])<0.06,m
trace=[r for r in csv.DictReader((ROOT/'gpt6astra-repro/results/gpt-6-astra/decision-trace.csv').open()) if r['trial']=='3']
for r in trace:
    for k in ['call','start_step','end_step','control_steps']:r[k]=int(r[k])
    r['decision_wait_seconds']=float(r['decision_wait_seconds'])
dump(DATA/'direct-trial03-timeline.json',trace)
# Every video has its own timeline; never reuse Astra trial-03 times.
def astra_trace(trial):
    result=[r for r in csv.DictReader((ROOT/'gpt6astra-repro/results/gpt-6-astra/decision-trace.csv').open()) if int(r['trial'])==trial]
    for r in result:
        for k in ('call','start_step','end_step','control_steps'):r[k]=int(r[k])
        r['decision_wait_seconds']=float(r['decision_wait_seconds'])
    return result
qwen_wire=source('gpt6astra-repro/results/qwen3.8-flash/trials/trial-03/harness/wire/*/*/calls.jsonl')
qwen_calls=[json.loads(line) for line in qwen_wire.read_text().splitlines()]
qwen_result=ROOT/'gpt6astra-repro/results/qwen3.8-flash/trials/trial-03/result.json'
qwen_final=json.loads(qwen_result.read_text())
qwen_starts=[]
for call in qwen_calls:
    observations=[int(step) for message in call['request']['messages'] if isinstance(message.get('content'),list) for part in message['content'] if part.get('type')=='text' for step in re.findall(r"camera .*?\(step (\d+)\)",part.get('text',''))]
    qwen_starts.append(observations[-1])
qwen_trace=[]
for i,call in enumerate(qwen_calls):
    function=call['response']['choices'][0]['message']['tool_calls'][0]['function']
    arguments=json.loads(function['arguments'])
    end=qwen_starts[i+1] if i+1<len(qwen_starts) else qwen_final['physics_steps']
    qwen_trace.append(dict(call=i+1,start_step=qwen_starts[i],end_step=end,control_steps=end-qwen_starts[i],decision_wait_seconds=call['duration_s'],tool=function['name'],targets_json=json.dumps(arguments.get('targets',{})),recorded_note=arguments.get('note',arguments.get('reason',''))))
assert len(qwen_trace)==qwen_final['calls']==19
TIMELINES={'direct-astra':trace,'direct-qwen':qwen_trace,'direct-boundary':astra_trace(1)}
KEY_ROUNDS={'direct-astra':{6:'首次闭爪',7:'第一次试举',8:'发现抓空并修正',10:'再次提升',13:'释放入碗'},'direct-qwen':{7:'首次闭爪',9:'提升尝试',13:'开爪释放',14:'尝试重新抓取',19:'give_up终止记录'},'direct-boundary':{5:'开爪下降',6:'闭爪抓取',7:'提升验证',9:'对齐并下降',10:'释放与反弹边界'}}
for media in MEDIA:
    timeline=TIMELINES[media['id']]
    assert timeline[0]['start_step']==0 and timeline[-1]['end_step']==media['steps']
    assert all(r['end_step']>r['start_step'] for r in timeline)
    assert all(a['end_step']==b['start_step'] for a,b in zip(timeline,timeline[1:]))
    media['timeline_file']='data/'+media['id']+'-timeline.json'
    dump(OUT/media['timeline_file'],timeline)
# Bundle minimal derived records and provenance, not full request payloads.
dump(DATA/'video-timeline-provenance.json',{'fps':20,'sources':[{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in [ROOT/'gpt6astra-repro/results/gpt-6-astra/decision-trace.csv',qwen_wire,qwen_result]],'qwen_mapping':'Latest camera step in each request to next request step; last endpoint from result.physics_steps. Round 19 is give_up; its final one-step interval is termination recording, not a new grasp action.'})
shutil.copy2(ROOT/'gpt6astra-repro/results/summary-20260916/montage-gpt-trial03-sequence.png',ASSETS/'direct-trial03-sequence.png')
handoff=json.loads((DATA/'handoff-media.json').read_text())
MEDIA.extend(handoff['available'])
for m in handoff['available']:
    assert sha(OUT/m['file'])==m['sha256'],m['id']
    assert mp4_metadata(OUT/m['file'])==m['metadata'],m['id']
GAPS={}
dump(DATA/'media-manifest.json',{'available':MEDIA,'missing':GAPS,'unrecorded':handoff['unrecorded'],'timeline_source':'data/video-timeline-provenance.json; data/handoff-media-provenance.json','media_verification':'container metadata and file hashes; playback verification separately recorded in validation.json'})
extras=[source('weeks/2026_1005*/deep-report/evidence/quantitative-conclusion.json'),ROOT/'gpt6astra-repro/results/gpt-6-astra/decision-trace.csv',ROOT/'gpt6astra-repro/results/qwen3.8-flash/summary.csv']
dump(DATA/'source-manifest.json',{'sources':SOURCES,'related_work':RELATED,'supplemental':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in extras]})

def data_uri(p):
    ext=p.suffix.lower();mime={'.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.mp4':'video/mp4'}.get(ext,'application/octet-stream')
    return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()

def legacy_video_markup(mid):
    m=next(m for m in MEDIA if m['id']==mid)
    timeline=TIMELINES[mid]
    rounds=ESC(json.dumps([dict(call=r['call'],start=r['start_step']/20,end=r['end_step']/20) for r in timeline]),quote=True)
    def segment_button(row,label=None):
        call=row['call']; label=label or KEY_ROUNDS[mid].get(call,'终止记录' if row['tool']=='give_up' else '动作执行')
        start=row['start_step']/20;end=row['end_step']/20
        return f'<button type="button" class="seek" data-video="video-{mid}" data-start="{start:.2f}" data-end="{end:.2f}" data-label="第{call}轮 · {ESC(label,quote=True)}" aria-pressed="false" disabled>播放第{call}轮 · {ESC(label)}<span class="segment-time">{start:.2f}–{end:.2f}秒</span></button>'
    out=f'<figure class="video-card" id="case-{mid}"><div class="figure-heading">{ESC(m["label"])}</div><video id="video-{mid}" data-rounds="{rounds}" data-fallback="assets/{mid}.mp4" controls playsinline preload="metadata" poster="{data_uri(ASSETS/(mid+"-poster.png"))}"><source src="{data_uri(ASSETS/(mid+".mp4"))}" type="video/mp4"/>浏览器无法播放，请使用报告包内MP4。</video><figcaption>{ESC(m["summary"])} 封面为初始侧视图，播放为三视角。</figcaption>'
    out+='<p class="fine">按钮对应上方同一条完整视频中的轮次片段。点击后从该轮起点播放，到终点自动暂停；短片段可切换半速观察。展开表格可播放全部轮次。视频为20 fps仿真时钟，省略模型等待。</p>'
    if mid=='direct-qwen':out+='<p class="fine">第19轮为give_up终止决策，31.00–31.05秒仅是终止记录，不代表新的抓取动作；动作标注描述指令，不将模型自述当作成功判定。</p>'
    out+=f'<p id="segment-status-{mid}" class="segment-status" role="status" aria-live="polite">分段控制尚未启用。若按钮一直为灰色，请用浏览器直接打开本地report.html；仍可拖动视频进度条查看标注时间区间。</p><div class="video-controls">'
    out+=f'<button type="button" class="play-full" data-video="video-{mid}" disabled>播放完整视频 · 0.00–{m["duration_s"]:.2f}秒</button>'
    for call,label in KEY_ROUNDS[mid].items():out+=segment_button(timeline[call-1],label)
    out+=f'</div><label class="fine" for="speed-{mid}">播放速度 </label><select id="speed-{mid}" disabled><option value="1">1× 正常速度</option><option value="0.5">0.5× 半速观察</option></select>'
    out+=f'<details><summary>展开{len(timeline)}轮动作时间与真实等待（全部轮次可播放）</summary><div class="table-wrap"><table><thead><tr><th>轮次播放</th><th>工具</th><th>控制步</th><th>决策等待 / s</th></tr></thead><tbody>'
    for row in timeline:out+=f'<tr><td>{segment_button(row)}</td><td>{ESC(row["tool"])}</td><td>{row["start_step"]}–{row["end_step"]}</td><td>{row["decision_wait_seconds"]:.2f}</td></tr>'
    out+='</tbody></table></div><p class="fine">跳转位置由本视频逐轮控制步÷20计算；视频有初始帧，片段终点为相应动作后的帧。等待期间仿真不推进，因此等待不占视频长度。</p></details>'
    return out+'</figure>'

def video_markup(mid):
    if mid in TIMELINES:
        return legacy_video_markup(mid)
    m=next(m for m in MEDIA if m['id']==mid)
    timeline=json.loads((OUT/m['timeline_file']).read_text())
    rounds=ESC(json.dumps([dict(call=r['call'],label=r['label'],start=r['start_s'],end=r['end_s']) for r in timeline]),quote=True)
    def button(r):
        start=r['start_s'];end=r['end_s'];label=ESC(r['label'],quote=True)
        return f'<button type="button" class="seek" data-video="video-{mid}" data-start="{start:.9f}" data-end="{end:.9f}" data-label="{label}" aria-pressed="false" disabled>播放{label}<span class="segment-time">{start:.2f}–{end:.2f}秒</span></button>'
    out=f'<figure class="video-card" id="case-{mid}"><div class="figure-heading">{ESC(m["label"])}</div><video id="video-{mid}" data-rounds="{rounds}" data-fallback="{m["file"]}" controls playsinline preload="metadata"><source src="{data_uri(OUT/m["file"])}" type="video/mp4"/>浏览器无法播放，请使用报告包内MP4。</video><figcaption>{ESC(m["summary"])}</figcaption>'
    out+=f'<p class="fine">{ESC(m["playback_note"])} 每个按钮播放上方同一条完整视频的指定区间，到终点自动暂停；可选半速。</p>'
    out+=f'<p id="segment-status-{mid}" class="segment-status" role="status" aria-live="polite">分段控制尚未启用；可用原生进度条查看。</p><div class="video-controls"><button type="button" class="play-full" data-video="video-{mid}" disabled>播放完整视频 · 0.00–{m["duration_s"]:.2f}秒</button>'
    for row in timeline:
        if row['call'] in m['key_calls']:out+=button(row)
    out+=f'</div><label class="fine" for="speed-{mid}">播放速度 </label><select id="speed-{mid}" disabled><option value="1">1× 正常速度</option><option value="0.5">0.5× 半速观察</option></select>'
    if m['timeline_kind']!='navigation':
        kind={'round':'轮决策','phase':'个生成阶段','call':'次推进物理的调用'}[m['timeline_kind']]
        out+=f'<details><summary>展开全部{len(timeline)}{kind}与时间映射</summary><div class="table-wrap"><table><thead><tr><th>片段播放</th><th>执行类型</th><th>步／帧区间</th><th>决策等待 / s</th></tr></thead><tbody>'
        for row in timeline:
            wait=f'{row["decision_wait_seconds"]:.2f}' if 'decision_wait_seconds' in row else '不适用'
            out+=f'<tr><td>{button(row)}</td><td>{ESC(row["tool"])}</td><td>{row["start_step"]}–{row["end_step"]}</td><td>{wait}</td></tr>'
        out+='</tbody></table></div></details>'
    out+=f'<details><summary>查看录像来源与核验信息</summary><p class="fine">原文件SHA256：<code>{m["sha256"]}</code>；{m["nframes"]}帧，{m["fps"]:g} fps。时间线：<a href="{m["timeline_file"]}">{m["timeline_file"]}</a>；完整路径与来源指纹见 <a href="data/handoff-media-provenance.json">交接媒体来源清单</a>。</p></details>'
    return out+'</figure>'

def gap_markup(key):
    d=GAPS[key]
    return '<aside class="media-gap"><span class="status">媒体未取得</span><h4>'+ESC(d['title'])+'</h4><p>'+ESC(d['reason'])+'</p><ul>'+''.join('<li><code>'+ESC(x)+'</code></li>' for x in d['paths'])+'</ul><p class="fine">案例：'+ESC(d['case'])+' · <a href="#ref-'+d['source']+'">来源 '+d['source']+'</a></p></aside>'

labels=['非π动作原语','π动作原语','工具附加','SDK内非工具','episode内SDK外']
def timing_markup():
    first=episodes[0]
    out='<section class="interactive" aria-label="RPent逐集计时查看器"><div class="figure-heading">逐集查看：动作路径与时间构成</div><label for="episode-select">有效episode：</label><select id="episode-select">'
    for i,e in enumerate(episodes):out+=f'<option value="{i}">{ESC(e["id"])} · {"成功" if e["environment_success"] else "失败"}</option>'
    out+='</select><div id="episode-stats" class="episode-stats"></div><div id="episode-bars" aria-live="polite"></div><div id="episode-table"></div><p class="fine">五项按原始报告口径闭合；SDK残差不等于纯模型推理。独立模型加载不计入本图。</p><button type="button" id="download-timing">下载8集计时CSV</button></section>'
    return out

def references_markup():
    out='<p class="fine">S为本地归档／源码，R为原论文或官方项目页。点击正文引用可跳到对应资料；包内副本保留原始文本，未改写其历史结论。</p><ol class="references">'
    for s in SOURCES:
        out+=f'<li id="ref-{s["id"]}"><strong>[{s["id"]}] {ESC(s["title"])}</strong><p><a href="{ESC(s["bundled"],quote=True)}">包内来源副本</a> · <a href="../../{ESC(s["path"],quote=True)}">项目原件</a> · <code>{ESC(s["path"])}</code></p><small>SHA256 {s["sha256"]}</small></li>'
    for s in RELATED:out+=f'<li id="ref-{s["id"]}"><strong>[{s["id"]}] <a href="{s["url"]}" target="_blank" rel="noopener">{ESC(s["title"])}</a></strong><p>{ESC(s["usage"])}；访问日期 {s["accessed"]}。</p></li>'
    return out+'</ol><p class="fine">包内来源副本链接需保留evidence目录，项目原件链接需保持原目录结构；单文件HTML的正文、图表、11段视频和内嵌统计可离线阅读。详细统计与来源指纹亦包含在完整报告包中。</p>'

# Narrow Markdown renderer for this report's headings, paragraphs, tables and lists.
def inline(s):
    keep=[]
    def stash(x):keep.append(x);return f'\x00{len(keep)-1}\x00'
    s=re.sub(r'`([^`]+)`',lambda m:stash('<code>'+ESC(m[1])+'</code>'),s)
    s=ESC(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\[([SR]\d{2})\]',lambda m:f'<a class="cite" href="#ref-{m[1]}">[{m[1]}]</a>',s)
    s=re.sub('\x00(\d+)\x00',lambda m:keep[int(m[1])],s)
    return s

headings=[]
def render_md(md):
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('<!--'):
            match=re.match(r'<!-- (VIDEO|MEDIA_GAP|INTERACTIVE):([^ ]+) -->',line)
            if match:
                kind,key=match.groups();out.append(video_markup(key) if kind=='VIDEO' else gap_markup(key) if kind=='MEDIA_GAP' else timing_markup())
            elif line=='<!-- REFERENCES -->':out.append(references_markup())
            i+=1;continue
        h=re.match(r'^(#{1,6}) (.*)',line)
        if h:
            level=len(h[1]);title=h[2];hid='section-'+str(len(headings));headings.append((level,title,hid))
            out.append(f'<h{level} id="{hid}">{inline(title)}</h{level}>');i+=1;continue
        im=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if im:
            p=OUT/im[2];out.append(f'<figure class="diagram"><img src="{data_uri(p)}" alt="{ESC(im[1])}"/><figcaption>{ESC(im[1])}</figcaption></figure>');i+=1;continue
        if line.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+$',lines[i+1]):
            headers=[c.strip() for c in line.strip('|').split('|')];i+=2
            table='<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in headers)+'</tr></thead><tbody>'
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                assert len(cells)==len(headers),(i,len(headers),cells)
                table+='<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>';i+=1
            out.append(table+'</tbody></table></div>');continue
        if re.match(r'^\d+\. ',line):
            li=[]
            while i<len(lines) and re.match(r'^\d+\. ',lines[i].strip()):
                li.append('<li>'+inline(re.sub(r'^\d+\. ','',lines[i].strip()))+'</li>');i+=1
            out.append('<ol>'+''.join(li)+'</ol>');continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|<!--|\d+\. )',lines[i].strip()):para.append(lines[i].strip());i+=1
        out.append('<p>'+inline(' '.join(para))+'</p>')
    return '\n'.join(out)

body=render_md((OUT/'report.md').read_text())
nav='<nav aria-label="报告目录"><a class="brand" href="#section-0">EMBODIED AI<br><span>研究报告 · 2026.11</span></a>'
for level,title,hid in headings:
    if level==2 and not title.startswith('基于'):
        nav+=f'<a href="#{hid}">{ESC(title)}</a>'
nav+='<div class="nav-note">6条研究路线 · 9张图<br>11段内嵌视频<br>统计截止 2026-11-23</div></nav>'
css='''
:root{--ink:#203642;--muted:#617584;--teal:#205f71;--line:#d9e2e5;--paper:#fff;--bg:#edf1f2}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.85}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}nav{position:fixed;inset:0 auto 0 0;width:250px;padding:33px 22px;overflow-y:auto;background:#e5ebed;border-right:1px solid var(--line)}nav a{display:block;font-size:13px;line-height:1.55;padding:9px 10px;border-radius:5px;margin:4px 0}.brand{font-size:20px;letter-spacing:2px;font-weight:750;margin-bottom:28px;color:#163a48}.brand span{font-size:11px;font-weight:400;letter-spacing:1px}.nav-note{font-size:11px;color:var(--muted);margin:30px 10px;border-top:1px solid #c9d6db;padding-top:20px}main{margin:32px 36px 70px 282px;max-width:1120px;background:white;padding:56px 64px 70px;border:1px solid var(--line);box-shadow:0 8px 35px #20364208}.eyebrow{color:#62808b;font-size:12px;letter-spacing:3px;font-weight:650}h1{font-size:42px;letter-spacing:-1.4px;line-height:1.3;margin:15px 0 20px;max-width:850px}h2{font-size:27px;line-height:1.5;margin:65px 0 25px;padding-top:20px;border-top:2px solid #a8c5cb;letter-spacing:-.3px}h2:first-of-type{font-size:20px;color:#617584;border:0;margin:0 0 28px;padding:0;font-weight:500}h3{font-size:19px;line-height:1.5;margin:34px 0 15px}h4{font-size:16px;margin:8px 0}p{margin:14px 0;text-align:justify}strong{font-weight:650}.cite{font-size:11px;margin:0 3px;white-space:nowrap;border-bottom:1px solid #b9ccd1}.table-wrap{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}th,td{text-align:left;vertical-align:top;padding:12px 13px;border-bottom:1px solid #e3eaed;min-width:70px}th{background:#eaf1f2;color:#244754;font-weight:650}tr:nth-child(even) td{background:#f8fafb}tr:last-child td{border-bottom:0}td code{font-size:11px}figure{margin:28px 0}figure img{display:block;width:100%;height:auto}.diagram{background:#f5f8f8;border-radius:10px;overflow:hidden}.diagram img{padding:8px}figcaption{font-size:12px;line-height:1.7;color:#687b87;padding:12px 16px}.figure-heading{font-size:15px;line-height:1.5;font-weight:650;margin-bottom:14px}.video-card{border:1px solid var(--line);border-radius:10px;padding:18px;background:#f8fafb}video{display:block;width:100%;min-height:170px;max-height:430px;background:#18252c;border-radius:6px}.video-card figcaption{padding:12px 0}button,select{font:inherit;font-size:12px;line-height:1.4;border:1px solid #b7cbd1;border-radius:6px;padding:9px 13px;background:#fff;color:#205f71;cursor:pointer}button:hover{background:#e8f1f2}.video-controls{display:flex;gap:8px;flex-wrap:wrap}details{border-top:1px solid var(--line);margin-top:15px;padding-top:12px}summary{cursor:pointer;font-size:13px;color:#205f71}.fine{font-size:12px;line-height:1.7;color:var(--muted)}.media-gap{margin:25px 0;padding:22px;border:1px dashed #c8b89e;background:#fcfaf6;border-radius:8px;font-size:13px}.status{font-size:11px;color:#836841;border:1px solid #d7c8af;border-radius:20px;padding:3px 9px;display:inline-block}.media-gap ul{padding-left:20px}.media-gap code{overflow-wrap:anywhere}.interactive{padding:24px;border:1px solid #a9c5ca;background:#f3f8f8;border-radius:10px;margin:25px 0}.interactive select{max-width:100%;margin:8px 0 20px;font-size:13px}.interactive label{font-size:13px}.episode-stats{display:flex;gap:22px;flex-wrap:wrap;padding:4px 0 20px;font-size:13px}.episode-stats b{font-size:22px;display:block;color:#205f71}.barrow{display:grid;grid-template-columns:130px 1fr 95px;gap:12px;align-items:center;font-size:12px;margin:11px 0}.bartrack{height:23px;background:#e5ecee;border-radius:3px;overflow:hidden}.barfill{height:100%;border-radius:3px;min-width:1px}.barvalue{text-align:right;font-variant-numeric:tabular-nums}.references{padding-left:24px;font-size:13px}.references li{margin:20px 0;scroll-margin-top:25px}.references p{margin:5px 0}.references code,.references small{font-size:11px;overflow-wrap:anywhere;color:#617584}.toolbar{display:flex;justify-content:flex-end;gap:10px;margin-bottom:20px}.toolbar a,.toolbar button{font-size:12px}.cover-summary{padding:20px 24px;background:#eff5f5;border-left:4px solid #2a6371;margin:30px 0;font-size:14px}.footer{font-size:12px;color:#6e818a;border-top:1px solid var(--line);padding-top:25px;margin-top:45px}.progress{position:fixed;top:0;left:0;height:3px;background:#2a6371;width:0;z-index:9}code{font-family:ui-monospace,SFMono-Regular,monospace;font-size:.9em;overflow-wrap:anywhere}ol{padding-left:25px}li{margin:8px 0}.mobile-nav{display:none}a:focus-visible,button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #c79553;outline-offset:3px}
@media(min-width:1600px){main{margin-left:calc(250px + (100vw - 1450px)/2)}}
@media(max-width:1100px){nav{width:210px;padding:24px 12px}main{margin-left:232px;margin-right:20px;padding:35px 30px}h1{font-size:34px}}
@media(max-width:760px){nav{display:none}main{margin:0;padding:24px 18px;border:0;width:100%}.mobile-nav{display:block;margin-bottom:18px}.mobile-nav select{width:100%}h1{font-size:30px}h2{font-size:23px;margin-top:45px}h3{font-size:18px}body{font-size:15px}.barrow{grid-template-columns:95px 1fr 75px;gap:7px;font-size:11px}.interactive{padding:15px}table{font-size:12px;min-width:600px}th,td{padding:9px}video{min-height:100px}.references{padding-left:17px}.toolbar{flex-wrap:wrap}.progress{height:2px}}
@media print{body{background:white;font-size:10pt;line-height:1.65}nav,.toolbar,.progress,.mobile-nav,.video-controls,button,select{display:none!important}main{margin:0!important;padding:0!important;border:0;box-shadow:none;max-width:none}h1{font-size:27pt}h2{font-size:18pt;margin-top:30pt;break-before:auto}h3{font-size:13pt}table{font-size:8pt}th,td{padding:6px}figure,.media-gap,.interactive{break-inside:avoid}video{max-height:150px}a{color:inherit}p{orphans:3;widows:3}.table-wrap{overflow:visible}thead{display:table-header-group}.references small{font-size:7pt}img{max-height:250px;object-fit:contain} .barrow{grid-template-columns:130px 1fr 95px}}
'''
js='''
const episodes=REPORT_DATA.rpent8;
const labels=['非π动作原语','π动作原语','工具附加','SDK内非工具','episode内SDK外'];
const keys=['manual_primitive_s','vla_primitive_s','tool_other_s','sdk_non_mcp_s','outside_sdk_s'];
const colors=['#205f71','#669b9b','#c79553','#6c719b','#b8c2cc'];
function showEpisode(index){const e=episodes[index];document.getElementById('episode-stats').innerHTML=`<span>原生结果<b>${e.environment_success?'成功':'失败'}</b></span><span>episode / s<b>${e.elapsed_s.toFixed(1)}</b></span><span>计量响应<b>${e.request_count}</b></span><span>π动作块<b>${e.vla_chunks}</b></span>`;document.getElementById('episode-bars').innerHTML=keys.map((k,i)=>`<div class="barrow"><span>${labels[i]}</span><div class="bartrack"><div class="barfill" style="width:${100*e[k]/e.elapsed_s}%;background:${colors[i]}"></div></div><span class="barvalue">${e[k].toFixed(2)} s</span></div>`).join('');document.getElementById('episode-table').innerHTML=`<p class="fine">${e.id} · ${e.attempt} · 分项合计 ${keys.reduce((a,k)=>a+e[k],0).toFixed(2)} s</p>`;}
document.getElementById('episode-select').addEventListener('change',e=>showEpisode(Number(e.target.value)));showEpisode(0);
document.getElementById('download-timing').addEventListener('click',()=>{const ks=['id','attempt','environment_success','elapsed_s','request_count','vla_chunks',...keys];const csv=ks.join(',')+'\\n'+episodes.map(e=>ks.map(k=>e[k]).join(',')).join('\\n');const url=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='rpent8-timing.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.getElementById('print-report').addEventListener('click',()=>window.print());
document.getElementById('jump-section').addEventListener('change',e=>{if(e.target.value)document.getElementById(e.target.value).scrollIntoView({behavior:'smooth'});});
window.addEventListener('scroll',()=>{const h=document.documentElement;document.querySelector('.progress').style.width=100*h.scrollTop/(h.scrollHeight-h.clientHeight)+'%';},{passive:true});
'''
mobile='<div class="mobile-nav"><label for="jump-section" class="fine">跳转章节</label><select id="jump-section"><option value="">选择章节</option>'+''.join(f'<option value="{hid}">{ESC(t)}</option>' for lv,t,hid in headings if lv==2 and not t.startswith('基于'))+'</select></div>'
css+='\n.segment-time{display:block;font-size:11px;margin-top:5px}.segment-status{font-size:13px;background:#eaf1f2;padding:10px 14px;border-radius:6px}.video-controls{margin-bottom:14px}.seek[aria-pressed="true"]{background:#205f71;color:white;border-color:#205f71}button:disabled,select:disabled{opacity:.55;cursor:not-allowed}'
js=(OUT/'video-segments.js').read_text()+'\n'+js
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>通用大模型对具身智能的影响｜完整研究报告</title><style>'+css+'</style></head><body><div class="progress" aria-hidden="true"></div>'+nav+'<main><div class="toolbar"><button type="button" id="print-report">打印 / 保存PDF</button></div>'+mobile+'<div class="eyebrow">EMPIRICAL REVIEW / EMBODIED AI</div>'+body+'<div class="footer">embodied-ai · 2026-11-25 · 控制、规划与经验复用比较综述 · 未重跑实验</div></main><script>const REPORT_DATA='+json.dumps(derived,ensure_ascii=False).replace('</','<\/')+';</script><script>'+js+'</script></body></html>'
(OUT/'report.html').write_text(page)

# Make the Markdown deliverable useful without HTML marker expansion.
md=(OUT/'report.md').read_text()
for m in MEDIA:
    repl=f'[{m["label"]}](assets/{m["id"]}.mp4) — {m["summary"]}'
    md=md.replace(f'<!-- VIDEO:{m["id"]} -->',repl)
for k,d in GAPS.items():md=md.replace(f'<!-- MEDIA_GAP:{k} -->','> **视频待补：'+d['title']+'。** '+d['reason']+' 来源路径：'+'；'.join('`'+x+'`' for x in d['paths']))
md=md.replace('<!-- INTERACTIVE:rpent-timing -->','交互版逐集时间分解见同目录 `report.html`；原始数值见 `data/rpent8-episodes.csv`。')
refs='\n\n'.join(f'**[{s["id"]}] {s["title"]}**  \n仓库路径：`{s["path"]}`' for s in SOURCES)
refs+='\n\n'+'\n\n'.join(f'**[{s["id"]}] [{s["title"]}]({s["url"]})**。{s["usage"]}。访问日期：{s["accessed"]}。' for s in RELATED)
md=md.replace('<!-- REFERENCES -->',refs)
(OUT/'full-report.md').write_text(md)

checks={
 'rpent120_count':len(rows)==120,
 'rpent120_successes':models['gpt-6-astra']['successes']==50 and models['gpt-6-sol']['successes']==49,
 'rpent120_responses':sum(m['request_count'] for m in models.values())==5014,
 'rpent8_count_success':len(episodes)==8 and sum(e['environment_success'] for e in episodes)==3,
 'rpent8_buckets_close':all(abs(e['elapsed_s']-sum(e[k] for k in fields[1:]))<.01 for e in episodes),
 'rpent8_total':abs(totals['elapsed_s']-10965.3)<.01,
 'trial03_steps':trace[-1]['end_step']==447 and len(trace)==13,
 'trial03_wait':abs(sum(r['decision_wait_seconds'] for r in trace)-337.93)<.01,
 'videos_sample_counts':all(m['metadata']['sample_count']==m['nframes'] for m in MEDIA),
 'source_citations_resolved':set(re.findall(r'\[([SR]\d{2})\]',(OUT/'report.md').read_text())).issubset({x['id'] for x in SOURCES+RELATED}),
 'unexpanded_html_markers':not bool(re.search(r'<!-- (VIDEO|MEDIA_GAP|INTERACTIVE|REFERENCES)',page)),
 'offline_assets':len(re.findall(r'<(?:img|source)\b[^>]*src="(?!data:)',page))==0,
 'no_external_scripts':'<script src=' not in page,
}
assert all(checks.values()), checks
dump(DATA/'validation.json',{'checks':checks,'html_bytes':(OUT/'report.html').stat().st_size,'included_videos':len(MEDIA),'architecture_diagrams':6,'timing_and_result_figures':3,'browser_verification':'pending','scope':'No robotics rerun; source statistics recomputed and local media metadata inspected.'})
print(json.dumps({'html':str(OUT/'report.html'),'bytes':(OUT/'report.html').stat().st_size,'checks':checks},ensure_ascii=False))
