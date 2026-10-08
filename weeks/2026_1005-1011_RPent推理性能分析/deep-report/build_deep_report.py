# coding: utf-8
"""Build the independent eight-episode report from one sanitized JSON file."""
import argparse
import html
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
KEYS=['manual_primitive_s','vla_primitive_s','tool_other_s','sdk_non_mcp_s','outside_sdk_s']
LABELS=['非 π 动作','π 动作','工具附加','SDK 未覆盖','外壳']
STATUS={'unreported':'未接入','planned':'待运行','running':'运行中','completed':'已结束',
        'infra_failed':'基础设施异常','needs_review':'待审查'}
def esc(v):return html.escape(str(v))
def number(v,unit=''):
    if v is None:return '<span class="unknown">—</span>'
    if not unit and float(v).is_integer():return '<span class="num">'+f'{int(v):,}'+'</span>'
    return '<span class="num">'+f'{float(v):,.2f}'+esc(unit)+'</span>'
def text(v,fallback='新批次数据尚未接入，暂不下结论。'):
    return esc(v if v else fallback)
def table(head,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def bullets(items,empty='证据待接入。'):
    return '<ul>'+''.join('<li>'+esc(i)+'</li>' for i in items)+'</ul>' if items else '<p class="muted">'+esc(empty)+'</p>'
def excluded_rows(d):
    rows=[]
    for source in d.get('excluded_attempts',[]):
        row=dict(source)
        if 'elapsed_s' in row and 'episode_elapsed_s' in row:
            assert row['elapsed_s']==row['episode_elapsed_s'],'conflicting excluded elapsed fields'
        row.setdefault('elapsed_s',row.get('episode_elapsed_s'))
        row.setdefault('reason',row.get('exclusion_category'))
        row.setdefault('evidence',row.get('review_id'))
        rows.append(row)
    return rows

def resource_epoch_rows(d):
    rows=[]
    for source in d.get('resource_epochs',[]):
        row=dict(source)
        row.setdefault('id',row.get('epoch'))
        description=row.get('change') or row.get('role')
        if description and row.get('sam_physical_device') is not None:
            description='SAM '+str(row.get('device_model',''))+' / device '+str(row['sam_physical_device'])+' · '+description
        row.setdefault('description',description)
        row.setdefault('evidence',row.get('review_sha256') or row.get('sam_launch_script_sha256'))
        rows.append(row)
    return rows

def validate(d):
    assert d['schema_version']==1
    for item in d.get('evidence_files',[]):
        href=item.get('href','')
        assert isinstance(href,str) and (href=='report-data.json' or href.startswith('evidence/'))
        assert '..' not in Path(href).parts and ':' not in href and '\\' not in href and href.endswith('.json')
        assert all(isinstance(item.get(k),str) for k in ('label','description'))
    expected={(m,t,s) for m in ('gpt-6-astra','gpt-6-sol') for t in (0,9) for s in (0,1)}
    excluded=excluded_rows(d);epochs=resource_epoch_rows(d)
    assert isinstance(excluded,list) and isinstance(epochs,list)
    identities=set();epoch_ids=set()
    def optional_seconds(value):
        return value is None or (type(value) in (int,float) and math.isfinite(value) and value>=0)
    for row in excluded:
        assert (row.get('model'),row.get('task'),row.get('seed')) in expected
        assert all(isinstance(row.get(k),str) and row[k] for k in ('attempt','reason','evidence'))
        assert row['attempt'].startswith('attempt-') and row['attempt'][8:].isdigit()
        identity=(row['model'],row['task'],row['seed'],row['attempt'])
        assert identity not in identities,'duplicate excluded attempt'
        identities.add(identity)
        assert optional_seconds(row.get('elapsed_s')) and optional_seconds(row.get('load_s'))
        assert row.get('resource_epoch') is None or isinstance(row['resource_epoch'],str)
    for row in epochs:
        assert all(isinstance(row.get(k),str) and row[k] for k in ('id','description','evidence'))
        assert row['id'] not in epoch_ids,'duplicate resource epoch'
        epoch_ids.add(row['id'])
        assert row.get('attempt_count') is None or (type(row['attempt_count']) is int and row['attempt_count']>=0)
        assert optional_seconds(row.get('model_load_work_s'))

    measured=d.get('questions',{}).get('q1',{}).get('measured_summary',[])
    assert isinstance(measured,list) and len(measured)<=6,'Q1 measured summary supports at most six rows'
    for row in measured:
        assert isinstance(row,dict) and all(isinstance(row.get(k),str) for k in ('label','scope','evidence'))
        v=row.get('seconds')
        assert v is None or (type(v) in (int,float) and math.isfinite(v) and v>=0),'invalid measured summary seconds'
    basis=d.get('questions',{}).get('q1',{}).get('measured_summary_basis') or {}
    if basis.get('episode_denominator_s') is not None:
        value=basis['episode_denominator_s']
        assert type(value) in (int,float) and math.isfinite(value) and value>0,'invalid component denominator'
        assert type(basis.get('episode_count')) is int and 0<basis['episode_count']<=8
    eps=d['episodes']
    assert len(eps)==8 and {(e['model'],e['task'],e['seed']) for e in eps}==expected
    assert len({e['id'] for e in eps})==8
    for e in eps:
        assert e['suite']=='libero_10_swap' and e['reasoning_effort']=='medium'
        if e['status']=='completed' and e.get('measurement_invalid') is False:
            assert (e['model'],e['task'],e['seed'],e.get('attempt','attempt-001')) not in identities,'admitted attempt also excluded'
        assert e['status'] in STATUS
        assert e['environment_success'] is None or type(e['environment_success']) is bool
        for k in KEYS+['elapsed_s','load_s']:
            assert e[k] is None or (type(e[k]) in (float,int) and math.isfinite(e[k]) and e[k]>=0)
        top=e.get('top_timing') or {}
        if top.get('verified') is True:
            assert e['elapsed_s'] is not None
            assert all(type(top.get(k)) in (float,int) and math.isfinite(top[k]) and top[k]>=0 for k in ('sdk_interval_s','outside_sdk_s'))
            assert abs(top['sdk_interval_s']+top['outside_sdk_s']-e['elapsed_s'])<=.01,'top-level boundary does not reconcile'
        if e['elapsed_s'] is not None and all(e[k] is not None for k in KEYS):
            assert abs(sum(e[k] for k in KEYS)-e['elapsed_s'])<=.01,'time buckets do not reconcile'
    return d

def build(d):
    eps=d['episodes']
    admitted=[e for e in eps if e['status']=='completed' and e['measurement_invalid'] is False and e['elapsed_s'] is not None]
    finished=sum(e['status']=='completed' for e in eps)
    top=[e for e in admitted if (e.get('top_timing') or {}).get('verified') is True]
    top_values=[sum(e['top_timing'][k] for e in top) if top else None for k in ('sdk_interval_s','outside_sdk_s')]
    expanded=bool(admitted) and all(
        (e.get('five_bucket_evidence') or {}).get('classification')=='source_backed_step_association' and
        all(type(e.get(k)) in (int,float) and math.isfinite(e[k]) and e[k]>=0 for k in KEYS) and
        abs(sum(e[k] for k in KEYS)-e['elapsed_s'])<=.01 for e in admitted)
    bucket_labels=['非 π 动作','π 动作','工具附加（残差）','SDK 非工具（残差）','SDK 外开销（残差）']
    display_values=[sum(e[k] for e in admitted) for k in KEYS] if expanded else top_values
    display_labels=bucket_labels if expanded else ['SDK 报告时长','episode − SDK 残差']
    formula_terms=['动作非π','π动作','工具附加(残差)','SDK非工具(残差)','SDK外开销(残差)'] if expanded else ['SDK区间','episode−SDK残差']
    mode_note=('同步调用语义包含验证；非纯物理 / 纯 thinking。所有纳入样本均由 step_id 唯一关联原语与 MCP，并有同步 handler 源码支撑；后三项仍是残差。〔E1+E2〕' if expanded else
               '条件模式：五桶仍需原语串行、MCP 包含及数值闭合证据；代数闭合不能独立证明互斥，暂不展示五桶精确占比。〔E3〕')
    css='''
:root{--ink:#18313f;--muted:#627783;--line:#d7e1e7;--accent:#126580;--paper:#fbfcfd}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:72px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;overflow-wrap:anywhere}a{color:var(--accent)}nav{position:sticky;top:0;z-index:5;background:#fbfcfdf5;border-bottom:1px solid var(--line);display:flex;gap:25px;padding:13px max(20px,calc((100vw - 1060px)/2));overflow:auto;white-space:nowrap;backdrop-filter:blur(8px)}nav a{font-size:13px;text-decoration:none;color:#355566;transition:color .15s}nav a:hover{color:#0891b2}main{max-width:1140px;margin:auto;padding:0 40px 60px}header{padding:35px 0 14px}.eyebrow,.section-no{color:var(--accent);font-size:11px;letter-spacing:1.5px;font-weight:650}h1{font-size:clamp(28px,3.4vw,43px);line-height:1.25;margin:12px 0}h2{font-size:27px;line-height:1.4;margin:8px 0 18px}h3{font-size:19px;line-height:1.5;margin:22px 0 10px}p{margin:10px 0 16px}.muted{color:var(--muted);font-size:13px}.meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:13px}.formula-board{background:#173642;color:#fff;padding:25px 28px;margin:10px 0 18px}.formula-title{font-size:12px;color:#bbd3de}.formula{display:flex;flex-wrap:wrap;gap:0 8px;font-size:clamp(18px,2.3vw,27px);line-height:1.8;margin:12px 0}.formula .term{white-space:nowrap}.terms{display:grid;grid-template-columns:repeat(5,1fr);gap:18px;border-top:1px solid #ffffff35;padding-top:17px}.terms strong{font-size:25px;display:block;font-weight:550}.terms span{font-size:12px;color:#c4d7df}.formula-note{font-size:12px;color:#c4d7df;margin:16px 0 0}.num{white-space:nowrap}.unknown{color:#7b8f99;font-variant-numeric:tabular-nums}.formula-board .unknown{color:#a9c1cd}.statusline{border-left:3px solid var(--accent);padding:9px 15px;background:#eff5f8;font-size:14px;margin:20px 0}.mechanism{font-size:17px;padding:16px 0;border-bottom:1px solid var(--line)}section{padding:35px 0;border-bottom:1px solid var(--line)}#formula{padding-top:0}.answer{font-size:19px;max-width:880px;margin-bottom:22px}.boundary{font-size:13px;color:var(--muted);border-left:2px solid var(--line);padding-left:15px;margin:20px 0}.table-wrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;color:#375b6b;background:#edf3f6;white-space:nowrap}td,th{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}td{line-height:1.6}details{border-top:1px solid var(--line);padding:0;margin:18px 0 0}summary{cursor:pointer;font-size:14px;font-weight:550;color:#365c6e;padding:15px 0;transition:color .15s}summary:hover{color:#0d7a9d}details>div{padding:0 0 16px}.scene{padding:15px 0 20px;border-bottom:1px solid var(--line)}.scene:last-child{border-bottom:0}.label{font-size:12px;letter-spacing:.6px;color:var(--accent)}.figures{display:grid;grid-template-columns:1fr 1fr;gap:22px}.figures figure{margin:0}.figures img{width:100%;display:block}.figures figcaption{font-size:12px;color:var(--muted)}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:15px 0}select,button{font:inherit;font-size:13px;padding:8px 12px;background:#fff;border:1px solid #b9ccd6;color:var(--ink);border-radius:2px}button{cursor:pointer}pre{font:12px/1.7 ui-monospace,monospace;background:#eef3f6;padding:18px;overflow:auto;white-space:pre-wrap}footer{padding-top:25px;color:var(--muted);font-size:12px}ul{padding-left:21px}li{margin:5px 0}@media(max-width:700px){main{padding:0 19px 40px}header{padding-top:25px}nav{padding:12px 19px;gap:19px}h2{font-size:23px}.answer{font-size:17px}.formula-board{padding:21px 18px}.terms{grid-template-columns:repeat(3,1fr);gap:15px}.terms strong{font-size:22px}.formula{font-size:19px}section{padding:28px 0}.figures{grid-template-columns:1fr}th,td{padding:8px}th{white-space:normal}.meta{gap:8px 18px}#q3 th{white-space:normal}#q3 th,#q3 td{padding:5px;font-size:12px}.scene th{white-space:normal}.scene th,.scene td{padding:8px;font-size:13px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}@page{size:A4;margin:14mm}@media print{h2,h3,.section-no{break-after:avoid}.formula-board{background:white;color:#18313f;border:1px solid #18313f}.formula-board .unknown,.formula-title,.terms span,.formula-note{color:#526772}.terms{border-color:#cbd6dc}tr,.formula-board{break-inside:avoid}nav,.controls{display:none}main{max-width:none;padding:0}body{font-size:11px;background:white;color:#111}h1{font-size:28px}h2{font-size:21px}.formula-board{print-color-adjust:exact;-webkit-print-color-adjust:exact}section{padding:20px 0}.table-wrap{overflow:visible}table{font-size:10px}th,td{padding:6px}.answer{font-size:14px}details:not([open]){display:none}}
'''
    out=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>'+esc(d['title'])+'</title><style>'+css+'</style></head><body>']
    out.append('<nav aria-label="报告导航"><a href="#formula">总公式</a><a href="#q1">01 · 时间拆分</a><a href="#q2">02 · 轨迹与基线</a><a href="#q3">03 · 输入与上下文</a><a href="#episodes">8 集明细</a></nav><main><header><div class="eyebrow">RPENT / DEEP TIMING / INDEPENDENT BATCH</div><h1>'+esc(d['title'])+'</h1><div class="meta"><span>Long Swap · t0 / t9</span><span>Astra / Sol · medium · seed0 / 1</span><span>数据更新：'+text(d['updated_at'],'尚未接入')+'</span></div></header>')
    out.append('<section id="formula"><div class="formula-board"><div class="formula-title">总公式 · '+('源码与逐步关联支持的五项分解' if expanded else '顶层计时与条件分解')+'</div><div class="formula"><span class="term">T<sub>episode</sub> =</span>'+''.join('<span class="term">'+(' + ' if i else '')+'T<sub>'+esc(term)+'</sub></span>' for i,term in enumerate(formula_terms))+'</div><div class="terms">'+''.join('<div><strong>'+number(v,' s')+'</strong><span>'+esc(label)+'</span></div>' for v,label in zip(display_values,display_labels))+'</div><p class="formula-note">'+mode_note+' 数值是纳入样本的累计工作量，不是并行批次墙钟。模型加载另列。</p></div>')
    out.append('<div class="mechanism"><b>优化看两层：轮数 × 每轮成本。</b><br>每轮 ≈ 模型等待 + 工具执行；本地 π = 前处理 + prefix 合并 + denoise + 后处理。<p class="muted">工具动作内部含策略、控制、物理、渲染和 RPC；这些是嵌套组成，不再加到五桶。模型等待还混有网络、排队与编排，不能当成纯 thinking。prefix 合并项不继续臆分编码 / prefill。〔E3：结构模型〕</p></div>')
    out.append('<div class="statusline">'+esc(d.get('note',''))+' 已结束 '+str(finished)+'/8，计时验收通过 '+str(len(admitted))+'/8。</div><p class="answer">'+text(d['conclusion'],'接入新数据后，先确定最大耗时项，再确定应减少调用次数，还是降低单次执行成本。当前不填入旧 120 集的占比。')+'</p>')
    out.append(bullets(d['optimization'],'优化排序将在新批次测量闭合后给出。'))
    out.append('<details><summary>顶层计时边界与五桶对账</summary><div><p>T_episode = SDK 报告时长 + episode−SDK 残差。episode 在 init_runtime 后起算，toolkit.close / daemons.stop 后取差，不含后续 transcript 写盘 / finalize_run。SDK 在 MCP 启动后起算，MCP stop 后取差；外 SDK 项由两项 round 到一位小数的时长相减，不是独立实测阶段，也不是全进程墙钟。</p>'+table(['记账项','测量边界'],[['非 π 动作', 'move_to、move_pose、夹爪等原语；不是人工接管。'],['π 动作','π 原语内部，包括模型、RPC、环境执行。'],['工具附加','已完成工具区间并集 − 动作原语时长。'],['SDK 非工具（残差）','SDK 总时长 − 已完成工具区间并集；位于 SDK 区间内部。'],['SDK 外开销（残差）','episode 总时长 − SDK 总时长；属于 episode 内、SDK 区间之外。']])+'<p class="muted">细阶段 span 为父项的组成，不能与父项再次相加。使用同一时钟域的区间并集/独占时间；不同机器单调时钟不能直接相减。算术对账只是必要条件；还须验证原语串行、MCP 包含和区间不重复。未证明这些条件时，五桶只作条件估计。</p></div></details></section>')
    batch_timing=d.get('batch_timing') or {}
    out.insert(-1, '<p><b>批次时间另计。</b> 两条 lane 并行，不能把 8 集 episode 相加当作批次墙钟。</p>'+table(['闸门起止墙钟（已知范围）','已观测模型加载工作量','可观测配对间隔'],[[number(batch_timing.get(k),' s') for k in ('gate_envelope_wall_s','model_load_work_s','observed_inter_pair_gap_s')]])+'<p class="muted">闸门墙钟按首个 before_pair 至最终 after_pair 定义，批次未结束时仍未知；不含此前启动及此后收尾。配对间隔混有额度检查、审查与重启，不等于纯审查暂停；独立审查时长和 lane 关键等待未证实则不报。模型加载覆盖所有实际attempt的已闭合init span；未闭合部分不估算，测量未齐时仅为已观测下限。它是累计工作量，可能并行，三列不能相加。'+('全进程墙钟（独立边界）：'+number(batch_timing.get('wall_s'),' s')+'。' if batch_timing.get('wall_s') is not None else '')+'</p>')
    excluded=excluded_rows(d);epochs=resource_epoch_rows(d)
    if excluded or epochs:
        summary='<p><b>分析分母与复现成本分开。</b> 上方公式仅含 '+str(len(admitted))+'/8 集计时验收通过的样本。'
        if excluded:
            known=[r['elapsed_s'] for r in excluded if r.get('elapsed_s') is not None]
            summary+='另有 '+str(len(excluded))+' 次 attempt 排除；'
            summary+=('已观测 episode 消耗 '+number(sum(known),' s')+('（完整覆盖这些排除项）' if len(known)==len(excluded) else '（仍有缺测，仅为下限）')+'。' if known else 'episode 消耗尚未接入。')
        summary+='排除项不进入结论分母，其执行成本仍属于复现成本；累计秒数不能当作并行批次墙钟。</p>'
        detail='<details><summary>排除 attempt 与资源变更成本</summary><div>'
        if excluded:
            detail+=table(['排除的 attempt','原因 / 资源 epoch','episode 秒数','冷加载工作量','证据'],[[esc(r['model'].replace('gpt-6-',''))+' · t'+str(r['task'])+' / s'+str(r['seed'])+'<br>'+esc(r['attempt']),esc(r['reason'])+'<br>'+esc(r.get('resource_epoch') or 'epoch 未接入'),number(r.get('elapsed_s'),' s'),number(r.get('load_s'),' s'),esc(r['evidence'])] for r in excluded])
        if epochs:
            costs=any(r.get('attempt_count') is not None or r.get('model_load_work_s') is not None for r in epochs)
            detail+=table(['资源 epoch','变更 / 适用范围']+(['实际 attempt 数','冷加载工作量'] if costs else [])+['证据'],[[esc(r['id']),esc(r['description'])]+([number(r.get('attempt_count')),number(r.get('model_load_work_s'),' s')] if costs else [])+[esc(r['evidence'])] for r in epochs])
        detail+='<p class="muted">episode 与冷加载遵循各自计时边界；冷加载已包含于上方全 attempt 加载工作量，不能重复加总。资源 epoch 用于说明配置条件与成本覆盖，不构成优化收益的因果证明。未知值保持 —。</p></div></details>'
        out.insert(-1,summary+detail)
    q=d['questions']['q1']
    out.append('<section id="q1"><div class="section-no">问题 01 / 推理与仿真耗时</div><h2>Prefill、thinking 和本地执行，分别花了多久？</h2><p class="answer">'+text(q['conclusion'])+'</p>')
    measured=q.get('measured_summary',[])
    if measured:
        basis=q.get('measured_summary_basis') or {}
        denominator=basis.get('episode_denominator_s')
        note=('同一组 '+str(basis.get('episode_count'))+' 集 episode 累计 '+number(denominator,' s')+' 为占比分母。' if denominator is not None else '')
        rows=[[esc(r['label']),number(r.get('seconds'),' s')]+([number(r['seconds']/denominator*100,'%') if r.get('seconds') is not None and denominator>0 else number(None)] if denominator is not None else []) for r in measured]
        out.append('<h3>实测重点 · 嵌套项，不求和</h3>'+table(['组件 / 原语窗','累计秒数']+(['占 episode'] if denominator is not None else []),rows)+'<p class="muted">'+note+'组件范围可能互相嵌套；prefix / denoise 已包含于 π handler，不与父项重复相加，也不再加到 episode 五桶。</p><details><summary>组件计时边界与证据</summary><div>'+table(['组件','计时范围','来源'],[[esc(r['label']),esc(r['scope']),esc(r['evidence'])] for r in measured])+'</div></details>')
    out.append(table(['阶段','本批报告口径','数据来源'],[['云端 prefill / 完整 thinking','仅在真实服务端遥测提供时填秒数；否则保持未知。','服务端阶段记录'],['客户端请求','请求流墙钟与传输区间并集；包含排队、网络、重连与可见输出，不等于纯服务端推理。TTFT 仅在请求身份可关联时展示。','请求事件与用量'],['π0.5 / SAM3','前处理、前缀/图像编码、去噪/提示解码、后处理及 RPC。','按调用闭合的组件 span'],['仿真执行','控制器、physics、观测/渲染、传输；按层级排除重复。','同域父子 span']]))
    out.append(('<details><summary>本问证据</summary><div>'+bullets(q['evidence'])+'</div></details>')+'<p class="boundary"><b>边界：</b>'+esc(q['boundary'])+'</p><details><summary>逐集细阶段计时</summary><div>')
    stage_rows=[]
    for e in eps:
        for st in e['stages']:
            stage_rows.append([esc(e['id']),esc(st['name']),number(st.get('seconds'),' s'),esc(st.get('scope','')),esc(st.get('status','')),esc(st.get('evidence',''))])
    out.append(table(['样本','阶段','耗时','inclusive / exclusive','状态','证据'],stage_rows) if stage_rows else '<p class="muted">尚未接入组件计时。空白不代表耗时为零。</p>')
    out.append('</div></details></section>')
    q=d['questions']['q2']
    out.append('<section id="q2"><div class="section-no">问题 02 / 重复尝试与性能基线</div><h2>哪些轮次没有推进任务？理想路径能省多少？</h2><p class="answer">'+text(q['conclusion'])+'</p>')
    for scene in d['scenes']:
        selected=[e for e in eps if e['task']==scene['task']]
        out.append('<article class="scene"><div class="label">LONG SWAP / t'+str(scene['task'])+' · 4 集配对</div><h3>'+esc(scene['name'])+'</h3><p>'+esc(scene['description'])+'</p><p><b>结论：</b>'+text(scene['conclusion'])+'</p>')
        out.append(table(['模型 / seed','状态 / 结果','总耗时','计量响应','π chunks'],[[esc(e['model'].replace('gpt-6-',''))+' / '+str(e['seed']),esc(STATUS[e['status']])+' / '+('未定' if e['environment_success'] is None else ('成功' if e['environment_success'] else '失败')),number(e['elapsed_s'],' s'),number(e['request_count']),number(e['vla_chunks'])] for e in selected]))
        out.append('<p><b>重复尝试证据：</b></p>'+bullets(scene['trajectory_findings'],'等待完整轨迹；不预判本批会重复历史行为。'))
        out.append('<p><b>理想闭环节点：</b>'+number(scene['ideal_nodes'])+'　<b>性能基线：</b>'+number(scene['baseline_s'],' s')+' '+text(scene['baseline_kind'],'尚未建立')+'</p>'+bullets(scene['ideal_assumptions'],'节点数与收益估算必须附成功、感知与局部控制假设。'))
        out.append('</article>')
    out.append(('<details><summary>本问证据</summary><div>'+bullets(q['evidence'])+'</div></details>')+'<p class="boundary"><b>边界：</b>'+esc(q['boundary'])+' 局部工具 success=false 不等于整次动作毫无进展；没有成功轨迹时不填写“最佳成功秒数”。</p></section>')
    q=d['questions']['q3']
    out.append('<section id="q3"><div class="section-no">问题 03 / 输入输出与上下文</div><h2>π0.5、Astra、SAM3 收发了哪些字段？</h2><p class="answer">'+text(q['conclusion'])+'</p>')
    out.append('<p><b>上下文看单次 input；累计 input 是跨请求重复计量的工作量。</b> cached ⊆ input，reasoning ⊆ output，两组子集均不可再与所属总量相加。〔E2：用量字段口径〕</p>')
    token_rows=[]
    for e in eps:
        t=(e.get('cloud') or {}).get('token_usage')
        if t:
            token_rows.append([esc(e['model'].replace('gpt-6-',''))+' · t'+str(e['task'])+'/s'+str(e['seed']),number(t.get('response_count')),number(t.get('input_min'))+' / '+number(t.get('input_median'))+' / '+number(t.get('input_max')),number(t.get('input_sum'))+'<br><span class="muted">其中缓存 '+number(t.get('cached_input_sum'))+'</span>',number(t.get('output_sum'))+'<br><span class="muted">其中 reasoning '+number(t.get('reasoning_sum'))+'</span>'])
    if token_rows:out.append(table(['样本','计量响应','单次 input：最小 / 中位 / 最大','累计 input（含重复）','累计 output'],token_rows))
    out.append('<p class="muted">π 的有效 prompt token 数用 nonpadding_count；张量 padding 长度用 shape，两者分列，不把填充位置算成有效文本。Astra / Sol 的 input 用量可能包括图片、工具和历史，不能仅用可见 prompt 字符数替代。逐请求证据收在样本折叠详情。</p>')
    if d['io']:
        out.append(table(['模块 / 接口','实际输入','实际输出','上下文 / 长度','证据'],[[esc(i['module']),esc(i['input']),esc(i['output']),esc(i['context']),esc(i['evidence'])] for i in d['io']]))
    else:
        out.append(table(['模块','本批需要核对的证据'],[['π0.5','图像/状态 shape、dtype、子任务文本长度与哈希、有效 prompt Token、动作块 shape 与实际执行步数。'],['Astra / Sol','实际消息组成、工具 Schema、图片、记忆与历史；逐请求 input/cached/output/reasoning Token 与请求身份对齐。'],['SAM3','图像、文本或像素点、阈值，mask/box/score；区分服务输出与工具追加的三维反投影。']]))
    io_details=d.get('io_details',[])
    if io_details:
        out.append('<details><summary>逐字段接口证据</summary><div>'+table(['模块','字段','类型 / shape','含义与边界','证据'],[[esc(row.get(k,'')) for k in ('module','field','type_shape','meaning','evidence')] for row in io_details])+'</div></details>')
    out.append(('<details><summary>本问证据</summary><div>'+bullets(q['evidence'])+'</div></details>')+'<p class="boundary"><b>边界：</b>'+esc(q['boundary'])+' “sim3”在本项目按 SAM3 服务解释；最终字段以本批捕获为准。</p></section>')
    out.append('<section id="episodes"><div class="section-no">证据 / 新批次 8 集</div><h2>逐集状态与证据</h2><p class="muted">保留全部 8 个计划身份；未完成、未接入和基础设施异常单列。这里不展示或混算旧 120 集。</p><div class="controls"><label>模型 <select id="model-filter"><option value="">全部</option><option value="gpt-6-astra">Astra</option><option value="gpt-6-sol">Sol</option></select></label><label>场景 <select id="task-filter"><option value="">全部</option><option value="0">t0 · 双罐入篮</option><option value="9">t9 · 微波炉</option></select></label><button id="csv">导出 8 集计时表</button></div><div id="episode-details">')
    for e in eps:
        out.append('<details data-model="'+esc(e['model'])+'" data-task="'+str(e['task'])+'"><summary>'+esc(e['model'].replace('gpt-6-','').title())+' · t'+str(e['task'])+' / seed'+str(e['seed'])+'　'+esc(STATUS[e['status']])+'</summary><div>')
        out.append(table(['指标','结果'],[['attempt',text(e['attempt'],'尚未接入')],['任务结果','未定' if e['environment_success'] is None else ('成功' if e['environment_success'] else '失败')],['计时验收','未定' if e['measurement_invalid'] is None else ('未通过' if e['measurement_invalid'] else '通过')],['episode',number(e['elapsed_s'],' s')],['模型加载（另计）',number(e['load_s'],' s')]]+[[('关联验证五桶 · ' if (e.get('five_bucket_evidence') or {}).get('classification')=='source_backed_step_association' else '条件五桶 · ')+l,number(e[k],' s')] for k,l in zip(KEYS,bucket_labels)]))
        cloud=e.get('cloud') or {}
        out.append('<h3>云端调用口径</h3>'+table(['指标','本集'],[[label,number(cloud.get(key),unit)] for key,label,unit in (
            ('logical_rounds','逻辑推理轮次',''),('sampling_attempts','采样尝试',''),
            ('completed_responses','完整响应',''),('http_attempts','HTTP 尝试',''),
            ('websocket_attempts','WebSocket 尝试',''),('transport_union_s','传输区间并集',' s'),
            ('prefill_s','服务端 prefill',' s'),('hidden_thinking_s','服务端隐藏 thinking',' s'))]))
        out.append('<p class="muted">逻辑轮次、采样尝试、完整响应和传输尝试是不同计数，不相互替代。传输区间并集不等于服务端推理；prefill / hidden thinking 无服务端遥测时保持未知。</p>')
        if e['requests']:
            out.append('<h3>逐请求</h3><p class="muted">stream_lifetime_s：客户端请求流墙钟，非纯服务端推理。TTFT 未关联到本请求时显示 —；不按行号或时间接近强配。</p>'+table(['序号','请求流墙钟 / s','关联 TTFT / s','input','cached','reasoning'],[[str(i+1),number(r.get('stream_lifetime_s')),number(r.get('ttft_s') if r.get('ttft_request_linked') is True else None),number(r.get('input_tokens')),number(r.get('cached_input_tokens')),number(r.get('reasoning_tokens'))] for i,r in enumerate(e['requests'])]))
        ttft=cloud.get('response_ttft_distribution')
        if ttft:
            out.append('<p class="muted">独立响应 TTFT 分布（未绑定请求，不用于拆出 prefill）：样本 '+number(ttft.get('count'))+'；p50 '+number(ttft.get('p50_s'),' s')+'；p95 '+number(ttft.get('p95_s'),' s')+'。</p>')
        if e['trajectory']:out.append('<h3>逐动作证据</h3>'+table(['动作','指令','秒','可见进展','判断依据'],[[esc(r['step']),esc(r['action']),number(r.get('seconds')),esc(r.get('progress','')),esc(r.get('assessment',''))] for r in e['trajectory']]))
        out.append(bullets(e['evidence'],'该样本证据尚未接入。')+'</div></details>')
    related='<p>关联 GPT-as-Policy：<a href="../../2026_0921-0927_同机仿真与云边验证/2026-09-22-GPT-Token与Prefill-Decoder规律分析.md">Token 与 Prefill / Decoder 分析</a>；<a href="../../2026_0921-0927_同机仿真与云边验证/2026-09-23-推理效率与纠错效果深挖.md">推理效率与纠错效果</a>。承接 TTFT 不等于 prefill、端到端瓶颈不等于 π 内核的分析方法；旧数字不进入本批。</p>'
    if d.get('evidence_files'):
        related+=table(['可打开的脱敏证据','用途'],[['<a href="'+esc(x['href'])+'">'+esc(x['label'])+'</a>',esc(x['description'])] for x in d['evidence_files']])
    out.append('</div><details><summary>证据来源与适用范围</summary><div>'+related+bullets(d['evidence'],'尚未接入新批次证据文件。')+'<p class="muted">E1＝源码证据；E2＝本批配置、事件与轨迹；E3＝条件推导。旧实验仅用于选题，不能代替本批测量。公开数据只接受脱敏汇总，禁止原始认证、地址和会话内容。</p></div></details></section><footer>'+esc(d['batch'])+' · 独立深度计时报告<br>单一数据源：report-data.json。未知值显示 —；不补零、不借用历史成功率。</footer></main>')
    data=json.dumps(eps,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    out.append('<script id="data" type="application/json">'+data+'</script><script>const E=JSON.parse(document.getElementById("data").textContent);const applyFilters=()=>{const t=document.getElementById("task-filter").value,m=document.getElementById("model-filter").value;document.querySelectorAll("#episode-details>details").forEach(d=>d.hidden=(!!t&&d.dataset.task!==t)||(!!m&&d.dataset.model!==m))};document.getElementById("task-filter").onchange=applyFilters;document.getElementById("model-filter").onchange=applyFilters;document.getElementById("csv").onclick=()=>{const K=["id","status","environment_success","measurement_invalid","elapsed_s","load_s","request_count","vla_chunks","manual_primitive_s","vla_primitive_s","tool_other_s","sdk_non_mcp_s","outside_sdk_s"];const q=v=>String.fromCharCode(34)+String(v??"").replaceAll(String.fromCharCode(34),String.fromCharCode(34,34))+String.fromCharCode(34);const csv="\\uFEFF"+[K.join(","),...E.map(e=>K.map(k=>q(e[k])).join(","))].join(String.fromCharCode(13,10));const u=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));const a=document.createElement("a");a.href=u;a.download="rpent-deep-timing.csv";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></body></html>')
    return '\n'.join(out)+'\n'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE/'report-data.json')
    parser.add_argument('--output',type=Path,default=HERE/'RPent-两场景深度计时报告.html')
    args=parser.parse_args()
    data=validate(json.loads(args.data.read_text()))
    args.output.write_text(build(data),encoding='utf-8')
    print(args.output)
