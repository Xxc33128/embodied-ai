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
        v2=e.get('timeline_v2')
        if v2:
            def check_intervals(rows,window):
                assert type(window) in (int,float) and math.isfinite(window) and window>0
                previous=-1
                for row in rows:
                    a,b=row['start_s'],row['end_s']
                    assert all(type(v) in (int,float) and math.isfinite(v) for v in (a,b))
                    assert 0<=a<=b<=window+1e-6,'timeline interval outside its own clock window'
                    assert a>=previous,'timeline must be ordered by start time'
                    assert isinstance(row['name'],str)
                    previous=a
            client=v2['client'];local=v2['local']
            for rows in client['lanes'].values():check_intervals(rows,client['window_s'])
            check_intervals(client['events'],client['window_s'])
            assert len(client['events'])==sum(map(len,client['lanes'].values()))
            assert len(client['lanes']['gpt'])==e['cloud']['sampling_attempts']
            assert abs(client['gpt_union_s']-e['cloud']['transport_union_s'])<.001
            check_intervals(local['actions'],local['window_s'])
            assert len(local['actions'])==len(e['trajectory'])
            for action,record in zip(local['actions'],e['trajectory']):
                detail=action['detail']
                assert action['name']==record['action']
                assert abs(detail['window_s']-record['seconds'])<.011
                assert abs(action['end_s']-action['start_s']-detail['window_s'])<1e-6
                for rows in detail['lanes'].values():check_intervals(rows,detail['window_s'])
    return d

def timeline_section(eps):
    out=['<section id="timeline"><div class="section-no">真实区间 / 8 集时间轴 · E2</div><h2>每一集，模型请求与本地执行如何交错？</h2><p>客户端两行共用横坐标，只画实际记录的请求与工具区间；空白保持空白，重叠保持重叠。点击色块查看起止与耗时。</p><p class="boundary">客户端采用同一 Mac 墙钟；本地动作采用 Linux monotonic 时钟，首个动作记为 0 秒。两者不是共同原点，不能上下对齐。动作细节再以该动作起点归零；π 服务、仿真与渲染存在包含关系，不能相加。客户端 GPT 区间为非预热传输尝试的请求生命周期，不是纯 thinking / prefill；它可以跨到工具执行阶段。</p><fieldset class="timeline-picker"><legend>选择要展示的集（可多选）</legend><div class="timeline-options">']
    for i,e in enumerate(eps):
        out.append('<label><input type="checkbox" data-timeline-select="'+str(i)+'"'+(' checked' if i==0 else '')+'> '+esc(e['model'].replace('gpt-6-','').title())+' · t'+str(e['task'])+' / seed'+str(e['seed'])+'</label>')
    out.append('</div><div class="controls"><button type="button" id="timeline-all">全选</button><button type="button" id="timeline-none">清空</button><span id="timeline-count" aria-live="polite"></span></div></fieldset><p id="timeline-empty" class="muted" hidden>请勾选上方的集，查看时间轴。</p>')
    for i,e in enumerate(eps):
        out.append('<article class="timeline-episode" data-timeline-index="'+str(i)+'"'+(' hidden' if i else '')+'><h3>'+esc(e['model'].replace('gpt-6-','').title())+' · t'+str(e['task'])+' / seed'+str(e['seed'])+' <small>'+('未定' if e['environment_success'] is None else ('成功' if e['environment_success'] else '失败'))+' · '+esc(e['attempt'])+'</small></h3><div class="timeline-content"></div></article>')
    out.append('<p class="muted">时间轴仅使用 timeline_v2 的实测区间；旧补集时间轴不参与展示。首尾未记录区间不补齐，各窗均为独立时间尺度。原始轮次与秒数保留于数据和明细。</p></section>')
    return ''.join(out)


def quantitative_conclusion():
    return """<div class="quantitative-conclusion"><h3>定量结论：下一轮应争取多大的改善？</h3><p>分析口径 T* = 10965.30 − 4391.18 = <b>6574.12 s</b>，只扣除原语执行窗中实测的渲染。它仍含仿真控制、物理、通信等，且工具外围可能含未独立归属的图像工作，<b>不是完整去渲染测量，更不是真机时间预测</b>。以下均为 8 集累计工作量。〔E2〕</p>
<p class="factor-equation">ΔT* = 1164.85α + 1047.43β + 4187.54γ + 174.30δ（秒）</p><p>α：动作非渲染部分的降幅；β：工具附加项的降幅；γ：SDK非工具项的降幅；δ：SDK外差额的降幅。降幅取0–1。固定其余条件时，式子给出条件收益；这些是账目降幅，不是已经测得的算法收益，也不是采样次数降幅。〔E3〕</p>
<div class="table-wrap"><table><thead><tr><th>独立假设情景〔E3〕</th><th>节省累计秒数</th><th>T* 降幅</th></tr></thead><tbody><tr><td>SDK非工具项下降10% / 20% / 30%</td><td>418.75 / 837.51 / 1256.26</td><td>6.37% / 12.74% / 19.11%</td></tr><tr><td>动作非渲染部分下降20%</td><td>232.97</td><td>3.54%</td></tr><tr><td>工具附加项下降20%</td><td>209.49</td><td>3.19%</td></tr><tr><td>π handler耗时减半，其余不变</td><td>99.33</td><td>1.51%</td></tr></tbody></table></div><p class="boundary">各行分别计算，π handler 包含于动作项，不能与动作降幅直接叠加。改善某策略后须重新测量各账目；增加诊断或本地技能成本也应计入。压缩上下文20%不等于等待下降20%。</p>
<h4>先校正“恢复很贵”的解释</h4><p>Sol t0/seed1 第18–34步恢复为562.94 s，其中渲染447.54 s；扣除后动作成本为<b>115.40 s，占该集T*的8.04%</b>。Sol t9/seed1 第14–18步对应156.22 −119.61 = <b>36.61 s，占该集T*的5.23%</b>。两段合计152.01 s，占整批T*的2.31%。〔E2：按同一原语窗口扣减；未四舍五入原值计算〕</p><p>所以这两段的<b>直接动作成本</b>不足以单独支撑整批10%的优化目标，即使全部移除也只有2.31%的账面空间；这不是可实现承诺，也不涵盖其他恢复。真正值得验证的是：减少恢复后是否同时减少了GPT决策与工具往返；目前尚未把这些额外往返精确归属到两段恢复，不能预先把它们算作收益。</p>
<h4>把10%的目标变成试验指标</h4><p>若只改SDK非工具项，需减少<b>657.41 s / 15.70%</b>。本批448次逻辑采样对应4187.54 s差额，摊销为<b>9.35 s/次</b>；仅在每轮边际成本等于平均值的假设下，相当于约71次采样，平均约9次/集。<b>这不是实测GPT单次延迟，也不是允许删掉71次调用的证据</b>，只用于判断目标量级。若只改动作非渲染部分，则需下降56.44%，要求明显更高。〔E2计算/E3〕</p><p><b>下一轮顺序：</b>先测试连续技能与检查点调整能否减少往返；同时测试无进展检测和分类恢复，记录它们对动作、工具、采样三项的联动影响。再测试上下文/载荷压缩与重试治理。以成功率、成功集耗时和所有attempt成本联合验收，不将失败更快视为改善。已有证据支持试验优先级，尚不支持把这些条件收益写成已实现加速。</p><p class="muted">计算依据：<a href="evidence/quantitative-conclusion.json">逐项预算与两段恢复扣减数据</a>。范围限本次两场景8集；12/14条件理想采样不用于计算上述收益。</p></div>"""


def front_answers():
    return """<section id="answers"><div class="section-no">先读结论 / 仅本次 8 集深度测试</div><h2>三个问题，先给答案</h2>
<article><h3>01 · prefill 和 thinking 能拆开吗？</h3><p><b>云端仍不能精确拆分；本地 π 阶段已测到。</b> 473 次请求尝试有客户端时序，但没有服务端 prefill / 完整 hidden thinking 边界。π handler 为 198.65 s，其中 prefix 合并 96.19 s、denoise 核心 86.14 s；它们不是 Astra 的阶段计时。精确云端拆分需要服务端遥测，补跑与 TTFT 本身不能替代。〔E2〕 <a href="#q1">测量证据 →</a></p></article>
<article><h3>02 · 是否有过多尝试？理想轮次与基线是什么？</h3><p><b>有耗时较大的恢复与预算耗尽，但不能把所有局部失败都判成无效。</b> t0 的 Sol seed1 后段恢复用了 562.95 s；t9 的 Astra seed0 有 14 次预算耗尽移动，共 672.90 s。〔E2〕</p><p><b>双罐入篮 t0：</b>候选 10 动作 / 12 次采样，最快已观测成功 1049.1 s。<br><b>杯入微波炉并关门 t9：</b>候选 12 动作 / 14 次采样，四集全失败，无成功耗时基线。</p><p class="boundary">12 / 14 是每动作一次采样，再加 finish 与最终回复的条件推演〔E3〕；额外观察另加，合批可能减少，非已证明最少轮次。没有最优路径秒数实跑，恢复耗时也不等于保证可节省时间。<a href="#q2">轨迹与路径假设 →</a></p></article>
<article><h3>03 · 收发哪些内容？上下文长度能一一对应吗？</h3><p><b>π0.5：图像 + 状态 + 动作指令 → 动作块；Astra / Sol：规则 + 任务 + 历史 / 记忆 + 工具反馈 → 决策；SAM3：图像 + 定位提示 → 检测 / 分割结果。</b> “sim3”按实际组件 SAM3 分析。〔E1/E2〕</p><p>逐字段形状、prompt 构成与官方逐次 token 数已有举证；<b>缺完整出站快照，不能证明每段内容与每次 input_tokens 一一对应</b>。字符数、补齐长度、累计输入均不能替代单次有效上下文。<a href="#q3">逐字段与 token 证据 →</a></p></article></section>"""


def factor_breakdown():
    rows=[
      ('T动作非π', 'Σ动作 [动作准备 + Σ控制步 (控制计算 + 物理推进 + 相机渲染 + 观察处理 / 通信) + 动作收尾]',
       '次数：路径段数、恢复次数。步数：目标距离与姿态、控制步幅、到达容差、接触状态、最大步数。每步：相机数量 / 分辨率、渲染实现、物理模型、设备与 RPC。',
       '优先减少无进展步数与恢复路径；渲染不列入后续优化目标。改变步幅、预算或容差会改变控制条件，须另设成功率对照，不能以提前失败换取较短耗时。'),
      ('Tπ动作', 'Σ技能 [技能准备 + Σ动作块 (π 请求往返 + 动作块执行与观察 + 终止检查) + 技能收尾]；π handler 内含前处理、prefix 合并、denoise、后处理',
       '块数：抓取 / 任务成功谓词、最大预算、失败恢复、指令与感知质量。单块策略成本：图像与有效 token 长度、denoise 步数、设备、首次调用 / 缓存状态。执行成本：本批每块 5 个控制步及其渲染、物理与通信。',
       '优先减少不成功动作块。371 次 handler 共 198.65 s，而完整 π 动作共 1110.09 s；单独加速 π 网络只作用于其中一部分。不能把整段 π 动作都归为神经网络推理。'),
      ('T工具附加（残差）', '精确定义：工具区间并集 − 非π动作 − π动作；内部包括其他工具与动作工具外围工作',
       '其他工具的调用数与类型（记忆、观察、定位）；SAM3 图像大小、提示、设备；深度反投影；图像生成 / 编码、返回载荷大小、文件 I/O、封装及传输。现有差额不能逐秒分配到这些因素。',
       '按工具追踪高频调用和大载荷，补独占阶段 span 后再优化缓存、图像编码和通信；不把必要感知工作当作全部可删除的开销。'),
      ('TSDK非工具（残差）', '精确定义：SDK 报告时长 − 工具区间并集；受采样轮数、每次等待及编排共同影响',
       '采样轮数受规划与失败恢复影响；请求等待受输入 / 缓存、模型推理与输出、服务排队、网络、重试 / 重连影响；客户端还存在编排等待。云端各因素没有独立计时。',
       '减少多余轮次与无用上下文、检查重试与缓存命中。云端 prefill / thinking 的精确优化归因须服务端遥测；不能把 4187.54 s 全算成 thinking。GPT 请求生命周期可能与工具重叠，不能直接求和代替此项。'),
      ('TSDK外开销（残差）', '精确定义：episode 报告时长 − SDK 报告时长',
       'episode 边界内、SDK 边界外的外围准备 / 清理及其等待；具体函数贡献仍缺独立 span，原始时长的一位小数舍入也影响差额。VLA 冷加载不在这一项里。',
       '先补同一时钟的边界事件再归因；目前只占约 1.59%，不优先做未经定位的重构。冷加载优化应计入另列的复现成本。')]
    return rows



def formula_explanation(eps):
    definitions=[
        ('T动作非π','执行不调用 π 的机器人动作',
         '例如 move_to 移动、move_pose 调姿态、开合夹爪。这些动作直接通过控制器驱动仿真；耗时已经包含动作内部的物理计算、相机渲染和通信。',
         '把本集所有非 π 动作原语的 elapsed_s 相加。原语就是工具内部执行一次机器人动作的函数。'),
        ('Tπ动作','执行由 π 策略驱动的完整动作',
         '例如 pi0_pick 抓取、pi0_doubled 操作。一次动作可能反复执行“π 推理 → 动作块 → 仿真与渲染 → 新观察”，直到结束或达到预算。因此这一项不是纯 π 推理时间。',
         '把本集所有 π 动作原语的 elapsed_s 相加。π handler 推理服务耗时只是其中一部分。'),
        ('T工具附加（残差）','工具调用中，除上述机器人动作以外的时间',
         '包括读记忆、观察、定位等其他工具调用，以及动作工具返回前的图像产物生成、封装、通信等。它既含必要工作，也含额外开销，不能一概当成浪费。',
         '工具调用总时间 − T动作非π − Tπ动作。工具调用总时间按已完成调用区间的并集统计；若区间重叠，只计一次。'),
        ('TSDK非工具（残差）','GPT 决策循环内，工具没有执行时的其余时间',
         '包含 GPT 请求与响应等待，以及网络、重连和客户端编排等。它体现了 GPT 相关等待，但不能全部归为 GPT 计算，更不能叫纯 thinking。',
         'SDK 报告时长 − 工具调用总时间。SDK 指本实验的模型与工具调用运行框架，不是另一个模型。'),
        ('TSDK外开销（残差）','episode 内，SDK 计时范围之外的时间',
         '属于本次 episode 边界内的外围准备与清理等工作；现有证据只支持总差额，不能把每秒进一步分配给某个初始化或收尾步骤。',
         'episode 报告时长 − SDK 报告时长。两项原始时长保留一位小数，差额继承其舍入误差；它不包含另列的 VLA 冷加载。')]
    out=['<div class="formula-explanation"><h3>公式怎么读：一集的时间记到五个互不重复的账目里</h3><p><b>T_episode 是单集 transcript 中记录的 elapsed_s。</b> 它表示本报告采用的单集评测时长，不是整个进程从启动到退出的墙钟；VLA 冷加载另列。上方秒数是 8 集对应账目的累计值，不是并行运行整批的等待时间。〔E2〕</p><p class="boundary">下列结构由源码支持〔E1〕、次数与时长由本批记录支持〔E2〕。未逐项计时的影响因素与优化方向为待验证分析〔E3〕，不为它们虚构秒数；子阶段不能再次加到总公式。</p><dl class="formula-definitions">']
    factors = factor_breakdown()
    for i,(symbol,title,meaning,method) in enumerate(definitions):
        out.append('<div><dt>'+esc(symbol)+'<strong>'+esc(title)+'</strong></dt><dd><p>'+esc(meaning)+'</p><p class="formula-method"><b>统计方法：</b>'+esc(method)+'</p><p class="factor-equation"><b>内部结构：</b>'+esc(factors[i][1])+'</p><p><b>影响因素：</b>'+esc(factors[i][2])+'</p><p><b>优化判断：</b>'+esc(factors[i][3])+'</p></dd></div>')
    out.append('</dl><p class="boundary"><b>“残差”是什么意思？</b> 用一个较大的计时范围减去已经单独记账的部分，得到剩余时间。三个差额的外层边界分别是工具调用、SDK、episode；这里的后三项都由相减得到，没有对其内部每个子阶段逐一计时。名称中的“残差”表示测量方式，不表示这些时间不存在。〔E1/E2〕</p>')
    example=next((e for e in eps if e['model']=='gpt-6-astra' and e['task']==0 and e['seed']==0),None)
    if example and all(example.get(k) is not None for k in KEYS):
        out.append('<div class="formula-example"><b>具体例子 · Astra / t0 / seed0</b><p>'+number(example['elapsed_s'],' s')+' = '+ ' + '.join(number(example[k],' s') for k in KEYS)+'</p><p class="muted">依次对应：非 π 动作、π 动作、工具其余时间、SDK 内非工具时间、SDK 外时间。五项合计回到本集总时长；上面两个动作项已经含内部渲染，不能再把渲染秒数加一次。此处显示两位小数。〔E2〕</p></div>')
    out.append('<p><b>这是一份耗时账本，不是执行顺序。</b> 一集会反复经历 GPT 请求、工具调用和动作执行；真实先后看下面的时间轴。也不能把另列的 GPT 请求生命周期再加到公式里，它与工具区间可能重叠。</p></div>')
    out.append('<div class="formula-example"><b>从记账公式到可调因素〔E3：优化模型〕</b><p>T执行 ≈ N动作 × 平均动作成本；平均动作成本受控制步数、π块数和恢复影响。<br>T高层等待 ≈ N采样 × 平均非工具等待；T工具外围受调用次数与返回载荷影响。</p><p>这些是定位用的近似关系，不替代上方精确五项账本；请求与工具可能重叠，不能用请求时长直接相加。减少一次失败恢复，可能同时减少动作、工具和后续决策，具有跨项优化空间；仅加速某个算子，只降低该子阶段。具体节省量须通过成功对照测得。</p><p><b>后续可调因素：</b>恢复策略与无进展阈值 → 技能粒度与检查点 → 历史/图像载荷与重试 → 单次执行成本。渲染保留为本次仿真测量证据，按用户场景不列为后续优化目标。</p></div>')
    return ''.join(out)


TIMELINE_SCRIPT=r"""
const tlNode=(tag,cls,value)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=value;return n};
const tlTime=v=>Number(v).toFixed(3);
const tlLabel=x=>`${x.name} · ${tlTime(x.start_s)}–${tlTime(x.end_s)} s · 耗时 ${tlTime(x.end_s-x.start_s)} s`;
function tlTable(parent,events,title){
 const fold=tlNode('details'),summary=tlNode('summary','',title+' · '+events.length+' 段');fold.append(summary);
 const wrap=tlNode('div','table-wrap'),table=tlNode('table'),head=tlNode('thead'),tr=tlNode('tr');
 ['行 / 内容','起止 / s','耗时 / s'].forEach(x=>tr.append(tlNode('th','',x)));head.append(tr);table.append(head);
 const body=tlNode('tbody');for(const e of events){const r=tlNode('tr');r.append(tlNode('td','',(e.lane?e.lane+' · ':'')+e.name),tlNode('td','',tlTime(e.start_s)+' → '+tlTime(e.end_s)),tlNode('td','',tlTime(e.end_s-e.start_s)));body.append(r)}
 table.append(body);wrap.append(table);fold.append(wrap);parent.append(fold);
}
function tlChart(parent,windowS,lanes,output){
 const chart=tlNode('div','timeline-chart');
 for(const [key,label,events] of lanes){
  const row=tlNode('div','timeline-lane'),name=tlNode('span','timeline-lane-label',label),track=tlNode('div','timeline-track');
  track.setAttribute('aria-label',label+'，0 至 '+tlTime(windowS)+' 秒');
  for(const x of events){const button=tlNode('button','tl-'+key);button.type='button';button.style.left=(windowS>0?100*x.start_s/windowS:0)+'%';button.style.width=(windowS>0?100*(x.end_s-x.start_s)/windowS:0)+'%';button.title=tlLabel(x);button.setAttribute('aria-label',tlLabel(x));button.onclick=()=>{output.textContent=tlLabel(x)};track.append(button)}
  row.append(name,track);chart.append(row);
 }
 const axisrow=tlNode('div','timeline-lane'),axis=tlNode('div','timeline-axis');axisrow.append(tlNode('span'));
 [0,windowS/2,windowS].forEach(v=>axis.append(tlNode('span','',tlTime(v)+' s')));axisrow.append(axis);chart.append(axisrow);parent.append(chart);
}
function tlRenderAction(parent,action){
 parent.replaceChildren();const detail=action.detail;
 parent.append(tlNode('p','muted','选中动作：'+tlLabel(action)+'（本地动作窗）。以下 0 秒仅为该动作起点。'));
 if(!detail){parent.append(tlNode('p','muted','该动作细阶段待接入。'));return}
 const output=tlNode('p','timeline-output','点击细阶段色块查看；小段可在明细中读取。');output.setAttribute('aria-live','polite');
 const lanes=[['pi','π 服务',detail.lanes.pi||[]],['sim','仿真 step（含 render）',detail.lanes.sim||[]],['render','render',detail.lanes.render||[]]];
 tlChart(parent,detail.window_s,lanes,output);parent.append(output);
 const events=lanes.flatMap(([key,label,items])=>items.map(x=>({...x,lane:label}))).sort((a,b)=>a.start_s-b.start_s);
 tlTable(parent,events,'选中动作细阶段明细');
}
function tlRenderEpisode(article,index){
 if(article.dataset.rendered)return;article.dataset.rendered='true';const host=article.querySelector('.timeline-content'),t=E[index].timeline_v2;
 if(!t){host.append(tlNode('p','muted','真实区间 timeline_v2 待接入；不使用旧 gap 推算 GPT 时间。'));return}
 const client=t.client;host.append(tlNode('h4','','客户端 · 同一 Mac 墙钟'));
 if(client){
  host.append(tlNode('p','muted','客户端事件窗 '+tlTime(client.window_s)+' s。GPT 请求与工具共用下列横坐标。'));
  const output=tlNode('p','timeline-output','点击请求或工具色块查看实际区间。');output.setAttribute('aria-live','polite');
  tlChart(host,client.window_s,[['gpt','GPT 请求',client.lanes.gpt||[]],['tools','工具调用',client.lanes.tools||[]]],output);host.append(output);
  tlTable(host,client.events||[],'客户端事件明细（按开始时间）');
 }else host.append(tlNode('p','muted','客户端实测区间待接入。'));
 const local=t.local,block=tlNode('div','timeline-local');block.append(tlNode('h4','','本地执行 · 独立 Linux 时钟'));
 block.append(tlNode('p','boundary','与上方客户端无共同原点，不能按横向位置比较发生时刻。选择一个动作，查看它内部的独立时间窗。'));
 if(!local||!local.actions.length){block.append(tlNode('p','muted','本地动作区间待接入。'));host.append(block);return}
 const controls=tlNode('div','controls'),label=tlNode('label','','动作 '),select=tlNode('select','timeline-action-select');
 select.setAttribute('aria-label','选择本地动作');local.actions.forEach((a,i)=>{const o=tlNode('option','',(i+1)+'. '+a.name+' · '+tlTime(a.start_s)+'–'+tlTime(a.end_s)+' s');o.value=i;select.append(o)});
 label.append(select);controls.append(label);block.append(tlNode('p','muted','本地动作窗 '+tlTime(local.window_s)+' s；首动作起点为 0。'),controls);
 const detailHost=tlNode('div','timeline-action-detail');block.append(detailHost);host.append(block);
 const firstPi=local.actions.findIndex(a=>String(a.name).startsWith('pi0'));select.value=String(firstPi<0?0:firstPi);
 select.onchange=()=>tlRenderAction(detailHost,local.actions[Number(select.value)]);select.onchange();
}
const timelineChoices=[...document.querySelectorAll('[data-timeline-select]')];
const updateTimeline=()=>{const selected=new Set(timelineChoices.filter(c=>c.checked).map(c=>c.dataset.timelineSelect));document.querySelectorAll('[data-timeline-index]').forEach(a=>{a.hidden=!selected.has(a.dataset.timelineIndex);if(!a.hidden)tlRenderEpisode(a,Number(a.dataset.timelineIndex))});document.getElementById('timeline-count').textContent='已展示 '+selected.size+' / '+timelineChoices.length+' 集';document.getElementById('timeline-empty').hidden=selected.size>0};
timelineChoices.forEach(c=>c.onchange=updateTimeline);document.getElementById('timeline-all').onclick=()=>{timelineChoices.forEach(c=>c.checked=true);updateTimeline()};document.getElementById('timeline-none').onclick=()=>{timelineChoices.forEach(c=>c.checked=false);updateTimeline()};updateTimeline();
"""

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
    mode_note=('前两项统计机器人动作执行，后三项由较大的计时范围减去已记账部分得到；每项的含义和计算方法见下方。口径经源码与逐步关联核验。〔E1/E2〕' if expanded else
               '条件模式：五桶仍需原语串行、MCP 包含及数值闭合证据；代数闭合不能独立证明互斥，暂不展示五桶精确占比。〔E3〕')
    css='''

#answers article{padding:8px 0 16px;border-top:1px solid var(--line)}#answers article:first-of-type{border-top:0}#answers h3{font-size:21px}.factor-breakdown article{border-top:1px solid var(--line);padding:10px 0}.factor-breakdown h4{font-size:17px;margin:12px 0}.factor-equation{background:#edf3f6;border-left:3px solid #126580;padding:12px 16px;font-size:14px}.factor-breakdown article p{font-size:14px}.formula-definitions{margin:20px 0}.formula-definitions>div{display:grid;grid-template-columns:235px minmax(0,1fr);gap:20px;padding:18px 0;border-top:1px solid #d7e1e7}.formula-definitions dt{font-size:13px;color:#126580}.formula-definitions dt strong{display:block;font-size:16px;color:#18313f;line-height:1.5;margin-top:7px}.formula-definitions dd{margin:0;font-size:14px}.formula-definitions dd p{margin:0 0 9px}.formula-method{color:#526b78;font-size:13px}.formula-example{border-left:3px solid #126580;background:#eff5f8;padding:15px 18px;margin:20px 0}.formula-example p{margin:8px 0}.formula-example .num{display:inline-block}@media(max-width:700px){.formula-definitions>div{grid-template-columns:1fr;gap:8px}.formula-definitions dt strong{margin-top:3px}.formula-example{padding:12px}}@media print{.formula-definitions>div{break-inside:avoid;grid-template-columns:190px minmax(0,1fr);gap:12px}.formula-definitions dd{font-size:11px}}
:root{--ink:#18313f;--muted:#627783;--line:#d7e1e7;--accent:#126580;--paper:#fbfcfd}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:72px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;overflow-wrap:anywhere}a{color:var(--accent)}nav{position:sticky;top:0;z-index:5;background:#fbfcfdf5;border-bottom:1px solid var(--line);display:flex;gap:25px;padding:13px max(20px,calc((100vw - 1060px)/2));overflow:auto;white-space:nowrap;backdrop-filter:blur(8px)}nav a{font-size:13px;text-decoration:none;color:#355566;transition:color .15s}nav a:hover{color:#0891b2}main{max-width:1140px;margin:auto;padding:0 40px 60px}header{padding:35px 0 14px}.eyebrow,.section-no{color:var(--accent);font-size:11px;letter-spacing:1.5px;font-weight:650}h1{font-size:clamp(28px,3.4vw,43px);line-height:1.25;margin:12px 0}h2{font-size:27px;line-height:1.4;margin:8px 0 18px}h3{font-size:19px;line-height:1.5;margin:22px 0 10px}p{margin:10px 0 16px}.muted{color:var(--muted);font-size:13px}.meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:13px}.formula-board{background:#173642;color:#fff;padding:25px 28px;margin:10px 0 18px}.formula-title{font-size:12px;color:#bbd3de}.formula{display:flex;flex-wrap:wrap;gap:0 8px;font-size:clamp(18px,2.3vw,27px);line-height:1.8;margin:12px 0}.formula .term{white-space:nowrap}.terms{display:grid;grid-template-columns:repeat(5,1fr);gap:18px;border-top:1px solid #ffffff35;padding-top:17px}.terms strong{font-size:25px;display:block;font-weight:550}.terms span{font-size:12px;color:#c4d7df}.formula-note{font-size:12px;color:#c4d7df;margin:16px 0 0}.num{white-space:nowrap}.unknown{color:#7b8f99;font-variant-numeric:tabular-nums}.formula-board .unknown{color:#a9c1cd}.statusline{border-left:3px solid var(--accent);padding:9px 15px;background:#eff5f8;font-size:14px;margin:20px 0}.mechanism{font-size:17px;padding:16px 0;border-bottom:1px solid var(--line)}section{padding:35px 0;border-bottom:1px solid var(--line)}#formula{padding-top:0}.answer{font-size:19px;max-width:880px;margin-bottom:22px}.boundary{font-size:13px;color:var(--muted);border-left:2px solid var(--line);padding-left:15px;margin:20px 0}.table-wrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;color:#375b6b;background:#edf3f6;white-space:nowrap}td,th{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}td{line-height:1.6}details{border-top:1px solid var(--line);padding:0;margin:18px 0 0}summary{cursor:pointer;font-size:14px;font-weight:550;color:#365c6e;padding:15px 0;transition:color .15s}summary:hover{color:#0d7a9d}details>div{padding:0 0 16px}.scene{padding:15px 0 20px;border-bottom:1px solid var(--line)}.scene:last-child{border-bottom:0}.label{font-size:12px;letter-spacing:.6px;color:var(--accent)}.figures{display:grid;grid-template-columns:1fr 1fr;gap:22px}.figures figure{margin:0}.figures img{width:100%;display:block}.figures figcaption{font-size:12px;color:var(--muted)}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:15px 0}select,button{font:inherit;font-size:13px;padding:8px 12px;background:#fff;border:1px solid #b9ccd6;color:var(--ink);border-radius:2px}button{cursor:pointer}pre{font:12px/1.7 ui-monospace,monospace;background:#eef3f6;padding:18px;overflow:auto;white-space:pre-wrap}footer{padding-top:25px;color:var(--muted);font-size:12px}ul{padding-left:21px}li{margin:5px 0}@media(max-width:700px){main{padding:0 19px 40px}header{padding-top:25px}nav{padding:12px 19px;gap:19px}h2{font-size:23px}.answer{font-size:17px}.formula-board{padding:21px 18px}.terms{grid-template-columns:repeat(3,1fr);gap:15px}.terms strong{font-size:22px}.formula{font-size:19px}section{padding:28px 0}.figures{grid-template-columns:1fr}th,td{padding:8px}th{white-space:normal}.meta{gap:8px 18px}#q3 th{white-space:normal}#q3 th,#q3 td{padding:5px;font-size:12px}.scene th{white-space:normal}.scene th,.scene td{padding:8px;font-size:13px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}@page{size:A4;margin:14mm}@media print{h2,h3,.section-no{break-after:avoid}.formula-board{background:white;color:#18313f;border:1px solid #18313f}.formula-board .unknown,.formula-title,.terms span,.formula-note{color:#526772}.terms{border-color:#cbd6dc}tr,.formula-board{break-inside:avoid}nav,.controls{display:none}main{max-width:none;padding:0}body{font-size:11px;background:white;color:#111}h1{font-size:28px}h2{font-size:21px}.formula-board{print-color-adjust:exact;-webkit-print-color-adjust:exact}section{padding:20px 0}.table-wrap{overflow:visible}table{font-size:10px}th,td{padding:6px}.answer{font-size:14px}details:not([open]){display:none}}

[hidden]{display:none!important}.timeline-picker{border:1px solid var(--line);margin:20px 0;padding:12px 16px}.timeline-picker legend{font-size:14px;font-weight:600}.timeline-options{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px 18px}.timeline-options label{font-size:14px;cursor:pointer;display:flex;align-items:center;gap:5px;min-height:32px}.timeline-options input{accent-color:#216c83;width:16px;height:16px}#timeline-count{font-size:13px;color:var(--muted)}.timeline-episode{padding:14px 0 22px;border-bottom:1px solid var(--line)}.timeline-episode h3{margin:12px 0 5px}.timeline-episode h4{font-size:16px;margin:24px 0 10px}.timeline-episode small{font-size:12px;font-weight:400}.timeline-lane{display:grid;grid-template-columns:110px minmax(0,1fr);gap:10px;align-items:center;margin:8px 0}.timeline-lane-label{font-size:12px;line-height:1.4}.timeline-track{position:relative;height:32px;width:100%;background:#edf1f4;overflow:hidden}.timeline-track button{position:absolute;top:0;height:32px;min-width:0;padding:0;border:0;border-radius:0;opacity:.85}.timeline-track button:hover,.timeline-track button:focus-visible{opacity:1;outline:2px solid #172f3b;outline-offset:-2px;z-index:2}.tl-gpt{background:#526581}.tl-tools{background:#247b85}.tl-pi{background:#b77c23}.tl-sim{background:#36868a}.tl-render{background:#8964a3}.timeline-axis{display:flex;justify-content:space-between;font:10px/2 ui-monospace,monospace;color:var(--muted)}.timeline-output{font-size:13px;min-height:24px;color:#375b6b}.timeline-local{border-top:2px solid var(--line);margin-top:26px;padding-top:4px}.timeline-action-select{max-width:100%;width:100%;white-space:normal}.timeline-local .controls label{min-width:0;width:100%}.timeline-episode th{white-space:normal}.timeline-episode details{margin-top:5px}@media(max-width:700px){.timeline-lane{grid-template-columns:76px minmax(0,1fr);gap:7px}.timeline-axis{font-size:9px}.timeline-episode th,.timeline-episode td{padding:5px;font-size:12px}}@media print{.timeline-picker,#timeline-empty{display:none}.timeline-track{print-color-adjust:exact;-webkit-print-color-adjust:exact}.timeline-output{display:none}.timeline-lane{break-inside:avoid}}
'''
    out=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>'+esc(d['title'])+'</title><style>'+css+'</style></head><body>']
    out.append('<nav aria-label="报告导航"><a href="#answers">三问结论</a><a href="#formula">总公式</a><a href="#timeline">逐集时间轴</a><a href="#q1">01 · 时间拆分</a><a href="#q2">02 · 轨迹与基线</a><a href="#q3">03 · 输入与上下文</a><a href="#episodes">8 集明细</a></nav><main><header><div class="eyebrow">RPENT / DEEP TIMING / INDEPENDENT BATCH</div><h1>'+esc(d['title'])+'</h1><div class="meta"><span>Long Swap · t0 / t9</span><span>Astra / Sol · medium · seed0 / 1</span><span>数据更新：'+text(d['updated_at'],'尚未接入')+'</span></div></header>')
    out.append(front_answers())
    out.append('<section id="formula"><div class="formula-board"><div class="formula-title">总公式 · '+('源码与逐步关联支持的五项分解' if expanded else '顶层计时与条件分解')+'</div><div class="formula"><span class="term">T<sub>episode</sub> =</span>'+''.join('<span class="term">'+(' + ' if i else '')+'T<sub>'+esc(term)+'</sub></span>' for i,term in enumerate(formula_terms))+'</div><div class="terms">'+''.join('<div><strong>'+number(v,' s')+'</strong><span>'+esc(label)+'</span></div>' for v,label in zip(display_values,display_labels))+'</div><p class="formula-note">'+mode_note+' 数值是纳入样本的累计工作量，不是并行批次墙钟。模型加载另列。</p></div>')
    if expanded:out.append(formula_explanation(eps))
    out.append('<div class="mechanism"><b>优化看两层：轮数 × 每轮成本。</b><br>每轮 ≈ 模型等待 + 工具执行；本地 π = 前处理 + prefix 合并 + denoise + 后处理。<p class="muted">工具动作内部含策略、控制、物理、渲染和 RPC；这些是嵌套组成，不再加到五桶。模型等待还混有网络、排队与编排，不能当成纯 thinking。prefix 合并项不继续臆分编码 / prefill。〔E3：结构模型〕</p></div>')
    out.append('<div class="statusline">'+esc(d.get('note',''))+' 已结束 '+str(finished)+'/8，计时验收通过 '+str(len(admitted))+'/8。</div><h3>最终结论：公式指向哪些优化因素</h3><p class="answer">'+text(d['conclusion'],'接入新数据后，先确定最大耗时项，再确定应减少调用次数，还是降低单次执行成本。当前不填入旧 120 集的占比。')+'</p>')
    out.append(quantitative_conclusion())
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
    out.append(timeline_section(eps))
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
    out.append('<script id="data" type="application/json">'+data+'</script><script>const E=JSON.parse(document.getElementById("data").textContent);'+TIMELINE_SCRIPT+'const applyFilters=()=>{const t=document.getElementById("task-filter").value,m=document.getElementById("model-filter").value;document.querySelectorAll("#episode-details>details").forEach(d=>d.hidden=(!!t&&d.dataset.task!==t)||(!!m&&d.dataset.model!==m))};document.getElementById("task-filter").onchange=applyFilters;document.getElementById("model-filter").onchange=applyFilters;document.getElementById("csv").onclick=()=>{const K=["id","status","environment_success","measurement_invalid","elapsed_s","load_s","request_count","vla_chunks","manual_primitive_s","vla_primitive_s","tool_other_s","sdk_non_mcp_s","outside_sdk_s"];const q=v=>String.fromCharCode(34)+String(v??"").replaceAll(String.fromCharCode(34),String.fromCharCode(34,34))+String.fromCharCode(34);const csv="\\uFEFF"+[K.join(","),...E.map(e=>K.map(k=>q(e[k])).join(","))].join(String.fromCharCode(13,10));const u=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));const a=document.createElement("a");a.href=u;a.download="rpent-deep-timing.csv";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></body></html>')
    return '\n'.join(out)+'\n'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE/'report-data.json')
    parser.add_argument('--output',type=Path,default=HERE/'RPent-两场景深度计时报告.html')
    args=parser.parse_args()
    data=validate(json.loads(args.data.read_text()))
    args.output.write_text(build(data),encoding='utf-8')
    print(args.output)
