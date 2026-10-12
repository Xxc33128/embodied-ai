# coding: utf-8
"""Offline report; consumes only payload-free audited evidence. Python stdlib."""
import json,statistics,html,math
from pathlib import Path
B=Path(__file__).resolve().parent
D=json.loads((B/'audited-data.json').read_text());T=json.loads((B/'timeline-evidence.json').read_text());I=json.loads((B/'io-evidence.json').read_text())
def esc(x):return html.escape(str(x))
def num(x):return '—' if x is None else f'{x:,.2f}'
def q(a,p):
 a=sorted(x for x in a if x is not None)
 if not a:return None
 pos=(len(a)-1)*p;j=int(pos);return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(pos-j)
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def span(e,process,name):
 for fam in e['local_execution'].get(process,{}).values():
  if isinstance(fam,dict) and name in fam.get('by_span',{}):return fam['by_span'][name]
 return {'spans':0,'sum_ms':0}
parts=[]
def add(s):parts.append(s)
def section(n,title):add(f'<section id="s{n}"><div class="eyebrow">{n:02d} / ANALYSIS</div><h2>{title}</h2>')
add('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RPent × DeepSeek｜两场景性能审计</title><style>
:root{--ink:#172c39;--muted:#5c6c75;--accent:#087e8b;--line:#d4dde0}*{box-sizing:border-box}body{margin:0;background:#fafbf9;color:var(--ink);font:16px/1.8 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}main{max-width:1160px;margin:auto;padding:42px 32px 90px}header{padding:28px 0 35px;border-bottom:3px solid var(--ink)}h1{font-size:42px;line-height:1.2;letter-spacing:-1px;max-width:850px}h2{font-size:28px;line-height:1.35;margin:9px 0 24px}h3{font-size:19px;margin-top:26px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--accent);font-weight:700}.lead{font-size:20px;max-width:850px}nav{display:flex;flex-wrap:wrap;gap:18px;padding:20px 0;border-bottom:1px solid var(--line)}a{color:var(--accent)}section{padding:42px 0;border-bottom:1px solid var(--line)}p{max-width:960px}li{margin:12px 0}.formula{font:18px/1.9 ui-monospace,monospace;padding:24px;background:#eaf2f1;overflow:auto}.muted,small{color:var(--muted)}.scroll{overflow:auto;max-height:650px}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);min-width:80px}th{background:#eaf0ef;position:sticky;top:0;white-space:nowrap}details{margin:18px 0;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer;font-weight:650}pre{overflow:auto;font-size:12px;padding:20px;background:#edf2f2;max-height:450px}select,button{padding:10px;border:1px solid var(--line);background:white;color:var(--ink);font-size:15px}button{cursor:pointer}canvas{width:100%;height:270px;display:block;margin:24px 0;background:white}.notice{border-left:3px solid var(--accent);padding:8px 20px}.metric{font-size:32px;color:var(--accent)}@media(max-width:700px){main{padding:22px 18px}h1{font-size:32px}h2{font-size:24px}td,th{padding:9px}.lead{font-size:18px}}@media print{nav,button{display:none}.scroll{max-height:none;overflow:visible}section{break-inside:auto}body{font-size:11px}main{padding:0}canvas{max-height:220px}details{display:block}}
</style><main><header><div class="eyebrow">RPENT / DEEPSEEK-V4.1-FLASH / 2026.10.12</div><h1>两场景性能审计<br>耗时、瓶颈与优化收益</h1><p class="lead">DeepSeek 推理耗时分析：实测数据、分阶段模型与优化预算。</p><p class="muted">2026-10-12 · v1.3 · E1源码，E2日志实测，E3条件估算。</p></header><nav><a href="#conclusions">结论与公式</a><a href="#proxy-audit">数据附录</a><a href="#s2">① 推理计时</a><a href="#s3">② 场景与轨迹</a><a href="#s4">③ 输入输出</a><a href="#s5">逐集时间轴</a><a href="#s6">执行开销</a><a href="#s7">证据与明细</a></nav>''')
from report_proxy import render as render_proxy
proxy_appendix=render_proxy(table)
from report_conclusions import render as render_conclusions
add(render_conclusions(table))
from report_intro import render
add('<details><summary>原成功4集：耗时模型与优化预算</summary>'+render(D, table)+'</details>')
section(2,'① 原8+8集：客户端实测、统计派生与估算的边界')
rows=[]
for effort in ['medium','none']:
 es=[e for e in D['batches']['流式补测'] if e['effort']==effort];rs=[r for e in es for r in e['llm']['per_request']];tt=[r['ttft_ms']/1000 for r in rs if r['ttft_ms'] is not None];tp=[r['tpot_ms'] for r in rs if r['tpot_ms'] is not None];think=[r['thinking_visible_ms']/1000 for r in rs if r['thinking_visible_ms'] is not None]
 rows.append([effort,len(rs),len(tt),num(q(tt,.5)),num(q(tt,.9)),num(min(tt)),num(max(tt)),num(q(tp,.5)),len(think),num(q(think,.5))])
add(table(['档位','响应attempt','有效样本','TTFT P50 s','P90 s','最小 s','最大 s','TPOT P50 ms/token','thinking窗口n','窗口P50 s'],rows))
add('''<p>这些分布混合不同上下文长度与任务阶段，不能把最大/最小比直接叫同输入网络抖动。分位数按所有有效请求做线性插值，不是逐集中位数的平均；177 个有效样本 / 178 个响应 attempt，177 个逐集去重逻辑请求。medium t0/seed1 有一次 12.412 s 失败 attempt，无 usage、无有效 delta；保留其成本，不将缺值当零。〔E2〕</p>''')
add(table(['指标','准确口径','不能推断'],[
['TTFT〔E1/E2〕','从客户端请求起点到首个非空 reasoning/content/tool-call delta；取三者最早值','不是纯 prefill；tool delta 可能先到元数据，不保证第一个文本 token。'],
['TPOT proxy〔E1/E2〕','(最后 chunk 到达 − TTFT)/(completion_tokens − 1)，每请求派生；包含 reasoning 输出','不是逐 token 间隔；尾部可能包含 usage/[DONE]，也不是设备纯 decode。'],
['thinking 可见窗口〔E1/E2〕','首 reasoning delta → 首 content delta，medium 有82条，共535.721 s','不是完整思考时间；86条有 reasoning，直接工具调用或截断可能没有content。none缺值不等于内部计算为0。'],
['prefill〔E3〕','旧校准拟合 TTFT 对 prompt 长度斜率 .0368 ms/token、截距约7.042 s','不能拆出纯服务端 prefill；缓存、队列和传输混杂。不把倒数写成实测 prefill吞吐。'],
['上传〔E3〕','旧字节阶梯带宽估计1.804 MB/s；请求JSON字节/该速率只作敏感性估计','不是本请求精确上传耗时；JSON体长度不等于完整网络线速字节。'],
['服务端阶段','需要提供方 trace：排队、prefill起止、decode起止、缓存命中，以及token事件','仅靠更多客户端重复实验也不能保证精确拆分。自托管后可加服务端埋点。']]))
add('<p>原始非流式批没有可用的逐请求 TTFT/TPOT，不能用后来补测的值回填。stream shim 改变了线上请求模式；结构兼容验证不证明性能或策略轨迹等价。〔E1/E3〕</p></section>')
section(3,'② 两个场景：一个可完成，一个暴露长轨迹与预算问题')
add('''<p><strong>t0：两罐入篮。</strong>alphabet soup 与 tomato sauce 均进入篮子区域。<strong>t9：杯入微波炉并关门。</strong>yellow-and-white mug 入炉，同时门关闭。每场景 seed0/1、medium/none，共 8 集/批。官方目标的区域/关门谓词才是成功标准，不能把一次技能返回当整集成功。〔E1〕</p><p>下表“成功”严格指日志 finish success；现有汇总 solved 字段由 finish_status 构造，并非本次重新核验所有 states 的独立判定。t9 的提前退出与预算异常均单列。〔E1/E2〕</p>''')
for batch,eps in D['batches'].items():
 add('<h3>'+batch+'</h3>');rows=[]
 for e in eps:
  r=e['result'];status='日志成功' if r['finish_status']=='success' else ('进程异常/length' if r['process_exit_code'] else ('finish失败' if r['finish_status']=='failure' else '未finish成功'))
  rows.append([e['identity'],e['effort'],status,r['turns'],e['decisions']['tool_calls_total'],num(r['rpent_elapsed_s']),num(e['llm']['wall_sum_ms']/1000),span(e,'vla_process','pi.handler')['spans'],num(r['vla_model_load_s'])])
 add(table(['场景/seed','档位','结果','轮次','工具次数','episode s','LLM s','π chunks','独立加载 s'],rows))
add('''<h3>重复调用的规模与可优化部分</h3><p>原始 medium t9 两集共 249 次 π chunk、7 次 pick、5 次 doubled、48 次 move_to，仍 finish failure；none t9/seed0 连 π 都未调用便停下。查询次数和动作次数应分别看，back_project 与 read_text_file 不是机器人动作。缺少逐步子目标进展标注时，不能把重复调用全算无效。〔E2〕</p><p><strong>重复动作的可节省比例尚未量化。</strong>需标注每次动作前后的子目标变化，再与成功最短对照路径比较。〔E3〕</p><h3>复合技能设计目标：尚未重放验证</h3><p>如果感知已可靠、复合技能能稳定完成一个子目标：t0 可用“规划/定位 → 第一罐后检查 → 第二罐后验收”约 3 次高层决策；t9 可用“定位/检查门态 → 抓取入炉 → 关门 → 验收”约 4 次。这是<strong>新技能粒度下的设计目标</strong>，不是当前工具接口的数学下界，也不是现有日志实测。当前 t0 补测最快成功轨迹为 26 轮、805.1 s；26→3 不能直接宣称节省88.5%，因为工具与感知仍需执行。〔E2/E3〕</p><p>更可实施的首个优化实验：保持任务、seed、模型和物理设置，压缩重复历史/图像并合并独立查询；对比成功率、决策轮数、总输入tokens、LLM墙钟与去渲染后的关键路径。每项单独消融。〔E3〕</p></section>''')
section(4,'③ 每轮究竟输入什么，输出什么')
add('''<p>结构是 <strong>DeepSeek 高层决策 → RPent 工具 → π0.5 动作 / SAM3 分割 → 仿真与观测 → 下一轮 DeepSeek</strong>。这里的 SAM3 是分割服务，不是仿真器；Astra 是此前对照模型，本批模型是服务返回的 deepseek-v4.1-flash。〔E1/E2〕</p>''')
add(table(['组件','输入字段','输出与长度口径'],[
['DeepSeek','model、messages、tools、tool_choice、reasoning_effort、max_completion_tokens、stream；shim加stream_options','content / reasoning_content / tool_calls；finish_reason、usage。以返回usage统计token，字符数和JSON字节只作结构证据。'],
['messages','system/user任务；assistant历史；tool返回；可能含image_url图像块','每次发送保留的历史上下文（源码有图像历史裁剪）；prompt_tokens为该次总输入，累计总和重复计入历史。单个prompt组件没有独立官方token账单。'],
['π0.5 handler〔本批实测〕','main_images与wrist_images [1,256,256,3] uint8；states [1,8] float32；task_descriptions；extra_view_images为空','样例动作[1,5,7] float32。transform后state[1,32]；prompt张量[1,200]是padding长度，不能称200有效tokens。'],
['π prompt mask','tokenized_prompt_mask [1,200] bool','样例nonpadding_count=8；是局部技能instruction，不能用DeepSeek十万级上下文代替。'],
['SAM3','图像bytes/base64、text_prompt或point、min_score','found/score/box/mask等；本批client输入样例1024×1024，732512 bytes，文本6字符；服务时段归属为windowed。'],
['仿真与控制','7维动作chunk逐步执行；控制状态与物理步进','返回观测/状态与终止信息；render、env.step和外层primitive嵌套，不能相加。']]))
rows=[]
for e in D['batches']['流式补测']:
 rs=e['llm']['per_request'];ts=e['tokens'];v=I.get(e['episode_key'],{}).get('requests',[])
 rows.append([e['identity']+'/'+e['effort'],num(ts['prompt_tokens']),num(ts['cached_tokens']),num(ts['completion_tokens']),num(ts['reasoning_tokens']),num(max((r.get('prompt_tokens') or 0) for r in rs)),num(max((r.get('n_messages') or 0) for r in v)),num(max((r.get('n_images') or 0) for r in v)),num(max((r.get('body_bytes') or 0) for r in v))])
add(table(['episode','累计input','其中cache','累计output','其中reason','单次最大input','最大消息数','最大图像数','最大JSON bytes'],rows))
add('''<p>cached是input子集；reasoning是output明细，不能重复加总。输入tokens包含工具schema与多模态编码等，不能由文本字符数直接换算。每轮的tokens、缓存、请求字节、消息数量已分别列出，支持逐条对照；不把“长请求体”直接等同“长文本上下文”。〔E2〕</p><h3>每轮新增输入与工具返回：上下文为什么变长</h3><p>下表按出站请求顺序，与上一请求做消息hash多重集差分。新增包括新消息和改写消息；改写按整条计，字符列只统计content文本，不含工具参数、reasoning_content或schema。图像列是这些消息中的块数，不代表首次出现的独特图片；图像裁剪造成的删除另计。官方prompt增量是两次usage之差，含工具schema与多模态编码影响，不能等同于新增文字token。工具结果进入后续请求，通常由上一决策产生，重试时不一定对应紧邻前一请求，下表明确标为“新带入的历史工具结果”。</p><p class="notice"><strong>具体例子〔E2〕：</strong>medium t0/seed0 的第6次请求，官方input从31,272增到54,517 tokens（+23,245）。新带入三份文件正文：strict_hybrid_guide.md 33,429字符、pro_hybrid_guide.md 26,286字符、env_calibration.md 7,492字符；加上其他消息，共新增/改写72,144个content字符及1个图像块。工具返回中一项segment found=true，另一项报错。文字/图像/结果共同进入上下文，不能把23,245 tokens全部归给某一份文件。</p><select id="ctxselect"></select><div id="ctxsummary"></div><div id="contextrows"></div><details><summary>逐轮输入结构：role、字符数与hash（无原始正文）</summary><select id="ioselect"></select><pre id="iopre"></pre></details><p>system来自LIBERO evaluation模板local-memory分支，经CLI渲染为Agent instructions：包含角色/评测、记忆读取规则、已验证操作、runtime、goal、rules、定位、首步、workflow、超参数和输出约束11节。记忆读取规则不等于完整记忆正文预注入。首轮medium：system 32598字符、user 1193字符；none：system 32594、user 1187字符，均有17个工具。各集system hash不同，因此不能宣称不同seed/档位prompt逐字相同。π实际记录的有效prompt均为8 tokens（有π调用的四集），其余集未调用π。结构证明出站消息的组成，不证明完整prompt语义相同。为了不公开原始会话，报告保留长度与hash；本批工具schema hash可逐轮核对。π逐阶段采集使用stages同步，sync_disturbed=true，墙钟结果包含同步扰动，不能直接当无插桩生产延迟。〔E1/E2〕</p></section>''')
section(5,'按真实先后顺序看一集')
add('''<p>只显示所选一集；横轴为同机 monotonic 时钟相对秒。零点是<strong>首个被选中的 span</strong>，不是进程启动。8 集各自核验只有一个 clock domain；不同轨道有嵌套，不能把条长相加。空白表示未展示类别，不能自动解释为 GPT 时间。〔E2〕</p><select id="epselect"></select><canvas id="timeline"></canvas><p id="tlcaption" class="muted"></p><details><summary>时间轴事件表（LLM/动作/π/SAM，渲染见轨道）</summary><div id="events"></div></details></section>''')
section(6,'π很慢吗？把模型计算和动作执行分开')
rows=[]
for e in D['batches']['流式补测']:
 h=span(e,'vla_process','pi.handler');pr=span(e,'vla_process','pi.prefix');den=span(e,'vla_process','pi.denoise_core');rend=span(e,'planner_process','sim.render');sam=span(e,'planner_process','sam.client.total')
 rows.append([e['identity']+'/'+e['effort'],h['spans'],num(h['sum_ms']/1000),num(h['sum_ms']/h['spans'] if h['spans'] else None),num(rend['sum_ms']/1000),num(sam['sum_ms']/1000)])
add(table(['episode','π handler次数','π总 s','π均值 ms/chunk','sim.render累计 s','SAM client累计 s'],rows))
add('''<p>例如补测 medium t0/seed0：28 次 π handler 总计约16.98 s、约606 ms/chunk；但外层 pi0_pick 总计约94.89 s，里面还包含动作执行、仿真、观测和通信。因此“pick耗时”不能叫“π神经网络推理耗时”。模型独立加载约165.6 s另计，未塞入每次推理。〔E2〕</p><p>本表为各类span累计，<strong>不是互斥总账</strong>。本报告按你的实际部署假设不把渲染作为首要优化方向；也不直接用episode减render累计来承诺真实机器人耗时，因为观测图像、控制闭环和嵌套边界会变化。优化π的收益上限，必须按π独占关键路径占比计算。〔E3〕</p>''')
add('<details><summary>全部16集工具计数与本地分阶段耗时</summary><pre>'+esc(json.dumps([{ 'batch':b,'episode':e['identity']+'/'+e['effort'],'tools':e['decisions']['tool_calls_by_name'],'local_execution':e['local_execution']} for b,es in D['batches'].items() for e in es],ensure_ascii=False,indent=2))+'</pre></details></section>')
section(7,'可复核明细与证据边界')
add('''<p><a href="round-context.json">逐轮新增上下文与工具摘要</a> · <a href="requests.csv">两批逐请求CSV</a> · <a href="measurement-source.md">计时源码与行号</a> · <a href="scene-goals.json">场景目标谓词</a> · <a href="audited-data.json">统计输入 JSON</a> · <a href="io-evidence.json">输入输出元数据</a> · <a href="timeline-evidence.json">时间轴与源文件hash</a> · <a href="build_report.py">可复现构建脚本</a></p><p>采集源码〔E1〕：ds_http_hooks.py::_scan / _emit_transport；ds_stream_shim.py 流式重组；timing_hooks.py::_metadata_input / _metadata_output；extract_metrics.py 请求计时与finish判定。原始指标文件SHA256见audited-data.json.sources。只导出白名单元数据，未发布认证、服务地址、原始prompt或原始日志。</p><p>范围限制：每场景仅2个seed；未作随机顺序/服务负载控制，不能因果比较模型或推理档位。成功仅复核既有日志字段，未重跑环境；HTTP状态/错误分类未在旧metrics完整保留。SAM按同机时间窗归属，非严格逐请求匹配。本报告不将未知项补零。</p><h3>流式每请求明细（全部两批明细另附CSV）</h3><p>筛选集数与时间轴同步。单位明确标在列名中；缺失值为“—”。</p><div id="requests"></div></section>''')
add('<details id="proxy-audit"><summary>附录：代理计时、批次归属与完整数据</summary>'+proxy_appendix.replace('id="proxy-audit"','id="proxy-data"')+'</details>')
R=json.loads((B/'round-context.json').read_text())
J={'rounds':R,'episodes':D['batches']['流式补测'],'timeline':T,'io':I}
add('<script>const D='+json.dumps(J,ensure_ascii=False).replace('</','<\/')+';')
add('''
const sel=document.querySelector('#epselect'), ios=document.querySelector('#ioselect');for(const e of D.episodes){for(const s of [sel,ios,document.querySelector('#ctxselect')])s.add(new Option(e.identity+' / '+e.effort,e.episode_key))}
const fmt=x=>x==null?'—':Number(x).toFixed(3);function tab(h,rows){return '<div class="scroll"><table><thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function draw(){const key=sel.value,t=D.timeline[key],rows=t.rows,e=D.episodes.find(x=>x.episode_key===key);const c=document.querySelector('#timeline'),ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=270*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);const w=c.clientWidth,max=Math.max(...rows.map(r=>r.end)),left=100,right=18;const names=['LLM','动作外层','π client','SAM client','render'];const lane=n=>n==='llm.response'?0:n.startsWith('primitive')?1:n.startsWith('pi.')?2:n.startsWith('sam.')?3:4;ctx.font='12px sans-serif';ctx.fillStyle='#5c6c75';for(let j=0;j<5;j++){ctx.fillText(names[j],0,40+j*39);ctx.strokeStyle='#e1e8e8';ctx.beginPath();ctx.moveTo(left,45+j*39);ctx.lineTo(w-right,45+j*39);ctx.stroke()}for(const r of rows){ctx.fillStyle=['#087e8b','#799c9d','#264653','#a4babb','#cbd9d8'][lane(r.name)];ctx.fillRect(left+r.start/max*(w-left-right),26+lane(r.name)*39,Math.max(1,(r.end-r.start)/max*(w-left-right)),19)}for(let j=0;j<=4;j++){ctx.fillStyle='#5c6c75';ctx.fillText((max*j/4).toFixed(0)+'s',left+j/4*(w-left-right)-10,247)}document.querySelector('#tlcaption').textContent='已捕获窗口 '+max.toFixed(2)+' s；episode 日志 '+e.result.rpent_elapsed_s+' s。二者起止边界不同。';document.querySelector('#events').innerHTML=tab(['起点s','终点s','耗时s','事件'],rows.filter(r=>r.name!=='sim.render').map(r=>[fmt(r.start),fmt(r.end),fmt(r.end-r.start),r.name+(r.lid?' / '+r.lid:'')]));document.querySelector('#requests').innerHTML=tab(['请求','attempt','墙钟s','TTFT s','TPOT ms/token','thinking窗口s','input','cache','output','reason','结束'],e.llm.per_request.map(r=>[r.logical_request_id,r.attempt,fmt(r.duration_ms/1000),r.ttft_ms==null?'—':fmt(r.ttft_ms/1000),fmt(r.tpot_ms),r.thinking_visible_ms==null?'—':fmt(r.thinking_visible_ms/1000),r.prompt_tokens??'—',r.cached_tokens??'—',r.completion_tokens??'—',r.reasoning_tokens??'—',r.finish_reason??'—']));}
function context(){const key=document.querySelector('#ctxselect').value, e=D.episodes.find(e=>e.episode_key===key);let prev=null;const rows=D.rounds[key].map(x=>{const r=e.llm.per_request.find(r=>r.logical_request_id===x.lid&&r.attempt===x.attempt);const tok=r?.prompt_tokens;const delta=tok==null||prev==null?null:tok-prev;if(tok!=null)prev=tok;const incoming=x.new_or_changed_messages;const results=incoming.filter(m=>m.role==='tool');const format=m=>{let a=[m.tool_name||'tool'];if(m.file)a.push(m.file);if(m.result)for(const [k,v] of Object.entries(m.result))a.push(({'file_content_chars':'文件正文字符','size':'文件大小','bytes_written':'写入字节','terminated':'环境终止','truncated':'截断','success':'技能成功','found':'找到目标','error':'异常','chunks_used':'执行chunks','max_chunks':'chunks上限','count':'数量'}[k]||k)+'='+v);if(!m.result||!Object.keys(m.result).length)a.push('返回文本 '+m.text_chars+' 字符');return safe(a.join(' · '))};return [safe(x.lid)+' / '+x.attempt,tok??'—',delta==null?'—':(delta>=0?'+':'')+delta,x.messages,incoming.length+' / '+x.removed_or_changed_count,x.new_text_chars,x.new_images,results.map(format).join('<br>')||'—','<details><summary>新增消息组成</summary>'+incoming.map(m=>'<p>'+safe(m.role)+'：'+m.text_chars+'字符，'+m.images+'图'+(m.calls?'；调用 '+m.calls.map(c=>safe(c.name)+(c.file?'('+safe(c.file)+')':'')).join('、'):'')+'</p>').join('')+'</details>']});document.querySelector('#contextrows').innerHTML=tab(['请求/attempt','官方input tokens','较前次Δtokens','总消息','新增/改写 · 删除/改写','新增/改写content字符','新增/改写消息中的图像块','新带入的历史工具结果','展开'],rows);document.querySelector('#ctxsummary').textContent='当前 '+e.identity+' / '+e.effort+'。首轮显示初始prompt；重试若请求未变，新增为0。末轮发出的工具若无后续请求，结果不在此数据源中。'}
function safe(v){return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
document.querySelector('#ctxselect').onchange=context;
function io(){document.querySelector('#iopre').textContent=JSON.stringify(D.io[ios.value],null,2)}sel.onchange=draw;ios.onchange=io;window.onresize=draw;draw();io();context();
</script><footer><p class="muted">RPent 多模型性能分析 · 本地离线报告 · 只读审计，无新实验</p></footer></main></html>''')
(B/'RPent-DeepSeek-两场景性能审计.html').write_text('\n'.join(parts))
print('built',len(''.join(parts)))
