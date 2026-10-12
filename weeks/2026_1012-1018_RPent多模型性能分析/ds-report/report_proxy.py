# coding: utf-8
"""Render separately scoped proxy audit; never merge with ds-adapter episodes."""
import json, html, statistics, csv
from pathlib import Path
B=Path(__file__).resolve().parent

def valid(r):
 return (r.get('converted_stream') is True or r.get('client_mode')=='nonstream->stream') and isinstance(r.get('ttft_s'),(int,float)) and r.get('duration_s',0)>r['ttft_s']>=0 and r.get('status')==200

def metrics(r):
 ok=valid(r); u=r.get('usage',{}); n=u.get('completion_tokens',0)
 window= r.get('first_content_s',0)-r.get('ttft_s',0) if ok and u.get('completion_tokens_details',{}).get('reasoning_tokens',0)>0 and r.get('first_content_s',0)>r['ttft_s'] else None
 return {'valid_stream_ttft':ok,'tpot_mean_proxy_ms':1000*(r['duration_s']-r['ttft_s'])/(n-1) if ok and n>1 else None,'thinking_visible_proxy_s':window}

def render(table):
 data=json.loads((B/'proxy-inventory.json').read_text()); files=data['files']
 mapping=json.loads((B/'proxy-run-mapping.json').read_text()) if (B/'proxy-run-mapping.json').exists() else None
 fnum=lambda x:'—' if x is None else f'{x:,.3f}'
 esc=lambda x:html.escape(str(x))
 allrows=[dict(file=f['file'],**r,**metrics(r)) for f in files for r in f['rows']]
 with (B/'proxy-requests.csv').open('w',newline='') as fp:
  cols=['file','line','status','client_mode','converted_stream','duration_s','ttft_s','valid_stream_ttft','first_content_s','thinking_visible_proxy_s','tpot_mean_proxy_ms','tpot_ms_p50','tpot_ms_p90','n_chunks','proxy_prep_s','upload_s','api_wait_s','gen_s','prompt_tokens','completion_tokens','reasoning_tokens']
  w=csv.DictWriter(fp,fieldnames=cols);w.writeheader()
  for r in allrows:
   row={k:r.get(k) for k in cols};u=r.get('usage',{});row.update(prompt_tokens=u.get('prompt_tokens'),completion_tokens=u.get('completion_tokens'),reasoning_tokens=u.get('completion_tokens_details',{}).get('reasoning_tokens'));w.writerow(row)
 nvalid=sum(r['valid_stream_ttft'] for r in allrows); nw=sum(r['thinking_visible_proxy_s'] is not None for r in allrows)
 s='<section id="proxy-audit"><div class="eyebrow">新增核对 / 25 份代理日志</div><h2>确实有更细的计时；先纠正字段名，再按批次使用</h2>'
 s+=f'<p class="lead">补核的25份日志共有 {len(allrows)} 条记录，其中 {nvalid} 条符合本报告的流式首输出筛选（DeepSeek 287条、Qwen 68条），{nw} 条可计算首输出到正文/工具的 thinking 可见代理窗口。这些是另一些实验与回放记录（含Qwen），<strong>不能合并进原8+8集，也不能把记录数当成独立实验数</strong>。〔E1/E2〕</p>'
 s+='<p>此前“仍不可识别”的表述过于笼统：已经记录的客户端阶段必须补上。准确结论是：<strong>可测客户端等待与可见输出窗口；目前导出的日志没有服务端 prefill、完整 thinking 的独立起止事件。</strong>拿到更多客户端时间戳，不等于拿到了服务端计算时间。〔E1/E2〕</p>'
 s+=table(['指标','本次实际找到什么','报告采用口径'],[
 ['prefill','请求准备、连接到响应头、响应头到首输出；另有TTFT拟合','这些客户端区间为实测；纯prefill未单独实测。input/TTFT不能称prefill吞吐。拟合及速率反推为E3估算。'],
 ['thinking','reasoning_tokens及部分first_content_s、ttft_s','两时间戳之差为客户端可见代理窗口；first_content可能是tool delta。未保存逐token/末reason事件，不等于完整服务端thinking。'],
 ['TTFT',f'{nvalid}条有明确stream转换标记且0≤TTFT<duration、HTTP200','采用首非空输出delta口径；其余保留原值但不放入有效TTFT统计。非流式读完整响应后打点的TTFT应改称响应完成耗时。'],
 ['TPOT','原tpot_ms_p50/p90/max与新算均值是两种东西','原字段为相邻reason/content数据块间隔，单位ms/chunk间隔；本次另算(duration−TTFT)/(completion_tokens−1)，单位ms/token，标记客户端均值proxy。'],
 ['upload_s','计时止于urlopen返回响应头','包括连接、请求发送和等待响应头；不是纯上传。Files API图片上传可能在此前准备阶段。不能从TTFT扣掉它后宣称剩余全是prefill或排队。']])
 s+='<p class="muted">证据：当前采集源码行号与hash见 <a href="proxy-source-audit.md">代理计时审计</a>（E1）；各原日志hash及白名单字段见 <a href="proxy-inventory.json">代理日志清单</a>（E2）。当前源码未与每次历史运行的代码hash绑定，因此源代码用于解释字段，历史版本不确定性仍保留。筛选是保守可用口径，不声称其余记录都无效。</p>'
 s+='<p><strong>归属核验：</strong>450条与25个完整运行区块的请求数、input/output总量和时间窗一致；32条仅候选关联、2条未归属。repro2/3/4属于Qwen，已从DeepSeek性能结论中分离。HTTP200或transcript的success不是本次重新验收环境终态。〔E2〕</p>'
 s+='<h3>性能模型：用同一份15请求日志逐层算清楚</h3><p>以下仅用 llm_t2par.jsonl，对应 ds_t2_par：libero_object_swap/task2/seed0、日志success、episode 205.3 s。全15条为converted_stream=true；不混入其他任务。数字是请求区间累计，不是多请求并行墙钟。每个请求内四个客户端阶段互不重叠；不同请求是否重叠未由此证明；rounding保留毫秒四舍五入误差。〔E2，统计派生〕</p>'
 rs=next(f['rows'] for f in files if f['file']=='llm_t2par.jsonl')
 sums={k:sum(r[k] for r in rs) for k in ['duration_s','ttft_s','gen_s','proxy_prep_s','upload_s','api_wait_s']}
 dur=sums['duration_s'];tt=sums['ttft_s'];gen=sums['gen_s'];prep=sums['proxy_prep_s'];head=sums['upload_s'];wait=sums['api_wait_s'];rounding=dur-prep-head-wait-gen
 s+=f'<div class="formula">T<sub>episode</sub> = T<sub>LLM</sub> + T<sub>其余账面差额</sub><br><strong>205.300 = {dur:.3f} + {205.3-dur:.3f} s</strong><br>LLM累计/episode比值 {100*dur/205.3:.2f}%；差额不是独立工具时长，也不证明时间占用互斥。<br>T<sub>LLM</sub> = Σ(TTFT + T<sub>首输出→结束</sub>) + ε<br><strong>{dur:.3f} = {tt:.3f} + {gen:.3f} − 0.001 s</strong><br>ΣTTFT = T<sub>代理准备</sub> + T<sub>连接/发送/等响应头</sub> + T<sub>响应头→首输出</sub> + ε<br><strong>{tt:.3f} = {prep:.3f} + {head:.3f} + {wait:.3f} + {tt-prep-head-wait:.3f} s</strong></div>'
 rows=[]
 for name,t,meaning,factors,direction in [
 ('代理准备',prep,'接收请求、解析/改写、图像处理等，具体以采集版本为准','请求JSON大小、图片数、Files API上传/缓存、序列化','减少重复base64；缓存已上传图片；避免反复编解码'),
 ('连接/发送/等响应头',head,'urlopen调用至响应头返回','连接复用、RTT、发送字节、上游何时发送响应头','复用连接；压缩请求；单独测网络，不能只靠带宽估算'),
 ('响应头→首输出',wait,'响应头返回至首非空delta','响应头刷新策略、服务排队/处理、网络缓冲','需要服务端trace区分；固定输入做负载与缓存对照'),
 ('首输出→结束',gen,'reasoning、正文、工具参数输出及尾部接收','输出tokens、生成速率、chunk缓冲、网络；thinking长度','优先验证缩短reasoning/工具参数和减轮次，约束成功率')]:
  rows.append([name,fnum(t)+' s',f'{100*t/dur:.2f}%',meaning,factors,direction,f'{t*.2:.3f} s / {100*.2*t/dur:.2f}%'])
 s+=table(['项','累计耗时','占LLM','边界','影响因素','优化方向','此项降20%可省〔E3〕'],rows)
 s+=f'<p><strong>本样本的优先级：</strong>首输出后的区间占LLM {100*gen/dur:.2f}%，远大于响应头后的首输出等待。若该输出区间缩短20%，LLM累计省 {gen*.2:.3f} s；若整个TTFT缩短20%，省 {tt*.2:.3f} s。若保持相同轨迹及成功结果，且所省时间均位于串行关键路径，换成整集205.3 s分母，条件收益分别为8.65%和2.12%；否则整集收益需按区间并集重算。两者分别是条件敏感性，不是已测优化效果。不能把这个比例套到原t0四集的33.15%首输出占整集结论上：任务、代理和分母不同。</p>'
 windows=[metrics(r)['thinking_visible_proxy_s'] for r in rs];ws=sum(x for x in windows if x is not None)
 s+=f'<p>其中 {sum(x is not None for x in windows)}/15 请求有可用thinking代理窗口，累计 <strong>{ws:.3f} s</strong>。这段已经包含在首输出后的 {gen:.3f} s 中，不能再相加。剩余差额 {gen-ws:.3f} s 包括正文/工具输出、协议尾部及不满足窗口筛选的区间，不能全叫decode。把reasoning_tokens乘平均TPOT得到的thinking时间属于均匀速率假设估算，本报告不拿它替代这段观测。〔E2/E3〕</p>'
 s+='<div class="formula">T<sub>episode</sub> = Σ T<sub>LLM</sub> + T<sub>执行与编排差额</sub><br>ΔT = Σ<sub>保留请求</sub>(α<sub>准备</sub>P + α<sub>头</sub>H + α<sub>首</sub>W + α<sub>输出</sub>G)<br>　　 + Σ<sub>被省请求</sub>T<sub>LLM</sub> + ΔT<sub>经验证可省的执行</sub></div><p>α表示该阶段实际能缩短的比例。减轮次与阶段加速只对各自请求计一次；最后一项须有可删除动作证据，未证明时不填数。必要的物理动作和观测不会因删除一次LLM请求自动消失。渲染按部署假设不列为优化方向；这里只给当前日志分母，不推造无渲染实机总时间。〔E3〕</p>'
 s+='<h3>批次归属：13轮不是参考报告的11轮</h3><p>现存 llm_files_api.jsonl 的13条与 ds_final_v2 的请求数、时间窗和776729 input /19755 output tokens相符，episode 205.9 s；参考报告对应 ds_files_api 的11轮、500355 /10722 tokens。不能用同名日志回填参考报告。现存13条TTFT均等于duration，故只使用其<strong>LLM完成耗时108.415 s</strong>，余下97.485 s为episode账面差额，不补造TTFT/TPOT。〔E2〕</p>'
 s+='<details><summary>25份日志的完整指标覆盖与统计（逐请求CSV另附）</summary><p>每行按文件统计；允许同一实验有多条采集链，因此不跨文件相加为实验总数。窗口缺失保留缺失，不当0。P50是请求均值proxy的中位数，并非所有token间隔的中位数。</p>'
 rows=[]
 for f in files:
  rr=f['rows'];good=[r for r in rr if valid(r)];tp=[metrics(r)['tpot_mean_proxy_ms'] for r in good if metrics(r)['tpot_mean_proxy_ms'] is not None];win=[metrics(r)['thinking_visible_proxy_s'] for r in good if metrics(r)['thinking_visible_proxy_s'] is not None]
  rows.append([esc(f['file']),esc(', '.join(sorted({str(r.get('model','未记')) for r in rr}))),len(rr),len(good),fnum(sum(r['duration_s'] for r in rr)) if all('duration_s' in r for r in rr) else '未完整记录',fnum(sum(r['ttft_s'] for r in good)) if good else '—',fnum(statistics.median(tp)) if tp else '—',str(len(win))+' / '+fnum(sum(win)) if win else '0 / —',sum(r.get('usage',{}).get('prompt_tokens',0) for r in rr),sum(r.get('usage',{}).get('completion_tokens',0) for r in rr)])
 s+=table(['文件','日志model','记录','流式TTFT覆盖','响应累计s','可用TTFT累计s','均值TPOT的P50 ms/token','thinking窗口n / Σs','累计input','累计output'],rows)+'</details>'
 s+='<details><summary>逐批对应、耗时、成功状态与无法归属项</summary><p>精确匹配要求连续行范围的请求数、两项token总量及时间窗一致；候选项不进入精确匹配总数。详细时间窗、模型和证据hash见JSON。</p>'
 rows=[]
 for f in mapping['files']:
  for m in f['matches']:
   e=m['run_evidence']; rr=next(z['rows'] for z in files if z['file']==f['file']);lo,hi=m['line_range'];rr=rr[lo-1:hi]
   llm=sum(r['duration_s'] for r in rr) if all('duration_s' in r for r in rr) else None
   rows.append([esc(f['file'])+f' L{lo}–{hi}',esc(m['run']), '精确' if m['match']=='exact_count_tokens_time' else '仅候选',esc(e['suite'])+f" t{e['task']}/s{e['seed']}",esc(e.get('finish_status')),e.get('turns'),fnum(e.get('elapsed_s')),fnum(llm),str(m['proxy_input_tokens'])+' / '+str(m['proxy_output_tokens'])])
  if not f['matches']:rows.append([esc(f['file']),'未归属','未归属','—','—','—','—','—','—'])
 s+=table(['文件/行','运行','匹配','场景','finish','轮数','episode s','LLM记录s','input/output'],rows)
 s+='<p>phase文件第1行未归属，表中匹配仅覆盖第2–16行。official1条404、repro4的23条、v6的8条仅候选，缺完整usage不能硬对账。</p></details>'
 s+='<details><summary>其余已记录字段：phase子阶段、缓存、图像与请求体</summary><p>以下保留原字段名与样本数，便于发现采集覆盖。sum是字段累计，不代表彼此互斥，不能横向全部相加。尤其download_first_tok在当前源码实际覆盖首响应字节到结束，并非首token下载；full_phase未保存完整duration或明确流模式，不能强行构造纯prefill/完整thinking。</p>'
 rows=[]
 for f in files:
  rr=f['rows']; keys=sorted({k for r in rr for k in r.get('phases',{})})
  fields={k:[r['phases'][k] for r in rr if k in r.get('phases',{})] for k in keys}
  for k in ['body_bytes','fwd_body_bytes','fwd_bytes','file_new_uploads','file_new','file_replaced','stripped_images','obs_dropped','thinking_tok','cached_tok','n_chunks']:
   vals=[r[k] for r in rr if isinstance(r.get(k),(int,float))]
   if vals:fields[k]=vals
  rows.append([esc(f['file']),'<br>'.join(esc(k)+f': n={len(v)}, Σ={sum(v):,.3f}' for k,v in fields.items()) or '—'])
 s+=table(['文件','字段覆盖与累计（_s为秒，bytes为字节，tok为token，其余计数）'],rows)+'</details>'
 s+='<p><a href="proxy-requests.csv">全部484条代理记录与派生指标CSV</a> · <a href="proxy-run-mapping.json">批次匹配证据</a> · <a href="proxy-inventory.json">字段与原文件hash</a>。另有原8+8集的484条响应记录，两个484恰好相同，属于不同来源，不能据此视为重复或互相替代。</p></section>'
 return s
