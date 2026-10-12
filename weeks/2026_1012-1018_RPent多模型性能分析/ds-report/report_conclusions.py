# coding: utf-8
"""Concise conclusions calculated from the two separately scoped datasets."""
import json, statistics
from pathlib import Path
from report_proxy import metrics
B=Path(__file__).resolve().parent

def render(table):
 files=json.loads((B/'proxy-inventory.json').read_text())['files']
 rs=next(f['rows'] for f in files if f['file']=='llm_t2par.jsonl')
 sums={k:sum(r[k] for r in rs) for k in ['duration_s','ttft_s','gen_s','proxy_prep_s','upload_s','api_wait_s']}
 d,tt,g,p,h,w=[sums[k] for k in ['duration_s','ttft_s','gen_s','proxy_prep_s','upload_s','api_wait_s']]
 wins=[metrics(r)['thinking_visible_proxy_s'] for r in rs];thinking=sum(x for x in wins if x is not None)
 tp=statistics.median(metrics(r)['tpot_mean_proxy_ms'] for r in rs)
 old=json.loads((B/'audited-data.json').read_text());eps=[e for e in old['batches']['流式补测'] if e['result']['finish_status']=='success']
 oldd=sum(e['llm']['wall_sum_ms']/1000 for e in eps);oldt=sum(r['ttft_ms']/1000 for e in eps for r in e['llm']['per_request'] if r['ttft_ms'] is not None)
 saved=sum(e['llm']['wall_sum_ms']/1000/len({r['logical_request_id'] for r in e['llm']['per_request']})*5 for e in eps)
 s='<section id="conclusions"><div class="eyebrow">结论 / 数据与优化收益</div><h2>先判断时间花在等待还是输出，再选优化方向</h2>'
 s+='<p>主例：DeepSeek，object_swap task2 / seed0，15次请求，任务日志成功，整集205.3秒。对照：流式实验中成功的4集t0，合计4174.9秒。两组分别统计。〔E2〕</p>'
 s+=table(['结论','数据','优化方向与收益'],[
 ['1. 新样本主要耗在首输出之后',f'LLM累计{d:.3f}s；首输出后{g:.3f}s，占{100*g/d:.1f}%；TTFT合计{tt:.3f}s，占{100*tt/d:.1f}%。',f'优先缩短不必要的reasoning、正文和工具参数输出。该段缩短20%，请求累计省{g*.2:.3f}s；整个TTFT缩短20%只省{tt*.2:.3f}s。〔E3〕'],
 ['2. 上下文增长来自指南、工具结果和图像','流式t0/seed0 medium第6次逻辑请求：input从31,272增至54,517 tokens；新带入三份指南共67,207字符及1个图像块。','先精简重复指南与历史输入，再验证TTFT下降幅度。单次token增量不是指南独占token数，现有数据未测出该优化的节省值。'],
 ['3. 减少重复决策有明确的时间预算',f'流式成功4集共有120次逻辑请求，LLM累计{oldd:.1f}s。',f'每集少5次同等平均成本的调用，4集估计省{saved:.1f}s，约整集7.5%。这是减轮数预算，不是已经证明存在20次无效调用。〔E3〕'],
 ['4. 不同批次的瓶颈不同',f'流式成功4集TTFT合计{oldt:.1f}s，占LLM {100*oldt/oldd:.1f}%；新15请求样本则为{100*tt/d:.1f}%。','四集组优先检查长上下文、重复图像、缓存与服务等待；新样本优先压缩输出。现有数据不能把两组差异归因于某一项改动。']])
 s+='<p><strong>思考开关：</strong>流式8集中，两档成功率均为2/4（50%）。共同成功的t0两集关闭思考后2326.7→1848.2s，减少478.5s（20.6%），轮数63→57。详细对比及每次请求的prefill/thinking/decode计时列见下一节。〔E2〕</p>'
 s+='<h3>耗时公式：每一项各占多少</h3>'
 s+=f'<div class="formula">T<sub>LLM累计</sub> ≈ T<sub>准备</sub> + T<sub>请求至响应头</sub> + T<sub>响应头至首输出</sub> + T<sub>首输出至结束</sub><br><strong>{d:.3f}s ≈ {p:.3f}s + {h:.3f}s + {w:.3f}s + {g:.3f}s</strong></div>'
 s+=table(['项','占LLM累计','受什么影响','此项缩短20%可省〔E3〕'],[
 ['准备 '+f'{p:.3f}s',f'{100*p/d:.2f}%','JSON/图像处理、文件上传与缓存',f'{p*.2:.3f}s'],
 ['请求至响应头 '+f'{h:.3f}s',f'{100*h/d:.2f}%','连接复用、请求大小、网络与服务端响应头等待',f'{h*.2:.3f}s'],
 ['响应头至首输出 '+f'{w:.3f}s',f'{100*w/d:.2f}%','服务处理、排队、网络缓冲',f'{w*.2:.3f}s'],
 ['首输出至结束 '+f'{g:.3f}s',f'{100*g/d:.2f}%','输出长度、生成速率、reasoning长度与接收开销',f'{g*.2:.3f}s']])
 s+='<p class="muted">时间为客户端实测区间的累计；分项舍入相差0.002s。thinking窗口包含在输出段内。节省值是保持轨迹与成功结果不变的估算；只有落在关键路径上的节省才会等量缩短整集。渲染不列为优化目标。〔E2/E3〕</p>'
 s+='<h3>四个推理指标</h3>'
 s+=table(['指标','该15请求样本','数据性质'],[
 ['TTFT',f'中位数{statistics.median(r["ttft_s"] for r in rs):.3f}s；累计{tt:.3f}s','客户端首非空输出等待，实测'],
 ['TPOT',f'每请求均值的中位数{tp:.3f}ms/token','由(响应耗时−TTFT)/(输出tokens−1)派生；不是逐token时钟'],
 ['thinking',f'14条可见窗口，累计{thinking:.3f}s','首输出→首正文/工具的客户端窗口；非完整服务端计算时间'],
 ['prefill','无独立实测值','TTFT包含多种等待；回归或速率反推只能标为估算']])
 return s+'</section>'
