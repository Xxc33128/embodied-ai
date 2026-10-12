"""Evidence-backed opening: all accounting values derived from audited data."""
def render(D, table):
 eps=[e for e in D['batches']['流式补测'] if e['result']['finish_status']=='success']
 rs=[r for e in eps for r in e['llm']['per_request']]
 total=sum(e['result']['rpent_elapsed_s'] for e in eps)
 llm=sum(e['llm']['wall_sum_ms']/1000 for e in eps)
 tt=sum(r['ttft_ms']/1000 for r in rs if r['ttft_ms'] is not None)
 output=sum((r['last_chunk_ms']-r['ttft_ms'])/1000 for r in rs if r['ttft_ms'] is not None)
 missing=sum(r['duration_ms']/1000 for r in rs if r['ttft_ms'] is None)
 n=sum(len({r['logical_request_id'] for r in e['llm']['per_request']}) for e in eps)
 other=total-llm
 assert abs(llm-tt-output-missing)<.02
 def f(x):return f'{x:,.3f}'
 def pc(x):return f'{x/total*100:.2f}%'
 def block(label,eq,explain):return f'<h3>{label}</h3><div class="formula">{eq}</div><p class="muted">{explain}</p>'
 s='<section id="s1"><div class="eyebrow">01 / 结论与优化预算</div><h2>优先减少等待与重复决策；π单次推理不是主要耗时</h2>'
 s+=f'<p class="lead">成功的四集共 {total:,.1f} 秒，其中等待首输出 {tt:,.1f} 秒（{pc(tt)}），后续流式输出 {output:,.1f} 秒（{pc(output)}）。若保持轨迹不变、把这两段都缩短20%，四集合计预计省 <strong>{(tt+output)*.2:.1f} 秒（{pc((tt+output)*.2)}）</strong>。这是敏感性估算，尚未实跑验证。</p>'
 s+='<p>主账只用流式补测中日志成功的 t0/seed0、seed1 × medium、none 四集，避免把提前失败算成提速。单位均为四集累计工作量，不是并行作业墙钟。实测来源为E2；求和/差额标为统计派生；优化预算为E3。</p>'
 s+=block('第一层：总时间 = LLM响应 + 其他执行与编排',f'T<sub>episode</sub> = T<sub>LLM记录</sub> + T<sub>其余</sub><br><strong>{f(total)} s = {f(llm)} s + {f(other)} s</strong><br>100% = {pc(llm)} + {pc(other)}','左侧是episode日志汇总；LLM是响应span累计；其余=总时间−LLM，包含工具、观测、通信、SDK与收尾，是账面差额，不是独立测得的“工具总时间”。')
 s+=block('第二层：LLM响应 = 等首输出 + 后续输出段 + 缺测失败请求',f'T<sub>LLM记录</sub> = Σ TTFT + Σ T<sub>首delta→末chunk</sub> + T<sub>缺测/失败</sub><br><strong>{f(llm)} = {f(tt)} + {f(output)} + {f(missing)} s</strong><br>占整集：{pc(llm)} = {pc(tt)} + {pc(output)} + {pc(missing)}','这些区间来自客户端时间戳。TTFT包括传输、排队、输入处理和首输出前计算；后续段含reasoning、正文、工具参数及协议尾部，均不能直接命名为纯prefill或纯decode。')
 s+=block('第三层：每次请求怎样受上下文与输出长度影响','TTFT = 上传/网络 + 服务等待 + 输入处理 + 首输出前计算<br>T<sub>后续输出,i</sub> = (N<sub>completion,i</sub> − 1) × TPOT<sub>proxy,i</sub><br>Σ T<sub>后续输出,i</sub> = '+f(output)+' s','第二层可代入实测；本层TTFT内部四项没有独立时间戳，不能硬拆数字。输出段公式按每个请求计算后求和，不能用全体中位数相乘。reasoning tokens已包含在completion中。')
 s+=block('第四层：轮数是杠杆，但不能把工具成本一起删掉',f'{n} 次逻辑调用（{len(rs)} 个响应attempt）<br>T<sub>episode</sub> = {n} × ({llm/n:.3f} s 响应 + {other/n:.3f} s 其余摊销)<br>= {f(total)} s','这是当前样本的平均摊销，不是每轮固定成本。合并查询可能减少LLM调用，却未必减少机器人动作；因此下面“减轮次”只扣云端响应时间。')
 s+='<h3>耗时、占比、改什么、预计省多少</h3>'
 rows=[['首输出等待',f(tt)+' s',pc(tt),'实测时间戳求和','减少重复历史/图像传输；固定输入测缓存与负载','TTFT降20%：省 '+f(tt*.2)+' s（'+pc(tt*.2)+'）'],['后续流式输出',f(output)+' s',pc(output),'实测时间戳求和','缩短不必要输出；验证thinking档位与成功率','该段降20%：省 '+f(output*.2)+' s（'+pc(output*.2)+'）'],['失败attempt',f(missing)+' s',pc(missing),'实测；1次无TTFT','定位适配/连接错误，减少同类重试','若该次完全避免：上限 '+f(missing)+' s'],['执行与编排差额',f(other)+' s',pc(other),'统计差额，非独立工具计时','审查动作与查询路径；分开观测、物理、RPC','未归因到可删动作，不承诺节省值']]
 s+=table(['项目','当前耗时','占整集','数据性质','优化方向','条件预算〔E3〕'],rows)
 saved=sum(e['llm']['wall_sum_ms']/1000/len({r['logical_request_id'] for r in e['llm']['per_request']})*5 for e in eps)
 s+=f'<p><strong>减轮次预算：</strong>每集减少5次同等平均响应成本的调用，四集共少20次，预计省 {saved:.1f} s（{pc(saved)}）。必要工具照常执行。它和上表阶段提速作用于同一批请求，不能直接相加；组合收益须重新计算。</p>'
 s+='<p class="notice">按实际部署不渲染的假设，渲染不列为本次优化目标。上述占比仍以当前仿真episode为分母；不可把它当成去渲染后的占比。π与仿真的嵌套统计见第6节。</p>'
 s+='<details><summary>另外两条结论：关闭思考的收益与失败样本</summary><p>原始t0/seed0关闭思考后轮次26→46，总时间增加43.1%；seed1减少20.4%。补测两个成功配对分别减少30.4%和2.9%，不能承诺稳定提速。两批共同成功四集合计4170.0→4174.9 s，几乎不变。补测medium t9两集在8192输出预算处异常退出，应先修预算处理，再验证性能。</p></details></section>'
 start=s.index('<h3>耗时、占比');end=s.index('<details>',start);at=s.index('<h3>第一层')
 s=s[:at]+s[start:end]+s[at:start]+s[end:]
 return s
