# coding: utf-8
import json,html,csv,statistics
from pathlib import Path
from report_proxy import valid
B=Path(__file__).resolve().parent

def render(D,table):
 es=D['batches']['流式补测'];groups={};out=[]
 for e in es:
  rows=[];prev=None
  for r in e['llm']['per_request']:
   end=r.get('last_chunk_ms');tt=r.get('ttft_ms');u=r.get('completion_tokens');reason=r.get('reasoning_tokens');fr=r.get('first_reasoning_ms');targets=[r[k] for k in ['first_content_ms','first_tool_ms'] if r.get(k) is not None];boundary=min(targets) if targets else None
   window=(boundary-fr)/1000 if fr is not None and boundary is not None and boundary>=fr else (0 if reason==0 and tt is not None else None)
   post=(end-tt)/1000 if end is not None and tt is not None else None
   decode=(end-boundary)/1000 if end is not None and boundary is not None else (post if reason==0 else None)
   tok=r.get('prompt_tokens');delta=tok-prev if tok is not None and prev is not None else tok
   if tok is not None:prev=tok
   rows.append(dict(id=r['logical_request_id']+'/'+str(r['attempt']),total=r['duration_ms']/1000,ttft=tt/1000 if tt is not None else None,prep=None,headers=r.get('first_byte_ms')/1000 if r.get('first_byte_ms') is not None else None,after_headers=(tt-r['first_byte_ms'])/1000 if tt is not None and r.get('first_byte_ms') is not None else None,post=post,thinking=window,decode_window=decode,input=tok,delta=delta,cache=r.get('cached_tokens'),output=u,reason=reason,tpot=r.get('tpot_ms'),finish=r.get('finish_reason')))
  groups['流式8集 / '+e['identity']+' / '+e['effort']]=rows
 F={f['file']:f['rows'] for f in json.loads((B/'proxy-inventory.json').read_text())['files']};mapping=json.loads((B/'proxy-run-mapping.json').read_text())
 for f in mapping['files']:
  for m in f['matches']:
   if m['match']!='exact_count_tokens_time' or 'deepseek' not in m['run_evidence']['model']:continue
   lo,hi=m['line_range'];rr=F[f['file']][lo-1:hi]
   if not any(valid(r) for r in rr):continue
   rr=[r for r in rr if valid(r)]
   rows=[];prev=None
   for r in rr:
    u=r.get('usage',{});tok=u.get('prompt_tokens');n=u.get('completion_tokens');reason=u.get('completion_tokens_details',{}).get('reasoning_tokens');ok=valid(r);tt=r.get('ttft_s') if ok else None;end=r.get('duration_s');boundary=r.get('first_content_s');post=end-tt if tt is not None else None
    thinking=(boundary-tt) if ok and boundary is not None and boundary>=tt and reason is not None and reason>0 else (0 if ok and reason==0 else None)
    decode=end-boundary if ok and boundary is not None else (post if ok and reason==0 else None)
    delta=tok-prev if tok is not None and prev is not None else tok
    if tok is not None:prev=tok
    rows.append(dict(id=r['line'],total=end,ttft=tt,prep=r.get('proxy_prep_s'),headers=r.get('upload_s'),after_headers=r.get('api_wait_s'),post=post,thinking=thinking,decode_window=decode,input=tok,delta=delta,cache=u.get('prompt_cache_hit_tokens',u.get('prompt_tokens_details',{}).get('cached_tokens')),output=n,reason=reason,tpot=1000*post/(n-1) if post is not None and n and n>1 else None,finish=r.get('finish_status',r.get('status'))))
   groups['代理流式 / '+m['run']]=rows
 for name,rows in groups.items():
  for r in rows:
   n=r['output'];reason=r['reason'];post=r['post'];r['prefill']=r['ttft']
   r['allocation_tpot_ms']=1000*post/n if post is not None and n else None
   r['think_est']=post*reason/n if post is not None and reason is not None and n else None
   r['decode_est']=post*(n-reason)/n if post is not None and reason is not None and n else None
   r['nonreason_tokens']=n-reason if n is not None and reason is not None else None
   out.append(dict(episode=name,**r))
 with (B/'streaming-phases.csv').open('w',newline='') as fp:
  wr=csv.DictWriter(fp,fieldnames=list(out[0]),lineterminator='\n');wr.writeheader();wr.writerows(out)
 s='<section id="s2"><h2>开启 / 关闭思考：同场景、同seed对比</h2><p>流式8集：medium与none各4集。成功按任务日志finish success统计。〔E2〕</p>'
 rows=[]
 for effort in ['medium','none']:
  ee=[e for e in es if e['effort']==effort];success=[e for e in ee if e['result']['finish_status']=='success'];rr=[r for e in ee for r in e['llm']['per_request']]
  rows.append([effort,str(len(success))+'/4（'+str(25*len(success))+'%）',sum(e['result']['process_exit_code']!=0 for e in ee),f"{sum(e['result']['rpent_elapsed_s'] for e in ee):.1f}",f"{statistics.mean(e['result']['rpent_elapsed_s'] for e in ee):.1f}",sum(e['result']['turns'] for e in success),f"{sum(e['result']['rpent_elapsed_s'] for e in success):.1f}",f"{sum(e['llm']['wall_sum_ms']/1000 for e in success):.1f}",sum((r.get('reasoning_tokens') or 0) for r in rr)])
 s+=table(['思考档位','成功率','进程异常集数','全部4集耗时s','全部4集均值s','成功2集轮数','成功2集耗时s','成功2集LLM s','全部reasoning tokens'],rows)
 rows=[]
 for ident in ['t0_s0','t0_s1','t9_s0','t9_s1']:
  pair=[next(e for e in es if e['identity']==ident and e['effort']==eff) for eff in ['medium','none']];a,b=pair;ra,rb=a['result'],b['result'];ok=ra['finish_status']==rb['finish_status']=='success'
  rows.append([ident,ra['finish_status'] or '未成功',rb['finish_status'] or '未成功',str(ra['turns'])+' → '+str(rb['turns']),f"{ra['rpent_elapsed_s']:.1f} → {rb['rpent_elapsed_s']:.1f}",f"{(1-rb['rpent_elapsed_s']/ra['rpent_elapsed_s'])*100:.1f}%" if ok else '未完成，不作提速结论'])
 s+=table(['场景/seed','medium结果','none结果','轮数 medium→none','耗时s medium→none','关闭思考节省'],rows)
 a=sum(e['result']['rpent_elapsed_s'] for e in es if e['effort']=='medium' and e['result']['finish_status']=='success');b=sum(e['result']['rpent_elapsed_s'] for e in es if e['effort']=='none' and e['result']['finish_status']=='success')
 s+=f'<p><strong>结论：</strong>本批两档成功率均50%；在共同成功的t0两集，关闭思考耗时从{a:.1f}s降至{b:.1f}s，减少{a-b:.1f}s（{100*(a-b)/a:.1f}%），轮数63→57。t9两档均未完成；medium两集因输出预算异常退出。每档仅4集，成功率相同不证明性能等价。〔E2〕</p>'
 s+='<h3>共同成功两集：LLM时间与输出长度</h3>'
 rows=[]
 for effort in ['medium','none']:
  ee=[e for e in es if e['effort']==effort and e['result']['finish_status']=='success'];rr=[r for e in ee for r in e['llm']['per_request']]
  llm=sum(r['duration_ms'] for r in rr)/1000;tt=sum(r['ttft_ms'] for r in rr if r['ttft_ms'] is not None)/1000;post=sum(r['last_chunk_ms']-r['ttft_ms'] for r in rr if r['ttft_ms'] is not None)/1000;failed=sum(r['duration_ms'] for r in rr if r['ttft_ms'] is None)/1000
  phase_rows=[x for e in ee for x in groups['流式8集 / '+e['identity']+' / '+effort]]
  th=sum(r['think_est'] for r in phase_rows if r['think_est'] is not None);de=sum(r['decode_est'] for r in phase_rows if r['decode_est'] is not None)
  rows.append([effort,f'{llm:.3f}',f'{tt:.3f}',f'{th:.3f}',f'{de:.3f}',f'{failed:.3f}',sum(r.get('prompt_tokens') or 0 for r in rr),sum(r.get('completion_tokens') or 0 for r in rr),sum(r.get('reasoning_tokens') or 0 for r in rr)])
 s+=table(['档位','LLM总s〔实测〕','prefill代理s＝TTFT','thinking s〔估算〕','decode s〔估算〕','失败请求s〔实测〕','input tokens','output tokens','reasoning tokens'],rows)
 s+='<p>开启思考：1238.435 = 776.094（prefill代理）+ 352.550（thinking估算）+ 97.379（decode估算）+ 12.412（失败请求）秒。关闭思考：715.819 = 607.904 + 0 + 107.915秒。关闭后，prefill代理减少168.190秒，thinking估算减少352.550秒，decode估算增加10.536秒；LLM累计减少522.616秒，整集减少478.5秒。〔E2，求和/差额〕</p>'
 s+='<h3>逐请求分段：秒、tokens及估算值逐列列出</h3><select id="phaseselect"></select><p id="phase-summary"></p><div id="phase-table"></div><p><a href="streaming-phases.csv">下载全部流式逐请求分段CSV</a></p>'
 s+='<p><strong>统一估算口径：</strong>prefill代理值取实测TTFT（含网络和等待）；thinking/decode按各自token数分摊首输出后的实测时间。decode包括正文与工具参数。〔E2/E3〕</p><div class="formula">Prefill<sub>proxy</sub> = TTFT<br>τ = (T<sub>总</sub> − TTFT) / N<sub>输出</sub><br>Thinking<sub>估</sub> = N<sub>reasoning</sub> × τ<br>Decode<sub>估</sub> = (N<sub>输出</sub> − N<sub>reasoning</sub>) × τ<br>T<sub>总</sub> ≈ Prefill<sub>proxy</sub> + Thinking<sub>估</sub> + Decode<sub>估</sub></div><p class="muted">τ为本轮均摊token耗时，假设thinking与正文/工具输出速率相同；与传统TPOT的(N输出−1)分母分开列示。失败请求缺分段数据时仅保留实测总耗时。实测输出窗口、首输出前子阶段及tokens在展开表中保留。</p></section>'
 s+='<script>const PHASES='+json.dumps(groups,ensure_ascii=False).replace('</','<\/')+''';const ps=document.querySelector('#phaseselect');for(const k of Object.keys(PHASES))ps.add(new Option(k,k));function phaseDraw(){const rs=PHASES[ps.value],f=x=>x==null?'—':Number(x).toFixed(3);const table=(h,rows)=>'<div class="scroll"><table><thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(a=>'<tr>'+a.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';const h=['请求','总耗时s〔实测〕','prefill代理s＝TTFT','thinking s〔估算〕','decode s〔估算〕','分段合计s','均摊τ ms/token'];const rows=rs.map(r=>[r.id,f(r.total),f(r.prefill),f(r.think_est),f(r.decode_est),r.prefill!=null&&r.think_est!=null&&r.decode_est!=null?f(r.prefill+r.think_est+r.decode_est):'—',f(r.allocation_tpot_ms)]);document.querySelector('#phase-table').innerHTML=table(h,rows)+'<details><summary>实测窗口、首输出前子阶段与输入输出tokens</summary>'+table(['请求','thinking窗口s〔实测〕','正文/工具窗口s〔实测〕','准备s','至响应头s','响应头→首输出s','首输出后总s','TPOT ms/token','input','Δinput','cache','thinking tokens','正文/工具tokens','output','结束'],rs.map(r=>[r.id,f(r.thinking),f(r.decode_window),f(r.prep),f(r.headers),f(r.after_headers),f(r.post),f(r.tpot),r.input??'—',r.delta??'—',r.cache??'—',r.reason??'—',r.nonreason_tokens??'—',r.output??'—',r.finish??'—']))+'</details>';const sum=k=>rs.reduce((a,r)=>a+(r[k]??0),0);const count=k=>rs.filter(r=>r[k]!=null).length;document.querySelector('#phase-summary').textContent=rs.length+'条请求；总耗时 '+f(sum('total'))+'s；prefill代理 '+f(sum('prefill'))+'s；thinking估算 '+f(sum('think_est'))+'s；decode估算 '+f(sum('decode_est'))+'s。完整分段 '+count('decode_est')+'/'+rs.length+'条。';}ps.onchange=phaseDraw;phaseDraw();</script>'''
 return s
