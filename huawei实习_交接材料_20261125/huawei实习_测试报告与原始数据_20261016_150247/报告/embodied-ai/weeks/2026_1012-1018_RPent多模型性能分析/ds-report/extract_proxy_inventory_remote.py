import json,hashlib,re,datetime
from pathlib import Path
B=Path('/data/rpent2');out={'schema_version':1,'files':[]}
metrics=['ttft_s','first_content_s','duration_s','gen_s','upload_s','api_wait_s','proxy_prep_s','ttft_upstream_s','tpot_ms_p50','tpot_ms_p90','tpot_ms_max','n_chunks','body_bytes','fwd_body_bytes','fwd_bytes','thinking_tok','cached_tok','thinking_chars','file_new_uploads','file_replaced','file_new','stripped_images','obs_dropped']
def texts(c):
 if isinstance(c,str):return c
 if isinstance(c,list):return '\n'.join(b.get('text','') for b in c if isinstance(b,dict) and b.get('type')=='text')
 return ''
for p in sorted(B.glob('llm_*.jsonl')):
 data=p.read_bytes();rec={'file':p.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':[],'parse_errors':0}
 for idx,line in enumerate(data.splitlines(),1):
  try:r=json.loads(line)
  except Exception:rec['parse_errors']+=1;continue
  x={'line':idx};x.update({k:r[k] for k in metrics if isinstance(r.get(k),(int,float,bool))})
  for k in ['status','stream','converted_stream','client_mode','finish','model','ts']:
   v=r.get(k)
   if isinstance(v,(int,float,bool)) or (isinstance(v,str) and len(v)<90 and not re.search(r'https?://|Bearer|sk-',v)):x[k]=v
  for k in ['phases','_phases']:
   if isinstance(r.get(k),dict):x[k]={a:b for a,b in r[k].items() if b is None or isinstance(b,(int,float))}
  u=r.get('usage') or {};x['usage']={k:u[k] for k in ['prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'] if isinstance(u.get(k),(int,float))}
  for k in ['prompt_tokens_details','completion_tokens_details']:
   if isinstance(u.get(k),dict):x['usage'][k]={a:b for a,b in u[k].items() if isinstance(b,(int,float))}
  msgs=r.get('messages') or []
  if isinstance(msgs,list):
   x['message_count']=len(msgs);x['logged_messages_sha256']=hashlib.sha256(json.dumps(msgs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
   user='\n'.join(texts(m.get('content')) for m in msgs if isinstance(m,dict) and m.get('role')=='user')
   ident={}
   for k,pat in [('suite',r'(?m)^\s*-?\s*suite:\s*([A-Za-z0-9_]+)'),('task',r'(?m)^\s*-?\s*task:\s*(\d+)'),('seed',r'(?m)^\s*-?\s*seed:\s*(\d+)')]:
    a=re.search(pat,user)
    if a:ident[k]=a.group(1)
   if ident:x['identity']=ident
  calls=r.get('tool_calls') or [];x['tool_names']=[]
  for c in calls:
   if not isinstance(c,dict):continue
   f=c.get('function',c);name=f.get('name');x['tool_names'].append(name)
   if name=='finish':
    a=f.get('arguments',{})
    if isinstance(a,str):
     try:a=json.loads(a)
     except:a={}
    if isinstance(a,dict) and a.get('status') in ['success','failure']:x['finish_status']=a['status']
  rec['rows'].append(x)
 out['files'].append(rec)
print(json.dumps(out,ensure_ascii=False))
