import json,gzip,hashlib,collections
from pathlib import Path
B=Path('/data/rpent2/ds-adapter-20261009');out={}
def fingerprint(m):return hashlib.sha256(json.dumps(m,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def summarize(m,toolmap):
 c=m.get('content');blocks=c if isinstance(c,list) else []
 texts=[c] if isinstance(c,str) else [b.get('text','') for b in blocks if isinstance(b,dict) and b.get('type')=='text']
 text='\n'.join(texts);x={'role':m.get('role'),'text_chars':len(text),'images':sum(1 for b in blocks if isinstance(b,dict) and b.get('type') in ['image_url','image'])}
 calls=[]
 for t in m.get('tool_calls') or []:
  f=t.get('function',{});a=f.get('arguments','{}')
  try:a=json.loads(a) if isinstance(a,str) else a
  except Exception:a={}
  calls.append({'name':f.get('name'),'file':Path(a['path']).name if isinstance(a,dict) and isinstance(a.get('path'),str) else None,'argument_chars':len(f.get('arguments',''))})
 if calls:x['calls']=calls
 if m.get('role')=='tool':
  x['tool_name']=toolmap.get(m.get('tool_call_id'),'unknown')
  try:v=json.loads(text)
  except Exception:v=None
  if isinstance(v,dict):
   x['result_keys']=sorted(v.keys());x['result']={k:v[k] for k in ['terminated','truncated','found','success','count','size','bytes_written'] if isinstance(v.get(k),(int,float,bool))}
   if 'error' in v:x['result']['error']='tool error (message omitted)'
   if isinstance(v.get('path'),str):x['file']=Path(v['path']).name
   if isinstance(v.get('content'),str):x['result']['file_content_chars']=len(v['content'])
   r=(v.get('log') or {}).get('result',{}) if isinstance(v.get('log'),dict) else {}
   if isinstance(r,dict):x['result'].update({k:r[k] for k in ['success','chunks_used','max_chunks','steps','steps_used'] if isinstance(r.get(k),(bool,int,float))})
 return x
for p in sorted((B/'measurements').glob('shim-deepseek*/libero_10_swap/*/attempt-001')):
 rs=[]
 for f in p.glob('spans-*.jsonl'):
  for line in f.open():
   r=json.loads(line)
   if r.get('name')=='llm.transport' and r.get('event')!='start':rs.append(r)
 prev=[];items=[]
 for r in sorted(rs,key=lambda r:r['start_ns']):
  fp=B/'private/llm-bodies'/(str(r.get('body_sha256'))+'.json.gz')
  if not fp.exists():continue
  body=json.loads(gzip.decompress(fp.read_bytes()));msgs=body.get('messages',[]);counts=collections.Counter(prev);new=[];current=[];toolmap={t.get('id'):t.get('function',{}).get('name') for m in msgs for t in m.get('tool_calls',[])}
  for m in msgs:
   h=fingerprint(m);current.append(h)
   if counts[h]:counts[h]-=1
   else:new.append(summarize(m,toolmap))
  items.append({'lid':r.get('logical_request_id'),'attempt':r.get('attempt'),'messages':len(msgs),'new_or_changed_messages':new,'removed_or_changed_count':sum(counts.values()),'new_text_chars':sum(x['text_chars'] for x in new),'new_images':sum(x['images'] for x in new),'body_bytes':r.get('body_bytes')});prev=current
 out[str(p.relative_to(B/'measurements')).removesuffix('/attempt-001')]=items
print(json.dumps(out,ensure_ascii=False))
