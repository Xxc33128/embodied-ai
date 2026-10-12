import json,gzip,hashlib
from pathlib import Path
B=Path('/data/rpent2/ds-adapter-20261009/measurements');out={}
for p in sorted(B.glob('shim-deepseek*/libero_10_swap/*/attempt-001')):
 key=str(p.relative_to(B)).removesuffix('/attempt-001');a={'requests':[],'pi':[],'sam':[]}
 for f in list(p.glob('spans-*.jsonl'))+list((B/'vla'/key/'attempt-001').glob('spans-*.jsonl')):
  for l in f.open():
   r=json.loads(l);n=r.get('name')
   if r.get('event')=='start':continue
   if n=='llm.transport':
    keys=['logical_request_id','attempt','body_bytes','wire_body_bytes','param_keys','param_model','param_max_tokens','param_reasoning_effort','param_stream','n_messages','n_images','image_approx_bytes','n_tools','tool_names','tools_sha256']
    x={k:r[k] for k in keys if k in r};x['request_params']=r.get('request_params',{});x['messages']=[{k:m[k] for k in ['role','types','text_chars','text_sha256','n_tool_calls','is_tool_result'] if k in m} for m in r.get('messages',[])];a['requests'].append(x)
    bp=B.parent/'private'/'llm-bodies'/(str(r.get('body_sha256'))+'.json.gz')
    if bp.exists():
     body=json.loads(gzip.decompress(bp.read_bytes()));x['messages']=[]
     for m in body.get('messages',[]):
      c=m.get('content'); texts=[c] if isinstance(c,str) else [z.get('text','') for z in (c or []) if isinstance(z,dict) and z.get('type')=='text']
      x['messages'].append({'role':m.get('role'),'text_chars':sum(map(len,texts)),'text_sha256':hashlib.sha256(''.join(texts).encode()).hexdigest(),'n_tool_calls':len(m.get('tool_calls') or []),'is_tool_result':bool(m.get('tool_call_id'))})
     x['tool_names']=[t.get('function',{}).get('name') for t in body.get('tools',[])]
     x['max_completion_tokens']=body.get('max_completion_tokens')

   if n in ['pi.handler','pi.input_transform','sam.client.total']:
    x={k:r[k] for k in ['name','input_meta','output_meta','sync_disturbed','measurement_mode'] if k in r}
    a['sam' if n.startswith('sam') else 'pi'].append(x)
 out[key]=a
print(json.dumps(out))
