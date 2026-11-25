import re,hashlib,json,pathlib
import backup
Q=rb'\\*"';W=rb'(?:\s|\\+[nrt])*'
HEAD=Q+rb'type'+Q+W+rb':'+W+Q+rb'image'+Q+W+rb','+W+Q+rb'data'+Q+W+rb':'+W
IM=re.compile(rb'(?P<prefix>'+HEAD+rb'(?P<quote>'+Q+rb'))(?P<data>.*?)(?P=quote)(?='+W+rb','+W+Q+rb'mimeType'+Q+rb')',re.DOTALL)
MARKER=re.compile(rb'\[OMITTED_MCP_IMAGE stage_sha256=[0-9a-f]{64} original_chars=[0-9]+\]')
def clean(data):
 count=0;chars=0
 def callback(m):
  nonlocal count,chars
  raw=m.group('data');count+=1;chars+=len(raw)
  original_chars=len(raw)
  for old in re.finditer(rb'\[OMITTED_IMAGE sha256=[0-9a-f]{64} chars=([0-9]+)\]',raw):original_chars+=int(old.group(1))-len(old.group(0))
  return m.group('prefix')+b'[OMITTED_MCP_IMAGE stage_sha256='+hashlib.sha256(raw).hexdigest().encode()+b' original_chars='+str(original_chars).encode()+b']'+m.group('quote')
 new=IM.sub(callback,data)
 expected=IM.sub(lambda m:m.group('prefix')+m.group('quote'),data)
 if MARKER.sub(b'',new)!=expected:raise RuntimeError('Non-image MCP data changed')
 return new,count,chars,hashlib.sha256(expected).hexdigest()
def run(stage,name='codex-eval-20260929_远程原始记录',dest=None):
 P=pathlib.Path;stage=P(stage);root=stage/'stat-snapshots'/name;mp=root/'_SOURCE_PROVENANCE.json';m=json.loads(mp.read_text());changed=0
 for e in m['original_files']:
  if e['original_sample_retained'] or not e['path'].endswith(('.json','.jsonl','.txt','.log','.out')):continue
  p=root/e['path'];raw=p.read_bytes();new,count,chars,proof=clean(raw)
  if not count:continue
  previous=e['archived_sha256'];p.write_bytes(new);e['image_payloads_omitted']+=count;e['base64_chars_omitted']+=chars;e['archived_sha256']=hashlib.sha256(new).hexdigest();e['archived_bytes']=len(new);e['additional_mcp_image_filter']={'before_sha256':previous,'after_sha256':e['archived_sha256'],'image_fields_omitted':count,'all_other_bytes_unchanged':True,'retained_bytes_sha256':proof};changed+=1
 m['summary']['archived_bytes']=sum(e['archived_bytes'] for e in m['original_files']);m['summary']['image_payloads_omitted']=sum(e['image_payloads_omitted'] for e in m['original_files']);m['summary']['base64_chars_omitted']=sum(e['base64_chars_omitted'] for e in m['original_files']);m['summary']['mcp_typed_image_files_filtered']=changed;backup.writejson(mp,m)
 task={'name':name,'dest':str(dest if dest is not None else stage/'packages-stat'),'roots':[{'src':str(p),'prefix':p.name} for p in sorted(root.iterdir())],'notes':['Statistical records and necessary original samples; both MIME/base64 and typed MCP image payloads omitted; every other byte verified unchanged.']}
 backup.bundle(task,task['dest']);backup.writejson(P(task['dest'])/(name+'.source-provenance.json'),m);backup.log('REFINED MCP READY '+str(changed))
if __name__=='__main__':run(__import__('sys').argv[1])
