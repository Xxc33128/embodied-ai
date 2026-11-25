import os,sys,json,pathlib,hashlib,gzip,re,shutil,time,collections
import backup
P=pathlib.Path
IMAGE=re.compile(rb'(?P<uri>data:image/[A-Za-z0-9.+-]+;base64,)(?P<payload>[A-Za-z0-9+/_=-]{128,})|(?P<raw>(?<![A-Za-z0-9+/_=-])(?:iVBORw0KGgo|/9j/|R0lGOD|UklGR)[A-Za-z0-9+/_=-]{128,})')
MARKER=re.compile(rb'\[OMITTED_IMAGE sha256=[0-9a-f]{64} chars=[0-9]+\]')
def clean(data):
 count=0;chars=0
 def replace(m):
  nonlocal count,chars
  payload=m.group('payload') or m.group('raw');count+=1;chars+=len(payload)
  return (m.group('uri') or b'')+b'[OMITTED_IMAGE sha256='+hashlib.sha256(payload).hexdigest().encode()+b' chars='+str(len(payload)).encode()+b']'
 new=IMAGE.sub(replace,data)
 if count:
  old_without=IMAGE.sub(lambda m:m.group('uri') or b'',data);new_without=MARKER.sub(b'',new)
  if old_without!=new_without:raise RuntimeError('Non-image bytes changed')
  proof=hashlib.sha256(old_without).hexdigest()
 else:proof=hashlib.sha256(data).hexdigest()
 return new,count,chars,proof

def video_group(path):
 bits=path.split('/')
 for marker in ['formal-results','runs']:
  if marker in bits:
   i=bits.index(marker);return '/'.join(bits[:i+3])
 return str(P(path).parent)
def run(task,base,samplehashes):
 entries,excluded=backup.inventory(task);stage=P(base)/'stat-snapshots'/task['name'];stage.mkdir(parents=True,exist_ok=True);provenance=[];images=collections.Counter();videos=collections.Counter();rawsamples=collections.Counter();totalimages=0;totalchars=0;originalbytes=0;archivedbytes=0
 backup.log('STAT SNAPSHOT START '+task['name'])
 for e in entries:
  path=e['path'];source=P(e['source']);target=stage/path;typ=e['type'];name=source.name
  if typ=='dir':target.mkdir(parents=True,exist_ok=True);continue
  if typ=='symlink':
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.is_symlink():target.symlink_to(e['target'])
   continue
  ext=source.suffix.lower();why=None
  if ext in {'.png','.jpg','.jpeg','.webp','.gif'} and not any(p in path.split('/') for p in ['figures','plots']):
   group=str(P(path).parent);images[group]+=1
   if images[group]>3:why='optional-image-payload; first-3-images-per-directory retained'
  if ext in {'.mp4','.webm','.avi','.mov'}:
   group=video_group(path);videos[group]+=1
   if videos[group]>1:why='optional-video; representative per model/suite/directory retained'
  if ext=='.tar' and (re.fullmatch(r'ri[0-9]*\.tar',name) or name.startswith(('req_imgs','record_av'))):why='optional-historical-image-carrier-archive'
  if name in {'source.tar','runtime.tar.gz','remaining.tar.zst'}:why='optional-runtime/source/QA-transfer-container; expanded result and metric files retained'
  if why:excluded.append({'path':path,'reason':why,'original_bytes':e['size']});continue
  target.parent.mkdir(parents=True,exist_ok=True);before=source.stat();data=source.read_bytes();origsha=hashlib.sha256(data).hexdigest();new=data;count=chars=0;proof=None;sample=any(h in name or h==origsha for h in samplehashes)
  if name.endswith('.txt.stream.jsonl'):
   group=video_group(path);rawsamples[group]+=1
   if rawsamples[group]==1:sample=True
  if '/codex-home/sessions/' in path:
   group=path.split('/codex-home/sessions/')[0];rawsamples[group]+=1
   if rawsamples[group]==1:sample=True
  text=name.endswith(('.json','.jsonl','.txt','.log','.out'))
  gzjson=name.endswith(('.json.gz','.jsonl.gz'))
  if not sample and (text or gzjson):
   raw=gzip.decompress(data) if gzjson else data
   filtered,count,chars,proof=clean(raw)
   if count:new=gzip.compress(filtered,compresslevel=1,mtime=0) if gzjson else filtered
  after=source.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed: '+path)
  target.write_bytes(new);os.chmod(target,e['mode']);os.utime(target,ns=(after.st_atime_ns,after.st_mtime_ns))
  provenance.append({'path':path,'original_sha256':origsha,'original_bytes':len(data),'archived_sha256':hashlib.sha256(new).hexdigest(),'archived_bytes':len(new),'image_payloads_omitted':count,'base64_chars_omitted':chars,'all_non_image_bytes_unchanged':True,'non_image_bytes_sha256':proof,'original_sample_retained':sample,'source_mtime_ns':before.st_mtime_ns})
  totalimages+=count;totalchars+=chars;originalbytes+=len(data);archivedbytes+=len(new)
 after,_=backup.inventory(task)
 if backup.signature(entries)!=backup.signature(after):raise RuntimeError('Source inventory changed during filtered snapshot')
 source_manifest={'format':2,'scope':'all-statistical-records-with-necessary-original-samples','source_stable':True,'image_filter':'base64 image payloads only; every byte outside image payloads verified unchanged','original_files':provenance,'excluded':excluded,'summary':{'retained_files':len(provenance),'retained_original_bytes':originalbytes,'archived_bytes':archivedbytes,'image_payloads_omitted':totalimages,'base64_chars_omitted':totalchars,'raw_original_samples':sum(e['original_sample_retained'] for e in provenance)}}
 backup.writejson(stage/'_SOURCE_PROVENANCE.json',source_manifest)
 cfg={'name':task['name'],'dest':task['dest'],'roots':[{'src':str(p),'prefix':p.name} for p in sorted(stage.iterdir())],'notes':['User chose statistical originals and necessary samples; other image payloads intentionally omitted.','All bytes outside omitted image payloads retained exactly; original fingerprints in _SOURCE_PROVENANCE.json.']}
 result=backup.bundle(cfg,task['dest']);backup.writejson(P(task['dest'])/(task['name']+'.source-provenance.json'),source_manifest);backup.log('STAT SNAPSHOT READY '+json.dumps(source_manifest['summary']));return result
if __name__=='__main__':
 config=json.loads(P(sys.argv[1]).read_text());samples=set(config['necessary_sample_hashes'])
 for task in config['tasks']:run(task,config['stage_base'],samples)
