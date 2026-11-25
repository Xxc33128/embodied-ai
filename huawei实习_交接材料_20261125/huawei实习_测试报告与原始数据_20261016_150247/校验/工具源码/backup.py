import os,sys,json,hashlib,tarfile,time,stat,pathlib,shutil,argparse
P=pathlib.Path
SKIPDIR={'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','node_modules','.pi','.zcode','.cache'}
SECRET={'auth.json','.glmkey','.deepseekkey','gpt.local.json','credentials.json','credentials','id_rsa','id_ed25519','.env'}
def reason(path,rel):
 n=path.name
 if 'codex-home/' in rel and ('.sqlite' in n or n in {'models_cache.json','version.json'}):return 'optional-Codex-runtime-database/cache'
 if rel.startswith('rpent-repo/archive'):return 'duplicate-tar-of-retained-extracted-batch'
 if rel.startswith(('rpent2/codex-eval-20260929/','rpent2/deep-timing-20261008/','rpent2/ds-adapter-20261009/','rpent2/fw_','rpent2/logs/')) and n.endswith(('.png','.npz')) and path.is_dir():return 'optional-render-frame/depth-duplicate; video,structured-logs,request-bodies-retained'
 if n in SKIPDIR or n.startswith('.venv'): return 'dependency/cache/VCS'
 if n in SECRET or n.startswith('.env.') or n.endswith(('.pem','.key','.sock','.pid')) or '私发勿入库' in n: return 'credential/private-guide/runtime-handle'
 if n=='.DS_Store' or n.startswith(('~$','.~')): return 'desktop/office-temporary'
 if rel=='rpent-repo/AGENTS.md': return 'private-node-connection-instructions'
 return None

def inventory(cfg):
 out=[]; excluded=[]
 for root in cfg['roots']:
  src=P(root['src']); prefix=root['prefix'].rstrip('/')
  if not src.exists() and not src.is_symlink(): raise RuntimeError('MISSING REQUIRED SOURCE: '+str(src))
  stack=[(src,prefix)]
  while stack:
   p,r=stack.pop(); why=reason(p,r)
   if why: excluded.append({'path':r,'reason':why});continue
   s=p.lstat()
   if stat.S_ISDIR(s.st_mode):
    out.append({'source':str(p),'path':r,'type':'dir','mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'size':0})
    for child in sorted(p.iterdir(),reverse=True): stack.append((child,r+'/'+child.name))
   elif stat.S_ISREG(s.st_mode): out.append({'source':str(p),'path':r,'type':'file','size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode)})
   elif stat.S_ISLNK(s.st_mode):
    target=os.readlink(p)
    if p.name in SECRET or any(x in target for x in ('auth.json','id_rsa','id_ed25519')): excluded.append({'path':r,'reason':'credential-symlink'});continue
    out.append({'source':str(p),'path':r,'type':'symlink','target':target,'size':0,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode)})
   else: excluded.append({'path':r,'reason':'special-file/socket/device'})
 return out,excluded

def signature(entries):
 return [(e['path'],e['type'],e['size'],e['mtime_ns'],e.get('target')) for e in entries if e['type']!='dir']
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def writejson(p,v):
 P(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def log(msg):print(time.strftime('%H:%M:%S'),msg,flush=True)
class Reader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256()
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);return b

def bundle(cfg,dest):
 name=cfg['name']; dest=P(dest); dest.mkdir(parents=True,exist_ok=True)
 final=dest/(name+'.tar.gz'); part=dest/(name+'.tar.gz.partial')
 for attempt in range(1,4):
  entries,excluded=inventory(cfg); log('%s start attempt=%d files=%d bytes=%d'%(name,attempt,sum(e['type']=='file' for e in entries),sum(e['size'] for e in entries)))
  unstable=[]; start=time.time(); done=0
  with tarfile.open(part,'w:gz',compresslevel=1,format=tarfile.PAX_FORMAT,dereference=True) as tf:
   for e in entries:
    p=P(e['source']); ti=tarfile.TarInfo(e['path']);ti.mode=e['mode'];ti.mtime=e['mtime_ns']/1e9
    if e['type']=='dir':ti.type=tarfile.DIRTYPE;tf.addfile(ti)
    elif e['type']=='symlink':ti.type=tarfile.SYMTYPE;ti.linkname=e['target'];tf.addfile(ti)
    else:
     ti.size=e['size']
     with p.open('rb') as f:
      before=os.fstat(f.fileno());reader=Reader(f);tf.addfile(ti,reader);after=os.fstat(f.fileno())
     e['sha256']=reader.h.hexdigest()
     if before.st_size!=e['size'] or before.st_mtime_ns!=e['mtime_ns'] or after.st_size!=before.st_size or after.st_mtime_ns!=before.st_mtime_ns:unstable.append(e['path'])
     done+=1
     if done%2000==0:log('%s files=%d'%(name,done))
   manifest={'format':1,'name':name,'snapshot_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'entries':entries,'excluded':excluded,'scope_notes':cfg.get('notes',[])}
   import io
   data=json.dumps(manifest,ensure_ascii=False).encode();ti=tarfile.TarInfo('_ARCHIVE_MANIFEST.json');ti.size=len(data);tf.addfile(ti,io.BytesIO(data))
  after,_=inventory(cfg)
  if unstable or signature(entries)!=signature(after):
   log(name+' SOURCE CHANGED; retry');continue
  os.replace(part,final);writejson(dest/(name+'.manifest.json'),manifest)
  result={'name':name,'files':done,'bytes':sum(e['size'] for e in entries),'archive_bytes':final.stat().st_size,'archive_sha256':digest(final),'source_stable':True,'excluded_count':len(excluded),'seconds':round(time.time()-start,1)}
  writejson(dest/(name+'.ready.json'),result);log(json.dumps(result));return result
 raise RuntimeError(name+' source remained unstable after 3 attempts')

def copytree(cfg,dest):
 dest=P(dest);entries,excluded=inventory(cfg);log('reports start files='+str(sum(e['type']=='file' for e in entries)))
 for e in entries:
  p=P(e['source']);q=dest/e['path'];q.parent.mkdir(parents=True,exist_ok=True)
  if e['type']=='dir':q.mkdir(exist_ok=True);continue
  if e['type']=='symlink':raise RuntimeError('Readable-copy symlink needs archive: '+str(p))
  for retry in range(3):
   h=hashlib.sha256();before=p.stat()
   with p.open('rb') as f,q.open('wb') as out:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b);out.write(b)
   after=p.stat()
   if (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and digest(q)==h.hexdigest():break
  else:raise RuntimeError('Unstable report: '+str(p))
  e['size']=after.st_size;e['mtime_ns']=after.st_mtime_ns;e['sha256']=h.hexdigest();os.utime(q,ns=(after.st_atime_ns,after.st_mtime_ns))
 after,_=inventory(cfg)
 if signature(entries)!=signature(after):raise RuntimeError('Report source inventory changed')
 result={'format':1,'entries':entries,'excluded':excluded,'source_stable':True}
 writejson(dest/'reports.manifest.json',result);log('reports COPIED AND SHA256 VERIFIED files='+str(sum(e['type']=='file' for e in entries)))
 return result

def verifyarchive(path):
 # Manifest is also saved alongside so verification can run in a single streaming pass.
 path=P(path);mp=path.with_name(path.name[:-7]+'.manifest.json');manifest=json.loads(mp.read_text())
 expected={e['path']:e for e in manifest['entries']};seen=set();embedded=False;count=0;total=0
 with tarfile.open(path,'r|gz') as tf:
  for m in tf:
   if m.name=='_ARCHIVE_MANIFEST.json':
    if json.load(tf.extractfile(m))!=manifest:raise RuntimeError('Embedded manifest mismatch: '+str(path))
    embedded=True;continue
   if m.name in seen or m.name not in expected:raise RuntimeError('Unexpected/duplicate member: '+m.name)
   e=expected[m.name];seen.add(m.name)
   if e['type']=='file':
    if not m.isfile() or m.size!=e['size']:raise RuntimeError('Type/size mismatch: '+m.name)
    h=hashlib.sha256()
    f=tf.extractfile(m)
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if h.hexdigest()!=e['sha256']:raise RuntimeError('SHA256 mismatch: '+m.name)
    count+=1;total+=m.size
    if count%5000==0:log(path.name+' verified files='+str(count))
   if e['type']=='symlink' and (not m.issym() or m.linkname!=e['target']):raise RuntimeError('Symlink mismatch')
   if e['type']=='dir' and not m.isdir():raise RuntimeError('Directory mismatch')
 if not embedded or seen!=set(expected):raise RuntimeError('Missing archive members')
 return count,total

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);args=a.parse_args();config=json.loads(P(args.config).read_text())
 for task in config['tasks']:
  if task.get('kind')=='copy':copytree(task,task['dest'])
  else:bundle(task,task['dest'])
