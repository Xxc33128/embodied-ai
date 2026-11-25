#!/usr/bin/env python3
"""Read-only verification of report files and all 8 source-data archives."""
import pathlib,json,hashlib,tarfile,sys,time,subprocess,shutil
BASE=pathlib.Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def verify_archive(path):
 is_zst=path.name.endswith('.tar.zst');is_tar=path.name.endswith('.tar');stem=path.name[:-8] if is_zst else (path.name[:-4] if is_tar else path.name[:-7])
 mpath=path.with_name(stem+'.manifest.json');rpath=path.with_name(stem+('.zst.ready.json' if is_zst else ('.tar.ready.json' if is_tar else '.ready.json')))
 manifest=json.loads(mpath.read_text());ready=json.loads(rpath.read_text());expected={e['path']:e for e in manifest['entries']};seen=set();embedded=False;count=0;size=0
 if len(expected)!=len(manifest['entries']):raise ValueError('Duplicate source paths: '+path.name)
 if not ready['source_stable']:raise ValueError('Unstable source: '+path.name)
 if sha(path)!=ready['archive_sha256']:raise ValueError('Whole archive SHA256 mismatch: '+path.name)
 proc=None
 if is_zst:
  executable=shutil.which('zstd')
  if not executable:raise RuntimeError('zstd executable required for .tar.zst archives')
  proc=subprocess.Popen([executable,'-q','-d','--long=31','-c',str(path)],stdout=subprocess.PIPE)
  tf=tarfile.open(fileobj=proc.stdout,mode='r|')
 else:tf=tarfile.open(path,'r|' if is_tar else 'r|gz')
 with tf:
  for member in tf:
   if member.name=='_ARCHIVE_MANIFEST.json':
    if embedded or json.load(tf.extractfile(member))!=manifest:raise ValueError('Embedded manifest mismatch: '+path.name)
    embedded=True;continue
   if member.name in seen or member.name not in expected:raise ValueError('Unexpected or duplicate member: '+member.name)
   seen.add(member.name);e=expected[member.name]
   if e['type']=='file':
    if not member.isfile() or member.size!=e['size']:raise ValueError('File size/type mismatch: '+member.name)
    h=hashlib.sha256();f=tf.extractfile(member)
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=e['sha256']:raise ValueError('File SHA256 mismatch: '+member.name)
    count+=1;size+=member.size
   elif e['type']=='dir':
    if not member.isdir():raise ValueError('Directory type mismatch: '+member.name)
   elif e['type']=='symlink':
    if not member.issym() or member.linkname!=e['target']:raise ValueError('Symlink mismatch: '+member.name)
   else:raise ValueError('Unsupported manifest type')
 if proc is not None:
  # Consume remaining tar padding to ensure the entire compressed stream is checked.
  while proc.stdout.read(8*1024*1024):pass
  proc.stdout.close()
  if proc.wait()!=0:raise ValueError('zstd decompression failed: '+path.name)
 if not embedded or seen!=set(expected):raise ValueError('Missing members: '+path.name)
 if count!=ready['files'] or size!=ready['bytes']:raise ValueError('Aggregate count mismatch: '+path.name)
 return {'path':str(path.relative_to(BASE)),'files':count,'original_bytes':size,'archive_bytes':path.stat().st_size,'sha256':ready['archive_sha256'],'symlinks':sum(e['type']=='symlink' for e in expected.values()),'passed':True}

def main():
 started=time.time();reportdir=BASE/'报告';m=json.loads((reportdir/'reports.manifest.json').read_text());count=0;size=0;expected=set()
 for e in m['entries']:
  if e['type']!='file':continue
  p=reportdir/e['path'];expected.add(e['path'])
  if not p.is_file() or p.stat().st_size!=e['size'] or sha(p)!=e['sha256']:raise ValueError('Report mismatch: '+e['path'])
  count+=1;size+=e['size']
 actual={str(p.relative_to(reportdir)) for p in reportdir.rglob('*') if p.is_file() and p.name!='reports.manifest.json' and not p.name.startswith('._')}
 if actual!=expected:raise ValueError('Report inventory mismatch')
 print('Reports verified:',count,'files,',size,'bytes',flush=True)
 archives=sorted(p for p in (BASE/'原始数据').rglob('*') if p.name.endswith(('.tar.gz','.tar.zst','.tar')) and not p.name.startswith('._'))
 if len(archives)!=8:raise ValueError('Expected 8 raw data archives, found '+str(len(archives)))
 results=[]
 for p in archives:
  result=verify_archive(p);results.append(result);print('PASS',p.name,result['files'],'files',flush=True)
 result={'passed':True,'verified_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'report_files':count,'report_bytes':size,'raw_files':sum(x['files'] for x in results),'raw_original_bytes':sum(x['original_bytes'] for x in results),'archive_bytes':sum(x['archive_bytes'] for x in results),'archive_count':len(results),'archives':results,'seconds':round(time.time()-started,1)}
 if '--write-result' in sys.argv:(BASE/'校验/完整性校验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  print('VERIFICATION FAILED:',type(error).__name__,str(error),file=sys.stderr);sys.exit(1)
