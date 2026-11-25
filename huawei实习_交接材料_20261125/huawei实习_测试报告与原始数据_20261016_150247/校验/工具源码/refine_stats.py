import pathlib,json,zipfile,ast,struct,hashlib,io,os,re,time
import backup,stat_snapshot
P=pathlib.Path

def is_image(name,shape,dtype):
 if len(shape)>=3 and ((shape[-1] in (1,3,4) and min(shape[-3:-1])>=16) or (shape[-3] in (1,3,4) and min(shape[-2:])>=16)):return True
 return len(shape)>=2 and min(shape[-2:])>=16 and any(x in name.lower() for x in ['depth','world_image','main_images','wrist_images'])
def npz_filter(data):
 out=io.BytesIO();kept=[];omitted=[]
 with zipfile.ZipFile(io.BytesIO(data)) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as target:
  for info in src.infolist():
   shape=();dtype=None
   if info.filename.endswith('.npy'):
    with src.open(info) as f:
     magic=f.read(8)
     if magic[:6]==b'\x93NUMPY':
      n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];header=ast.literal_eval(f.read(n).decode('latin1'));shape=header['shape'];dtype=header['descr']
   if is_image(info.filename,shape,dtype):omitted.append({'member':info.filename,'original_shape':shape,'dtype':dtype,'original_bytes':info.file_size,'original_crc32':info.CRC});continue
   raw=src.read(info);target.writestr(info.filename,raw);kept.append({'member':info.filename,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'shape':shape,'dtype':dtype})
 if not omitted:return data,None
 new=out.getvalue()
 with zipfile.ZipFile(io.BytesIO(new)) as dst:
  for e in kept:
   if hashlib.sha256(dst.read(e['member'])).hexdigest()!=e['sha256']:raise RuntimeError('Non-image array changed')
 return new,{'omitted_image_arrays':omitted,'retained_non_image_arrays':kept,'non_image_arrays_byte_identical':True}

def run(stage,name):
 stage=P(stage);root=stage/'stat-snapshots'/name;mp=root/'_SOURCE_PROVENANCE.json';m=json.loads(mp.read_text());changed=0
 for e in m['original_files']:
  if not e['path'].endswith('.npz'):continue
  p=root/e['path'];raw=p.read_bytes()
  try:new,audit=npz_filter(raw)
  except zipfile.BadZipFile:continue
  if audit is None:continue
  p.write_bytes(new);e['archived_sha256']=hashlib.sha256(new).hexdigest();e['archived_bytes']=len(new);e['transformation']='npz-omit-image-arrays';e['all_non_image_bytes_unchanged']=None;e['non_image_array_members_byte_identical']=True;e['omitted_npz_image_arrays']=audit['omitted_image_arrays'];e['retained_npz_non_image_arrays']=audit['retained_non_image_arrays'];backup.writejson(p.with_name(p.name+'.array-audit.json'),audit);changed+=1
 m['summary']['archived_bytes']=sum(e['archived_bytes'] for e in m['original_files']);m['summary']['npz_files_image_arrays_omitted']=changed;backup.writejson(mp,m)
 task={'name':name,'dest':str(stage/'packages-stat'),'roots':[{'src':str(p),'prefix':p.name} for p in sorted(root.iterdir())],'notes':['User selected statistical originals and necessary samples; NPZ image arrays omitted; all non-image NPY member bytes verified identical.']}
 backup.bundle(task,task['dest']);backup.writejson(stage/'packages-stat'/(name+'.source-provenance.json'),m);backup.log('REFINED NPZ READY '+str(changed))
if __name__=='__main__':run(__import__('sys').argv[1],'RPent_早期批次及代理原始记录')
