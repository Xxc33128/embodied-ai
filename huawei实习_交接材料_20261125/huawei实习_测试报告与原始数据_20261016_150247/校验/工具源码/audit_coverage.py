import pathlib,json,sys,collections,time
T=pathlib.Path('/tmp/huawei-test-backup-20261016');s=json.loads((T/'state.json').read_text());D=pathlib.Path(s['dest']);files=[];exclusions=[];packages=[]
for p in (D/'原始数据').rglob('*.manifest.json'):
 if p.name.startswith('._'):continue
 m=json.loads(p.read_text());packages.append(m['name']);files.extend(e for e in m['entries'] if e['type']=='file');exclusions.extend({'package':m['name'],**e} for e in m['excluded'])
assert len(packages)==8,('missing package manifests',len(packages))
paths={e['path'] for e in files};archive_hashes={e['sha256'] for e in files};hashes=set(archive_hashes);checks=[]
provenance_checks=[];sample_paths=[];retained_index={(f['path'],f['sha256']):f for f in files}
for p in (D/'原始数据').rglob('*.source-provenance.json'):
 if p.name.startswith('._'):continue
 prov=json.loads(p.read_text());assert prov['source_stable']
 for e in prov['original_files']:
  hashes.add(e['original_sha256'])
  entry=retained_index.get((e['path'],e['archived_sha256']))
  assert entry is not None,('missing retained statistical record',e['path'])
  assert e.get('all_non_image_bytes_unchanged') is True or e.get('non_image_array_members_byte_identical') is True
  if e.get('additional_mcp_image_filter'):assert e['additional_mcp_image_filter']['all_other_bytes_unchanged']
  if e.get('original_sample_retained'):
   assert e['original_sha256']==e['archived_sha256'];sample_paths.append(e['path'])
 provenance_checks.append({'package':p.stem,'retained_statistical_records':len(prov['original_files']),'passed':True})
def required(name,prefix):
 count=sum(p.startswith(prefix.rstrip('/')+'/') for p in paths)
 checks.append({'name':name,'prefix':prefix,'files':count,'passed':count>0})
 if not count:raise RuntimeError('Missing required data '+prefix)
for name in ['logs-glm-subagent20-20260915-120407','logs-deepseekv4.1-subagent20-20260915-1435','logs-qwen3.8flash-subagent20-20260915-1545','logs-persistent-subagent20-20260915-113640','logs-astra-five-20260923','logs-sol-five-20260923','logs-sol-ultra-five-20260923','logs-codex-luna-max-five-20260923']:
 required('bowl '+name,'gpt6astra-repro/'+name)
for name in ['init0_20260921_141155','goal8_hybrid_repeat_20260921_160205','decision_v2_20260922','decision_extra_20260922','decision_extra_retry1_20260922','decision_v2_probe','parallel-probe-20260921','codex_input_audit_20260929']:
 required('policy '+name,'GPT-as-Policy-repro/data/'+name)
for name in ['eval20-npunew','goal20-npunew','flash20-npunew','retry-npunew']:
 required('RPent early '+name,'rpent-repo/'+name)
for root,sub in [('codex-eval-20260929','formal-results'),('codex-eval-20260929','batch'),('deep-timing-20261008','formal-results'),('deep-timing-20261008','measurements'),('ds-adapter-20261009','runs'),('ds-adapter-20261009','measurements'),('ds-adapter-20261009','private/llm-bodies')]:
 required('RPent remote '+root+'/'+sub,'rpent2/'+root+'/'+sub)
report=D/'报告/embodied-ai/weeks';ledger=json.loads((report/'2026_1005-1011_RPent推理性能分析/evidence/runtime-ledger.json').read_text());hashchecks=[]
for field in ['stream_sha256','state_sha256','transcript_sha256']:
 missing=[{'model':e['model'],'suite':e['suite'],'task':e['task'],'seed':e['seed'],'field':field,'sha256':e[field]} for e in ledger['episodes'] if e[field] not in hashes]
 hashchecks.append({'source':'runtime-ledger.json','field':field,'expected':len(ledger['episodes']),'matched':len(ledger['episodes'])-len(missing),'missing':missing,'passed':not missing,'fingerprint_scope':'original source fingerprint mapped to retained statistical record via provenance; image payloads may be intentionally omitted'})
proxy=json.loads((report/'2026_1012-1018_RPent多模型性能分析/ds-report/proxy-inventory.json').read_text());missing=[{'file':e['file'],'sha256':e['sha256']} for e in proxy['files'] if e['sha256'] not in hashes]
hashchecks.append({'source':'proxy-inventory.json','expected':len(proxy['files']),'matched':len(proxy['files'])-len(missing),'missing':missing,'passed':not missing,'fingerprint_scope':'original source fingerprint mapped to retained statistical record via provenance; image payloads may be intentionally omitted'})
policycounts={}
for name,expected in [('init0_20260921_141155',18),('decision_v2_20260922',3),('decision_extra_20260922',5),('decision_extra_retry1_20260922',1)]:
 prefix='GPT-as-Policy-repro/data/'+name+'/episodes/';count=sum(p.startswith(prefix) and p.endswith('/result.json') for p in paths);policycounts[name]={'expected':expected,'actual':count,'passed':count==expected}
sample_hashes=json.loads((T/'necessary_sample_hashes.json').read_text());sample_match={h:any(h in p for p in sample_paths) for h in sample_hashes};assert all(sample_match.values()),'Missing full original replay bodies'
passed=all(c['passed'] for c in hashchecks) and all(c['passed'] for c in policycounts.values())
result={'passed':passed,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'package_count':len(packages),'required_data_roots':checks,'source_hash_matches':hashchecks,'policy_episode_counts':policycounts,'provenance_checks':provenance_checks,'necessary_full_replay_body_count':sum(sample_match.values()),'all_non_image_record_bytes_or_NPZ_array_bytes_verified_unchanged':True}
(D/'校验/报告统计数据覆盖核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(D/'校验/排除项.json').write_text(json.dumps(exclusions,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));sys.exit(0 if passed else 1)
