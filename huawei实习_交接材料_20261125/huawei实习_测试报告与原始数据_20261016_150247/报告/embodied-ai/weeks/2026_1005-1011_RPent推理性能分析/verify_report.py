# coding: utf-8
"""Check the sanitized ledger against the authoritative final acceptance manifest."""
from pathlib import Path
import json, re
P=Path(__file__).resolve().parent
D=json.loads((P/'evidence/runtime-ledger.json').read_text());A=D['episodes']
M=json.loads((P.parent/'2026_0928-1004_RPent双脑对照与回馈/evidence/2026-10-01-RPent-Codex-120集验收.json').read_text())['episodes']
def identity(x):return x['model'],x['suite'],x['task'],x['seed']
idx={identity(x):x for x in M}
assert len(A)==len({identity(x) for x in A})==120
assert {identity(x) for x in A}==set(idx)
assert sum(x['environment_success'] for x in A)==99
for x in A:
 m=idx[identity(x)]
 for k in ('attempt','elapsed_s','environment_success','video_frames'):assert x[k]==m[k],(identity(x),k)
 assert x['state_sha256']==m['artifact_sha256']['states.json']
 assert x['transcript_sha256']==next(v for k,v in m['artifact_sha256'].items() if k.startswith('transcript'))
 assert abs(sum(x[k] for k in ('manual_primitive_s','vla_primitive_s','tool_other_s','sdk_non_mcp_s','outside_sdk_s'))-x['elapsed_s'])<0.01
 for k in ('input_tokens','cached_input_tokens','output_tokens'):
  assert sum(y[k] for y in x['request_usage'])==x[k]
 assert sum(y['reasoning_output_tokens'] for y in x['request_usage'])==x['reasoning_tokens']
 assert len(x['request_usage'])==x['request_count']
 assert x['load_s']>0 and x['actual_model']==x['model'] and x['effort']=='medium'
 assert min(x[k] for k in ('manual_primitive_s','vla_primitive_s','tool_other_s','sdk_non_mcp_s','outside_sdk_s'))>=0
text=(P/'2026-10-08-RPent-120集推理耗时与优化报告.html').read_text()
for anchor in re.findall(r'href="#([^"]+)"',text):assert f'id="{anchor}"' in text,anchor
for f in P.rglob('*'):
 if f.suffix not in ('.html','.json','.md'):continue
 s=f.read_text()
 for pattern in (r'/Users/[^\s"<>]+',r'/data/[^\s"<>]+',r'\b(?:\d{1,3}\.){3}\d{1,3}\b',r'github_pat_[A-Za-z0-9_]+',r'sk-[A-Za-z0-9]{20,}'):
  assert not re.search(pattern,s),(f.name,pattern)
result={'date':'2026-10-08','episodes':120,'success':99,'failure':21,'unique_metered_responses':sum(x['request_count'] for x in A),'identity_attempt_and_outcome_match':True,'timing_reconciliation':True,'all_token_totals_match':True,'remote_artifact_hash_check':D['hash_check'],'text_sensitive_pattern_scan':True,'html_anchor_check':True}
(P/'evidence/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
