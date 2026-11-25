#!/usr/bin/env python3
"""Verify report provenance, metrics, HTML links, inline media and JS syntax."""
from pathlib import Path
from html.parser import HTMLParser
import base64, hashlib, json, re, subprocess, tempfile
P=Path(__file__).resolve().parent
class Parser(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.media=[]; self.scripts=[]; self.inscript=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag in ('img','source') and 'src' in a:self.media.append(a['src'])
        if tag=='script':self.inscript=True;self.scripts.append('')
    def handle_endtag(self,tag):
        if tag=='script':self.inscript=False
    def handle_data(self,data):
        if self.inscript:self.scripts[-1]+=data
h=Parser();html=(P/'report.html').read_text();h.feed(html)
checks={}
checks['unique_html_ids']=len(h.ids)==len(set(h.ids))
checks['all_internal_anchors_resolve']=all(x[1:] in h.ids for x in h.links if x.startswith('#'))
checks['local_links_exist']=all((P/x).exists() for x in h.links if not x.startswith(('#','https:','http:')))
checks['all_media_inline']=all(x.startswith('data:') for x in h.media)
checks['nine_diagrams_three_videos']=len(h.media)==12
checks['all_inline_media_decode']=all(len(base64.b64decode(x.split(',',1)[1],validate=True))>0 for x in h.media)
manifest=json.loads((P/'data/source-manifest.json').read_text())
checks['twenty_four_local_sources']=len(manifest['sources'])==24
checks['ten_primary_references']=len(manifest['related_work'])==10
checks['source_copies_match_hash']=all(hashlib.sha256((P/x['bundled']).read_bytes()).hexdigest()==x['sha256'] for x in manifest['sources'])
media=json.loads((P/'data/media-manifest.json').read_text())['available']
checks['video_bytes_match_hash']=all(hashlib.sha256((P/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in media)
for index,m in enumerate(media):
    raw=base64.b64decode([x for x in h.media if x.startswith('data:video/')][index].split(',',1)[1])
    checks['embedded_video_'+m['id']]=hashlib.sha256(raw).hexdigest()==m['sha256']
with tempfile.TemporaryDirectory(prefix='survey-js-') as tmp:
    js=Path(tmp)/'report.js';js.write_text('\n'.join(h.scripts))
    result=subprocess.run(['node','--check',str(js)],capture_output=True,text=True)
    checks['javascript_syntax']=result.returncode==0
    if result.returncode:print(result.stderr)
md=(P/'report.md').read_text()
chapters=re.findall(r'^## (\d+)　',md,re.M)
checks['ten_numbered_chapters']=chapters==[str(x) for x in range(1,11)]
checks['figures_numbered_in_order']=re.findall(r'^!\[图(\d+)',md,re.M)==[str(x) for x in range(1,10)]
checks['no_unexpanded_placeholders']=not re.search(r'<!-- (VIDEO|MEDIA_GAP|INTERACTIVE|REFERENCES)',html)
checks['rsi_evidence_boundary']='没有正式的 RSI' in md
checks['motus_latest_not_best']='| 10000步 | 30/50' in md
validation=json.loads((P/'data/validation.json').read_text())
validation['extended_checks']=checks
browser_record=P/'data/browser-verification.json'
if browser_record.exists():
    browser=json.loads(browser_record.read_text())
    current_hash=hashlib.sha256((P/'report.html').read_bytes()).hexdigest()
    if browser.get('report_sha256')==current_hash:
        validation['browser_verification']=browser
        validation['media_playback_verification']=browser.get('scope', 'See data/browser-verification.json')
    else:
        validation['browser_verification']={'status':'stale','reason':'HTML changed since recorded browser playback; repeat browser verification.'}
        validation['media_playback_verification']='Current HTML has not been browser-playback verified.'
else:
    validation['browser_verification']={'status':'not_performed','reason':'No real browser verification record for this HTML.'}
    validation['media_playback_verification']='Container metadata and byte hashes verified; browser playback not recorded.'
validation['report_version']='2.0'
validation['report_date']='2026-11-25'
(P/'data/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
assert all(checks.values()),{k:v for k,v in checks.items() if not v}
print(json.dumps({'checks_passed':len(checks),'html_bytes':len(html.encode()),'source_characters':len(md),'source_files':len(manifest['sources']),'references':len(manifest['related_work'])},ensure_ascii=False))
