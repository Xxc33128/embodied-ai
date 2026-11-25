"""Check standalone knowledge links and embedded assets without a browser."""
import base64
import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.anchors=[]; self.images=[]; self.downloads=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and a.get('href','').startswith('#'):self.anchors.append(a['href'][1:])
        if tag=='img':self.images.append(a.get('src',''))
        if tag=='a' and 'download' in a:self.downloads.append(a.get('href',''))

def verify(campaign):
    root=Path(campaign); p=ReportParser();p.feed((root/'index.html').read_text())
    duplicates=[k for k,v in Counter(p.ids).items() if v>1]
    assert not duplicates,duplicates
    assert not (set(p.anchors)-set(p.ids)),set(p.anchors)-set(p.ids)
    assert len(p.images)==4 and all(x.startswith('data:image/png;base64,') for x in p.images)
    assert all(x.startswith('data:') for x in p.downloads)
    embedded={hashlib.sha256(base64.b64decode(x.split(',',1)[1])).hexdigest() for x in p.images+p.downloads}
    manifest=json.loads((root/'knowledge_manifest.json').read_text())
    assert set(manifest['files_sha256'].values())<=embedded,'Missing or altered embedded source'
    assert manifest['sections']==10
    return {'knowledge_sections':manifest['sections'],'embedded_figures':len(p.images),'embedded_downloads':len(p.downloads),'anchor_targets':len(p.anchors),'source_hashes_verified':len(manifest['files_sha256']),'browser_rendering':'not_performed'}

if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),ensure_ascii=False))
