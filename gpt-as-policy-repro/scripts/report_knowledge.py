"""Render the maintained Markdown companion into an offline HTML knowledge section."""
import base64
import hashlib
import html
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote
import markdown

TOPICS = {'scope':'1','interfaces':'2','states':'3','execution':'4','timing':'5','tokens':'6','results':'7','optimization':'8'}

def find_companion(explicit=None):
    if explicit:
        return Path(explicit).resolve(strict=True)
    repo = Path(__file__).resolve().parent.parent
    for docs in (repo, repo.parent/'embodied-ai', repo.parent):
        p = docs/'weeks/2026_0921-0927_同机仿真与云边验证/2026-09-23-实验报告对话疑问与解答.md'
        if p.is_file():
            return p
    raise FileNotFoundError('Use --knowledge-md to locate the maintained report companion.')

def render_knowledge(path):
    source = path.read_text(encoding='utf-8')
    version = re.search(r'版本：([^\n]+)', source).group(1)
    files = {}
    def embed(target, image=False):
        p = (path.parent/unquote(target)).resolve(strict=True)
        # Only publish references within the authoritative public repository.
        p.relative_to(path.parents[2].resolve())
        raw = p.read_bytes()
        files[target] = hashlib.sha256(raw).hexdigest()
        mime = mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
        return 'data:'+mime+';base64,'+base64.b64encode(raw).decode()
    def link(label, target):
        target = html.unescape(target)
        if target.startswith('#'):
            topic = TOPICS.get(target[1:])
            dest = 'kb-'+topic if topic else target[1:]
            return f'<a href="#{dest}" onclick="openKnowledge(\'{dest}\')">{label}</a>'
        if target.startswith(('https://','http://')):
            return f'<a href="{html.escape(target,quote=True)}" target="_blank" rel="noopener">{label}</a>'
        data = embed(target)
        return f'<a href="{data}" download="{html.escape(Path(target).name,quote=True)}">{label}（内嵌附件）</a>'
    def render(text):
        out = markdown.markdown(text, extensions=['tables','fenced_code','toc'], extension_configs={'toc':{'slugify':lambda title,sep:'kb-'+re.match(r'[\d.]+',title).group(0).rstrip('.').replace('.','-') if re.match(r'\d',title) else 'kb-note-'+hashlib.sha256(title.encode()).hexdigest()[:10]}})
        out = re.sub(r'<img alt="([^"]*)" src="([^"]*)"\s*/?>',lambda m:f'<img alt="{m[1]}" src="{embed(html.unescape(m[2]),True)}" loading="lazy">',out)
        out = re.sub(r'<a href="([^"]*)">(.*?)</a>',lambda m:link(m[2],m[1]),out)
        return out.replace('<table>','<div class="knowledge-table"><table>').replace('</table>','</table></div>')
    source = re.sub(r'^<a id="[^"]+"></a>\s*$', '', source, flags=re.M)
    chunks = re.split(r'^## ',source,flags=re.M)[1:]
    body=[]; nav=[]
    for index,chunk in enumerate(chunks,1):
        title,content = chunk.split('\n',1)
        ident='kb-'+str(index) if index<=8 else 'kb-appendix-'+str(index)
        nav.append(f'<a href="#{ident}" onclick="openKnowledge(\'{ident}\')">{html.escape(title)}</a>')
        body.append(f'<details class="knowledge-topic" id="{ident}"><summary>{html.escape(title)}</summary>{render(content)}</details>')
    source_uri=embed(path.name)
    result='<section id="knowledge"><h2>知识与分析</h2><p>固定批次说明：九个 init 0 回合。此区内容不随上方任务和轮次切换；图表的任务、轮次范围以各图标注为准。</p>'
    result+=f'<p class="note">同步自补充 MD · {html.escape(version)}。〔E1〕源码或记录，〔E2〕配置事实，〔E3〕解释或推断。外部网站引用需联网；正文、配图和本地附件均已内嵌。</p><nav class="knowledge-nav" aria-label="知识目录">'+''.join(nav)+'</nav>'+''.join(body)
    result+=f'<p><a href="{source_uri}" download="{html.escape(path.name)}">下载本页采用的补充 MD</a> · <a href="#experiment">返回实验记录</a></p></section>'
    return result, {'source':path.name,'version':version,'files_sha256':files,'sections':len(body)}
