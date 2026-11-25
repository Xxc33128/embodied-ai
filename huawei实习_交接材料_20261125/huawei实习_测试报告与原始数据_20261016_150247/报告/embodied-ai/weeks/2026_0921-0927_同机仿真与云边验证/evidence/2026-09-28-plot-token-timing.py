"""Render the knowledge companion figures from the recorded aggregate statistics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'assets/token-timing'
OUT.mkdir(parents=True,exist_ok=True)
D=json.loads((ROOT/'evidence/2026-09-28-cache-ttft-decode-analysis.json').read_text())
font=FontProperties(family='Arial Unicode MS')
plt.rcParams.update({'font.family':font.get_name(),'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
BLUE='#3076B8'; ORANGE='#D97B30'; INK='#25354A'; TEAL='#23867B'; GRAY='#63748A'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white',bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)
def box(ax,x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.02',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',color='white',fontsize=13)

fig,ax=plt.subplots(figsize=(11,4.4)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
ax.text(0,.96,'TTFT、Decode 与 TPOT：客户端看到的时间',fontsize=19,weight='bold',color=INK)
ax.text(0,.86,'定义示意 · 区段宽度不代表实测时长',color=GRAY)
box(ax,.03,.49,.35,.17,'TTFT',BLUE); box(ax,.38,.49,.56,.17,'Decode 可见输出时间',TEAL)
for x,label in [(.03,'请求开始'),(.38,'首个非空可见分片'),(.94,'最后可见分片')]:
    ax.plot([x,x],[.43,.70],color=INK,lw=1)
    ax.text(x,.38,label,ha='center' if x==.38 else ('left' if x==.03 else 'right'),color=INK,fontsize=11)
ax.text(.03,.24,'TTFT：网络 + 排队 + 输入处理 + 首输出前计算（无法由此拆开）',color=GRAY)
ax.text(.03,.13,'TPOT 估算 = Decode 间隔 ÷（输出 Token − 推理 Token − 1）',color=INK)
ax.text(.03,.02,'分片 ≠ 单个 Token；TTFT ≠ 纯 Prefill；Decode 间隔 ≠ 服务端纯解码。',color=GRAY,fontsize=11)
save(fig,'01-timing-definitions')

E=D['episodes'][0]; a=E['first_floor_quarter']; b=E['last_floor_quarter']; groups=[a,b]
fig,axs=plt.subplots(1,2,figsize=(11.8,5.3),gridspec_kw={'width_ratios':[1.15,1]})
fig.suptitle('输入增至约 3.2 倍，TTFT 与 Decode 没有同比增长',fontsize=18,weight='bold',color=INK,y=1.02)
ax=axs[0]
c=[g['cached_tokens_mean']/1000 for g in groups]; u=[g['uncached_tokens_mean']/1000 for g in groups]
ax.bar([0,1],c,color=BLUE,width=.48,label='缓存输入'); ax.bar([0,1],u,bottom=c,color=ORANGE,width=.48,label='未缓存输入')
for i,g in enumerate(groups):
    ax.text(i,c[i]+u[i]+1.4,f"总计 {g['input_tokens_mean']:,.1f}",ha='center',fontsize=11)
    ax.text(i,c[i]/2,f"缓存 {g['cached_tokens_mean']:,.0f}\n命中 {g['cached_tokens_mean']/g['input_tokens_mean']:.1%}",ha='center',va='center',color='white',fontsize=11)
ax.set(xticks=[0,1],xticklabels=['第 1–12 轮','第 40–51 轮'],ylabel='输入 Token / 千',ylim=(0,83),title='每轮平均输入构成'); ax.legend(frameon=False,loc='upper left'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
ax=axs[1]
for offset,g,label,color in [(-.18,a,'第 1–12 轮',BLUE),(.18,b,'第 40–51 轮',ORANGE)]:
    vals=[g['ttft_s_mean'],g['decode_client_s_mean']]
    bars=ax.bar([offset,1+offset],vals,width=.34,label=label,color=color)
    ax.bar_label(bars,fmt='%.2f',padding=4)
ax.set(xticks=[0,1],xticklabels=['TTFT','Decode 可见输出'],ylabel='每轮平均耗时 / s',ylim=(0,12),title='客户端观测耗时'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
fig.text(.06,-.015,'罐子 Hybrid · 共 51 轮，比较前后各 12 轮。均值来自记录；并非缓存效果的因果实验。〔E1 / E3〕',color=GRAY,fontsize=11)
fig.tight_layout(); save(fig,'02-cache-and-latency')

fig,ax=plt.subplots(figsize=(11,4.4)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
ax.text(0,.96,'长历史命中缓存：复用旧状态，但仍需处理新内容',fontsize=18,weight='bold',color=INK)
ax.text(0,.85,'机制示意 · 60,000 + 1,300 为说明用数字，区块不按数量比例绘制',color=GRAY,fontsize=11)
box(ax,.02,.55,.42,.18,'60,000 个旧 Token\n匹配前缀 → 复用 KV 状态',BLUE)
box(ax,.55,.55,.42,.18,'1,300 个新 Token\n仍需进行输入计算',ORANGE)
box(ax,.32,.14,.42,.17,'生成本轮新输出\n仍需逐步计算',TEAL)
for x in [.23,.76]:
    ax.annotate('',xy=(.53,.33),xytext=(x,.53),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.8})
ax.text(.02,.02,'复用减少重复计算；新内容仍可能访问长历史。缓存 Token 不是显存占用量。〔E3〕',color=GRAY,fontsize=11)
save(fig,'03-cache-mechanism')

fig,ax=plt.subplots(figsize=(10.7,3.9))
vals=[e['pearson_visible_tokens_decode'] for e in D['episodes']]
bars=ax.barh([2,1,0],vals,color=[BLUE,TEAL,ORANGE],height=.52)
for bar,v,e in zip(bars,vals,D['episodes']):
    ax.text(v+.02,bar.get_y()+bar.get_height()/2,f"r = {v:.2f}  ·  {e['n']} 轮",va='center',color=INK)
ax.set(yticks=[2,1,0],yticklabels=['罐子 Hybrid','底层抽屉 Hybrid','微波炉 Hybrid'],xlim=(0,1.08),xticks=[0,.2,.4,.6,.8,1],xlabel='Pearson 相关系数 r（各任务内单独统计）')
ax.set_title('可见输出越长，Decode 通常越久',fontsize=18,weight='bold',loc='left',pad=20,color=INK)
ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
fig.text(.02,-.025,'本图展示相关系数，不是逐轮散点。正相关不等于因果，也不代表每个 Token 耗时固定。〔E1 / E3〕',fontsize=11,color=GRAY)
fig.tight_layout(); save(fig,'04-output-decode-correlation')
print('Generated 4 PNG and 4 SVG figures from recorded statistics.')
