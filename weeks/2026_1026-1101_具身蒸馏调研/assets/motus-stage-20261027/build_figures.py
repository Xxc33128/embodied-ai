"""Reproduce the report's static charts from its sanitized JSON snapshot.
Uses bundled ReportLab charts; PNG previews are rasterized from the same
in-memory vector drawing with PDFium. No network or training dependencies.
"""
from pathlib import Path
import json
from reportlab.graphics.shapes import Drawing, String, Rect, Line, Circle
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib.colors import HexColor, Color, white
import pypdfium2

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/"experiment_snapshot.json").read_text())
ROWS=DATA["results"]
assert all(r["completed"]==r["assigned"]==50 and sum(r["per_task"])==r["successes"] for r in ROWS)
INK=HexColor("#183247");TEAL=HexColor("#116C78");MUTED=HexColor("#526677")
ORANGE=HexColor("#AF5C00");GRID=HexColor("#E2E8ED")
def label(d,x,y,text,size=12,color=INK,font="Helvetica",anchor="start"):
    d.add(String(x,y,text,fontName=font,fontSize=size,fillColor=color,textAnchor=anchor))
def save(d,name):
    (HERE/(name+".svg")).write_text(renderSVG.drawToString(d),encoding="utf-8")
    raw=renderPDF.drawToString(d)
    doc=pypdfium2.PdfDocument(raw);page=doc[0];bitmap=page.render(scale=1.6)
    im=bitmap.to_pil();im.save(HERE/(name+".png"))
    bitmap.close();page.close();doc.close()
    print(name,im.size)

d=Drawing(960,500)
d.add(Rect(0,0,960,500,fillColor=white,strokeColor=None))
label(d,65,461,"LIBERO-Long: fixed-50 success rate",22,font="Helvetica-Bold")
label(d,65,437,"10 tasks x 5 official initial states | completed evaluations only",12,color=MUTED)
lp=LinePlot();lp.x=85;lp.y=100;lp.width=800;lp.height=295
full=[(r["step"],r["success_rate"]*100) for r in ROWS if r["step"]>=1000]
lp.data=[full,[(200,0)],[(200,0),(1000,2)]]
lp.joinedLines=1
lp.lines[0].strokeColor=TEAL;lp.lines[0].strokeWidth=3
lp.lines[0].symbol=makeMarker("FilledCircle");lp.lines[0].symbol.fillColor=TEAL;lp.lines[0].symbol.strokeColor=white;lp.lines[0].symbol.size=8
lp.lines[1].strokeColor=ORANGE;lp.lines[1].symbol=makeMarker("FilledDiamond");lp.lines[1].symbol.fillColor=ORANGE;lp.lines[1].symbol.size=9
lp.lines[2].strokeColor=HexColor("#9BA8B2");lp.lines[2].strokeWidth=1;lp.lines[2].strokeDashArray=[4,4]
lp.xValueAxis.valueMin=0;lp.xValueAxis.valueMax=8500;lp.xValueAxis.valueSteps=list(range(0,8001,1000))
lp.yValueAxis.valueMin=0;lp.yValueAxis.valueMax=100;lp.yValueAxis.valueSteps=list(range(0,101,20));lp.yValueAxis.labelTextFormat="%d%%"
for ax in (lp.xValueAxis,lp.yValueAxis):
    ax.labels.fontName="Helvetica";ax.labels.fontSize=11;ax.strokeColor=GRID;ax.tickStrokeColor=GRID
lp.yValueAxis.visibleGrid=1;lp.yValueAxis.gridStrokeColor=GRID
d.add(lp)
for r in ROWS:
    if r["step"]==200:continue
    x=85+r["step"]/8500*800;y=100+r["success_rate"]*100*2.95
    label(d,x,y+14,str(round(r["success_rate"]*100))+"%",12,TEAL,font="Helvetica-Bold",anchor="middle")
label(d,85+200/8500*800+7,118,"0% pilot",10,ORANGE)
label(d,485,64,"Absolute global optimizer step",13,anchor="middle")
label(d,65,412,"Success rate",11,color=MUTED)
d.add(Circle(74,31,4,fillColor=TEAL,strokeColor=None));label(d,87,27,"Full TRAIN windows",11,color=MUTED)
d.add(Rect(252,27,8,8,fillColor=ORANGE,strokeColor=None));label(d,268,27,"200-step pilot",11,color=MUTED)
label(d,65,9,"Pilot: 900 TRAIN windows. The 200 -> 1,000 segment switches to all 30,639 TRAIN windows.",10,color=MUTED)
save(d,"training_curve")

cols=[r for r in ROWS if r["step"]>=1000]
d=Drawing(960,580);d.add(Rect(0,0,960,580,fillColor=white,strokeColor=None))
label(d,65,542,"Per-task outcomes across checkpoints",22,font="Helvetica-Bold")
label(d,65,518,"Each cell is successful episodes / 5; the same five initial states are reused.",12,color=MUTED)
x0=190;top=470;cw=83;ch=34
for c,r in enumerate(cols):label(d,x0+(c+.5)*cw,top+15,str(r["step"]),12,font="Helvetica-Bold",anchor="middle")
for task in range(10):
    label(d,145,top-(task+.5)*ch-4,"Task "+str(task),12,anchor="end")
    for c,r in enumerate(cols):
        value=r["per_task"][task];t=value/5
        lo=(.94,.96,.97);hi=(.07,.42,.47)
        color=Color(*(lo[i]+t*(hi[i]-lo[i]) for i in range(3)))
        x=x0+c*cw;y=top-(task+1)*ch
        d.add(Rect(x,y,cw,ch,fillColor=color,strokeColor=white,strokeWidth=2))
        label(d,x+cw/2,y+ch/2-4,str(value)+"/5",12,white if value>=3 else INK,font="Helvetica-Bold",anchor="middle")
d.add(Rect(x0,top-9*ch,cw*len(cols),ch,fillColor=None,strokeColor=ORANGE,strokeWidth=2))
label(d,65,106,"Task 8: both moka pots onto the stove. 3/5 at step 5,000, then 0/5 at 6,000-8,000.",11,color=ORANGE)
label(d,65,84,"Overall: 35/50 at both 7,000 and 8,000; individual cases still change (9 improved, 9 regressed).",11,color=MUTED)
for value in range(6):
    t=value/5;lo=(.94,.96,.97);hi=(.07,.42,.47);color=Color(*(lo[i]+t*(hi[i]-lo[i]) for i in range(3)))
    x=222+value*91;d.add(Rect(x,36,40,18,fillColor=color,strokeColor=white))
    label(d,x+20,20,str(value)+"/5",10,anchor="middle")
label(d,65,9,"Task names and provenance are provided in the report and experiment_snapshot.json.",10,color=MUTED)
save(d,"task_heatmap")
