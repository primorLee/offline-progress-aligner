"""Build Comic Sans MS SVG/PDF with vectorized, colored hand-drawn clip-art.

Requires PyMuPDF, ReportLab, svglib. Optional --retrace requires Pillow/vtracer.
Supply installed Comic Sans MS via --font-dir. Fonts are not redistributed.
Scenes/heatmap are conceptual, never experiment evidence. See hero-prompt.md.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse, math, random, sys, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
deps=ROOT.parent/'build/figure/deps'
if deps.exists(): sys.path.insert(0,str(deps))
import fitz
from reportlab.graphics import renderPDF
from svglib.fonts import FontMap
from svglib.svglib import svg2rlg
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--font-dir',type=Path,default=Path('C:/Windows/Fonts'))
ap.add_argument('--retrace',action='store_true')
args=ap.parse_args()
fm=FontMap()
for file,weight in [('comic.ttf','normal'),('comicbd.ttf','bold')]:
    if not (args.font_dir/file).exists(): raise SystemExit('Install Comic Sans MS or supply --font-dir.')
    assert fm.register_font('Comic Sans MS',str(args.font_dir/file),weight=weight)[1]
if args.retrace:
    from PIL import Image
    import vtracer
    work=ROOT.parent/'build/figure'; work.mkdir(parents=True,exist_ok=True)
    (ROOT/'clipart').mkdir(exist_ok=True)
    im=Image.open(ROOT/'overview-web-comic.png').convert('RGB')
    crops={'human-approach':(36,151,147,258),'human-grasp':(165,151,276,258),
           'human-place':(294,151,405,258),'robot-approach':(37,423,147,530),
           'robot-grasp':(165,423,276,530),'robot-place':(294,423,405,530)}
    for name,bounds in crops.items():
        crop=work/(name+'.png'); im.crop(bounds).save(crop)
        vtracer.convert_image_to_svg_py(str(crop),str(ROOT/'clipart'/(name+'.svg')),
            colormode='color',hierarchical='stacked',mode='spline',filter_speckle=3,
            color_precision=5,layer_difference=24,corner_threshold=65,
            length_threshold=3.5,max_iterations=10,splice_threshold=45,path_precision=2)
W,H=2000,1050
INK='#172136'; SUB='#425371'; BLUE='#0758C4'; GREEN='#217849'; PURPLE='#7130AF'; GRAY='#8E99A8'; AMBER='#EAA91C'
parts=[]
def emit(s): parts.append(s)
def group(name,transform=None): emit(f'<g id="{name}"'+(f' transform="{transform}"' if transform else '')+'>')
def end(): emit('</g>')
def text(x,y,value,size=27,color=INK,weight='normal',anchor='start',rotate=None):
    tr=f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    emit(f'<text xml:space="preserve" x="{x}" y="{y}" font-family="Comic Sans MS" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{tr}>{escape(value)}</text>')
def path(d,fill='none',stroke=INK,sw=2.4,dash=None):
    dash=f' stroke-dasharray="{dash}"' if dash else ''
    emit(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
def rect(x,y,w,h,fill='white',stroke='none',sw=1):
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def rough(x,y,w,h,fill,stroke=INK,sw=3,r=13):
    rng=random.Random(int(x*17+y*101+w*11)); j=lambda:rng.uniform(-2,2)
    d=(f'M {x+r},{y} Q {x+w*.5},{y+j()} {x+w-r},{y} Q {x+w},{y} {x+w},{y+r} '
       f'Q {x+w+j()},{y+h*.5} {x+w},{y+h-r} Q {x+w},{y+h} {x+w-r},{y+h} '
       f'Q {x+w*.5},{y+h+j()} {x+r},{y+h} Q {x},{y+h} {x},{y+h-r} '
       f'Q {x+j()},{y+h*.5} {x},{y+r} Q {x},{y} {x+r},{y} Z')
    path(d,fill,stroke,sw)
def line(x1,y1,x2,y2,color=INK,sw=2.6,dash=None): path(f'M {x1},{y1} L {x2},{y2}',stroke=color,sw=sw,dash=dash)
def arrow(points,color=INK,sw=3,dashed=False,fat=False):
    pts=list(points); d='M '+' L '.join(f'{x},{y}' for x,y in pts)
    if fat:
        path(d,stroke=color,sw=10); path(d,stroke='#DCEFFC' if color==BLUE else '#DCC6F5',sw=5.5)
    else: path(d,stroke=color,sw=sw,dash='7 6' if dashed else None)
    x,y=pts[-1]; px,py=pts[-2]; a=math.atan2(y-py,x-px)
    length,half=(17,9) if fat else (12,5.5); bx,by=x-length*math.cos(a),y-length*math.sin(a)
    path(f'M {x},{y} L {bx+half*math.sin(a)},{by-half*math.cos(a)} L {bx-half*math.sin(a)},{by+half*math.cos(a)} Z',color,color,1.4)
def dot(x,y,r=4,fill=INK,stroke='none',sw=1): emit(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def sparks(x,y,color=INK,scale=1):
    for a,b,c,d in [(-12,-12,-23,-23),(0,-18,-1,-33),(13,-12,22,-25)]: line(x+a*scale,y+b*scale,x+c*scale,y+d*scale,color,3.5*scale)
def scene(name,x,y,w,role,phase,muted=False):
    key=f'{role}-{("approach","grasp","place")[phase]}'
    t=ET.parse(ROOT/'clipart'/(key+'.svg')).getroot(); aw,ah=float(t.attrib['width']),float(t.attrib['height'])
    group(name); rough(x-4,y-4,w+8,w+8,'#FFFFFF','#FFFFFF',4,5)
    group(name+'-illustration',f'translate({x} {y}) scale({w/aw} {w/ah})')
    for child in t:
        if muted:
            for element in child.iter():
                for attr in ('fill','stroke'):
                    value=element.attrib.get(attr,'')
                    if value.startswith('#') and len(value)==7:
                        channels=[int(value[i:i+2],16) for i in (1,3,5)]
                        element.attrib[attr]='#%02X%02X%02X'%tuple(round(c*.40+255*.60) for c in channels)
        emit(ET.tostring(child,encoding='unicode'))
    end()
    rough(x-2,y-2,w+4,w+4,'none',GRAY if muted else INK,2.3,4); end()
def tokens(name,x,y,color=BLUE,big=False):
    group(name); cols=['#8DD7FF','#AEE6A9','#FFDC7E'] if color==BLUE else ['#BA8DE7','#9BC9F5','#F7CAAF']
    tw,th,gap=(27,19,8) if big else (22,14,8)
    for row in range(3):
        for c in range(3): rough(x+c*(tw+gap),y+row*(th+6),tw,th,cols[c],INK,1.7,2)
    text(x+3*(tw+gap)+1,y+30,'...',25,color,'bold'); end()
def embedding(name,x,y,color=BLUE):
    group(name); cols=['#225B9D','#5C9FDD','#A2DFFF'] if color==BLUE else ['#7350A1','#A674D2','#D7ACED']
    for i in range(3):
        rough(x+i*26,y,18,65,cols[i],INK,1.9,2); line(x+i*26+4,y+6,x+i*26+4,y+46,'#FFFFFF',1.3)
    end()
def box(name,x,y,w,h,rows,fill='#F5EBFC',stroke=PURPLE,size=26):
    group(name); rough(x,y,w,h,fill,stroke,2.7,10); first=y+h/2-(len(rows)-1)*16+9
    face=fitz.Font(fontfile=str(args.font_dir/'comicbd.ttf'))
    fitted=min(size,*(size*(w-20)/max(1,face.text_length(v,fontsize=size)) for v in rows))
    for i,val in enumerate(rows): text(x+w/2,first+i*32,val,fitted,INK,'bold','middle')
    end()

group('overview')
rough(16,67,930,606,'#F1FAFE',BLUE,3.5); rough(965,67,416,606,'#F4FCEF',GREEN,3.5); rough(1401,67,580,606,'#F3FAFF',BLUE,3.5)
for x,w,title,fill,edge in [(29,645,'(a) Shared representations','#DBF4FF',BLUE),(980,379,'(b) Monotonic matching','#E3F7DA',GREEN),(1416,542,'(c) Filtered correspondences','#E0F2FF',BLUE)]:
    rough(x,24,w,72,fill,edge,3.4); text(x+20,73,title,29 if title.startswith('(b)') else 32,weight='bold')
for role,y,color in [('human',206,BLUE),('robot',452,PURPLE)]:
    group(role+'-source-lane'); rough(31,y-90,895,211,'#E8F7FE' if role=='human' else '#F1ECFC','#84C9EE' if role=='human' else '#C1A1DF',1.8)
    text(48,y-42,'Human video' if role=='human' else 'Robot video',31,color,'bold'); text(48,y-13,'Same task, different timing',21,SUB)
    for i in range(3): scene(f'{role}-context-{i}',50+112*i,y,99,role,i)
    arrow([(51,y+111),(372,y+111)],SUB,1.8)
    for tx in ([65,129,348] if role=='human' else [61,259,323]): line(tx,y+105,tx,y+117,SUB,1.7)
    text(448,y-42,'Cached tokens',27,color,'bold','middle'); text(448,y-11,'512-D / token',22,SUB,anchor='middle')
    tokens(role+'-cached-token-sets',401,y+21,color); arrow([(378,y+52),(394,y+52)],color,2.3,True)
    text(466,y+104,'variable K / frame',19,SUB,anchor='middle'); arrow([(522,y+52),(563,y+52)],color,fat=True)
    box(role+'-shared-adapter',581,y+14,137,77,['fθ'],'#D5F0FD' if role=='human' else '#E7D7F8',color,39)
    arrow([(722,y+52),(767,y+52)],color,fat=True); embedding(role+'-frame-embeddings',787,y+20,color)
    text(823,y-41,'128-D',29,color,'bold','middle'); text(823,y-10,'frame embeddings',21,SUB,anchor='middle'); end()
line(648,298,648,357,PURPLE,2.4,'6 6'); line(648,400,648,458,PURPLE,2.4,'6 6')
rough(557,354,181,44,'#EAD9FC',PURPLE,2.3,8); text(647,385,'Shared weights',22,PURPLE,'bold','middle')
text(47,636,'Video context · cached token sets · shared adapter',24,SUB)
path('M 880,258 L 936,258 L 936,186 L 954,186',stroke=BLUE,sw=4)
path('M 880,504 L 947,504 L 947,186 L 954,186',stroke=PURPLE,sw=3)
dot(954,186,3.5)
arrow([(954,186),(1180,186),(1180,201)],INK,2.4)
text(1173,146,'Cosine similarity',29,weight='bold',anchor='middle'); text(1173,177,'matrix (schematic)',23,SUB,anchor='middle')
mx,my,cell,nh,nr=1034,202,29,10,8; centers=[0,1,3,4,4,6,7,9]
for i in range(nr):
    for j in range(nh):
        v=math.exp(-((j-centers[i])/1.8)**2); rgb=tuple(round(a+(b-a)*v) for a,b in zip((238,248,251),(66,127,172)))
        rect(mx+j*cell,my+(nr-1-i)*cell,cell,cell,'#%02X%02X%02X'%rgb,'#DAEAF2',.7)
indices=[(0,0),(1,1),(2,2),(2,3),(3,4),(4,4),(5,5),(5,6),(6,7),(6,8),(7,9)]
pp=[(mx+(j+.5)*cell,my+(nr-i-.5)*cell) for i,j in indices]
path('M '+' L '.join(f'{x},{y}' for x,y in pp),stroke=AMBER,sw=5)
for x,y in pp: dot(x,y,4.7,'#FFD442','#C98D00',1)
rough(mx,my,nh*cell,nr*cell,'none',INK,2.4,3)
arrow([(mx,my+nr*cell+12),(mx+nh*cell,my+nr*cell+12)],INK,2)
arrow([(mx-17,my+nr*cell),(mx-17,my)],INK,2)
text(mx+nh*cell/2,481,'Human time',24,SUB,anchor='middle'); text(mx-34,my+nr*cell/2,'Robot time',24,SUB,anchor='middle',rotate=-90)
arrow([(1179,491),(1179,516)],GREEN)
box('two-way-dtw',1058,524,241,60,['Two-way DTW'],'#E9DBF8',PURPLE,29)
text(1179,618,'R to H     H to R',26,PURPLE,'bold','middle'); text(1179,650,'Complete sequences · offline',21,SUB,anchor='middle')
arrow([(1302,554),(1390,554),(1390,302),(1421,302)],PURPLE,fat=True)
group('filtering-gates'); rough(1431,136,213,394,'#F1E5FD',PURPLE,2.9); text(1538,172,'Filtering criteria',24,weight='bold',anchor='middle')
path('M 1467,210 Q 1535,196 1608,209 L 1561,258 L 1561,286 L 1535,295 L 1534,258 Z','#C7A6E4',INK,2.6)
path('M 1467,210 Q 1535,184 1608,209 Q 1537,236 1467,210 Z','#FCF6FF',INK,2.6)
for k in range(7): line(1488+k*14,202,1493+k*12,218,'#8C6CA6',1.1)
for yy in (205,212,219): path(f'M 1483,{yy} Q 1540,{yy-8} 1593,{yy}',stroke='#8C6CA6',sw=1)
sparks(1538,195,PURPLE,.58)
for yy,label in [(315,'Similarity'),(377,'Cycle consistency'),(439,'Match margin')]:
    face=fitz.Font(fontfile=str(args.font_dir/'comicbd.ttf'))
    label_size=min(23,23*164/face.text_length(label,fontsize=23))
    rough(1445,yy,184,47,'#FFFFFF',PURPLE,2.1,7); text(1537,yy+31,label,label_size,weight='bold',anchor='middle')
text(1537,516,'ALL must pass',24,PURPLE,'bold','middle'); end()
group('retained-and-rejected-pairs'); rough(1664,126,296,336,'#EEFAEA',GREEN,2.4)
text(1811,165,'Retained frame pairs',27,GREEN,'bold','middle'); text(1729,206,'Human',23,BLUE,'bold','middle'); text(1875,206,'Robot',23,PURPLE,'bold','middle')
for row,phase in enumerate((1,2)):
    yy=224+118*row; scene(f'retained-human-{row}',1685,yy,88,'human',phase); scene(f'retained-robot-{row}',1830,yy,88,'robot',phase)
    line(1780,yy+44,1821,yy+44,GREEN,4); dot(1780,yy+44,5,GREEN); dot(1821,yy+44,5,GREEN)
    path(f'M 1929,{yy+47} L 1937,{yy+56} L 1951,{yy+33}',stroke=GREEN,sw=4)
rough(1664,487,296,172,'#EEF0F2',GRAY,2.1); text(1812,518,'Rejected candidate',25,SUB,'bold','middle')
scene('rejected-human',1685,542,88,'human',0,True); scene('rejected-robot',1830,542,88,'robot',2,True)
line(1780,586,1821,586,GRAY,3,'5 5'); path('M 1930,575 L 1948,596 M 1948,575 L 1930,596',stroke=GRAY,sw=4)
arrow([(1647,509),(1655,509),(1655,290),(1676,290)],GREEN,2.2); arrow([(1655,403),(1676,403)],GREEN,2.2)
arrow([(1647,485),(1654,485),(1654,586),(1676,586)],GRAY,2.2,True); end()
text(1537,582,'Cosine ≥ 0.50',21,SUB,anchor='middle'); text(1537,613,'Cycle ≤ 2 frames*',21,SUB,anchor='middle'); text(1537,644,'Margin ≥ 0.01',21,SUB,anchor='middle'); end()
group('shared-adapter-detail'); rough(17,740,1965,280,'#FAF6FD',PURPLE,3); rough(31,707,906,64,'#F0E3FB',PURPLE,3)
text(52,751,'(d) Shared adapter: learned pooling + causal context',31,weight='bold'); text(1954,729,'Frozen at inference',24,PURPLE,'bold','end')
tokens('detail-input-tokens',48,840,BLUE,True); text(98,956,'512-D tokens',24,BLUE,anchor='middle'); text(98,989,'K per frame',21,SUB,anchor='middle')
arrow([(172,876),(201,876)]); box('token-projection',215,833,170,83,['LN + linear','GELU'],'#EEE0F9',PURPLE,25); text(300,956,'512 to 128',23,SUB,anchor='middle')
arrow([(389,876),(430,876)]); box('learned-weighted-pooling',444,833,218,83,['Learned scores','Weighted pooling'],'#DDF2FC',BLUE,24); text(553,956,'Softmax · valid tokens only',21,SUB,anchor='middle')
arrow([(666,876),(699,876)]); dot(699,876,4)
arrow([(699,876),(718,876),(718,810),(754,810)]); arrow([(699,876),(754,876)]); arrow([(699,876),(718,876),(718,942),(754,942)])
box('pooled-feature',769,787,223,47,['Pooled feature'],'#E0F5DA',GREEN,24); box('first-difference',769,853,223,47,['First difference'],'#FFF0C8','#AB7D12',24)
box('causal-conv1d',769,919,223,47,['Causal Conv1D'],'#FADDDD','#A95168',24); text(880,996,'k = 3; past and current',21,SUB,anchor='middle')
for yy in (810,876,942): path(f'M 996,{yy} L 1040,{yy} L 1040,876',stroke=INK,sw=2.5)
arrow([(1040,876),(1080,876)]); box('concatenate',1094,833,166,83,['Concat'],'#EBDDFA',PURPLE,29); text(1177,956,'384-D',23,SUB,anchor='middle')
arrow([(1264,876),(1306,876)]); box('readout-normalize',1320,833,267,83,['LN + MLP + L2 norm'],'#EEE1FA',PURPLE,24); text(1453,956,'384 to 128',23,SUB,anchor='middle')
arrow([(1592,876),(1640,876)]); embedding('normalized-output',1654,845,BLUE)
text(1855,870,'128-D frame',27,BLUE,'bold','middle'); text(1855,910,'unit-norm embedding',21,SUB,anchor='middle'); end()
text(31,1041,'* Cycle error in sampled robot frames. Margin: competitors >3 human frames away; none: reject.',19,SUB)
text(1960,1041,'Offline Progress Aligner',19,SUB,anchor='end')
description=('Same-task complete human/robot sequences provide cached per-frame sets of 512-D interaction tokens. '
    'Shared f_theta produces unit-norm 128-D embeddings. Endpoint-constrained DTW runs in both directions '
    'with cosine cost. Similarity, cycle-return, and distant-margin gates filter heuristic pseudo-correspondences. '
    'No time IDs enter the adapter. Scenes and heatmap are schematic. Comic Sans MS text remains editable; '
    'six original web-generated clip-art scenes are traced into colored vector paths.')
svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="190mm" height="{190*H/W:.5f}mm" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">\n'
     '<title id="title">Offline Progress Aligner — hand-drawn method overview</title>\n'
     f'<desc id="description">{escape(description)}</desc>\n<rect width="{W}" height="{H}" fill="white"/>\n'+'\n'.join(parts)+'</svg>\n')
svg_path=ROOT/'overview.svg'; svg_path.write_text(svg,encoding='utf-8'); tree=ET.fromstring(svg)
assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
ids=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]; assert len(ids)==len(set(ids))
drawing=svg2rlg(str(svg_path),font_map=fm); assert drawing is not None
pdf_path=ROOT/'overview.pdf'; renderPDF.drawToFile(drawing,str(pdf_path),title='Offline Progress Aligner — hand-drawn overview')
with fitz.open(pdf_path) as check:
    page=check[0]; assert len(check)==1 and not page.get_images(full=True)
    assert len(page.get_drawings())>500
    fonts=page.get_fonts(); assert any('ComicSansMS' in f[3] for f in fonts),fonts
    pix=page.get_pixmap(matrix=fitz.Matrix(W/page.rect.width,W/page.rect.width),alpha=False); pix.save(ROOT/'overview.png')
    print({'raster_images':0,'vector_paths':len(page.get_drawings()),'live_svg_text':len(tree.findall('.//{http://www.w3.org/2000/svg}text')),'pdf_fonts':[f[3] for f in fonts]})
