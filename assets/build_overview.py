"""Rebuild reference A as semantic SVG objects at the original 1774 x 887 layout.

The diagram, labels, arrows, matrix and sieve are native vector geometry. Six
small colored clip-art illustrations reuse the separately vectorized assets.
The fidelity SVG preserves A's separately traced lettering. A second SVG keeps
live Comic Sans MS text for typing edits. Both figures use native vector geometry.
No image, pixel-grid, full-image tracing, or training result is embedded.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse, copy, json, math, random, sys, shutil, subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DEPS = ROOT.parent / 'build/figure/deps'
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
import fitz
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
args = ap.parse_args()
FONTS = {key: TTFont(args.font_dir / name) for key, name in
         [('normal', 'comic.ttf'), ('bold', 'comicbd.ttf')]}
W, H = 1774, 887
INK = '#080c10'
NAVY = '#09204e'
BLUE = '#034fdf'
CYAN = '#65cdf0'
PURPLE = '#751bd1'
GREEN = '#178327'
parts, defs, text_jobs = [], [], []
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

def emit(value):
    parts.append(value)

def group(name, transform=None, opacity=None):
    emit(f'<g id="{name}"' + (f' transform="{transform}"' if transform else '') +
         (f' opacity="{opacity}"' if opacity is not None else '') + '>')

def end():
    emit('</g>')

def path(d, fill='none', stroke=INK, sw=2, dash=None, opacity=None):
    emit(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
         ' stroke-linejoin="round" stroke-linecap="round"' +
         (f' stroke-dasharray="{dash}"' if dash else '') +
         (f' opacity="{opacity}"' if opacity is not None else '') + '/>')

def line(x1, y1, x2, y2, color=INK, sw=2, dash=None):
    path(f'M {x1},{y1} L {x2},{y2}', stroke=color, sw=sw, dash=dash)

def circle(x, y, radius, fill, stroke='none', sw=1):
    emit(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def ellipse(x, y, rx, ry, fill, stroke='none', sw=1, transform=None):
    emit(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' +
         (f' transform="{transform}"' if transform else '') + '/>')

def rect(x, y, w, h, fill, stroke='none', sw=1):
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def rough(x, y, w, h, fill, stroke=INK, sw=2, radius=10, wobble=1.4):
    # Sparse Bezier geometry: editable edges, deliberately not pixel contours.
    r = radius
    rng = random.Random(round(x * 17 + y * 101 + w * 13 + h * 7))
    j = lambda: round(rng.uniform(-wobble, wobble), 2)
    d = (f'M {x+r},{y} Q {x+w*.48},{y+j()} {x+w-r},{y} '
         f'Q {x+w},{y} {x+w},{y+r} Q {x+w+j()},{y+h*.5} {x+w},{y+h-r} '
         f'Q {x+w},{y+h} {x+w-r},{y+h} Q {x+w*.47},{y+h+j()} {x+r},{y+h} '
         f'Q {x},{y+h} {x},{y+h-r} Q {x+j()},{y+h*.54} {x},{y+r} '
         f'Q {x},{y} {x+r},{y} Z')
    path(d, fill, stroke, sw)

def gradient(name, c1, c2, direction='vertical', mid=None):
    xy = 'x1="0%" y1="0%" x2="100%" y2="0%"' if direction == 'horizontal' else 'x1="0%" y1="0%" x2="0%" y2="100%"'
    stops = f'<stop offset="0" stop-color="{c1}"/>'
    if mid:
        stops += f'<stop offset="0.52" stop-color="{mid}"/>'
    stops += f'<stop offset="1" stop-color="{c2}"/>'
    defs.append(f'<linearGradient id="{name}" {xy}>{stops}</linearGradient>')
    return f'url(#{name})'

def text(x, y, label, width=None, height=25, weight='bold', color=INK, name=None, rotate=None):
    """Place the visible glyph bounding box, making editable letters fit A's boxes."""
    name = name or f'label-{len(text_jobs)+1:03d}'
    font = FONTS[weight]
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    cursor, bounds = 0, []
    for ch in label:
        glyph_name = cmap.get(ord(ch), '.notdef')
        bp = BoundsPen(glyphs)
        glyphs[glyph_name].draw(bp)
        if bp.bounds:
            a,b,c,d = bp.bounds
            bounds.append((cursor+a,b,cursor+c,d))
        cursor += font['hmtx'].metrics[glyph_name][0]
    xmin, ymin = min(b[0] for b in bounds), min(b[1] for b in bounds)
    xmax, ymax = max(b[2] for b in bounds), max(b[3] for b in bounds)
    sy = height / (ymax-ymin)
    sx = width / (xmax-xmin) if width is not None else sy
    width = (xmax-xmin)*sx
    upm = font['head'].unitsPerEm
    # Use a normal editable font size with only horizontal shape matching.
    # Disabling kerning keeps the live label and outlined print copy identical.
    fontsize = sy * upm
    tx, ty = x-xmin*sx, y+ymax*sy
    transform = f'translate({tx:.5f} {ty:.5f}) scale({sx/sy:.7f} 1)'
    if rotate is not None:
        transform = f'rotate({rotate} {x+width/2:.5f} {y+height/2:.5f}) ' + transform
    emit(f'<text id="{name}" x="0" y="0" transform="{transform}" font-family="Comic Sans MS" '
         f'font-size="{fontsize:.7f}" font-weight="{weight}" fill="{color}" stroke="{color}" stroke-width="0.3" paint-order="stroke fill" style="font-kerning:none;font-variant-ligatures:none" '
         f'xml:space="preserve">{escape(label)}</text>')
    text_jobs.append((name, label, weight))
    return width

def arrow(points, color=NAVY, sw=2.4, head=11, half=6, dash=None):
    path('M '+' L '.join(f'{x},{y}' for x,y in points), stroke=color, sw=sw, dash=dash)
    x,y = points[-1]
    px,py = points[-2]
    angle = math.atan2(y-py, x-px)
    bx,by = x-head*math.cos(angle), y-head*math.sin(angle)
    path(f'M {x},{y} L {bx+half*math.sin(angle)},{by-half*math.cos(angle)} '
         f'L {bx-half*math.sin(angle)},{by+half*math.cos(angle)} Z', color, color, .6)

def dotted_arrow(x1, y, x2, color=NAVY):
    for x in range(round(x1), round(x2-11), 8):
        circle(x, y, 2.1, color)
    arrow([(x2-10,y),(x2,y)], color, 1.5, 10, 6)

def double_stroke(d, color=BLUE, fill='#b4e7fb', width=12):
    path(d, stroke='white', sw=width+5)
    path(d, stroke=color, sw=width)
    path(d, stroke=fill, sw=width-5)

def sparks(coords, color=INK, sw=3):
    for x1,y1,x2,y2 in coords:
        path(f'M {x1},{y1} Q {(x1+x2)/2-.7},{(y1+y2)/2} {x2},{y2}', stroke=color, sw=sw)

def scene(name, x, y, w, h, role, phase, muted=False):
    exact = ROOT/'clipart'/f'{name}.svg'
    asset = exact if exact.exists() else ROOT/'clipart'/f'{role}-{phase}.svg'
    src = ET.parse(asset).getroot()
    aw, ah = float(src.attrib['width']), float(src.attrib['height'])
    group(name)
    rough(x-2,y-2,w+4,h+4,'#fffaf2', '#171b1f', 1.9, 4, .8)
    group(name+'-colored-illustration', f'translate({x} {y}) scale({w/aw:.7f} {h/ah:.7f})', .43 if muted and not exact.exists() else None)
    for child in src:
        emit(ET.tostring(child, encoding='unicode'))
    end()
    if muted:
        rough(x-1,y-1,w+2,h+2,'none','#aebac0',1.3,3,.8)
    end()

def token_grid(name, x, y, tw=24, th=21, gx=8, gy=7, robot=False):
    group(name)
    fills = [token_purple, robot_token_green, token_orange] if robot else [token_blue, token_green, token_gold]
    for r in range(3):
        for c in range(3):
            xx, yy = x+c*(tw+gx), y+r*(th+gy)
            rough(xx,yy,tw,th,fills[c],INK,1.9,2,.45)
            path(f'M {xx+3},{yy+th-3} L {xx+tw-3},{yy+th-3}',stroke='#ffffff',sw=.7,opacity=.6)
    end()

def embedding(name, x, y, w=20, h=68, gap=11, purple=False):
    group(name)
    cols = ['#6d3ed6','#8e55f3','#b08efb'] if purple else ['#207ae9','#388dfa','#6dc2fc']
    for c, col in enumerate(cols):
        xx = x+c*(w+gap)
        rough(xx,y,w,h,col,NAVY,1.8,1.1,.35)
        line(xx+2,y+3,xx+2,y+h-3,'#d8f1ff' if not purple else '#f0dfff',.9)
    for xx in [x+3*(w+gap)+2, x+3*(w+gap)+10, x+3*(w+gap)+18]:
        circle(xx,y+h/2-1,2.2,NAVY)
    end()

def label_box(name, x,y,w,h,label,fill,stroke=PURPLE,ht=25,text_width=None):
    group(name)
    rough(x,y,w,h,fill,stroke,2.3,9,1.2)
    tw = text_width or w-28
    text(x+(w-tw)/2,y+(h-ht)/2,label,tw,ht)
    end()

bg_a = gradient('panel-a-wash','#d4f5fe','#d6f6fd',mid='#d7f6fe')
bg_b = gradient('panel-b-wash','#e3fde4','#d9fcdf',mid='#e5fde7')
bg_c = gradient('panel-c-wash','#d8f5fe','#d0f3fd',mid='#e5faff')
bg_d = gradient('panel-d-wash','#f4eefe','#f4eefe',mid='#f5effd')
title_a = gradient('title-blue','#d8f6fd','#cef4fe',mid='#d1f4fe')
title_b = gradient('title-green','#e0fde4','#e0fde4')
title_d = gradient('title-violet','#ecd8fe','#ebd7fe')
lane = gradient('source-lane','#dff8fe','#dff8fd',mid='#e8fbfe')
filter_fill = gradient('filter-wash','#eddafd','#ecdafd',mid='#f1e2fd')
retained_fill = gradient('retained-wash','#d6fdda','#e4fee7',mid='#ebfeee')
rejected_fill = gradient('rejected-wash','#f1f4f6','#e6edf2')
token_blue = gradient('token-blue','#49a5fa','#5cb6fc')
token_green = gradient('token-green','#89eb8f','#8def91')
robot_token_green = gradient('robot-token-green','#9ab4d4','#88c3b9',mid='#9cbcc9')
token_gold = gradient('token-gold','#fbc645','#fedb89')
token_purple = gradient('token-purple','#ae86fa','#a880f9')
token_orange = gradient('token-orange','#facc8e','#fbdbaa')
adapter_blue = gradient('adapter-blue','#c0e4fd','#c5e7fd','horizontal')
node_purple = gradient('node-purple','#e7cffc','#ead2fd',mid='#e9d3fd')
node_blue = gradient('node-blue','#c5dffd','#bbdbfd',mid='#c8e2fd')
node_green = gradient('node-green','#d8fddb','#d1fdd4')
node_yellow = gradient('node-yellow','#fef5b7','#fef3a9')
node_red = gradient('node-red','#fdd7d7','#fed7d5',mid='#fed4d3')
funnel_body = gradient('funnel-body','#d5d8eb','#7579a7','horizontal', '#a4a7d0')
funnel_rim = gradient('funnel-rim','#cbd5ea','#7988af',mid='#f1f4fd')
funnel_inside = gradient('funnel-inside','#3b4b70','#a5b2ce','horizontal', '#5f6d94')

# Major contours traced manually from reference A: same canvas and panel boxes.
group('panel-a-shared-representations')
path('M 13,71 Q 15,45 37,46 Q 445,38 816,41 Q 858,40 865,55 Q 871,76 871,143 L 868,583 '
     'Q 868,619 858,626 Q 849,635 799,632 Q 606,630 380,632 Q 203,630 74,634 '
     'Q 24,634 17,626 Q 9,622 9,595 Q 9,331 11,109 Z',bg_a,BLUE,3.2)
rough(20,91,836,252,lane,CYAN,1.8,20,3)
rough(20,360,836,255,lane,CYAN,1.8,20,3)
path('M 18,42 Q 19,32 35,29 Q 150,16 287,20 Q 411,18 458,24 Q 473,26 476,41 '
     'Q 481,65 470,76 Q 464,84 373,82 L 39,84 Q 15,85 15,69 Z',title_a,BLUE,3.4)
text(32,34,'(a) Shared representations',413,39)

for role, top, art_y, axis_y, tokens_y, adapter_y in [
        ('human',101,150,278,197,193), ('robot',373,422,556,484,478)]:
    is_robot = role == 'robot'
    group(role+'-representation-lane')
    title = 'Robot video' if is_robot else 'Human video'
    text(33,top,title,175 if not is_robot else 158,30,color='#072a70')
    text(215 if not is_robot else 204,top+7,'(place cube into tray)',198 if not is_robot else 205,23,weight='normal',color=NAVY)
    for i,phase in enumerate(['approach','grasp','place']):
        scene(role+'-'+phase,35+129*i,art_y,112,109,role,phase)
    arrow([(35,axis_y),(411,axis_y)],NAVY,2.2,12,7)
    for x,h in [(43,8),(125,5),(169,5),(260,4),(306,3),(350,6)]:
        line(x,axis_y-h,x,axis_y+5,NAVY,1.7)
    # T_H and T_R are represented by live subscript spans in a dedicated group.
    time_y = axis_y+15
    text(82,time_y,'Time (T',83,23,weight='normal',color=NAVY)
    text(168,time_y+13,'R' if is_robot else 'H',9,12,weight='normal',color=NAVY)
    text(184,time_y+1,'frames, variable length)',191,23,weight='normal',color=NAVY)
    arrow([(418,art_y+81),(445,art_y+81)],BLUE,2.5,11,7)
    cy = 402 if is_robot else 113
    text(431,cy,'Cached interaction tokens',211 if is_robot else 229,20 if is_robot else 22)
    text(478,cy+30,'512-D per token',135,19,weight='normal',color=NAVY)
    text(430,cy+53,'variable token count per frame',205 if is_robot else 232,18 if is_robot else 19,weight='normal',color=NAVY)
    token_grid(role+'-512d-token-set',453,tokens_y,23,21,9,7,is_robot)
    dotted_arrow(550,art_y+84,584)
    group(role+'-shared-adapter')
    path(f'M 591,{adapter_y} L 678,{adapter_y+20} L 678,{adapter_y+66} L 591,{adapter_y+85} Z',adapter_blue,BLUE,2.3)
    text(622,adapter_y+27,'f',19,32,color='#053995')
    text(638,adapter_y+40,'θ',12,19,color='#053995')
    end()
    arrow([(687,art_y+86),(714,art_y+86)],BLUE,2.4,11,7)
    label_y = 130 if not is_robot else 413
    text(733,label_y,'128-D',53,20,weight='normal',color=NAVY)
    text(693,label_y+28,'frame embeddings',147,20,weight='normal',color=NAVY)
    embedding(role+'-128d-embeddings',722,204 if not is_robot else 485,19,66,12,is_robot)
    end()
group('shared-weights-tie')
arrow([(635,320),(635,277)],PURPLE,2.2,13,7,'8 5')
arrow([(656,367),(656,481)],PURPLE,2.2,13,7,'8 5')
rough(555,323,167,43,node_purple,PURPLE,2.5,13,1.8)
text(566,333,'Shared weights',144,23,color='#3e126f')
end()
end()

group('panel-b-monotonic-matching')
path('M 888,53 Q 892,43 910,42 L 1299,41 Q 1317,41 1321,54 Q 1325,74 1322,276 '
     'L 1322,593 Q 1321,622 1311,627 Q 1304,634 1268,632 Q 1094,631 924,634 '
     'Q 893,634 889,621 Q 881,606 884,468 L 884,92 Q 884,59 888,53 Z',bg_b,'#187f24',3.1)
rough(898,94,412,527,'#edfef0','#a1e9ac',1.35,21,2.5)
path('M 896,40 Q 899,26 924,23 Q 980,17 1087,20 L 1265,21 Q 1302,21 1310,31 '
     'Q 1318,43 1316,62 Q 1317,79 1301,81 Q 1170,83 921,81 Q 895,82 894,65 Z',title_b,'#156b1d',3.4)
text(910,34,'(b) Monotonic matching',389,40)
text(918,108,'Cosine similarity matrix (schematic)',373,24)

group('cosine-similarity-matrix')
mx,my,ww,hh,nx,ny = 984,149,264,236,11,10
# A small schematic similarity grid, not an experimental heat map.
matrix_colors = [
 ['#d3edf9','#d9eff9','#bae2f7','#a9dcf6','#a5d9f6','#a9dbf6','#98d2f4','#54a8e4','#2a81cb','#135aac','#0c53a7'],
 ['#d2ecf9','#dcf0f9','#cae9f8','#b7e1f7','#b3dff7','#6eb9eb','#3e94d7','#2a7ec8','#1c69b8','#135bad','#277ac6'],
 ['#bde4f7','#cfebf9','#b4e0f7','#bce3f7','#69b6e9','#3990d5','#1c69b9','#125dac','#1f6cb8','#3488cf','#63b1e6'],
 ['#d1ebf8','#b5e1f7','#aadcf6','#5eaee5','#4297da','#1f71c0','#0e51a2','#1a64b1','#479bda','#76beeb','#7dc2ed'],
 ['#c0e5f7','#9bd4f4','#65b4e8','#368dd3','#2c81cb','#165dad','#1a67b6','#489cdb','#7fc4ee','#b1def6','#c0e6f8'],
 ['#8fcdf2','#73bded','#348ad1','#1c6bb9','#125aab','#1866b2','#4296d7','#7cc2ed','#b0def6','#c9e9f8','#daf0f9'],
 ['#6db7e9','#4196d7','#2173c1','#1457a2','#1c68b6','#4297d8','#80c4ee','#b7e0f6','#cdebf9','#b1dff7','#cfecf9'],
 ['#3c90d4','#1b6cbb','#145aaa','#1767b5','#4fa2dd','#85c7ef','#bbe3f7','#ceebf9','#b3e0f7','#9bd5f5','#c4e8f9'],
 ['#1d6ab9','#1457a5','#2d82ca','#51a3df','#9ad2f3','#bee4f7','#cceaf8','#b7e1f7','#a4daf6','#bbe4f8','#9fd7f5'],
 ['#14549d','#2578c4','#5dace4','#99d2f3','#cae9f8','#d2edf8','#bce4f8','#acddf6','#99d4f4','#adddf6','#8bcdf2']]
for row in range(ny):
    for col in range(nx):
        color = matrix_colors[row][col]
        rect(mx+col*ww/nx,my+row*hh/ny,ww/nx+.15,hh/ny+.15,color)
rough(mx,my,ww,hh,'none',NAVY,2.0,1,.5)
points = [(994,374),(1002,363),(1023,353),(1031,326),(1067,325),(1068,302),
          (1077,297),(1083,277),(1113,274),(1126,267),(1131,233),(1160,225),
          (1163,204),(1200,198),(1211,173),(1235,159)]
path('M '+' L '.join(f'{x},{y}' for x,y in points),stroke='#ec9b05',sw=4.5)
for x,y in points:
    circle(x,y,5.2,'#ffdc37','#f3a10b',1.6)
arrow([(963,385),(963,151)],NAVY,2.6,13,7)
arrow([(984,400),(1254,400)],NAVY,2.2,12,6.8)
text(888,261,'Robot time',126,24,rotate=-90)
text(1048,409,'Human time',128,24)
line(973,452,1015,452,'#f0a012',3.8)
circle(973,452,6,'#ffe244','#ef9c06',1.6)
circle(1015,452,6,'#ffe244','#ef9c06',1.6)
text(1036,443,'DTW path (monotonic)',194,21)
end()
arrow([(1108,466),(1108,485)],PURPLE,2.4,12,7)
label_box('two-way-dtw',987,489,241,43,'Two-way DTW',node_purple,ht=28,text_width=174)
arrow([(1018,533),(1018,555)],NAVY,2.1,12,7)
arrow([(1200,533),(1200,555)],NAVY,2.1,12,7)
rough(919,558,179,61,'#b4eafc','#0984d9',2.4,16,1.8)
text(939,569,'Robot-to-human',140,21)
text(969,594,'matches',78,18)
rough(1123,558,171,61,'#dcf7d0','#159e25',2.4,16,1.8)
text(1140,569,'Human-to-robot',140,21)
text(1172,594,'matches',78,18)
end()

# Outlined colored routes retain A's hand-drawn connector silhouette.
group('embedding-input-routes')
double_stroke('M 839,229 L 857,229 Q 867,229 867,241 L 867,319 L 892,319',BLUE,'#58c1fd',11)
path('M 891,301 L 910,319 L 891,337 L 891,325 L 866,325 L 866,313 L 891,313 Z','#58c1fd',BLUE,2.5)
double_stroke('M 839,501 L 858,501 Q 867,501 867,489 L 867,381 L 892,381',BLUE,'#58c1fd',11)
path('M 891,364 L 910,381 L 891,398 L 891,387 L 866,387 L 866,375 L 891,375 Z','#58c1fd',BLUE,2.5)
end()

group('panel-c-filtered-correspondences')
path('M 1341,63 Q 1342,33 1363,30 Q 1436,24 1500,30 L 1735,31 Q 1760,31 1763,55 '
     'Q 1768,98 1768,258 L 1766,591 Q 1765,622 1756,629 Q 1650,633 1486,632 '
     'L 1369,633 Q 1341,634 1338,613 Q 1334,481 1337,252 Z',bg_c,BLUE,3.1)
path('M 1350,39 Q 1353,26 1382,24 Q 1510,17 1628,22 Q 1738,21 1753,29 '
     'Q 1762,37 1760,62 Q 1761,80 1746,82 Q 1551,83 1381,82 Q 1344,84 1346,63 Z',title_a,BLUE,3.3)
text(1361,36,'(c) Filtered correspondences',386,34)

group('three-filter-gates')
rough(1344,118,153,455,filter_fill,PURPLE,2.6,18,3.2)
text(1354,139,'Filtering criteria',138,23)
group('hand-drawn-filter-sieve')
path('M 1391,216 Q 1404,242 1417,250 L 1422,268 Q 1428,273 1438,267 L 1434,247 '
     'Q 1448,228 1458,207 Z',funnel_body,INK,2.3)
path('M 1417,249 Q 1426,253 1434,247',stroke='#242c48',sw=1.7)
path('M 1425,253 L 1428,266 L 1433,265 L 1429,252 Z','#ecf1fd','none')
path('M 1458,197 L 1477,189 Q 1484,187 1486,192 Q 1488,198 1479,201 L 1464,205 Z',funnel_rim,INK,2)
path('M 1465,197 L 1479,193',stroke='#546583',sw=2)
path('M 1381,207 C 1388,192 1410,185 1438,186 C 1456,186 1465,190 1461,200 '
     'C 1457,212 1433,222 1409,224 C 1391,225 1376,219 1381,207 Z',funnel_rim,INK,2.3)
path('M 1387,207 C 1396,195 1419,191 1438,191 C 1453,191 1457,194 1453,199 '
     'C 1446,210 1424,218 1407,219 C 1392,220 1384,215 1387,207 Z',funnel_inside,'#313c59',1.6)
path('M 1391,211 Q 1410,195 1440,195',stroke='#cdd7eb',sw=1.7)
path('M 1394,216 Q 1415,218 1436,208',stroke='#dce4f3',sw=1.2)
sparks([(1365,187,1373,198),(1355,210,1366,213),(1378,253,1386,246),(1396,267,1401,257)],INK,3.4)
end()
label_box('similarity-gate',1353,286,134,46,'Similarity','#fffefe','#20184e',ht=24,text_width=91)
label_box('cycle-consistency-gate',1353,343,134,46,'Cycle consistency','#fffefe','#20184e',ht=22,text_width=124)
label_box('match-margin-gate',1353,399,134,46,'Match margin','#fffefe','#20184e',ht=22,text_width=112)
arrow([(1419,455),(1418,489)],PURPLE,2.5,13,7)
text(1384,510,'All pass',72,24)
sparks([(1363,501,1372,507),(1358,532,1368,527),(1374,550,1381,540),
        (1468,506,1478,500),(1469,525,1481,531),(1460,542,1466,551)],INK,3.4)
end()

group('retained-correspondences')
rough(1508,102,252,313,retained_fill,'#13a92a',2.8,16,2.6)
path('M 1512,149 Q 1563,145 1618,149 Q 1721,145 1756,149',stroke='#9bf1a6',sw=1.5)
text(1526,111,'Retained frame pairs',216,23)
text(1533,155,'Human',68,24,color='#006fbe')
text(1663,156,'Robot',53,24,color='#0061af')
for i, (yy,phase) in enumerate([(190,'grasp'),(296,'place')]):
    hh = 83 if i == 0 else 100
    scene('retained-human-'+phase,1522,yy,81,hh,'human',phase)
    scene('retained-robot-'+phase,1641,yy,79,hh,'robot',phase)
    yline = 233 if i == 0 else 342
    line(1605,yline,1638,yline,'#0dab32',5.2)
    circle(1605,yline,7.5,'#0dac33','#ffffff',1.3)
    circle(1639,yline,7.5,'#0dac33','#ffffff',1.3)
    path(f'M 1732,{yline-9} L 1740,{yline+1} L 1753,{yline-25}',stroke='#11ab35',sw=6)
end()
group('rejected-correspondence')
rough(1505,432,254,177,rejected_fill,'#a7b6c6',2.6,15,2)
text(1517,444,'Rejected',62,21)
text(1582,449,'(e.g., low similarity or inconsistent)',171,15,weight='normal',color=NAVY)
scene('rejected-human-approach',1521,485,79,100,'human','approach',True)
scene('rejected-robot-place',1642,485,79,100,'robot','place',True)
line(1604,530,1638,530,'#8b98a8',3,'5 5')
circle(1604,530,7,'#8795a5','#ffffff',1.4)
circle(1639,530,7,'#8795a5','#ffffff',1.4)
path('M 1732,521 L 1752,543 M 1751,520 L 1732,543',stroke='#677b92',sw=5.5)
end()
end()

group('dtw-filter-route')
double_stroke('M 1230,510 L 1284,510 Q 1299,510 1299,498 L 1299,329 Q 1299,321 1308,321 L 1328,321',PURPLE,'#c99bfa',10)
path('M 1327,303 L 1347,320 L 1328,339 L 1328,327 L 1300,327 L 1300,316 L 1327,316 Z','#c99bfa',PURPLE,2.6)
double_stroke('M 1489,315 L 1504,315',PURPLE,'#c99bfa',10)
path('M 1501,298 L 1519,315 L 1502,332 L 1502,321 L 1489,321 L 1489,309 L 1501,309 Z','#c99bfa',PURPLE,2.5)
end()

group('panel-d-shared-adapter-detail')
path('M 16,675 Q 17,653 35,650 Q 239,647 492,650 Q 850,648 1123,649 '
     'Q 1391,648 1567,650 Q 1720,646 1744,650 Q 1759,651 1759,669 '
     'Q 1763,747 1760,832 Q 1760,856 1747,859 Q 1655,863 1440,861 '
     'Q 1094,863 835,861 Q 483,861 242,863 L 43,863 Q 20,863 17,849 '
     'Q 12,823 15,756 Z',bg_d,'#8629ee',2.7)
path('M 17,666 Q 20,651 42,650 Q 274,647 527,647 Q 567,645 578,656 '
     'Q 585,667 580,687 Q 577,699 562,701 L 44,701 Q 16,702 17,682 Z',title_d,PURPLE,3.1)
text(35,660,'(d) Shared adapter (frozen at inference)',527,33)

token_grid('adapter-input-token-set',54,715,26,17,9,6)
for x in (164,172,181):
    circle(x,742,2.3,NAVY)
arrow([(195,742),(233,742)],NAVY,2.5,12,7)
rough(240,713,141,63,node_purple,PURPLE,2.2,10,1.5)
text(283,722,'Token',58,19)
text(268,746,'projection',97,24)
arrow([(388,742),(428,742)],NAVY,2.3,12,7)
token_grid('projected-token-set',438,715,25,17,9,6)
dotted_arrow(543,741,583)
rough(590,706,185,71,node_blue,BLUE,2.4,8,1.7)
text(641,719,'Learned',74,20)
text(605,744,'weighted pooling',154,24)
arrow([(777,722),(801,721),(801,688),(837,687)],NAVY,2.4,11,6)
arrow([(779,740),(838,739)],NAVY,2.4,11,6)
arrow([(778,755),(801,756),(801,788),(838,789)],NAVY,2.4,11,6)
label_box('pooled-frame-feature',843,667,198,39,'Pooled feature',node_green,'#138b24',ht=20,text_width=126)
label_box('first-temporal-difference',843,716,198,40,'First difference',node_yellow,'#a99e21',ht=20,text_width=132)
label_box('causal-temporal-convolution',843,769,199,39,'Causal Conv1D',node_red,'#b83032',ht=21,text_width=131)
path('M 1047,687 L 1072,687 L 1072,789 L 1048,789 M 1046,737 L 1072,738',stroke=NAVY,sw=2.6)
arrow([(1072,741),(1100,741)],NAVY,2.4,11,6)
label_box('concatenate-three-features',1106,711,132,64,'Concat',node_purple,PURPLE,ht=23,text_width=67)
arrow([(1245,741),(1289,739)],NAVY,2.4,11,6)
label_box('mlp-and-l2-normalization',1296,710,230,65,'MLP + L2 norm',node_purple,PURPLE,ht=25,text_width=155)
arrow([(1533,740),(1574,739)],NAVY,2.4,11,6)
embedding('adapter-128d-normalized-output',1586,708,20,63,13)

text(50,789,'512-D token set',143,21,weight='normal',color=NAVY)
text(73,818,'(K',24,26,weight='normal',color=NAVY)
text(99,832,'t',6,9,weight='normal',color=NAVY)
text(108,819,'× 512)',62,24,weight='normal',color=NAVY)
text(259,791,'512',32,21,weight='normal',color=NAVY)
arrow([(300,801),(319,801)],NAVY,1.7,7,4)
text(330,791,'128',32,21,weight='normal',color=NAVY)
text(608,791,'Valid tokens only',139,21,weight='normal',color=NAVY)
text(861,819,'k = 3 (t-2, t-1, t)',164,22,weight='normal',color=NAVY)
text(1147,791,'384-D',53,20,weight='normal',color=NAVY)
text(1358,791,'384',33,21,weight='normal',color=NAVY)
arrow([(1401,801),(1421,801)],NAVY,1.7,7,4)
text(1431,791,'128',32,21,weight='normal',color=NAVY)
text(1620,788,'128-D',52,20,weight='normal',color=NAVY)
text(1602,813,'(normalized)',101,24,weight='normal',color=NAVY)
end()

description = ('Editable reconstruction of reference A. Same-task complete human and robot sequences supply cached '
    'per-frame sets of 512-dimensional interaction tokens. The shared adapter produces normalized 128-dimensional '
    'frame embeddings. Cosine similarities enter monotonic two-way DTW and three filtering criteria. '
    'Reference lettering outlines and named vector groups; colored clip-art illustrations; no hand or gripper pose annotations. '
    'Illustrations and the similarity matrix are schematic, not experimental evidence.')
svg = (f'<svg xmlns="{NS}" xmlns:xlink="http://www.w3.org/1999/xlink" width="190mm" height="95mm" '
       f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="figure-title figure-description">\n'
       '<title id="figure-title">Offline Progress Aligner</title>\n'
       f'<desc id="figure-description">{escape(description)}</desc>\n'
       '<defs>'+''.join(defs)+'</defs>\n'
       f'<rect id="white-canvas" width="{W}" height="{H}" fill="#ffffff"/>\n' + '\n'.join(parts) + '\n</svg>\n')
svg_path = ROOT / 'overview.svg'
# Keep the independently editable live-text edition. The main view preserves the
# generated reference lettering, whose letterforms are not a system font.
(ROOT/'overview-editable-text.svg').write_text(svg,encoding='utf-8')
tree = ET.fromstring(svg)
live_text = {e.attrib['id']: e for e in tree.findall('.//{'+NS+'}text')}
parents = {child:parent for parent in tree.iter() for child in parent}
ref = ET.parse(ROOT/'lettering/reference-a.svg').getroot()
replaced = set()
for original in ref:
    if original.tag != '{'+NS+'}g':
        continue
    ident = original.attrib['id']
    group_copy = copy.deepcopy(original)
    extra = [i for i in group_copy.attrib.pop('data-replaces','').split(',') if i]
    ids_for_label = [ident]+extra
    group_copy.set('aria-label',' '.join(live_text[k].text or '' for k in ids_for_label))
    target = live_text[ident]
    parent = parents[target]
    index = list(parent).index(target)
    parent.remove(target)
    parent.insert(index,group_copy)
    replaced.add(ident)
    for key in extra:
        node = live_text[key]
        parents[node].remove(node)
        replaced.add(key)
assert replaced == set(live_text), (set(live_text)-replaced)
# The two dimension arrows are part of their original lettering groups.
for parent in tree.iter():
    for child in list(parent):
        if child.tag == '{'+NS+'}path' and child.attrib.get('d','').startswith(
            ('M 300,801','M 319,801','M 1401,801','M 1421,801')):
            parent.remove(child)
svg_path.write_bytes(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
assert not tree.findall('.//{'+NS+'}image')
ids = [el.attrib['id'] for el in tree.iter() if 'id' in el.attrib]
assert len(ids) == len(set(ids)), 'Duplicate object IDs'

# Outlined print export preserves the exact installed font without embedding a
# bitmap or depending on an SVG converter's font-substitution behavior.
outlined = copy.deepcopy(tree)
parents = {child:parent for parent in outlined.iter() for child in parent}
for item in list(outlined.iter('{'+NS+'}text')):
    font = FONTS[item.attrib.get('font-weight', 'normal')]
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale = float(item.attrib['font-size']) / font['head'].unitsPerEm
    g = ET.Element('{'+NS+'}g', {k:v for k,v in item.attrib.items() if k in ('id','transform','fill')})
    g.set('aria-label', item.text or '')
    cursor = 0
    for ch in item.text or '':
        gn = cmap.get(ord(ch), '.notdef')
        pen = SVGPathPen(glyphs)
        glyphs[gn].draw(pen)
        d = pen.getCommands()
        if d:
            ET.SubElement(g, '{'+NS+'}path', {'d':d,'transform':f'translate({cursor*scale:.6f} 0) scale({scale:.8f} {-scale:.8f})'})
        cursor += font['hmtx'].metrics[gn][0]
    parent = parents[item]
    index = list(parent).index(item)
    parent.remove(item)
    parent.insert(index,g)
work = ROOT.parent/'build/figure/native'
work.mkdir(parents=True,exist_ok=True)
outlined_path = work/'overview-print-outlines.svg'
outlined_path.write_bytes(ET.tostring(outlined,encoding='utf-8',xml_declaration=True))
# SVG-to-PDFKit preserves native axial gradients (MuPDF's SVG importer does not).
# The generated converter is a temporary export helper, not another figure source.
exporter = work/'export-pdf.cjs'
exporter.write_text('''const fs=require('node:fs');
const localRequire=require('node:module').createRequire(require('node:path').join(__dirname,'pdf-deps/package.json'));
const PDFDocument=localRequire('pdfkit');
const SVGtoPDF=localRequire('svg-to-pdfkit');
const src=process.argv[2],dest=process.argv[3];
const w=190*72/25.4,h=95*72/25.4;
const doc=new PDFDocument({size:[w,h],margin:0,compress:true,
  info:{Title:'Offline Progress Aligner — reference A',Subject:'Native vector reconstruction'}});
const stream=fs.createWriteStream(dest);doc.pipe(stream);
SVGtoPDF(doc,fs.readFileSync(src,'utf8'),0,0,{width:w,height:h,preserveAspectRatio:'xMinYMin meet'});
doc.end();stream.on('finish',()=>console.log('Vector PDF exported'));\n''',encoding='utf-8')
node = shutil.which('node') or 'C:/Program Files/nodejs/node.exe'
subprocess.run([node,str(exporter),str(outlined_path),str(ROOT/'overview.pdf')],check=True)
with fitz.open(ROOT/'overview.pdf') as pdf:
    assert len(pdf) == 1
    p = pdf[0]
    assert not p.get_images(full=True), 'Raster embedded in vector PDF'
    pix = p.get_pixmap(matrix=fitz.Matrix(W/p.rect.width,W/p.rect.width),alpha=False)
    pix.save(ROOT/'overview.png')
    info = {'canvas':[W,H], 'svg_bytes':svg_path.stat().st_size,
            'native_svg_text_labels':len(tree.findall('.//{'+NS+'}text')),
            'original_lettering_groups':len(ref.findall('{'+NS+'}g')),
            'live_text_alternative_labels':len(text_jobs),
            'named_vector_groups':len(tree.findall('.//{'+NS+'}g')),
            'svg_paths':len(tree.findall('.//{'+NS+'}path')),
            'pdf_bytes':(ROOT/'overview.pdf').stat().st_size,
            'pdf_raster_images':len(p.get_images(full=True)),
            'pdf_vector_drawings':len(p.get_drawings()),
            'source_reference':'overview-web-comic.png',
            'svg_text':'original reference lettering outlines; separate live Comic Sans MS edition',
            'pdf_text':'original reference lettering outlines',
            'reconstruction':'native diagram; separate original label outlines and small color-vector clip art; no whole-image tracing'}
    (work/'verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))
