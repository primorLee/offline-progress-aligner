"""Build the editable method figure and export its SVG to vector PDF/PNG.

Diagram content follows model.py, aligner.py and dtw.py. All scenes and matrix
colors are schematic. No experiment measurements or raster assets are used.
Requires PyMuPDF (PDF export / preview); Python's standard library authors SVG.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import math
import xml.etree.ElementTree as ET
import fitz

ROOT = Path(__file__).resolve().parent
W, H = 1800, 920
INK = '#1D2735'
SUB = '#566172'
LIGHT = '#DCE2E8'
HUMAN = '#167D87'
ROBOT = '#285A9A'
GREEN = '#248265'
AMBER = '#C88B00'
GRAY = '#9AA3AE'
parts, definitions = [], []


def emit(value):
    parts.append(value)


def group(name, transform=None, opacity=None):
    extra = (f' transform="{transform}"' if transform else '')
    extra += (f' opacity="{opacity}"' if opacity is not None else '')
    emit(f'<g id="{name}"{extra}>')


def end():
    emit('</g>')


def text(x, y, value, size=27, color=INK, weight='normal', anchor='start', italic=False, rotate=None):
    rotation = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    emit(f'<text xml:space="preserve" x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
         f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
         f'font-style="{"italic" if italic else "normal"}"{rotation}>{escape(value)}</text>')


def mathtext(x, y, fragments, size=27, color=INK, anchor='start'):
    """Editable typographic subscripts/superscripts, without Unicode fake glyphs."""
    widths=[fitz.Font('heit' if italic else 'helv').text_length(value,fontsize=size*scale)
            for value,scale,shift,italic in fragments]
    total=sum(widths)
    cursor=x-total/2 if anchor=='middle' else x-total if anchor=='end' else x
    group(f'math-label-{len(parts)}')
    for (value,scale,shift,italic),width in zip(fragments,widths):
        # Explicit offsets keep leading spaces consistent across SVG/PDF renderers.
        leading=len(value)-len(value.lstrip(' '))
        gap=fitz.Font('heit' if italic else 'helv').text_length(' '*leading,fontsize=size*scale)
        text(cursor+gap,y+shift,value.strip(' '),size*scale,color,italic=italic)
        cursor+=width
    end()


def rect(x, y, w, h, fill='white', stroke=LIGHT, sw=1.5, radius=0):
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, color=INK, sw=2.2, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    emit(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
         f'stroke-width="{sw}" stroke-linecap="round"{extra}/>')


def path(d, fill='none', stroke=INK, sw=2, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    emit(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
         f'stroke-linejoin="round" stroke-linecap="round"{extra}/>')


def arrow(points, color=INK, sw=2.3, dashed=False):
    pts = list(points)
    path('M ' + ' L '.join(f'{x},{y}' for x, y in pts), stroke=color, sw=sw, dash='6 5' if dashed else None)
    x, y = pts[-1]
    px, py = pts[-2]
    a = math.atan2(y-py, x-px)
    length, half = 11, 4.5
    bx, by = x-length*math.cos(a), y-length*math.sin(a)
    p1 = (bx+half*math.sin(a), by-half*math.cos(a))
    p2 = (bx-half*math.sin(a), by+half*math.cos(a))
    path(f'M {x},{y} L {p1[0]},{p1[1]} L {p2[0]},{p2[1]} Z', fill=color, stroke=color, sw=0.6)


def dot(x, y, r=4, fill=INK):
    emit(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')


def scene(name, x, y, w, role, phase, muted=False):
    """Original vector hand/gripper glyphs; illustrative, not simulator frames."""
    h = w*0.88
    clip_id = name+'-clip'
    definitions.append(f'<clipPath id="{clip_id}"><rect width="100" height="88" rx="3"/></clipPath>')
    group(name, f'translate({x} {y}) scale({w/100})', 0.40 if muted else None)
    emit(f'<g clip-path="url(#{clip_id})">')
    rect(0, 0, 100, 88, '#FAFBFC', 'none')
    rect(0, 68, 100, 20, '#EBECEB', 'none')
    line(0, 68, 100, 68, '#C5CCD0', 1.2)
    cx, cy = ((35, 64), (49, 46), (69, 64))[phase]
    # Cube uses flat faces, not a decorative gradient.
    path(f'M {cx-9},{cy-8} L {cx+5},{cy-8} L {cx+5},{cy+6} L {cx-9},{cy+6} Z', '#E8B747', '#967529', 0.6)
    path(f'M {cx-9},{cy-8} L {cx-4},{cy-13} L {cx+10},{cy-13} L {cx+5},{cy-8} Z', '#F4D88B', '#967529', 0.6)
    path(f'M {cx+5},{cy-8} L {cx+10},{cy-13} L {cx+10},{cy+1} L {cx+5},{cy+6} Z', '#C28E2C', '#967529', 0.6)
    if role == 'human':
        hx, hy = ((45, 39), (49, 35), (69, 51))[phase]
        path(f'M 98,1 L 98,20 C 82,21 {hx+25},{hy-11} {hx+16},{hy-5} '
             f'L {hx+7},{hy+11} C {hx+4},{hy+15} {hx},{hy+12} {hx+2},{hy+7} '
             f'L {hx+6},{hy-2} L {hx-1},{hy+1} L {hx-8},{hy+10} '
             f'C {hx-11},{hy+14} {hx-15},{hy+10} {hx-12},{hy+6} '
             f'L {hx-4},{hy-7} C {hx+1},{hy-14} {hx+10},{hy-18} {hx+17},{hy-20} Z',
             '#EAC0A6', '#A1735D', 1.3)
        path(f'M {hx+6},{hy-2} L {hx+12},{hy-10}', stroke='#BB8D75', sw=1.1)
    else:
        gx, gy = ((42, 26), (49, 26), (69, 37))[phase]
        path(f'M 92,7 L 75,19 L {gx+12},{gy-8}', stroke='#6B7D8F', sw=13)
        path(f'M 92,7 L 75,19 L {gx+12},{gy-8}', stroke='#D5DFE7', sw=8)
        rect(gx-12, gy-7, 25, 14, '#526478', '#293B50', 1, 2)
        spread = 17 if phase != 1 else 12
        path(f'M {gx-10},{gy+7} L {gx-spread},{gy+13} L {gx-spread},{gy+23}', stroke='#263C50', sw=4)
        path(f'M {gx+10},{gy+7} L {gx+spread},{gy+13} L {gx+spread},{gy+23}', stroke='#263C50', sw=4)
    end()
    rect(0, 0, 100, 88, 'none', '#BCC6CF', 1.3, 3)
    end()


def token_set(name, x, y, color):
    group(name)
    for frame, count in enumerate((4, 3)):
        for row in range(count):
            for col in range(3):
                rect(x+frame*47+col*9, y+row*17, 7.5, 14, color, 'none')
    text(x+36, y+41, '⋯', 21, SUB, anchor='middle')
    end()


def embedding(name, x, y, color):
    group(name)
    for i in range(3):
        rect(x+i*25, y, 17, 64, color, 'none')
    end()


def adapter(name, x, y, color):
    group(name)
    rect(x, y, 128, 76, '#F6F8FA', color, 2, 4)
    mathtext(x+64, y+46, [('f',1,0,True),('θ',.7,8,False)], 37, color, 'middle')
    end()


def box(name, x, y, w, h, rows, stroke=INK, fill='white', size=26):
    group(name)
    rect(x, y, w, h, fill, stroke, 1.7, 4)
    first = y+h/2-(len(rows)-1)*16+9
    for i, row in enumerate(rows):
        text(x+w/2, first+i*32, row, size, anchor='middle')
    end()


# Overview: sources are context; cached token sets are the package input.
group('overview')
text(28, 47, '(a) Shared interaction representations', 30, weight='bold')
text(794, 47, '(b) Monotonic matching', 30, weight='bold')
text(1223, 47, '(c) Filtered correspondences', 30, weight='bold')
line(765, 74, 765, 558, LIGHT, 1.4, '5 7')
line(1208, 74, 1208, 558, LIGHT, 1.4, '5 7')

for role, y, color, suffix in [('human', 185, HUMAN, 'H'), ('robot', 400, ROBOT, 'R')]:
    label = 'Human video' if role == 'human' else 'Robot video'
    text(30, y-27, label, 28, color, weight='bold')
    for i in range(3):
        scene(f'{role}-context-{i}', 30+76*i, y, 69, role, i)
    arrow([(31, y+83), (248, y+83)], color=SUB, sw=1.5)
    ticks = [45, 93, 210] if role == 'human' else [43, 158, 202]
    for tick in ticks:
        line(tick, y+78, tick, y+88, SUB, 1.4)
    mathtext(138,y+117,[('T',1,0,True),(suffix,.7,6,False),(' sampled frames',1,0,False)],24,SUB,'middle')
    token_set(f'{role}-cached-token-sets', 288, y+2, color)
    arrow([(255, y+37), (277, y+37)], color=GRAY, dashed=True, sw=1.7)
    text(325, y-29, '512-D tokens', 25, color, anchor='middle')
    text(325, y+99, 'Kₜ × 512', 26, color, anchor='middle')
    text(325, y+129, 'per frame', 23, SUB, anchor='middle')
    arrow([(365, y+37), (393, y+37)])
    adapter(f'{role}-shared-adapter', 401, y, color)
    arrow([(532, y+37), (573, y+37)])
    embedding(f'{role}-frame-embeddings', 584, y+4, color)
    text(620, y-29, '128-D', 26, color, anchor='middle')
    mathtext(620,y+99,[('Z',1,0,True),(suffix,.7,-8,False),(' : T',1,0,True),
                      (suffix,.7,6,False),(' × 128',1,0,False)],25,color,'middle')

line(465, 266, 465, 311, SUB, 1.6, '5 5')
line(465, 365, 465, 390, SUB, 1.6, '5 5')
text(465, 341, 'Shared θ', 25, SUB, anchor='middle')
path('M 680,222 L 736,222 L 736,301',stroke=INK,sw=2.3)
path('M 680,437 L 736,437 L 736,301', stroke=INK, sw=2.3)
arrow([(736,301),(752,301),(752,180),(915,180),(915,199)])
text(30, 553, 'Same task · independently sampled timelines', 25, SUB)

# S is cosine similarity because the adapter output is L2 normalized.
group('cosine-similarity-and-dtw')
text(807, 120, 'Cosine similarity', 27, weight='bold')
mathtext(909,155,[('S = Z',1,0,True),('R',.7,-9,False),(' (Z',1,0,True),
                 ('H',.7,-9,False),(')',1,0,False),('T',.7,-9,False)],27,anchor='middle')
mx, my, cell, nh, nr = 820, 206, 19, 10, 8
centers = [0, 1, 3, 4, 4, 6, 7, 9]
for i in range(nr):
    for j in range(nh):
        value = math.exp(-((j-centers[i])/1.8)**2)
        low, high = (241, 247, 250), (76, 126, 164)
        rgb = tuple(round(a+(b-a)*value) for a, b in zip(low, high))
        rect(mx+j*cell, my+(nr-1-i)*cell, cell, cell,
             '#%02X%02X%02X' % rgb, '#E0E8EE', 0.7)
path_indices = [(0,0),(1,1),(2,2),(2,3),(3,4),(4,4),(5,5),(5,6),(6,7),(6,8),(7,9)]
path_points = [(mx+(j+.5)*cell, my+(nr-i-.5)*cell) for i,j in path_indices]
path('M '+' L '.join(f'{x},{y}' for x,y in path_points), stroke=AMBER, sw=3.3)
for x,y in path_points:
    dot(x,y,3.3,AMBER)
rect(mx,my,nh*cell,nr*cell,'none','#A9BAC8',1.3)
arrow([(mx, my+nr*cell+10), (mx+nh*cell, my+nr*cell+10)], color=SUB, sw=1.3)
arrow([(mx-10,my+nr*cell), (mx-10,my)], color=SUB, sw=1.3)
text(mx+nh*cell/2, my+nr*cell+41, 'Human j', 24, SUB, anchor='middle')
text(mx-31,my+nr*cell/2,'Robot i',24,SUB,anchor='middle',rotate=-90)
line(831, 431, 865, 431, AMBER, 3)
text(877, 439, 'DTW path', 23, SUB)
text(909, 471, '(schematic)', 23, SUB, anchor='middle')
arrow([(1019, 281), (1040, 281)])
box('two-way-dtw', 1049, 214, 144, 151, ['Two-way','DTW','cost 1 − S'], size=25)
text(1121, 410, 'g: R → H', 25, anchor='middle')
text(1121, 443, 'h: H → R', 25, anchor='middle')
text(1000, 526, 'Endpoint-constrained', 24, SUB, anchor='middle')
text(1000, 556, 'Complete sequences (offline)', 24, SUB, anchor='middle')
end()

# Only filtered candidates reach the output: no matrix-to-output bypass.
arrow([(1195, 281), (1232, 281)])
group('correspondence-filters')
text(1363, 156, 'Accept if all pass', 27, weight='bold', anchor='middle')
rect(1243, 188, 241, 237, '#FFFFFF', INK, 1.7, 4)
text(1363, 233, 'Cosine ≥ 0.50', 26, anchor='middle')
line(1260, 255, 1467, 255, LIGHT, 1.2)
text(1363, 284, 'Cycle ≤ 2', 26, anchor='middle')
text(1363, 309, 'sampled robot frames', 23, SUB, anchor='middle')
line(1260, 315, 1467, 315, LIGHT, 1.2)
text(1363, 351, 'Margin ≥ 0.01', 26, anchor='middle')
text(1363, 399, 'AND', 24, SUB, weight='bold', anchor='middle')
text(1363, 476, 'Distant: |j − g(i)| > 3', 24, SUB, anchor='middle')
text(1363, 512, 'No competitor → reject', 23, SUB, anchor='middle')
end()

group('output-frame-pairs')
text(1560, 120, 'Human', 25, HUMAN, anchor='middle', weight='bold')
text(1701, 120, 'Robot', 25, ROBOT, anchor='middle', weight='bold')
for row, phase in enumerate((0,1,2)):
    sy = 177+130*row
    rejected = row==2
    scene(f'output-human-{row}',1517,sy,86,'human',phase,rejected)
    scene(f'output-robot-{row}',1658,sy,86,'robot',phase if not rejected else 0,rejected)
    line(1609,sy+36,1651,sy+36,GRAY if rejected else GREEN,2.5,'5 5' if rejected else None)
    dot(1609,sy+36,3,GRAY if rejected else GREEN)
    dot(1651,sy+36,3,GRAY if rejected else GREEN)
    if rejected:
        path(f'M 1758,{sy+29} L 1772,{sy+43} M 1772,{sy+29} L 1758,{sy+43}',stroke=GRAY,sw=2.6)
    else:
        path(f'M 1755,{sy+36} L 1762,{sy+43} L 1776,{sy+28}',stroke=GREEN,sw=2.7)
arrow([(1486,281),(1501,281),(1501,213),(1511,213)],color=GREEN,sw=1.8)
arrow([(1501,281),(1501,343),(1511,343)],color=GREEN,sw=1.8)
arrow([(1486,400),(1497,400),(1497,473),(1511,473)],color=GRAY,sw=1.6,dashed=True)
text(1630, 548, 'Retained / rejected candidates', 23, SUB, anchor='middle')
end()
end()

# Exact internal computation of f_theta, with no future frames in its context.
line(29, 594, 1771, 594, LIGHT, 1.4)
group('shared-adapter-detail')
text(29, 644, '(d) Adapter detail: token pooling and causal temporal context', 30, weight='bold')
text(1767, 644, 'Frozen at inference', 25, SUB, anchor='end')
group('detail-input-set')
for row in range(5):
    for col in range(6):
        rect(45+col*10,731+row*15,8,12,HUMAN,'none')
end()
text(77, 851, 'Xₜ', 29, HUMAN, anchor='middle', italic=True)
text(77, 885, 'Kₜ × 512', 24, SUB, anchor='middle')
arrow([(125,770),(163,770)])
box('token-projection',176,725,178,92,['LN + linear','GELU'],size=25)
text(265, 851, '512 → 128', 25, SUB, anchor='middle')
arrow([(356,770),(396,770)])
box('learned-attention-pooling',409,725,199,92,['Score → softmax','Weighted sum'],size=24)
text(508, 852, 'Valid pairs only', 24, SUB, anchor='middle')
text(640, 748, 'pₜ', 27, anchor='middle', italic=True)
arrow([(610,770),(657,770)])
dot(657,770,3.5)
arrow([(657,770),(683,770),(683,702),(711,702)])
arrow([(657,770),(711,770)])
arrow([(657,770),(683,770),(683,841),(711,841)])
box('current-pooled-token',725,679,202,45,['pₜ'],size=27)
box('temporal-difference',725,748,202,45,['pₜ − pₜ₋₁'],size=27)
box('causal-convolution',725,818,202,45,['Causal Conv1D'],size=25)
text(826, 895, 'k = 3 · t−2, t−1, t', 24, SUB, anchor='middle')
for yy in (702,770,841):
    path(f'M 928,{yy} L 966,{yy} L 966,770',stroke=INK,sw=2.3)
arrow([(966,770),(1002,770)])
box('concatenation',1016,725,174,92,['Concatenate','384-D'],size=26)
arrow([(1192,770),(1230,770)])
box('readout-and-normalization',1244,725,192,92,['LN + MLP','L2 normalize'],size=26)
text(1340, 852, '384 → 128', 25, SUB, anchor='middle')
arrow([(1438,770),(1484,770)])
embedding('detail-normalized-output',1500,737,ROBOT)
mathtext(1639,772,[('z',1,0,True),('t',.7,6,False),(' ∈ R',1,0,True),
                  ('128',.7,-10,False)],29,anchor='middle')
text(1637, 812, 'unit norm', 24, SUB, anchor='middle')
end()

description = ('Offline human-robot progress alignment. The package reads cached per-frame sets '
               'of 512-D interaction tokens, not raw pixels. Shared f_theta produces unit-norm '
               '128-D embeddings. Endpoint-constrained DTW runs in both directions, followed '
               'by cosine, cycle-return, and distant-margin filtering. Output pairs are heuristic '
               'pseudo-correspondences, not verified semantic labels. Scenes and matrix are schematic.')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="190mm" height="{190*H/W:.6f}mm" '
       f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">\n'
       f'<title id="title">Offline progress alignment: shared temporal tokens, DTW and correspondence filtering</title>\n'
       f'<desc id="description">{escape(description)}</desc>\n'
       '<defs>'+''.join(definitions)+'</defs>\n'
       f'<rect width="{W}" height="{H}" fill="white"/>\n'+ '\n'.join(parts)+'</svg>\n')
svg_path = ROOT/'overview.svg'
svg_path.write_text(svg, encoding='utf-8')
tree = ET.fromstring(svg)
assert not tree.findall('.//{http://www.w3.org/2000/svg}image'), 'Raster embedding is prohibited'
ids = [e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
assert len(ids) == len(set(ids)), 'Duplicate semantic group IDs'

source = fitz.open(svg_path)
pdf = fitz.open('pdf', source.convert_to_pdf())
pdf.set_metadata({'title':'Offline Progress Aligner - method overview',
                  'subject':description,'creator':'SVG vector source'})
pdf_path = ROOT/'overview.pdf'
pdf.save(pdf_path,garbage=4,deflate=True)
pdf.close()
with fitz.open(pdf_path) as check:
    assert len(check)==1
    assert len(check[0].get_images(full=True))==0, 'PDF contains raster objects'
    assert len(check[0].get_drawings())>200, 'Missing vector geometry'
    pix=check[0].get_pixmap(matrix=fitz.Matrix(W/check[0].rect.width,W/check[0].rect.width),alpha=False)
    pix.save(ROOT/'overview.png')
    print({'svg':str(svg_path),'pdf':str(pdf_path),'raster_images':0,
           'vector_paths':len(check[0].get_drawings()),'svg_text_elements':len(tree.findall('.//{http://www.w3.org/2000/svg}text')),
           'pdf_fonts':len(check[0].get_fonts()),'pdf_text_chars':len(check[0].get_text())})
