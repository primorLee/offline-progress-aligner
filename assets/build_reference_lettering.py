"""Recover reference A's individual label contours, not a whole-image trace.

Each named label remains an independent vector group at its original location.
This solves the generated handwriting's mismatch with installed Comic Sans MS.
The alternative live-text figure is kept by build_overview.py.
"""
from pathlib import Path
import json, sys, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'build/figure/deps'))
import cv2, numpy as np, vtracer
from PIL import Image

# (replacement label, source bounds, ink family, additional labels replaced)
REGIONS=[
 ('001',(29,31,452,77),'black',[]),
 ('002',(30,97,213,134),'navy',[]),
 ('003',(216,103,418,133),'navy',[]),
 ('004',(79,289,376,319),'navy',['005','006']),
 ('007',(429,110,665,136),'black',[]),
 ('008',(476,141,613,161),'navy',[]),
 ('009',(428,163,664,184),'navy',[]),
 ('010',(618,216,653,257),'qblue',['011']),
 ('012',(728,127,789,151),'navy',[]),
 ('013',(688,154,844,180),'navy',[]),
 ('014',(30,368,194,405),'navy',[]),
 ('015',(198,374,410,402),'navy',[]),
 ('016',(79,568,380,598),'navy',['017','018']),
 ('019',(427,399,645,424),'black',[]),
 ('020',(472,426,608,447),'navy',[]),
 ('021',(424,447,645,470),'navy',[]),
 ('022',(618,500,653,540),'qblue',['023']),
 ('024',(734,410,794,434),'navy',[]),
 ('025',(688,439,844,464),'navy',[]),
 ('026',(563,328,714,360),'purple',[]),
 ('027',(906,32,1300,77),'black',[]),
 ('028',(915,104,1295,135),'black',[]),
 ('029',(928,206,955,341),'black',[]),
 ('030',(1044,405,1179,431),'black',[]),
 ('031',(1033,439,1235,466),'black',[]),
 ('032',(1020,495,1202,528),'black',[]),
 ('033',(937,564,1081,590),'black',[]),
 ('034',(967,591,1051,613),'black',[]),
 ('035',(1138,564,1277,590),'black',[]),
 ('036',(1170,591,1254,613),'black',[]),
 ('037',(1357,33,1752,72),'black',[]),
 ('038',(1351,134,1493,164),'black',[]),
 ('039',(1373,295,1470,324),'black',[]),
 ('040',(1354,351,1483,382),'black',[]),
 ('041',(1359,406,1483,434),'black',[]),
 ('042',(1380,505,1463,537),'black',[]),
 ('043',(1523,106,1746,138),'black',[]),
 ('044',(1528,153,1604,181),'cyan',[]),
 ('045',(1658,153,1719,181),'cyan',[]),
 ('046',(1513,440,1758,468),'navy',['047']),
 ('048',(33,656,568,694),'black',[]),
 ('049',(282,717,340,742),'black',[]),
 ('050',(265,741,366,770),'black',[]),
 ('051',(639,714,719,740),'black',[]),
 ('052',(601,739,764,770),'black',[]),
 ('053',(877,671,1010,699),'black',[]),
 ('054',(875,722,1015,751),'black',[]),
 ('055',(876,774,1011,802),'black',[]),
 ('056',(1134,730,1209,758),'black',[]),
 ('057',(1329,729,1494,758),'black',[]),
 ('058',(47,784,195,810),'navy',[]),
 ('059',(68,813,172,842),'navy',['060','061']),
 ('062',(256,784,366,810),'navy',['063']),
 ('064',(604,784,752,816),'navy',[]),
 ('065',(858,813,1029,843),'navy',[]),
 ('066',(1141,784,1207,810),'navy',[]),
 ('067',(1353,784,1469,810),'navy',['068']),
 ('069',(1617,779,1679,805),'navy',[]),
 ('070',(1595,805,1709,835),'navy',[]),
]

A=np.array(Image.open(ROOT/'overview-web-comic.png').convert('RGB'))
work=ROOT.parent/'build/figure/native/text-matching'
work.mkdir(parents=True,exist_ok=True)
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
out=ET.Element('{'+NS+'}svg',{'viewBox':'0 0 1774 887','width':'1774','height':'887'})
data=[]
for ident,box,family,also in REGIONS:
 x0,y0,x1,y1=box
 crop=A[y0:y1,x0:x1].astype(float)
 r,g,b=[crop[:,:,i] for i in range(3)]
 if family=='black':
  valid=(b-r<65)&(g-r<65)&(np.max(crop,axis=2)<175)
 elif family=='navy':
  valid=(r<135)&(g<140)&(b<180)
 elif family=='qblue':
  valid=(r<100)&(g<125)&(b<183)
 elif family=='purple':
  valid=(r<135)&(g<105)&(b<200)
 else:
  valid=(r<120)&(g<170)&(b<230)&(b>g)
 pixels=crop[valid]
 if len(pixels)<5: raise ValueError((ident,'empty label'))
 lum=pixels@np.array([.2126,.7152,.0722])
 core=pixels[lum<=np.quantile(lum,.28)]
 ink=np.median(core,axis=0)
 bg=np.percentile(crop.reshape(-1,3),85,axis=0)
 delta=bg-ink
 alpha=np.clip(np.sum((bg-crop)*delta,axis=2)/np.sum(delta*delta),0,1)
 alpha[~valid]=0
 # Upsample the opacity field, then fit smooth outlines at its half-opacity edge.
 up=cv2.resize(alpha,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
 mask=(up>=.49).astype(np.uint8)
 n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
 for i in range(1,n):
  if stats[i,cv2.CC_STAT_AREA]<7: mask[labels==i]=0
 raster=work/(ident+'.png'); vector=work/(ident+'.svg')
 Image.fromarray(255-mask*255).convert('RGB').save(raster)
 vtracer.convert_image_to_svg_py(str(raster),str(vector),colormode='binary',mode='spline',
  filter_speckle=3,corner_threshold=60,length_threshold=2,max_iterations=10,splice_threshold=45,path_precision=3)
 root=ET.parse(vector).getroot()
 fill='#%02x%02x%02x'%tuple(map(lambda v:round(float(v)),ink))
 group=ET.SubElement(out,'{'+NS+'}g',{'id':'label-'+ident,'transform':f'translate({x0} {y0}) scale(0.25)',
  'data-replaces':','.join('label-'+j for j in also),'data-source':'reference-A lettering'})
 for child in root:
  if child.tag.endswith('path'):
   child.set('fill',fill);group.append(child)
 data.append({'id':'label-'+ident,'source_box':box,'ink':fill,'replace_ids':['label-'+j for j in also]})
dest=ROOT/'lettering';dest.mkdir(exist_ok=True)
ET.ElementTree(out).write(dest/'reference-a.svg',encoding='utf-8',xml_declaration=True)
(work/'regions.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps({'labels':len(data),'vector_bytes':(dest/'reference-a.svg').stat().st_size}))
