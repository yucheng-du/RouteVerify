#!/usr/bin/env python3
"""Generate the RouteVerify overview schematic as editable SVG and PDF.

ReportLab measures Arial text; Inkscape exports the embedded-font PDF.
SVG objects retain named text and shape elements for later editing.
"""
from pathlib import Path
import os,shutil
from html import escape
import json, math, subprocess
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE=Path(os.environ.get("ROUTEVERIFY_FIGURE_OUTPUT", "build/figures")).resolve()
QA=HERE.parent / "qa"
HERE.mkdir(parents=True,exist_ok=True)
QA.mkdir(exist_ok=True)
for name,file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path(os.environ.get('ROUTEVERIFY_FONT_DIR','/System/Library/Fonts/Supplemental'))/file)))
INK='#183653'; LINE='#24537C'; MUTED='#48637A'; WHITE='#FFFFFF'
CLEAN='#49BEB0'; ERROR='#F76B63'; BLUE_EDGE='#C4DEED'
ROUTER='#2B9CB2'; OUTPUT='#CB941E'; RESIDUAL='#D85E7E'

class Drawing:
    def __init__(self,stem,w,h):
        self.stem=stem;self.w=w;self.h=h;self.parts=[];self.texts=[];self.boxes={};self.edges=[];self.window_widths=[]
        self.parts.append('''<defs>
          <linearGradient id="verifier" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#FA736A"/><stop offset=".48" stop-color="#AF577D"/><stop offset="1" stop-color="#143F7E"/></linearGradient>
          <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FCFEFF"/><stop offset="1" stop-color="#F1F9FD"/></linearGradient>
          <linearGradient id="clean" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#53C9BC"/><stop offset="1" stop-color="#3AB6AC"/></linearGradient>
          <linearGradient id="error" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FF776F"/><stop offset="1" stop-color="#F15A58"/></linearGradient>
          <linearGradient id="footer" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#EDF2F7"/><stop offset="1" stop-color="#E6EEF5"/></linearGradient>
        </defs>''')
    def rect(self,id,x,y,w,h,fill=WHITE,stroke=BLUE_EDGE,lw=.55,r=2,shadow=False,dash=None):
        if shadow:
            # Small low-opacity vector silhouettes emulate the reference's soft depth.
            for dx,dy,spread,opacity in [(0,.6,.65,.035),(0,.35,.3,.045)]:
                self.parts.append(f'<rect x="{x-spread+dx}" y="{y-spread+dy}" width="{w+2*spread}" height="{h+2*spread}" rx="{r+spread}" fill="#34546C" opacity="{opacity}"/>')
        attr=f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{attr}/>')
        self.boxes[id]=(x,y,x+w,y+h)
    def text(self,id,x,y,value,size=7,color=INK,bold=False,anchor='middle',owner=None,leading=None):
        lines=value.split('\n');lead=leading or size*1.18
        font='Arial-Bold' if bold else 'Arial'
        widths=[pdfmetrics.stringWidth(t,font,size) for t in lines]
        top=y-(len(lines)-1)*lead/2
        self.parts.append(f'<text id="{id}" font-family="Arial" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">')
        for i,line in enumerate(lines):
            self.parts.append(f'<tspan x="{x}" y="{top+i*lead+size*.34}">{escape(line)}</tspan>')
        self.parts.append('</text>')
        x0=x-(max(widths) if anchor=='end' else max(widths)/2 if anchor=='middle' else 0)
        self.texts.append({'id':id,'text':value,'font_pt':size,'owner':owner,'bbox':[x0,top-size*.52,x0+max(widths),top+(len(lines)-1)*lead+size*.52]})
    def line(self,id,pts,color=LINE,lw=.7,dash=None):
        data='M '+' L '.join(f'{x},{y}' for x,y in pts)
        attr=f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<path id="{id}" d="{data}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round"{attr}/>')
    def arrow(self,id,pts,color=LINE,lw=.8,head=2.3,dash=None):
        self.line(id,pts,color,lw,dash)
        x,y=pts[-1];a,b=pts[-2];t=math.atan2(y-b,x-a);dx=math.cos(t);dy=math.sin(t);wid=head*.45
        p=[(x,y),(x-head*dx+wid*dy,y-head*dy-wid*dx),(x-head*dx-wid*dy,y-head*dy+wid*dx)]
        self.parts.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in p)+f'" fill="{color}"/>')
        self.edges.append({'id':id,'points':pts,'style':'dashed' if dash else 'solid'})
    def circle(self,id,x,y,r,fill=WHITE,stroke=LINE,lw=.8):
        self.parts.append(f'<circle id="{id}" cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
    def trim_top(self,amount):
        """Remove unused top canvas without scaling any surviving element."""
        self.parts=[f'<g transform="translate(0,{-amount})">',*self.parts,'</g>']
        self.h-=amount
        for t in self.texts:
            t['bbox'][1]-=amount;t['bbox'][3]-=amount
        self.boxes={key:(x,y-amount,x1,y1-amount) for key,(x,y,x1,y1) in self.boxes.items()}
        for edge in self.edges:
            edge['points']=[(x,y-amount) for x,y in edge['points']]
    def save(self):
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" viewBox="0 0 {self.w} {self.h}">\n'+ '\n'.join(self.parts)+'</svg>'
        (HERE/(self.stem+'.svg')).write_text(svg)
        subprocess.run([os.environ.get('INKSCAPE',shutil.which('inkscape') or '/Applications/Inkscape.app/Contents/MacOS/inkscape'),str(HERE/(self.stem+'.svg')),'--export-type=pdf','--export-filename='+str(HERE/(self.stem+'.pdf'))],check=True,capture_output=True)
        subprocess.run(['pdftoppm','-r','240','-png','-singlefile',str(HERE/(self.stem+'.pdf')),str(HERE/self.stem)],check=True,capture_output=True)
        issues=[]
        for t in self.texts:
            a,b,c,e=t['bbox']
            if a<0 or b<0 or c>self.w or e>self.h:issues.append(['canvas',t['id']])
            if t['owner']:
                x,y,x1,y1=self.boxes[t['owner']]
                if a<x+.4 or b<y+.4 or c>x1-.4 or e>y1-.4:issues.append(['owner',t['id'],t['owner']])
        for i,a in enumerate(self.texts):
            for b in self.texts[i+1:]:
                ax,ay,ax1,ay1=a['bbox'];bx,by,bx1,by1=b['bbox']
                if min(ax1,bx1)-max(ax,bx)>.3 and min(ay1,by1)-max(ay,by)>.3:issues.append(['text_overlap',a['id'],b['id']])
        (QA/('figure1_geometry.json')).write_text(json.dumps({'canvas_pt':[self.w,self.h],'texts':self.texts,'edges':self.edges,'issues':issues,'window_widths_pt':self.window_widths,'fonts_pt':sorted(set(t['font_pt'] for t in self.texts))},indent=2)+'\n')
        print(self.stem,'geometry',issues)
        if issues: raise AssertionError(issues)

def badge(d,id,x,y,w,h,label,role,size):
    fill,stroke,color={'clean':('url(#clean)','none',WHITE),'error':('url(#error)','none',WHITE),'clean_receiver':('#E8F6F5','#6CC7C0',INK),'error_receiver':('#FFF1EE','#F88B89',INK),'same':('#EAF1F6','#CCDDE8',INK)}[role]
    d.rect(id,x,y,w,h,fill,stroke,r=1.9)
    d.text(id+'_text',x+w/2,y+h/2,label,size,color,owner=id)


def traces(d):
    x,y,w,h=5.5,34.5,76,127.5
    d.rect('traces',x,y,w,h,'url(#panel)',r=3,shadow=True)
    d.text('traces_title',x+w/2,y+13,'Paired reasoning\ntraces',7.2,bold=True,owner='traces',leading=8.1)
    xx=[x+10.5,x+35,x+59]
    yc,ye=83.5,121
    boundary_end=133
    d.line('first_error_boundary',[(xx[-1],yc+5.3),(xx[-1],boundary_end)],'#A6BDCC',.55,'2.2 2.2')
    for role,yy,title in [('clean',yc,'Correct trace'),('error',ye,'First-error trace')]:
        d.text(role+'_trace_title',x+6.2,yy-12,title,6.2,bold=True,anchor='start',owner='traces')
        for j,xxx in enumerate(xx):
            if j<2:d.arrow(role+'_step_edge_'+str(j),[(xxx+5.9,yy),(xx[j+1]-6.0,yy)],head=2.2)
            d.circle(role+'_step_'+str(j),xxx,yy,5.2,'url(#'+role+')' if j==2 else WHITE,lw=.9)
    d.text('first_error_label',xx[-1],139,'First error',5.7,owner='traces')
    d.text('future_hidden',x+w/2,152,'Later steps\nhidden',5.7,owner='traces',leading=6.3)


def sites(d):
    x,y,w,h=89,66,142.5,59
    d.rect('sites',x,y,w,h,'none','#C2DCEA',r=2,dash='2.3 1.7')
    d.text('sites_title',x+w/2,y+5,'Intervention sites',6.6,owner='sites')
    cw=w/3
    for i,(title,col,fill) in enumerate([('Router logits',ROUTER,'#E6F5F7'),('MoE output',OUTPUT,'#FFF0E1'),('Residual state',RESIDUAL,'#FAE7EE')]):
        cx=x+(i+.5)*cw
        if i:d.line('site_separator_'+str(i),[(x+i*cw,y+10),(x+i*cw,y+h-3)],'#D4E3EB',.45)
        d.circle('site_dot_'+str(i),cx,y+10.5,1.4,col,'none')
        d.line('site_stem_'+str(i),[(cx,y+10.5),(cx,y+15)],col,.65)
        d.rect('site_label_'+str(i),x+i*cw+1.5,y+14,cw-3,8,fill,'none',r=1.4)
        d.text('site_text_'+str(i),cx,y+18,title,6.0,bold=True,owner='site_label_'+str(i))
    cy=102; rx=x+3; ex=x+cw-13
    d.rect('router_icon',rx,cy-4.2,20,8.4,WHITE,'#6E90B7',r=1.8)
    d.text('router_text',rx+10,cy,'Router',5.3,owner='router_icon')
    for j in range(3):
        yy=cy-8.8+j*8.8
        d.arrow('selection_'+str(j),[(rx+20.55,cy),(ex-.55,yy)],lw=.5,head=1.25,dash='1.2 1')
        d.rect('routed_expert_'+str(j),ex,yy-3.65,11,7.3,WHITE,'#A3BFDB',lw=.45,r=1)
        d.text('routed_E_'+str(j),ex+5.5,yy,'E'+str(j+1),5.2,owner='routed_expert_'+str(j))
    outx=x+2*cw-8
    for j in range(3):
        yy=cy-8.8+j*8.8; bx=x+cw+3.5
        d.rect('output_expert_'+str(j),bx,yy-3.65,11,7.3,WHITE,'#A3BFDB',lw=.45,r=1)
        d.text('output_E_'+str(j),bx+5.5,yy,'E'+str(j+1),5.2,owner='output_expert_'+str(j))
        d.arrow('routed_state_'+str(j),[(bx+11.55,yy),(outx-4.5,cy)],lw=.6,head=1.6)
    d.circle('output_patch',outx,cy,4.0,OUTPUT,'none')
    plus=x+2*cw+cw*.4; res=x+3*cw-7
    for j,dy in enumerate([-8,8]):d.arrow('addition_input_'+str(j),[(x+2*cw+4,cy+dy),(plus-4.4,cy+dy*.25)],lw=.6,head=1.6)
    d.circle('addition',plus,cy,3.8,WHITE,lw=.65)
    d.boxes['addition_region']=(plus-3.8,cy-3.8,plus+3.8,cy+3.8)
    d.text('addition_glyph',plus,cy,'+',6.2,owner='addition_region')
    d.arrow('post_addition',[(plus+4.4,cy),(res-4.2,cy)],lw=.65,head=1.6)
    d.circle('residual_patch',res,cy,3.6,RESIDUAL,LINE,.6)
    for i,txt in enumerate(['Select + weight','Routed update','After addition']):
        d.text('site_definition_'+str(i),x+(i+.5)*cw,120.5,txt,5.6,owner='sites')


def support(d):
    x,y,w,h=86.5,128,148,34
    d.rect('support',x,y,w,h,'url(#panel)',r=2.8,shadow=True)
    d.text('support_title',x+4,y+4.5,'Layer support · T window examples',6.4,bold=True,anchor='start',owner='support')
    layerw=40
    for i,txt in enumerate(['One anchor','All-router']):
        d.rect('layer_'+str(i),x+3.5,y+10.6+i*9.6,layerw-5,7.8,'#E9F2F8',r=1.7)
        d.text('layer_text_'+str(i),x+3.5+(layerw-5)/2,y+14.5+i*9.6,txt,5.3,owner='layer_'+str(i))
    d.text('layer_note',x+3,y+h-3,'All routed layers',5.0,anchor='start',owner='support')
    winw=(w-layerw-8)/2
    for i,(label,color,fill) in enumerate([('Divergence-8','#CC4C5F','#FAC7C4'),('Suffix-8','#188C86','#ADE6DD')]):
        xx=x+layerw+5+i*winw
        if i:d.line('window_separator',[(xx-1,y+12),(xx-1,y+h-3)],'#D1E2EA',.45)
        d.text('window_label_'+str(i),xx+winw/2,y+11.8,label,5.8,bold=True,color=color,owner='support')
        d.text('window_tokens_'+str(i),xx+winw/2,y+18,'8 tokens',5.0,color,owner='support')
        ty=y+24.2; trackw=winw-5; selected=trackw*.55; start=xx+trackw*(.22 if i==0 else .45)
        d.rect('track_'+str(i),xx,ty-1,trackw,3.4,'#E4EDF3','none',r=1.7)
        d.rect('selected_window_'+str(i),start,ty-1,selected,3.4,fill,'none',r=1.4)
        d.line('window_brace_'+str(i),[(start,ty-2),(start,ty-3),(start+selected,ty-3),(start+selected,ty-2)],color,.45)
        d.window_widths.append(selected)
        if i==0:
            d.line('divergence_token',[(start+selected*.44,ty-.4),(start+selected*.44,ty+1.8)],'#708596',1)
            d.text('first_divergence_text',xx+trackw/2,y+h-3,'First divergence',5.0,owner='support')


def protocols(d):
    x,w=241.5,148.5
    fs=6.2
    y,th=34.5,57
    d.rect('T_card',x,y,w,th,'url(#panel)',r=3,shadow=True)
    d.text('T_heading',x+6,y+8,'T  Paired response',8.2,bold=True,anchor='start',owner='T_card')
    d.text('T_protocol',x+w/2,y+18.5,'Same pair · independent runs',fs,owner='T_card')
    left,unit,gap=x+7.5,54.75,24
    for j,(donor,receiver) in enumerate([('Clean','Error'),('Error','Clean')]):
        yy=58+j*14
        badge(d,'T_donor_'+str(j),left,yy,unit,10,donor+' donor',donor.lower(),fs)
        badge(d,'T_receiver_'+str(j),left+unit+gap,yy,unit,10,receiver+' receiver',receiver.lower()+'_receiver',fs)
        d.arrow('T_transfer_'+str(j),[(left+unit+1.4,yy+5),(left+unit+gap-1.4,yy+5)],head=2.6,lw=.75,dash='1.8 1.4')
    d.text('T_readout',x+w/2,87.1,'Signed donor-aligned margin shifts',fs,owner='T_card')
    yy,dh=95,67
    d.rect('D_card',x,yy,w,dh,'url(#panel)',r=3,shadow=True)
    d.text('D_heading',x+6,yy+8,'D  Donor-role contrast',8.2,bold=True,anchor='start',owner='D_card')
    d.text('D_protocol',x+w/2,yy+18.5,'External matched pair · separate runs',fs,owner='D_card')
    unit,gap=62,9.5
    for j,role in enumerate(['clean','error']):
        xx=x+7.5+j*(unit+gap)
        badge(d,'D_donor_'+role,xx,119,unit,10,'Matched '+role+' donor',role,fs)
        badge(d,'D_receiver_'+role,xx,136,unit,10,'Same receiver','same',fs)
        d.arrow('D_transfer_'+role,[(xx+unit/2,130.1),(xx+unit/2,134.9)],head=1.7,lw=.7)
    d.text('D_comparison',x+w/2,151.2,'Compare error-donor − clean-donor margins',fs,owner='D_card')
    d.text('D_repeat',x+w/2,158.2,'Repeat for both receiver roles',fs,owner='D_card')


def verify_semantics(d):
    """Finite layout/semantic checks, independent of scientific experiment code."""
    byid={t['id']:t for t in d.texts}
    assert not any(k in d.boxes for k in ['T_margin', 'D_margin', 'shared_margin']), 'No verdict ruler is permitted.'
    assert d.w==396 and abs(d.h-157.16)<1e-8
    assert 'title' not in d.boxes and not any(k in byid for k in ['project_title','question'])
    assert len({d.boxes[k][1] for k in ['traces','verifier','T_card']})==1
    assert len({d.boxes[k][3] for k in ['traces','support','D_card']})==1
    assert len(d.window_widths)==2 and d.window_widths[0]==d.window_widths[1]
    assert all(byid[k]['text']=='8 tokens' for k in ['window_tokens_0','window_tokens_1'])
    assert all(byid[k]['font_pt']>=6.2 for k in byid if k.startswith(('T_', 'D_')))
    assert all(t['owner'] for t in d.texts), 'Visible labels must have a layout owner.'
    expected={
        'T_donor_0_text':'Clean donor','T_receiver_0_text':'Error receiver',
        'T_donor_1_text':'Error donor','T_receiver_1_text':'Clean receiver',
        'D_donor_clean_text':'Matched clean donor','D_donor_error_text':'Matched error donor',
        'D_receiver_clean_text':'Same receiver','D_receiver_error_text':'Same receiver',
        'D_repeat':'Repeat for both receiver roles',
        'D_comparison':'Compare error-donor − clean-donor margins',
        'site_text_0':'Router logits','site_text_1':'MoE output','site_text_2':'Residual state',
        'site_definition_1':'Routed update','site_definition_2':'After addition',
        'margin_sign':'M(x) < 0 favors CORRECT; M(x) > 0 favors INCORRECT',
    }
    assert all(byid[k]['text']==v for k,v in expected.items())
    for edge in d.edges:
        points=edge['points']
        length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        assert length>2.2, (edge['id'],length)
    # Finite segment-vs-text checks catch connectors routed through labels.
    def segment_hits(a,b,box):
        x0,y0,x1,y1=box
        dx,dy=b[0]-a[0],b[1]-a[1]
        lo,hi=0.,1.
        for p,q in [(-dx,a[0]-x0),(dx,x1-a[0]),(-dy,a[1]-y0),(dy,y1-a[1])]:
            if abs(p)<1e-9:
                if q<0:return False
            elif p<0:lo=max(lo,q/p)
            else:hi=min(hi,q/p)
            if lo>hi:return False
        return True
    arrow_text_hits=[]
    for edge in d.edges:
        for t in d.texts:
            a,b,c,e=t['bbox']; box=(a-.15,b-.15,c+.15,e+.15)
            if any(segment_hits(p,q,box) for p,q in zip(edge['points'],edge['points'][1:])):
                arrow_text_hits.append([edge['id'],t['id']])
    assert not arrow_text_hits, arrow_text_hits
    # Inset contracts for the transfers and verifier destinations.
    edges={e['id']:e for e in d.edges}
    for i in range(2):
        donor=d.boxes['T_donor_'+str(i)];receiver=d.boxes['T_receiver_'+str(i)]
        a,b=edges['T_transfer_'+str(i)]['points']
        assert a[0]-donor[2]>=1.3 and receiver[0]-b[0]>=1.3
        assert donor[1]<a[1]<donor[3] and receiver[1]<b[1]<receiver[3]
    for role in ['clean','error']:
        donor=d.boxes['D_donor_'+role];receiver=d.boxes['D_receiver_'+role]
        a,b=edges['D_transfer_'+role]['points']
        assert a[1]-donor[3]>=1.0 and receiver[1]-b[1]>=1.0
        assert donor[0]<a[0]<donor[2] and receiver[0]<b[0]<receiver[2]
    for which in ['T','D']:
        tip=edges['verifier_'+which]['points'][-1];box=d.boxes[which+'_card']
        assert box[0]-tip[0]>=1.4 and box[1]+4<tip[1]<box[3]-4
    assert edges['verifier_D']['points'][1][0]-d.boxes['sites'][2]>=4
    (QA/'figure1_semantic_checks.json').write_text(json.dumps({
        'status':'PASS','source':'figure_vector_refine_20260914/build_vector.py:faithful',
        'canvas_pt':[d.w,d.h], 'rulers':0, 'window_widths_pt':d.window_widths,
        'title_card_removed':True, 'top_canvas_trimmed_pt':32, 'surviving_content_scaled':False,
        'column_top_alignment_pt':d.boxes['traces'][1],
        'column_bottom_alignment_pt':d.boxes['traces'][3],
        'right_card_font_pt':[6.2,8.2], 'retained_labels':expected,
        'arrow_text_intersections':arrow_text_hits, 'transfer_endpoint_clearance':'PASS',
        'verifier_D_interface_corridor_clearance_pt':4.5,
        'no_text_below_6_2_claim':'The T/D cards retain 6.2 pt. Central interface names are 6.0 pt, definitions 5.6 pt, and microcircuit/window/layer details 5.0–5.8 pt.',
        'central_microcircuit_font_pt':5.2,
        'minimum_font_pt':min(t['font_pt'] for t in d.texts)
    },indent=2)+'\n')


def draw_overview():
    d=Drawing('abc_overview_20260908',396,189.16)
    traces(d)
    d.rect('verifier',99,34.5,129,27.5,'url(#verifier)',WHITE,lw=.7,r=3,shadow=True)
    d.text('verifier_title',163.5,48.25,'MoE verifier',10.5,bold=True,color=WHITE,owner='verifier')
    d.arrow('verifier_input',[(83,83.5),(86,83.5),(86,48.25),(97.5,48.25)],head=2.7)
    d.arrow('verifier_T',[(229.5,48.25),(234,48.25),(240,47)],head=2.4)
    # Corridor remains entirely to the right of the interface box (right = 231.5).
    d.arrow('verifier_D',[(229.5,58),(236,58),(236,103),(240,103)],head=1.8)
    sites(d)
    support(d)
    protocols(d)
    d.boxes['margin_sign_region']=(5.5,164,390.5,171.8)
    d.text('margin_sign',198,168,'M(x) < 0 favors CORRECT; M(x) > 0 favors INCORRECT',6.2,owner='margin_sign_region')
    d.rect('conclusion',5.5,174,385,12.5,'url(#footer)','#90A8BC',r=3)
    d.text('conclusion_text',198,180.25,'Intervention response and donor-role directionality require separate evidence',7,bold=True,owner='conclusion')
    d.trim_top(32)
    verify_semantics(d)
    d.save()

if __name__=='__main__':
    draw_overview()
