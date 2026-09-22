#!/usr/bin/env python3
"""Render the four numerical paper figures from archived statistics on CPU."""
from pathlib import Path
from collections import Counter
import argparse,csv,importlib.util,json,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import main_style as style
import appendix_style as appendix

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/figures'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    p=json.loads((HERE/'inputs/palette.json').read_text());checks=[]
    import craft
    measurements={'contrast':{f:{b:craft.contrast_ratio(p[f],p[b]) for b in ('paper','band','window_fill')} for f in ('ink','muted')}}
    def save(fig,stem,extra=None):
        fig.savefig(out/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'RouteVerify figure reproduction'})
        fig.savefig(out/(stem+'.png'),dpi=200)
        with matplotlib.rc_context({'svg.fonttype':'none'}):fig.savefig(out/(stem+'.svg'),metadata={'Date':None})
        reference=HERE.parent/'figures'/(stem+'.pdf')
        text=lambda q:subprocess.check_output(['pdftotext','-layout',str(q),'-'],text=True)
        assert Counter(text(reference).split())==Counter(text(out/(stem+'.pdf')).split()),stem
        from pypdf import PdfReader
        a=PdfReader(reference);b=PdfReader(out/(stem+'.pdf'))
        assert list(a.pages[0].mediabox)==list(b.pages[0].mediabox),stem
        checks.append({'figure':stem,'text_tokens_equal_to_reviewed_asset':True,'page_geometry_equal':True,**(extra or {})});plt.close(fig)
    for n,stem in [(2,'abc_window_support_20260908'),(3,'wide_d_roles_20260908')]:
        with plt.rc_context():
            m=load('figure'+str(n));style.apply_palette(m,n,p,measurements)
            if n==2:
                m.configure(17.0);m.frozen_structure();drawn=m.build_figure();fig=drawn
            else:
                rows,secondary=m.load_rows();m.assert_frozen_structure(rows)
                drawn=m.build_figure(rows,m.load_contrasts(secondary));fig=drawn['fig']
            style.apply_surfaces(fig,m,n,p,drawn);style.apply_final_layout(fig,m,n)
            save(fig,stem)
    common={'#000000':p['ink'],'#202b36':p['ink'],'#54616b':p['muted'],'#d9dfe4':p['grid'],'#808080':p['rule'],'#4c4c4c':p['ink'],'#ffffff':p['paper']}
    with plt.rc_context():
        plt.rcdefaults();m=load('window_profiles');m.OUT=out;m.QA=out
        fig=appendix.capture(m.window_profiles)
        mapping={**common,'#596572':p['single_router_edge'],'#b37d19':p['output'],'#087f8c':p['router'],'#aa466b':p['residual']}
        appendix.recolour(fig,mapping,p,single_router_old='#596572');appendix.typography(fig,'abc_window_profiles_20260908')
        assert len(m.INVENTORY)==144
        save(fig,'abc_window_profiles_20260908',{'intervals':144,'layout':appendix.layout_census(fig)})
    with plt.rc_context():
        plt.rcdefaults();m=load('processbench_figure')
        folder=HERE/'inputs/processbench';pairs=list(csv.DictReader((folder/'pair_effects.csv').open()));strata=list(csv.DictReader((folder/'receiver_state_effects.csv').open()))
        for r in pairs:r['delta']=float(r['delta']);r['receiver_judgment_correct']=r['receiver_judgment_correct']=='True'
        for r in strata:
            r['receiver_judgment_correct']=r['receiver_judgment_correct']=='True';r['n_pairs']=int(r['n_pairs'])
            for k in ('estimate','ci_low','ci_high'):r[k]=float(r[k])
        fig=appendix.capture(lambda:m.figures(out,pairs,strata))
        appendix.recolour(fig,{**common,'#196c9f':p['pb_detected'],'#c77728':p['pb_missed']},p)
        appendix.receiver_status_markers(fig,p);appendix.typography(fig,'processbench_receiver_state')
        save(fig,'processbench_receiver_state',{'layout':appendix.layout_census(fig)})
    (out/'reproduction_check.json').write_text(json.dumps({'passed':True,'model_forwards':False,'resampling':False,'figures':checks},indent=2)+'\n')
    print(json.dumps({'passed':True,'figures':len(checks)}))
if __name__=='__main__':main()
