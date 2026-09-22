"""Recompute 126 D and all-layer output headline panels from native pair records."""
import argparse,gzip,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/lib'))
from routeverify.analysis.donor_contrast import _source_bootstrap

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'build/headline')
    out=ap.parse_args().output;out.mkdir(parents=True,exist_ok=True)
    reference=json.loads((HERE/'headline_d_bundle_v1.json').read_text())
    panels=defaultdict(list)
    for row in reference:panels[row['model'],row['family']].append(row)
    results=[]; max_error=0.;checked_cells=0
    for (model,family),rows in panels.items():
        records=json.loads(gzip.decompress((HERE/'headline_inputs'/model/(family+'.json.gz')).read_bytes()))
        bysite=defaultdict(list)
        for r in records:bysite[r['site']].append(r)
        for site,pairs in bysite.items():
            pairs.sort(key=lambda x:x['pair_id'])
            assert len(pairs)==290 and len({p['source_component'] for p in pairs})==235
            arm=site.split('.',1)[1] if family=='D' else 'all_moe_output'
            matched=[r for r in rows if r['arm']==arm]
            assert matched
            columns=[]
            for ref in matched:
                condition=ref['condition']
                if family=='D':
                    col=[]
                    for pair in pairs:
                        cells=pair['four_combinations']
                        ds={r:cells[r]['corrupt']['mean']-cells[r]['clean']['mean'] for r in ['clean','corrupt']}
                        ds['common']=(ds['clean']+ds['corrupt'])/2
                        val=ds[condition];assert abs(val-pair['d_'+condition+'_mean'])<1e-12
                        col.append(val);checked_cells+=4
                else:
                    group='contrasts' if '_minus_' in condition else 'effects'
                    col=[p[group]['mean_logprob'][condition]['t'] for p in pairs]
                columns.append(col)
            boot=_source_bootstrap(pairs,np.asarray(columns).T,seed=matched[0]['seed'],n_resamples=50000,alpha=.05)
            for i,ref in enumerate(matched):
                row={k:ref[k] for k in ['model','family','arm','condition','N','K','seed','draws']}
                for key,src in [('mean','point'),('ci_low','ci_low'),('ci_high','ci_high')]:
                    val=boot[src][i];err=abs(val-ref[key]);max_error=max(max_error,err)
                    assert err<1e-12,(model,family,ref['condition'],key,val,ref[key],err)
                    row[key]=val
                results.append(row)
        print(model,family,'PASS',flush=True)
    assert len(results)==126
    (out/'headline_d_bundle_recomputed.json').write_text(json.dumps(results,indent=2)+'\n')
    receipt={'status':'PASS','panels':len(results),'draws_per_panel':50000,'maximum_absolute_error':max_error,'four_cell_occurrences_checked':checked_cells,'GPU_forwards':0,'bundle_input':'native per-pair effects and contrasts','D_input':'native four-cell margins'}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
