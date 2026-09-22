#!/usr/bin/env python3
"""Recompute saved-score statistics in a separate directory; never load a model."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/verification_metrics'))
    parser.add_argument('--stages',nargs='+',choices=['baseline','flips','normalized_t','wide_d','tables'],default=['baseline','flips','normalized_t','wide_d','tables'])
    args=parser.parse_args()
    source=Path(__file__).resolve().parent; output=args.output.resolve()
    if output==source or source in output.parents:
        parser.error('Choose an output directory outside the immutable input directory.')
    output.mkdir(parents=True,exist_ok=True)
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='reproduce.py':
            target=output/p.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for stage in args.stages:
        script='make_paper_tables.py' if stage=='tables' else 'analyze_'+stage+'.py'
        subprocess.run([sys.executable,str(output/script)],cwd=output,env=env,check=True)
    checks=[]
    import pandas as pd
    folders={'baseline':'baseline','flips':'flips','normalized_t':'normalized_t','wide_d':'wide_d'}
    for stage in args.stages:
        if stage not in folders:continue
        for original in sorted((source/folders[stage]).glob('*.csv')):
            actual=output/original.relative_to(source)
            left=pd.read_csv(original,float_precision='round_trip');right=pd.read_csv(actual,float_precision='round_trip')
            # CSVs contain numeric and scientific identity columns, not timings.
            pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=0,atol=1e-12)
            checks.append({'file':str(original.relative_to(source)),'rows':len(left),'match_atol':1e-12})
    (output/'reproduction_check.json').write_text(json.dumps({'passed':True,'stages':args.stages,'checks':checks,'model_forwards':False},indent=2)+'\n')
    print(json.dumps({'passed':True,'compared_csvs':len(checks)}))
if __name__=='__main__':main()
