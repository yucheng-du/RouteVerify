#!/usr/bin/env python3
"""Recompute published supplementary statistics from saved, text-free scores (CPU only).
No model loading, network access, classifier fitting, or model forwards.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
CHECKS=[]
def read_csv(p):
 with p.open() as f:return list(csv.DictReader(f))
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def close(label,a,b):
 assert np.isfinite(float(a)) and np.isfinite(float(b)), f'{label}: nonfinite comparison'
 err=abs(float(a)-float(b));CHECKS.append({'check':label,'max_abs_error':err})
 if err>2e-12:raise AssertionError(f'{label}: computed={a}, archived={b}, error={err}')
def interval(x):
 assert np.all(np.isfinite(x))
 return np.quantile(x,[.025,.975])
def bootstrap_auc(rows, seed=20260829, n_resamples=1000, expected_rows=580, expected_groups=235):
 """Algebraically equal to duplicating all rows in sampled source groups.
 Ties receive half credit. Fixed ordering and PCG64 draws match saved analysis.
 """
 groups=sorted({r['analysis_group_id'] for r in rows})
 assert len(rows)==expected_rows and len(groups)==expected_groups
 gi={g:i for i,g in enumerate(groups)}
 seen={}
 for r in rows:
  g=r['analysis_group_id'];fold=r['fold_index']
  if g in seen:assert seen[g]==fold
  seen[g]=fold
 scores=np.array([r['score'] for r in rows]);pos=np.array([r['label']=='corrupt' for r in rows]);order=np.argsort(scores,kind='stable');s=scores[order]
 idx=np.random.default_rng(seed).integers(0,len(groups),(n_resamples,len(groups)))
 count=np.zeros((n_resamples,len(groups)),dtype=np.int64)
 np.add.at(count,(np.arange(n_resamples)[:,None],idx),1)
 weights=count[:,[gi[r['analysis_group_id']] for r in rows]]
 weights=np.vstack([np.ones(len(rows),dtype=np.int64),weights])[:,order]
 p=weights*pos[order];n=weights*(~pos[order]);starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
 pp=np.add.reduceat(p,starts,axis=1);nn=np.add.reduceat(n,starts,axis=1)
 numerator=(pp*(np.cumsum(nn,axis=1)-.5*nn)).sum(axis=1)
 auc=numerator/(pp.sum(axis=1)*nn.sum(axis=1))
 return float(auc[0]),auc[1:]
def check_est_ci(label,point,draws,expected,field='estimate'):
 lo,hi=interval(draws)
 for key,v in [(field,point),('ci_low',lo),('ci_high',hi)]:close(label+':'+key,v,expected[key])
def probes():
 results={};summary=[]
 for name in ['canonical_six','glm','gpt120']:
  root=ROOT/'probe'/name;rows=read_jsonl(root/'interface_access_oof_predictions.jsonl');grouped=defaultdict(list)
  for r in rows:grouped[r['model_id'],r['anchor'],r['feature_family']].append(r)
  reference=read_csv(root/'interface_access_summary.csv')
  for r in reference:
   key=(r['model_id'],r['anchor'],r['feature_family']);point,draws=bootstrap_auc(grouped[key]);results[key]=(point,draws)
   check_est_ci('probe/'+name+'/'+str(key),point,draws,r,'micro_oof_auroc');summary.append({'panel':name,'model_id':key[0],'anchor':key[1],'feature':key[2],'auroc':point,'ci_low':float(interval(draws)[0]),'ci_high':float(interval(draws)[1])})
  for fn in ['interface_access_adjacent_differences.csv','interface_access_direct_differences.csv']:
   p=root/fn
   if p.exists():
    for r in read_csv(p):
     a=results[r['model_id'],r['anchor'],r['left_feature']];b=results[r['model_id'],r['anchor'],r['right_feature']]
     check_est_ci('probe contrast/'+str(r['model_id'])+'/'+r['anchor']+'/'+r['left_feature']+'/'+r['right_feature'],a[0]-b[0],a[1]-b[1],r,'difference')
 textrows=defaultdict(list)
 for r in read_jsonl(ROOT/'probe/text_baseline/oof_predictions.jsonl'):textrows[r['model_id']].append(r)
 text={m:bootstrap_auc(v) for m,v in textrows.items()}
 for r in read_csv(ROOT/'probe/text_baseline/summary.csv'):
  p,d=text[r['model_id']];check_est_ci('text/'+r['model_id'],p,d,r,'micro_oof_auroc')
 for r in read_csv(ROOT/'probe/text_baseline/router_minus_text.csv'):
  a=results[r['model_id'],r['anchor'],'router_logits'];b=text[r['model_id']]
  check_est_ci('router-minus-text/'+r['model_id']+'/'+r['anchor'],a[0]-b[0],a[1]-b[1],r,'difference')
 historical=json.loads((ROOT/'probe/qwen3_historical/matched_estimator_comparison.json').read_text())
 h={}
 for feature in ['routing','surface']:
  rr=[dict(r,analysis_group_id=r['source_problem_id'],score=r[feature+'_score']) for r in historical['predictions']]
  h[feature]=bootstrap_auc(rr,seed=20260808,n_resamples=2000,expected_rows=1160,expected_groups=384)
 expected=historical['paired_auroc_difference']
 close('historical Qwen routing',h['routing'][0],expected['left']['auroc'])
 close('historical Qwen surface',h['surface'][0],expected['right']['auroc'])
 check_est_ci('historical Qwen routing-minus-text',h['routing'][0]-h['surface'][0],h['routing'][1]-h['surface'][1],expected,'difference')
 narrow_rows=defaultdict(list)
 for r in read_jsonl(ROOT/'probe/native238/interface_access_oof_predictions.jsonl'):narrow_rows[r['model_id'],r['anchor'],r['feature_family']].append(r)
 narrow={}
 for r in read_csv(ROOT/'probe/native238/interface_access_summary.csv'):
  key=(r['model_id'],r['anchor'],r['feature_family']);point,draws=bootstrap_auc(narrow_rows[key],expected_groups=238);narrow[key]=(point,draws)
  check_est_ci('native238/'+str(key),point,draws,r,'micro_oof_auroc')
  summary.append({'panel':'native238','model_id':key[0],'anchor':key[1],'feature':key[2],'auroc':point,'ci_low':float(interval(draws)[0]),'ci_high':float(interval(draws)[1])})
 for r in read_csv(ROOT/'probe/native238/interface_access_adjacent_differences.csv'):
  a=narrow[r['model_id'],r['anchor'],r['left_feature']];b=narrow[r['model_id'],r['anchor'],r['right_feature']]
  check_est_ci('native238 contrast/'+r['model_id']+'/'+r['anchor']+'/'+r['left_feature']+'/'+r['right_feature'],a[0]-b[0],a[1]-b[1],r,'difference')
 selected=['post_moe_residual','router_input','router_logits','expert_id_aligned_sparse_weights']
 max_delta=max(abs(v[0]-results[k][0]) for k,v in narrow.items() if k[1]=='middle' and k[2] in selected)
 assert max_delta<=.031
 return summary
def processbench():
 p=ROOT/'processbench';pairrows=read_csv(p/'analysis/pair_effects.csv');derived={(r['model_id'],r['pair_id'],r['site_family'],r['direction']):r for r in pairrows};computed=[]
 effects=read_csv(p/'analysis/effect_distribution.csv');strata=read_csv(p/'analysis/receiver_state_effects.csv');contrasts=read_csv(p/'analysis/receiver_state_contrasts.csv');comp=read_csv(p/'analysis/problem_component_sensitivity.csv')
 construction=read_csv(p/'analysis/construction_pairs.csv');ids=sorted(r['pair_id'] for r in construction);assert len(ids)==64
 cmap={r['pair_id']:r['problem_component_id'] for r in construction};components=sorted(set(cmap.values()));assert len(components)==61
 ci={v:i for i,v in enumerate(components)};indices=np.random.default_rng(0).integers(0,64,(50000,64))
 cidx=np.random.default_rng(0).integers(0,61,(50000,61));counts=np.zeros((50000,61),dtype=np.int64);np.add.at(counts,(np.arange(50000)[:,None],cidx),1);weights=counts[:,[ci[cmap[x]] for x in ids]]
 for model in ['gpt_oss_20b','gemma4','nemotron3_nano']:
  rows=read_jsonl(p/'runs'/model/'transfer/predictions.jsonl');assert len(rows)==768
  behavior=read_jsonl(p/'runs'/model/'behavior/predictions.jsonl');assert len(behavior)==128
  def margin(r):
   m=r['scores']['INCORRECT']['mean_logprob']-r['scores']['CORRECT']['mean_logprob'];close('score margin',m,r['margin_incorrect_minus_correct_mean']);return m
  base={(r['pair_id'],r['receiver_role']):margin(r) for r in rows if r['treatment']=='unpatched'}
  interventions={(r['pair_id'],r['site'].split('.')[-1],r['treatment']):r for r in rows if r['treatment'] in ['clean_to_corrupt','corrupt_to_clean']}
  assert len(base)==128 and len(interventions)==256
  native=0;identity=0
  for r in rows:
   if r['treatment'] in ['self_patch','unpatched_rerun']:
    close('identity',margin(r),base[r['pair_id'],r['receiver_role']]);identity+=1
   if r['site'] and r['site'].endswith('router_logits') and r['treatment'] in ['clean_to_corrupt','corrupt_to_clean']:
    assert r['metadata']['manipulation']['all_router_dispatch_verified'];native+=r['metadata']['manipulation']['candidate_count']
  assert identity==384 and native==256
  pred=np.array([r['prediction']==r['gold'] for r in behavior]);bm=json.loads((p/'runs'/model/'behavior/metrics.json').read_text())
  close('behavior accuracy/'+model,pred.mean(),bm['accuracy'])
  labels=np.array([r['gold']=='INCORRECT' for r in behavior]);scores=np.array([margin(r) for r in behavior]);auc=np.mean((scores[labels,None]>scores[~labels]).astype(float)+.5*(scores[labels,None]==scores[~labels]))
  close('behavior auroc/'+model,auc,bm['auroc'])
  vectors={}
  for direction,receiver in [('clean_to_corrupt','first_error'),('corrupt_to_clean','matched_correct')]:
   for site in ['post_moe_residual','router_logits']:
    x=[];mask=[]
    for pid in ids:
     r=interventions[pid,site,direction];d=margin(r)-base[pid,receiver];expected=derived[model,pid,site,direction];close('atomic-to-pair effect',d,expected['delta']);x.append(d);mask.append(expected['receiver_judgment_correct']=='True')
     assert r['metadata']['alpha']==1.0
    x=np.array(x);mask=np.array(mask);vectors[direction,site]=x
    key={'model_id':model,'site_family':site,'direction':direction}
    filt=lambda rows:next(r for r in rows if all(r[k]==v for k,v in key.items()))
    point=x.mean();draws=x[indices].mean(axis=1);check_est_ci('PB direction/'+str(key),point,draws,filt(effects))
    rng=np.random.default_rng(0);sd={};sp={}
    for correct in [True,False]:
     sx=x[mask==correct];si=rng.integers(0,len(sx),(50000,len(sx)));sp[correct]=sx.mean();sd[correct]=sx[si].mean(axis=1)
     er=next(r for r in strata if all(r[k]==v for k,v in key.items()) and r['receiver_judgment_correct']==str(correct));check_est_ci('receiver-stratum/'+str(key)+'/'+str(correct),sp[correct],sd[correct],er)
    check_est_ci('receiver contrast/'+str(key),sp[False]-sp[True],sd[False]-sd[True],filt(contrasts))
    cd={}
    for c in [True,False]:
     w=weights[:,mask==c];den=w.sum(axis=1)
     cd[c]=np.divide(w@x[mask==c],den,out=np.full(50000,np.nan),where=den>0)
    vals={'all_pairs':(point,weights@x/weights.sum(axis=1)),'receiver_correctly_judged':(sp[True],cd[True]),'receiver_misjudged':(sp[False],cd[False]),'misjudged_minus_correctly_judged':(sp[False]-sp[True],cd[False]-cd[True])}
    for stat,(pt,dr) in vals.items():
     er=next(r for r in comp if all(r[k]==v for k,v in key.items()) and r['statistic']==stat)
     valid=np.isfinite(dr);assert valid.sum()==int(er['valid_replicates']) and (~valid).sum()==int(er['empty_stratum_replicates'])
     check_est_ci('component/'+str(key)+'/'+stat,pt,dr[valid],er)
    computed.append(key|{'estimate':point,'ci_low':float(interval(draws)[0]),'ci_high':float(interval(draws)[1])})
  ts={s:(vectors['corrupt_to_clean',s]-vectors['clean_to_corrupt',s])/2 for s in ['post_moe_residual','router_logits']}
  ts['residual_minus_router']=ts['post_moe_residual']-ts['router_logits']
  for s,x in ts.items():
   er=next(r for r in comp if r['model_id']==model and r['site_family']==s and r['direction']=='paired_directions');check_est_ci('component T/gap/'+model+'/'+s,x.mean(),weights@x/weights.sum(axis=1),er)
 return computed
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT.parent/'build/secondary');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 pr=probes();pb=processbench();payload={'status':'pass','scope':'CPU recomputation from archived OOF and atomic ProcessBench scores; no forwards or refits','numeric_checks':len(CHECKS),'maximum_absolute_error':max(r['max_abs_error'] for r in CHECKS),'numpy':np.__version__,'probe_cells':len(pr),'probe':pr,'processbench':pb}
 (args.output/'validation.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k not in ['probe','processbench']},indent=2))
if __name__=='__main__':main()
