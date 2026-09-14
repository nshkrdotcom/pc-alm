"""Paired descriptive statistics; no independence assumption across checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def key(row):
    # Keep every delay/rate/block setting distinct; remove only the paired seed.
    return row['label'].rsplit('_seed', 1)[0]


def summarize(root):
    grouped={}
    for path in sorted(root.glob('depth*_seed*/analysis.json')):
        summary=json.loads((path.parent/'summary.json').read_text())
        assert summary['matmul_precision']=='highest' and summary['backend']=='gpu'
        data=json.loads(path.read_text())
        sync=next(r for r in data if r['mode']=='sync')
        for row in data:
            row.update(seed=summary['seed'],depth=summary['model']['depth'],
                       paired_cosine_delta=row['cosine']-sync['cosine'],
                       paired_early_cosine_delta=row['early_layer_cosine']-sync['early_layer_cosine'])
            grouped.setdefault((row['depth'],key(row)),[]).append(row)
    out=[]
    for (depth,mode),group in sorted(grouped.items()):
        r=dict(depth=depth,mode=mode,n=len(group),ok=sum(g['status']=='ok' for g in group))
        for name in ['cosine','early_layer_cosine','paired_cosine_delta','paired_early_cosine_delta',
                     'early_gradient_norm_ratio_min','early_gradient_norm_ratio_mean','bp_early_gradient_norm_min',
                     'residual','max_state','state_distance','dual_distance','R_final_0.25','R_any_final_0.25','max_gap_0.25']:
            a=np.array([g[name] for g in group],dtype=float)
            r[name+'_mean']=float(a.mean());r[name+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
            r[name+'_min']=float(a.min());r[name+'_max']=float(a.max())
        for threshold in (.1,.25,.5,.75):
            for fraction in (.25,.5,.75,.9):
                name=f't{fraction}_threshold{threshold}'
                a=[g[name] for g in group if g[name] is not None]
                r[name+'_reached']=len(a)
                r[name+'_median_reached_only']=float(np.median(a)) if a else None
        out.append(r)
    (root/'aggregate.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    if out:
        with (root/'aggregate.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    for r in out:
        print(r['depth'],r['mode'],f"ok {r['ok']}/{r['n']}",
              f"cos {r['cosine_mean']:.4f} +/- {r['cosine_sd'] or 0:.4f}",
              f"early {r['early_layer_cosine_mean']:.4f}",f"paired delta {r['paired_cosine_delta_mean']:+.4f}",
              f"R {r['R_final_0.25_mean']:.1f}",f"t50 {r['t0.5_threshold0.25_median_reached_only']}")
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
