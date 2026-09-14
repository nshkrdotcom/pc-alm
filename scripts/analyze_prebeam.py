"""Cross-cell analysis from saved raw traces; never changes inference semantics."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_async_dynamics import analyze, rows
from summarize_async_study import summarize


def write_csv(path, data):
    if not data:
        return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)


def reference_sensitivity(root):
    reference=json.loads((root/'reference256/summary.json').read_text())['reference_terminal_dual_norms_input_to_output']
    output=[]
    for directory in [root/'supplied_scout',root/'scout/depth32_seed0']:
        if not (directory/'layer_trace.csv').exists():continue
        table=rows(directory/'layer_trace.csv'); grouped=defaultdict(list)
        for row in table:grouped[row['mode_label'],float(row['sweep_equivalents'])].append(row)
        fronts=defaultdict(list)
        for (label,time),layers in sorted(grouped.items()):
            layers.sort(key=lambda r:int(r['distance_from_output']))
            for threshold in (.1,.25,.5,.75):
                for scale in ('128','256'):
                    reach=0
                    for row in layers:
                        denom=float(row['reference_dual_norm']) if scale=='128' else reference[int(row['layer_index_input_to_output'])]
                        if float(row['dual_norm'])/denom < threshold:break
                        reach+=1
                    fronts[label,threshold,scale].append((time,reach))
        for (label,threshold,scale),curve in fronts.items():
            record=dict(source=str(directory),mode=label,threshold=threshold,reference_steps=int(scale),R_final=curve[-1][1])
            for fraction in (.25,.5,.75,.9):
                record[f't{fraction}']=next((t for t,r in curve if r>=math.ceil(31*fraction)),None)
            output.append(record)
    write_csv(root/'reference_sensitivity.csv',output)


def aggregate_plots(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=sorted(stage.glob('depth*_seed*/summary.json'))
    depths=sorted({json.loads(p.read_text())['model']['depth'] for p in cells})
    for depth in depths:
        groups=defaultdict(list)
        for p in cells:
            s=json.loads(p.read_text())
            if s['model']['depth']!=depth:continue
            table=rows(p.parent/'trace.csv');front=rows(p.parent/'front.csv');grad=rows(p.parent/'gradient_trace.csv')
            for label in s['modes']:
                name=label.rsplit('_seed',1)[0]
                t=[r for r in table if r['mode_label']==label]
                f=[r for r in front if r['mode_label']==label and float(r['threshold'])==.25]
                early=[];early_magnitude=[]
                for row in t:
                    g=[r for r in grad if r['mode_label']==label and r['checkpoint_index']==row['checkpoint_index'] and int(r['weight_layer_index_input_to_output'])<depth//4]
                    early.append(np.mean([float(r['weight_grad_cos_to_bp']) for r in g]))
                    early_magnitude.append(np.mean([float(r['weight_grad_norm']) / max(float(r['bp_weight_grad_norm']),1e-30) for r in g]))
                groups[name].append((np.array([float(r['sweep_equivalents']) for r in t]),
                    np.array([float(r['weight_grad_cos_to_bp']) for r in t]),np.array(early),
                    np.array([float(r['R_contiguous_layers'])/(depth-1) for r in f]),
                    np.array([float(r['dispersion_gap_layers']) for r in f]),
                    np.array([float(r['total_residual_norm']) for r in t]),np.array(early_magnitude)))
        fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
        labels=['Gradient cosine to BP','Early-quarter mean layer cosine','Contiguous front / free layers (threshold .25)','Front gap (layers)','Residual norm','Early-quarter gradient / BP norm ratio']
        for name,curves in groups.items():
            for i,ax in enumerate(axes.flat):
                # Only completed common-budget traces are aggregated; no extrapolation.
                assert all(np.array_equal(c[0],curves[0][0]) for c in curves)
                values=np.stack([c[i+1] for c in curves]);mean=values.mean(axis=0)
                line,=ax.plot(curves[0][0],mean,label=name)
                ax.fill_between(curves[0][0],values.min(axis=0),values.max(axis=0),alpha=.09,color=line.get_color())
                ax.set(xlabel='Sweep equivalents',ylabel=labels[i])
        axes[1,1].set_yscale('symlog',linthresh=.1)
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3)
        fig.suptitle(f'{stage.name}: depth {depth}; mean curves, min–max seed bands')
        fig.savefig(stage/f'depth{depth}_aggregate.png',dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    inventory=[]
    for p in sorted(args.root.rglob('summary.json')):
        s=json.loads(p.read_text())
        if 'modes' not in s:continue
        analyze(p.parent,plots=False)
        cfg=json.loads((p.parent/'config.json').read_text())
        for label,mode in s['modes'].items():
            spec=next(r for r in cfg['modes'] if label==__import__('run_async_dynamics').mode_label(r))
            inventory.append(dict(path=str(p.parent),dataset=s['dataset'],seed=s['seed'],
                depth=s['model']['depth'],width=s['model']['width'],activation=s['model']['activation'],
                state_lr=s['inference']['state_lr'],alpha=s['inference']['alpha'],rho=s['inference']['rho'],
                work=s['inference']['work_sweep_equivalents'],reference=s['inference']['reference_budget'],
                mode=label,scheduler_seed=spec.get('scheduler_seed'),block_size=spec.get('block_size',1),
                rate_strength=spec.get('heterogeneous_rate_strength',0),tau=spec.get('tau_max',0),
                tau_sweeps=mode['tau_max_sweep_equivalents'],status=mode['status']))
    write_csv(args.root/'experiment_inventory.csv',inventory)
    reference_sensitivity(args.root)
    for stage in ('replication','depth','real'):
        directory=args.root/stage
        if directory.exists():summarize(directory);aggregate_plots(directory)


if __name__=='__main__':main()
