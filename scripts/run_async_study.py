"""Small staged, paired-seed studies; every cell uses the audited dynamics harness."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_async_dynamics as dynamics
from scripts.analyze_async_dynamics import analyze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['scout','replication','depth','real','reference'])
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--depths',default='16,64')
    parser.add_argument('--output-root',type=Path,default=Path('results/prebeam'))
    args=parser.parse_args()
    base=yaml.safe_load(Path('configs/async_dynamics.yaml').read_text())
    with Path('configs/eta_by_depth.csv').open() as f:
        rates={int(r['L']):float(r['state_lr']) for r in csv.DictReader(f)}
    seeds=[int(s) for s in args.seeds.split(',')]
    depths=[int(s) for s in args.depths.split(',')] if args.stage=='depth' else [32]
    if args.stage in ('scout','reference'): seeds=[0]
    for depth in depths:
        for seed in seeds:
            n=depth-1
            cfg=copy.deepcopy(base)
            cfg['seed']=seed
            cfg['model']['depth']=depth
            cfg['inference'].update(work_sweep_equivalents=2*depth,reference_budget=4*depth,
                                    state_lr=rates[depth],diagnostic_interval_sweeps=max(1,depth//16))
            sched=101+1009*seed
            def mode(name,**kw): return dict(mode=name,scheduler_seed=sched,**kw)
            if args.stage=='scout':
                modes=[{'mode':'sync'},mode('random_permutation'),mode('forward_ordered_sweep'),
                       mode('reverse_ordered_sweep'),mode('random_block',block_size=24)]
                modes += [mode('bounded_staleness',tau_max=tau) for tau in (0,8,16,31,62)]
                modes += [mode('fully_async_local',tau_max=tau) for tau in (0,8,31)]
                modes += [mode('fully_async_local',tau_max=8,heterogeneous_rate_strength=.5)]
            elif args.stage=='reference':
                modes=[{'mode':'sync'}];cfg['inference']['reference_budget']=8*depth
            else:
                modes=[{'mode':'sync'},mode('random_coordinate'),mode('random_permutation'),
                       mode('bounded_staleness',tau_max=n),mode('fully_async_local',tau_max=0),
                       mode('fully_async_local',tau_max=n)]
                if args.stage=='replication':
                    modes += [mode('bounded_staleness',tau_max=2*n),
                              mode('fully_async_local',tau_max=round(n*.25)),
                              mode('heterogeneous_rates',heterogeneous_rate_strength=1.5)]
            if args.stage=='real':
                cfg['dataset']='fashion_mnist';cfg['model']['input_dim']=784
                with Path('configs/eta_best_by_cell.csv').open() as f:
                    calibration=list(csv.DictReader(f))
                cell=next(r for r in calibration if r['dataset']=='fashion_mnist'
                          and r['activation']=='relu' and int(r['N'])==32 and int(r['L'])==32)
                cfg['inference']['state_lr']=float(cell['eta_best_1_over_lambda_median'])
            cfg['modes']=modes;cfg['plots']=False
            root=args.output_root/args.stage/f'depth{depth}_seed{seed}'
            if (root/'summary.json').exists():
                print('Existing completed cell:',root,flush=True);continue
            root.mkdir(parents=True,exist_ok=True)
            cfg['output_dir']=str(root)
            config_path=root/'input.yaml';config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            sys.argv=['run_async_dynamics.py','--config',str(config_path)]
            dynamics.main()
            analyze(root, plots=seed==0)
            print('STUDY_CELL_COMPLETE',root,flush=True)


if __name__=='__main__': main()
