"""Validate raw dynamics outputs and derive work-normalized endpoint/arrival tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_async_dynamics import mode_label


def mode_specs(config):
    return {mode_label(spec): spec for spec in config['modes']}


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def analyze(root: Path, plots: bool = False):
    summary = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    specs = mode_specs(config)
    assert set(summary['modes']) == set(specs), 'Requested and completed modes differ'
    tables = {name: rows(root / (name + '.csv')) for name in
              ('trace', 'layer_trace', 'gradient_trace', 'front', 'events')}
    for name, table in tables.items():
        for row in table:
            for value in row.values():
                assert value.lower() not in ('nan', 'inf', '-inf', 'infinity'), (name, row)
    n = summary['model']['number_of_free_layers']
    reference = summary['reference_terminal_dual_norms_input_to_output']
    assert min(reference) > 1e-6 * max(reference), 'Reference has effectively unidentifiable layers'
    endpoints = []
    uniform_selections = {}
    for label, result in summary['modes'].items():
        trace = [r for r in tables['trace'] if r['mode_label'] == label]
        front = [r for r in tables['front'] if r['mode_label'] == label]
        events = [r for r in tables['events'] if r['mode_label'] == label]
        for r in trace:
            assert r['finite'] == 'True'
            assert math.isclose(float(r['sweep_equivalents']), int(r['layer_update_events']) / n)
        counts = np.zeros(n, dtype=int)
        for event in events:
            selected = [int(x) for x in event['selected_layers_input_to_output'].split(';')]
            counts[selected] += 1
            assert int(event['selected_layer_count']) == len(selected)
            assert counts.sum() == int(event['layer_update_events'])
            assert int(event['state_version_before']) - int(event['read_version']) == int(event['staleness_used'])
        if events:
            assert counts.sum() == result['final_layer_update_events']
            spec = specs[label]
            assert max(int(e['staleness_used']) for e in events) <= spec.get('tau_max', 0)
            if result['mode'] in ('random_coordinate', 'bounded_staleness', 'fully_async_local') and not spec.get('heterogeneous_rate_strength', 0):
                seed = spec.get('scheduler_seed', 0)
                sequence = [e['selected_layers_input_to_output'] for e in events]
                if seed in uniform_selections:
                    previous = uniform_selections[seed]
                    assert sequence[:min(len(sequence), len(previous))] == previous[:min(len(sequence), len(previous))]
                uniform_selections[seed] = sequence
            if result['mode'] == 'fully_async_local':
                assert result['final_dual_update_events'] == counts.sum()
            else:
                assert result['final_dual_update_events'] == counts.sum() // (n * summary['inference']['inner_steps']) * n
        final = trace[-1]
        if result['status'] == 'ok':
            assert int(final['layer_update_events']) == summary['inference']['layer_update_budget']
        gradients = [r for r in tables['gradient_trace'] if r['mode_label'] == label and
                     r['checkpoint_index'] == final['checkpoint_index']]
        first_quarter = [float(r['weight_grad_cos_to_bp']) for r in gradients
                         if int(r['weight_layer_index_input_to_output']) < max(1, (n+1)//4)]
        early_gradients = [r for r in gradients
                          if int(r['weight_layer_index_input_to_output']) < max(1, (n+1)//4)]
        early_norm_ratios = [float(r['weight_grad_norm']) / max(float(r['bp_weight_grad_norm']), 1e-30)
                             for r in early_gradients]
        endpoint = dict(label=label, mode=result['mode'], status=result['status'],
                        cosine=float(final['weight_grad_cos_to_bp']), early_layer_cosine=float(np.mean(first_quarter)),
                        residual=float(final['total_residual_norm']), max_state=max(float(r['max_abs_state']) for r in trace),
                        state_distance=result['final_state_l2_to_sync_same_work'],
                        dual_distance=result['final_dual_l2_to_sync_same_work'],
                        layer_count_min=int(counts.min()) if events else None,
                        layer_count_max=int(counts.max()) if events else None)
        endpoint['early_gradient_norm_ratio_min'] = min(early_norm_ratios)
        endpoint['early_gradient_norm_ratio_mean'] = float(np.mean(early_norm_ratios))
        endpoint['bp_early_gradient_norm_min'] = min(float(r['bp_weight_grad_norm']) for r in early_gradients)
        for threshold in summary['inference']['front_thresholds']:
            f = [r for r in front if float(r['threshold']) == threshold]
            endpoint[f'R_final_{threshold}'] = int(f[-1]['R_contiguous_layers'])
            endpoint[f'R_any_final_{threshold}'] = int(f[-1]['R_any_layers'])
            endpoint[f'max_gap_{threshold}'] = max(int(r['dispersion_gap_layers']) for r in f)
            for fraction in (.25, .5, .75, .9):
                reached = [float(r['sweep_equivalents']) for r in f if int(r['R_contiguous_layers']) >= math.ceil(n*fraction)]
                endpoint[f't{fraction}_threshold{threshold}'] = min(reached) if reached else None
        endpoints.append(endpoint)
    (root/'analysis.json').write_text(json.dumps(endpoints, indent=2, allow_nan=False)+'\n')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for label in summary['modes']:
            t=[r for r in tables['trace'] if r['mode_label']==label]
            f=[r for r in tables['front'] if r['mode_label']==label and float(r['threshold'])==.25]
            for ax, rr, key in [(axes[0,0],f,'R_contiguous_layers'),(axes[0,1],t,'weight_grad_cos_to_bp'),
                                (axes[1,0],f,'dispersion_gap_layers'),(axes[1,1],t,'total_residual_norm')]:
                ax.plot([float(r['sweep_equivalents']) for r in rr],[float(r[key]) for r in rr],label=label)
                ax.set(xlabel='Layer updates / free layers',ylabel=key)
        axes[1,1].set_yscale('symlog',linthresh=.1)
        axes[0,0].legend(fontsize=6)
        fig.savefig(root/'dynamics.png',dpi=150);plt.close(fig)
        sync=[r for r in tables['layer_trace'] if r['mode']=='sync']
        times=sorted(set(float(r['sweep_equivalents']) for r in sync))
        heat=np.array([[float(r['credit_fraction']) for r in sorted(
            (r for r in sync if float(r['sweep_equivalents'])==t),key=lambda r:int(r['distance_from_output']))] for t in times])
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        im=ax.imshow(heat.T,origin='lower',aspect='auto',extent=[min(times),max(times),-.5,n-.5],vmin=0,vmax=1)
        ax.set(xlabel='Sweep equivalents',ylabel='Distance from output (0 = output-adjacent)',title='Official synchronous dual / reference dual')
        fig.colorbar(im,ax=ax);fig.savefig(root/'sync_dual_growth.png',dpi=150);plt.close(fig)
    print(root, 'QC PASS', 'reference range', min(reference), max(reference))
    for e in endpoints:
        print(e['label'], e['status'], 'cos',round(e['cosine'],4),'early',round(e['early_layer_cosine'],4),
              'R25',e.get('R_final_0.25'),'t50',e.get('t0.5_threshold0.25'),'residual',round(e['residual'],4))
    return endpoints


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directories',type=Path,nargs='+');parser.add_argument('--plots',action='store_true')
    args=parser.parse_args()
    for directory in args.directories: analyze(directory,args.plots)
