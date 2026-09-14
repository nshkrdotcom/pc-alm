from scripts.analyze_async_dynamics import mode_specs
from scripts.summarize_async_study import key


def test_analysis_does_not_confuse_rate_or_seed_prefixes():
    config = {'modes': [
        dict(mode='fully_async_local', tau_max=8, heterogeneous_rate_strength=h, scheduler_seed=s)
        for h in (0, .5) for s in (1, 11)
    ]}
    specs = mode_specs(config)
    assert len(specs) == 4
    for h in (0, .5):
        for seed in (1, 11):
            spec = specs[f'fully_async_local_tau8_h{h:g}_seed{seed}']
            assert spec['heterogeneous_rate_strength'] == h
            assert spec['scheduler_seed'] == seed
    assert key({'label':'fully_async_local_tau8_h0_seed1'}) != key({'label':'fully_async_local_tau8_h0.5_seed1'})
    assert key({'label':'fully_async_local_tau8_h0_seed1'}) == key({'label':'fully_async_local_tau8_h0_seed11'})
