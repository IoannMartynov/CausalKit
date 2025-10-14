import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import add_score_flags


def _base_rep():
    # minimal skeleton; summary is optional (function can build one)
    return {
        'meta': {'n': 10000, 'score': 'ATE'},
        'influence_diagnostics': {
            'se_plugin': 0.1,
            'p99_over_med': 9.0,
            'kurtosis': 8.0,
        },
        # OOS test placeholder (overridden in specific tests)
        'oos_moment_test': {
            'tstat_fold_agg': 0.05,
            'tstat_strict': 0.05,
        },
    }


def test_effect_size_guard_downgrades_red_when_oos_green():
    rep = _base_rep()
    # Orthogonality with large t, but tiny constant-basis derivatives
    ortho = pd.DataFrame({
        'basis': [0, 1],
        'd_g1':  [0.010, 0.010],
        'se_g1': [0.001, 0.001],
        't_g1':  [10.0, 10.0],
        'd_g0':  [0.015, 0.020],
        'se_g0': [0.001, 0.001],
        't_g0':  [6.0,  6.0],
        'd_m':   [0.005, 0.005],
        'se_m':  [0.001, 0.001],
        't_m':   [8.0,  8.0],
    })
    rep['orthogonality_derivatives'] = ortho
    # OOS green (near zero t)
    rep['oos_moment_test'] = {'tstat_fold_agg': 0.01, 'tstat_strict': 0.01}

    out = add_score_flags(rep)  # default effect_size_guard=0.02, oos_gate=True
    f = out['flags']
    # Initially would be RED due to large |t|; guard should downgrade to GREEN because OOS is GREEN
    assert f['ortho_max_|t|_g1'] == 'GREEN'
    assert f['ortho_max_|t|_g0'] == 'GREEN'
    assert f['ortho_max_|t|_m'] == 'GREEN'
    # Overall should not be RED now
    assert out['overall_flag'] in {'GREEN', 'YELLOW'}


def test_effect_size_guard_downgrades_red_to_yellow_when_oos_not_green():
    rep = _base_rep()
    ortho = pd.DataFrame({
        'basis': [0, 1],
        'd_g1':  [0.010, 0.010],
        'se_g1': [0.001, 0.001],
        't_g1':  [10.0, 10.0],
        'd_g0':  [0.015, 0.020],
        'se_g0': [0.001, 0.001],
        't_g0':  [6.0,  6.0],
        'd_m':   [0.005, 0.005],
        'se_m':  [0.001, 0.001],
        't_m':   [8.0,  8.0],
    })
    rep['orthogonality_derivatives'] = ortho
    # OOS not green (tstat high)
    rep['oos_moment_test'] = {'tstat_fold_agg': 3.5, 'tstat_strict': 3.5}

    out = add_score_flags(rep)
    f = out['flags']
    # Guard should downgrade RED to YELLOW when OOS isn't GREEN
    assert f['ortho_max_|t|_g1'] == 'YELLOW'
    assert f['ortho_max_|t|_g0'] == 'YELLOW'
    assert f['ortho_max_|t|_m'] == 'YELLOW'


def test_huge_n_relaxes_tail_and_kurtosis_flags():
    # Construct with large n and borderline tail/kurtosis values
    rep = {
        'meta': {'n': 250_000, 'score': 'ATE'},
        'influence_diagnostics': {
            'se_plugin': 0.05,
            'p99_over_med': 11.5,  # → initial YELLOW
            'kurtosis': 45.0,      # → initial RED (>30)
        },
        'orthogonality_derivatives': pd.DataFrame({
            'basis': [0], 'd_g1': [0.0], 'se_g1': [1.0], 't_g1': [0.0],
            'd_g0': [0.0], 'se_g0': [1.0], 't_g0': [0.0],
            'd_m':  [0.0], 'se_m':  [1.0], 't_m':  [0.0],
        }),
        'oos_moment_test': {'tstat_fold_agg': 0.0, 'tstat_strict': 0.0},
    }

    out = add_score_flags(rep)
    f = out['flags']
    assert f['psi_tail_ratio'] == 'GREEN'   # YELLOW→GREEN under gate
    assert f['psi_kurtosis'] == 'YELLOW'    # RED→YELLOW under gate
