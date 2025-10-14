import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import add_score_flags


def _minimal_rep(se_plugin: float) -> dict:
    # Minimal report with required pieces
    return {
        'influence_diagnostics': {
            'se_plugin': float(se_plugin),
            'p99_over_med': np.nan,
            'kurtosis': np.nan,
        },
        'orthogonality_derivatives': pd.DataFrame({
            'basis': [0],
            't_g1': [np.nan],
            't_g0': [np.nan],
            't_m':  [np.nan],
            'd_g1': [0.0],
            'd_g0': [0.0],
            'd_m':  [0.0],
            'se_g1': [np.nan],
            'se_g0': [np.nan],
            'se_m':  [np.nan],
        }),
        'summary': pd.DataFrame([
            {'metric': 'se_plugin', 'value': float(se_plugin)},
        ]),
        'meta': {'n': 1000, 'score': 'ATE'},
    }


def test_se_plugin_flag_model_green_yellow_red():
    rep_base = _minimal_rep(0.20)

    # GREEN: exact match
    rep_g = add_score_flags(rep_base, se_rule="model", se_ref=0.20)
    assert rep_g['flags'].get('se_plugin') == 'GREEN'
    # thresholds exposed
    thr = rep_g.get('thresholds', {})
    assert 'se_rel_warn' in thr and 'se_rel_strong' in thr

    # YELLOW: 0.20 vs 0.28 → rel ≈ 0.2857 > 0.25 and < 0.50
    rep_y = add_score_flags(rep_base, se_rule="model", se_ref=0.28)
    assert rep_y['flags'].get('se_plugin') == 'YELLOW'

    # RED: 0.20 vs 0.40 → rel = 0.50 ≥ 0.50
    rep_r = add_score_flags(rep_base, se_rule="model", se_ref=0.40)
    assert rep_r['flags'].get('se_plugin') == 'RED'

    # Summary should reflect se_plugin flag
    summ = rep_r['summary']
    row = summ.loc[summ['metric'] == 'se_plugin']
    assert not row.empty and row['flag'].iloc[0] == 'RED'


def test_se_plugin_flag_na_when_not_requested():
    rep_base = _minimal_rep(0.33)
    rep = add_score_flags(rep_base)  # no se_rule/se_ref -> NA
    assert rep['flags'].get('se_plugin') == 'NA'
    row = rep['summary'].loc[rep['summary']['metric'] == 'se_plugin']
    assert not row.empty and row['flag'].iloc[0] == 'NA'
