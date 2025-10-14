import numpy as np
import pandas as pd

from causalis.refutation.score.score_validation import run_score_diagnostics, add_score_flags


def _make_synth(seed=0, n=250):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta0 = np.array([0.7, -0.4, 0.2])
    tau = 1.0
    g0 = X @ beta0
    g1 = g0 + tau
    logits = X @ np.array([0.6, -0.1, 0.15])
    m = 1 / (1 + np.exp(-logits))
    d = rng.binomial(1, m).astype(float)
    y = g0 + d * tau + rng.normal(scale=1.0, size=n)
    # AIPW plug-in theta using true nuisances for stability
    m_clip = np.clip(m, 1e-3, 1-1e-3)
    theta_terms = (g1 - g0) + d * (y - g1) / m_clip - (1 - d) * (y - g0) / (1 - m_clip)
    theta = float(theta_terms.mean())
    return y, d, g0, g1, m, theta


def test_add_score_flags_integration_with_run_score_diag():
    y, d, g0, g1, m, theta = _make_synth(seed=123, n=200)
    rep = run_score_diagnostics(y=y, d=d, g0=g0, g1=g1, m=m, theta=theta, score='ATE', return_summary=True)
    rep2 = add_score_flags(rep)
    # keys added
    assert 'flags' in rep2
    assert 'overall_flag' in rep2
    assert 'thresholds' in rep2
    # summary has flag column
    assert 'summary' in rep2 and isinstance(rep2['summary'], pd.DataFrame)
    assert 'flag' in rep2['summary'].columns
    # overall flag is one of allowed
    assert rep2['overall_flag'] in {"GREEN","YELLOW","RED","NA"}


def test_add_score_flags_deterministic_rules():
    # handcrafted minimal report
    infl = {
        'se_plugin': 0.1,
        'p99_over_med': 15.0,  # between 10 warn and 20 strong -> YELLOW
        'kurtosis': 35.0,      # > 30 strong -> RED
    }
    # orthogonality table
    df = pd.DataFrame({
        'basis': [0,1,2],
        't_g1': [0.5, 1.0, 1.5],    # max 1.5 -> GREEN (t_warn=2)
        't_g0': [2.5, 0.1, -0.2],   # max 2.5 -> YELLOW
        't_m':  [5.0, 0.0, -0.1],   # max 5.0 -> RED
    })
    oos = {
        'tstat_fold_agg': 3.5,  # -> RED (oos_strong=3)
        'tstat_strict': 2.5,    # -> YELLOW (oos_warn=2, oos_strong=3)
    }
    rep = {
        'influence_diagnostics': infl,
        'orthogonality_derivatives': df,
        'oos_moment_test': oos,
        'summary': pd.DataFrame([
            {'metric':'se_plugin','value': infl['se_plugin']},
            {'metric':'psi_p99_over_med','value': infl['p99_over_med']},
            {'metric':'psi_kurtosis','value': infl['kurtosis']},
            {'metric':'max_|t|_g1','value': float(np.max(np.abs(df['t_g1'])))},
            {'metric':'max_|t|_g0','value': float(np.max(np.abs(df['t_g0'])))},
            {'metric':'max_|t|_m','value': float(np.max(np.abs(df['t_m'])))},
            {'metric':'oos_tstat_fold','value': oos['tstat_fold_agg']},
            {'metric':'oos_tstat_strict','value': oos['tstat_strict']},
        ])
    }
    rep2 = add_score_flags(rep, thresholds={
        'tail_ratio_warn': 10.0,
        'tail_ratio_strong': 20.0,
        'kurt_warn': 10.0,
        'kurt_strong': 30.0,
        't_warn': 2.0,
        't_strong': 4.0,
        'oos_warn': 2.0,
        'oos_strong': 3.0,
    })

    f = rep2['flags']
    assert f['psi_tail_ratio'] == 'YELLOW'
    assert f['psi_kurtosis'] == 'RED'
    assert f['ortho_max_|t|_g1'] == 'GREEN'
    assert f['ortho_max_|t|_g0'] == 'YELLOW'
    assert f['ortho_max_|t|_m'] == 'RED'
    # OOS: strict present -> canonical YELLOW; fold RED also present
    assert f['oos_tstat_fold'] == 'RED'
    assert f['oos_tstat_strict'] == 'YELLOW'
    assert f['oos_moment'] == 'YELLOW'
    # Overall should be RED due to some RED flags
    assert rep2['overall_flag'] == 'RED'

    # Summary should carry flags
    df_sum = rep2['summary']
    assert set(['metric','value','flag']).issubset(df_sum.columns)
    # check that some mapped flags appear
    m = {r['metric']: r['flag'] for _, r in df_sum.iterrows()}
    assert m.get('psi_kurtosis') == 'RED'
    assert m.get('max_|t|_m') == 'RED'
