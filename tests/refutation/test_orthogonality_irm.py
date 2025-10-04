from causalkit.data.generators import CausalDatasetGenerator
from causalkit.refutation.score.score_validation import refute_irm_orthogonality
from causalkit.inference.atte.dml_atte import dml_atte

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def test_refute_irm_orthogonality_att_runs_and_returns_keys():
    # Generate synthetic data and convert to CausalData
    gen = CausalDatasetGenerator(seed=123)
    # Avoid constant columns like 'm_obs' by specifying explicit confounders
    cd = gen.to_causal_data(n=300, confounders=['x1','x2','x3','x4','x5'])

    # Lightweight sklearn learners to avoid optional dependencies
    ml_g = RandomForestRegressor(n_estimators=30, random_state=1)
    ml_m = RandomForestClassifier(n_estimators=30, random_state=2)

    # Run orthogonality diagnostics with ATT estimator
    res = refute_irm_orthogonality(
        dml_atte,
        cd,
        n_folds_oos=3,
        score="ATTE",  # explicit modern API
        ml_g=ml_g,
        ml_m=ml_m,
    )

    # Basic shape and key checks
    assert isinstance(res, dict)
    assert "theta" in res
    assert "oos_moment_test" in res
    assert "orthogonality_derivatives" in res
    assert "influence_diagnostics" in res

    # Check that derivative outputs are dataframes with expected columns
    ortho = res["orthogonality_derivatives"]["full_sample"]
    assert set(["d_m0", "d_g", "t_m0", "t_g"]).issubset(set(ortho.columns))

    # OOS test outputs should include a tstat
    assert "tstat" in res["oos_moment_test"]
