import importlib


def test_inference_top_level_exports():
    # These are referenced in docs autosummary under causalis.inference
    mod = importlib.import_module('causalis.inference')
    for name in ['ttest', 'conversion_z_test', 'bootstrap_diff_means', 'dml_atte_source']:
        assert hasattr(mod, name), f"causalis.inference missing expected export: {name}"


def test_inference_subpackages_functions():
    # ATT/ATE/CATE/GATE functions referenced in docs by fully qualified names
    atte = importlib.import_module('causalis.inference.atte')
    assert hasattr(atte, 'dml_atte')

    ate = importlib.import_module('causalis.inference.ate')
    assert hasattr(ate, 'dml_ate')

    cate = importlib.import_module('causalis.inference.cate')
    assert hasattr(cate, 'cate_esimand')

    gate = importlib.import_module('causalis.inference.gate')
    assert hasattr(gate, 'gate_esimand')


def test_refutation_exports():
    ref = importlib.import_module('causalis.refutation')
    for name in [
        'refute_placebo_outcome',
        'refute_placebo_treatment',
        'refute_subset',
        'sensitivity_analysis',
        'get_sensitivity_summary',
        'refute_irm_orthogonality',
    ]:
        assert hasattr(ref, name), f"causalis.refutation missing expected export: {name}"
