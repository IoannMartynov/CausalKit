import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
NB_PATH = os.path.join(PROJECT_ROOT, 'examples', 'observational_binary_treatment.ipynb')

def test_notebook_exists():
    assert os.path.exists(NB_PATH), "Example notebook for observational binary treatment must exist"


def test_notebook_has_required_steps():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    # Check that it's a notebook-like file with our key steps and uses causalkit only
    assert '#%%' in content or 'cells' in content.lower(), "Notebook/script should contain cell markers or JSON"
    # Required components per the scenario
    assert 'CausalData' in content, "CausalData object creation should be shown"
    assert 'CausalEDA' in content, "EDA step using CausalEDA should be present"
    assert 'from causalkit.inference' in content and 'dml' in content, "Inference step using causalkit.inference should be present"
    assert 'gate_esimand' in content, "GATE analysis should be performed"
    # Ensure we do not import external causal libraries
    forbidden = ['econml', 'causalml', 'dowhy']
    assert not any(lib in content for lib in forbidden), "Only causalkit should be used in the example"
