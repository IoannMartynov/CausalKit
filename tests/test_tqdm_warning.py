import warnings
import pytest


def test_causalkit_suppresses_tqdm_iprogress_warning(recwarn):
    # Importing causalkit should install a filter for the specific TqdmWarning
    import importlib
    ck = importlib.import_module('causalkit')

    try:
        from tqdm import TqdmWarning
    except Exception:
        pytest.skip("tqdm not installed; cannot test TqdmWarning suppression")

    # Emit the exact style of message that tqdm.auto emits
    warnings.warn(
        "IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html",
        category=TqdmWarning,
        stacklevel=2,
    )

    # Ensure the warning was suppressed by our filter
    msgs = [str(w.message) for w in recwarn]
    assert not any("IProgress not found" in m for m in msgs), msgs
