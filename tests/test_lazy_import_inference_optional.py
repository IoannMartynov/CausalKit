import sys
import subprocess


def test_importing_data_does_not_import_inference():
    """
    Importing causalkit.data should not import the heavy optional
    subpackage causalkit.inference (which may depend on catboost).
    This guards against ModuleNotFoundError in environments where
    optional dependencies are not installed (e.g., docs renderer).
    """
    code = (
        "import sys\n"
        "from causalkit.data import CausalDatasetGenerator, CausalData\n"
        "import causalkit\n"
        "assert 'causalkit.inference' not in sys.modules\n"
        "assert 'catboost' not in sys.modules\n"
        "print('OK')\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert (
        proc.returncode == 0
    ), f"Subprocess failed with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    assert "OK" in proc.stdout
