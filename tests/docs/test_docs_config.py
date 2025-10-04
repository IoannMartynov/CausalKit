import importlib.util
import os
import runpy
from pathlib import Path


def load_conf_module(conf_path: str):
    # Use runpy to execute the conf.py in its own globals dict
    # This avoids polluting sys.modules and works without module name conflicts.
    return runpy.run_path(conf_path)


def test_notebook_execution_disabled():
    # Locate docs/conf.py relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    conf_path = repo_root / 'docs' / 'conf.py'
    assert conf_path.exists(), f"docs/conf.py not found at {conf_path}"

    conf_globals = load_conf_module(str(conf_path))

    # nbsphinx should never execute
    assert conf_globals.get('nbsphinx_execute') == 'never'

    # myst-nb should be configured to not execute (supporting both keys)
    assert conf_globals.get('jupyter_execute_notebooks') == 'off', (
        'Expected jupyter_execute_notebooks == "off" to disable myst-nb execution.'
    )
    assert conf_globals.get('nb_execution_mode') == 'off', (
        'Expected nb_execution_mode == "off" to disable myst-nb execution.'
    )

    # Optional safety: do not raise if someone accidentally triggers execution
    assert conf_globals.get('nb_execution_raise_on_error') is False
