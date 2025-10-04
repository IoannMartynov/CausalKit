import importlib.util
import sys
from pathlib import Path


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_docs_conf_version_release_are_strings_and_parseable():
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    conf_py = docs_dir / "conf.py"
    assert conf_py.exists(), f"conf.py not found at {conf_py}"

    mod = load_module_from_path("docs_conf", str(conf_py))

    # Check attributes exist and are strings
    assert hasattr(mod, "version"), "conf.py must define 'version'"
    assert hasattr(mod, "release"), "conf.py must define 'release'"
    assert isinstance(mod.version, str), "version must be a string"
    assert isinstance(mod.release, str), "release must be a string"

    # Ensure packaging can parse them
    from packaging.version import Version

    Version(mod.version)
    Version(mod.release)


if __name__ == "__main__":
    # Allow running this test directly
    import pytest

    sys.exit(pytest.main(["-xvs", __file__]))
