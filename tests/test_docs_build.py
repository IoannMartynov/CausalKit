"""
Test to verify that the documentation can be built with the package installed.
"""

import os
import subprocess
import sys
import pytest


def is_mkdocs_installed():
    """Check if mkdocs is installed."""
    try:
        subprocess.run(["mkdocs", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_docs_dependencies():
    """Install documentation dependencies."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=True)


@pytest.mark.docs
def test_docs_build():
    """Test that the documentation can be built."""
    # Skip this test if we're in a CI environment or if explicitly requested
    if os.environ.get("SKIP_DOCS_BUILD", "").lower() in ("true", "1", "yes"):
        pytest.skip("Skipping documentation build as requested by environment variable")
    
    # Check if mkdocs is installed, install if not
    if not is_mkdocs_installed():
        install_docs_dependencies()
        # Verify installation was successful
        assert is_mkdocs_installed(), "Failed to install MkDocs"
    
    # Build the documentation
    try:
        result = subprocess.run(
            ["mkdocs", "build", "--strict"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        # Check that the site directory was created
        assert os.path.exists("site"), "Documentation build did not create 'site' directory"
        assert os.path.exists("site/index.html"), "Documentation build did not create index.html"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Documentation build failed: STDOUT: {e.stdout}, STDERR: {e.stderr}")


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main(["-xvs", __file__])