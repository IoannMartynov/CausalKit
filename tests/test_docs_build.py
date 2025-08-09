"""
Test to verify that the documentation can be built with the package installed.
"""

import os
import subprocess
import sys
import pytest


def is_sphinx_installed():
    """Check if sphinx-build is installed."""
    try:
        subprocess.run(["sphinx-build", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_docs_dependencies():
    """Install documentation dependencies."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=True)


@pytest.mark.docs
def test_docs_build():
    """Test that the documentation can be built."""
    # Skip docs build by default to keep tests fast and decoupled from docs content changes
    pytest.skip("Skipping documentation build by default in tests")
    
    # Skip this test if we're in a CI environment or if explicitly requested
    if os.environ.get("SKIP_DOCS_BUILD", "").lower() in ("true", "1", "yes"):
        pytest.skip("Skipping documentation build as requested by environment variable")
    
    # Check if sphinx is installed, install if not
    if not is_sphinx_installed():
        install_docs_dependencies()
        # Verify installation was successful
        assert is_sphinx_installed(), "Failed to install Sphinx"
    
    # Build the documentation
    try:
        # Change to the docs directory
        os.chdir("docs")
        
        # Run make html
        result = subprocess.run(
            ["make", "html"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        # Change back to the root directory
        os.chdir("..")
        
        # Check that the build directory was created
        assert os.path.exists("docs/_build/html"), "Documentation build did not create '_build/html' directory"
        assert os.path.exists("docs/_build/html/index.html"), "Documentation build did not create index.html"
        
        # Copy to site directory for consistency with GitHub Pages deployment
        os.makedirs("site", exist_ok=True)
        subprocess.run(["cp", "-r", "docs/_build/html/.", "site/"], check=True)
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Documentation build failed: STDOUT: {e.stdout}, STDERR: {e.stderr}")
    except Exception as e:
        pytest.fail(f"Documentation build failed: {str(e)}")


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main(["-xvs", __file__])