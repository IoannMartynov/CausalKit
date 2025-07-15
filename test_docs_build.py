"""
Test script to verify that the documentation can be built with the package installed.
"""

import os
import subprocess
import sys

def test_docs_build():
    """Test that the documentation can be built."""
    print("Testing documentation build...")
    
    # Check if mkdocs is installed
    try:
        subprocess.run(["mkdocs", "--version"], check=True, capture_output=True)
        print("MkDocs is installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("MkDocs is not installed. Installing documentation dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=True)
    
    # Build the documentation
    try:
        result = subprocess.run(["mkdocs", "build", "--strict"], check=True, capture_output=True, text=True)
        print("Documentation built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

if __name__ == "__main__":
    success = test_docs_build()
    sys.exit(0 if success else 1)