# Installation

This guide shows the quickest way to install CausalKit directly from GitHub.

## Prerequisites
- Python 3.7+ (per project configuration)
- A clean virtual environment is recommended (venv or conda)

## Install from GitHub (recommended)
```bash
pip install --upgrade pip
pip install "git+https://github.com/ioannmartynov/causalkit.git"
```

### Verify the installation
```bash
python -c "import causalkit; print(causalkit.__version__)"
```

## Optional: editable install (for contributors)
If you plan to work with the source code, clone the repo and install in editable mode:
```bash
git clone https://github.com/ioannmartynov/CausalKit.git
cd CausalKit
pip install -e .
# or include development tools
pip install -e ".[dev]"
```

## Optional: conda environment after cloning the repo
You can also create a conda env using the provided file (useful for local development):
```bash
conda env create -f environment.yml
conda activate causalis
```

You're all set! Import the library to get started:

```python
import causalis
```
