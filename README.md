# CausalKit

A Python toolkit for causal inference and experimentation.

## Overview

CausalKit provides tools and utilities for causal inference, A/B testing, and experimental design. The toolkit aims to simplify the process of designing, implementing, and analyzing experiments to determine causal effects.

## Features

### Traffic Splitting

The `split_traffic` function provides a flexible way to split traffic (users, sessions, etc.) for A/B testing and experimentation:

- Simple random splits with customizable ratios
- Support for multiple variants (A/B/C/...)
- Stratified splitting to maintain balanced distributions of important variables
- Reproducible results with random state control

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/CausalKit.git
cd CausalKit

# Install in development mode
pip install -e .

# Install with documentation dependencies
pip install -e ".[docs]"

# Install with development dependencies
pip install -e ".[dev]"
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/CausalKit.git
cd CausalKit

# Create and activate conda environment
conda env create -f environment.yml
conda activate causalkit

# Install the package in development mode
pip install -e .
```

## Usage Examples

### Traffic Splitting

```python
import pandas as pd
from causalkit.utils.traffic_splitter import split_traffic

# Load your data
df = pd.DataFrame({
    'user_id': range(1000),
    'country': ['US', 'UK', 'CA'] * 333 + ['US'],
    'device': ['mobile', 'desktop'] * 500
})

# Simple 50/50 split
control_df, treatment_df = split_traffic(df, random_state=42)

# Custom ratio split (80% control, 20% treatment)
control_df, treatment_df = split_traffic(df, split_ratio=0.8, random_state=42)

# Multiple variants (A/B/C test with 60/20/20 split)
control_df, variant_b_df, variant_c_df = split_traffic(
    df, split_ratio=[0.6, 0.2], random_state=42
)

# Stratified split to ensure balanced country distribution
control_df, treatment_df = split_traffic(
    df, split_ratio=0.5, stratify_column='country', random_state=42
)
```

For more detailed examples, see the [examples directory](examples/).

## Documentation

The documentation for CausalKit is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Online Documentation

The latest documentation is available at: [https://yourusername.github.io/causalkit/](https://yourusername.github.io/causalkit/)

### Building Documentation Locally

To build and view the documentation locally:

```bash
# Install MkDocs and required plugins
pip install mkdocs-material mkdocstrings[python] pymdown-extensions

# Serve the documentation (with live reloading)
mkdocs serve

# Build the documentation
mkdocs build
```

The documentation will be available at http://localhost:8000/.

## Project Structure

```
causalkit/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   └── ttest.py
├── data/
│   ├── __init__.py
│   └── generators.py
├── design/
│   ├── __init__.py
│   └── traffic_splitter.py
├── tests/
│   ├── __init__.py
│   └── test_traffic_splitter.py
└── examples/
    ├── __init__.py
    ├── generators_notebook.ipynb
    ├── traffic_splitting_example.ipynb
    └── ttest.ipynb
```

## Running Tests

### Using pip environment

```bash
# Run all tests
python -m unittest discover

# Run specific test file
python -m unittest causalkit.tests.test_traffic_splitter
```

### Using conda environment

```bash
# Activate the conda environment first
conda activate causalkit

# Run all tests
python -m unittest discover

# Run specific test file
python -m unittest causalkit.tests.test_traffic_splitter
```

## License

[MIT License](LICENSE)
