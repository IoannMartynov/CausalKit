# Installation

CausalKit can be installed using pip or from source.

## Prerequisites

CausalKit requires:

- Python 3.7 or later
- NumPy
- Pandas
- SciPy
- Statsmodels
- DoubleML (for advanced methods)

## Installing with pip

The simplest way to install CausalKit is using pip:

```bash
pip install causalkit
```

This will install CausalKit and all its dependencies.

## Installing from source

To install CausalKit from source:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/causalkit.git
```

2. Navigate to the cloned directory:

```bash
cd causalkit
```

3. Install the package in development mode:

```bash
pip install -e .
```

## Verifying the installation

You can verify that CausalKit is installed correctly by importing it in Python:

```python
import causalkit
print(causalkit.__version__)
```

## Installing optional dependencies

For development or running tests, you can install additional dependencies:

```bash
pip install causalkit[dev]
```

This will install additional packages like pytest, flake8, and sphinx for development and testing.