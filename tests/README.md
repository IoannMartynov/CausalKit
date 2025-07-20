# CausalKit Tests

This directory contains tests for the CausalKit package. The tests are written using pytest and can be run individually or as a suite.

## Prerequisites

Before running the tests, make sure you have pytest installed:

```bash
pip install pytest
```

Alternatively, you can install all development dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

### Running all tests

To run all tests:

```bash
python -m pytest
```

### Running specific tests

To run a specific test file:

```bash
python -m pytest tests/test_file_name.py
```

For example:

```bash
python -m pytest tests/test_data_generators.py
```

To run a specific test function:

```bash
python -m pytest tests/test_file_name.py::test_function_name
```

For example:

```bash
python -m pytest tests/test_data_generators.py::test_generate_rct_data_default_parameters
```

### Verbose output

For more detailed output, use the `-v` flag:

```bash
python -m pytest -v
```

### Skipping documentation build tests

The documentation build test can be skipped by setting the `SKIP_DOCS_BUILD` environment variable:

```bash
SKIP_DOCS_BUILD=true python -m pytest
```

## Test Files

- `test_causaldata_get_df.py`: Tests for the get_df method in causaldata class
- `test_ckit.py`: Tests for the causaldata class
- `test_data_generators.py`: Tests for the data generator functions
- `test_docs_build.py`: Tests for building documentation
- `test_installation.py`: Tests for package installation and imports
- `test_mde.py`: Tests for the calculate_mde function
- `test_ttest.py`: Tests for the ttest function

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file or add to an existing one based on the functionality being tested
2. Use pytest fixtures for common setup
3. Write clear test functions with descriptive names
4. Add docstrings to explain the purpose of each test
5. Use assertions to verify expected behavior