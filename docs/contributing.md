# Contributing to CausalKit

Thank you for your interest in contributing to CausalKit! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to CausalKit. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to CausalKit:

1. **Report bugs**: If you find a bug, please create an issue on GitHub with a detailed description of the bug, steps to reproduce it, and your environment.

2. **Suggest features**: If you have an idea for a new feature or improvement, please create an issue on GitHub describing your suggestion.

3. **Contribute code**: If you'd like to contribute code, please follow the steps below.

## Contributing Code

### Setting up the Development Environment

1. Fork the repository on GitHub.

2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/causalkit.git
   cd causalkit
   ```

3. Create a virtual environment and install the development dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate causalkit
   pip install -e ".[dev]"
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests if applicable.

3. Run the tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. Format your code using the project's style guidelines:
   ```bash
   black causalkit
   isort causalkit
   ```

5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub.

### Pull Request Guidelines

- Provide a clear and descriptive title for your pull request.
- Include a detailed description of the changes you've made.
- Reference any related issues using the GitHub issue number (e.g., "Fixes #123").
- Make sure all tests pass.
- Update documentation if necessary.

## Documentation

If you're adding new features or changing existing functionality, please update the documentation accordingly. CausalKit uses MkDocs for documentation:

1. Make changes to the documentation in the `docs/` directory.

2. Preview your changes locally:
   ```bash
   mkdocs serve
   ```

3. Include documentation updates in your pull request.

## Testing

CausalKit uses pytest for testing. Please add tests for any new functionality or bug fixes:

1. Write tests in the `causalkit/tests/` directory.

2. Run the tests to ensure they pass:
   ```bash
   pytest
   ```

## Code Style

CausalKit follows the PEP 8 style guide for Python code. We use the following tools to enforce code style:

- Black for code formatting
- isort for import sorting
- flake8 for linting

Please ensure your code adheres to these standards before submitting a pull request.

## License

By contributing to CausalKit, you agree that your contributions will be licensed under the project's license.

## Questions

If you have any questions about contributing, please create an issue on GitHub or reach out to the maintainers.

Thank you for contributing to CausalKit!