CausalKit
=========

CausalKit is a Python package for causal inference that provides tools for designing, implementing, and analyzing causal inference experiments.

Overview
--------

CausalKit simplifies the process of conducting causal inference studies by providing:

- **Data Generation**: Tools for generating synthetic data for A/B tests and randomized controlled trials
- **Experimental Design**: Utilities for splitting traffic and designing experiments
- **Statistical Analysis**: Methods for analyzing experimental results using various statistical approaches

.. toctree::
   :maxdepth: 1
   :caption: Main Sections

   user-guide
   examples
   api

Installation
-----------

.. code-block:: bash

   pip install causalkit

Or clone the repository and install from source:

.. code-block:: bash

   git clone https://github.com/ioannmartynov/causalkit.git
   cd causalkit
   pip install -e .

Quick Start
----------

Here's a simple example of generating A/B test data and analyzing the results:

.. code-block:: python

   import causalkit
   from causalkit.data import generate_ab_test_data
   from causalkit.inference import compare_ab

   # Generate synthetic A/B test data
   df = generate_ab_test_data(
       n_samples={"A": 5000, "B": 5000},
       conversion_rates={"A": 0.10, "B": 0.12}
   )

   # Extract control and treatment data
   control = df[df['group'] == 'A']['conversion'].values
   treatment = df[df['group'] == 'B']['conversion'].values

   # Compare the results
   compare_ab(control, treatment)

Features
-------

Data Generation
~~~~~~~~~~~~~~

- Generate A/B test data with customizable parameters
- Create randomized controlled trial (RCT) data
- Generate observational data for more complex causal inference scenarios

Experimental Design
~~~~~~~~~~~~~~~~~

- Split traffic for experiments with customizable ratios
- Support for stratified splitting to maintain distribution of key variables

Analysis
~~~~~~~

- Two-sample t-tests for comparing control and treatment groups
- OLS regression with treatment dummies
- Advanced methods like Partial Linear Regression (PLR) using DoubleML

License
------

This project is licensed under the terms of the `LICENSE <https://github.com/ioannmartynov/causalkit/blob/main/LICENSE>`_ file.