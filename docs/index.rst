Causalis
=========

Evaluate the impact of the treatment on the outcome metric within the sample population, while controlling for confounding factors, to inform resource allocation decisions.


CausalKit simplifies the process of conducting causal inference studies by providing:

- **RCT Inference**: Estimate effect in your AB test
- **Observational Data Inference**: Estimate effect on observational data

.. toctree::
   :maxdepth: 1
   :caption: Main Sections
   :hidden:


   user-guide
   examples
   research
   api

Installation
-----------
Install from github directly

.. code-block:: bash

   pip install git+https://github.com/ioannmartynov/causalis.git

Then import the library

.. code-block:: python

    import causalis

Scenarios Covered
--------

.. list-table::
   :header-rows: 1

   * - Is RCT
     - Treatment
     - Outcome
     - EDA
     - Estimands
     - Refutation
     - Docs
   * - Observational
     - Binary
     - Continuous
     - Yes
     - ATE
     - Yes
     - Example
   * - Observational
     - Binary
     - Continuous
     - Yes
     - ATT
     - Yes
     - Example

Start using with :doc:`User Guide <user-guide>`
----------------------------------------------

References
-----------
https://github.com/DoubleML/doubleml-for-py

https://github.com/py-why/EconML