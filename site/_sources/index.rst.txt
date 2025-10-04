CausalKit
=========

Evaluate the impact of the treatment on the target metric within the sample population, while controlling for confounding factors, to inform resource allocation decisions.


Scenarios Covered
--------

CausalKit simplifies the process of conducting causal inference studies by providing:

- **RCT Inference**: Estimate effect in your AB test
- **Observational Data Inference**: Estimate effect on observational data

.. toctree::
   :maxdepth: 1
   :caption: Main Sections
   :hidden:


   user-guide
   examples
   api

Installation
-----------
Install from github directly

.. code-block:: bash

   pip install git+https://github.com/ioannmartynov/causalkit.git

Then import the library

.. code-block:: python

    import causalkit

Start using with :doc:`User Guide <user-guide>`
----------------------------------------------

References
-----------
https://github.com/DoubleML/doubleml-for-py