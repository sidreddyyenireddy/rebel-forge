Installation
============

To install *skala*, you can use pip:

.. code-block:: bash

   pip install microsoft-skala

This will install the *skala* packages and all its dependencies, including

* `torch <https://pytorch.org>`__,
  `e3nn <https://e3nn.org>`__, and
  `opt_einsum_fx <https://opt-einsum-fx.readthedocs.io>`__
  for running the exchange-correlation model
* `pyscf <https://pyscf.org>`__
  for running the self-consistent field (SCF) calculations and evaluating the density features
* `dftd3 <https://dftd3.readthedocs.io>`__
  for computing the D3 dispersion correction to the total energy

The default Pytorch installation is the GPU version, which the *skala* package in combination with PySCF doesn't leverage. To install only the much smaller CPU version of Pytorch, run the following before installing the *skala* package:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

Installing from source
----------------------

If you prefer to install Skala from the source code, you can clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/microsoft/skala
   cd skala
   mamba env create -n skala -f environment.yml
   mamba activate skala
   pip install -e .

To install the development dependencies, you can run:

.. code-block:: bash

    pip install -e .[dev]

For development purposes, please initialize the pre-commit hooks via:

.. code-block:: bash

   pre-commit install

To test your installation, you can run the tests:

.. code-block:: bash

   pytest -v tests/


Model checkpoints
-----------------

The pre-trained Skala model checkpoints are hosted [on Hugging Face](https://huggingface.co/microsoft/skala) and downloaded automatically by the Python package in this repository from there for running calculations.
