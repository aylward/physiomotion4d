============
Installation
============

This guide covers the installation of PhysioMotion4D and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.10, 3.11, or 3.12
* **GPU**: NVIDIA GPU with CUDA 13 (default) or CUDA 12 — recommended for production use; CPU-only installation is supported but will be slow and will emit a runtime warning
* **RAM**: 16GB minimum (32GB+ recommended for large datasets)
* **Storage**: 10GB+ for package and model weights
* **Visualization**: NVIDIA Omniverse (optional, for USD visualization)

Software Dependencies
---------------------

PhysioMotion4D relies on several key packages:

* **Medical Imaging**: ITK, TubeTK, MONAI, nibabel, PyVista
* **AI/ML**: PyTorch, CuPy (CUDA 13 default; CUDA 12 via ``[cuda12]`` extra), transformers, MONAI
* **Registration**: icon-registration, unigradicon
* **Visualization**: USD-core, PyVista
* **Segmentation**: TotalSegmentator

Installation Methods
====================

Method 1: Install from PyPI (Recommended)
------------------------------------------

The simplest way to install PhysioMotion4D is from PyPI.

CPU-only install (evaluation / no GPU):

.. code-block:: bash

   pip install physiomotion4d

This works immediately. CuPy is absent, so a ``UserWarning`` is emitted at
import time (visible by default in all standard Python runs):

.. code-block:: text

   CuPy is not installed — GPU acceleration is unavailable and processing will be
   slow. Re-install with uv to get CuPy and CUDA-enabled PyTorch in one step
   (pip alone will not select the correct CUDA wheel):
     uv pip install 'physiomotion4d[cuda13]'  # CUDA 13
     uv pip install 'physiomotion4d[cuda12]'  # CUDA 12

CUDA 13 install (recommended for production):

.. code-block:: bash

   uv pip install "physiomotion4d[cuda13]"

CUDA 12 install:

.. code-block:: bash

   uv pip install "physiomotion4d[cuda12]"

The ``[cuda13]`` and ``[cuda12]`` extras install both CuPy and the correct
CUDA-built PyTorch wheel in one step. PyTorch is listed inside the extras so
that uv's dependency resolver fetches the GPU wheel from the PyTorch index
rather than the CPU wheel from PyPI. There is no need to install PyTorch
separately.

For development with NVIDIA NIM cloud services:

.. code-block:: bash

   pip install physiomotion4d[nim]

Method 2: Install from Source
------------------------------

For development or to get the latest features:

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/Project-MONAI/physiomotion4d.git
   cd physiomotion4d

**Step 2: Create virtual environment**

.. tabs::

   .. tab:: Linux/macOS

      .. code-block:: bash

         python -m venv venv
         source venv/bin/activate

   .. tab:: Windows

      .. code-block:: bash

         python -m venv venv
         venv\Scripts\activate

**Step 3: Install uv package manager** (optional but recommended)

.. code-block:: bash

   pip install uv

**Step 4: Install PhysioMotion4D**

CPU-only (evaluation / no GPU):

.. code-block:: bash

   uv pip install -e "."

With uv (CUDA 13, recommended for production):

.. code-block:: bash

   uv pip install -e ".[cuda13]"

With uv (CUDA 12):

.. code-block:: bash

   uv pip install -e ".[cuda12]"

Optional Dependencies
=====================

Development Tools
-----------------

To install development dependencies (testing, linting, formatting):

.. code-block:: bash

   pip install physiomotion4d[dev]

This includes:

* **ruff** (fast linting and formatting)
* **mypy** (type checking)
* **pytest, pytest-cov** (testing)
* **pre-commit** (git hooks for automatic checks)

.. note::
   As of 2026, PhysioMotion4D uses Ruff as the primary linter and formatter,
   replacing the previous black, isort, flake8, and pylint tools for improved
   speed and simplicity.

Documentation Tools
-------------------

To build documentation locally:

.. code-block:: bash

   pip install physiomotion4d[docs]

Testing Dependencies
--------------------

To run tests:

.. code-block:: bash

   pip install physiomotion4d[test]

Verify Installation
===================

After installation, verify that PhysioMotion4D is correctly installed:

.. code-block:: python

   import physiomotion4d
   from physiomotion4d import ProcessHeartGatedCT
   
   print(f"PhysioMotion4D version: {physiomotion4d.__version__}")

Expected output:

.. code-block:: text

   PhysioMotion4D version: 2025.05.0

Command-Line Tools
==================

PhysioMotion4D provides command-line interfaces that should be available after installation:

.. code-block:: bash

   # Check CLI is available
   physiomotion4d --help
   physiomotion4d-heart-gated-ct --help

GPU Setup
=========

CUDA Installation
-----------------

An NVIDIA GPU is strongly recommended. Two CUDA versions are supported via
optional extras:

* **CUDA 13** — installed when you use the ``[cuda13]`` extra (recommended)
* **CUDA 12** — installed when you use the ``[cuda12]`` extra

A plain ``pip install physiomotion4d`` installs a CPU-only build. It runs
without error but emits a ``UserWarning`` at import time and will be
significantly slower than a GPU-enabled install.

If CUDA is not yet installed, download the CUDA Toolkit from
`NVIDIA's website <https://developer.nvidia.com/cuda-downloads>`_, then verify:

.. code-block:: bash

   nvcc --version
   nvidia-smi

PyTorch with CUDA
-----------------

A plain install pulls a CPU-only PyTorch wheel from PyPI. The ``[cuda13]``
extra sources PyTorch, torchvision, and torchaudio from the
``https://download.pytorch.org/whl/cu130`` index. The ``[cuda12]`` extra
sources them from ``https://download.pytorch.org/whl/cu128``. To verify the
active version:

.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")

Troubleshooting
===============

Common Issues
-------------

**Issue: CUDA out of memory**

Solution: Reduce batch sizes or process smaller images. Most PhysioMotion4D functions work with limited GPU memory.

**Issue: Import errors for ITK or VTK**

Solution: These packages sometimes require system dependencies. On Ubuntu:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0

**Issue: TotalSegmentator download fails**

Solution: TotalSegmentator downloads models on first use. Ensure you have:

* Stable internet connection
* Sufficient disk space (~2GB for models)
* Write permissions in the cache directory

**Issue: USD files not rendering in Omniverse**

Solution:

1. Ensure NVIDIA Omniverse is installed
2. Check USD file integrity with ``usdview`` (included with usd-core)
3. Verify file paths are accessible to Omniverse

Getting Help
------------

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Search `GitHub Issues <https://github.com/Project-MONAI/physiomotion4d/issues>`_
3. Open a new issue with:

   * Python version
   * CUDA version
   * Error messages
   * Minimal code to reproduce

Next Steps
==========

* Continue to :doc:`quickstart` for your first PhysioMotion4D workflow
* Explore :doc:`examples` for common use cases
* Read :doc:`cli_scripts/overview` for detailed command-line workflows

