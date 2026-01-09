============
Installation
============

This guide covers the installation of PhysioMotion4D and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.10, 3.11, or 3.12
* **GPU**: NVIDIA GPU with CUDA 12.6+ (for AI models and registration)
* **RAM**: 16GB minimum (32GB+ recommended for large datasets)
* **Storage**: 10GB+ for package and model weights
* **Visualization**: NVIDIA Omniverse (optional, for USD visualization)

Software Dependencies
---------------------

PhysioMotion4D relies on several key packages:

* **Medical Imaging**: ITK, TubeTK, MONAI, nibabel, PyVista
* **AI/ML**: PyTorch (CUDA 12.6), transformers, MONAI
* **Registration**: icon-registration, unigradicon
* **Visualization**: USD-core, PyVista
* **Segmentation**: TotalSegmentator, VISTA-3D models

Installation Methods
====================

Method 1: Install from PyPI (Recommended)
------------------------------------------

The simplest way to install PhysioMotion4D is from PyPI:

.. code-block:: bash

   pip install physiomotion4d

For development with NVIDIA NIM cloud services:

.. code-block:: bash

   pip install physiomotion4d[nim]

Method 2: Install from Source
------------------------------

For development or to get the latest features:

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/aylward/PhysioMotion4d.git
   cd PhysioMotion4D

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

With uv:

.. code-block:: bash

   uv pip install -e .

Or with pip:

.. code-block:: bash

   pip install -e .

Optional Dependencies
=====================

Development Tools
-----------------

To install development dependencies (testing, linting, formatting):

.. code-block:: bash

   pip install physiomotion4d[dev]

This includes:

* black (code formatting)
* isort (import sorting)
* flake8, pylint (linting)
* pytest, pytest-cov (testing)
* mypy (type checking)

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

PhysioMotion4D requires CUDA 12.6+ for GPU acceleration. If you don't have CUDA installed:

1. Download and install CUDA Toolkit from `NVIDIA's website <https://developer.nvidia.com/cuda-downloads>`_
2. Verify CUDA installation:

.. code-block:: bash

   nvcc --version
   nvidia-smi

PyTorch with CUDA
-----------------

The package automatically installs PyTorch with CUDA 12.6 support. To verify:

.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")

Expected output:

.. code-block:: text

   PyTorch version: 2.x.x+cu126
   CUDA available: True
   CUDA version: 12.6

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
2. Search `GitHub Issues <https://github.com/aylward/PhysioMotion4d/issues>`_
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

