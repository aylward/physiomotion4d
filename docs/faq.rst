===
FAQ
===

Frequently Asked Questions about PhysioMotion4D.

General Questions
=================

What is PhysioMotion4D?
-----------------------

PhysioMotion4D is a medical imaging package that converts 4D CT scans into dynamic 
3D models for visualization in NVIDIA Omniverse.

What data formats are supported?
---------------------------------

* **Input**: NRRD, MHA, NIfTI, DICOM
* **Output**: USD (Universal Scene Description), VTK

Do I need NVIDIA Omniverse?
----------------------------

No, Omniverse is optional for visualization. You can also use:

* usdview (comes with usd-core)
* PyVista
* ParaView

Installation Questions
======================

Do I need a GPU?
----------------

No. A plain ``pip install physiomotion4d`` works without a GPU. At import time
a ``UserWarning`` is emitted (visible by default in all standard Python runs):

.. code-block:: text

   CuPy is not installed — GPU acceleration is unavailable and processing will be
   slow. Re-install with uv to get CuPy and CUDA-enabled PyTorch in one step
   (pip alone will not select the correct CUDA wheel):
     uv pip install 'physiomotion4d[cuda13]'  # CUDA 13
     uv pip install 'physiomotion4d[cuda12]'  # CUDA 12

CPU-only mode is suitable for evaluation and small datasets. For production
workloads an NVIDIA GPU is strongly recommended.

Which CUDA version is required?
--------------------------------

CUDA 13 and CUDA 12 are both supported. Install the extra that matches your
system CUDA version:

.. code-block:: bash

   # CUDA 13 (recommended)
   uv pip install "physiomotion4d[cuda13]"

   # CUDA 12
   uv pip install "physiomotion4d[cuda12]"

Each extra installs both CuPy and a CUDA-built PyTorch wheel in one step —
there is no need to install PyTorch separately. The ``[cuda13]`` extra provides
``cupy-cuda13x>=13.6.0`` and sources PyTorch, torchvision, and torchaudio from
``https://download.pytorch.org/whl/cu130``. The ``[cuda12]`` extra provides
``cupy-cuda12x>=12.0.0`` and sources them from
``https://download.pytorch.org/whl/cu128``. PyTorch is listed in both extras
so that uv's dependency resolver fetches the GPU wheel instead of the CPU wheel
from PyPI.

What Python version is required?
---------------------------------

Python 3.10, 3.11, or 3.12 are supported.

Usage Questions
===============

How long does processing take?
-------------------------------

Typical processing time for 10-frame cardiac CT (with GPU):

* 4D to 3D conversion: ~1 minute
* Registration: ~5-10 minutes
* Segmentation: ~1-2 minutes
* USD creation: ~1 minute
* **Total**: ~10-15 minutes

Which segmentation method should I use?
----------------------------------------

* **TotalSegmentator**: Fast, good quality, general purpose
* **Simpleware**: Best quality for cardiac imaging, requires Simpleware Medical

See :doc:`api/segmentation/index` for comparison.

Which registration method should I use?
----------------------------------------

* **ICON**: Recommended for cardiac/lung (fast, GPU)
* **ANTs**: Best for brain imaging and general purpose

See :doc:`api/registration/index` for comparison.

Troubleshooting
===============

See :doc:`troubleshooting` for common issues and solutions.

More Questions?
===============

* Check the :doc:`cli_scripts/heart_gated_ct`
* Browse :doc:`examples`
* Open an issue on `GitHub <https://github.com/Project-MONAI/physiomotion4d/issues>`_

