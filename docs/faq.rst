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

* **Recommended**: NVIDIA GPU with CUDA 12.6+ for fast processing
* **Optional**: CPU-only mode available (slower)

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

* **TotalSegmentator**: Fast, good quality
* **VISTA-3D**: Best quality, requires more GPU memory
* **Ensemble**: Best quality, slowest

See :doc:`api/segmentation` for comparison.

Which registration method should I use?
----------------------------------------

* **ICON**: Recommended for cardiac/lung (fast, GPU)
* **ANTs**: Best for brain imaging and general purpose

See :doc:`api/registration` for comparison.

Troubleshooting
===============

See :doc:`troubleshooting` for common issues and solutions.

More Questions?
===============

* Check the :doc:`user_guide/heart_gated_ct`
* Browse :doc:`examples`
* Open an issue on `GitHub <https://github.com/NVIDIA/PhysioMotion4D/issues>`_

