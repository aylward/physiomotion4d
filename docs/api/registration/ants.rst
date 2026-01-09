====================================
ANTs Registration
====================================

.. currentmodule:: physiomotion4d

Traditional optimization-based image registration using ANTs (Advanced Normalization Tools).

Class Reference
===============

.. autoclass:: RegisterImagesANTs
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

ANTs provides robust, accurate registration using iterative optimization. Best for high-precision registration where processing time is less critical.

**Key Features**:
   * Excellent accuracy with traditional methods
   * Multi-resolution pyramid optimization
   * Multiple similarity metrics (MI, CC, MSQ)
   * Rigid, affine, and deformable registration
   * Well-established and validated

Usage Examples
==============

Basic Registration
------------------

.. code-block:: python

   from physiomotion4d import RegisterImagesANTs
   
   registrar = RegisterImagesANTs(verbose=True)
   
   transform = registrar.register(
       fixed_image="reference.nrrd",
       moving_image="moving.nrrd",
       transform_type="SyN"  # Symmetric normalization
   )

See Also
========

* :doc:`index` - Registration overview
* :doc:`icon` - Faster deep learning alternative

.. rubric:: Navigation

:doc:`base` | :doc:`index` | :doc:`icon`
