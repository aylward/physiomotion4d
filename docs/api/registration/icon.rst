====================================
Icon Deep Learning Registration
====================================

.. currentmodule:: physiomotion4d

GPU-accelerated deep learning-based deformable registration using Icon algorithm.

Class Reference
===============

.. autoclass:: RegisterImagesICON
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

Icon provides fast, accurate registration using deep learning, ideal for 4D sequences and real-time applications.

**Key Features**:
   * Fast inference (~10 seconds per pair on GPU)
   * Learned from large medical image datasets
   * Excellent for cardiac and respiratory motion
   * GPU-accelerated computation
   * Smooth, diffeomorphic transforms

Usage Examples
==============

Basic Registration
------------------

.. code-block:: python

   from physiomotion4d import RegisterImagesIcon
   
   registrar = RegisterImagesIcon(
       device="cuda:0",
       verbose=True
   )
   
   displacement_field = registrar.register(
       fixed_image="reference.nrrd",
       moving_image="moving.nrrd"
   )

See Also
========

* :doc:`index` - Registration overview
* :doc:`time_series` - 4D sequence registration

.. rubric:: Navigation

:doc:`ants` | :doc:`index` | :doc:`time_series`
