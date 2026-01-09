====================================
Image Registration Modules
====================================

.. currentmodule:: physiomotion4d

Align medical images using traditional and deep learning-based registration methods.

Overview
========

PhysioMotion4D provides multiple image registration approaches:

* **ANTs**: Traditional optimization-based registration
* **Icon**: Deep learning-based deformable registration
* **Time Series**: 4D image sequence registration

All registration classes inherit from :class:`RegisterImagesBase` for consistent interfaces.

Quick Links
===========

**Registration Classes**:
   * :doc:`base` - Base class for all registration methods
   * :doc:`ants` - ANTs registration (traditional)
   * :doc:`icon` - Icon deep learning registration
   * :doc:`time_series` - 4D time series registration

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   base
   ants
   icon
   time_series

Choosing a Method
=================

+------------------+------------------+------------------+------------------+
| Method           | Speed            | Accuracy         | Best For         |
+==================+==================+==================+==================+
| ANTs             | Slow (~5 min)    | Excellent        | High precision   |
+------------------+------------------+------------------+------------------+
| Icon             | Fast (~10 sec)   | Very Good        | 4D sequences     |
+------------------+------------------+------------------+------------------+
| Time Series      | Medium           | Excellent        | Cardiac/lung 4D  |
+------------------+------------------+------------------+------------------+

Quick Start
===========

ANTs Registration
-----------------

.. code-block:: python

   from physiomotion4d import RegisterImagesANTs
   
   registrar = RegisterImagesANTs(verbose=True)
   
   transform = registrar.register(
       fixed_image="reference.nrrd",
       moving_image="moving.nrrd"
   )
   
   # Apply transform
   registered = registrar.apply_transform(
       moving_image="moving.nrrd",
       transform=transform
   )

Icon Registration
-----------------

.. code-block:: python

   from physiomotion4d import RegisterImagesICON
   
   registrar = RegisterImagesICON(
       device="cuda:0",
       verbose=True
   )
   
   displacement_field = registrar.register(
       fixed_image="reference.nrrd",
       moving_image="moving.nrrd"
   )

Time Series Registration
------------------------

.. code-block:: python

   from physiomotion4d import RegisterTimeSeriesImages
   
   registrar = RegisterTimeSeriesImages(
       method="icon",
       device="cuda:0",
       verbose=True
   )
   
   transforms = registrar.register_sequence(
       image_files=["phase_00.nrrd", "phase_01.nrrd", "phase_02.nrrd"]
   )

See Also
========

* :doc:`../segmentation/index` - Segment images before registration
* :doc:`../model_registration/index` - Register 3D models
* :doc:`../workflows` - Complete registration workflows

.. rubric:: Navigation

:doc:`../segmentation/ensemble` | :doc:`../index` | :doc:`base`
