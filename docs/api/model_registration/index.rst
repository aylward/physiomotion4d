====================================
Model Registration Modules
====================================

.. currentmodule:: physiomotion4d

Register 3D anatomical models (meshes, point clouds) to each other or to images.

Overview
========

Model registration methods for aligning anatomical models:

* **ICP**: Iterative Closest Point (pure Python)
* **ICP-ITK**: ICP using ITK backend
* **Distance Maps**: Distance field-based registration
* **PCA**: Principal component analysis-based alignment

Quick Links
===========

**Registration Classes**:
   * :doc:`icp` - Iterative Closest Point (Python)
   * :doc:`icp_itk` - ICP with ITK
   * :doc:`distance_maps` - Distance map-based
   * :doc:`pca` - PCA-based registration

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   icp
   icp_itk
   distance_maps
   pca

Quick Start
===========

ICP Registration
----------------

.. code-block:: python

   from physiomotion4d import RegisterModelsICP
   
   registrar = RegisterModelsICP(verbose=True)
   
   transform = registrar.register(
       fixed_model="reference_heart.vtk",
       moving_model="template_heart.vtk"
   )

See Also
========

* :doc:`../registration/index` - Image registration
* :doc:`../workflows` - Model-to-patient workflows

.. rubric:: Navigation

:doc:`../registration/time_series` | :doc:`../index` | :doc:`icp`
