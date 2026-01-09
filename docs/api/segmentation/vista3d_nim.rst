==========================================
VISTA-3D NIM (Inference Microservice)
==========================================

.. currentmodule:: physiomotion4d

NVIDIA Inference Microservice version of VISTA-3D for cloud deployment.

Class Reference
===============

.. autoclass:: SegmentChestVista3DNIM
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

VISTA-3D NIM provides optimized inference through a REST API, ideal for production deployments and scalable cloud applications.

**Key Features**:
   * Optimized for high-throughput inference
   * REST API interface
   * Cloud/server deployment
   * Scalable for multiple concurrent requests

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from physiomotion4d import SegmentChestVista3DNIM
   
   # Initialize with API endpoint
   segmentator = SegmentChestVista3DNIM(
       api_endpoint="https://api.nvidia.com/nim/vista3d",
       api_key="your_api_key",
       verbose=True
   )
   
   # Segment via API
   labelmap = segmentator.segment(
       image_path="ct_scan.nrrd",
       structures=["heart", "lungs"]
   )

See Also
========

* :doc:`vista3d` - Local deployment version
* :doc:`index` - Segmentation overview

.. rubric:: Navigation

:doc:`vista3d` | :doc:`index` | :doc:`ensemble`
