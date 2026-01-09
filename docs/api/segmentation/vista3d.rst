====================================
VISTA-3D Foundation Model
====================================

.. currentmodule:: physiomotion4d

MONAI-based foundation model for high-accuracy medical image segmentation.

Class Reference
===============

.. autoclass:: SegmentChestVista3D
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

VISTA-3D is a foundation model architecture that provides state-of-the-art accuracy, especially for cardiac structures. It supports interactive segmentation with point prompts and bounding boxes.

**Key Features**:
   * Foundation model trained on diverse medical datasets
   * Interactive prompting with points and boxes
   * Excellent accuracy on cardiac structures
   * Supports automatic prompt generation
   * Requires GPU for optimal performance

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from physiomotion4d import SegmentChestVista3D
   
   # Initialize with GPU
   segmentator = SegmentChestVista3D(
       device="cuda:0",
       verbose=True
   )
   
   # Segment with automatic prompts
   labelmap = segmentator.segment(
       image_path="cardiac_ct.nrrd",
       structures=["heart_left_ventricle", "heart_myocardium"]
   )

With Manual Prompts
-------------------

.. code-block:: python

   # Segment with point prompts
   labelmap = segmentator.segment(
       image_path="ct.nrrd",
       point_prompts=[(128, 128, 150)],  # (x, y, z) coordinates
       structure_name="heart_left_ventricle"
   )
   
   # Or with bounding box
   labelmap = segmentator.segment(
       image_path="ct.nrrd",
       bbox_prompt=[100, 100, 120, 150, 150, 180],  # xmin,ymin,zmin,xmax,ymax,zmax
       structure_name="heart"
   )

See Also
========

* :doc:`index` - Segmentation overview
* :doc:`vista3d_nim` - Cloud deployment version
* :doc:`totalsegmentator` - Alternative method

.. rubric:: Navigation

:doc:`totalsegmentator` | :doc:`index` | :doc:`vista3d_nim`
