====================================
Ensemble Segmentation
====================================

.. currentmodule:: physiomotion4d

Combine multiple segmentation methods for improved accuracy and robustness.

Class Reference
===============

.. autoclass:: SegmentChestEnsemble
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

Ensemble segmentation combines predictions from multiple methods using voting or averaging strategies to achieve higher accuracy than any single method.

**Key Features**:
   * Combines TotalSegmentator and VISTA-3D
   * Voting or averaging fusion strategies
   * Improved boundary delineation
   * Higher confidence in predictions
   * Better robustness to image variations

Usage Examples
==============

Basic Ensemble
--------------

.. code-block:: python

   from physiomotion4d import SegmentChestEnsemble
   
   # Combine methods with voting
   segmentator = SegmentChestEnsemble(
       methods=['totalsegmentator', 'vista3d'],
       fusion_strategy='voting',
       verbose=True
   )
   
   labelmap = segmentator.segment("ct_scan.nrrd")
   
   # Get confidence map
   confidence = segmentator.get_confidence_map()

With Weighted Fusion
--------------------

.. code-block:: python

   # Weight methods differently
   segmentator = SegmentChestEnsemble(
       methods=['totalsegmentator', 'vista3d'],
       fusion_strategy='weighted',
       weights=[0.4, 0.6],  # Trust VISTA-3D more
       verbose=True
   )
   
   labelmap = segmentator.segment("cardiac_ct.nrrd")

See Also
========

* :doc:`index` - Segmentation overview
* :doc:`totalsegmentator` - Fast baseline method
* :doc:`vista3d` - High-accuracy method

.. rubric:: Navigation

:doc:`vista3d_nim` | :doc:`index` | :doc:`../registration/index`
