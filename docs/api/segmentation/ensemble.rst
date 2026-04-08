====================================
Ensemble Segmentation
====================================

.. currentmodule:: physiomotion4d

Ensemble segmentation provides a stable API entry point that currently delegates
to :class:`SegmentChestTotalSegmentator`.

Class Reference
===============

.. autoclass:: SegmentChestEnsemble
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

``SegmentChestEnsemble`` inherits from :class:`SegmentChestTotalSegmentator` and
exposes the same interface.  Using this class keeps downstream code stable if the
ensemble strategy changes in a future release.

Usage Example
=============

.. code-block:: python

   from physiomotion4d import SegmentChestEnsemble

   segmenter = SegmentChestEnsemble()
   result = segmenter.segment(ct_image, contrast_enhanced_study=False)
   labelmap = result['labelmap']

See Also
========

* :doc:`index` - Segmentation overview
* :doc:`totalsegmentator` - Underlying segmentation method

.. rubric:: Navigation

:doc:`totalsegmentator` | :doc:`index` | :doc:`../registration/index`
