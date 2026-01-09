====================================
Segmentation Modules
====================================

.. currentmodule:: physiomotion4d

AI-powered anatomical structure identification from medical images using state-of-the-art deep learning models.

Overview
========

PhysioMotion4D supports multiple segmentation approaches:

* **TotalSegmentator**: Whole-body CT segmentation (100+ structures)
* **VISTA-3D**: MONAI-based foundation model for medical imaging
* **VISTA-3D NIM**: NVIDIA Inference Microservice version
* **Ensemble**: Combine multiple methods for improved accuracy

All segmentation classes inherit from :class:`SegmentChestBase` and provide consistent interfaces.

Quick Links
===========

**Segmentation Classes**:
   * :doc:`base` - Base class for all segmentation methods
   * :doc:`totalsegmentator` - TotalSegmentator implementation
   * :doc:`vista3d` - VISTA-3D foundation model
   * :doc:`vista3d_nim` - VISTA-3D NIM for cloud deployment
   * :doc:`ensemble` - Ensemble segmentation

Choosing a Method
=================

+------------------+------------------+------------------+------------------+
| Method           | Speed            | Accuracy         | Best For         |
+==================+==================+==================+==================+
| TotalSegmentator | Fast (~30s)      | Good             | General purpose  |
+------------------+------------------+------------------+------------------+
| VISTA-3D         | Medium (~60s)    | Excellent        | Cardiac imaging  |
+------------------+------------------+------------------+------------------+
| VISTA-3D NIM     | Fast (cloud)     | Excellent        | Production       |
+------------------+------------------+------------------+------------------+
| Ensemble         | Slow (~90s)      | Best             | Research/QC      |
+------------------+------------------+------------------+------------------+

Quick Start
===========

Basic Segmentation
------------------

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   
   # Initialize segmentator
   segmentator = SegmentChestTotalSegmentator(fast=True, verbose=True)
   
   # Segment image
   labelmap = segmentator.segment("ct_scan.nrrd")
   
   # Extract specific structure
   heart = segmentator.extract_structure(labelmap, "heart")
   
   # Save results
   segmentator.save_labelmap(labelmap, "output_labels.mha")

With VISTA-3D
-------------

.. code-block:: python

   from physiomotion4d import SegmentChestVista3D
   
   # Initialize with GPU
   segmentator = SegmentChestVista3D(
       device="cuda:0",
       use_auto_prompts=True,
       verbose=True
   )
   
   # Segment specific structures
   labelmap = segmentator.segment(
       image_path="cardiac_ct.nrrd",
       structures=["heart_left_ventricle", "heart_myocardium"]
   )

Ensemble Approach
-----------------

.. code-block:: python

   from physiomotion4d import SegmentChestEnsemble
   
   # Combine multiple methods
   segmentator = SegmentChestEnsemble(
       methods=['totalsegmentator', 'vista3d'],
       fusion_strategy='voting',
       verbose=True
   )
   
   labelmap = segmentator.segment("ct_scan.nrrd")

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   base
   totalsegmentator
   vista3d
   vista3d_nim
   ensemble

Common Operations
=================

Structure Extraction
--------------------

Extract individual anatomical structures from segmentation results:

.. code-block:: python

   # Segment entire image
   labelmap = segmentator.segment("ct.nrrd")
   
   # Extract cardiac structures
   lv = segmentator.extract_structure(labelmap, "heart_left_ventricle")
   rv = segmentator.extract_structure(labelmap, "heart_right_ventricle")
   myocardium = segmentator.extract_structure(labelmap, "heart_myocardium")
   
   # Compute volumes
   lv_volume = segmentator.compute_volume(lv)
   print(f"LV volume: {lv_volume} mm³")

Batch Processing
----------------

Process multiple images efficiently:

.. code-block:: python

   from pathlib import Path
   
   segmentator = SegmentChestTotalSegmentator(fast=True)
   
   for image_file in Path("data").glob("*.nrrd"):
       print(f"Segmenting {image_file}...")
       
       labelmap = segmentator.segment(str(image_file))
       
       output_file = f"{image_file.stem}_labels.mha"
       segmentator.save_labelmap(labelmap, output_file)

Quality Control
---------------

Validate segmentation quality:

.. code-block:: python

   import numpy as np
   
   def validate_segmentation(labelmap, ground_truth):
       """Compute Dice coefficient."""
       intersection = np.logical_and(labelmap, ground_truth).sum()
       dice = 2 * intersection / (labelmap.sum() + ground_truth.sum())
       return dice
   
   # Validate results
   dice_score = validate_segmentation(labelmap, reference_labelmap)
   print(f"Dice score: {dice_score:.3f}")

Best Practices
==============

Method Selection
----------------

* **TotalSegmentator**: Use for general-purpose segmentation, fast iterations
* **VISTA-3D**: Use for cardiac structures, when accuracy is critical
* **Ensemble**: Use when maximum accuracy is needed, can tolerate longer processing

Parameter Tuning
----------------

* Start with default parameters
* Enable ``fast`` mode for quick prototyping
* Use GPU (``device="cuda:0"``) for VISTA-3D when available
* Adjust post-processing based on image quality

Error Handling
--------------

.. code-block:: python

   try:
       labelmap = segmentator.segment(image_path)
   except RuntimeError as e:
       print(f"Segmentation failed: {e}")
       # Fallback to alternative method
       segmentator_backup = SegmentChestTotalSegmentator(fast=True)
       labelmap = segmentator_backup.segment(image_path)

Performance Tips
================

* Use GPU acceleration when available
* Enable fast mode for development
* Process time series in batch
* Cache segmentation results for repeated use

See Also
========

* :doc:`../workflows` - Using segmentation in workflows
* :doc:`../registration/index` - Register segmented images
* :doc:`../usd/index` - Convert segmentations to USD
* :doc:`../../cli_scripts/overview` - Command-line tools

.. rubric:: Navigation

:doc:`../index` | :doc:`base` | :doc:`totalsegmentator`
