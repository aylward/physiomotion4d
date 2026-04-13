====================================
Segmentation Development Guide
====================================

This guide covers developing with and extending PhysioMotion4D's segmentation capabilities.

For complete API documentation, see :doc:`../api/segmentation/index`.

Overview
========

The segmentation module provides AI-powered anatomical structure identification from medical images using state-of-the-art deep learning models.

Overview
========

PhysioMotion4D supports multiple segmentation approaches:

* **TotalSegmentator**: Whole-body CT segmentation (100+ structures)
* **Simpleware**: High-quality cardiac segmentation via Synopsys Simpleware Medical

All segmentation classes inherit from :class:`SegmentAnatomyBase` and provide consistent interfaces.

Base Segmentation Class
=======================

SegmentAnatomyBase
----------------

Abstract base class for all segmentation methods.

.. autoclass:: physiomotion4d.SegmentAnatomyBase
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods**:
   * ``segment(image)``: Main segmentation method
   * ``post_process(labelmap)``: Refine segmentation
   * ``get_label_names()``: Get structure names
   * ``extract_structure(labelmap, label_id)``: Extract single structure

Segmentation Methods
====================

TotalSegmentator
----------------

Uses the TotalSegmentator model for comprehensive anatomical segmentation.

.. autoclass:: physiomotion4d.SegmentChestTotalSegmentator
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * 100+ anatomical structures
   * Fast inference (~30 seconds per CT)
   * CPU and GPU support
   * Pre-trained on large dataset

**Example Usage**:

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   
   # Initialize segmentator
   segmentator = SegmentChestTotalSegmentator(
       fast=True,  # Fast mode for speed
       verbose=True
   )
   
   # Segment image
   labelmap = segmentator.segment(image_path="ct_scan.nrrd")
   
   # Get structure names
   label_names = segmentator.get_label_names()
   print(f"Segmented structures: {label_names}")
   
   # Extract specific structure
   heart = segmentator.extract_structure(labelmap, "heart")

**Segmented Structures** (partial list):
   * Heart chambers (LV, RV, LA, RA)
   * Myocardium
   * Aorta, pulmonary artery, vena cava
   * Lungs (left, right, lobes)
   * Airways (trachea, bronchi)
   * Bones (ribs, spine, sternum)
   * Liver, kidneys, spleen
   * And many more...

Common Usage Patterns
=====================

Basic Segmentation
------------------

Simple segmentation workflow:

.. code-block:: python

   # Choose and initialize segmentator
   from physiomotion4d import SegmentChestTotalSegmentator
   
   segmentator = SegmentChestTotalSegmentator()
   
   # Segment image
   labelmap = segmentator.segment("input.nrrd")
   
   # Save results
   segmentator.save_labelmap(labelmap, "output_labels.mha")

Contrast-Enhanced vs Non-Contrast
----------------------------------

Handle different imaging protocols:

.. code-block:: python

   # For contrast-enhanced CT
   segmentator = SegmentChestTotalSegmentator(
       contrast_enhanced=True,
       optimize_for_vessels=True
   )
   
   # For non-contrast CT
   segmentator = SegmentChestTotalSegmentator(
       contrast_enhanced=False,
       optimize_for_soft_tissue=True
   )

Structure Extraction
--------------------

Extract and process individual structures:

.. code-block:: python

   # Segment entire image
   labelmap = segmentator.segment("ct.nrrd")
   
   # Extract specific structures
   heart_lv = segmentator.extract_structure(labelmap, "heart_left_ventricle")
   heart_rv = segmentator.extract_structure(labelmap, "heart_right_ventricle")
   aorta = segmentator.extract_structure(labelmap, "aorta")
   
   # Process extracted structures
   lv_volume = segmentator.compute_volume(heart_lv)
   print(f"LV volume: {lv_volume} mm³")

Customization
=============

Custom Post-Processing
----------------------

Add custom post-processing steps:

.. code-block:: python

   from physiomotion4d import SegmentAnatomyBase
   import numpy as np
   
   class CustomSegmentator(SegmentAnatomyBase):
       """Custom segmentator with post-processing."""
       
       def post_process(self, labelmap):
           """Custom post-processing."""
           # Call parent post-processing
           labelmap = super().post_process(labelmap)
           
           # Add custom processing
           labelmap = self.fill_holes(labelmap)
           labelmap = self.smooth_boundaries(labelmap)
           
           return labelmap
       
       def fill_holes(self, labelmap):
           """Fill holes in segmentation."""
           # Implement hole filling
           return labelmap
       
       def smooth_boundaries(self, labelmap):
           """Smooth structure boundaries."""
           # Implement smoothing
           return labelmap

Custom Structure Selection
--------------------------

Segment only specific structures:

.. code-block:: python

   class CardiacOnlySegmentator(SegmentChestTotalSegmentator):
       """Segment cardiac structures only."""
       
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           
           # Define cardiac structures
           self.cardiac_structures = [
               'heart_left_ventricle',
               'heart_right_ventricle',
               'heart_left_atrium',
               'heart_right_atrium',
               'heart_myocardium',
               'aorta',
               'pulmonary_artery'
           ]
       
       def segment(self, image_path):
           """Segment cardiac structures only."""
           # Get full segmentation
           full_labelmap = super().segment(image_path)
           
           # Filter for cardiac structures
           cardiac_labelmap = self.filter_structures(
               full_labelmap,
               self.cardiac_structures
           )
           
           return cardiac_labelmap

Performance Optimization
========================

GPU Acceleration
----------------

Leverage GPU for faster inference:

.. code-block:: python

   import torch
   
   # Check GPU availability
   if torch.cuda.is_available():
       device = "cuda:0"
       print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   else:
       device = "cpu"
       print("Using CPU")
   
   # Initialize with GPU
   segmentator = SegmentChestTotalSegmentator()

Batch Processing
----------------

Process multiple images efficiently:

.. code-block:: python

   def batch_segment(image_files, output_dir):
       """Batch segmentation of multiple images."""
       from pathlib import Path
       
       segmentator = SegmentChestTotalSegmentator(fast=True)
       
       for image_file in image_files:
           print(f"Segmenting {image_file}...")
           
           # Segment
           labelmap = segmentator.segment(image_file)
           
           # Save result
           output_file = Path(output_dir) / f"{Path(image_file).stem}_labels.mha"
           segmentator.save_labelmap(labelmap, str(output_file))

Quality Control
===============

Validation Metrics
------------------

Assess segmentation quality:

.. code-block:: python

   def validate_segmentation(labelmap, ground_truth):
       """Compute validation metrics."""
       from scipy import ndimage
       
       # Dice coefficient
       intersection = np.logical_and(labelmap, ground_truth).sum()
       dice = 2 * intersection / (labelmap.sum() + ground_truth.sum())
       
       # Hausdorff distance
       hausdorff = compute_hausdorff(labelmap, ground_truth)
       
       # Volume difference
       vol_diff = abs(labelmap.sum() - ground_truth.sum()) / ground_truth.sum()
       
       return {
           'dice': dice,
           'hausdorff': hausdorff,
           'volume_difference': vol_diff
       }

Best Practices
==============

Method Selection
----------------

* **TotalSegmentator**: General purpose, fast, comprehensive
* **Ensemble**: When accuracy is critical, can tolerate longer processing

Parameter Tuning
----------------

* Start with default parameters
* Enable ``fast`` mode for quick iterations
* Use GPU when available for speed
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

See Also
========

* :doc:`../api/segmentation/index` - Complete segmentation API
* :doc:`../api/workflows` - Using segmentation in workflows
* :doc:`registration_images` - Registering segmented images
* :doc:`usd_generation` - Converting segmentations to USD
