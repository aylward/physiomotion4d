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
* **VISTA-3D**: MONAI-based foundation model for medical imaging
* **VISTA-3D NIM**: NVIDIA Inference Microservice version
* **Ensemble**: Combine multiple methods for improved accuracy

All segmentation classes inherit from :class:`SegmentChestBase` and provide consistent interfaces.

Base Segmentation Class
=======================

SegmentChestBase
----------------

Abstract base class for all segmentation methods.

.. autoclass:: physiomotion4d.SegmentChestBase
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

VISTA-3D
--------

MONAI-based foundation model for medical image segmentation.

.. autoclass:: physiomotion4d.SegmentChestVista3D
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Foundation model architecture
   * Supports point prompts and bounding boxes
   * High accuracy on cardiac structures
   * Requires GPU

**Example Usage**:

.. code-block:: python

   from physiomotion4d import SegmentChestVista3D
   
   # Initialize with GPU
   segmentator = SegmentChestVista3D(
       device="cuda:0",
       use_auto_prompts=True,
       verbose=True
   )
   
   # Segment with automatic prompts
   labelmap = segmentator.segment(
       image_path="cardiac_ct.nrrd",
       structures=["heart_left_ventricle", "heart_myocardium"]
   )
   
   # Or provide manual prompts
   labelmap = segmentator.segment(
       image_path="cardiac_ct.nrrd",
       point_prompts=[(128, 128, 150)],  # (x, y, z)
       structure_name="heart_left_ventricle"
   )

VISTA-3D NIM
------------

NVIDIA Inference Microservice version for cloud deployment.

.. autoclass:: physiomotion4d.SegmentChestVista3DNIM
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Optimized inference
   * Cloud/server deployment
   * REST API interface
   * Scalable for multiple requests

**Example Usage**:

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

Ensemble Segmentation
---------------------

Combines multiple segmentation methods for improved accuracy.

.. autoclass:: physiomotion4d.SegmentChestEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Combines TotalSegmentator + VISTA-3D
   * Voting or averaging strategies
   * Improved robustness
   * Better boundary delineation

**Example Usage**:

.. code-block:: python

   from physiomotion4d import SegmentChestEnsemble
   
   # Initialize ensemble
   segmentator = SegmentChestEnsemble(
       methods=['totalsegmentator', 'vista3d'],
       fusion_strategy='voting',  # or 'averaging'
       verbose=True
   )
   
   # Segment with ensemble
   labelmap = segmentator.segment(image_path="ct_scan.nrrd")
   
   # Get confidence maps
   confidence = segmentator.get_confidence_map()

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

   from physiomotion4d import SegmentChestBase
   import numpy as np
   
   class CustomSegmentator(SegmentChestBase):
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
   segmentator = SegmentChestVista3D(device=device)

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

Confidence Assessment
---------------------

Assess segmentation confidence:

.. code-block:: python

   # For VISTA-3D (supports confidence)
   segmentator = SegmentChestVista3D()
   
   labelmap, confidence_map = segmentator.segment_with_confidence("ct.nrrd")
   
   # Check low-confidence regions
   low_confidence_mask = confidence_map < 0.5
   print(f"Low confidence regions: {low_confidence_mask.sum()} voxels")

Best Practices
==============

Method Selection
----------------

* **TotalSegmentator**: General purpose, fast, comprehensive
* **VISTA-3D**: High accuracy, especially for cardiac structures
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
