====================================
TotalSegmentator
====================================

.. currentmodule:: physiomotion4d

Comprehensive anatomical segmentation using the TotalSegmentator deep learning model.

Class Reference
===============

.. autoclass:: SegmentChestTotalSegmentator
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

:class:`SegmentChestTotalSegmentator` provides fast, accurate segmentation of 100+ anatomical structures from CT scans using the TotalSegmentator model.

**Key Features**:
   * 100+ anatomical structures in whole body
   * Fast inference (~30 seconds per CT)
   * CPU and GPU support
   * Pre-trained on large diverse dataset
   * Excellent generalization to various scan protocols

Segmented Structures
====================

Cardiac Structures
------------------
* Heart chambers: left ventricle, right ventricle, left atrium, right atrium
* Myocardium
* Aorta (ascending, descending, arch)
* Pulmonary artery and veins
* Vena cava (superior, inferior)
* Coronary arteries (with contrast)

Pulmonary Structures
--------------------
* Lungs (left, right, upper/middle/lower lobes)
* Airways (trachea, main bronchi)
* Pulmonary vessels

Thoracic Structures
-------------------
* Ribs, sternum, clavicles
* Spine (vertebrae, discs)
* Esophagus
* Thyroid

Abdominal Organs
----------------
* Liver, spleen, pancreas
* Kidneys, adrenal glands
* Gallbladder
* Stomach, intestines

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from physiomotion4d import SegmentChestTotalSegmentator
   
   # Initialize with default settings
   segmentator = SegmentChestTotalSegmentator(verbose=True)
   
   # Segment image
   labelmap = segmentator.segment("ct_scan.nrrd")
   
   # Get available structures
   structures = segmentator.get_label_names()
   print(f"Available structures: {len(structures)}")
   
   # Save results
   segmentator.save_labelmap(labelmap, "segmentation.mha")

Fast Mode
---------

For quick iterations and prototyping:

.. code-block:: python

   # Enable fast mode for ~2x speedup
   segmentator = SegmentChestTotalSegmentator(
       fast=True,
       verbose=True
   )
   
   labelmap = segmentator.segment("ct_scan.nrrd")

Cardiac-Focused Segmentation
-----------------------------

For cardiac imaging workflows:

.. code-block:: python

   # Optimize for cardiac structures
   segmentator = SegmentChestTotalSegmentator(
       contrast_enhanced=True,  # For contrast CT
       optimize_for_cardiac=True,
       verbose=True
   )
   
   labelmap = segmentator.segment("cardiac_ct.nrrd")
   
   # Extract cardiac structures
   lv = segmentator.extract_structure(labelmap, "heart_left_ventricle")
   rv = segmentator.extract_structure(labelmap, "heart_right_ventricle")
   myocardium = segmentator.extract_structure(labelmap, "heart_myocardium")
   aorta = segmentator.extract_structure(labelmap, "aorta")

Structure Extraction
--------------------

Extract and analyze specific structures:

.. code-block:: python

   # Segment entire scan
   labelmap = segmentator.segment("ct.nrrd")
   
   # Extract lungs
   left_lung = segmentator.extract_structure(labelmap, "lung_left")
   right_lung = segmentator.extract_structure(labelmap, "lung_right")
   
   # Compute volumes
   left_volume = segmentator.compute_volume(left_lung)
   right_volume = segmentator.compute_volume(right_lung)
   
   print(f"Left lung: {left_volume:.0f} mm³")
   print(f"Right lung: {right_volume:.0f} mm³")
   print(f"Total: {left_volume + right_volume:.0f} mm³")

Batch Processing
----------------

Process multiple images efficiently:

.. code-block:: python

   from pathlib import Path
   
   # Initialize once
   segmentator = SegmentChestTotalSegmentator(fast=True, verbose=True)
   
   # Process all images in directory
   image_dir = Path("data/ct_scans")
   output_dir = Path("results/segmentations")
   output_dir.mkdir(exist_ok=True)
   
   for image_file in image_dir.glob("*.nrrd"):
       print(f"Processing {image_file.name}...")
       
       # Segment
       labelmap = segmentator.segment(str(image_file))
       
       # Save with descriptive name
       output_file = output_dir / f"{image_file.stem}_segmentation.mha"
       segmentator.save_labelmap(labelmap, str(output_file))

Advanced Usage
==============

GPU Acceleration
----------------

Use GPU for faster processing:

.. code-block:: python

   import torch
   
   # Check GPU availability
   if torch.cuda.is_available():
       device = "cuda:0"
   else:
       device = "cpu"
   
   segmentator = SegmentChestTotalSegmentator(
       device=device,
       verbose=True
   )
   
   labelmap = segmentator.segment("ct.nrrd")

Custom Post-Processing
----------------------

Add application-specific post-processing:

.. code-block:: python

   class CustomTotalSegmentator(SegmentChestTotalSegmentator):
       """TotalSegmentator with custom post-processing."""
       
       def post_process(self, labelmap):
           """Enhanced post-processing."""
           # Apply default post-processing
           labelmap = super().post_process(labelmap)
           
           # Add custom steps
           labelmap = self.refine_cardiac_structures(labelmap)
           labelmap = self.fill_gaps_in_vessels(labelmap)
           
           return labelmap
       
       def refine_cardiac_structures(self, labelmap):
           """Refine cardiac chamber boundaries."""
           # Custom refinement logic
           return labelmap

Multi-Structure Analysis
------------------------

Analyze relationships between structures:

.. code-block:: python

   # Segment scan
   labelmap = segmentator.segment("ct.nrrd")
   
   # Extract related structures
   heart = segmentator.extract_structure(labelmap, "heart_left_ventricle")
   aorta = segmentator.extract_structure(labelmap, "aorta")
   
   # Compute distances
   from scipy.ndimage import distance_transform_edt
   
   heart_dist = distance_transform_edt(~heart.astype(bool))
   contact_region = (heart_dist < 5) & aorta.astype(bool)
   
   print(f"Heart-aorta interface: {contact_region.sum()} voxels")

Performance Optimization
========================

Speed vs Quality
----------------

Choose the right mode for your needs:

.. code-block:: python

   # Fast mode: ~2x faster, slightly lower quality
   fast_seg = SegmentChestTotalSegmentator(fast=True)
   
   # Full mode: Best quality, takes longer
   full_seg = SegmentChestTotalSegmentator(fast=False)

Memory Management
-----------------

For large datasets or limited RAM:

.. code-block:: python

   def process_large_dataset(image_files):
       """Process many images with memory management."""
       
       # Create segmentator
       segmentator = SegmentChestTotalSegmentator(fast=True)
       
       for i, image_file in enumerate(image_files):
           print(f"Processing {i+1}/{len(image_files)}: {image_file}")
           
           # Segment
           labelmap = segmentator.segment(image_file)
           
           # Process and save immediately
           output_file = f"seg_{i:04d}.mha"
           segmentator.save_labelmap(labelmap, output_file)
           
           # Clear variables to free memory
           del labelmap
           
           if i % 10 == 0:
               import gc
               gc.collect()

Quality Control
===============

Validation
----------

Check segmentation quality:

.. code-block:: python

   import numpy as np
   
   def validate_segmentation(labelmap):
       """Basic quality checks."""
       # Check for empty segmentation
       if labelmap.max() == 0:
           print("Warning: Empty segmentation")
           return False
       
       # Check for expected structures
       unique_labels = set(np.unique(labelmap))
       expected = {1, 2, 3, 4, 5}  # Major organs
       
       if not expected.issubset(unique_labels):
           print(f"Warning: Missing expected structures")
           return False
       
       # Check volumes are reasonable
       for label_id in unique_labels:
           if label_id == 0:
               continue
           volume = (labelmap == label_id).sum()
           if volume < 100:  # Too small
               print(f"Warning: Label {label_id} very small ({volume} voxels)")
       
       return True
   
   # Use validation
   labelmap = segmentator.segment("ct.nrrd")
   is_valid = validate_segmentation(labelmap)

Comparison with Ground Truth
-----------------------------

.. code-block:: python

   def compute_dice(pred, truth):
       """Compute Dice coefficient."""
       intersection = np.logical_and(pred, truth).sum()
       return 2.0 * intersection / (pred.sum() + truth.sum())
   
   # Compare with manual segmentation
   predicted = segmentator.segment("ct.nrrd")
   ground_truth = segmentator.load_image("manual_segmentation.mha")
   
   # Extract specific structure
   pred_heart = predicted == 1  # Assuming heart is label 1
   true_heart = ground_truth == 1
   
   dice = compute_dice(pred_heart, true_heart)
   print(f"Dice coefficient: {dice:.3f}")

Best Practices
==============

1. **Use fast mode for development**, full mode for final results
2. **Enable GPU** when available for significant speedup
3. **Validate inputs** - check image orientation, spacing
4. **Post-process results** - fill holes, smooth boundaries as needed
5. **Save intermediate results** for checkpoint/resume capability

Common Issues
=============

Missing Structures
------------------

If certain structures are not segmented:
* Check input image quality (resolution, contrast, noise)
* Verify proper windowing/leveling for CT
* Try different preprocessing (denoising, normalization)
* Consider ensemble with VISTA-3D for better coverage

Incorrect Boundaries
--------------------

If boundaries are imprecise:
* Disable fast mode for better quality
* Apply custom post-processing (smoothing, morphological operations)
* Use ensemble approach combining multiple methods

Performance Issues
------------------

If segmentation is too slow:
* Enable fast mode
* Use GPU acceleration
* Process images at lower resolution (with resampling)
* Batch process during off-hours

See Also
========

* :doc:`index` - Segmentation overview
* :doc:`vista3d` - VISTA-3D for cardiac structures
* :doc:`ensemble` - Combine with other methods
* :doc:`../workflows` - Use in complete workflows

.. rubric:: Navigation

:doc:`base` | :doc:`index` | :doc:`vista3d`
