===========================
Segmentation API Reference
===========================

PhysioMotion4D provides multiple AI-based segmentation methods for chest CT images,
including heart, vessels, lungs, and bones.

Base Segmentation Class
========================

.. autoclass:: physiomotion4d.SegmentChestBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Abstract base class for chest segmentation implementations. All segmentation
   methods inherit from this class and implement the :meth:`segment` method.

   **Common Interface:**

   All segmentation classes provide these methods:

   * :meth:`segment`: Main segmentation method
   * :meth:`set_device`: Set computation device (CPU/GPU)
   * :meth:`preprocess`: Optional preprocessing
   * :meth:`postprocess`: Optional postprocessing

TotalSegmentator
================

.. autoclass:: physiomotion4d.SegmentChestTotalSegmentator
   :members:
   :undoc-members:
   :show-inheritance:

   Segmentation using the TotalSegmentator model. Provides fast and accurate
   segmentation of 104 anatomical structures.

   **Features:**

   * Fast inference (~30 seconds per volume)
   * Pre-trained on diverse dataset
   * Supports both contrast and non-contrast CT
   * Automatic model download

   **Example:**

   .. code-block:: python

      from physiomotion4d import SegmentChestTotalSegmentator
      import itk

      segmenter = SegmentChestTotalSegmentator()
      image = itk.imread("chest_ct.nrrd")
      masks = segmenter.segment(image, contrast_enhanced_study=True)

   **Output Masks:**

   Returns tuple of 8 masks:

   1. ``heart_mask`` - Heart and pericardium
   2. ``vessels_mask`` - Major vessels (aorta, vena cava, pulmonary)
   3. ``lungs_mask`` - Left and right lungs
   4. ``bones_mask`` - Ribs, sternum, spine
   5. ``soft_tissue_mask`` - Muscles and soft tissues
   6. ``contrast_mask`` - Contrast-enhanced regions
   7. ``all_mask`` - Combined mask of all structures
   8. ``dynamic_mask`` - Moving structures (heart + lungs)

VISTA-3D
========

.. autoclass:: physiomotion4d.SegmentChestVista3D
   :members:
   :undoc-members:
   :show-inheritance:

   Segmentation using NVIDIA's VISTA-3D foundation model. Provides state-of-the-art
   segmentation with fine-grained anatomical detail.

   **Features:**

   * Foundation model trained on 10,000+ CT scans
   * Superior edge detection and small structure segmentation
   * Point-prompt and auto-segmentation modes
   * Supports custom anatomical classes

   **Example:**

   .. code-block:: python

      from physiomotion4d import SegmentChestVista3D
      import itk

      segmenter = SegmentChestVista3D()
      image = itk.imread("chest_ct.nrrd")
      
      # Automatic segmentation
      masks = segmenter.segment(image, contrast_enhanced_study=True)
      
      # With custom points (point-prompt mode)
      point_prompts = [(120, 150, 80)]  # [x, y, z] in voxel coordinates
      masks = segmenter.segment(
          image,
          point_prompts=point_prompts,
          contrast_enhanced_study=True
      )

VISTA-3D with NVIDIA NIM
=========================

.. autoclass:: physiomotion4d.SegmentChestVista3DNIM
   :members:
   :undoc-members:
   :show-inheritance:

   Cloud-based VISTA-3D segmentation using NVIDIA NIM (NVIDIA Inference Microservices).
   Requires API credentials and internet connection.

   **Features:**

   * No local GPU required
   * Access to latest VISTA-3D models
   * Scalable cloud inference
   * Automatic updates

   **Setup:**

   .. code-block:: python

      import os
      
      # Set NIM API credentials
      os.environ['NGC_API_KEY'] = 'your_api_key_here'
      
      from physiomotion4d import SegmentChestVista3DNIM
      
      segmenter = SegmentChestVista3DNIM()
      masks = segmenter.segment(image, contrast_enhanced_study=True)

   **Requirements:**

   Install with NIM support:

   .. code-block:: bash

      pip install physiomotion4d[nim]

Ensemble Segmentation
=====================

.. autoclass:: physiomotion4d.SegmentChestEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

   Ensemble segmentation combining multiple methods for improved accuracy.
   Uses voting or weighted averaging of predictions.

   **Features:**

   * Combines TotalSegmentator + VISTA-3D
   * Reduces false positives/negatives
   * Improves boundary detection
   * Configurable fusion strategies

   **Example:**

   .. code-block:: python

      from physiomotion4d import SegmentChestEnsemble
      import itk

      segmenter = SegmentChestEnsemble(
          methods=['totalsegmentator', 'vista3d'],
          fusion_strategy='voting'  # or 'weighted_average'
      )
      
      image = itk.imread("chest_ct.nrrd")
      masks = segmenter.segment(image, contrast_enhanced_study=True)

   **Fusion Strategies:**

   * ``voting``: Majority vote across methods (default)
   * ``weighted_average``: Weighted combination with confidence scores
   * ``unanimous``: Only include regions agreed by all methods

Segmentation Comparison
========================

.. list-table:: Segmentation Method Comparison
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Method
     - Speed
     - Accuracy
     - GPU Memory
     - Best For
   * - TotalSegmentator
     - Fast (~30s)
     - Good
     - 4GB
     - Quick results, batch processing
   * - VISTA-3D
     - Moderate (~60s)
     - Excellent
     - 8GB
     - High quality, fine details
   * - VISTA-3D NIM
     - Moderate (~60s)
     - Excellent
     - 0GB (cloud)
     - No local GPU available
   * - Ensemble
     - Slow (~90s)
     - Best
     - 12GB
     - Critical applications, research

Custom Segmentation
===================

To implement a custom segmentation method, inherit from :class:`SegmentChestBase`:

.. code-block:: python

   from physiomotion4d import SegmentChestBase
   import itk

   class MyCustomSegmenter(SegmentChestBase):
       def __init__(self):
           super().__init__()
           # Initialize your model here
           
       def segment(self, image, contrast_enhanced_study=False):
           """
           Segment chest CT image.
           
           Args:
               image: ITK image to segment
               contrast_enhanced_study: Whether image is contrast-enhanced
               
           Returns:
               Tuple of 8 masks: (heart, vessels, lungs, bones,
                                  soft_tissue, contrast, all, dynamic)
           """
           # Your segmentation logic here
           return (heart_mask, vessels_mask, lungs_mask, bones_mask,
                   soft_tissue_mask, contrast_mask, all_mask, dynamic_mask)

   # Use your custom segmenter
   segmenter = MyCustomSegmenter()
   masks = segmenter.segment(image)

