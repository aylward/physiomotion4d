====================================
Segmentation Base Class
====================================

.. currentmodule:: physiomotion4d

Abstract base class for all segmentation methods.

Class Reference
===============

.. autoclass:: SegmentChestBase
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

:class:`SegmentChestBase` provides the foundation for all segmentation implementations in PhysioMotion4D. It defines the common interface and shared functionality that all segmentation methods must implement.

**Key Responsibilities**:
   * Define standard segmentation interface
   * Provide common post-processing operations
   * Handle structure extraction and labeling
   * Manage segmentation parameters and validation

Key Methods
===========

Abstract Methods
----------------

Subclasses must implement these methods:

.. method:: segment(image_path)

   Main segmentation method.
   
   :param image_path: Path to input image
   :type image_path: str
   :returns: Segmented labelmap
   :rtype: numpy.ndarray

.. method:: get_label_names()

   Get list of structure names that can be segmented.
   
   :returns: List of anatomical structure names
   :rtype: list[str]

Common Methods
--------------

These methods are provided by the base class:

.. method:: post_process(labelmap)

   Apply post-processing to segmentation.
   
   :param labelmap: Input labelmap
   :type labelmap: numpy.ndarray
   :returns: Post-processed labelmap
   :rtype: numpy.ndarray

.. method:: extract_structure(labelmap, structure_name)

   Extract a single anatomical structure.
   
   :param labelmap: Full segmentation
   :type labelmap: numpy.ndarray
   :param structure_name: Name of structure to extract
   :type structure_name: str
   :returns: Binary mask of structure
   :rtype: numpy.ndarray

.. method:: compute_volume(mask)

   Calculate volume of segmented structure.
   
   :param mask: Binary mask
   :type mask: numpy.ndarray
   :returns: Volume in cubic millimeters
   :rtype: float

Creating Custom Segmentation Classes
=====================================

To create a new segmentation method, inherit from :class:`SegmentChestBase`:

Basic Implementation
--------------------

.. code-block:: python

   from physiomotion4d import SegmentChestBase
   import numpy as np
   
   class CustomSegmentator(SegmentChestBase):
       """Custom segmentation implementation."""
       
       def __init__(self, param1=None, verbose=False):
           """Initialize segmentator.
           
           Args:
               param1: Custom parameter
               verbose: Enable verbose logging
           """
           super().__init__(verbose=verbose)
           self.param1 = param1
           
           self.log("CustomSegmentator initialized", level="INFO")
       
       def segment(self, image_path):
           """Segment image using custom method.
           
           Args:
               image_path: Path to input image
           
           Returns:
               Segmented labelmap
           """
           self.log(f"Segmenting {image_path}", level="INFO")
           
           # Load image
           image = self.load_image(image_path)
           
           # Apply custom segmentation algorithm
           labelmap = self._custom_algorithm(image)
           
           # Post-process
           labelmap = self.post_process(labelmap)
           
           return labelmap
       
       def get_label_names(self):
           """Get structure names."""
           return ['heart', 'lungs', 'aorta']
       
       def _custom_algorithm(self, image):
           """Internal segmentation algorithm."""
           # Implement your segmentation method here
           labelmap = np.zeros_like(image)
           # ... processing ...
           return labelmap

With Custom Post-Processing
----------------------------

.. code-block:: python

   class CustomSegmentator(SegmentChestBase):
       """Segmentator with custom post-processing."""
       
       def post_process(self, labelmap):
           """Apply custom post-processing."""
           # Call parent post-processing first
           labelmap = super().post_process(labelmap)
           
           # Add custom operations
           labelmap = self.fill_holes(labelmap)
           labelmap = self.smooth_boundaries(labelmap)
           labelmap = self.remove_small_components(labelmap)
           
           return labelmap
       
       def fill_holes(self, labelmap):
           """Fill holes in segmentation."""
           from scipy.ndimage import binary_fill_holes
           
           result = np.zeros_like(labelmap)
           for label_id in np.unique(labelmap):
               if label_id == 0:  # Skip background
                   continue
               mask = labelmap == label_id
               filled = binary_fill_holes(mask)
               result[filled] = label_id
           
           return result
       
       def smooth_boundaries(self, labelmap):
           """Smooth structure boundaries."""
           from scipy.ndimage import gaussian_filter
           
           # Apply smoothing to each label separately
           result = np.zeros_like(labelmap)
           for label_id in np.unique(labelmap):
               if label_id == 0:
                   continue
               mask = (labelmap == label_id).astype(float)
               smoothed = gaussian_filter(mask, sigma=1.0)
               result[smoothed > 0.5] = label_id
           
           return result

Integration Examples
====================

Using Custom Segmentator
-------------------------

.. code-block:: python

   # Create custom segmentator instance
   segmentator = CustomSegmentator(param1="value", verbose=True)
   
   # Use like any other segmentator
   labelmap = segmentator.segment("input.nrrd")
   
   # Extract structures
   heart = segmentator.extract_structure(labelmap, "heart")
   
   # Compute metrics
   volume = segmentator.compute_volume(heart)
   print(f"Heart volume: {volume} mm³")

In Workflows
------------

Custom segmentators can be integrated into workflows:

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   # Create workflow with custom segmentator
   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=["phase0.nrrd", "phase1.nrrd"],
       segmentator=CustomSegmentator(verbose=True),
       output_directory="./results"
   )
   
   result = workflow.process()

Common Patterns
===============

Structure Validation
--------------------

Validate that required structures are present:

.. code-block:: python

   class ValidatingSegmentator(SegmentChestBase):
       """Segmentator with validation."""
       
       def segment(self, image_path):
           """Segment with validation."""
           labelmap = super().segment(image_path)
           
           # Validate required structures
           required_structures = ['heart', 'lungs']
           present_labels = set(np.unique(labelmap))
           
           for structure in required_structures:
               label_id = self.get_label_id(structure)
               if label_id not in present_labels:
                   self.log(f"Warning: {structure} not found", level="WARNING")
           
           return labelmap

Progress Tracking
-----------------

Track segmentation progress for long operations:

.. code-block:: python

   class ProgressSegmentator(SegmentChestBase):
       """Segmentator with progress tracking."""
       
       def segment(self, image_path):
           """Segment with progress updates."""
           self.log("Loading image...", level="INFO")
           image = self.load_image(image_path)
           
           self.log("Preprocessing (10%)", level="INFO")
           preprocessed = self.preprocess(image)
           
           self.log("Running inference (50%)", level="INFO")
           labelmap = self.run_inference(preprocessed)
           
           self.log("Post-processing (90%)", level="INFO")
           result = self.post_process(labelmap)
           
           self.log("Complete (100%)", level="INFO")
           return result

Best Practices
==============

Implementation Guidelines
-------------------------

1. **Always call super().__init__()** in constructor
2. **Validate inputs** before processing
3. **Provide progress updates** for long operations
4. **Handle errors gracefully** with try-except
5. **Document parameters** clearly in docstrings
6. **Test with various input types** (different resolutions, spacings, etc.)

Performance Considerations
--------------------------

* Cache models and weights after first load
* Use GPU when available
* Implement batch processing for multiple images
* Minimize memory usage for large 3D volumes

Error Handling
--------------

.. code-block:: python

   def segment(self, image_path):
       """Robust segmentation with error handling."""
       try:
           # Validate input
           image_path = self.validate_file_exists(image_path)
           
           # Load and process
           image = self.load_image(image_path)
           
           # Check image properties
           if image.ndim != 3:
               raise ValueError("Expected 3D image")
           
           # Segment
           labelmap = self._segment_impl(image)
           
           return labelmap
           
       except FileNotFoundError as e:
           self.log(f"File not found: {e}", level="ERROR")
           raise
       except RuntimeError as e:
           self.log(f"Segmentation failed: {e}", level="ERROR")
           raise
       except Exception as e:
           self.log(f"Unexpected error: {e}", level="ERROR")
           raise

See Also
========

* :doc:`index` - Segmentation module overview
* :doc:`totalsegmentator` - TotalSegmentator implementation
* :doc:`vista3d` - VISTA-3D implementation
* :doc:`../../developer/extending` - Extending PhysioMotion4D

.. rubric:: Navigation

:doc:`index` | :doc:`totalsegmentator` | :doc:`vista3d`
