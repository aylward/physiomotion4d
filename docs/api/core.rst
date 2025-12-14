====================
Core API Reference
====================

This section documents the core workflow processors and main entry points for PhysioMotion4D.

All workflow classes inherit from :class:`~physiomotion4d.PhysioMotion4DBase` and support
standardized logging. See :doc:`utilities` for logging configuration options.

Workflow Classes
================

Heart Gated CT to USD Workflow
-------------------------------

.. autoclass:: physiomotion4d.HeartGatedCTToUSDWorkflow
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Complete workflow for processing heart-gated CT data to animated USD models.
   This class orchestrates the full pipeline from 4D CT input to Omniverse-ready output.

   **Key Methods:**

   * :meth:`process`: Run the complete workflow
   * :meth:`convert_4d_to_3d`: Convert 4D NRRD to 3D time frames
   * :meth:`register_images`: Perform image registration
   * :meth:`segment_reference_image`: Generate AI-based segmentation
   * :meth:`transform_contours`: Transform contours across time
   * :meth:`create_usd_models`: Generate USD models
   * :meth:`paint_anatomy`: Apply realistic materials and textures

   **Example:**

   .. code-block:: python

      from physiomotion4d import HeartGatedCTToUSDWorkflow

      workflow = HeartGatedCTToUSDWorkflow(
          input_filenames=["cardiac_4d.nrrd"],
          contrast_enhanced=True,
          output_directory="./results",
          project_name="patient_001",
          registration_method='icon'  # or 'ants'
      )
      final_usd = workflow.process()

Heart Model to Patient Workflow
--------------------------------

.. autoclass:: physiomotion4d.HeartModelToPatientWorkflow
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Register generic anatomical models to patient-specific imaging data.
   Implements a three-stage registration pipeline: ICP → mask-to-mask → mask-to-image.

   **Key Methods:**

   * :meth:`run_workflow`: Execute full three-stage registration
   * :meth:`register_icp`: ICP-based rough alignment
   * :meth:`register_masks`: Mask-based deformable registration
   * :meth:`register_to_image`: Optional image-based refinement
   * :meth:`set_roi_dilation_mm`: Configure region of interest
   * :meth:`set_mask_blur_sigma`: Configure mask smoothing

   **Example:**

   .. code-block:: python

      from physiomotion4d import HeartModelToPatientWorkflow
      import pyvista as pv
      import itk

      # Load data
      model_mesh = pv.read("generic_heart.vtu")
      patient_surfaces = [pv.read("lv.stl"), pv.read("rv.stl")]
      patient_image = itk.imread("patient_ct.nii.gz")

      # Initialize and run workflow
      workflow = HeartModelToPatientWorkflow(
          moving_mesh=model_mesh,
          fixed_meshes=patient_surfaces,
          fixed_image=patient_image
      )
      
      # Configure parameters
      workflow.set_roi_dilation_mm(20)
      workflow.set_mask_blur_sigma(5)
      
      # Run registration
      registered_mesh = workflow.run_workflow()

Data Conversion
===============

NRRD 4D to 3D Converter
------------------------

.. autoclass:: physiomotion4d.ConvertNRRD4DTo3D
   :members:
   :undoc-members:
   :show-inheritance:

   Converts 4D NRRD files to multiple 3D time frames.

VTK to USD Converters
---------------------

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDBase
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for VTK to USD conversion.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDPolyMesh
   :members:
   :undoc-members:
   :show-inheritance:

   Convert VTK polygonal meshes to USD with time-varying geometry.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDTetMesh
   :members:
   :undoc-members:
   :show-inheritance:

   Convert VTK tetrahedral meshes to USD with time-varying geometry.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSD
   :members:
   :undoc-members:
   :show-inheritance:

   High-level VTK to USD converter that automatically selects appropriate mesh type.

Utility Classes
===============

Image Tools
-----------

.. autoclass:: physiomotion4d.ImageTools
   :members:
   :undoc-members:
   :show-inheritance:

   Utilities for medical image processing operations.

   **Key Methods:**

   * :meth:`resample_image`: Resample image to new spacing
   * :meth:`normalize_intensity`: Normalize image intensities
   * :meth:`crop_to_roi`: Crop image to region of interest
   * :meth:`pad_image`: Pad image to desired size
   * :meth:`compute_statistics`: Calculate image statistics

Transform Tools
---------------

.. autoclass:: physiomotion4d.TransformTools
   :members:
   :undoc-members:
   :show-inheritance:

   Utilities for image and contour transformations using deformation fields.

   **Key Methods:**

   * :meth:`apply_transform_to_image`: Transform an image using a deformation field
   * :meth:`apply_transform_to_contour`: Transform contours/meshes
   * :meth:`compose_transforms`: Combine multiple transformations
   * :meth:`invert_transform`: Compute inverse deformation field

USD Tools
---------

.. autoclass:: physiomotion4d.USDTools
   :members:
   :undoc-members:
   :show-inheritance:

   Utilities for USD file manipulation and merging.

   **Key Methods:**

   * :meth:`merge_usd_files`: Combine multiple USD files
   * :meth:`set_time_samples`: Add time-varying attributes
   * :meth:`flatten_stage`: Flatten USD stage composition
   * :meth:`validate_usd`: Check USD file integrity

Contour Tools
-------------

.. autoclass:: physiomotion4d.ContourTools
   :members:
   :undoc-members:
   :show-inheritance:

   Tools for extracting and processing contours from segmentation masks.

   **Key Methods:**

   * :meth:`extract_surface`: Extract surface mesh from segmentation
   * :meth:`smooth_contour`: Apply smoothing to surface meshes
   * :meth:`decimate_mesh`: Reduce mesh complexity
   * :meth:`compute_normals`: Calculate surface normals

Anatomy Painting
----------------

.. autoclass:: physiomotion4d.USDAnatomyTools
   :members:
   :undoc-members:
   :show-inheritance:

   Apply anatomically realistic materials and textures to USD models.

   **Features:**

   * Procedural texture generation
   * Anatomical coloring schemes
   * Material property assignment
   * Custom texture mapping

