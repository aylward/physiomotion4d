========================
Utilities API Reference
========================

This section documents utility classes and helper functions in PhysioMotion4D.

Transform Tools
===============

.. automodule:: physiomotion4d.transform_tools
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`TransformTools` class provides utilities for applying and manipulating
deformation fields and transforms.

Applying Transforms
-------------------

.. code-block:: python

   from physiomotion4d import TransformTools
   import itk

   tools = TransformTools()
   
   # Load deformation field
   phi = itk.imread("deformation_field.mha")
   
   # Transform an image
   moving_image = itk.imread("moving.mha")
   warped_image = tools.apply_transform_to_image(moving_image, phi)
   
   # Transform a mesh/contour
   import pyvista as pv
   mesh = pv.read("contour.vtp")
   warped_mesh = tools.apply_transform_to_contour(mesh, phi)

Transform Composition
---------------------

.. code-block:: python

   # Compose multiple transforms
   phi_AB = itk.imread("transform_A_to_B.mha")
   phi_BC = itk.imread("transform_B_to_C.mha")
   
   # Compute phi_AC = phi_BC ∘ phi_AB
   phi_AC = tools.combine_displacement_field_transforms(phi_AB, phi_BC, reference_image, tfm1_weight=1.0, tfm2_weight=1.0, mode="compose")

Transform Inversion
-------------------

.. code-block:: python

   # Compute inverse transform
   phi_forward = itk.imread("forward_transform.mha")
   phi_inverse = tools.invert_transform(
       phi_forward,
       iterations=20,  # More iterations = better accuracy
       regularization=0.1
   )

USD Tools
=========

.. automodule:: physiomotion4d.usd_tools
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`USDTools` class provides utilities for working with USD files.

Merging USD Files
-----------------

.. code-block:: python

   from physiomotion4d import USDTools

   tools = USDTools()
   
   # Merge multiple USD files
   files_to_merge = [
       "heart_animation.usd",
       "lungs_animation.usd",
       "vessels_static.usd"
   ]
   
   tools.merge_usd_files(
       input_files=files_to_merge,
       output_file="complete_model.usd",
       flatten=True  # Flatten stage composition
   )

Time Sample Management
----------------------

.. code-block:: python

   # Set time-varying attributes
   from pxr import Usd, UsdGeom
   
   tools = USDTools()
   stage = Usd.Stage.Open("model.usd")
   
   # Add time samples to mesh
   mesh_prim = stage.GetPrimAtPath("/World/Mesh")
   mesh = UsdGeom.Mesh(mesh_prim)
   
   # Set points at different time codes
   for frame in range(10):
       time_code = frame
       points = compute_points_at_frame(frame)
       tools.set_time_sample(mesh, "points", points, time_code)

USD Validation
--------------

.. code-block:: python

   # Validate USD file
   is_valid, errors = tools.validate_usd("model.usd")
   
   if not is_valid:
       print("USD validation errors:")
       for error in errors:
           print(f"  - {error}")

Contour Tools
=============

.. automodule:: physiomotion4d.contour_tools
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`ContourTools` class provides utilities for extracting and processing
surface meshes from segmentation masks.

Surface Extraction
------------------

.. code-block:: python

   from physiomotion4d import ContourTools
   import itk

   tools = ContourTools()
   
   # Load segmentation mask
   mask = itk.imread("heart_mask.nrrd")
   
   # Extract surface mesh
   mesh = tools.extract_surface(
       mask,
       threshold=0.5,  # Isosurface value
       smoothing_iterations=10,
       decimate_target=10000  # Target number of triangles
   )
   
   # Save mesh
   import pyvista as pv
   pv_mesh = pv.wrap(mesh)
   pv_mesh.save("heart_surface.vtp")

Mesh Processing
---------------

.. code-block:: python

   import pyvista as pv
   
   tools = ContourTools()
   mesh = pv.read("surface.vtp")
   
   # Smooth mesh
   smoothed = tools.smooth_contour(
       mesh,
       iterations=20,
       relaxation_factor=0.1
   )
   
   # Decimate mesh (reduce complexity)
   decimated = tools.decimate_mesh(
       mesh,
       target_reduction=0.5  # 50% reduction
   )
   
   # Compute normals
   with_normals = tools.compute_normals(mesh)

Multi-label Surface Extraction
-------------------------------

.. code-block:: python

   # Extract surfaces for multiple labels
   multi_label_mask = itk.imread("all_organs.nrrd")
   
   label_names = {
       1: "heart",
       2: "left_lung", 
       3: "right_lung",
       4: "aorta"
   }
   
   meshes = {}
   for label_id, name in label_names.items():
       mesh = tools.extract_surface_from_label(
           multi_label_mask,
           label_id=label_id,
           smoothing_iterations=10
       )
       meshes[name] = mesh
       mesh.save(f"{name}_surface.vtp")

Anatomy Painting
================

.. automodule:: physiomotion4d.usd_anatomy_tools
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`USDAnatomyTools` class applies anatomically realistic materials to USD models.

Basic Material Application
---------------------------

.. code-block:: python

   from physiomotion4d import USDAnatomyTools

   painter = USDAnatomyTools()
   
   # Paint anatomical structures
   painter.paint_usd_file(
       input_usd="model.usd",
       output_usd="painted_model.usd",
       anatomy_mapping={
           "/World/Heart": "cardiac_muscle",
           "/World/Aorta": "arterial_vessel",
           "/World/LeftLung": "lung_tissue",
           "/World/RightLung": "lung_tissue",
           "/World/Bone": "cortical_bone"
       }
   )

Available Anatomy Materials
----------------------------

.. code-block:: python

   # Available pre-defined materials
   materials = painter.get_available_materials()
   
   # Materials include:
   # - cardiac_muscle: Pink-red heart tissue
   # - arterial_vessel: Red blood vessels
   # - venous_vessel: Blue veins
   # - lung_tissue: Light pink lungs
   # - cortical_bone: White/ivory bone
   # - soft_tissue: Beige muscle
   # - fat_tissue: Yellow adipose tissue
   # - liver_tissue: Dark red liver
   # - kidney_tissue: Dark red kidney

Custom Materials
----------------

.. code-block:: python

   # Define custom material
   painter.define_custom_material(
       name="my_custom_material",
       diffuse_color=(0.8, 0.2, 0.2),  # RGB
       metallic=0.0,
       roughness=0.5,
       opacity=1.0,
       emissive_color=(0.0, 0.0, 0.0)
   )
   
   # Apply custom material
   painter.apply_material_to_prim(
       prim_path="/World/CustomMesh",
       material_name="my_custom_material"
   )

Procedural Textures
-------------------

.. code-block:: python

   # Apply procedural texture (e.g., for vessels)
   painter.apply_procedural_texture(
       prim_path="/World/Aorta",
       texture_type="vessel",  # or "muscle", "fat"
       scale=1.0,
       noise_amplitude=0.1
   )

Data Management Utilities
==========================

DirLab 4D-CT Dataset
--------------------

.. code-block:: python

   from physiomotion4d.data import DirLab4DCT

   # Download DirLab dataset
   downloader = DirLab4DCT()
   
   # Download specific case
   downloader.download_case(
       case_number=1,
       output_dir="./data/DirLab"
   )
   
   # Load case data
   case_data = downloader.load_case(1)
   
   # Access images
   inhale_image = case_data["inhale"]
   exhale_image = case_data["exhale"]
   
   # Access landmarks (for validation)
   landmarks_fixed = case_data["landmarks_inhale"]
   landmarks_moving = case_data["landmarks_exhale"]

File I/O Utilities
==================

Medical Image I/O
-----------------

.. code-block:: python

   from physiomotion4d.io_utils import (
       read_medical_image,
       write_medical_image,
       convert_to_itk,
       convert_to_numpy
   )
   
   # Read image (auto-detects format)
   image = read_medical_image("scan.nrrd")  # or .mha, .nii, .dcm
   
   # Convert to NumPy
   array = convert_to_numpy(image)
   spacing = image.GetSpacing()
   origin = image.GetOrigin()
   
   # Convert back to ITK
   new_image = convert_to_itk(array, spacing, origin)
   
   # Write image
   write_medical_image(new_image, "output.nrrd")

Mesh I/O
--------

.. code-block:: python

   from physiomotion4d.io_utils import read_mesh, write_mesh
   
   # Read mesh (VTK, OBJ, STL, PLY)
   mesh = read_mesh("model.vtp")
   
   # Write mesh
   write_mesh(mesh, "output.vtp")

Visualization Utilities
=======================

Quick Plotting
--------------

.. code-block:: python

   from physiomotion4d.visualization import (
       plot_image_slice,
       plot_mesh,
       plot_registration_result,
       create_animation
   )
   
   import itk
   
   # Plot image slice
   image = itk.imread("ct_scan.nrrd")
   plot_image_slice(image, slice_index=50, axis='z')
   
   # Plot mesh
   import pyvista as pv
   mesh = pv.read("heart.vtp")
   plot_mesh(mesh, color='red', opacity=0.8)
   
   # Compare registration
   fixed = itk.imread("fixed.mha")
   moving = itk.imread("moving.mha")
   registered = itk.imread("registered.mha")
   
   plot_registration_result(fixed, moving, registered)

Animation Creation
------------------

.. code-block:: python

   # Create animation from time series
   image_files = [f"frame_{i:03d}.mha" for i in range(10)]
   
   create_animation(
       image_files,
       output_file="cardiac_cycle.gif",
       fps=10,
       slice_index=50
   )

Base Classes
============

PhysioMotion4D Base Class
--------------------------

.. autoclass:: physiomotion4d.PhysioMotion4DBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The :class:`PhysioMotion4DBase` class provides standardized logging and debug
settings for all PhysioMotion4D classes. Workflow and registration classes
inherit from this base class to provide consistent logging behavior.

**Features:**

* Unified logging interface with standard log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
* Class-based log filtering to show/hide logs from specific classes
* Progress reporting for long-running operations
* Shared logger named "PhysioMotion4D" for all logs

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import logging
   from physiomotion4d import HeartModelToPatientWorkflow, PhysioMotion4DBase
   
   # Create workflow instance (inherits from PhysioMotion4DBase)
   workflow = HeartModelToPatientWorkflow(
       moving_mesh=model,
       fixed_meshes=surfaces,
       fixed_image=image,
       log_level=logging.DEBUG
   )
   
   # Workflow will now show debug-level messages
   workflow.run_workflow()

Global Logging Control
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Control log level for all PhysioMotion4D classes
   PhysioMotion4DBase.set_log_level(logging.INFO)
   
   # Or use string
   PhysioMotion4DBase.set_log_level('DEBUG')

Class-Based Filtering
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Show logs only from specific classes
   PhysioMotion4DBase.set_log_classes([
       "RegisterModelToImagePCA",
       "HeartModelToPatientWorkflow"
   ])
   
   # All other classes' logs will be hidden
   
   # Show all classes again
   PhysioMotion4DBase.set_log_all_classes()
   
   # Query which classes are currently filtered
   filtered_classes = PhysioMotion4DBase.get_log_classes()
   print(filtered_classes)  # [] if all shown, or list of class names

Creating Custom Classes
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase
   
   class MyProcessor(PhysioMotion4DBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(
               logger_name="MyProcessor",
               log_level=log_level
           )
       
       def process(self, data):
           self.log_info("Starting processing...")
           self.log_debug(f"Processing {len(data)} items")
           
           # Report progress
           for i, item in enumerate(data):
               if i % 10 == 0:
                   self.log_progress(i+1, len(data), prefix="Processing")
               # ... do work ...
           
           self.log_info("Processing complete!")

See also the :doc:`../user_guide/logging` guide for more detailed examples.

Logging and Progress
====================

.. code-block:: python

   from physiomotion4d.utils import setup_logger, ProgressTracker
   
   # Setup logging
   logger = setup_logger(
       name="my_workflow",
       log_file="processing.log",
       level="INFO"
   )
   
   logger.info("Starting processing...")
   
   # Track progress
   tracker = ProgressTracker(total_steps=100, description="Processing")
   
   for i in range(100):
       # Do work
       process_step(i)
       tracker.update(1)
   
   tracker.close()

Configuration Management
========================

.. code-block:: python

   from physiomotion4d.config import Config
   
   # Load configuration
   config = Config.from_file("config.yaml")
   
   # Access settings
   output_dir = config.get("output_directory")
   num_iterations = config.get("registration.iterations", default=100)
   
   # Update settings
   config.set("segmentation.method", "vista3d")
   
   # Save configuration
   config.save("updated_config.yaml")

Example configuration file (``config.yaml``):

.. code-block:: yaml

   output_directory: ./results
   project_name: cardiac_study
   
   segmentation:
     method: vista3d
     device: cuda
     
   registration:
     method: icon
     iterations: 100
     smoothness: 0.5
     device: cuda
     
   usd_export:
     frame_rate: 30
     flatten_stage: true
     apply_materials: true

See Also
========

* :doc:`../user_guide/usd_conversion` - USD conversion guide
* :doc:`../tutorials/vtk_to_usd` - VTK to USD tutorial
* :doc:`core` - Core API reference

