====================================
Utilities Development Guide
====================================

This guide covers using utility modules for common operations.

For complete API documentation, see :doc:`../api/utilities/index`.

Overview
========

The utilities module provides helper functions and tools for image processing, mesh manipulation, coordinate transforms, and USD operations.

Overview
========

PhysioMotion4D includes utility modules for:

* **Image Tools**: Medical image I/O and processing
* **Transform Tools**: Spatial transformation utilities
* **Contour Tools**: Mesh generation and manipulation
* **USD Tools**: USD file operations
* **USD Anatomy Tools**: Anatomical material management

These utilities are used throughout PhysioMotion4D workflows and are available for custom applications.

Image Tools
===========

Medical Image I/O and Processing
---------------------------------

.. automodule:: physiomotion4d.image_tools
   :members:
   :undoc-members:

**Key Functions**:

.. code-block:: python

   from physiomotion4d.image_tools import (
       read_image,
       write_image,
       resample_image,
       crop_image,
       pad_image,
       normalize_intensity,
       compute_statistics
   )
   
   # Read medical image
   image = read_image("ct_scan.nrrd")
   print(f"Image shape: {image.GetSize()}")
   print(f"Spacing: {image.GetSpacing()}")
   
   # Resample to isotropic spacing
   isotropic = resample_image(
       image,
       new_spacing=[1.0, 1.0, 1.0],
       interpolation='linear'
   )
   
   # Normalize intensity
   normalized = normalize_intensity(
       image,
       window_min=-100,
       window_max=400
   )
   
   # Compute image statistics
   stats = compute_statistics(image)
   print(f"Mean: {stats['mean']}, Std: {stats['std']}")

**Image Manipulation**:

.. code-block:: python

   # Crop to region of interest
   cropped = crop_image(
       image,
       start_index=[50, 50, 50],
       size=[200, 200, 200]
   )
   
   # Pad image
   padded = pad_image(
       image,
       pad_size=[[10, 10], [10, 10], [10, 10]],
       constant_value=0
   )
   
   # Extract slice
   slice_2d = extract_slice(
       image,
       slice_index=100,
       orientation='axial'  # or 'sagittal', 'coronal'
   )

Transform Tools
===============

Spatial Transformation Utilities
---------------------------------

.. automodule:: physiomotion4d.transform_tools
   :members:
   :undoc-members:

**Key Functions**:

.. code-block:: python

   from physiomotion4d.transform_tools import (
       read_transform,
       write_transform,
       apply_transform_to_image,
       apply_transform_to_points,
       compose_transforms,
       invert_transform,
       convert_transform_format
   )
   
   # Read transform (HDF5, TFM, or MAT)
   transform = read_transform("registration.hdf5")
   
   # Apply to image
   transformed_img = apply_transform_to_image(
       image="moving.mha",
       transform=transform,
       reference="fixed.mha",
       interpolation='linear'
   )
   
   # Apply to point set
   transformed_points = apply_transform_to_points(
       points=point_array,
       transform=transform
   )
   
   # Compose multiple transforms
   combined = compose_transforms(
       [transform1, transform2, transform3]
   )
   
   # Invert transform
   inverse = invert_transform(transform)

**Displacement Field Operations**:

.. code-block:: python

   # Get displacement field from transform
   displacement_field = get_displacement_field(
       transform=transform,
       reference_image="fixed.mha"
   )
   
   # Compute deformation statistics
   magnitude = compute_displacement_magnitude(displacement_field)
   print(f"Max displacement: {magnitude.max()} mm")
   print(f"Mean displacement: {magnitude.mean()} mm")
   
   # Warp mesh with displacement field
   warped_mesh = warp_mesh_with_field(
       mesh="mesh.vtk",
       displacement_field=displacement_field
   )

Contour Tools
=============

Mesh Generation and Manipulation
---------------------------------

.. automodule:: physiomotion4d.contour_tools
   :members:
   :undoc-members:

**Key Functions**:

.. code-block:: python

   from physiomotion4d.contour_tools import (
       extract_surface_mesh,
       smooth_mesh,
       decimate_mesh,
       compute_mesh_properties,
       create_mesh_from_points,
       merge_meshes
   )
   
   # Extract surface from labelmap
   mesh = extract_surface_mesh(
       labelmap="segmentation.mha",
       label_value=1,
       smooth_iterations=20,
       decimate_target_reduction=0.3
   )
   
   # Smooth mesh
   smoothed = smooth_mesh(
       mesh,
       iterations=50,
       relaxation_factor=0.1
   )
   
   # Decimate for performance
   decimated = decimate_mesh(
       mesh,
       target_reduction=0.5,
       preserve_topology=True
   )
   
   # Compute properties
   props = compute_mesh_properties(mesh)
   print(f"Surface area: {props['area']} mm²")
   print(f"Volume: {props['volume']} mm³")
   print(f"Number of triangles: {props['num_cells']}")

**Mesh Processing**:

.. code-block:: python

   # Fill holes in mesh
   filled = fill_mesh_holes(
       mesh,
       hole_size=10.0
   )
   
   # Compute normals
   with_normals = compute_mesh_normals(
       mesh,
       consistent_orientation=True
   )
   
   # Create distance map from mesh
   distance_image = mesh_to_distance_image(
       mesh=mesh,
       reference_image="ct.mha",
       signed=True
   )
   
   # Merge multiple meshes
   combined_mesh = merge_meshes([mesh1, mesh2, mesh3])

USD Tools
=========

USD File Operations
-------------------

.. automodule:: physiomotion4d.usd_tools
   :members:
   :undoc-members:

**Key Functions**:

.. code-block:: python

   from physiomotion4d.usd_tools import (
       create_usd_stage,
       add_mesh_to_stage,
       apply_transform_to_prim,
       merge_usd_files,
       export_usd_to_format,
       validate_usd
   )
   
   # Create new USD stage
   stage = create_usd_stage("scene.usd")
   
   # Add mesh to stage
   mesh_prim = add_mesh_to_stage(
       stage=stage,
       vertices=vertices,
       faces=faces,
       prim_path="/Anatomy/Heart",
       colors=vertex_colors
   )
   
   # Apply transformation
   apply_transform_to_prim(
       prim=mesh_prim,
       translation=[10, 0, 0],
       rotation=[0, 0, 90],
       scale=[1.0, 1.0, 1.0]
   )
   
   # Merge USD files
   merge_usd_files(
       input_files=["heart.usd", "lungs.usd"],
       output_file="anatomy.usd"
   )
   
   # Validate USD file
   is_valid, errors = validate_usd("scene.usd")
   if not is_valid:
       print(f"Validation errors: {errors}")

**Time-Varying Operations**:

.. code-block:: python

   # Add time samples to geometry
   add_time_varying_geometry(
       stage=stage,
       prim_path="/Anatomy/Heart",
       vertices_per_time={
           0.0: vertices_t0,
           0.5: vertices_t1,
           1.0: vertices_t2
       }
   )
   
   # Set animation timing
   set_stage_timerange(
       stage=stage,
       start_time=0.0,
       end_time=2.0,
       fps=30
   )

USD Anatomy Tools
=================

Anatomical Material Management
-------------------------------

.. automodule:: physiomotion4d.usd_anatomy_tools
   :members:
   :undoc-members:

**Key Functions**:

.. code-block:: python

   from physiomotion4d.usd_anatomy_tools import (
       get_anatomical_material,
       apply_anatomical_material,
       create_material_library,
       map_structure_to_material
   )
   
   # Get material for anatomical structure
   material = get_anatomical_material("heart_left_ventricle")
   print(f"Material: {material}")
   
   # Apply material to USD prim
   apply_anatomical_material(
       stage=stage,
       prim_path="/Anatomy/Heart/LeftVentricle",
       structure_name="heart_left_ventricle"
   )
   
   # Create custom material library
   library = create_material_library({
       'custom_cardiac': {
           'base_color': [0.8, 0.2, 0.2],
           'metallic': 0.0,
           'roughness': 0.7,
           'opacity': 0.95
       }
   })

**Structure Name Mapping**:

.. code-block:: python

   # Map mesh name to material
   structure_name = map_structure_to_material("heart_lv_smooth")
   # Returns: "heart_left_ventricle"
   
   # Get all available anatomical structures
   structures = get_available_structures()
   print(f"Available: {len(structures)} structures")
   
   # Get material color palette
   palette = get_material_palette("cardiac")
   # Returns colors for all cardiac structures

Common Utility Patterns
========================

Pipeline Integration
--------------------

Combine utilities in processing pipelines:

.. code-block:: python

   from physiomotion4d.image_tools import read_image, resample_image
   from physiomotion4d.contour_tools import extract_surface_mesh
   from physiomotion4d.usd_tools import create_usd_stage, add_mesh_to_stage
   from physiomotion4d.usd_anatomy_tools import apply_anatomical_material
   
   def labelmap_to_usd(labelmap_file, output_usd):
       """Convert labelmap to USD with materials."""
       # Load image
       labelmap = read_image(labelmap_file)
       
       # Resample to standard spacing
       resampled = resample_image(labelmap, [1.0, 1.0, 1.0])
       
       # Extract meshes for each label
       stage = create_usd_stage(output_usd)
       
       for label_id in range(1, 10):
           mesh = extract_surface_mesh(resampled, label_id)
           
           if mesh.GetNumberOfPoints() > 0:
               prim = add_mesh_to_stage(
                   stage, mesh, f"/Anatomy/Structure_{label_id}"
               )
               apply_anatomical_material(
                   stage, prim.GetPath(), f"structure_{label_id}"
               )
       
       stage.Save()

Batch Processing
----------------

Process multiple files with utilities:

.. code-block:: python

   from pathlib import Path
   from physiomotion4d.image_tools import read_image, normalize_intensity
   
   def batch_normalize(input_dir, output_dir):
       """Normalize all images in directory."""
       input_path = Path(input_dir)
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)
       
       for img_file in input_path.glob("*.nrrd"):
           # Read
           image = read_image(str(img_file))
           
           # Normalize
           normalized = normalize_intensity(image, -100, 400)
           
           # Write
           output_file = output_path / f"{img_file.stem}_normalized.mha"
           write_image(normalized, str(output_file))
           
           print(f"Processed: {img_file.name}")

Custom Utility Functions
========================

Extending Image Tools
---------------------

Create custom image processing functions:

.. code-block:: python

   import SimpleITK as sitk
   from physiomotion4d.image_tools import read_image, write_image
   
   def apply_custom_filter(image_file, output_file):
       """Apply custom image filter."""
       # Read
       image = read_image(image_file)
       
       # Apply custom processing
       smoothed = sitk.CurvatureFlow(
           image,
           timeStep=0.125,
           numberOfIterations=5
       )
       
       # Threshold
       thresholded = sitk.BinaryThreshold(
           smoothed,
           lowerThreshold=50,
           upperThreshold=250
       )
       
       # Write
       write_image(thresholded, output_file)

Custom Mesh Processing
----------------------

Add custom mesh operations:

.. code-block:: python

   import pyvista as pv
   from physiomotion4d.contour_tools import smooth_mesh
   
   def custom_mesh_processing(mesh_file, output_file):
       """Custom mesh processing pipeline."""
       # Load
       mesh = pv.read(mesh_file)
       
       # Custom operations
       mesh = mesh.clean()
       mesh = mesh.fill_holes(hole_size=5.0)
       mesh = smooth_mesh(mesh, iterations=30)
       
       # Compute curvature
       mesh = mesh.compute_normals()
       curvature = mesh.curvature('mean')
       mesh['curvature'] = curvature
       
       # Save
       mesh.save(output_file)

Performance Tips
================

Image Processing
----------------

.. code-block:: python

   # Use appropriate data types
   image_uint8 = sitk.Cast(image, sitk.sitkUInt8)  # For labels
   image_float = sitk.Cast(image, sitk.sitkFloat32)  # For processing
   
   # Resample efficiently
   from physiomotion4d.image_tools import resample_image
   
   # Downsample for speed
   downsampled = resample_image(image, [2.0, 2.0, 2.0])
   
   # Process downsampled, then upsample result
   processed = apply_expensive_operation(downsampled)
   upsampled = resample_image(processed, original_spacing)

Mesh Processing
---------------

.. code-block:: python

   # Decimate before processing
   from physiomotion4d.contour_tools import decimate_mesh
   
   decimated = decimate_mesh(mesh, target_reduction=0.7)
   
   # Process decimated mesh (faster)
   processed = apply_expensive_mesh_operation(decimated)
   
   # Use efficient VTK operations
   import vtk
   
   # Fast mesh smoothing
   smoother = vtk.vtkWindowedSincPolyDataFilter()
   smoother.SetInputData(mesh)
   smoother.SetNumberOfIterations(20)
   smoother.Update()

Best Practices
==============

Error Handling
--------------

.. code-block:: python

   from physiomotion4d.image_tools import read_image
   
   try:
       image = read_image("input.nrrd")
   except FileNotFoundError:
       print("Input file not found")
   except RuntimeError as e:
       print(f"Failed to read image: {e}")

Memory Management
-----------------

.. code-block:: python

   # Clear large objects when done
   import gc
   
   large_image = read_image("large_scan.nrrd")
   # ... process ...
   del large_image
   gc.collect()
   
   # Process in chunks for very large data
   for slice_idx in range(image_depth):
       slice_2d = extract_slice(image, slice_idx)
       process_slice(slice_2d)
       del slice_2d

File I/O
--------

.. code-block:: python

   # Use appropriate formats
   write_image(image, "output.mha")  # Fast, compressed
   write_image(labelmap, "labels.nrrd")  # Good for 4D
   
   # Compress USD files
   stage.GetRootLayer().Save()
   stage.Export("output.usdz")  # Compressed archive

See Also
========

* :doc:`workflows` - Using utilities in workflows
* :doc:`architecture` - System architecture overview
* :doc:`extending` - Creating custom utilities
