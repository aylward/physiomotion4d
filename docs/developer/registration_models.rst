====================================
Model Registration Development Guide
====================================

This guide covers developing with 3D model registration methods.

For complete API documentation, see :doc:`../api/model_registration/index`.

Overview
========

The model registration module provides methods for aligning 3D mesh models, including surface-based and point-based registration approaches.

Overview
========

PhysioMotion4D supports multiple model registration approaches:

* **ICP**: Iterative Closest Point registration (VTK and ITK implementations)
* **Distance Maps**: Distance field-based registration
* **PCA**: Statistical shape model-based registration

These methods are used for model-to-model and model-to-image registration tasks.

Registration Methods
====================

ICP Registration (VTK)
----------------------

Iterative Closest Point using VTK's implementation.

.. autoclass:: physiomotion4d.RegisterModelsICP
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Fast surface-to-surface alignment
   * Rigid transformation only
   * Suitable for initial alignment
   * Works with VTK meshes

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterModelsICP
   
   # Initialize ICP registrator
   registrator = RegisterModelsICP(
       max_iterations=100,
       tolerance=1e-6,
       verbose=True
   )
   
   # Register meshes
   transform = registrator.register(
       fixed_mesh_path="reference_heart.vtp",
       moving_mesh_path="template_heart.vtp"
   )
   
   # Apply transform to mesh
   registered_mesh = registrator.apply_transform(
       mesh_path="template_heart.vtp",
       transform=transform
   )
   
   # Save registered mesh
   registrator.save_mesh(registered_mesh, "registered_heart.vtp")

**Parameters**:
   * ``max_iterations``: Maximum ICP iterations
   * ``tolerance``: Convergence threshold
   * ``use_landmarks``: Use landmark-based initialization
   * ``rigid_only``: Restrict to rigid transformations

ICP Registration (ITK)
----------------------

ITK-based ICP implementation with additional features.

.. autoclass:: physiomotion4d.RegisterModelsICPITK
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Supports point clouds and meshes
   * Multiple distance metrics
   * Outlier rejection
   * ITK transform compatibility

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterModelsICPITK
   
   # Initialize with ITK
   registrator = RegisterModelsICPITK(
       max_iterations=200,
       outlier_rejection=True,
       outlier_threshold=3.0,
       verbose=True
   )
   
   # Register with outlier handling
   transform = registrator.register(
       fixed_mesh_path="target.vtk",
       moving_mesh_path="source.vtk"
   )
   
   # Get registration error
   rms_error = registrator.get_rms_error()
   print(f"RMS error: {rms_error:.3f} mm")

Distance Map Registration
-------------------------

Register models using distance fields.

.. autoclass:: physiomotion4d.RegisterModelsDistanceMaps
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Uses signed distance fields
   * Handles partial overlaps
   * Robust to missing data
   * Supports deformable registration

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterModelsDistanceMaps
   
   # Initialize distance-based registrator
   registrator = RegisterModelsDistanceMaps(
       distance_threshold=5.0,
       smooth_sigma=1.0,
       use_signed_distance=True,
       verbose=True
   )
   
   # Register using distance fields
   transform = registrator.register(
       fixed_mesh_path="reference.vtp",
       moving_mesh_path="moving.vtp"
   )
   
   # Get distance map for visualization
   distance_map = registrator.get_distance_map(
       mesh_path="reference.vtp"
   )

**Parameters**:
   * ``distance_threshold``: Maximum distance for correspondence
   * ``use_signed_distance``: Use signed vs unsigned distance
   * ``smooth_sigma``: Smoothing for distance field
   * ``allow_deformation``: Enable deformable registration

PCA-Based Registration
----------------------

Register using statistical shape models.

.. autoclass:: physiomotion4d.RegisterModelsPCA
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Population-based shape priors
   * Constrained deformations
   * Shape parameter extraction
   * Handles shape variations

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterModelsPCA
   
   # Initialize with PCA model
   registrator = RegisterModelsPCA(
       pca_model_path="heart_shape_model.pkl",
       num_modes=10,
       verbose=True
   )
   
   # Register target to model
   transform, shape_params = registrator.register(
       target_mesh_path="patient_heart.vtp"
   )
   
   # Get shape parameters
   print(f"Shape parameters: {shape_params}")
   
   # Generate mesh from parameters
   reconstructed = registrator.reconstruct_from_parameters(shape_params)
   
   # Assess shape abnormality
   abnormality_score = registrator.compute_abnormality_score(shape_params)
   print(f"Abnormality score: {abnormality_score}")

Common Usage Patterns
=====================

Initial Alignment
-----------------

Use ICP for initial rigid alignment:

.. code-block:: python

   from physiomotion4d import RegisterModelsICP
   
   # Quick initial alignment
   icp = RegisterModelsICP(max_iterations=50)
   
   initial_transform = icp.register(
       fixed_mesh_path="target.vtp",
       moving_mesh_path="source.vtp"
   )
   
   # Apply initial transform
   aligned_mesh = icp.apply_transform(
       mesh_path="source.vtp",
       transform=initial_transform
   )

Multi-Stage Registration
-------------------------

Combine multiple methods for better results:

.. code-block:: python

   # Stage 1: Rigid ICP
   icp = RegisterModelsICP()
   rigid_tfm = icp.register(fixed, moving)
   aligned = icp.apply_transform(moving, rigid_tfm)
   
   # Stage 2: Distance map deformable
   dist_reg = RegisterModelsDistanceMaps(allow_deformation=True)
   deform_tfm = dist_reg.register(fixed, aligned)
   final = dist_reg.apply_transform(aligned, deform_tfm)
   
   # Combine transforms
   combined_tfm = registrator.compose_transforms(rigid_tfm, deform_tfm)

Model-to-Image Registration
----------------------------

Register mesh model to medical image:

.. code-block:: python

   from physiomotion4d import RegisterModelsDistanceMaps
   from physiomotion4d import RegisterImagesICON
   
   # Step 1: Convert mesh to distance map image
   dist_reg = RegisterModelsDistanceMaps()
   model_distance_image = dist_reg.mesh_to_distance_image(
       mesh_path="model.vtp",
       reference_image="patient_ct.mha"
   )
   
   # Step 2: Segment patient image to get distance map
   patient_distance_image = dist_reg.create_distance_image(
       labelmap_path="patient_segmentation.mha"
   )
   
   # Step 3: Register distance images
   image_reg = RegisterImagesICON(device="cuda:0")
   transform = image_reg.register(
       fixed_image_path=patient_distance_image,
       moving_image_path=model_distance_image
   )
   
   # Step 4: Apply transform to mesh
   registered_mesh = dist_reg.transform_mesh(
       mesh_path="model.vtp",
       transform=transform
   )

Advanced Features
=================

Partial Shape Matching
----------------------

Handle incomplete or partial meshes:

.. code-block:: python

   class PartialMatchingRegistrator(RegisterModelsICP):
       """ICP with partial shape matching."""
       
       def __init__(self, overlap_ratio=0.5, **kwargs):
           super().__init__(**kwargs)
           self.overlap_ratio = overlap_ratio
       
       def register(self, fixed_mesh_path, moving_mesh_path):
           """Register with partial matching."""
           # Identify overlapping regions
           overlap_mask = self.find_overlap(
               fixed_mesh_path,
               moving_mesh_path
           )
           
           # Register only overlapping parts
           transform = self.register_partial(
               fixed_mesh_path,
               moving_mesh_path,
               overlap_mask=overlap_mask
           )
           
           return transform

Landmark-Based Initialization
------------------------------

Use anatomical landmarks for initialization:

.. code-block:: python

   # Define landmarks (e.g., apex, base points)
   fixed_landmarks = [
       [10.5, 20.3, 15.7],  # Apex
       [10.2, 45.6, 18.3],  # Base center
       [15.8, 42.1, 20.5],  # Base lateral
   ]
   
   moving_landmarks = [
       [50.1, 30.2, 25.4],
       [49.8, 55.3, 27.9],
       [55.2, 51.7, 30.1],
   ]
   
   # Initialize with landmarks
   registrator = RegisterModelsICP(use_landmarks=True)
   
   transform = registrator.register_with_landmarks(
       fixed_mesh_path="target.vtp",
       moving_mesh_path="source.vtp",
       fixed_landmarks=fixed_landmarks,
       moving_landmarks=moving_landmarks
   )

Quality Assessment
==================

Registration Error Metrics
--------------------------

Evaluate registration quality:

.. code-block:: python

   def assess_model_registration(fixed_mesh, registered_mesh):
       """Compute registration quality metrics."""
       import numpy as np
       from vtk.util.numpy_support import vtk_to_numpy
       
       # Get point coordinates
       fixed_points = vtk_to_numpy(fixed_mesh.GetPoints().GetData())
       registered_points = vtk_to_numpy(registered_mesh.GetPoints().GetData())
       
       # Compute closest point distances
       from scipy.spatial import cKDTree
       tree = cKDTree(fixed_points)
       distances, _ = tree.query(registered_points)
       
       # Compute metrics
       mean_error = np.mean(distances)
       std_error = np.std(distances)
       max_error = np.max(distances)
       rms_error = np.sqrt(np.mean(distances ** 2))
       
       return {
           'mean': mean_error,
           'std': std_error,
           'max': max_error,
           'rms': rms_error
       }

Surface Distance Visualization
-------------------------------

Visualize registration errors:

.. code-block:: python

   import pyvista as pv
   
   def visualize_registration_error(fixed_mesh, registered_mesh):
       """Visualize surface-to-surface distances."""
       # Compute distances
       errors = compute_surface_distances(fixed_mesh, registered_mesh)
       
       # Add as scalar data
       registered_mesh['registration_error'] = errors
       
       # Visualize
       plotter = pv.Plotter()
       plotter.add_mesh(
           registered_mesh,
           scalars='registration_error',
           cmap='jet',
           clim=[0, 5.0],
           show_scalar_bar=True
       )
       plotter.add_mesh(fixed_mesh, opacity=0.3, color='white')
       plotter.show()

Customization
=============

Custom Distance Metrics
-----------------------

Define custom distance metrics for ICP:

.. code-block:: python

   class CustomICPRegistrator(RegisterModelsICP):
       """ICP with custom distance metric."""
       
       def compute_correspondence(self, fixed_points, moving_points):
           """Custom correspondence computation."""
           # Implement custom metric (e.g., feature-based)
           feature_fixed = self.compute_features(fixed_points)
           feature_moving = self.compute_features(moving_points)
           
           # Find correspondences in feature space
           correspondences = self.match_features(
               feature_fixed,
               feature_moving
           )
           
           return correspondences

Constrained Registration
------------------------

Add anatomical constraints:

.. code-block:: python

   class ConstrainedModelRegistrator(RegisterModelsDistanceMaps):
       """Registration with anatomical constraints."""
       
       def __init__(self, constraint_landmarks=None, **kwargs):
           super().__init__(**kwargs)
           self.constraint_landmarks = constraint_landmarks
       
       def register(self, fixed_mesh_path, moving_mesh_path):
           """Register with constraints."""
           # Perform initial registration
           transform = super().register(fixed_mesh_path, moving_mesh_path)
           
           # Apply landmark constraints
           if self.constraint_landmarks:
               transform = self.apply_constraints(
                   transform,
                   self.constraint_landmarks
               )
           
           return transform

Best Practices
==============

Method Selection
----------------

* **ICP**: Fast, rigid alignment, good initialization
* **Distance Maps**: Handles partial data, allows deformation
* **PCA**: Uses shape priors, good for incomplete data

Preprocessing
-------------

.. code-block:: python

   # Clean and preprocess meshes
   def preprocess_mesh(mesh_path):
       """Prepare mesh for registration."""
       import pyvista as pv
       
       mesh = pv.read(mesh_path)
       
       # Remove duplicate points
       mesh = mesh.clean()
       
       # Smooth if needed
       mesh = mesh.smooth(n_iter=10)
       
       # Ensure consistent normals
       mesh.compute_normals(inplace=True)
       
       return mesh

Parameter Tuning
----------------

* Start with default parameters
* Increase iterations for difficult cases
* Use multi-stage registration (coarse to fine)
* Validate results with quality metrics

See Also
========

* :doc:`registration_images` - Register medical images
* :doc:`workflows` - Using model registration in workflows
* :doc:`utilities` - Mesh processing utilities
