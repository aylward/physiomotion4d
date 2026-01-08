===========================
Registration API Reference
===========================

PhysioMotion4D provides multiple image registration methods, including deep learning-based
and classical optimization-based approaches.

Base Registration Class
========================

.. autoclass:: physiomotion4d.RegisterImagesBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Abstract base class for image registration implementations. All registration
   methods inherit from this class.

   **Common Interface:**

   All registration classes provide:

   * :meth:`set_fixed_image`: Set reference/target image
   * :meth:`set_moving_image`: Set image to be registered
   * :meth:`set_modality`: Set imaging modality (CT, MRI, etc.)
   * :meth:`register`: Perform registration
   * :meth:`get_transform`: Retrieve computed transformation

Deep Learning Registration
===========================

ICON (Inverse Consistent Optimization Network)
-----------------------------------------------

.. autoclass:: physiomotion4d.RegisterImagesICON
   :members:
   :undoc-members:
   :show-inheritance:

   GPU-accelerated deep learning registration using the ICON algorithm.
   Provides fast, accurate, and diffeomorphic registration.

   **Features:**

   * GPU acceleration (5-10x faster than classical methods)
   * Diffeomorphic (topology-preserving) transforms
   * Inverse consistency built-in
   * Pre-trained on medical imaging datasets
   * Supports multi-resolution registration

   **Example:**

   .. code-block:: python

      from physiomotion4d.register_images_icon import RegisterImagesICON
      import itk

      # Initialize registration
      registerer = RegisterImagesICON()
      
      # Load images
      fixed_image = itk.imread("reference.mha")
      moving_image = itk.imread("target.mha")
      
      # Configure
      registerer.set_modality('ct')
      registerer.set_fixed_image(fixed_image)
      registerer.set_iterations(100)
      registerer.set_device('cuda')  # or 'cpu'
      
      # Register
      results = registerer.register(moving_image)
      
      # Get results
      forward_transform = results["forward_transform"]  # Forward deformation field
      inverse_transform = results["inverse_transform"]  # Inverse deformation field
      registered_image = results["registered_image"]
      similarity = results["similarity_score"]

   **Output Dictionary:**

   * ``forward_transform``: Used to warp an image from moving to fixed space
   * ``inverse_transform``: Used to warp an image from fixed to moving space
   * ``registered_image``: Moving image warped to fixed space
   * ``similarity_score``: Registration quality metric
   * ``inverse_consistency_error``: Inverse consistency metric

   **Advanced Options:**

   .. code-block:: python

      registerer = RegisterImagesICON()
      
      # Multi-resolution pyramid
      registerer.set_pyramid_levels([4, 2, 1])
      
      # Regularization
      registerer.set_smoothness_weight(0.5)
      
      # Loss function
      registerer.set_similarity_metric('ncc')  # or 'mse', 'mi'
      
      # Memory optimization
      registerer.set_batch_size(1)

Classical Registration Methods
===============================

ANTs Registration
-----------------

.. autoclass:: physiomotion4d.RegisterImagesANTs
   :members:
   :undoc-members:
   :show-inheritance:

   Advanced Normalization Tools (ANTs) based registration. Provides state-of-the-art
   classical registration algorithms.

   **Features:**

   * Symmetric normalization (SyN)
   * Diffeomorphic registration
   * Multi-stage optimization
   * Robust to initialization

   **Example:**

   .. code-block:: python

      from physiomotion4d import RegisterImagesANTs
      import itk

      registerer = RegisterImagesANTs()
      
      fixed_image = itk.imread("reference.mha")
      moving_image = itk.imread("target.mha")
      
      registerer.set_fixed_image(fixed_image)
      registerer.set_transform_type('SyN')  # Symmetric Normalization
      
      results = registerer.register(moving_image)

Time Series Registration
=========================

.. autoclass:: physiomotion4d.RegisterTimeSeriesImages
   :members:
   :undoc-members:
   :show-inheritance:

   Register an ordered sequence of images (time series) to a fixed reference image.
   Supports both ANTs and ICON registration backends with optional temporal smoothing.

   **Features:**

   * Sequential registration with bidirectional processing
   * Support for ANTs and ICON backends
   * Optional temporal coherence via prior transform propagation
   * Configurable starting point in the time series
   * Returns all transforms and loss values

   **Example:**

   .. code-block:: python

      from physiomotion4d import RegisterTimeSeriesImages
      import itk

      # Initialize with ANTs backend
      registerer = RegisterTimeSeriesImages(registration_method='ants')
      
      # Load reference and time series images
      fixed_image = itk.imread("reference_frame.mha")
      time_series = [itk.imread(f"frame_{i:03d}.mha") for i in range(20)]
      
      # Configure
      registerer.set_modality('ct')
      registerer.set_fixed_image(fixed_image)
      registerer.set_number_of_iterations([30, 15, 5])
      
      # Register time series
      results = registerer.register_time_series(
          moving_images=time_series,
          starting_index=10,  # Start from middle frame
          register_start_to_reference=True,
          portion_of_prior_transform_to_init_next_transform=0.5
      )
      
      # Access results
      forward_transforms_list = results["forward_transforms"]  # List of transforms
      inverse_transforms_list = results["inverse_transforms"]  # List of inverse transforms
      registration_losses = results["losses"]  # List of registration losses

   **Use Cases:**

   * 4D cardiac CT reconstruction
   * Respiratory motion tracking
   * Dynamic contrast-enhanced imaging
   * Any ordered sequence of medical images

   **Parameters Explained:**

   * ``starting_index``: Frame to begin registration (typically mid-cycle)
   * ``register_start_to_reference``: If True, register starting frame; if False, use identity
   * ``portion_of_prior_transform_to_init_next_transform``: Weight for temporal smoothing (0.0-1.0)

     * 0.0: No temporal smoothing (each frame registered independently)
     * 0.5: Moderate smoothing (good for cardiac CT)
     * 1.0: Maximum smoothing (strong temporal coherence)

Model-to-Image Registration
============================

These methods register statistical shape models or segmentation meshes to medical images.

PCA-based Registration
----------------------

.. autoclass:: physiomotion4d.RegisterModelsPCA
   :members:
   :undoc-members:
   :show-inheritance:

   Register statistical shape models using PCA-based optimization.

   **Features:**

   * Shape prior from statistical model
   * PCA parameter optimization
   * Efficient low-dimensional search
   * Robust to poor initialization

   **Example:**

   .. code-block:: python

      from physiomotion4d import RegisterModelsPCA
      import itk

      registerer = RegisterModelsPCA()
      
      # Load shape model and image
      shape_model = load_statistical_shape_model("heart_model.pkl")
      image = itk.imread("patient_ct.nrrd")
      
      registerer.set_shape_model(shape_model)
      registerer.set_target_image(image)
      
      # Register
      fitted_mesh = registerer.register()

Combined Model-Patient Workflow
--------------------------------

.. autoclass:: physiomotion4d.HeartModelToPatientWorkflow
   :members:
   :undoc-members:
   :show-inheritance:

   Complete three-stage registration workflow for fitting generic anatomical models to 
   patient-specific data. Combines ICP, mask-based registration, and optional image-based 
   refinement for robust model-to-patient registration.

   **Example:**

   .. code-block:: python

      from physiomotion4d import HeartModelToPatientWorkflow
      import pyvista as pv
      import itk

      # Load generic model and patient data
      model_mesh = pv.read("generic_heart_model.vtu")
      patient_surfaces = [pv.read("patient_lv.stl"), pv.read("patient_rv.stl")]
      reference_image = itk.imread("patient_ct.nii.gz")

      # Initialize workflow
      workflow = HeartModelToPatientWorkflow(
          moving_mesh=model_mesh,
          fixed_meshes=patient_surfaces,
          fixed_image=reference_image
      )

      # Run complete three-stage registration
      registered_mesh = workflow.run_workflow()

Model-to-Model Registration
============================

ICP Registration
----------------

.. autoclass:: physiomotion4d.RegisterModelsICP
   :members:
   :undoc-members:
   :show-inheritance:

   Iterative Closest Point (ICP) registration for surface meshes.

   **Features:**

   * Point cloud alignment
   * Rigid and non-rigid variants
   * Outlier rejection
   * Fast convergence

   **Example:**

   .. code-block:: python

      from physiomotion4d import RegisterModelsICP
      import pyvista as pv

      # Load meshes
      fixed_mesh = pv.read("reference_mesh.vtp")
      moving_mesh = pv.read("target_mesh.vtp")
      
      # Initialize registrar
      registrar = RegisterModelsICP(
          moving_mesh=moving_mesh,
          fixed_mesh=fixed_mesh
      )
      
      # Register
      result = registrar.register(mode='rigid', max_iterations=100)
      aligned_mesh = result['moving_mesh']
      forward_point_transform = result['forward_point_transform']
      inverse_point_transform = result['inverse_point_transform']

Mask-based Registration
-----------------------

.. autoclass:: physiomotion4d.RegisterModelsDistanceMaps
   :members:
   :undoc-members:
   :show-inheritance:

   Register models using binary mask overlap optimization.

   **Features:**

   * Segmentation mask alignment
   * Dice coefficient optimization
   * Robust to mesh quality
   * Handles topology differences

Registration Comparison
=======================

.. list-table:: Registration Method Comparison
   :header-rows: 1
   :widths: 25 15 15 20 25

   * - Method
     - Speed
     - Accuracy
     - GPU Required
     - Best Use Case
   * - ICON
     - Fast (30-60s)
     - Excellent
     - Recommended
     - 4D CT, cardiac motion, large deformations
   * - ANTs
     - Slow (5-10min)
     - Excellent
     - No
     - Brain imaging, research validation
   * - ICP
     - Fast (<10s)
     - Good
     - No
     - Mesh alignment, initialization
   * - PCA Model
     - Fast (10-30s)
     - Good
     - No
     - Shape models, prior knowledge

Evaluation Metrics
==================

PhysioMotion4D provides utilities for evaluating registration quality:

.. code-block:: python

   from physiomotion4d.registration_metrics import (
       compute_dice_coefficient,
       compute_target_registration_error,
       compute_inverse_consistency_error,
       compute_jacobian_determinant
   )
   
   # Dice coefficient (segmentation overlap)
   dice = compute_dice_coefficient(fixed_mask, warped_moving_mask)
   
   # Target Registration Error (landmark-based)
   tre = compute_target_registration_error(fixed_landmarks, warped_landmarks)
   
   # Inverse consistency (bidirectional registration)
   ice = compute_inverse_consistency_error(inverse_transform, forward_transform)
   
   # Jacobian determinant (topology preservation)
   jac_det = compute_jacobian_determinant(inverse_transform)
   folding_points = jac_det < 0  # Locations with folding

Best Practices
==============

Choosing a Registration Method
-------------------------------

1. **For 4D cardiac/lung CT**:

   * Use :class:`RegisterImagesICON` for speed and accuracy
   * GPU recommended for large datasets

2. **For brain imaging and general purpose**:

   * Use :class:`RegisterImagesANTs` for validated results

3. **For initialization/coarse alignment**:

   * Start with :class:`RegisterModelsICP`
   * Then refine with image-based registration

4. **For shape model fitting**:

   * Use :class:`RegisterModelsPCA`
   * Especially when prior knowledge available

Parameter Selection
-------------------

.. code-block:: python

   # ICON parameters for cardiac CT
   registerer = RegisterImagesICON()
   registerer.set_iterations(100)  # 50-200 typical
   registerer.set_smoothness_weight(0.5)  # 0.1-1.0
   registerer.set_pyramid_levels([4, 2, 1])  # Coarse to fine
   
   # For respiratory motion (larger deformations)
   registerer.set_smoothness_weight(0.3)  # Lower = more flexible
   registerer.set_iterations(150)  # More iterations

   # For subtle cardiac motion
   registerer.set_smoothness_weight(0.7)  # Higher = smoother
   registerer.set_iterations(75)  # Fewer iterations sufficient

Troubleshooting Registration
-----------------------------

**Poor registration quality:**

1. Check image preprocessing (normalization, masking)
2. Adjust regularization parameters
3. Use multi-resolution approach
4. Verify image orientation and spacing

**Registration diverges:**

1. Reduce smoothness weight
2. Increase number of iterations
3. Start with coarser pyramid level
4. Pre-align with rigid registration

**Out of memory:**

1. Reduce image size with downsampling
2. Use CPU instead of GPU
3. Process smaller regions of interest
4. Reduce batch size

See Also
========

* :doc:`../tutorials/image_registration` - Detailed registration tutorial
* :doc:`../user_guide/registration` - User guide for registration
* :doc:`../examples` - Registration examples

