====================================
Image Registration Development Guide
====================================

This guide covers developing with image registration methods.

For complete API documentation, see :doc:`../api/registration/index`.

Overview
========

The image registration module provides deformable and rigid registration methods for aligning medical images across time or between subjects.

Overview
========

PhysioMotion4D supports multiple image registration approaches:

* **ICON**: Deep learning-based deformable registration (GPU-accelerated)
* **ANTs**: Traditional optimization-based registration (ANTsPy wrapper)
* **Time Series**: Specialized registration for 4D medical images

All registration classes inherit from :class:`RegisterImagesBase` and provide consistent interfaces.

Base Registration Class
=======================

RegisterImagesBase
------------------

Abstract base class for all image registration methods.

.. autoclass:: physiomotion4d.RegisterImagesBase
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods**:
   * ``register(fixed, moving)``: Register two images
   * ``apply_transform(image, transform)``: Apply transform to image
   * ``invert_transform(transform)``: Invert transformation
   * ``save_transform(transform, filename)``: Save transform to disk
   * ``load_transform(filename)``: Load transform from disk

Registration Methods
====================

ICON Registration
-----------------

Deep learning-based deformable registration with mass preservation.

.. autoclass:: physiomotion4d.RegisterImagesICON
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * GPU-accelerated neural network
   * Mass-preserving deformations
   * Fast inference (~5-10 seconds per pair)
   * Pre-trained on medical images
   * Suitable for cardiac and respiratory motion

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterImagesICON
   
   # Initialize ICON registrator
   registrator = RegisterImagesICON(
       device="cuda:0",
       network_weights="path/to/weights.trch",
       iterations=1,
       verbose=True
   )
   
   # Register images
   transform = registrator.register(
       fixed_image_path="reference.mha",
       moving_image_path="frame_01.mha"
   )
   
   # Apply transform to moving image
   registered = registrator.apply_transform(
       image_path="frame_01.mha",
       transform=transform
   )
   
   # Save transform
   registrator.save_transform(transform, "transform.hdf5")
   
   # Get displacement field
   displacement_field = registrator.get_displacement_field(transform)

**Parameters**:
   * ``device``: CPU or CUDA device
   * ``network_weights``: Path to pre-trained weights
   * ``iterations``: Refinement iterations (default: 1)
   * ``smooth_sigma``: Smoothing for regularization
   * ``lambda_weight``: Mass preservation weight

ANTs Registration
-----------------

Traditional optimization-based registration using ANTsPy.

.. autoclass:: physiomotion4d.RegisterImagesANTs
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Symmetric normalization (SyN)
   * Multiple similarity metrics
   * Rigid, affine, and deformable stages
   * CPU-based optimization
   * Highly accurate for diverse anatomy

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterImagesANTs
   
   # Initialize ANTs registrator
   registrator = RegisterImagesANTs(
       type_of_transform='SyN',  # Symmetric normalization
       iterations=[100, 70, 50, 20],
       shrink_factors=[8, 4, 2, 1],
       smoothing_sigmas=[3, 2, 1, 0],
       verbose=True
   )
   
   # Register with mutual information
   transform = registrator.register(
       fixed_image_path="reference.mha",
       moving_image_path="moving.mha",
       metric='MI'  # Mutual information
   )
   
   # Apply transform
   registered = registrator.apply_transform(
       image_path="moving.mha",
       transform=transform
   )
   
   # Get forward and inverse transforms
   forward_tfm = registrator.get_forward_transform(transform)
   inverse_tfm = registrator.get_inverse_transform(transform)

**Transform Types**:
   * ``Rigid``: Translation + rotation
   * ``Affine``: Rigid + scaling + shearing
   * ``SyN``: Symmetric deformable
   * ``SyNOnly``: Deformable without initial alignment

**Similarity Metrics**:
   * ``MI``: Mutual information (for multi-modal)
   * ``CC``: Cross-correlation (for same modality)
   * ``Demons``: Demons algorithm
   * ``MeanSquares``: Mean squared difference

Time Series Registration
------------------------

Specialized registration for 4D medical image sequences.

.. autoclass:: physiomotion4d.RegisterTimeSeriesImages
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Register multiple frames to reference
   * Temporal smoothness constraints
   * Parallel processing support
   * Handles missing frames

**Example Usage**:

.. code-block:: python

   from physiomotion4d import RegisterTimeSeriesImages
   
   # Initialize time series registrator
   registrator = RegisterTimeSeriesImages(
       registration_method='icon',  # or 'ants'
       reference_index=7,  # Use frame 7 as reference (70%)
       temporal_smoothing=True,
       verbose=True
   )
   
   # Register time series
   transforms = registrator.register_series(
       image_files=[
           "frame_00.mha",
           "frame_01.mha",
           "frame_02.mha",
           # ... more frames
       ]
   )
   
   # Apply transforms to segmentation
   registered_labels = registrator.transform_labelmap_series(
       labelmap_path="reference_labels.mha",
       transforms=transforms
   )
   
   # Get motion statistics
   motion_magnitude = registrator.compute_motion_magnitude(transforms)
   print(f"Average motion: {motion_magnitude.mean()} mm")

Common Usage Patterns
=====================

Basic Registration
------------------

Simple pairwise registration:

.. code-block:: python

   from physiomotion4d import RegisterImagesICON
   
   # Initialize
   registrator = RegisterImagesICON(device="cuda:0")
   
   # Register
   transform = registrator.register(
       fixed_image_path="fixed.mha",
       moving_image_path="moving.mha"
   )
   
   # Apply to image
   registered_img = registrator.apply_transform(
       image_path="moving.mha",
       transform=transform
   )
   
   # Apply to segmentation
   registered_seg = registrator.apply_transform(
       image_path="moving_labels.mha",
       transform=transform,
       interpolation='nearest'  # For labels
   )

Multi-Modal Registration
------------------------

Register images from different modalities:

.. code-block:: python

   from physiomotion4d import RegisterImagesANTs
   
   # Use mutual information for multi-modal
   registrator = RegisterImagesANTs(
       type_of_transform='Affine',
       verbose=True
   )
   
   # Register CT to MRI
   transform = registrator.register(
       fixed_image_path="mri.nii",
       moving_image_path="ct.nii",
       metric='MI'  # Mutual information
   )

Masked Registration
-------------------

Register using region of interest masks:

.. code-block:: python

   # Register only within heart region
   registrator = RegisterImagesICON(device="cuda:0")
   
   transform = registrator.register(
       fixed_image_path="fixed.mha",
       moving_image_path="moving.mha",
       fixed_mask_path="heart_mask_fixed.mha",
       moving_mask_path="heart_mask_moving.mha"
   )

Advanced Features
=================

Registration Quality Assessment
-------------------------------

Evaluate registration quality:

.. code-block:: python

   def assess_registration(fixed, moving, transform, registrator):
       """Assess registration quality."""
       import numpy as np
       
       # Apply transform
       registered = registrator.apply_transform(moving, transform)
       
       # Compute similarity metrics
       mse = np.mean((fixed - registered) ** 2)
       correlation = np.corrcoef(fixed.flatten(), registered.flatten())[0, 1]
       
       # Compute deformation statistics
       disp_field = registrator.get_displacement_field(transform)
       max_deformation = np.max(np.linalg.norm(disp_field, axis=-1))
       mean_deformation = np.mean(np.linalg.norm(disp_field, axis=-1))
       
       return {
           'mse': mse,
           'correlation': correlation,
           'max_deformation': max_deformation,
           'mean_deformation': mean_deformation
       }

Iterative Refinement
--------------------

Refine registration iteratively:

.. code-block:: python

   registrator = RegisterImagesICON(device="cuda:0")
   
   # Initial registration
   transform = registrator.register(fixed, moving)
   
   # Refine with higher resolution
   registrator.set_resolution(high_res=True)
   transform = registrator.refine_registration(
       fixed, moving, initial_transform=transform
   )
   
   # Further refinement
   for i in range(3):
       transform = registrator.refine_registration(
           fixed, moving, initial_transform=transform
       )

Customization
=============

Custom Loss Functions
---------------------

Define custom similarity metrics:

.. code-block:: python

   from physiomotion4d import RegisterImagesICON
   import torch
   
   class CustomIconRegistrator(RegisterImagesICON):
       """ICON with custom loss function."""
       
       def custom_similarity_loss(self, fixed, moving):
           """Custom similarity metric."""
           # Combine multiple metrics
           mse_loss = torch.mean((fixed - moving) ** 2)
           grad_loss = self.gradient_difference_loss(fixed, moving)
           
           return mse_loss + 0.1 * grad_loss
       
       def gradient_difference_loss(self, fixed, moving):
           """Penalize gradient differences."""
           # Compute gradients
           grad_fixed = self.compute_gradient(fixed)
           grad_moving = self.compute_gradient(moving)
           
           return torch.mean((grad_fixed - grad_moving) ** 2)

Custom Transform Constraints
-----------------------------

Add constraints to deformations:

.. code-block:: python

   class ConstrainedRegistrator(RegisterImagesICON):
       """Registration with deformation constraints."""
       
       def __init__(self, max_deformation=10.0, **kwargs):
           super().__init__(**kwargs)
           self.max_deformation = max_deformation
       
       def apply_constraints(self, displacement_field):
           """Limit maximum deformation."""
           import numpy as np
           
           # Compute deformation magnitude
           magnitude = np.linalg.norm(displacement_field, axis=-1)
           
           # Clip excessive deformations
           scale = np.minimum(1.0, self.max_deformation / (magnitude + 1e-8))
           constrained_field = displacement_field * scale[..., np.newaxis]
           
           return constrained_field

Performance Optimization
========================

GPU Acceleration
----------------

Optimize GPU usage for ICON:

.. code-block:: python

   import torch
   
   # Check GPU memory
   if torch.cuda.is_available():
       gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
       print(f"GPU memory: {gpu_mem:.1f} GB")
       
       # Adjust batch size based on memory
       if gpu_mem > 16:
           batch_size = 8
       else:
           batch_size = 4
   
   # Initialize with optimal settings
   registrator = RegisterImagesICON(
       device="cuda:0",
       batch_size=batch_size,
       mixed_precision=True  # Use FP16 for speed
   )

Parallel Processing
-------------------

Register multiple pairs in parallel:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   from physiomotion4d import RegisterImagesICON
   
   def register_pair(fixed_file, moving_file, output_dir):
       """Register a single pair."""
       registrator = RegisterImagesICON(device="cuda:0")
       transform = registrator.register(fixed_file, moving_file)
       registrator.save_transform(transform, f"{output_dir}/transform.hdf5")
       return transform
   
   def parallel_registration(pairs):
       """Register multiple pairs in parallel."""
       with ThreadPoolExecutor(max_workers=4) as executor:
           futures = [
               executor.submit(register_pair, p['fixed'], p['moving'], p['output'])
               for p in pairs
           ]
           results = [f.result() for f in futures]
       return results

Best Practices
==============

Method Selection
----------------

* **ICON**: Fast, GPU-based, good for cardiac/respiratory motion
* **ANTs**: High accuracy, CPU-based, versatile for all anatomy
* **Time Series**: Use for 4D sequences with temporal coherence

Parameter Tuning
----------------

**ICON**:
   * Start with ``iterations=1`` (usually sufficient)
   * Increase for difficult registrations
   * Adjust ``lambda_weight`` for mass preservation

**ANTs**:
   * Use multi-resolution (coarse to fine)
   * More iterations for difficult cases
   * Choose appropriate metric for modality

Quality Control
---------------

.. code-block:: python

   # Always assess registration quality
   quality = assess_registration(fixed, moving, transform, registrator)
   
   if quality['correlation'] < 0.8:
       print("Warning: Low registration quality")
       # Try alternative method or parameters

See Also
========

* :doc:`registration_models` - Register mesh models
* :doc:`workflows` - Using registration in workflows
* :doc:`segmentation` - Segment images before registration
