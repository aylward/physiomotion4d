====================
Registration Guide
====================

Comprehensive guide to image and model registration methods in PhysioMotion4D.

Available Registration Methods
===============================

Image-to-Image Registration
----------------------------

PhysioMotion4D provides multiple registration methods for aligning medical images:

* **ICON (RegisterImagesICON)**: Deep learning-based, GPU-accelerated, diffeomorphic registration
  
  - Best for: 4D cardiac/lung CT, large deformations
  - Speed: Fast (30-60 seconds with GPU)
  - Accuracy: Excellent
  - Requirements: GPU recommended (CUDA)

* **ANTs (RegisterImagesANTs)**: Classical optimization-based registration with SyN algorithm
  
  - Best for: Brain imaging, validation studies
  - Speed: Slow (5-10 minutes)
  - Accuracy: Excellent
  - Requirements: CPU only

* **Time Series (RegisterTimeSeriesImages)**: Specialized for 4D CT sequences
  
  - Best for: Cardiac gating, respiratory motion
  - Features: Temporal smoothing, bidirectional registration
  - Backend: Uses ICON or ANTs

Model-to-Image/Model Registration
----------------------------------

For registering anatomical models to patient data:

* **ICP (RegisterModelsICP)**: Iterative Closest Point for surface alignment
  
  - Best for: Initial rough alignment, mesh registration
  - Speed: Very fast (<10 seconds)
  - Type: Rigid/affine registration

* **Mask-based (RegisterModelsDistanceMaps)**: Deformable registration using binary masks
  
  - Best for: Model-to-patient fitting
  - Features: Dice coefficient optimization
  - Type: Deformable registration

* **PCA-based (RegisterModelsPCA)**: Statistical shape model registration
  
  - Best for: Shape prior constraints
  - Features: Low-dimensional optimization
  - Type: Parametric model fitting

Complete Workflows
------------------

* **HeartModelToPatientWorkflow**: Three-stage model-to-patient registration
  
  - Stage 1: ICP rough alignment
  - Stage 2: Mask-based deformable registration
  - Stage 3: Optional image-based refinement

Quick Start Examples
====================

Image Registration with ICON
-----------------------------

.. code-block:: python

   from physiomotion4d import RegisterImagesICON
   import itk

   # Initialize
   registerer = RegisterImagesICON()
   registerer.set_modality('ct')
   
   # Load images
   fixed = itk.imread("reference.mha")
   moving = itk.imread("target.mha")
   
   registerer.set_fixed_image(fixed)
   results = registerer.register(moving)
   
   # Get displacement fields
   inverse_transform = results["inverse_transform"]  # Fixed to moving
   forward_transform = results["forward_transform"]  # Moving to fixed

Time Series Registration
-------------------------

.. code-block:: python

   from physiomotion4d import RegisterTimeSeriesImages

   # Initialize with ICON backend
   reg = RegisterTimeSeriesImages(
       reference_index=0,
       registration_method='icon'
   )
   
   # Register sequence
   image_files = [f"frame_{i:03d}.mha" for i in range(20)]
   results = reg.register_time_series(image_filenames=image_files)
   
   # Access transforms
   transforms_inverse = results["inverse_transforms"]
   transforms_forward = results["forward_transforms"]

Model to Patient Registration
------------------------------

.. code-block:: python

   from physiomotion4d import HeartModelToPatientWorkflow
   import pyvista as pv
   import itk

   # Load data
   model = pv.read("generic_heart.vtu")
   patient_surfaces = [pv.read("lv.stl"), pv.read("rv.stl")]
   patient_image = itk.imread("patient_ct.nii.gz")

   # Initialize workflow
   workflow = HeartModelToPatientWorkflow(
       moving_mesh=model,
       fixed_meshes=patient_surfaces,
       fixed_image=patient_image
   )
   
   # Run complete registration
   registered = workflow.run_workflow()

Choosing the Right Method
==========================

For 4D Cardiac/Lung CT
----------------------

**Recommended**: RegisterImagesICON or RegisterTimeSeriesImages with ICON backend

- Fast processing with GPU
- Excellent accuracy for large cardiac/respiratory motion
- Diffeomorphic (topology-preserving) transforms

For Model-to-Patient Fitting
-----------------------------

**Recommended**: HeartModelToPatientWorkflow

- Combines multiple registration stages
- Robust to initialization
- Handles both geometric and intensity information

For Validation Studies
----------------------

**Recommended**: RegisterImagesANTs

- Well-validated classical method
- CPU-only (no GPU required)
- Gold standard for comparison

Best Practices
==============

1. **Preprocessing**
   
   - Normalize image intensities
   - Apply appropriate masking
   - Verify image orientation and spacing

2. **Parameter Tuning**
   
   - Start with default parameters
   - Adjust smoothness/regularization for motion magnitude
   - Use multi-resolution for large deformations

3. **Quality Assessment**
   
   - Visualize deformation fields
   - Check inverse consistency errors
   - Validate with anatomical landmarks

4. **Performance Optimization**
   
   - Use GPU for ICON registration
   - Downsample for initial alignment
   - Process regions of interest only

Advanced Topics
===============

See the following for more details:

* :doc:`../api/registration` - Complete API reference
* :doc:`../tutorials/image_registration` - Step-by-step tutorials
* :doc:`../tutorials/model_to_image_registration` - Model fitting guide
* :doc:`../examples` - Practical examples

Troubleshooting
===============

**Registration produces poor results:**

- Check image preprocessing and masking
- Adjust regularization parameters
- Try different initialization
- Verify image quality and contrast

**Registration is too slow:**

- Use GPU acceleration for ICON
- Reduce image resolution
- Process smaller regions of interest
- Consider faster methods (ICP for initialization)

**Out of memory errors:**

- Reduce batch size
- Downsample images
- Use CPU instead of GPU
- Process smaller ROI

