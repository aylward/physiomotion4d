==========================
Basic Workflow Tutorial
==========================

This tutorial walks you through a complete PhysioMotion4D workflow, from raw 4D CT data
to an animated USD model ready for visualization in NVIDIA Omniverse.

Overview
========

In this tutorial, you will:

1. Prepare 4D cardiac CT data
2. Convert 4D NRRD to 3D time frames
3. Register cardiac phases
4. Generate AI-based segmentation
5. Transform contours across time
6. Create animated USD model
7. Visualize in Omniverse

Prerequisites
=============

* PhysioMotion4D installed (see :doc:`../installation`)
* 4D cardiac CT data (NRRD format) or use sample data
* NVIDIA GPU with CUDA 12.6+
* NVIDIA Omniverse (optional, for visualization)

Tutorial Dataset
================

We'll use sample cardiac CT data from the Slicer-Heart-CT collection.

Download Sample Data
--------------------

.. code-block:: python

   import urllib.request
   import os

   # Create data directory
   os.makedirs("tutorial_data", exist_ok=True)

   # Download sample 4D cardiac CT
   # Replace with actual download URL
   url = "https://example.com/cardiac_4d_sample.nrrd"
   output_file = "tutorial_data/cardiac_4d.nrrd"
   
   print("Downloading sample data...")
   urllib.request.urlretrieve(url, output_file)
   print(f"Downloaded to: {output_file}")

Or use your own 4D cardiac CT data in NRRD format.

Step 1: Setup and Initialization
==================================

Create a Python script or Jupyter notebook:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT
   import os

   # Define paths
   input_file = "tutorial_data/cardiac_4d.nrrd"
   output_dir = "tutorial_output"
   project_name = "my_first_cardiac_model"

   # Create output directory
   os.makedirs(output_dir, exist_ok=True)

   # Initialize processor
   processor = ProcessHeartGatedCT(
       input_filenames=[input_file],
       contrast_enhanced=True,  # Set False if non-contrast CT
       output_directory=output_dir,
       project_name=project_name,
       verbose=True  # Enable progress messages
   )

   print("Processor initialized successfully!")

Step 2: Convert 4D to 3D Frames
================================

The first step is to extract individual 3D volumes from the 4D dataset:

.. code-block:: python

   # Convert 4D NRRD to 3D time frames
   print("\nStep 1: Converting 4D to 3D frames...")
   processor.convert_4d_to_3d()

   # Check output
   frame_files = processor.get_frame_filenames()
   print(f"Extracted {len(frame_files)} time frames")
   for i, frame in enumerate(frame_files[:3]):  # Show first 3
       print(f"  Frame {i}: {frame}")

**Expected Output:**

.. code-block:: text

   Step 1: Converting 4D to 3D frames...
   Extracted 10 time frames
     Frame 0: tutorial_output/frames/frame_000.mha
     Frame 1: tutorial_output/frames/frame_001.mha
     Frame 2: tutorial_output/frames/frame_002.mha

**What's Happening:**

* 4D NRRD file is split into individual 3D volumes
* Each volume represents one cardiac phase
* Files are saved as MetaImage (.mha) format
* Frame 0 is typically end-diastole (reference phase)

Step 3: Register Cardiac Phases
================================

Register all frames to the reference (frame 0) using deep learning:

.. code-block:: python

   print("\nStep 2: Registering cardiac phases...")
   
   # Configure registration
   processor.set_registration_method('icon')  # Use ICON deep learning
   processor.set_registration_iterations(100)
   processor.set_registration_device('cuda')  # or 'cpu'
   
   # Perform registration
   processor.register_images()

   print("Registration complete!")
   
   # Check registration quality
   metrics = processor.get_registration_metrics()
   for i, metric in enumerate(metrics):
       print(f"  Frame {i+1}: similarity = {metric['similarity']:.3f}, "
             f"inverse consistency = {metric['ic_error']:.4f}")

**Expected Output:**

.. code-block:: text

   Step 2: Registering cardiac phases...
   Registering frame 1/9...
   Registering frame 2/9...
   ...
   Registration complete!
     Frame 1: similarity = 0.945, inverse consistency = 0.0023
     Frame 2: similarity = 0.932, inverse consistency = 0.0031
     ...

**What's Happening:**

* Each frame is registered to the reference frame 0
* ICON computes forward and inverse deformation fields
* Higher similarity score = better registration
* Lower inverse consistency error = more accurate transforms

**Troubleshooting:**

If registration is slow or fails:

.. code-block:: python

   # Reduce iterations
   processor.set_registration_iterations(50)
   
   # Use CPU if GPU out of memory
   processor.set_registration_device('cpu')
   
   # Downsample large images
   processor.set_downsample_factor(2)

Step 4: Generate Segmentation
==============================

Use AI to segment anatomical structures in the reference frame:

.. code-block:: python

   print("\nStep 3: Generating segmentation...")
   
   # Configure segmentation
   processor.set_segmentation_method('vista3d')  # or 'totalsegmentator'
   processor.set_segmentation_device('cuda')
   
   # Perform segmentation
   processor.segment_reference_image()
   
   print("Segmentation complete!")
   
   # Check generated masks
   masks = processor.get_segmentation_masks()
   print(f"Generated {len(masks)} anatomical masks:")
   for name, mask_file in masks.items():
       print(f"  {name}: {mask_file}")

**Expected Output:**

.. code-block:: text

   Step 3: Generating segmentation...
   Loading VISTA-3D model...
   Segmenting reference image...
   Segmentation complete!
   Generated 8 anatomical masks:
     heart: tutorial_output/masks/heart_mask.nrrd
     vessels: tutorial_output/masks/vessels_mask.nrrd
     lungs: tutorial_output/masks/lungs_mask.nrrd
     bones: tutorial_output/masks/bones_mask.nrrd
     ...

**What's Happening:**

* AI model segments the reference frame
* Generates masks for heart, vessels, lungs, bones, etc.
* Masks are binary (0 = background, 1 = structure)
* Dynamic structures (heart, lungs) will be animated

**Segmentation Options:**

.. code-block:: python

   # Use TotalSegmentator (faster, less memory)
   processor.set_segmentation_method('totalsegmentator')
   
   # Use ensemble (best quality, slowest)
   processor.set_segmentation_method('ensemble')
   
   # Use NVIDIA NIM (cloud-based)
   processor.set_segmentation_method('vista3d_nim')

Step 5: Transform Contours
===========================

Propagate reference segmentation to all time frames:

.. code-block:: python

   print("\nStep 4: Transforming contours across time...")
   
   # Extract surfaces from masks
   processor.extract_surfaces_from_masks()
   
   # Apply transforms to propagate contours
   processor.transform_contours()
   
   print("Contour transformation complete!")
   
   # Check generated contours
   contours = processor.get_contour_files()
   print(f"Generated {len(contours)} contour files per structure")

**Expected Output:**

.. code-block:: text

   Step 4: Transforming contours across time...
   Extracting surface from heart mask...
   Extracting surface from vessels mask...
   ...
   Transforming heart contour to frame 1...
   Transforming heart contour to frame 2...
   ...
   Contour transformation complete!
   Generated 10 contour files per structure

**What's Happening:**

* Segmentation masks converted to 3D surface meshes
* Reference meshes warped to each time frame using registration
* Creates time-varying geometry for animation
* Static structures (bones) are not transformed

Step 6: Create USD Models
==========================

Convert VTK meshes to USD format for Omniverse:

.. code-block:: python

   print("\nStep 5: Creating USD models...")
   
   # Configure USD export
   processor.set_frame_rate(30)  # FPS for animation
   processor.set_apply_materials(True)  # Add anatomical materials
   processor.set_flatten_usd(True)  # Flatten for better performance
   
   # Create USD files
   final_usd = processor.create_usd_models()
   
   print(f"USD model created: {final_usd}")

**Expected Output:**

.. code-block:: text

   Step 5: Creating USD models...
   Converting heart contours to USD...
   Converting vessels contours to USD...
   Converting lungs contours to USD...
   Applying materials...
   Merging USD files...
   USD model created: tutorial_output/my_first_cardiac_model.usd

**What's Happening:**

* VTK meshes converted to USD PolyMesh
* Time samples added for animation
* Anatomical materials applied (red heart, pink lungs, etc.)
* All structures merged into single USD file

Step 7: Complete Workflow (All Steps)
======================================

Or run all steps at once:

.. code-block:: python

   from physiomotion4d import ProcessHeartGatedCT

   processor = ProcessHeartGatedCT(
       input_filenames=["tutorial_data/cardiac_4d.nrrd"],
       contrast_enhanced=True,
       output_directory="tutorial_output",
       project_name="my_first_cardiac_model"
   )

   # Run complete workflow
   final_usd = processor.process()
   
   print(f"\n✓ Complete! Generated: {final_usd}")

**Expected Runtime:**

* Convert 4D to 3D: ~1 minute
* Registration: ~5-10 minutes (10 frames, GPU)
* Segmentation: ~1 minute (VISTA-3D, GPU)
* Transform contours: ~2 minutes
* Create USD: ~1 minute
* **Total: ~10-15 minutes**

Step 8: Visualize Results
==========================

In NVIDIA Omniverse
-------------------

1. Open NVIDIA Omniverse
2. Launch **USD Composer** or **USD Presenter**
3. **File → Open** → Select ``tutorial_output/my_first_cardiac_model.usd``
4. Press **Play** button to view cardiac cycle animation
5. Adjust camera and lighting as desired

Using usdview
-------------

.. code-block:: bash

   # Command-line USD viewer
   usdview tutorial_output/my_first_cardiac_model.usd

Quick Preview with PyVista
---------------------------

.. code-block:: python

   import pyvista as pv
   import glob

   # Load a time frame
   mesh = pv.read("tutorial_output/contours/heart/heart_000.vtp")
   
   # Visualize
   mesh.plot(color='red', opacity=0.8)

   # Or animate the sequence
   plotter = pv.Plotter()
   mesh_files = sorted(glob.glob("tutorial_output/contours/heart/heart_*.vtp"))
   
   for i, mesh_file in enumerate(mesh_files):
       mesh = pv.read(mesh_file)
       plotter.clear()
       plotter.add_mesh(mesh, color='red')
       plotter.show(auto_close=False, interactive_update=True)

Intermediate Files
==================

The workflow generates several intermediate files:

.. code-block:: text

   tutorial_output/
   ├── frames/                    # 3D time frames
   │   ├── frame_000.mha
   │   ├── frame_001.mha
   │   └── ...
   ├── transforms/                # Registration transforms
   │   ├── inverse_transform_001.mha
   │   ├── forward_transform_001.mha
   │   └── ...
   ├── masks/                     # Segmentation masks
   │   ├── heart_mask.nrrd
   │   ├── vessels_mask.nrrd
   │   └── ...
   ├── contours/                  # Surface meshes
   │   ├── heart/
   │   │   ├── heart_000.vtp
   │   │   ├── heart_001.vtp
   │   │   └── ...
   │   └── vessels/
   │       └── ...
   ├── usd/                       # USD files
   │   ├── heart.usd
   │   ├── vessels.usd
   │   └── ...
   └── my_first_cardiac_model.usd # Final merged USD

Customization
=============

Adjust Quality Settings
------------------------

.. code-block:: python

   # Higher quality segmentation
   processor.set_segmentation_method('ensemble')
   
   # More accurate registration
   processor.set_registration_iterations(200)
   
   # Smoother contours
   processor.set_smoothing_iterations(30)
   processor.set_decimation_target(50000)  # More triangles

Adjust Performance Settings
----------------------------

.. code-block:: python

   # Faster processing
   processor.set_registration_iterations(50)
   processor.set_segmentation_method('totalsegmentator')
   processor.set_decimation_target(5000)  # Fewer triangles
   
   # Use CPU if no GPU
   processor.set_registration_device('cpu')
   processor.set_segmentation_device('cpu')

Process Specific Structures Only
---------------------------------

.. code-block:: python

   # Only process heart and vessels
   processor.set_structures_to_process(['heart', 'vessels'])

Troubleshooting
===============

Common Issues
-------------

**Issue: Out of GPU memory**

.. code-block:: python

   # Solution 1: Downsample images
   processor.set_downsample_factor(2)
   
   # Solution 2: Use CPU
   processor.set_registration_device('cpu')
   processor.set_segmentation_device('cpu')

**Issue: Poor segmentation quality**

.. code-block:: python

   # Solution 1: Try different method
   processor.set_segmentation_method('ensemble')
   
   # Solution 2: Preprocess image
   processor.set_intensity_normalization(True)
   processor.set_denoise(True)

**Issue: Registration not converging**

.. code-block:: python

   # Solution 1: More iterations
   processor.set_registration_iterations(200)
   
   # Solution 2: Different method
   processor.set_registration_method('ants')

**Issue: USD not animating**

.. code-block:: bash

   # Validate USD file
   usdchecker tutorial_output/my_first_cardiac_model.usd
   
   # Check time samples
   usdview tutorial_output/my_first_cardiac_model.usd

Next Steps
==========

Now that you've completed the basic workflow:

* Try :doc:`custom_segmentation` for advanced segmentation
* Learn :doc:`image_registration` for registration details
* Explore :doc:`vtk_to_usd` for custom USD creation
* See :doc:`../examples` for more use cases

Additional Resources
====================

* :doc:`../user_guide/heart_gated_ct` - Detailed user guide
* :doc:`../api/core` - API reference
* :doc:`../troubleshooting` - Troubleshooting guide
* `Sample Data <https://example.com/samples>`_ - Download more datasets

