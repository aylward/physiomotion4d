==============================
Heart-Gated CT Processing Guide
==============================

Complete guide to processing heart-gated CT data with PhysioMotion4D.

Overview
========

PhysioMotion4D provides a comprehensive workflow for converting 4D cardiac CT scans
into animated 3D models for visualization in NVIDIA Omniverse. The ``HeartGatedCTToUSDWorkflow``
class orchestrates the complete pipeline from raw 4D CT data to Omniverse-ready USD files.

Workflow Steps
==============

The complete pipeline consists of:

1. **Data Preparation**: Convert 4D NRRD to 3D time frames
2. **Image Registration**: Align cardiac phases using ICON or ANTs
3. **Segmentation**: AI-based anatomical structure identification (TotalSegmentator or VISTA-3D)
4. **Contour Extraction**: Extract surface meshes from segmentation masks
5. **Contour Transformation**: Propagate segmentation across time using displacement fields
6. **USD Generation**: Create time-varying USD models
7. **Material Application**: Apply anatomically realistic materials and textures

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from physiomotion4d import HeartGatedCTToUSDWorkflow

   # Initialize workflow
   workflow = HeartGatedCTToUSDWorkflow(
       input_filenames=["cardiac_4d_ct.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001",
       registration_method='icon'  # or 'ants'
   )

   # Run complete pipeline
   final_usd = workflow.process()

Command-Line Interface
----------------------

.. code-block:: bash

   # Process a single 4D cardiac CT file
   physiomotion4d-heart-gated-ct cardiac_4d.nrrd --contrast --output-dir ./results

   # Process multiple time frames
   physiomotion4d-heart-gated-ct frame_*.nrrd --contrast --project-name patient_001

   # With custom settings
   physiomotion4d-heart-gated-ct cardiac.nrrd \
       --contrast \
       --reference-image ref.mha \
       --registration-iterations 50 \
       --output-dir ./output

See :doc:`../tutorials/basic_workflow` for detailed tutorial.

Input Data Requirements
=======================

* **Format**: NRRD 4D or multiple 3D files (NRRD, MHA, NII)
* **Modality**: Cardiac CT (contrast or non-contrast)
* **Phases**: Minimum 2 phases, typically 10-20 phases
* **Resolution**: Isotropic recommended, 0.5-2mm typical
* **Field of View**: Include entire heart and major vessels

Best Practices
==============

1. **Use contrast-enhanced CT** for better segmentation
2. **Ensure good temporal sampling** (10+ phases)
3. **Use GPU acceleration** for faster processing
4. **Start with default parameters** then tune as needed

See :doc:`../examples` for practical examples.

