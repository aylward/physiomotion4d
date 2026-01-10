====================================
CLI & Scripts Overview
====================================

This section provides comprehensive guides for using PhysioMotion4D's command-line tools to process medical imaging data. These tools are designed for medical imaging experts and physiological simulation researchers who need efficient, reproducible pipelines for converting 4D medical images into dynamic anatomical models for NVIDIA Omniverse.

Target Audience
===============

These CLI tools are intended for users with:

* Strong medical image analysis expertise
* Understanding of physiological simulation requirements
* Modest Python experience for running scripts
* Familiarity with command-line interfaces

If you are a Python developer looking to extend or integrate PhysioMotion4D into your applications, please refer to the :doc:`../developer/architecture` section.

Available Scripts
=================

Current Scripts
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`heart_gated_ct`
     - Process cardiac gated CT to animated heart models with physiological motion
   * - :doc:`heart_model_to_patient`
     - Register generic heart models to patient-specific imaging data and surface models

Upcoming Scripts
----------------

The following scripts are planned for future releases:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Description
   * - :doc:`lung_gated_ct`
     - Process respiratory-gated CT to animated lung models
   * - :doc:`4dct_reconstruction`
     - Reconstruct 4D CT from multiple 3D acquisitions
   * - :doc:`vtk_to_usd`
     - Convert VTK anatomical models to USD format with material painting
   * - :doc:`brain_vessel_modeling`
     - Extract and model brain vasculature from angiography data

Installation
============

All scripts are installed with the PhysioMotion4D package:

.. code-block:: bash

   pip install physiomotion4d

After installation, scripts are available as command-line tools with the prefix ``physiomotion4d-``:

.. code-block:: bash

   physiomotion4d-heart-gated-ct --help

General Workflow
================

All PhysioMotion4D scripts follow a similar pattern:

1. **Input Data**: Provide medical image files (NRRD, NII, MHA formats)
2. **Configuration**: Set processing parameters via command-line flags
3. **Processing Pipeline**: Automated execution of segmentation, registration, and conversion
4. **Output Generation**: USD files ready for Omniverse visualization

Typical Command Structure
--------------------------

.. code-block:: bash

   physiomotion4d-<script-name> input_files [options]
       --output-dir <directory>
       --project-name <name>
       [script-specific options]

Output Organization
-------------------

Each script organizes outputs in a consistent structure:

.. code-block:: text

   output_directory/
   ├── intermediate/          # Segmentations, registrations
   ├── meshes/               # VTK mesh files
   └── usd/                  # Final USD files for Omniverse

Getting Help
============

Each script provides detailed help:

.. code-block:: bash

   physiomotion4d-<script-name> --help

For troubleshooting and common issues, see :doc:`../troubleshooting`.

Next Steps
==========

* Start with :doc:`heart_gated_ct` for a complete example
* See :doc:`best_practices` for optimization tips
* Refer to script-specific pages for detailed usage
