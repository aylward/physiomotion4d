====================================
Heart Model to Patient Registration
====================================

.. note::
   This script is planned for a future release. Documentation will be updated when available.

Overview
========

The ``physiomotion4d-heart-model-to-patient`` script will register population-based statistical heart models to patient-specific cardiac images, enabling:

* Patient-specific anatomical modeling
* Transfer of population knowledge to individual cases
* Shape analysis and abnormality detection
* Constrained segmentation using model priors

Planned Features
================

Input Data
----------

* **Population Model**: Statistical shape model (PCA-based)
* **Patient Image**: 3D cardiac CT or MRI
* **Optional**: Initial landmarks or segmentation masks

Processing Steps
----------------

1. Initial alignment using landmarks or image features
2. Coarse rigid/affine registration
3. Deformable model-to-image registration
4. Point correspondence mapping
5. Model parameter extraction

Expected Outputs
----------------

* Registered model mesh fitted to patient anatomy
* Transform parameters and displacement fields
* Patient-specific shape parameters
* Quality metrics and visualization

Use Cases
=========

* **Clinical**: Patient-specific heart modeling for surgical planning
* **Research**: Population analysis and shape statistics
* **Education**: Demonstration of anatomical variations
* **Validation**: Comparison of manual vs model-based segmentation

Workflow Class
==============

For Python API access, see :class:`physiomotion4d.WorkflowRegisterHeartModelToPatient` in :doc:`../developer/workflows`.

Related Scripts
===============

* :doc:`heart_gated_ct` - Process cardiac gated CT data
* :doc:`vtk_to_usd` - Convert model meshes to USD format
