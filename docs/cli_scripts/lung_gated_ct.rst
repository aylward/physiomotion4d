====================================
Lung Gated CT Processing
====================================

.. note::
   This script is planned for a future release. Documentation will be updated when available.

Overview
========

The ``physiomotion4d-lung-gated-ct`` script will process respiratory-gated CT images into dynamic lung models with physiological breathing motion for NVIDIA Omniverse visualization.

Planned Features
================

Input Requirements
------------------

* **Data Format**: 4D lung CT (respiration-gated)
* **Phases**: Typically 10 respiratory phases (inhale to exhale)
* **Modality**: Chest CT with lung parenchyma visible
* **Resolution**: 1-2mm recommended

Processing Pipeline
-------------------

1. **Respiratory Phase Organization**
   * Sort phases by respiratory cycle position
   * Identify inhale/exhale extremes
   
2. **Lung Segmentation**
   * Segment lungs, airways, vessels
   * Identify lobes and segments
   * Detect nodules or lesions if present

3. **Registration**
   * Register phases to reference (end-exhale typical)
   * Account for large deformations
   * Preserve lung topology

4. **Ventilation Analysis**
   * Compute local volume changes
   * Generate ventilation maps
   * Identify poorly ventilated regions

5. **USD Generation**
   * Create animated lung models
   * Visualize ventilation with colormaps
   * Include airways and vasculature

Expected Outputs
----------------

.. code-block:: text

   <project_name>.lung_parenchyma_painted.usd
       Animated lungs with ventilation colormaps
   
   <project_name>.airways_vessels_painted.usd
       Static airways and blood vessels
   
   <project_name>.respiratory_motion.usd
       Complete respiratory system model

Use Cases
=========

* **Pulmonary Function**: Visualize regional ventilation
* **Radiation Planning**: 4D treatment planning for lung tumors
* **COPD Analysis**: Assess ventilation heterogeneity
* **Research**: Study breathing mechanics and disease

Workflow Class
==============

For Python API access, see :class:`physiomotion4d.WorkflowConvertLungGatedCTToUSD` in :doc:`../developer/workflows`.

Related Scripts
===============

* :doc:`heart_gated_ct` - Similar workflow for cardiac imaging
* :doc:`4dct_reconstruction` - Reconstruct 4D-CT from acquisitions
