====================================
Brain Vessel Modeling
====================================

.. note::
   This script is planned for a future release. Documentation will be updated when available.

Overview
========

The ``physiomotion4d-brain-vessel-modeling`` script will extract, model, and convert brain vasculature from angiography data into dynamic USD models for visualization in NVIDIA Omniverse.

Planned Features
================

Input Requirements
------------------

* **Angiography Data**: CT angiography (CTA) or MR angiography (MRA)
* **Optional**: Time-resolved angiography (4D)
* **Optional**: Contrast bolus timing information
* **Resolution**: Sub-millimeter preferred for small vessels

Processing Pipeline
-------------------

1. **Vessel Enhancement**
   * Apply Hessian-based vessel filters
   * Enhance tubular structures
   * Suppress background and noise

2. **Vessel Segmentation**
   * Segment arterial and venous trees
   * Identify major vessels (Circle of Willis, etc.)
   * Classify vessel types

3. **Centerline Extraction**
   * Extract vessel centerlines
   * Compute vessel radii
   * Build connectivity graph

4. **Surface Reconstruction**
   * Generate smooth vessel surfaces
   * Preserve bifurcations and anastomoses
   * Model vessel wall thickness

5. **Flow Visualization** (4D data)
   * Track contrast bolus propagation
   * Visualize flow direction and speed
   * Generate flow animations

6. **USD Creation**
   * Create vessel tree models
   * Apply arterial/venous materials
   * Add flow visualizations

Expected Outputs
----------------

.. code-block:: text

   <project_name>.arterial_tree_painted.usd
       Arterial vasculature with materials
   
   <project_name>.venous_tree_painted.usd
       Venous vasculature (if present)
   
   <project_name>.vessel_flow.usd
       Flow visualization (4D data only)

Use Cases
=========

* **Neurovascular Planning**: Pre-surgical planning for aneurysms/AVMs
* **Stroke Analysis**: Visualize perfusion territories
* **Research**: Study vascular anatomy variations
* **Education**: Interactive brain vasculature teaching
* **CFD Validation**: Patient-specific flow simulations

Workflow Class
==============

For Python API access, see :class:`physiomotion4d.WorkflowModelBrainVessels` in :doc:`../developer/workflows`.

Related Scripts
===============

* :doc:`vtk_to_usd` - Convert vessel meshes to USD
* :doc:`heart_gated_ct` - Similar workflow for cardiac vessels
