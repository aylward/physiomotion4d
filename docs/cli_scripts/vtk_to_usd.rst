====================================
VTK to USD Conversion
====================================

.. note::
   This script is planned for a future release. Documentation will be updated when available.

Overview
========

The ``physiomotion4d-vtk-to-usd`` script will convert VTK anatomical mesh files to USD format with intelligent material painting based on anatomical structure names.

Planned Features
================

Input Requirements
------------------

* **VTK Meshes**: Surface meshes (``.vtk``, ``.vtp``) or volume meshes (``.vtu``)
* **Naming Convention**: Mesh names indicate anatomy (e.g., "heart_lv", "lung_left")
* **Optional**: Scalar data arrays for colormap visualization
* **Optional**: 4D time series VTK files

Material Painting
-----------------

1. **Anatomical Recognition**
   * Parse mesh names for anatomical structures
   * Apply organ-specific materials
   * Handle structure hierarchies

2. **Material Library**
   * Pre-defined materials for common organs
   * Transparency and color by structure type
   * Physically-based rendering properties

3. **Colormap Support**
   * Visualize scalar data (temperature, pressure, strain)
   * Multiple colormap options (viridis, plasma, heat, etc.)
   * Custom intensity ranges

4. **Time-Varying Data**
   * Convert 4D VTK sequences to animated USD
   * Preserve temporal coherence
   * Support dynamic colormaps

Processing Options
------------------

.. code-block:: bash

   # Basic conversion
   physiomotion4d-vtk-to-usd meshes/*.vtk --output model.usd

   # With colormap for scalar data
   physiomotion4d-vtk-to-usd mesh.vtk \
       --scalar-array "Temperature" \
       --colormap plasma \
       --output temperature_viz.usd

   # 4D time series
   physiomotion4d-vtk-to-usd heart_4d.vtk \
       --time-varying \
       --output animated_heart.usd

Expected Outputs
----------------

* USD file with painted anatomical models
* Hierarchical scene organization
* Time-varying geometry (if 4D input)
* Material definitions and textures

Use Cases
=========

* **Simulation Results**: Visualize CFD or FEA simulations in Omniverse
* **Anatomical Models**: Convert segmentation meshes to USD
* **Education**: Create anatomical teaching models
* **Pipeline Integration**: Bridge VTK workflows to Omniverse

Workflow Class
==============

For Python API access, see :class:`physiomotion4d.WorkflowConvertVTKToUSD` in :doc:`../developer/workflows`.

Related Scripts
===============

* :doc:`heart_gated_ct` - Generates VTK meshes from CT
* :doc:`lung_gated_ct` - Generates VTK meshes from lung CT
