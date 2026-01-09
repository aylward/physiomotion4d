====================================
USD Generation Modules
====================================

.. currentmodule:: physiomotion4d

Convert medical imaging data to Universal Scene Description (USD) format for NVIDIA Omniverse.

Overview
========

USD generation tools for creating animated 3D models from medical images:

* **USD Tools**: Core USD file operations
* **Anatomy Tools**: Anatomical structure handling
* **VTK Conversion**: Convert VTK meshes to USD
* **PolyMesh**: Surface mesh representation
* **TetMesh**: Tetrahedral mesh representation

Quick Links
===========

**USD Modules**:
   * :doc:`tools` - Core USD utilities
   * :doc:`anatomy_tools` - Anatomical structure tools
   * :doc:`vtk_conversion` - VTK to USD conversion
   * :doc:`polymesh` - Surface mesh USD
   * :doc:`tetmesh` - Tetrahedral mesh USD

Module Documentation
====================

.. toctree::
   :maxdepth: 2

   tools
   anatomy_tools
   vtk_conversion
   polymesh
   tetmesh

Quick Start
===========

Convert VTK to USD
------------------

.. code-block:: python

   from physiomotion4d import ConvertVTK4DToUSD
   
   converter = ConvertVTK4DToUSD(
       output_file="animated_heart.usd",
       colormap="rainbow",
       verbose=True
   )
   
   converter.convert(
       vtk_files=["heart_phase_00.vtk", "heart_phase_01.vtk"],
       time_points=[0.0, 0.1, 0.2]
   )

Create Anatomical Scene
-----------------------

.. code-block:: python

   from physiomotion4d import usd_anatomy_tools
   
   stage = usd_anatomy_tools.create_anatomical_stage()
   usd_anatomy_tools.add_heart_model(stage, "heart.vtk")
   usd_anatomy_tools.add_lungs_model(stage, "lungs.vtk")
   stage.Save()

See Also
========

* :doc:`../workflows` - Complete USD workflows
* :doc:`../segmentation/index` - Segmentation for USD generation

.. rubric:: Navigation

:doc:`../model_registration/pca` | :doc:`../index` | :doc:`tools`
