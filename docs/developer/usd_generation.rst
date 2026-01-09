====================================
USD Generation Development Guide
====================================

This guide covers developing with USD generation tools.

For complete API documentation, see :doc:`../api/usd/index`.

Overview
========

The USD generation module converts VTK meshes into Universal Scene Description (USD) format for visualization in NVIDIA Omniverse with anatomically-realistic materials and time-varying geometry.

Overview
========

PhysioMotion4D provides comprehensive USD conversion capabilities:

* **Base Converter**: Core USD generation functionality
* **PolyMesh Converter**: Surface mesh conversion
* **TetMesh Converter**: Volumetric tetrahedral mesh conversion
* **Anatomical Materials**: Organ-specific material painting
* **Time-Varying Geometry**: 4D animation support

All converters inherit from :class:`ConvertVTK4DToUSDBase`.

Base Converter Class
====================

ConvertVTK4DToUSDBase
---------------------

Abstract base class for USD conversion.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDBase
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods**:
   * ``convert(vtk_file, usd_file)``: Convert VTK to USD
   * ``add_material(mesh, material_name)``: Apply material
   * ``create_time_varying(vtk_files, usd_file)``: Create animated USD
   * ``merge_usd_files(usd_files, output)``: Combine USD files

USD Converters
==============

Polygon Mesh Converter
----------------------

Convert surface meshes (VTK PolyData) to USD.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDPolyMesh
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Surface mesh conversion
   * Anatomical material painting
   * Time-varying geometry
   * Scalar data visualization with colormaps

**Example Usage**:

.. code-block:: python

   from physiomotion4d import ConvertVTK4DToUSDPolyMesh
   
   # Initialize converter
   converter = ConvertVTK4DToUSDPolyMesh(
       start_time=0,
       end_time=1.0,
       fps=30,
       verbose=True
   )
   
   # Convert single mesh
   converter.convert(
       vtk_file="heart_mesh.vtk",
       usd_file="heart.usd",
       mesh_name="heart_lv",
       apply_materials=True
   )
   
   # Convert 4D time series
   vtk_files = [
       "heart_frame_00.vtk",
       "heart_frame_01.vtk",
       # ... more frames
   ]
   
   converter.create_time_varying(
       vtk_files=vtk_files,
       usd_file="heart_animated.usd",
       mesh_names=["heart_lv", "heart_rv", "aorta"]
   )

**Material Painting**:

.. code-block:: python

   # Automatic material from mesh name
   converter.convert(
       vtk_file="heart_lv.vtk",
       usd_file="output.usd",
       mesh_name="heart_lv",  # Auto-detected as heart
       apply_materials=True
   )
   
   # Custom material
   converter.convert(
       vtk_file="custom.vtk",
       usd_file="output.usd",
       material_name="cardiac_tissue",
       material_color=[0.8, 0.2, 0.2],
       material_opacity=0.9
   )

**Colormap Visualization**:

.. code-block:: python

   # Visualize scalar data with colormap
   converter.convert_with_colormap(
       vtk_file="heart_strain.vtk",
       usd_file="strain_viz.usd",
       scalar_array="Strain",
       colormap="plasma",
       value_range=[0, 0.2],
       show_colorbar=True
   )

Tetrahedral Mesh Converter
---------------------------

Convert volumetric meshes (VTK UnstructuredGrid) to USD.

.. autoclass:: physiomotion4d.ConvertVTK4DToUSDTetMesh
   :members:
   :undoc-members:
   :show-inheritance:

**Features**:
   * Volumetric mesh conversion
   * Internal structure visualization
   * Cut-plane rendering
   * Volume data representation

**Example Usage**:

.. code-block:: python

   from physiomotion4d import ConvertVTK4DToUSDTetMesh
   
   # Initialize for tetrahedral meshes
   converter = ConvertVTK4DToUSDTetMesh(verbose=True)
   
   # Convert volumetric mesh
   converter.convert(
       vtk_file="heart_volume.vtu",
       usd_file="heart_volume.usd",
       extract_surface=True,  # Also create surface
       internal_opacity=0.1   # Make volume semi-transparent
   )
   
   # Create cut-plane visualization
   converter.convert_with_cutplane(
       vtk_file="heart_volume.vtu",
       usd_file="heart_cutplane.usd",
       plane_normal=[1, 0, 0],
       plane_position=0.5
   )

Anatomical Material System
===========================

Material Library
----------------

PhysioMotion4D includes a comprehensive anatomical material library:

.. code-block:: python

   from physiomotion4d import USDAnatomyTools
   
   # Access material library
   anatomy_tools = USDAnatomyTools()
   
   # Get available materials
   materials = anatomy_tools.get_available_materials()
   print(f"Available: {materials}")
   
   # Get material for structure
   mat = anatomy_tools.get_material("heart_lv")
   print(f"Heart LV: {mat}")

**Material Categories**:
   * **Cardiac**: Heart chambers, myocardium, valves
   * **Vascular**: Arteries, veins, capillaries
   * **Pulmonary**: Lungs, airways, alveoli
   * **Skeletal**: Bones, cartilage
   * **Soft Tissue**: Muscles, fat, connective tissue
   * **Neural**: Brain, nerves
   * **Visceral**: Liver, kidneys, spleen, etc.

Custom Materials
----------------

Define custom anatomical materials:

.. code-block:: python

   from physiomotion4d import USDAnatomyTools
   
   anatomy_tools = USDAnatomyTools()
   
   # Define custom material
   anatomy_tools.add_custom_material(
       name="custom_cardiac_tissue",
       base_color=[0.85, 0.2, 0.2],
       metallic=0.0,
       roughness=0.8,
       opacity=0.95,
       emission_color=[0.1, 0.0, 0.0],
       emission_strength=0.2
   )
   
   # Use custom material
   converter.apply_material(
       usd_stage=stage,
       prim_path="/heart_lv",
       material_name="custom_cardiac_tissue"
   )

Material Properties
-------------------

Anatomical materials support physically-based rendering:

.. code-block:: python

   material_properties = {
       'base_color': [0.8, 0.2, 0.2],      # RGB color
       'metallic': 0.0,                     # 0=non-metal, 1=metal
       'roughness': 0.7,                    # 0=smooth, 1=rough
       'opacity': 0.9,                      # 0=transparent, 1=opaque
       'ior': 1.4,                          # Index of refraction
       'emission_color': [0.0, 0.0, 0.0],  # Emissive color
       'emission_strength': 0.0,            # Emission intensity
       'normal_map': None,                  # Normal map texture
       'displacement': None                 # Displacement map
   }

Time-Varying Geometry
=====================

Animation Creation
------------------

Create animated USD files from 4D VTK sequences:

.. code-block:: python

   from physiomotion4d import ConvertVTK4DToUSD
   
   # Initialize with timing
   converter = ConvertVTK4DToUSD(
       start_time=0.0,
       end_time=2.0,    # 2 second animation
       fps=30,          # 30 frames per second
       loop=True        # Loop animation
   )
   
   # Create 4D animation
   vtk_4d_files = [f"frame_{i:03d}.vtk" for i in range(60)]
   
   converter.create_animation(
       vtk_files=vtk_4d_files,
       usd_file="animated_heart.usd",
       mesh_names=["heart_lv", "heart_rv", "heart_myocardium"]
   )

Time Sampling
-------------

Control temporal resolution:

.. code-block:: python

   # High temporal resolution
   converter = ConvertVTK4DToUSD(
       start_time=0,
       end_time=1.0,
       fps=60  # Smooth animation
   )
   
   # Match source frame rate
   num_frames = len(vtk_files)
   cycle_duration = 1.0  # 1 second cardiac cycle
   
   converter = ConvertVTK4DToUSD(
       start_time=0,
       end_time=cycle_duration,
       fps=num_frames / cycle_duration
   )

Colormap Rendering
==================

Scalar Data Visualization
-------------------------

Visualize scalar data on meshes:

.. code-block:: python

   from physiomotion4d import ConvertVTK4DToUSDPolyMesh
   
   converter = ConvertVTK4DToUSDPolyMesh()
   
   # Available colormaps
   colormaps = [
       'plasma', 'viridis', 'inferno', 'magma',
       'rainbow', 'jet', 'hot', 'cool',
       'gray', 'bone', 'copper'
   ]
   
   # Apply colormap to scalar data
   converter.convert_with_colormap(
       vtk_file="heart_with_strain.vtk",
       usd_file="strain_visualization.usd",
       scalar_array="Strain",           # Array name in VTK
       colormap="plasma",
       value_range=[0.0, 0.15],        # Data range
       show_colorbar=True,
       colorbar_position="right"
   )

Time-Varying Colormaps
----------------------

Animate colormaps over time:

.. code-block:: python

   # Create time-varying colormap animation
   vtk_files_with_data = [
       "heart_t0_strain.vtk",
       "heart_t1_strain.vtk",
       # ... more timesteps
   ]
   
   converter.create_time_varying_colormap(
       vtk_files=vtk_files_with_data,
       usd_file="strain_animation.usd",
       scalar_array="Strain",
       colormap="viridis",
       global_range=[0, 0.2],  # Fixed range across time
       fps=30
   )

USD File Management
===================

Merging USD Files
-----------------

Combine multiple USD files:

.. code-block:: python

   from physiomotion4d import USDTools
   
   usd_tools = USDTools()
   
   # Merge multiple anatomical structures
   usd_tools.merge_usd_files(
       input_files=[
           "heart.usd",
           "lungs.usd",
           "vessels.usd"
       ],
       output_file="complete_anatomy.usd",
       preserve_hierarchy=True,
       resolve_conflicts=True
   )

Scene Organization
------------------

Organize USD scene hierarchy:

.. code-block:: python

   from pxr import Usd, UsdGeom
   
   # Create organized scene
   stage = Usd.Stage.CreateNew("organized_scene.usd")
   
   # Create hierarchy
   anatomy_root = UsdGeom.Xform.Define(stage, "/Anatomy")
   cardiac = UsdGeom.Xform.Define(stage, "/Anatomy/Cardiac")
   vessels = UsdGeom.Xform.Define(stage, "/Anatomy/Vessels")
   
   # Add meshes to hierarchy
   converter.add_mesh_to_stage(
       stage=stage,
       vtk_file="heart_lv.vtk",
       prim_path="/Anatomy/Cardiac/LeftVentricle"
   )
   
   stage.Save()

Advanced Features
=================

Level of Detail (LOD)
---------------------

Create multi-resolution meshes:

.. code-block:: python

   class LODConverter(ConvertVTK4DToUSDPolyMesh):
       """Converter with LOD support."""
       
       def create_lod_mesh(self, vtk_file, usd_file, lod_levels):
           """Create mesh with multiple LODs."""
           import pyvista as pv
           
           # Load high-res mesh
           mesh = pv.read(vtk_file)
           
           # Create LOD variants
           for lod, reduction in enumerate(lod_levels):
               decimated = mesh.decimate(reduction)
               self.convert(
                   vtk_mesh=decimated,
                   usd_file=usd_file,
                   variant_name=f"LOD_{lod}"
               )

Instancing
----------

Efficiently handle repeated structures:

.. code-block:: python

   from physiomotion4d import USDTools
   
   usd_tools = USDTools()
   
   # Create instances of the same mesh
   usd_tools.create_instances(
       stage=stage,
       source_prim="/Vessels/Artery",
       instance_paths=[
           "/Anatomy/Artery_1",
           "/Anatomy/Artery_2",
           "/Anatomy/Artery_3"
       ],
       transforms=[
           transform_1,
           transform_2,
           transform_3
       ]
   )

Performance Optimization
========================

Mesh Simplification
-------------------

Reduce polygon count for performance:

.. code-block:: python

   import pyvista as pv
   
   def optimize_mesh_for_usd(vtk_file, target_reduction=0.5):
       """Optimize mesh for USD conversion."""
       mesh = pv.read(vtk_file)
       
       # Decimate
       decimated = mesh.decimate(target_reduction)
       
       # Clean
       cleaned = decimated.clean()
       
       # Smooth
       smoothed = cleaned.smooth(n_iter=20)
       
       return smoothed

Batch Conversion
----------------

Convert multiple files efficiently:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   from pathlib import Path
   
   def batch_convert_to_usd(vtk_dir, usd_dir, num_workers=4):
       """Convert multiple VTK files to USD in parallel."""
       converter = ConvertVTK4DToUSDPolyMesh()
       
       vtk_files = list(Path(vtk_dir).glob("*.vtk"))
       
       def convert_single(vtk_file):
           usd_file = Path(usd_dir) / f"{vtk_file.stem}.usd"
           converter.convert(str(vtk_file), str(usd_file))
           return usd_file
       
       with ThreadPoolExecutor(max_workers=num_workers) as executor:
           results = list(executor.map(convert_single, vtk_files))
       
       return results

Best Practices
==============

Material Assignment
-------------------

* Use anatomically accurate colors and properties
* Maintain consistent naming conventions
* Apply appropriate transparency for visualization
* Use emission for highlighting structures

Animation
---------

* Match frame rate to source data
* Use smooth temporal interpolation
* Ensure consistent topology across frames
* Optimize polygon count for real-time playback

File Organization
-----------------

.. code-block:: text

   # Recommended structure
   output/
   ├── meshes/
   │   ├── frame_000.vtk
   │   ├── frame_001.vtk
   │   └── ...
   ├── usd/
   │   ├── static_anatomy.usd
   │   ├── dynamic_anatomy.usd
   │   └── complete_scene.usd
   └── textures/
       └── colormaps/

See Also
========

* :doc:`utilities` - USD and mesh utilities
* :doc:`workflows` - Using USD conversion in workflows
* :doc:`../cli_scripts/vtk_to_usd` - CLI for USD conversion
