#!/usr/bin/env python
# %% [markdown]
# # VTK to USD Converter Test Notebook
#
# This notebook demonstrates the usage of the new `vtk_to_usd` library for converting VTK files to USD format.
#
# The library is based on the ParaViewConnector architecture from Omniverse but simplified for file-based conversion only.
#
# ## Features
#
# - **File Format Support**: VTK legacy (.vtk), XML PolyData (.vtp), XML UnstructuredGrid (.vtu)
# - **Geometry Preservation**: Points, faces, normals, colors
# - **Data Arrays**: VTK point and cell data arrays → USD primvars
# - **Time-Series**: Support for animated/time-varying data
# - **Materials**: UsdPreviewSurface materials with customizable properties
# - **Coordinate System**: Automatic conversion from RAS (medical imaging) to USD Y-up
#
# ## Test Data
#
# We'll use the KCL Heart Model data:
# - `average_surface.vtp`: Surface mesh of the heart
# - `average_mesh.vtk`: Volumetric mesh of the heart

# %%
import copy
import logging
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom, UsdShade

from physiomotion4d import ContourTools

# Import the new vtk_to_usd library
from physiomotion4d.vtk_to_usd import (
    VTKToUSDConverter,
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    convert_vtk_file,
    read_vtk_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_HERE = Path(__file__).parent

# %% [markdown]
# ## 1. Basic Conversion: VTP to USD
#
# Let's start with the simplest use case: converting a single VTP file to USD.

# %%
# Define file paths
data_dir = _HERE / "../../data/KCL-Heart-Model"
output_dir = _HERE / "results"
output_dir.mkdir(exist_ok=True)

vtk_file = data_dir / "average_mesh.vtk"

print("Input files:")
print(f"  VTK: {vtk_file.exists()} - {vtk_file}")
print(f"\nOutput directory: {output_dir}")

# %%
vtp_file = output_dir / "average_surface.vtp"
if not vtp_file.exists():
    vtk_mesh = pv.read(vtk_file)
    contour_tools = ContourTools()
    vtp_surface = vtk_mesh.extract_surface(algorithm="dataset_surface")
    vtp_surface.save(vtp_file)
print(f"  VTP: {vtp_file.exists()} - {vtp_file}")

# %%
# Simple conversion using convenience function
output_usd = output_dir / "heart_surface_simple.usd"

stage = convert_vtk_file(
    vtk_file=vtp_file, output_usd=output_usd, mesh_name="HeartSurface"
)

print(f"\nCreated USD file: {output_usd}")
print("Stage info:")
print(f"  Root layer: {stage.GetRootLayer().identifier}")
print(f"  Default prim: {stage.GetDefaultPrim().GetPath()}")

# %% [markdown]
# ## 2. Inspect VTK File Data
#
# Let's examine what data arrays are present in the VTK files.

# %%
# Read and inspect the VTP file
mesh_data = read_vtk_file(vtp_file)

print("=" * 60)
print("VTP File (average_surface.vtp) Contents:")
print("=" * 60)
print("\nGeometry:")
print(f"  Points: {len(mesh_data.points)}")
print(f"  Faces: {len(mesh_data.face_vertex_counts)}")
print(f"  Normals: {'Yes' if mesh_data.normals is not None else 'No'}")
print(f"  Colors: {'Yes' if mesh_data.colors is not None else 'No'}")

print(f"\nData Arrays ({len(mesh_data.generic_arrays)}):")
for i, array in enumerate(mesh_data.generic_arrays, 1):
    print(f"  {i}. {array.name}:")
    print(f"     - Components: {array.num_components}")
    print(f"     - Type: {array.data_type.value}")
    print(f"     - Interpolation: {array.interpolation}")
    print(f"     - Shape: {array.data.shape}")
    if array.data.size > 0:
        print(f"     - Range: [{np.min(array.data):.6f}, {np.max(array.data):.6f}]")

# %%
# Read and inspect the VTK file
mesh_data_vtk = read_vtk_file(vtk_file, extract_surface=True)

print("=" * 60)
print("VTK File (average_mesh.vtk) Contents:")
print("=" * 60)
print("\nGeometry:")
print(f"  Points: {len(mesh_data_vtk.points)}")
print(f"  Faces: {len(mesh_data_vtk.face_vertex_counts)}")
print(f"  Normals: {'Yes' if mesh_data_vtk.normals is not None else 'No'}")
print(f"  Colors: {'Yes' if mesh_data_vtk.colors is not None else 'No'}")

print(f"\nData Arrays ({len(mesh_data_vtk.generic_arrays)}):")
for i, array in enumerate(mesh_data_vtk.generic_arrays, 1):
    print(f"  {i}. {array.name}:")
    print(f"     - Components: {array.num_components}")
    print(f"     - Type: {array.data_type.value}")
    print(f"     - Interpolation: {array.interpolation}")
    print(f"     - Shape: {array.data.shape}")
    if array.data.size > 0:
        print(f"     - Range: [{np.min(array.data):.6f}, {np.max(array.data):.6f}]")

# %% [markdown]
# ## 3. Advanced Conversion with Custom Settings
#
# Now let's use custom settings to control the conversion process.

# %%
# Create custom conversion settings
settings = ConversionSettings(
    triangulate_meshes=True,  # Ensure all faces are triangles
    compute_normals=True,  # Compute normals if not present
    preserve_point_arrays=True,  # Keep all point data as primvars
    preserve_cell_arrays=True,  # Keep all cell data as primvars
    meters_per_unit=0.001,  # Assume VTK data is in millimeters
    up_axis="Y",  # Use Y-up (USD standard)
)

# Create custom material
material = MaterialData(
    name="heart_material",
    diffuse_color=(0.9, 0.3, 0.3),  # Reddish color for heart
    roughness=0.4,
    metallic=0.0,
)

# Create converter
converter = VTKToUSDConverter(settings)

# Convert with custom settings
output_usd_custom = output_dir / "heart_surface_custom.usd"
stage_custom = converter.convert_file(
    vtk_file=vtp_file,
    output_usd=output_usd_custom,
    mesh_name="HeartSurface",
    material=material,
)

print(f"\nCreated custom USD file: {output_usd_custom}")

# %% [markdown]
# ## 4. Convert VTK Legacy Format
#
# Now let's convert the legacy VTK format file.

# %%
# Convert VTK file with custom material
output_usd_vtk = output_dir / "heart_mesh.usd"

material_mesh = MaterialData(
    name="heart_mesh_material",
    diffuse_color=(0.8, 0.4, 0.4),
    roughness=0.5,
    metallic=0.0,
)

stage_vtk = converter.convert_file(
    vtk_file=vtk_file,
    output_usd=output_usd_vtk,
    mesh_name="HeartMesh",
    material=material_mesh,
    extract_surface=True,  # Extract surface from volumetric mesh
)

print(f"\nCreated VTK USD file: {output_usd_vtk}")

# %% [markdown]
# ## 5. Inspect USD Output
#
# Let's examine the created USD file to verify all data was preserved.

# %%
# Open the USD file for inspection
inspect_stage = Usd.Stage.Open(str(output_usd_custom))

print("=" * 60)
print("USD File Inspection")
print("=" * 60)

# Get the mesh prim
mesh_path = "/World/Meshes/HeartSurface"
mesh_prim = inspect_stage.GetPrimAtPath(mesh_path)

if mesh_prim:
    mesh = UsdGeom.Mesh(mesh_prim)

    print(f"\nMesh: {mesh_path}")
    print(f"  Type: {mesh_prim.GetTypeName()}")

    # Geometry attributes
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    print("\nGeometry:")
    print(f"  Points: {len(points) if points else 0}")
    print(f"  Faces: {len(face_counts) if face_counts else 0}")
    print(f"  Face indices: {len(face_indices) if face_indices else 0}")

    # Check normals
    normals_attr = mesh.GetNormalsAttr()
    if normals_attr:
        normals = normals_attr.Get()
        print(f"  Normals: {len(normals) if normals else 0}")

    # List primvars
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    primvars = primvars_api.GetPrimvars()

    print(f"\nPrimvars ({len(primvars)}):")
    for primvar in primvars:
        name = primvar.GetPrimvarName()
        interpolation = primvar.GetInterpolation()
        type_name = primvar.GetTypeName()
        value = primvar.Get()
        size = len(value) if value else 0
        print(f"  - {name}: {type_name} ({interpolation}), {size} elements")

    # Check material binding
    binding_api = UsdShade.MaterialBindingAPI(mesh)
    material_binding = binding_api.GetDirectBinding()
    if material_binding:
        material_path = material_binding.GetMaterialPath()
        print(f"\nMaterial Binding: {material_path}")
else:
    print(f"\nMesh not found at path: {mesh_path}")

# %% [markdown]
# ## 6. Create Time-Series Data (Simulated)
#
# Demonstrate time-series conversion by creating synthetic deformation of the mesh.

# %%
# Create a simple time-series by deforming the mesh


def create_deformed_mesh(base_mesh_data, time_step, num_steps=10):
    """Create a deformed version of the mesh for animation."""
    # Clone the mesh data
    deformed_mesh = copy.deepcopy(base_mesh_data)

    # Apply sinusoidal deformation
    t = time_step / num_steps * 2 * np.pi
    scale_factor = 1.0 + 0.1 * np.sin(t)  # 10% amplitude

    # Scale points radially from centroid
    centroid = np.mean(deformed_mesh.points, axis=0)
    deformed_mesh.points = centroid + (deformed_mesh.points - centroid) * scale_factor

    # Add a time-varying scalar field (simulated pressure)
    num_points = len(deformed_mesh.points)
    pressure = np.sin(t + np.linspace(0, 2 * np.pi, num_points))
    pressure_array = GenericArray(
        name="pressure",
        data=pressure,
        num_components=1,
        data_type=DataType.FLOAT,
        interpolation="vertex",
    )

    # Add to generic arrays if not already present
    array_names = [arr.name for arr in deformed_mesh.generic_arrays]
    if "pressure" not in array_names:
        deformed_mesh.generic_arrays.append(pressure_array)
    else:
        # Replace existing pressure array
        for i, arr in enumerate(deformed_mesh.generic_arrays):
            if arr.name == "pressure":
                deformed_mesh.generic_arrays[i] = pressure_array
                break

    return deformed_mesh


# Create sequence of deformed meshes
num_time_steps = 10
mesh_sequence = []
time_codes = list(range(num_time_steps))

for t in range(num_time_steps):
    deformed = create_deformed_mesh(mesh_data, t, num_time_steps)
    mesh_sequence.append(deformed)
    print(f"Created time step {t + 1}/{num_time_steps}")

print(f"\nCreated {len(mesh_sequence)} time steps")

# %%
# Convert time series to USD
output_usd_anim = output_dir / "heart_surface_animated.usd"

material_anim = MaterialData(
    name="heart_animated_material",
    diffuse_color=(0.9, 0.2, 0.2),
    roughness=0.3,
    metallic=0.0,
)

stage_anim = converter.convert_mesh_data_sequence(
    mesh_data_sequence=mesh_sequence,
    output_usd=output_usd_anim,
    mesh_name="HeartAnimated",
    time_codes=time_codes,
    material=material_anim,
)

print(f"\nCreated animated USD file: {output_usd_anim}")
print(f"Time range: {stage_anim.GetStartTimeCode()} to {stage_anim.GetEndTimeCode()}")
print(f"Time codes per second: {stage_anim.GetTimeCodesPerSecond()}")

# %% [markdown]
# ## 7. Summary
#
# Let's summarize what we've created.

# %%
print("=" * 60)
print("Generated USD Files")
print("=" * 60)

usd_files = list(output_dir.glob("*.usd"))
usd_files.extend(output_dir.glob("*.usda"))
usd_files.extend(output_dir.glob("*.usdc"))

for usd_file in sorted(usd_files):
    size_kb = os.path.getsize(usd_file) / 1024
    print(f"\n{usd_file.name}:")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Path: {usd_file}")

    # Quick inspection
    stage = Usd.Stage.Open(str(usd_file))
    if stage:
        print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
        print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
        if stage.HasAuthoredTimeCodeRange():
            print(
                f"  Time range: {stage.GetStartTimeCode()} - {stage.GetEndTimeCode()}"
            )

print("\n" + "=" * 60)
print("✓ All conversions completed successfully!")
print("=" * 60)


# %% [markdown]
# ## 8. Verification
#
# Verify that the USD files can be opened and contain the expected data.


# %%
def verify_usd_file(usd_path):
    """Verify USD file integrity."""
    print(f"\nVerifying: {usd_path.name}")
    print("-" * 40)

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print("  ✗ Failed to open stage")
        return False

    # Check default prim
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print("  ✗ No default prim")
        return False
    print(f"  ✓ Default prim: {default_prim.GetPath()}")

    # Find mesh prims
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                print(f"  ✓ Mesh '{prim.GetName()}': {len(points)} points")

    if mesh_count == 0:
        print("  ✗ No meshes found")
        return False

    print(f"  ✓ Total meshes: {mesh_count}")
    return True


# Verify all generated files
print("=" * 60)
print("USD File Verification")
print("=" * 60)

all_valid = True
for usd_file in sorted(usd_files):
    valid = verify_usd_file(usd_file)
    all_valid = all_valid and valid

print("\n" + "=" * 60)
if all_valid:
    print("✓ All USD files are valid!")
else:
    print("✗ Some USD files have issues")
print("=" * 60)

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated the comprehensive features of the new `vtk_to_usd` library:
#
# 1. **Simple Conversion**: One-line conversion of VTK files
# 2. **Data Inspection**: Reading and analyzing VTK data arrays
# 3. **Custom Settings**: Fine-grained control over conversion
# 4. **Multiple Formats**: Support for VTP, VTK, VTU files
# 5. **Material System**: Custom materials with UsdPreviewSurface
# 6. **Time-Series**: Animated meshes with time-varying attributes
# 7. **Data Preservation**: All VTK arrays preserved as USD primvars
# 8. **Coordinate Systems**: Automatic RAS to Y-up conversion
#
# The library is production-ready and can be used for converting medical imaging data, simulation results, and other VTK-based datasets to USD for visualization in Omniverse, USDView, or other USD-compatible applications.
#
# ### Next Steps
#
# - View the generated USD files in USDView or Omniverse
# - Experiment with different conversion settings
# - Test with your own VTK datasets
# - Explore advanced features like custom colormaps and transfer functions
