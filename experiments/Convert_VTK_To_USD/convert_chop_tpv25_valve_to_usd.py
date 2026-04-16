#!/usr/bin/env python
# %% [markdown]
# # Cardiac Valve 4D Time-Series Conversion to USD
#
# This notebook demonstrates converting time-varying cardiac valve simulation data from VTK format to animated USD.
#
# ## Dataset: CHOP-Valve4D (TPV25)
#
# One cardiac valve model with time-varying geometry:
#
# - **TPV25**: 265 time steps (cardiac cycle simulation)
#
# This dataset represents 4D (3D + time) simulation of a prosthetic heart valve during a cardiac cycle.
#
# ## Goals
#
# 1. Load and inspect time-varying VTK data
# 2. Convert entire time series to animated USD
# 3. Handle large datasets efficiently
# 4. Preserve all simulation data as USD primvars
# 5. Create multiple variations (full resolution, subsampled, etc.)

# %% [markdown]
# ## Configuration
#
# Control which time series conversions to compute.

# %%
import re
from pathlib import Path

import numpy as np

from physiomotion4d.notebook_utils import running_as_test

# Import USDTools for post-processing colormap
from physiomotion4d.usd_tools import USDTools

from physiomotion4d import ConvertVTKToUSD

# cell_type_name_for_vertex_count and read_vtk_file are internal APIs used for diagnostics
from physiomotion4d.vtk_to_usd import cell_type_name_for_vertex_count
from physiomotion4d.vtk_to_usd.vtk_reader import read_vtk_file

# %% [markdown]
# ## 1. Discover and Organize Time-Series Files

# %%
# Set to True to use as a test.  Automatically done by
#    running_as_test() helper function.
quick_run = running_as_test()
quick_run_step = 4

# Define data directories (TPV25 only)
data_dir = Path.cwd().parent.parent / "data" / "CHOP-Valve4D"
tpv25_dir = data_dir / "TPV25"

output_dir = Path.cwd() / "results" / "valve4d-tpv25"
if quick_run:
    output_usd = output_dir / "tpv25_quick.usd"
else:
    output_usd = output_dir / "tpv25_full.usd"

colormap_primvar_substrs = ["stress", "strain"]
colormap_name = "jet"  # matplotlib colormap name
colormap_range_min = 25
colormap_range_max = 200

# Conversion parameters
separate_by = "connectivity"  # Essential for tpv25 vtk file
times_per_second = 60.0
solid_color = (0.5, 0.5, 0.5)

# %%
output_dir.mkdir(parents=True, exist_ok=True)

vtk_files = list(Path(tpv25_dir).glob("*.vtk"))
pattern = r"\.t(\d+)\.vtk$"

# Extract time step numbers and pair with files
tpv25_series = []
for vtk_file in vtk_files:
    match = re.search(pattern, vtk_file.name)
    if match:
        time_step = int(match.group(1))
        tpv25_series.append((time_step, vtk_file))

# Sort by time step
tpv25_series.sort(key=lambda x: x[0])

# %% [markdown]
# ## 2. Inspect First Frame
#
# Examine the first time step to understand the data structure.

# %%
# Debuggin
first_file = tpv25_series[0][1]
mesh_data = read_vtk_file(first_file, extract_surface=True)

print(f"\nFile: {first_file.name}")
print("\nGeometry:")
print(f"  Points: {len(mesh_data.points):,}")
print(f"  Faces: {len(mesh_data.face_vertex_counts):,}")
print(f"  Normals: {'Yes' if mesh_data.normals is not None else 'No'}")
print(f"  Colors: {'Yes' if mesh_data.colors is not None else 'No'}")

# Bounding box
bbox_min = np.min(mesh_data.points, axis=0)
bbox_max = np.max(mesh_data.points, axis=0)
bbox_size = bbox_max - bbox_min
print("\nBounding Box:")
print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
print(f"  Size: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}]")

print(f"\nData Arrays ({len(mesh_data.generic_arrays)}):")
for i, array in enumerate(mesh_data.generic_arrays, 1):
    print(f"  {i}. {array.name}:")
    print(f"     - Type: {array.data_type.value}")
    print(f"     - Components: {array.num_components}")
    print(f"     - Interpolation: {array.interpolation}")
    print(f"     - Elements: {len(array.data):,}")
    if array.data.size > 0:
        print(f"     - Range: [{np.min(array.data):.6f}, {np.max(array.data):.6f}]")

# Cell types (face vertex count = triangle, quad, etc.)
unique_counts, num_each = np.unique(mesh_data.face_vertex_counts, return_counts=True)
print("\nCell types (faces by vertex count):")
for u, n in zip(unique_counts, num_each):
    name = cell_type_name_for_vertex_count(int(u))
    print(f"  {name} ({u} vertices): {n:,} faces")

# %% [markdown]
# ## 3. Convert TPV25

# %%
tpv25_files = [file_path for _, file_path in tpv25_series]
tpv25_times = [float(time_step) for time_step, _ in tpv25_series]

if quick_run:
    tpv25_files = tpv25_files[::quick_run_step]
    tpv25_times = tpv25_times[::quick_run_step]

print(f"\nConverting to: {output_usd}")
print(f"Number of time steps: {len(tpv25_times)}")
print("\nThis may take several minutes...\n")

# topology validation and conversion happen inside from_files()
stage = ConvertVTKToUSD.from_files(
    data_basename="TPV25Valve",
    vtk_files=tpv25_files,
    extract_surface=True,
    separate_by=separate_by,
    times_per_second=times_per_second,
    solid_color=solid_color,
    time_codes=tpv25_times,
).convert(str(output_usd))

# %%
usd_tools = USDTools()
# ConvertVTKToUSD places prims at /World/{basename}/{part_name}
if separate_by == "connectivity":
    vessel_path = "/World/TPV25Valve/TPV25Valve_object4"
elif separate_by == "cell_type":
    vessel_path = "/World/TPV25Valve/TPV25Valve_Triangle"
else:
    vessel_path = "/World/TPV25Valve/Mesh"

# Select primvar for coloring
primvars = usd_tools.list_mesh_primvars(str(output_usd), vessel_path)
color_primvar = usd_tools.pick_color_primvar(
    primvars, keywords=tuple(colormap_primvar_substrs)
)
print(f"Chosen primvar = {color_primvar}")

if color_primvar:
    print(f"\nApplying colormap to '{color_primvar}' using {colormap_name}")
    usd_tools.apply_colormap_from_primvar(
        str(output_usd),
        vessel_path,
        color_primvar,
        intensity_range=(colormap_range_min, colormap_range_max),
        cmap=colormap_name,
        use_sigmoid_scale=True,
        bind_vertex_color_material=True,
    )
