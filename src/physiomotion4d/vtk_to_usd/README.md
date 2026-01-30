# VTK to USD Converter Library

A comprehensive Python library for converting VTK files (VTK, VTP, VTU) to USD (Universal Scene Description) format. Based on the architecture of NVIDIA's ParaViewConnector for Omniverse, but simplified for file-based conversion without ParaView or Qt dependencies.

## Features

### File Format Support
- **Legacy VTK** (`.vtk`): Binary and ASCII formats
- **XML PolyData** (`.vtp`): Surface meshes
- **XML UnstructuredGrid** (`.vtu`): Volumetric meshes (with surface extraction)

### Data Preservation
- **Geometry**: Points, faces, topology
- **Normals**: Automatic computation or preservation from source
- **Colors**: Vertex colors (RGB/RGBA)
- **Data Arrays**: All VTK point and cell data arrays converted to USD primvars
- **Time-Series**: Support for animated/time-varying data

### USD Features
- **Materials**: UsdPreviewSurface with customizable properties
- **Primvars**: VTK data arrays → USD primvars with appropriate interpolation
- **Coordinate Systems**: Automatic conversion from RAS (medical imaging) to USD Y-up
- **Time Sampling**: Efficient time-varying attribute encoding

### Architecture

The library is organized into modular components inspired by ParaViewConnector:

```
vtk_to_usd/
├── data_structures.py      # Data containers (MeshData, MaterialData, etc.)
├── vtk_reader.py           # VTK file readers (VTK, VTP, VTU)
├── usd_utils.py            # USD utility functions (coordinate conversion, primvars)
├── material_manager.py     # Material creation and binding
├── usd_mesh_converter.py   # Mesh conversion to USD
├── converter.py            # High-level API
└── __init__.py             # Public exports
```

## Installation

The library is part of the PhysioMotion4D package. Ensure you have the required dependencies:

```bash
pip install vtk pxr numpy
```

## Quick Start

### Simple Conversion

```python
from physiomotion4d.vtk_to_usd import convert_vtk_file

# Convert a single file
stage = convert_vtk_file('mesh.vtp', 'output.usd')
```

### With Custom Settings

```python
from physiomotion4d.vtk_to_usd import VTKToUSDConverter, ConversionSettings, MaterialData

# Configure conversion
settings = ConversionSettings(
    triangulate_meshes=True,
    compute_normals=True,
    preserve_point_arrays=True,
    meters_per_unit=0.001,  # mm to meters
)

# Define material
material = MaterialData(
    name="my_material",
    diffuse_color=(0.8, 0.3, 0.3),
    roughness=0.4,
)

# Convert
converter = VTKToUSDConverter(settings)
stage = converter.convert_file(
    vtk_file='mesh.vtp',
    output_usd='output.usd',
    material=material,
)
```

### Time-Series Data

```python
from physiomotion4d.vtk_to_usd import VTKToUSDConverter

converter = VTKToUSDConverter()

# Convert sequence of VTK files
files = ['frame_0.vtp', 'frame_1.vtp', 'frame_2.vtp']
time_codes = [0.0, 0.1, 0.2]  # seconds

stage = converter.convert_sequence(
    vtk_files=files,
    output_usd='animated.usd',
    time_codes=time_codes,
)
```

### Direct MeshData Conversion

```python
from physiomotion4d.vtk_to_usd import read_vtk_file, VTKToUSDConverter

# Read VTK file
mesh_data = read_vtk_file('mesh.vtp')

# Inspect data
print(f"Points: {len(mesh_data.points)}")
print(f"Faces: {len(mesh_data.face_vertex_counts)}")
print(f"Data arrays: {len(mesh_data.generic_arrays)}")

for array in mesh_data.generic_arrays:
    print(f"  - {array.name}: {array.num_components} components, {array.data_type}")

# Convert to USD
converter = VTKToUSDConverter()
stage = converter.convert_mesh_data(mesh_data, 'output.usd')
```

## API Reference

### ConversionSettings

Configuration for conversion process:

```python
@dataclass
class ConversionSettings:
    # Output settings
    output_binary: bool = False          # Binary or ASCII USD
    meters_per_unit: float = 1.0         # Unit scale
    up_axis: str = "Y"                   # "Y" or "Z"

    # Mesh processing
    triangulate_meshes: bool = True      # Convert all faces to triangles
    compute_normals: bool = True         # Compute normals if missing
    preserve_point_arrays: bool = True   # Keep point data arrays
    preserve_cell_arrays: bool = True    # Keep cell data arrays

    # Material settings
    use_preview_surface: bool = True     # Use UsdPreviewSurface
    default_color: tuple = (0.8, 0.8, 0.8)

    # Time settings
    times_per_second: float = 24.0       # FPS for animation
    use_time_samples: bool = True        # Use time sampling

    # Array prefixes
    point_array_prefix: str = "vtk_point_"
    cell_array_prefix: str = "vtk_cell_"
```

### MaterialData

Material properties:

```python
@dataclass
class MaterialData:
    name: str = "default_material"
    diffuse_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5
    use_vertex_colors: bool = False
```

### MeshData

Mesh geometry and data:

```python
@dataclass
class MeshData:
    points: NDArray                      # (N, 3) array
    face_vertex_counts: NDArray          # (F,) array
    face_vertex_indices: NDArray         # Flat array of indices
    normals: Optional[NDArray] = None    # (N, 3) or facevarying
    uvs: Optional[NDArray] = None        # (N, 2) texture coordinates
    colors: Optional[NDArray] = None     # (N, 3) or (N, 4) vertex colors
    generic_arrays: list[GenericArray] = []  # Data arrays
    material_id: str = "default_material"
```

### VTKToUSDConverter

Main converter class:

- `convert_file(vtk_file, output_usd, **kwargs)`: Convert single file
- `convert_sequence(vtk_files, output_usd, time_codes, **kwargs)`: Convert time series
- `convert_mesh_data(mesh_data, output_usd, **kwargs)`: Convert MeshData
- `convert_mesh_data_sequence(mesh_data_list, output_usd, **kwargs)`: Convert MeshData sequence

## Data Array Handling

VTK data arrays are automatically converted to USD primvars with appropriate types and interpolation:

### Point Data → Vertex Primvars
- Interpolation: `vertex`
- Naming: `vtk_point_<array_name>`
- Example: `vtk_point_pressure`, `vtk_point_temperature`

### Cell Data → Uniform Primvars
- Interpolation: `uniform` (per-face)
- Naming: `vtk_cell_<array_name>`
- Example: `vtk_cell_region_id`

### Type Mapping

| VTK Type     | USD Type    | Components |
| ------------ | ----------- | ---------- |
| Float/Double | FloatArray  | 1          |
| Float/Double | Float2Array | 2          |
| Float/Double | Float3Array | 3          |
| Float/Double | Float4Array | 4          |
| Int/Long     | IntArray    | 1-4        |
| UInt         | UIntArray   | 1+         |
| UChar/Char   | UCharArray  | 1+         |

## Coordinate System Conversion

The library automatically converts from RAS (Right-Anterior-Superior) coordinate system used in medical imaging to USD's Y-up coordinate system:

**RAS (Medical Imaging):**
- X: Patient's right
- Y: Patient's anterior (front)
- Z: Patient's superior (head)

**USD Y-up:**
- X: Right
- Y: Up
- Z: Back (toward camera)

**Conversion:** `USD(x, y, z) = RAS(x, z, -y)`

## Design Principles

Based on ParaViewConnector but adapted for file-based conversion:

1. **No Omniverse Dependencies**: Pure file-based USD output
2. **No ParaView/Qt**: Direct VTK API usage
3. **Modular Architecture**: Separate concerns (reading, conversion, materials)
4. **Data Preservation**: All VTK arrays preserved as primvars
5. **Standards Compliant**: Uses UsdPreviewSurface and standard USD schemas

## Comparison with ParaViewConnector

| Feature          | ParaViewConnector       | vtk_to_usd              |
| ---------------- | ----------------------- | ----------------------- |
| **Input**        | ParaView proxies        | VTK files               |
| **Output**       | Omniverse/Files         | Files only              |
| **Dependencies** | ParaView, Qt, Omniverse | VTK, USD                |
| **Use Case**     | Interactive pipeline    | Batch conversion        |
| **Materials**    | MDL + PreviewSurface    | PreviewSurface          |
| **Time Series**  | Full clip system        | Time-sampled attributes |
| **Volumes**      | OpenVDB support         | Surface extraction      |

## Examples

See `experiments/convert_vtk_to_usd_lib/test_vtk_to_usd_converter.ipynb` for comprehensive examples including:

1. Basic file conversion
2. Data array inspection
3. Custom materials and settings
4. Time-series animation
5. USD file verification

## Testing

Run the test notebook to verify the installation:

```bash
cd experiments/convert_vtk_to_usd_lib
jupyter notebook test_vtk_to_usd_converter.ipynb
```

## Known Limitations

1. **Volumetric Meshes**: VTU files are converted to surface meshes (via extract_surface)
2. **Complex Materials**: Only UsdPreviewSurface supported (no MDL)
3. **Topology Changes**: Time-varying topology requires separate prims per frame
4. **Large Datasets**: Memory-limited (entire mesh loaded at once)

## Future Enhancements

Potential improvements based on ParaViewConnector:

- [ ] OpenVDB volume support
- [ ] Clip-based time management for varying topology
- [ ] MDL material support
- [ ] Texture coordinate generation
- [ ] Point cloud support (UsdGeomPoints)
- [ ] Curve/line support (UsdGeomBasisCurves)
- [ ] Streaming for large datasets

## License

Part of the PhysioMotion4D project.

## References

- ParaViewConnector: https://github.com/NVIDIA-Omniverse/ParaViewConnector
- USD Documentation: https://openusd.org/
- VTK Documentation: https://vtk.org/
