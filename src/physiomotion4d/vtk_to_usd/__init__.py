"""VTK to USD conversion library.

A comprehensive library for converting VTK files (VTK, VTP, VTU) to USD format.
Based on the ParaViewConnector architecture but simplified for file-based conversion.

Features:
- Supports VTK legacy format (.vtk), XML PolyData (.vtp), and UnstructuredGrid (.vtu)
- Preserves geometry, topology, normals, and colors
- Converts VTK data arrays to USD primvars
- Supports time-series/animated data
- Material system with UsdPreviewSurface
- Coordinate conversion from RAS to USD Y-up

Example Usage:
    >>> from physiomotion4d.vtk_to_usd import convert_vtk_file
    >>> stage = convert_vtk_file('mesh.vtp', 'output.usd')

    >>> # Advanced usage with custom settings
    >>> from physiomotion4d.vtk_to_usd import VTKToUSDConverter, ConversionSettings
    >>> settings = ConversionSettings(triangulate_meshes=True, compute_normals=True)
    >>> converter = VTKToUSDConverter(settings)
    >>> converter.convert_sequence(['mesh_0.vtp', 'mesh_1.vtp'], 'output.usd')
"""

from .converter import VTKToUSDConverter, convert_vtk_file, convert_vtk_sequence
from .data_structures import (
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    MeshData,
    TimeStepData,
    VolumeData,
)
from .material_manager import MaterialManager
from .usd_mesh_converter import UsdMeshConverter
from .usd_utils import (
    compute_mesh_extent,
    create_primvar,
    ras_normals_to_usd,
    ras_points_to_usd,
    ras_to_usd,
    sanitize_primvar_name,
    triangulate_face,
)
from .vtk_reader import (
    LegacyVTKReader,
    PolyDataReader,
    UnstructuredGridReader,
    VTKReader,
    read_vtk_file,
    validate_time_series_topology,
)

__all__ = [
    # Main converter
    "VTKToUSDConverter",
    "convert_vtk_file",
    "convert_vtk_sequence",
    # Data structures
    "ConversionSettings",
    "DataType",
    "GenericArray",
    "MaterialData",
    "MeshData",
    "TimeStepData",
    "VolumeData",
    # Managers
    "MaterialManager",
    "UsdMeshConverter",
    # Utilities
    "ras_to_usd",
    "ras_points_to_usd",
    "ras_normals_to_usd",
    "create_primvar",
    "sanitize_primvar_name",
    "triangulate_face",
    "compute_mesh_extent",
    # Readers
    "VTKReader",
    "PolyDataReader",
    "LegacyVTKReader",
    "UnstructuredGridReader",
    "read_vtk_file",
    "validate_time_series_topology",
]

__version__ = "0.1.0"
