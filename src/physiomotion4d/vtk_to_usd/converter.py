"""Main VTK to USD converter interface.

Provides high-level API for converting VTK files to USD format.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pxr import Usd, UsdGeom

from .data_structures import ConversionSettings, MaterialData, MeshData
from .material_manager import MaterialManager
from .usd_mesh_converter import UsdMeshConverter
from .vtk_reader import read_vtk_file

logger = logging.getLogger(__name__)


class VTKToUSDConverter:
    """High-level converter for VTK files to USD.

    Provides simple API for converting single or multiple VTK files to USD format.
    Handles material creation, primvar mapping, and time-series data.

    Example:
        >>> converter = VTKToUSDConverter()
        >>> converter.convert_file('mesh.vtp', 'output.usd')

        >>> # Time-series conversion
        >>> converter = VTKToUSDConverter()
        >>> files = ['mesh_0.vtp', 'mesh_1.vtp', 'mesh_2.vtp']
        >>> converter.convert_sequence(files, 'output.usd')
    """

    def __init__(self, settings: Optional[ConversionSettings] = None) -> None:
        """Initialize converter.

        Args:
            settings: Optional conversion settings. If None, uses defaults.
        """
        self.settings = settings or ConversionSettings()
        self.stage: Optional[Usd.Stage] = None
        self.material_mgr: Optional[MaterialManager] = None
        self.mesh_converter: Optional[UsdMeshConverter] = None

    def convert_file(
        self,
        vtk_file: str | Path,
        output_usd: str | Path,
        mesh_name: str = "Mesh",
        material: Optional[MaterialData] = None,
        extract_surface: bool = True,
    ) -> Usd.Stage:
        """Convert a single VTK file to USD.

        Args:
            vtk_file: Path to VTK file (.vtk, .vtp, or .vtu)
            output_usd: Path to output USD file
            mesh_name: Name for the mesh in USD
            material: Optional material data. If None, uses default.
            extract_surface: For .vtu files, whether to extract surface

        Returns:
            Usd.Stage: Created USD stage
        """
        logger.info(f"Converting {vtk_file} to {output_usd}")

        # Read VTK file
        mesh_data = read_vtk_file(vtk_file, extract_surface=extract_surface)

        # Set material ID if provided
        if material is not None:
            mesh_data.material_id = material.name

        # Create USD stage
        self._create_stage(output_usd)
        stage = self.stage
        mesh_converter = self.mesh_converter
        material_mgr = self.material_mgr
        assert stage is not None
        assert mesh_converter is not None
        assert material_mgr is not None

        # Create material if provided
        if material is not None:
            material_mgr.get_or_create_material(material)

        # Create mesh
        mesh_path = f"/World/Meshes/{mesh_name}"
        self._ensure_parent_path(mesh_path)

        mesh_converter.create_mesh(mesh_data, mesh_path, bind_material=True)

        # Save stage
        stage.Save()
        logger.info(f"Saved USD file: {output_usd}")

        return stage

    def convert_sequence(
        self,
        vtk_files: list[str | Path],
        output_usd: str | Path,
        mesh_name: str = "Mesh",
        time_codes: Optional[list[float]] = None,
        material: Optional[MaterialData] = None,
        extract_surface: bool = True,
    ) -> Usd.Stage:
        """Convert a sequence of VTK files to time-varying USD.

        Args:
            vtk_files: List of VTK file paths (one per time step)
            output_usd: Path to output USD file
            mesh_name: Name for the mesh in USD
            time_codes: Optional list of time codes. If None, uses sequential integers.
            material: Optional material data. If None, uses default.
            extract_surface: For .vtu files, whether to extract surface

        Returns:
            Usd.Stage: Created USD stage
        """
        if len(vtk_files) == 0:
            raise ValueError("Empty file list")

        logger.info(f"Converting sequence of {len(vtk_files)} files to {output_usd}")

        # Generate time codes if not provided
        if time_codes is None:
            time_codes = [float(i) for i in range(len(vtk_files))]
        elif len(time_codes) != len(vtk_files):
            raise ValueError(
                f"Number of time codes ({len(time_codes)}) must match "
                f"number of files ({len(vtk_files)})"
            )

        # Read all mesh data
        mesh_data_sequence = []
        for vtk_file in vtk_files:
            mesh_data = read_vtk_file(vtk_file, extract_surface=extract_surface)
            if material is not None:
                mesh_data.material_id = material.name
            mesh_data_sequence.append(mesh_data)

        # Create USD stage
        self._create_stage(output_usd)
        stage = self.stage
        mesh_converter = self.mesh_converter
        material_mgr = self.material_mgr
        assert stage is not None
        assert mesh_converter is not None
        assert material_mgr is not None

        # Create material if provided
        if material is not None:
            material_mgr.get_or_create_material(material)

        # Set time range
        stage.SetStartTimeCode(time_codes[0])
        stage.SetEndTimeCode(time_codes[-1])
        stage.SetTimeCodesPerSecond(self.settings.times_per_second)

        # Create time-varying mesh
        mesh_path = f"/World/Meshes/{mesh_name}"
        self._ensure_parent_path(mesh_path)

        mesh_converter.create_time_varying_mesh(
            mesh_data_sequence, mesh_path, time_codes, bind_material=True
        )

        # Save stage
        stage.Save()
        logger.info(f"Saved USD file: {output_usd}")

        return stage

    def convert_mesh_data(
        self,
        mesh_data: MeshData,
        output_usd: str | Path,
        mesh_name: str = "Mesh",
        material: Optional[MaterialData] = None,
    ) -> Usd.Stage:
        """Convert MeshData directly to USD.

        Useful when you already have MeshData from other sources.

        Args:
            mesh_data: Mesh data to convert
            output_usd: Path to output USD file
            mesh_name: Name for the mesh in USD
            material: Optional material data

        Returns:
            Usd.Stage: Created USD stage
        """
        logger.info(f"Converting MeshData to {output_usd}")

        if material is not None:
            mesh_data.material_id = material.name

        # Create USD stage
        self._create_stage(output_usd)
        stage = self.stage
        mesh_converter = self.mesh_converter
        material_mgr = self.material_mgr
        assert stage is not None
        assert mesh_converter is not None
        assert material_mgr is not None

        # Create material if provided
        if material is not None:
            material_mgr.get_or_create_material(material)

        # Create mesh
        mesh_path = f"/World/Meshes/{mesh_name}"
        self._ensure_parent_path(mesh_path)

        mesh_converter.create_mesh(mesh_data, mesh_path, bind_material=True)

        # Save stage
        stage.Save()
        logger.info(f"Saved USD file: {output_usd}")

        return stage

    def convert_mesh_data_sequence(
        self,
        mesh_data_sequence: list[MeshData],
        output_usd: str | Path,
        mesh_name: str = "Mesh",
        time_codes: Optional[list[float]] = None,
        material: Optional[MaterialData] = None,
    ) -> Usd.Stage:
        """Convert sequence of MeshData to time-varying USD.

        Args:
            mesh_data_sequence: List of MeshData (one per time step)
            output_usd: Path to output USD file
            mesh_name: Name for the mesh in USD
            time_codes: Optional list of time codes
            material: Optional material data

        Returns:
            Usd.Stage: Created USD stage
        """
        if len(mesh_data_sequence) == 0:
            raise ValueError("Empty mesh data sequence")

        logger.info(
            f"Converting sequence of {len(mesh_data_sequence)} MeshData to {output_usd}"
        )

        # Generate time codes if not provided
        if time_codes is None:
            time_codes = [float(i) for i in range(len(mesh_data_sequence))]
        elif len(time_codes) != len(mesh_data_sequence):
            raise ValueError(
                f"Number of time codes ({len(time_codes)}) must match "
                f"number of mesh data ({len(mesh_data_sequence)})"
            )

        # Set material for all mesh data
        if material is not None:
            for mesh_data in mesh_data_sequence:
                mesh_data.material_id = material.name

        # Create USD stage
        self._create_stage(output_usd)
        stage = self.stage
        mesh_converter = self.mesh_converter
        material_mgr = self.material_mgr
        assert stage is not None
        assert mesh_converter is not None
        assert material_mgr is not None

        # Create material if provided
        if material is not None:
            material_mgr.get_or_create_material(material)

        # Set time range
        stage.SetStartTimeCode(time_codes[0])
        stage.SetEndTimeCode(time_codes[-1])
        stage.SetTimeCodesPerSecond(self.settings.times_per_second)

        # Create time-varying mesh
        mesh_path = f"/World/Meshes/{mesh_name}"
        self._ensure_parent_path(mesh_path)

        mesh_converter.create_time_varying_mesh(
            mesh_data_sequence, mesh_path, time_codes, bind_material=True
        )

        # Save stage
        stage.Save()
        logger.info(f"Saved USD file: {output_usd}")

        return stage

    def _create_stage(self, output_path: str | Path) -> None:
        """Create a new USD stage.

        Args:
            output_path: Path for the USD file
        """
        output_path = Path(output_path)

        # Remove existing file
        if output_path.exists():
            output_path.unlink()
            logger.debug(f"Removed existing file: {output_path}")

        # Create stage
        stage = Usd.Stage.CreateNew(str(output_path))
        assert stage is not None
        self.stage = stage

        # Set stage metadata
        UsdGeom.SetStageMetersPerUnit(stage, self.settings.meters_per_unit)
        if self.settings.up_axis.upper() == "Y":
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        else:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Create root
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # Initialize managers
        self.material_mgr = MaterialManager(stage)
        self.mesh_converter = UsdMeshConverter(stage, self.settings, self.material_mgr)

        logger.debug(f"Created USD stage: {output_path}")

    def _ensure_parent_path(self, path: str) -> None:
        """Ensure all parent prims in path exist.

        Args:
            path: USD path (e.g., "/World/Meshes/MyMesh")
        """
        parts = path.strip("/").split("/")
        current_path = ""
        stage = self.stage
        assert stage is not None
        for part in parts[:-1]:  # Skip the last part (the actual mesh)
            current_path += f"/{part}"
            if not stage.GetPrimAtPath(current_path):
                stage.DefinePrim(current_path, "Xform")


# Convenience functions


def convert_vtk_file(
    vtk_file: str | Path,
    output_usd: str | Path,
    settings: Optional[ConversionSettings] = None,
    **kwargs: Any,
) -> Usd.Stage:
    """Convenience function to convert a single VTK file.

    Args:
        vtk_file: Path to VTK file
        output_usd: Path to output USD file
        settings: Optional conversion settings
        **kwargs: Additional arguments passed to convert_file()

    Returns:
        Usd.Stage: Created USD stage
    """
    converter = VTKToUSDConverter(settings)
    return converter.convert_file(vtk_file, output_usd, **kwargs)


def convert_vtk_sequence(
    vtk_files: list[str | Path],
    output_usd: str | Path,
    settings: Optional[ConversionSettings] = None,
    **kwargs: Any,
) -> Usd.Stage:
    """Convenience function to convert a sequence of VTK files.

    Args:
        vtk_files: List of VTK file paths
        output_usd: Path to output USD file
        settings: Optional conversion settings
        **kwargs: Additional arguments passed to convert_sequence()

    Returns:
        Usd.Stage: Created USD stage
    """
    converter = VTKToUSDConverter(settings)
    return converter.convert_sequence(vtk_files, output_usd, **kwargs)
