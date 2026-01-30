"""Unified VTK to USD converter with advanced features.

This module provides a high-level interface for converting VTK/PyVista meshes to USD,
with support for:
- Time-series animation
- Anatomical region labeling (mask_ids)
- Colormap visualization
- Automatic topology change detection
- Both surface and volumetric meshes

Uses the vtk_to_usd library internally for core conversion functionality.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
import vtk
from pxr import Usd, UsdGeom

from .physiomotion4d_base import PhysioMotion4DBase
from .vtk_to_usd import (
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    MaterialManager,
    MeshData,
    UsdMeshConverter,
)


class ConvertVTKToUSD(PhysioMotion4DBase):
    """
    Advanced VTK to USD converter with colormap and anatomical labeling support.

    This class extends the basic vtk_to_usd library with:
    - Support for VTK/PyVista objects (not just files)
    - Anatomical region labeling via mask_ids
    - Colormap-based visualization
    - Automatic topology change detection
    - Time-series animation

    Example Usage:
        >>> # Create converter with time-series meshes
        >>> converter = ConvertVTKToUSD(
        ...     data_basename='CardiacModel',
        ...     input_polydata=meshes,  # List of PyVista/VTK meshes
        ...     mask_ids={1: 'ventricle', 2: 'atrium'}
        ... )
        >>>
        >>> # Configure colormap visualization
        >>> converter.set_colormap(
        ...     color_by_array='transmembrane_potential',
        ...     colormap='rainbow',
        ...     intensity_range=(-80.0, 20.0)
        ... )
        >>>
        >>> # Convert to USD
        >>> stage = converter.convert('output.usd')
    """

    def __init__(
        self,
        data_basename: str,
        input_polydata: Sequence[pv.DataSet | vtk.vtkDataSet],
        mask_ids: Optional[dict[int, str]] = None,
        compute_normals: bool = False,
        convert_to_surface: bool = True,
        times_per_second: float = 24.0,
        log_level: int | str = logging.INFO,
    ) -> None:
        """
        Initialize converter.

        Args:
            data_basename: Base name for USD data (used in prim paths)
            input_polydata: Sequence of PyVista/VTK meshes (one per time step)
            mask_ids: Optional mapping of label IDs to anatomical region names.
                     If provided, meshes will be split by labeled regions.
            compute_normals: Whether to compute vertex normals
            convert_to_surface: If True, extract surface from volumetric meshes
            times_per_second: Time codes per second (default 24.0).
                            For medical imaging time series where each frame = 1 second, use 1.0.
            log_level: Logging level
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.data_basename = data_basename
        self.input_polydata = list(input_polydata)
        self.mask_ids = mask_ids
        self.compute_normals = compute_normals
        self.convert_to_surface = convert_to_surface

        # Colormap settings
        self.color_by_array: Optional[str] = None
        self.colormap: str = "plasma"
        self.intensity_range: Optional[tuple[float, float]] = None

        # Conversion settings
        self.settings = ConversionSettings(
            triangulate_meshes=True,
            compute_normals=compute_normals,
            preserve_point_arrays=True,
            preserve_cell_arrays=True,
            meters_per_unit=1.0,
            up_axis="Y",
            times_per_second=times_per_second,
        )

        self.logger.info(
            f"Initialized converter with {len(input_polydata)} time steps, "
            f"mask_ids={'enabled' if mask_ids else 'disabled'}"
        )

    def supports_mesh_type(self, mesh: pv.DataSet | vtk.vtkDataSet) -> bool:
        """
        Check if mesh type is supported for conversion.

        Args:
            mesh: PyVista or VTK mesh to check

        Returns:
            bool: True if mesh type is supported
        """
        # Wrap VTK objects
        if isinstance(
            mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid, vtk.vtkImageData)
        ):
            mesh = pv.wrap(mesh)

        # Support most PyVista types
        return isinstance(
            mesh,
            (
                pv.PolyData,
                pv.UnstructuredGrid,
                pv.StructuredGrid,
                pv.ImageData,
                pv.RectilinearGrid,
            ),
        )

    def list_available_arrays(self) -> dict:
        """
        List all point data arrays available across all time steps.

        Returns:
            dict: Dictionary with array names as keys and metadata as values.
                  Metadata includes: 'n_components', 'dtype', 'range', 'present_in_steps'
        """
        available_arrays: dict[str, dict[str, Any]] = {}

        for time_idx, mesh in enumerate(self.input_polydata):
            # Wrap VTK objects
            if isinstance(mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
                mesh = pv.wrap(mesh)

            # Get point data arrays
            if hasattr(mesh, "point_data"):
                for array_name in mesh.point_data.keys():
                    if array_name not in available_arrays:
                        array_data = mesh.point_data[array_name]
                        available_arrays[array_name] = {
                            "n_components": int(
                                array_data.shape[1] if array_data.ndim > 1 else 1
                            ),
                            "dtype": str(array_data.dtype),
                            "range": (
                                float(np.min(array_data)),
                                float(np.max(array_data)),
                            ),
                            "present_in_steps": [time_idx],
                        }
                    else:
                        array_data = mesh.point_data[array_name]
                        meta = available_arrays[array_name]
                        current_min, current_max = cast(
                            tuple[float, float], meta["range"]
                        )
                        meta["range"] = (
                            min(current_min, float(np.min(array_data))),
                            max(current_max, float(np.max(array_data))),
                        )
                        cast(list[int], meta["present_in_steps"]).append(time_idx)

        self.logger.debug(f"Found {len(available_arrays)} data arrays")
        return available_arrays

    def set_colormap(
        self,
        color_by_array: Optional[str] = None,
        colormap: str = "plasma",
        intensity_range: Optional[tuple[float, float]] = None,
    ) -> "ConvertVTKToUSD":
        """
        Configure colormap for visualization.

        Args:
            color_by_array: Name of point data array to visualize. If None, uses solid colors.
            colormap: Colormap name. Supports all matplotlib colormaps plus aliases:
                - 'plasma', 'viridis', 'inferno', 'magma' (perceptually uniform)
                - 'rainbow', 'jet' (spectral)
                - 'hot', 'heat' (heat map, 'heat' is alias for 'hot')
                - 'coolwarm', 'seismic' (diverging)
                - 'gray', 'grayscale', 'grey', 'greyscale' (grayscale)
                - 'random', 'tab20' (categorical/discrete colors)
            intensity_range: Manual (vmin, vmax) range. If None, auto-computed from data.

        Returns:
            self: For method chaining
        """
        self.color_by_array = color_by_array
        self.colormap = colormap
        self.intensity_range = intensity_range

        self.logger.info(
            f"Colormap configured: array='{color_by_array}', "
            f"colormap='{colormap}', range={intensity_range}"
        )

        return self

    def convert(
        self,
        output_usd_file: str,
        convert_to_surface: Optional[bool] = None,
        compute_normals: Optional[bool] = None,
    ) -> Usd.Stage:
        """
        Convert VTK meshes to USD.

        Args:
            output_usd_file: Path to output USD file
            convert_to_surface: Override convert_to_surface setting
            compute_normals: Override compute_normals setting

        Returns:
            Usd.Stage: Created USD stage

        Raises:
            ValueError: If no valid meshes found
        """
        if convert_to_surface is not None:
            self.convert_to_surface = convert_to_surface
        if compute_normals is not None:
            self.settings.compute_normals = compute_normals

        self.logger.info(
            f"Converting {len(self.input_polydata)} meshes to {output_usd_file}"
        )

        # Remove existing file
        output_path = Path(output_usd_file)
        if output_path.exists():
            output_path.unlink()
            self.logger.debug(f"Removed existing file: {output_path}")

        # Create USD stage
        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageMetersPerUnit(stage, self.settings.meters_per_unit)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Create root
        root_path = f"/World/{self.data_basename}"
        UsdGeom.Xform.Define(stage, root_path)
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # Set time range for animation
        if len(self.input_polydata) > 1:
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(len(self.input_polydata) - 1)
            stage.SetTimeCodesPerSecond(self.settings.times_per_second)

        # Initialize managers
        material_mgr = MaterialManager(stage)
        mesh_converter = UsdMeshConverter(stage, self.settings, material_mgr)

        # Process meshes
        if self.mask_ids:
            # Split by anatomical regions
            self._convert_with_labels(stage, root_path, material_mgr, mesh_converter)
        else:
            # Single mesh conversion
            self._convert_unified(stage, root_path, material_mgr, mesh_converter)

        # Save stage
        stage.Save()
        self.logger.info(f"Saved USD file: {output_path}")

        return stage

    def _convert_unified(
        self,
        stage: Usd.Stage,
        root_path: str,
        material_mgr: MaterialManager,
        mesh_converter: UsdMeshConverter,
    ) -> None:
        """Convert all meshes as a single unified mesh."""
        self.logger.debug("Converting as unified mesh (no label splitting)")

        # Convert meshes to MeshData
        mesh_data_sequence = []
        for time_idx, vtk_mesh in enumerate(self.input_polydata):
            mesh_data = self._vtk_to_mesh_data(vtk_mesh, time_idx)
            mesh_data_sequence.append(mesh_data)

        # Create material
        material = self._create_material_from_colormap("unified_material")

        # Convert to USD
        mesh_path = f"{root_path}/Mesh"
        if len(mesh_data_sequence) == 1:
            # Single frame
            mesh_data_sequence[0].material_id = material.name
            material_mgr.get_or_create_material(material)
            mesh_converter.create_mesh(
                mesh_data_sequence[0], mesh_path, bind_material=True
            )
        else:
            # Time series
            time_codes = [float(i) for i in range(len(mesh_data_sequence))]
            for md in mesh_data_sequence:
                md.material_id = material.name
            material_mgr.get_or_create_material(material)
            mesh_converter.create_time_varying_mesh(
                mesh_data_sequence, mesh_path, time_codes, bind_material=True
            )

    def _convert_with_labels(
        self,
        stage: Usd.Stage,
        root_path: str,
        material_mgr: MaterialManager,
        mesh_converter: UsdMeshConverter,
    ) -> None:
        """Convert meshes split by anatomical labels."""
        mask_ids = self.mask_ids
        assert mask_ids is not None
        self.logger.debug(f"Converting with {len(mask_ids)} anatomical labels")

        # Extract labeled meshes for each time step
        labeled_meshes_by_time = []
        for time_idx, vtk_mesh in enumerate(self.input_polydata):
            labeled_meshes = self._split_by_labels(vtk_mesh, time_idx)
            labeled_meshes_by_time.append(labeled_meshes)

        # Get all unique labels
        all_labels: set[str] = set()
        for labeled_meshes in labeled_meshes_by_time:
            all_labels.update(labeled_meshes.keys())

        # Convert each label separately
        for label_name in sorted(all_labels):
            self.logger.debug(f"Processing label: {label_name}")

            # Collect mesh data for this label across time
            label_mesh_sequence = []
            for labeled_meshes in labeled_meshes_by_time:
                if label_name in labeled_meshes:
                    label_mesh_sequence.append(labeled_meshes[label_name])
                else:
                    # Label not present in this time step - use empty mesh or skip
                    self.logger.warning(f"Label '{label_name}' missing in time step")

            if not label_mesh_sequence:
                continue

            # Create material for this label
            material = self._create_material_from_colormap(f"{label_name}_material")

            # Convert to USD
            mesh_path = f"{root_path}/{label_name}"
            if len(label_mesh_sequence) == 1:
                label_mesh_sequence[0].material_id = material.name
                material_mgr.get_or_create_material(material)
                mesh_converter.create_mesh(
                    label_mesh_sequence[0], mesh_path, bind_material=True
                )
            else:
                time_codes = [float(i) for i in range(len(label_mesh_sequence))]
                for md in label_mesh_sequence:
                    md.material_id = material.name
                material_mgr.get_or_create_material(material)
                mesh_converter.create_time_varying_mesh(
                    label_mesh_sequence, mesh_path, time_codes, bind_material=True
                )

    def _vtk_to_mesh_data(
        self, vtk_mesh: pv.DataSet | vtk.vtkDataSet, time_idx: int
    ) -> MeshData:
        """Convert VTK/PyVista mesh to MeshData."""
        # Wrap VTK objects
        if isinstance(vtk_mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
            vtk_mesh = pv.wrap(vtk_mesh)

        # Extract surface if needed
        if isinstance(vtk_mesh, pv.UnstructuredGrid) and self.convert_to_surface:
            vtk_mesh = vtk_mesh.extract_surface()

        # Get points
        points = np.array(vtk_mesh.points, dtype=np.float64)

        # Get faces
        if hasattr(vtk_mesh, "faces"):
            faces = vtk_mesh.faces
            # Parse VTK face format: [n_points, i0, i1, ..., n_points, j0, j1, ...]
            face_counts_list: list[int] = []
            face_indices_list: list[int] = []
            idx = 0
            while idx < len(faces):
                n = int(faces[idx])
                face_counts_list.append(n)
                face_indices_list.extend([int(v) for v in faces[idx + 1 : idx + 1 + n]])
                idx += n + 1
            face_counts = np.array(face_counts_list, dtype=np.int32)
            face_indices = np.array(face_indices_list, dtype=np.int32)
        else:
            # No faces - might be point cloud or volumetric
            raise ValueError("Mesh has no faces - surface extraction may be needed")

        # Get normals
        normals = None
        if "Normals" in vtk_mesh.point_data:
            normals = np.array(vtk_mesh.point_data["Normals"], dtype=np.float64)

        # Get colors if using colormap
        colors = None
        if self.color_by_array and self.color_by_array in vtk_mesh.point_data:
            colors = self._apply_colormap(vtk_mesh.point_data[self.color_by_array])

        # Extract generic arrays
        generic_arrays = []
        for array_name in vtk_mesh.point_data.keys():
            array_data = vtk_mesh.point_data[array_name]
            num_components = int(array_data.shape[1] if array_data.ndim > 1 else 1)

            # Determine data type
            if array_data.dtype in [np.float32, np.float64]:
                data_type = DataType.FLOAT
            elif array_data.dtype in [np.int32, np.int64]:
                data_type = DataType.INT
            else:
                data_type = DataType.FLOAT

            generic_arrays.append(
                GenericArray(
                    name=array_name,
                    data=array_data,
                    num_components=num_components,
                    data_type=data_type,
                    interpolation="vertex",
                )
            )

        return MeshData(
            points=points,
            face_vertex_counts=face_counts,
            face_vertex_indices=face_indices,
            normals=normals,
            colors=colors,
            generic_arrays=generic_arrays,
        )

    def _split_by_labels(
        self, vtk_mesh: pv.DataSet | vtk.vtkDataSet, time_idx: int
    ) -> dict[str, MeshData]:
        """Split mesh by anatomical labels."""
        mask_ids = self.mask_ids
        assert mask_ids is not None
        # Wrap VTK objects
        if isinstance(vtk_mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
            vtk_mesh = pv.wrap(vtk_mesh)

        # Extract surface if needed
        if isinstance(vtk_mesh, pv.UnstructuredGrid) and self.convert_to_surface:
            vtk_mesh = vtk_mesh.extract_surface()

        # Get boundary labels
        if "boundary_labels" not in vtk_mesh.cell_data:
            self.logger.warning("No 'boundary_labels' array found - using unified mesh")
            return {"default": self._vtk_to_mesh_data(vtk_mesh, time_idx)}

        label_array = vtk_mesh.cell_data["boundary_labels"]

        # Create submeshes for each label
        labeled_meshes = {}
        for label_id, label_name in mask_ids.items():
            # Extract cells with this label
            mask = label_array == label_id
            if not np.any(mask):
                continue

            # Create submesh
            cell_ids = np.where(mask)[0].astype(int).tolist()
            submesh = vtk_mesh.extract_cells(cell_ids)

            # Convert to MeshData
            labeled_meshes[label_name] = self._vtk_to_mesh_data(submesh, time_idx)

        return labeled_meshes

    def _apply_colormap(self, scalar_data: np.ndarray) -> np.ndarray:
        """Apply colormap to scalar data."""
        from matplotlib import colormaps

        # Map common/intuitive names to actual matplotlib colormap names
        colormap_aliases = {
            "heat": "hot",
            "grayscale": "gray",
            "greyscale": "grey",
            "jet": "jet",
            "random": "tab20",  # Good for categorical data
        }

        # Flatten to 1D if needed
        if scalar_data.ndim > 1:
            scalar_data = np.linalg.norm(scalar_data, axis=1)

        # Normalize
        if self.intensity_range:
            vmin, vmax = self.intensity_range
        else:
            vmin, vmax = np.min(scalar_data), np.max(scalar_data)

        if vmax > vmin:
            normalized = (scalar_data - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            normalized = np.ones_like(scalar_data) * 0.5

        # Get colormap name (use alias if available)
        cmap_name = colormap_aliases.get(self.colormap, self.colormap)

        # Apply colormap with fallback
        try:
            cmap = colormaps[cmap_name]
        except KeyError:
            self.logger.warning(
                f"Colormap '{self.colormap}' not found, falling back to 'viridis'"
            )
            cmap = colormaps["viridis"]

        colors_rgba = cmap(normalized)

        # Return RGB (drop alpha)
        return colors_rgba[:, :3].astype(np.float32)

    def _create_material_from_colormap(self, name: str) -> MaterialData:
        """Create material based on colormap settings."""
        if self.color_by_array:
            # Use vertex colors
            return MaterialData(
                name=name,
                diffuse_color=(0.8, 0.8, 0.8),
                roughness=0.5,
                metallic=0.0,
                use_vertex_colors=True,
            )
        else:
            # Use solid color
            return MaterialData(
                name=name,
                diffuse_color=(0.8, 0.8, 0.8),
                roughness=0.5,
                metallic=0.0,
                use_vertex_colors=False,
            )
