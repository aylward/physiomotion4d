"""Unified facade for VTK to USD conversion supporting both PolyData and UnstructuredGrid."""

import logging

import pyvista as pv
import vtk
from pxr import Usd

from .convert_vtk_4d_to_usd_polymesh import ConvertVTK4DToUSDPolyMesh
from .convert_vtk_4d_to_usd_tetmesh import ConvertVTK4DToUSDTetMesh
from .physiomotion4d_base import PhysioMotion4DBase


class ConvertVTK4DToUSD(PhysioMotion4DBase):
    """
    Unified converter supporting both PolyData and UnstructuredGrid.

    Automatically routes meshes to appropriate specialized converter:
    - PolyData → ConvertVTK4DToUSDPolyMesh
    - UnstructuredGrid (volumetric) → ConvertVTK4DToUSDTetMesh
    - UnstructuredGrid (surface) → ConvertVTK4DToUSDPolyMesh

    Maintains API compatibility with original ConvertVTK4DToUSD class.

    Supports:
    - PolyData: Surface meshes exported as UsdGeomMesh
    - UnstructuredGrid:
        * By default: Volumetric tetrahedral meshes exported as UsdGeomTetMesh
                      (requires OpenUSD v24.03+)
        * With convert_to_surface=True: Extracted surface exported as UsdGeomMesh
                                        (compatible with all USD versions)
    - Time-varying topology: Automatically detects and handles topology changes
                            across time steps (varying number of points/faces).
                            When detected, creates separate mesh prims per timestep
                            with visibility control instead of time-sampled attributes.

    Example Usage:
        >>> # Create converter with meshes
        >>> converter = ConvertVTK4DToUSDAll(
        ...     data_basename="CardiacModel",
        ...     input_polydata=meshes,
        ...     mask_ids=None
        ... )
        >>>
        >>> # List available point data arrays
        >>> arrays = converter.list_available_arrays()
        >>> print(arrays.keys())  # ['transmembrane_potential', 'temperature', ...]
        >>>
        >>> # Configure colormap
        >>> converter.set_colormap(
        ...     color_by_array="transmembrane_potential",
        ...     colormap="rainbow",
        ...     intensity_range=(-80.0, 20.0)
        ... )
        >>>
        >>> # Convert to USD (automatically handles topology changes)
        >>> stage = converter.convert("output.usd")
    """

    def __init__(
        self,
        data_basename,
        input_polydata,
        mask_ids=None,
        compute_normals=False,
        convert_to_surface=False,
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize converter and store parameters for later routing.

        Args:
            data_basename (str): Base name for the USD data
            input_polydata (list): List of PyVista PolyData or UnstructuredGrid meshes,
                                   one per time step. For UnstructuredGrid, tetrahedral
                                   cells will be exported as UsdGeomTetMesh by default.
            mask_ids (dict or None): Optional mapping of label IDs to label names for
                                     organizing meshes by anatomical regions.
                                     Default: None
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.data_basename = data_basename
        self.input_polydata = input_polydata
        self.mask_ids = mask_ids

        self.compute_normals = compute_normals
        self.convert_to_surface = convert_to_surface

        # Colormap settings (will be applied to specialized converter)
        self.color_by_array = None
        self.colormap = 'plasma'
        self.intensity_range = None

        # Flag for surface conversion
        self.convert_to_surface = False

    def list_available_arrays(self):
        """
        List all point data arrays available for coloring across all time steps.

        Creates a temporary PolyMesh converter to analyze arrays since the
        method is common to both mesh types.

        Returns:
            dict: Dictionary with array names as keys and dict of metadata as values.
                  Metadata includes: 'n_components', 'dtype', 'range', 'present_in_steps'
        """
        # Create temporary converter to analyze arrays
        temp_converter = ConvertVTK4DToUSDPolyMesh(
            self.data_basename, self.input_polydata, self.mask_ids
        )
        return temp_converter.list_available_arrays()

    def set_colormap(
        self, color_by_array=None, colormap='plasma', intensity_range=None
    ):
        """
        Configure colormap settings for vertex coloring.

        Settings are stored and will be applied to the specialized converter
        during convert().

        Args:
            color_by_array (str or None): Name of point data array to use for
                                          vertex colors. If None, uses fixed label
                                          colors. Use list_available_arrays() to see
                                          available options.
            colormap (str): Colormap to use for color_by_array visualization.
                           Available options: 'plasma', 'viridis', 'rainbow', 'heat',
                           'coolwarm', 'grayscale', 'random'
            intensity_range (tuple or None): Manual intensity range (vmin, vmax) for
                                            colormap. If None, uses automatic range
                                            from data.

        Returns:
            self: Returns self for method chaining
        """
        self.color_by_array = color_by_array
        self.colormap = colormap
        self.intensity_range = intensity_range
        return self

    def convert(
        self, output_usd_file, convert_to_surface=None, compute_normals=None
    ) -> Usd.Stage:
        """
        Convert meshes to USD, automatically routing by mesh type.

        Analyzes input meshes and routes to appropriate specialized converter:
        1. Only PolyData → ConvertVTK4DToUSDPolyMesh
        2. Only UnstructuredGrid (volumetric) → ConvertVTK4DToUSDTetMesh
        3. Only UnstructuredGrid (surface mode) → ConvertVTK4DToUSDPolyMesh
        4. Mixed types → NotImplementedError (use original class)

        Args:
            output_usd_file (str): Path to output USD file
            convert_to_surface (bool): If True, convert UnstructuredGrid to surface
                                       PolyData before processing. Useful for compatibility
                                       with older USD versions or when volumetric data is
                                       not needed. Default: False (preserve volumetric data)

        Returns:
            Usd.Stage: The created USD stage

        Raises:
            NotImplementedError: If mixed mesh types are detected
            ValueError: If no valid mesh data found
        """
        if convert_to_surface is not None:
            self.convert_to_surface = convert_to_surface
        if compute_normals is not None:
            self.compute_normals = compute_normals

        # Analyze mesh types in input
        has_polydata = False
        has_ugrid = False

        for mesh in self.input_polydata:
            if isinstance(mesh, (pv.PolyData, vtk.vtkPolyData)):
                has_polydata = True
            elif isinstance(mesh, (pv.UnstructuredGrid, vtk.vtkUnstructuredGrid)):
                if convert_to_surface:
                    has_polydata = True
                else:
                    has_ugrid = True

        # Case 1: Only PolyData (or surface-converted UGrid)
        if has_polydata and not has_ugrid:
            self.log_info("Routing to PolyMesh converter (surface meshes)")
            converter = ConvertVTK4DToUSDPolyMesh(
                self.data_basename,
                self.input_polydata,
                self.mask_ids,
                convert_to_surface=self.convert_to_surface,
                compute_normals=self.compute_normals,
                log_level=self.log_level,
            )
            converter.set_colormap(
                self.color_by_array, self.colormap, self.intensity_range
            )
            return converter.convert(output_usd_file)

        # Case 2: Only UnstructuredGrid (tetmesh)
        elif has_ugrid and not has_polydata:
            self.log_info("Routing to TetMesh converter (volumetric meshes)")
            converter = ConvertVTK4DToUSDTetMesh(
                self.data_basename,
                self.input_polydata,
                self.mask_ids,
                convert_to_surface=self.convert_to_surface,
                compute_normals=self.compute_normals,
                log_level=self.log_level,
            )
            converter.set_colormap(
                self.color_by_array, self.colormap, self.intensity_range
            )
            return converter.convert(output_usd_file)

        # Case 3: Mixed - need custom handling
        elif has_polydata and has_ugrid:
            raise NotImplementedError(
                "Mixed PolyData and UnstructuredGrid not yet supported in "
                "refactored version. Please use one of the following solutions:\n"
                "1. Use the original ConvertVTK4DToUSD class from "
                "convert_vtk_4d_to_usd.py\n"
                "2. Separate your meshes by type and convert in two passes\n"
                "3. Convert all UnstructuredGrid to surface with convert_to_surface=True"
            )

        # Case 4: No valid meshes
        else:
            raise ValueError("No valid mesh data found in input_polydata")
