"""Abstract base class for converting 4D VTK data to animated USD meshes."""

import logging
import os
import time
from abc import ABC, abstractmethod

import cupy as cp
import matplotlib.cm as cm
import numpy as np
import pyvista as pv
import vtk
from pxr import Gf, Usd, UsdGeom, Vt

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase

# VTK Cell Type Constants
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14


class ConvertVTK4DToUSDBase(PhysioMotion4DBase, ABC):
    """
    Abstract base class for VTK to USD conversion.

    Provides shared utilities for coordinate conversion, colormap handling,
    topology change detection, and normal computation. Subclasses must implement
    mesh-specific processing and USD creation methods.
    """

    def __init__(
        self,
        data_basename,
        input_polydata,
        mask_ids=None,
        convert_to_surface=False,
        compute_normals=False,
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize VTK to USD converter.

        Args:
            data_basename (str): Base name for the USD data
            input_polydata (list): List of PyVista PolyData or UnstructuredGrid meshes,
                                   one per time step.
            mask_ids (dict or None): Optional mapping of label IDs to label names for
                                     organizing meshes by anatomical regions.
                                     Default: None
            convert_to_surface (bool): If True, convert UnstructuredGrid meshes to surface
                                      PolyData before processing. Only applicable for PolyMesh
                                      converter. Default: False
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.data_basename = data_basename
        self.input_polydata = input_polydata
        self.mask_ids = mask_ids

        self.convert_to_surface = convert_to_surface

        self.compute_normals = compute_normals

        # Colormap settings (set via set_colormap())
        self.color_by_array = None
        self.colormap = 'plasma'
        self.intensity_range = None

        self.times = list(range(len(input_polydata)))
        self.stage = None

        # Define a set of distinct colors for different objects
        self.colors = [
            (0.7, 0.3, 0.3),  # Red
            (0.3, 0.7, 0.3),  # Green
            (0.3, 0.3, 0.7),  # Blue
            (0.7, 0.7, 0.3),  # Yellow
            (0.7, 0.3, 0.7),  # Magenta
            (0.3, 0.7, 0.7),  # Cyan
            (0.7, 0.5, 0.3),  # Orange
            (0.5, 0.3, 0.7),  # Purple
            (0.3, 0.5, 0.5),  # Teal
            (0.5, 0.5, 0.3),  # Olive
        ]

    def list_available_arrays(self):
        """
        List all point data arrays available for coloring across all time steps.

        Returns:
            dict: Dictionary with array names as keys and dict of metadata as values.
                  Metadata includes: 'n_components', 'dtype', 'range', 'present_in_steps'
        """
        available_arrays = {}

        for time_idx, mesh in enumerate(self.input_polydata):
            # Convert to PyVista if needed
            if isinstance(mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
                mesh = pv.wrap(mesh)

            # Get point data array names
            for array_name in mesh.point_data.keys():
                if array_name not in available_arrays:
                    array_data = mesh.point_data[array_name]
                    available_arrays[array_name] = {
                        'n_components': (
                            array_data.shape[1] if len(array_data.shape) > 1 else 1
                        ),
                        'dtype': str(array_data.dtype),
                        'range': (float(np.min(array_data)), float(np.max(array_data))),
                        'present_in_steps': [time_idx],
                    }
                else:
                    # Update range and track presence
                    array_data = mesh.point_data[array_name]
                    current_min, current_max = available_arrays[array_name]['range']
                    available_arrays[array_name]['range'] = (
                        min(current_min, float(np.min(array_data))),
                        max(current_max, float(np.max(array_data))),
                    )
                    available_arrays[array_name]['present_in_steps'].append(time_idx)

        return available_arrays

    def set_colormap(
        self, color_by_array=None, colormap='plasma', intensity_range=None
    ):
        """
        Configure colormap settings for vertex coloring.

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

        """Validate that the chosen colormap is supported"""
        supported_colormaps = [
            'plasma',
            'viridis',
            'rainbow',
            'heat',
            'coolwarm',
            'grayscale',
            'random',
        ]
        if self.colormap not in supported_colormaps:
            raise ValueError(
                f"Unsupported colormap '{self.colormap}'. "
                f"Choose from: {', '.join(supported_colormaps)}"
            )

        # Initialize random seed for reproducible random colormap
        if self.colormap == 'random':
            np.random.seed(42)

        return self

    def _ras_to_usd(self, point):
        """Convert RAS coordinates to USD's right-handed Y-up system"""
        return Gf.Vec3f(float(point[0]), float(point[2]), float(-point[1]))

    def _get_matplotlib_colormap(self, colormap_name):
        """
        Get matplotlib colormap object, with custom implementations for special cases.

        Args:
            colormap_name (str): Name of the colormap

        Returns:
            matplotlib colormap object
        """
        colormap_mapping = {
            'plasma': 'plasma',
            'viridis': 'viridis',
            'rainbow': 'rainbow',
            'heat': 'hot',
            'coolwarm': 'coolwarm',
            'grayscale': 'gray',
        }

        if colormap_name == 'random':
            # Random colormap will be handled separately
            return None

        mpl_name = colormap_mapping.get(colormap_name, colormap_name)
        return cm.get_cmap(mpl_name)

    def _map_scalar_to_color(self, scalar_value, vmin, vmax, colormap='plasma'):
        """
        Map a scalar value to RGB color using a colormap.

        Args:
            scalar_value: The scalar value to map
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            colormap: Colormap name (default: 'plasma')

        Returns:
            Gf.Vec3f: RGB color value
        """
        if type(scalar_value) is list:
            scalar_value = np.linalg.norm(np.array(scalar_value))
        elif type(scalar_value) is np.ndarray:
            scalar_value = np.linalg.norm(scalar_value)
        elif type(scalar_value) is tuple:
            scalar_value = np.linalg.norm(np.array(scalar_value))
        elif type(scalar_value) is dict:
            scalar_value = np.linalg.norm(np.array(scalar_value.values()))
        elif type(scalar_value) is str:
            scalar_value = np.linalg.norm(np.array(scalar_value))

        # Handle random colormap specially
        if colormap == 'random':
            # Use hash of value to get consistent random color
            hash_val = hash(float(scalar_value))
            np.random.seed(hash_val % (2**32))
            rgb = np.random.rand(3)
            return Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))

        # Normalize value to [0, 1]
        if vmax > vmin:
            normalized = (float(scalar_value) - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            normalized = 0.5

        # Get colormap
        cmap = self._get_matplotlib_colormap(colormap)
        rgba = cmap(normalized)

        return Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2]))

    def _compute_intensity_range(self, mesh_time_data, label):
        """
        Compute the intensity range for colormap mapping.

        Args:
            mesh_time_data (dict): Time-series mesh data
            label (str): Label identifier for the mesh

        Returns:
            tuple: (vmin, vmax) intensity range
        """
        if self.intensity_range is not None:
            # Use user-specified range
            return self.intensity_range

        # Compute automatic range from data
        all_values = []
        for time_idx in range(len(self.times)):
            time_data = mesh_time_data[time_idx][label]
            if time_data.get('color_array') is not None:
                color_values = time_data['color_array']
                if isinstance(color_values, np.ndarray):
                    all_values.extend(color_values.flatten())
                else:
                    all_values.extend(color_values)

        if len(all_values) > 0:
            return (float(np.min(all_values)), float(np.max(all_values)))
        else:
            # Fallback
            return (0.0, 1.0)

    def _extract_color_array(self, mesh):
        """
        Extract color array data from mesh point data.

        Args:
            mesh: PyVista mesh (PolyData or UnstructuredGrid)

        Returns:
            numpy.ndarray or None: Array of scalar values for coloring
        """
        if self.color_by_array is None:
            return None

        # Convert VTK to PyVista if needed
        if isinstance(mesh, (vtk.vtkPolyData, vtk.vtkUnstructuredGrid)):
            mesh = pv.wrap(mesh)

        # Check if array exists in point data
        if self.color_by_array in mesh.point_data:
            color_values = []
            for scalar_value in mesh.point_data[self.color_by_array]:
                if isinstance(scalar_value, list):
                    scalar_value = np.linalg.norm(np.array(scalar_value))
                elif isinstance(scalar_value, np.ndarray):
                    scalar_value = np.linalg.norm(scalar_value)
                elif isinstance(scalar_value, tuple):
                    scalar_value = np.linalg.norm(np.array(scalar_value))
                elif isinstance(scalar_value, dict):
                    scalar_value = np.linalg.norm(np.array(scalar_value.values()))
                elif isinstance(scalar_value, str):
                    scalar_value = np.linalg.norm(np.array(scalar_value))
                else:
                    scalar_value = float(scalar_value)
                color_values.append(scalar_value)
            return np.array(color_values)
        else:
            self.log_warning("Array '%s' not found in point data", self.color_by_array)
            return None

    def _check_topology_changes(self, mesh_time_data):
        """
        Check if mesh topology changes across time steps.

        Args:
            mesh_time_data (dict): Dictionary mapping time_idx -> label -> mesh_data

        Returns:
            dict: Dictionary mapping label -> bool (True if topology changes)
        """
        topology_changes = {}

        # Get all labels from first time step
        first_time = mesh_time_data[0]
        labels = list(first_time.keys())

        for label in labels:
            has_change = False
            first_data = mesh_time_data[0][label]
            mesh_type = first_data.get('mesh_type', 'polymesh')

            # Get reference topology from first timestep
            if mesh_type == 'polymesh':
                ref_num_points = len(first_data['points'])
                ref_num_faces = len(first_data['face_vertex_counts'])
            elif mesh_type == 'tetmesh':
                ref_num_points = len(first_data['points'])
                ref_num_tets = len(first_data['tet_indices'])
            else:
                # Unknown mesh type, assume no change
                topology_changes[label] = False
                continue

            # Check all subsequent time steps
            for time_idx in range(1, len(mesh_time_data)):
                if label not in mesh_time_data[time_idx]:
                    # Label doesn't exist in this timestep - topology change
                    has_change = True
                    break

                curr_data = mesh_time_data[time_idx][label]

                if mesh_type == 'polymesh':
                    curr_num_points = len(curr_data['points'])
                    curr_num_faces = len(curr_data['face_vertex_counts'])
                    if (
                        curr_num_points != ref_num_points
                        or curr_num_faces != ref_num_faces
                    ):
                        has_change = True
                        break
                elif mesh_type == 'tetmesh':
                    curr_num_points = len(curr_data['points'])
                    curr_num_tets = len(curr_data['tet_indices'])
                    if (
                        curr_num_points != ref_num_points
                        or curr_num_tets != ref_num_tets
                    ):
                        has_change = True
                        break

            topology_changes[label] = has_change

            if has_change:
                self.log_info(
                    "Detected topology changes for label '%s' - will use time-varying mesh approach",
                    label,
                )

        return topology_changes

    def _compute_facevarying_normals_tri(
        self, points_vt, faceCounts_vt, faceIndices_vt
    ):
        """
        Vectorized face-varying normals for a triangulated mesh.

        points_vt: Vt.Vec3fArray
        faceCounts_vt: Vt.IntArray (all must be 3)
        faceIndices_vt: Vt.IntArray (len == 3 * numFaces)

        Returns: Vt.Vec3fArray of length len(faceIndices_vt), one normal per corner.
        """

        # Convert Vt arrays to NumPy
        points = np.array(points_vt).astype(np.float32)  # (N, 3)
        counts = np.array(faceCounts_vt).astype(np.int32)  # (F,)
        indices = np.array(faceIndices_vt).astype(np.int32)  # (3F,)

        # Sanity: assume triangulated mesh
        if not np.all(counts == 3):
            raise ValueError(
                "Mesh must be fully triangulated (all faceVertexCounts == 3)"
            )

        # Reshape indices into (F, 3)
        faces = indices.reshape(-1, 3)  # (F, 3)

        # Gather per-face vertex positions (F, 3, 3)
        tris = points[faces]  # (F, 3, 3)

        # Compute normals via vectorized cross product
        v1 = tris[:, 1] - tris[:, 0]  # (F, 3)
        v2 = tris[:, 2] - tris[:, 0]  # (F, 3)
        v1 = cp.array(v1)
        v2 = cp.array(v2)
        n = cp.cross(v1, v2)  # (F, 3)

        # Normalize
        lengths = cp.linalg.norm(n, axis=1, keepdims=True)  # (F, 1)
        mask = lengths[:, 0] > 0
        n[mask] /= lengths[mask]

        # Broadcast each face normal to 3 corners -> (F, 3, 3), then flatten
        n_fv = cp.repeat(n[:, cp.newaxis, :], 3, axis=1).reshape(-1, 3)  # (3F, 3)

        # Convert back to Vt.Vec3fArray
        fv_vt = Vt.Vec3fArray.FromNumpy(n_fv.get())

        return fv_vt

    # Abstract methods that subclasses must implement

    @abstractmethod
    def supports_mesh_type(self, mesh) -> bool:
        """
        Check if this converter supports the given mesh type.

        Args:
            mesh: PyVista or VTK mesh object

        Returns:
            bool: True if this converter can process the mesh
        """
        pass

    @abstractmethod
    def _process_mesh_data(self, mesh) -> dict:
        """
        Process mesh and extract geometry data.

        Args:
            mesh: PyVista or VTK mesh object

        Returns:
            dict: Processed mesh data organized by labels or mesh type
        """
        pass

    @abstractmethod
    def _create_usd_mesh(
        self, transform_path, label, mesh_time_data, label_colors, has_topology_change
    ):
        """
        Create USD mesh prim(s) for this label.

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
            has_topology_change: Whether topology varies over time
        """
        pass

    def convert(
        self, output_usd_file, convert_to_surface=None, compute_normals=None
    ) -> Usd.Stage:
        """
        Convert VTK meshes to USD format.

        Args:
            output_usd_file (str): Path to output USD file
            convert_to_surface (bool or None): If True, convert UnstructuredGrid to surface
                                       PolyData before processing. If None, uses the value
                                       set in __init__. Default: None

        Returns:
            Usd.Stage: The created USD stage
        """
        # Only override if explicitly provided
        if convert_to_surface is not None:
            self.convert_to_surface = convert_to_surface

        if compute_normals is not None:
            self.compute_normals = compute_normals

        # Remove existing file if it exists to avoid USD layer conflicts
        if os.path.exists(output_usd_file):
            os.remove(output_usd_file)

        # Create USD stage
        self.stage = Usd.Stage.CreateNew(output_usd_file)
        # Set the stage's linear scale to meters
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)

        # Create a parent scope for all meshes
        root_path = f"/World/{self.data_basename}"
        UsdGeom.Xform.Define(self.stage, root_path)

        basename = os.path.basename(output_usd_file).split(".")[0]
        self.log_info("Converting %s", basename)
        root_path = f"{root_path}/Transform_{basename}"
        UsdGeom.Xform.Define(self.stage, root_path)

        root_scope = UsdGeom.Scope.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(root_scope.GetPrim())

        # Collect the label data from each time point
        polydata_time_data = {}
        for fnum, mesh_data in enumerate(self.input_polydata):
            self.log_progress(
                fnum + 1, len(self.input_polydata), prefix="Processing time point"
            )
            polydata_time_data[fnum] = self._process_mesh_data(mesh_data)

        # Check for topology changes across time steps
        topology_changes = self._check_topology_changes(polydata_time_data)

        # Assign a unique color to each label
        label_colors = {}
        for fnum in range(len(polydata_time_data)):
            for label, _ in polydata_time_data[fnum].items():
                if label not in label_colors:
                    label_colors[label] = self.colors[
                        len(label_colors) % len(self.colors)
                    ]

        # Process first polydata to get label groups
        first_data = polydata_time_data[0]

        # Create a mesh prim for each label group
        num_labels = len(first_data.items())
        for idx, (label, data) in enumerate(first_data.items()):
            # Create a transform for each mesh
            transform_path = f"{root_path}/Transform_{label}"
            UsdGeom.Xform.Define(self.stage, transform_path)

            # Determine if topology changes for this label
            has_topology_change = topology_changes.get(label, False)

            # Call subclass-specific USD mesh creation
            start_time = time.time()
            self._create_usd_mesh(
                transform_path,
                label,
                polydata_time_data,
                label_colors,
                has_topology_change,
            )
            end_time = time.time()
            self.log_info(
                "Time taken to create USD mesh: %s seconds", end_time - start_time
            )

        # Set time range for the stage
        self.stage.SetStartTimeCode(self.times[0])
        self.stage.SetEndTimeCode(self.times[-1])
        self.stage.SetTimeCodesPerSecond(1.0)

        self.stage.Save()
        return self.stage
