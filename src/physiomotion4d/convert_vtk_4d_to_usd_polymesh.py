"""Converter for VTK PolyData to USD Mesh with surface meshes."""

import itertools
import time

import pyvista as pv
import vtk
from pxr import Gf, Sdf, UsdGeom
from typing import cast

from .convert_vtk_4d_to_usd_base import (
    ConvertVTK4DToUSDBase,
    MeshLabelData,
    MeshTimeData,
    RgbColor,
)


class ConvertVTK4DToUSDPolyMesh(ConvertVTK4DToUSDBase):
    """
    Converter for VTK PolyData to USD Mesh.

    Handles:
    - Surface meshes (PolyData)
    - UnstructuredGrid converted to surface
    - Time-varying topology via visibility control
    - Per-vertex colormap visualization

    Example Usage:
        >>> converter = ConvertVTK4DToUSDPolyMesh(
        ...     data_basename='SurfaceModel', input_polydata=meshes, mask_ids=None
        ... )
        >>> converter.set_colormap(color_by_array='pressure', colormap='rainbow')
        >>> stage = converter.convert('output.usd')
    """

    def supports_mesh_type(self, mesh: pv.DataSet | vtk.vtkDataSet) -> bool:
        """
        Check if this converter supports the given mesh type.

        Supports:
        - PolyData meshes
        - UnstructuredGrid when convert_to_surface=True

        Args:
            mesh: PyVista or VTK mesh object

        Returns:
            bool: True if mesh is PolyData or can be converted to surface
        """
        if isinstance(mesh, (pv.PolyData, vtk.vtkPolyData)):
            return True
        if (
            isinstance(mesh, (pv.UnstructuredGrid, vtk.vtkUnstructuredGrid))
            and self.convert_to_surface
        ):
            return True
        return False

    def _process_mesh_data(
        self, mesh: pv.DataSet | vtk.vtkDataSet
    ) -> dict[str, MeshLabelData]:
        """
        Process mesh and extract geometry data.

        Args:
            mesh: PyVista PolyData or UnstructuredGrid

        Returns:
            dict: Processed mesh data organized by labels or 'default'
        """
        if isinstance(mesh, (pv.UnstructuredGrid, vtk.vtkUnstructuredGrid)):
            if self.convert_to_surface:
                # Convert UnstructuredGrid to surface PolyData first
                surface_mesh = self._convert_ugrid_to_surface(mesh)
                return self._process_polydata(surface_mesh)
            raise TypeError(
                "UnstructuredGrid not supported by PolyMesh converter. "
                "Use convert_to_surface=True or TetMesh converter."
            )
        if isinstance(mesh, (pv.PolyData, vtk.vtkPolyData)):
            return self._process_polydata(mesh)
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    def _create_usd_mesh(
        self,
        transform_path: str,
        label: str,
        mesh_time_data: MeshTimeData,
        label_colors: dict[str, RgbColor],
        has_topology_change: bool,
    ) -> None:
        """
        Create USD mesh prim(s) for this label.

        Routes to appropriate method based on topology:
        - Constant topology: Single UsdGeomMesh with time-sampled points
        - Varying topology: Multiple UsdGeomMesh prims with visibility control

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
            has_topology_change: Whether topology varies over time
        """
        if has_topology_change:
            self.log_info(
                "Creating time-varying UsdGeomMesh for label: %s (topology changes detected)",
                label,
            )
            self._create_usd_polymesh_varying(
                transform_path, label, mesh_time_data, label_colors
            )
        else:
            self.log_info("Creating UsdGeomMesh for label: %s", label)
            self._create_usd_polymesh(
                transform_path, label, mesh_time_data, label_colors
            )

    def _convert_ugrid_to_surface(
        self, ugrid: pv.UnstructuredGrid | vtk.vtkUnstructuredGrid
    ) -> pv.PolyData:
        """
        Extract surface from UnstructuredGrid and convert to PolyData.

        Args:
            ugrid: PyVista UnstructuredGrid or VTK vtkUnstructuredGrid

        Returns:
            pv.PolyData: Surface mesh extracted from the UnstructuredGrid
        """
        # Convert VTK to PyVista if needed
        if isinstance(ugrid, vtk.vtkUnstructuredGrid):
            ugrid = pv.wrap(ugrid)

        # Extract surface using PyVista's built-in method
        surface = cast(pv.PolyData, ugrid.extract_surface())

        # Preserve point and cell data arrays
        # Point data is automatically preserved by extract_surface

        return surface

    def _process_polydata(
        self, polydata: pv.PolyData | vtk.vtkPolyData
    ) -> dict[str, MeshLabelData]:
        """
        Process PolyData and extract geometry, labels, and attributes.

        Args:
            polydata: VTK or PyVista PolyData mesh

        Returns:
            dict: Processed mesh data with structure:
                  {label: {'mesh_type', 'points', 'face_vertex_counts',
                           'face_vertex_indices', 'deformation_magnitude',
                           'color_array', 'point_mapping'}}
        """
        # Get points
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()

        # Get boundary labels if they exist
        boundary_labels = None
        if self.mask_ids is not None and polydata.GetCellData().HasArray(
            "boundary_labels"
        ):
            label_array = polydata.GetCellData().GetArray("boundary_labels")
            # Get all unique labels from both components
            tuple_values = [
                label_array.GetTuple(i) for i in range(label_array.GetNumberOfTuples())
            ]
            tuple_values_flattened = list(itertools.chain.from_iterable(tuple_values))
            boundary_labels = set(tuple_values_flattened)
            boundary_labels.discard(0)

        # Get deformation magnitude if it exists
        def_mag = None
        if polydata.GetPointData().HasArray("DeformationMagnitude"):
            intensity_array = polydata.GetPointData().GetArray("DeformationMagnitude")
            def_mag = [float(intensity_array.GetValue(i)) for i in range(num_points)]

        # Get color array if specified
        color_array = self._extract_color_array(polydata)

        # Get faces
        faces = polydata.GetPolys()
        connectivity = faces.GetConnectivityArray()
        offsets = faces.GetOffsetsArray()

        # Process face data
        num_faces = offsets.GetNumberOfValues() - 1
        start_idx = [offsets.GetValue(i) for i in range(num_faces)]
        end_idx = [offsets.GetValue(i + 1) for i in range(num_faces)]
        face_vertex_counts = [end_idx[i] - start_idx[i] for i in range(num_faces)]
        face_vertex_indices = []
        for i in range(num_faces):
            face_vertex_indices.extend(
                [connectivity.GetValue(j) for j in range(start_idx[i], end_idx[i])]
            )

        # for i in range(offsets.GetNumberOfValues() - 1):
        # start_idx = offsets.GetValue(i)
        # end_idx = offsets.GetValue(i + 1)
        # num_vertices = end_idx - start_idx
        # face_vertex_counts.append(num_vertices)
        # for j in range(start_idx, end_idx):
        # face_vertex_indices.append(connectivity.GetValue(j))

        # Create objects for each cell based on its labels
        if boundary_labels:
            assert self.mask_ids is not None
            # Create a dictionary to store objects for each label
            label_objects: dict[str, MeshLabelData] = {}

            # Initialize objects for each unique label
            for label_id in boundary_labels:
                if int(label_id) != 0:
                    label = self.mask_ids[int(label_id)]
                    label_objects[label] = {
                        "mesh_type": "polymesh",
                        "points": [],
                        "face_vertex_counts": [],
                        "face_vertex_indices": [],
                        "deformation_magnitude": [] if def_mag else None,
                        "color_array": [] if color_array is not None else None,
                        "point_mapping": {},
                    }

            label_array = None
            if polydata.GetCellData().HasArray("boundary_labels"):
                label_array = polydata.GetCellData().GetArray("boundary_labels")

            # Process each cell
            start_time = time.time()
            for cell_id in range(polydata.GetNumberOfCells()):
                if cell_id % 1000000 == 0 or cell_id == polydata.GetNumberOfCells() - 1:
                    self.log_progress(
                        cell_id + 1,
                        polydata.GetNumberOfCells(),
                        prefix="Processing cells",
                    )

                cell = polydata.GetCell(cell_id)
                cell_labels = set()

                # Get all labels for this cell
                if label_array:
                    tuple_values = label_array.GetTuple(cell_id)
                    labels = [
                        self.mask_ids[int(label_id)]
                        for label_id in tuple_values
                        if int(label_id) != 0
                    ]
                    cell_labels.update(labels)

                    # Get the points of this cell
                    n_points = cell.GetNumberOfPoints()
                    point_ids = [int(cell.GetPointId(i)) for i in range(n_points)]
                    usd_points = [
                        self._ras_to_usd(points.GetPoint(pnt_id))
                        for pnt_id in point_ids
                    ]

                    # For each label of this cell, create a copy of the cell
                    for label_str in cell_labels:
                        obj = label_objects[label_str]
                        point_mapping = cast(dict[int, int], obj["point_mapping"])
                        obj_points = cast(list[Gf.Vec3f], obj["points"])
                        obj_face_counts = cast(list[int], obj["face_vertex_counts"])
                        obj_face_indices = cast(list[int], obj["face_vertex_indices"])
                        cell_point_indices = []
                        for pnt_num, pnt_id in enumerate(point_ids):
                            indx = point_mapping.get(pnt_id, None)
                            if indx is None:
                                indx = len(obj_points)
                                obj_points.append(usd_points[pnt_num])
                                point_mapping[pnt_id] = indx
                                if def_mag is not None:
                                    obj_def_mag = cast(
                                        list[float], obj["deformation_magnitude"]
                                    )
                                    obj_def_mag.append(def_mag[pnt_id])
                                if color_array is not None:
                                    obj_color_array = cast(
                                        list[float], obj["color_array"]
                                    )
                                    obj_color_array.append(float(color_array[pnt_id]))
                            cell_point_indices.append(indx)

                        obj_face_counts.append(len(cell_point_indices))
                        obj_face_indices.extend(cell_point_indices)

            end_time = time.time()
            self.log_info(
                "Time taken to process cells %d: %s seconds",
                polydata.GetNumberOfCells(),
                end_time - start_time,
            )

            # Convert color_array lists to numpy arrays
            # for label, obj in label_objects.items():
            # if obj['color_array'] is not None:
            # obj['color_array'] = np.array(obj['color_array'])

            return label_objects
        # If no boundary labels, return single group with all points and faces
        points_data = [self._ras_to_usd(points.GetPoint(i)) for i in range(num_points)]
        return {
            "default": {
                "mesh_type": "polymesh",
                "points": points_data,
                "face_vertex_counts": face_vertex_counts,
                "face_vertex_indices": face_vertex_indices,
                "deformation_magnitude": def_mag,
                "color_array": (
                    color_array.tolist() if color_array is not None else None
                ),
            }
        }

    def _create_usd_polymesh(
        self,
        transform_path: str,
        label: str,
        mesh_time_data: MeshTimeData,
        label_colors: dict[str, RgbColor],
    ) -> None:
        """
        Create UsdGeomMesh for polygon surface data with constant topology.

        Uses time-sampled attributes for points, normals, and colors.

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
        """
        assert self.stage is not None
        data = mesh_time_data[0][label]

        # Create mesh prim under the transform
        mesh_path = f"{transform_path}/{label}"
        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

        # Set topology (assuming consistent topology across timesteps)
        mesh.CreateFaceVertexCountsAttr(data["face_vertex_counts"])
        mesh.CreatePointsAttr()
        mesh.CreateFaceVertexIndicesAttr(data["face_vertex_indices"])

        # Set mesh attributes for Index renderer compatibility
        mesh.CreateSubdivisionSchemeAttr("none")  # Prevent unwanted subdivision
        mesh.CreateDoubleSidedAttr(True)  # Ensure visibility from both sides

        # Create normals attribute
        # Normals will be computed per timestep since mesh deforms
        if self.compute_normals:
            normals_attr = mesh.CreateNormalsAttr()
            normals_attr.SetMetadata("interpolation", UsdGeom.Tokens.vertex)

        # Set display color - either per-vertex from color array or fixed label color
        use_color_array = self.color_by_array is not None and any(
            mesh_time_data[t][label].get("color_array") is not None
            for t in range(len(self.times))
        )

        if not use_color_array:
            # Use fixed label color with proper primvar
            display_color = label_colors[label]
            display_color_primvar = mesh.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.constant
            )
            display_color_primvar.Set([display_color])

        # Create points attribute with time samples
        points_attr = mesh.CreatePointsAttr()
        extent_attr = mesh.CreateExtentAttr()
        time_samples = {}

        # Create display color primvar if using color array
        scalar_primvar = None
        display_color_primvar = None
        display_opacity_primvar = None
        if use_color_array:
            display_color_primvar = mesh.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.vertex
            )
            display_opacity_primvar = mesh.CreateDisplayOpacityPrimvar(
                UsdGeom.Tokens.vertex
            )

            # Create custom primvar for raw scalar values (for colormap control)
            scalar_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                self.color_by_array,
                Sdf.ValueTypeNames.FloatArray,
                UsdGeom.Tokens.vertex,
            )

        # **USE NEW COLORMAP SYSTEM: Compute intensity range once**
        if use_color_array:
            vmin, vmax = self._compute_intensity_range(mesh_time_data, label)
            global_vmin = vmin
            global_vmax = vmax
        else:
            global_vmin = float("inf")
            global_vmax = float("-inf")

        num_times = len(self.times)
        for time_idx, time_code in enumerate(self.times):
            if time_idx % 10 == 0 or time_idx == num_times - 1:
                self.log_progress(
                    time_idx + 1,
                    num_times,
                    prefix=f"Processing time samples for {label}",
                )
            time_data = mesh_time_data[time_idx][label]

            if self.compute_normals:
                vertex_normals = self._compute_facevarying_normals_tri(
                    time_data["points"],
                    time_data["face_vertex_counts"],
                    time_data["face_vertex_indices"],
                )

            # Set points first
            time_samples[time_code] = {
                "points": time_data["points"],
                "extent": UsdGeom.Mesh.ComputeExtent(time_data["points"]),
            }
            if self.compute_normals:
                time_samples[time_code]["normals"] = vertex_normals

            # Compute per-vertex colors if using color array
            if use_color_array and time_data.get("color_array") is not None:
                color_values = cast(list[float], time_data["color_array"])

                # **USE CONFIGURED COLORMAP with consistent intensity range**
                vertex_colors = [
                    self._map_scalar_to_color(float(v), vmin, vmax, self.colormap)
                    for v in color_values
                ]
                time_samples[time_code]["vertex_colors"] = vertex_colors
                time_samples[time_code]["scalar_values"] = color_values
                time_samples[time_code]["vmin"] = vmin
                time_samples[time_code]["vmax"] = vmax

        # Set points, extents, and normals with explicit time codes
        for t_code, time_data_dict in time_samples.items():
            points_attr.Set(time_data_dict["points"], t_code)
            extent_attr.Set(time_data_dict["extent"], t_code)
            if self.compute_normals:
                normals_attr.Set(time_data_dict["normals"], t_code)
            if use_color_array and "vertex_colors" in time_data_dict:
                assert display_color_primvar is not None
                assert scalar_primvar is not None
                assert display_opacity_primvar is not None
                display_color_primvar.Set(time_data_dict["vertex_colors"], t_code)
                # Set raw scalar values for colormap control
                scalar_values = cast(list[float], time_data_dict["scalar_values"])
                scalar_primvar.Set(scalar_values, t_code)
                # Set opacity (full opacity by default)
                num_vertices = len(scalar_values)
                opacity_values = [1.0] * num_vertices
                display_opacity_primvar.Set(opacity_values, t_code)

        # Set initial values (non-timewarped)
        points_attr.Set(time_samples[self.times[0]]["points"])
        extent_attr.Set(time_samples[self.times[0]]["extent"])
        if self.compute_normals:
            normals_attr.Set(time_samples[self.times[0]]["normals"])
        if use_color_array and "vertex_colors" in time_samples[self.times[0]]:
            assert display_color_primvar is not None
            assert scalar_primvar is not None
            assert display_opacity_primvar is not None
            display_color_primvar.Set(time_samples[self.times[0]]["vertex_colors"])
            scalar_values_0 = cast(
                list[float], time_samples[self.times[0]]["scalar_values"]
            )
            scalar_primvar.Set(scalar_values_0)
            num_vertices = len(scalar_values_0)
            display_opacity_primvar.Set([1.0] * num_vertices)

        # Add metadata for colormap range and visualization controls (for colormap meshes only)
        if use_color_array:
            prim = mesh.GetPrim()
            prim.SetCustomDataByKey(f"{self.color_by_array}_min", global_vmin)
            prim.SetCustomDataByKey(f"{self.color_by_array}_max", global_vmax)
            prim.SetCustomDataByKey(f"{self.color_by_array}_colormap", self.colormap)
            prim.SetCustomDataByKey("visualizationDataArray", self.color_by_array)

        # Set deformation magnitude if it exists
        if any(
            mesh_time_data[time_idx][label]["deformation_magnitude"] is not None
            for time_idx in range(len(self.times))
        ):
            def_mag_attr = mesh.GetPrim().CreateAttribute(
                "deformationMagnitude", Sdf.ValueTypeNames.FloatArray
            )

            for time_idx, t_code in enumerate(self.times):
                def_mag = mesh_time_data[time_idx][label]["deformation_magnitude"]
                if def_mag is not None:
                    def_mag_attr.Set(def_mag, t_code)

    def _create_usd_polymesh_varying(
        self,
        transform_path: str,
        label: str,
        mesh_time_data: MeshTimeData,
        label_colors: dict[str, RgbColor],
    ) -> None:
        """
        Create separate UsdGeomMesh prims for each timestep with visibility control.

        Used when topology changes over time (varying number of points/faces).

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
        """
        assert self.stage is not None
        # Determine if using color array
        use_color_array = self.color_by_array is not None and any(
            mesh_time_data[t][label].get("color_array") is not None
            for t in range(len(self.times))
        )

        # Compute intensity range if using color array
        if use_color_array:
            vmin, vmax = self._compute_intensity_range(mesh_time_data, label)

        # Create a parent Xform for all time-varying meshes
        parent_path = f"{transform_path}/{label}"
        UsdGeom.Xform.Define(self.stage, parent_path)

        # Create separate mesh for each time step
        num_times = len(self.times)
        for time_idx, time_code in enumerate(self.times):
            if time_idx % 10 == 0 or time_idx == num_times - 1:
                self.log_progress(
                    time_idx + 1, num_times, prefix=f"Creating meshes for {label}"
                )
            # Skip if label doesn't exist at this timestep
            if label not in mesh_time_data[time_idx]:
                continue

            time_data = mesh_time_data[time_idx][label]

            # Create mesh prim for this time step
            mesh_path = f"{parent_path}/mesh_t{time_code}"
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

            # Set topology (unique for this timestep)
            mesh.CreateFaceVertexCountsAttr(time_data["face_vertex_counts"])
            mesh.CreateFaceVertexIndicesAttr(time_data["face_vertex_indices"])
            mesh.CreatePointsAttr(time_data["points"])

            # Set mesh attributes for Index renderer compatibility
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)

            # Compute and set normals
            if self.compute_normals:
                vertex_normals = self._compute_facevarying_normals_tri(
                    time_data["points"],
                    time_data["face_vertex_counts"],
                    time_data["face_vertex_indices"],
                )
                normals_attr = mesh.CreateNormalsAttr()
                normals_attr.SetMetadata("interpolation", UsdGeom.Tokens.vertex)
                normals_attr.Set(vertex_normals)

            # Set extent
            extent_attr = mesh.CreateExtentAttr()
            extent_attr.Set(UsdGeom.Mesh.ComputeExtent(time_data["points"]))

            # Set display color
            if use_color_array and time_data.get("color_array") is not None:
                color_values = cast(list[float], time_data["color_array"])

                # Map scalars to colors
                vertex_colors = [
                    self._map_scalar_to_color(float(v), vmin, vmax, self.colormap)
                    for v in color_values
                ]
                display_color_primvar = mesh.CreateDisplayColorPrimvar(
                    UsdGeom.Tokens.vertex
                )
                display_color_primvar.Set(vertex_colors)

                # Set opacity
                display_opacity_primvar = mesh.CreateDisplayOpacityPrimvar(
                    UsdGeom.Tokens.vertex
                )
                display_opacity_primvar.Set([1.0] * len(vertex_colors))

                # Store scalar values as primvar
                scalar_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    self.color_by_array,
                    Sdf.ValueTypeNames.FloatArray,
                    UsdGeom.Tokens.vertex,
                )
                scalar_list = (
                    color_values.tolist()
                    if hasattr(color_values, "tolist")
                    else list(color_values)
                )
                scalar_primvar.Set(scalar_list)

                # Add colormap metadata
                prim = mesh.GetPrim()
                prim.SetCustomDataByKey(f"{self.color_by_array}_min", vmin)
                prim.SetCustomDataByKey(f"{self.color_by_array}_max", vmax)
                prim.SetCustomDataByKey(
                    f"{self.color_by_array}_colormap", self.colormap
                )
                prim.SetCustomDataByKey("visualizationDataArray", self.color_by_array)
            else:
                # Use fixed label color
                display_color = label_colors[label]
                display_color_primvar = mesh.CreateDisplayColorPrimvar(
                    UsdGeom.Tokens.constant
                )
                display_color_primvar.Set([display_color])

            # Set deformation magnitude if exists
            if time_data.get("deformation_magnitude") is not None:
                def_mag_attr = mesh.GetPrim().CreateAttribute(
                    "deformationMagnitude", Sdf.ValueTypeNames.FloatArray
                )
                def_mag_attr.Set(time_data["deformation_magnitude"])

            # Set visibility based on time code
            # Mesh is visible only at its specific time code
            visibility_attr = mesh.CreateVisibilityAttr()
            for t_code in self.times:
                if t_code == time_code:
                    visibility_attr.Set(UsdGeom.Tokens.inherited, t_code)
                else:
                    visibility_attr.Set(UsdGeom.Tokens.invisible, t_code)

            # Set default visibility
            visibility_attr.Set(
                UsdGeom.Tokens.inherited
                if time_code == self.times[0]
                else UsdGeom.Tokens.invisible
            )
