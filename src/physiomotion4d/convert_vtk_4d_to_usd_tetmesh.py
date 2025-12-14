"""Converter for VTK UnstructuredGrid to USD TetMesh with volumetric meshes."""

import numpy as np
import pyvista as pv
import vtk
from pxr import Gf, Sdf, UsdGeom

from .convert_vtk_4d_to_usd_base import (
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    ConvertVTK4DToUSDBase,
)


class ConvertVTK4DToUSDTetMesh(ConvertVTK4DToUSDBase):
    """
    Converter for VTK UnstructuredGrid to USD TetMesh.

    Handles:
    - Volumetric tetrahedral meshes
    - Surface cells (triangles/quads) from UnstructuredGrid
    - Time-varying topology via visibility control

    Requires OpenUSD v24.03+ for UsdGeomTetMesh support.

    Example Usage:
        >>> converter = ConvertVTK4DToUSDTetMesh(
        ...     data_basename="VolumetricModel",
        ...     input_polydata=ugrid_meshes,
        ...     mask_ids=None
        ... )
        >>> stage = converter.convert("output.usd")
    """

    def supports_mesh_type(self, mesh) -> bool:
        """
        Check if this converter supports the given mesh type.

        Supports UnstructuredGrid when NOT converting to surface.

        Args:
            mesh: PyVista or VTK mesh object

        Returns:
            bool: True if mesh is UnstructuredGrid and not surface mode
        """
        return self._is_unstructured_grid(mesh) and not self.convert_to_surface

    def _process_mesh_data(self, mesh) -> dict:
        """
        Process mesh and extract geometry data.

        Args:
            mesh: PyVista UnstructuredGrid

        Returns:
            dict: Processed mesh data with tetrahedral or surface cell information
        """
        if not self._is_unstructured_grid(mesh):
            raise TypeError(
                f"TetMesh converter only supports UnstructuredGrid. "
                f"Got: {type(mesh)}"
            )
        return self._process_unstructured_grid(mesh)

    def _create_usd_mesh(
        self, transform_path, label, mesh_time_data, label_colors, has_topology_change
    ):
        """
        Create USD mesh prim(s) for this label.

        Routes to appropriate method based on topology and mesh type:
        - TetMesh with constant topology: Single UsdGeomTetMesh
        - TetMesh with varying topology: Multiple UsdGeomTetMesh prims with visibility
        - PolyMesh (surface cells): Delegates to PolyMesh creation

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
            has_topology_change: Whether topology varies over time
        """
        # Check mesh type from first timestep data
        mesh_type = mesh_time_data[0][label].get('mesh_type', 'tetmesh')

        if mesh_type == 'tetmesh':
            if has_topology_change:
                print(
                    f"Creating time-varying UsdGeomTetMesh for label: {label} "
                    f"(topology changes detected)"
                )
                self._create_usd_tetmesh_varying(
                    transform_path, label, mesh_time_data, label_colors
                )
            else:
                print(f"Creating UsdGeomTetMesh for label: {label}")
                self._create_usd_tetmesh(
                    transform_path, label, mesh_time_data, label_colors
                )
        else:
            # Surface cells from UnstructuredGrid - treat as polymesh
            # Note: This is a fallback for UnstructuredGrid with only surface cells
            raise ValueError(
                f"UnstructuredGrid contains surface cells, not tetrahedra. "
                f"Use convert_to_surface=True with PolyMesh converter."
            )

    def _process_unstructured_grid(self, ugrid) -> dict:
        """
        Process UnstructuredGrid and extract tetrahedral and surface geometry.

        Args:
            ugrid: PyVista UnstructuredGrid or VTK vtkUnstructuredGrid

        Returns:
            dict: Processed mesh data with 'mesh_type' key indicating 'tetmesh' or
                  'polymesh'
        """
        # Convert VTK to PyVista if needed
        if isinstance(ugrid, vtk.vtkUnstructuredGrid):
            ugrid = pv.wrap(ugrid)

        # Get points
        points = ugrid.points

        # Get deformation magnitude if it exists
        def_mag = None
        if 'DeformationMagnitude' in ugrid.point_data:
            def_mag = ugrid.point_data['DeformationMagnitude']

        # Get boundary labels if they exist
        boundary_labels = None
        if self.mask_ids is not None and 'boundary_labels' in ugrid.cell_data:
            label_array = ugrid.cell_data['boundary_labels']
            boundary_labels = set()
            if label_array.ndim > 1:
                # Multi-component array
                for row in label_array:
                    for value in row:
                        if int(value) != 0:
                            boundary_labels.add(value)
            else:
                # Single component
                for value in label_array:
                    if int(value) != 0:
                        boundary_labels.add(value)

        # Parse cells and cell types
        cells = ugrid.cells
        celltypes = ugrid.celltypes

        # Separate tetrahedral cells from surface cells
        tet_cells = []
        surface_cells = []
        cell_labels = []

        idx = 0
        cell_id = 0
        while idx < len(cells):
            n_points = cells[idx]
            cell_type = celltypes[cell_id]
            cell_connectivity = cells[idx + 1 : idx + 1 + n_points]

            # Get cell label if available
            cell_label = None
            if boundary_labels is not None and 'boundary_labels' in ugrid.cell_data:
                label_val = ugrid.cell_data['boundary_labels'][cell_id]
                if isinstance(label_val, (list, np.ndarray)):
                    # Multi-component, take first non-zero
                    for v in label_val:
                        if int(v) != 0:
                            cell_label = int(v)
                            break
                else:
                    if int(label_val) != 0:
                        cell_label = int(label_val)

            if cell_type == VTK_TETRA:
                tet_cells.append(cell_connectivity)
                cell_labels.append(cell_label)
            elif cell_type in [VTK_TRIANGLE, VTK_QUAD]:
                surface_cells.append((cell_connectivity, n_points))

            idx += n_points + 1
            cell_id += 1

        # Determine mesh type and process accordingly
        if len(tet_cells) > 0:
            # Process as tetrahedral mesh
            return self._process_tetrahedral_mesh(
                points, tet_cells, cell_labels, def_mag, boundary_labels
            )
        elif len(surface_cells) > 0:
            # Process as surface mesh
            return self._process_surface_cells(
                points, surface_cells, def_mag, boundary_labels
            )
        else:
            raise ValueError(
                "UnstructuredGrid contains no supported cell types (tetrahedra or surface cells)"
            )

    def _process_tetrahedral_mesh(
        self, points, tet_cells, cell_labels, def_mag, boundary_labels
    ):
        """
        Process tetrahedral cells for UsdGeomTetMesh export.

        Args:
            points: Array of point coordinates
            tet_cells: List of tetrahedral connectivity arrays
            cell_labels: List of cell labels (or None)
            def_mag: Deformation magnitude array (or None)
            boundary_labels: Set of boundary label IDs (or None)

        Returns:
            dict: Processed tetmesh data with structure:
                  {'default': {'mesh_type', 'points', 'tet_indices',
                               'surface_face_indices', 'deformation_magnitude'}}
        """
        # Convert points to USD coordinates
        points_usd = [self._ras_to_usd(p) for p in points]

        # Convert tetrahedral indices to Vec4i format (required by UsdGeomTetMesh)
        tet_indices_vec4 = [
            Gf.Vec4i(int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3]))
            for tet in tet_cells
        ]

        # Compute surface faces from tetrahedra for rendering
        # Each tetrahedron has 4 triangular faces
        surface_faces = []
        for tet in tet_cells:
            # The 4 faces of a tetrahedron with vertices [a, b, c, d]:
            # Face 0: [b, c, d], Face 1: [a, c, d], Face 2: [a, b, d], Face 3: [a, b, c]
            a, b, c, d = tet
            surface_faces.extend(
                [
                    [int(b), int(c), int(d)],
                    [int(a), int(c), int(d)],
                    [int(a), int(b), int(d)],
                    [int(a), int(b), int(c)],
                ]
            )

        # Flatten surface face indices to Vec3i format
        surface_face_indices = [Gf.Vec3i(f[0], f[1], f[2]) for f in surface_faces]

        result = {
            'default': {
                'mesh_type': 'tetmesh',
                'points': points_usd,
                'tet_indices': tet_indices_vec4,
                'surface_face_indices': surface_face_indices,
                'deformation_magnitude': (
                    def_mag.tolist() if def_mag is not None else None
                ),
            }
        }

        return result

    def _process_surface_cells(self, points, surface_cells, def_mag, boundary_labels):
        """
        Process surface cells (triangles/quads) for UsdGeomMesh export.

        Args:
            points: Array of point coordinates
            surface_cells: List of (connectivity, n_points) tuples
            def_mag: Deformation magnitude array (or None)
            boundary_labels: Set of boundary label IDs (or None)

        Returns:
            dict: Processed polymesh data with structure:
                  {'default': {'mesh_type', 'points', 'face_vertex_counts',
                               'face_vertex_indices', 'deformation_magnitude'}}
        """
        points_usd = [self._ras_to_usd(p) for p in points]

        face_vertex_counts = []
        face_vertex_indices = []

        for cell_connectivity, n_points in surface_cells:
            face_vertex_counts.append(n_points)
            face_vertex_indices.extend(cell_connectivity.tolist())

        return {
            'default': {
                'mesh_type': 'polymesh',
                'points': points_usd,
                'face_vertex_counts': face_vertex_counts,
                'face_vertex_indices': face_vertex_indices,
                'deformation_magnitude': (
                    def_mag.tolist() if def_mag is not None else None
                ),
            }
        }

    def _create_usd_tetmesh(self, transform_path, label, mesh_time_data, label_colors):
        """
        Create UsdGeomTetMesh for tetrahedral volume data with constant topology.

        Uses time-sampled attributes for points and normals.

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
        """
        data = mesh_time_data[0][label]

        # Create tetrahedral mesh prim under the transform
        mesh_path = f"{transform_path}/{label}"
        tetmesh = UsdGeom.TetMesh.Define(self.stage, mesh_path)

        # Set tetrahedral topology (assuming consistent topology across timesteps)
        tetmesh.CreateTetVertexIndicesAttr(data['tet_indices'])
        tetmesh.CreateSurfaceFaceVertexIndicesAttr(data['surface_face_indices'])

        # Set mesh attributes for Index renderer compatibility
        tetmesh.CreateDoubleSidedAttr(True)  # Ensure visibility from both sides

        # Create normals attribute (REQUIRED for IndeX renderer)
        # For tetrahedral meshes, we need normals for the surface vertices
        normals_attr = tetmesh.CreateNormalsAttr()
        normals_attr.SetMetadata('interpolation', UsdGeom.Tokens.vertex)

        # Assign a unique color to the mesh with proper primvar
        display_color = label_colors[label]
        display_color_primvar = tetmesh.CreateDisplayColorPrimvar(
            UsdGeom.Tokens.constant
        )
        display_color_primvar.Set([display_color])

        # Create points attribute with time samples
        points_attr = tetmesh.CreatePointsAttr()
        extent_attr = tetmesh.CreateExtentAttr()
        time_samples = {}

        for time_idx, time_code in enumerate(self.times):
            print(f"Processing time step {time_code} for label: {label}")
            time_data = mesh_time_data[time_idx][label]

            # Compute per-vertex normals for surface faces (REQUIRED for IndeX renderer)
            # For tetrahedral meshes, compute normals based on surface triangulation
            # First, need to convert surface face indices to face vertex counts
            surface_indices = time_data['surface_face_indices']
            # Surface faces are triangular, so each face has 3 vertices
            face_vertex_counts = [3] * (len(surface_indices) // 3)
            vertex_normals = self._compute_vertex_normals(
                time_data['points'], face_vertex_counts, surface_indices
            )

            # Set points first
            time_samples[time_code] = {
                'points': time_data['points'],
                'extent': UsdGeom.TetMesh.ComputeExtent(time_data['points']),
                'normals': vertex_normals,
            }

        # Set points, extents, and normals with explicit time codes
        for t_code, time_data_dict in time_samples.items():
            points_attr.Set(time_data_dict['points'], t_code)
            extent_attr.Set(time_data_dict['extent'], t_code)
            normals_attr.Set(time_data_dict['normals'], t_code)

        # Set initial values (non-timewarped)
        points_attr.Set(time_samples[self.times[0]]['points'])
        extent_attr.Set(time_samples[self.times[0]]['extent'])
        normals_attr.Set(time_samples[self.times[0]]['normals'])

        # Set deformation magnitude if it exists
        if any(
            mesh_time_data[time_idx][label]['deformation_magnitude'] is not None
            for time_idx in range(len(self.times))
        ):

            def_mag_attr = tetmesh.GetPrim().CreateAttribute(
                "deformationMagnitude", Sdf.ValueTypeNames.FloatArray
            )

            for time_idx, t_code in enumerate(self.times):
                def_mag = mesh_time_data[time_idx][label]['deformation_magnitude']
                if def_mag is not None:
                    def_mag_attr.Set(def_mag, t_code)

    def _create_usd_tetmesh_varying(
        self, transform_path, label, mesh_time_data, label_colors
    ):
        """
        Create separate UsdGeomTetMesh prims for each timestep with visibility control.

        Used when topology changes over time (varying number of points/tetrahedra).

        Args:
            transform_path: USD path for the transform
            label: Label identifier
            mesh_time_data: Time-series mesh data
            label_colors: Color assignments for labels
        """
        # Create a parent Xform for all time-varying tetmeshes
        parent_path = f"{transform_path}/{label}"
        UsdGeom.Xform.Define(self.stage, parent_path)

        # Create separate tetmesh for each time step
        for time_idx, time_code in enumerate(self.times):
            print(f"Creating UsdGeomTetMesh for label: {label} at time: {time_code}")
            # Skip if label doesn't exist at this timestep
            if label not in mesh_time_data[time_idx]:
                continue

            time_data = mesh_time_data[time_idx][label]

            # Create tetmesh prim for this time step
            mesh_path = f"{parent_path}/tetmesh_t{time_code}"
            tetmesh = UsdGeom.TetMesh.Define(self.stage, mesh_path)

            # Set topology (unique for this timestep)
            tetmesh.CreateTetVertexIndicesAttr(time_data['tet_indices'])
            tetmesh.CreateSurfaceFaceVertexIndicesAttr(
                time_data['surface_face_indices']
            )
            tetmesh.CreatePointsAttr(time_data['points'])

            # Set mesh attributes for Index renderer compatibility
            tetmesh.CreateDoubleSidedAttr(True)

            # Compute and set normals
            surface_indices = time_data['surface_face_indices']
            face_vertex_counts = [3] * (len(surface_indices) // 3)
            vertex_normals = self._compute_vertex_normals(
                time_data['points'], face_vertex_counts, surface_indices
            )
            normals_attr = tetmesh.CreateNormalsAttr()
            normals_attr.SetMetadata('interpolation', UsdGeom.Tokens.vertex)
            normals_attr.Set(vertex_normals)

            # Set extent
            extent_attr = tetmesh.CreateExtentAttr()
            extent_attr.Set(UsdGeom.TetMesh.ComputeExtent(time_data['points']))

            # Set display color
            display_color = label_colors[label]
            display_color_primvar = tetmesh.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.constant
            )
            display_color_primvar.Set([display_color])

            # Set deformation magnitude if exists
            if time_data.get('deformation_magnitude') is not None:
                def_mag_attr = tetmesh.GetPrim().CreateAttribute(
                    "deformationMagnitude", Sdf.ValueTypeNames.FloatArray
                )
                def_mag_attr.Set(time_data['deformation_magnitude'])

            # Set visibility based on time code
            # Mesh is visible only at its specific time code
            visibility_attr = tetmesh.CreateVisibilityAttr()
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
