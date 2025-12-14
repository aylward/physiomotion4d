"""
Tools for creating and manipulating contours.
"""

import itk
import numpy as np
import pyvista as pv
import trimesh

from physiomotion4d.transform_tools import TransformTools


class ContourTools:
    """
    Tools for creating and manipulating contours.
    """

    def __init__(self):
        pass

    def extract_contours(
        self,
        mask_image: itk.image,
    ) -> pv.PolyData:
        """
        Make contours from a mask image.

        Args:
            mask_image (itk.image): The mask image to create contours from
            output_file (str, optional): If provided, save the contours to this VTP
                file

        Returns:
            pv.PolyData: The contours as a PyVista PolyData object
        """
        labels = pv.wrap(itk.vtk_image_from_image(mask_image))
        contours = labels.contour_labels(
            boundary_style="all",
            pad_background=False,
            smoothing=True,
            smoothing_iterations=10,
            output_mesh_type="triangles",
        )

        contours.smooth_taubin(
            inplace=True,
            n_iter=50,
            pass_band=0.05,
        )

        # self.contours.decimate_pro(
        # inplace=True,
        # reduction=0.7,
        # feature_angle=45,
        # preserve_topology=True,
        # )

        return contours

    def transform_contours(
        self,
        contours: pv.PolyData,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.PolyData:
        """
        Transform contours using a given transform.

        Args:
            tfm (itk.Transform): The transform to use

        Returns:
            pv.PolyData: The transformed contours with deformation magnitude
        """
        new_contours = TransformTools().transform_pvcontour(
            contours, tfm, with_deformation_magnitude=with_deformation_magnitude
        )

        return new_contours

    def merge_meshes(self, meshes):
        """
        Merge multiple fixed meshes into a single mesh.

        Returns
        -------
        pv.PolyData
            Merged mesh
        """
        print("Merging meshes...")
        if hasattr(meshes[0], 'n_faces_strict'):
            meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points,
                    faces=mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:],
                )
                for mesh in meshes
            ]
        else:
            meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4]
                )
                for mesh in meshes
            ]

        # Merge meshes
        merged_mesh = trimesh.util.concatenate(meshes)
        flip_matrix = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        merged_mesh.apply_transform(flip_matrix)  # Apply flip transformation
        for mesh in meshes:
            mesh.apply_transform(flip_matrix)

        merged_mesh = pv.wrap(merged_mesh)
        pv_meshes = [pv.wrap(mesh) for mesh in meshes]

        return merged_mesh, pv_meshes

    def create_mask_from_mesh(
        self,
        mesh,
        reference_image,
        resample_to_reference=True,
    ):
        ref_spacing = np.array(reference_image.GetSpacing())

        # Create trimesh object with LPS coordinates
        if hasattr(mesh, 'n_faces_strict'):
            # PyVista PolyData
            faces = mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:]
        else:
            # Handle other mesh types
            faces = mesh.faces.reshape((-1, 4))[:, 1:]

        trimesh_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)

        # Determine voxel spacing (use minimum spacing from reference)
        voxel_pitch = float(np.min(ref_spacing))

        # Voxelize the mesh
        # trimesh.voxelized() creates a grid aligned with the mesh's bounding box
        # The voxel grid origin is at the minimum corner of the bounding box
        vox = trimesh_mesh.voxelized(pitch=voxel_pitch)
        binary_array = vox.matrix.astype(np.uint8)

        # Get the physical origin of the voxel grid in LPS space
        # trimesh voxel grids use a transformation matrix, and the voxel grid starts
        # at the mesh's minimum bounds. The physical origin is where voxel [0,0,0]
        # center is located.
        # Get mesh bounds in LPS coordinates
        mesh_bounds_lps = (
            trimesh_mesh.bounds
        )  # shape (2, 3): [[x_min, y_min, z_min], [x_max, y_max, z_max]]

        # The voxel grid origin is at the minimum corner, but ITK origin is the CENTER
        # of voxel (0,0,0)
        # So we need to add half a voxel pitch to each dimension
        voxel_grid_origin_lps = mesh_bounds_lps[0] + voxel_pitch / 2.0
        voxel_grid_origin_lps[2] = (
            voxel_grid_origin_lps[2] + voxel_pitch * binary_array.shape[2]
        )

        # Create ITK image from the voxel array
        # ITK uses ZYX ordering (numpy array convention), trimesh uses XYZ
        # Need to transpose: (X, Y, Z) -> (Z, Y, X)
        binary_array_zyx = np.transpose(binary_array, (2, 1, 0))
        binary_array_flip = np.flip(binary_array_zyx, axis=0)
        binary_image = itk.GetImageFromArray(binary_array_flip)

        # Set ITK image metadata in LPS coordinates
        # Origin: where the center of voxel (0,0,0) is located in physical space
        binary_image.SetOrigin(voxel_grid_origin_lps)

        # Spacing: uniform voxel pitch in all directions
        binary_image.SetSpacing([voxel_pitch] * 3)

        # Direction: use identity for now (axis-aligned), will be handled by resampling
        ref_dir = np.array(reference_image.GetDirection())
        ref_dir[2, 2] = -ref_dir[2, 2]
        binary_image.SetDirection(ref_dir)

        # Fill holes to create solid mask
        ImageType = type(binary_image)
        fill_filter = itk.BinaryFillholeImageFilter[ImageType].New()
        fill_filter.SetInput(binary_image)
        fill_filter.SetForegroundValue(1)
        fill_filter.Update()
        mask_image = fill_filter.GetOutput()

        if resample_to_reference:
            resampler = itk.ResampleImageFilter.New(Input=mask_image)
            resampler.SetReferenceImage(reference_image)
            resampler.SetUseReferenceImage(True)
            resampler.SetInterpolator(
                itk.NearestNeighborInterpolateImageFunction.New(mask_image)
            )
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            mask_image = resampler.GetOutput()

        return mask_image

    def create_contour_distance_map_from_mesh(
        self,
        mesh,
        reference_image,
        max_distance: float = 100.0,
        invert_distance_map: bool = False,
    ):
        # Convert mask to binary
        mesh_mask = self.create_mask_from_mesh(mesh, reference_image)
        mask_arr = itk.GetArrayFromImage(mesh_mask)
        binary_mask_arr = (mask_arr > 0).astype(np.uint8)
        binary_mask_image = itk.GetImageFromArray(binary_mask_arr)
        binary_mask_image.CopyInformation(reference_image)

        edge_filter = itk.BinaryContourImageFilter.New(Input=binary_mask_image)
        edge_filter.SetForegroundValue(1)
        edge_filter.SetBackgroundValue(0)
        edge_filter.SetFullyConnected(False)
        edge_filter.Update()
        edge_mask_image = edge_filter.GetOutput()

        # Compute signed distance map (positive inside, negative outside)
        print("  Computing signed distance map...")
        distance_filter = itk.SignedMaurerDistanceMapImageFilter.New(
            Input=edge_mask_image
        )
        distance_filter.SetSquaredDistance(False)
        distance_filter.SetUseImageSpacing(True)
        distance_filter.SetInsideIsPositive(False)
        distance_filter.Update()
        distance_image = distance_filter.GetOutput()

        distance_arr = itk.GetArrayFromImage(distance_image)
        min_val = distance_arr.min()
        if max_distance is None:
            max_val = distance_arr.max()
        else:
            max_val = max_distance
            distance_arr = np.clip(distance_arr, min_val, max_val)
        if invert_distance_map:
            distance_arr = (
                (1.0 - (distance_arr - min_val) / (max_val - min_val)) * max_distance
            ).astype(np.float32)
        else:
            distance_arr = (
                ((distance_arr - min_val) / (max_val - min_val)) * max_distance
            ).astype(np.float32)
        distance_image = itk.GetImageFromArray(distance_arr)
        distance_image.CopyInformation(reference_image)

        return distance_image
