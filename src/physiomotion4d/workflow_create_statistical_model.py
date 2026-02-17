"""Create a PCA statistical shape model from a sample of meshes.

This module provides the WorkflowCreateStatisticalModel class that implements
the pipeline from the Heart-Create_Statistical_Model experiment notebooks:

1. Extract surfaces from sample and reference meshes
2. ICP alignment: align each sample surface to the reference (template) surface
3. Deformable registration: establish dense correspondence via mask-based SyN
4. Correspondence: warp reference surface by each transform to get aligned shapes
5. PCA: compute mean and modes from corresponded shapes

Returns a dictionary of surfaces, meshes, and PCA model structure (no file I/O).
"""

import logging
from typing import Any, Optional

import itk
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_models_distance_maps import RegisterModelsDistanceMaps
from physiomotion4d.register_models_icp import RegisterModelsICP
from physiomotion4d.transform_tools import TransformTools


def _extract_surface(mesh: pv.DataSet) -> pv.PolyData:
    """Extract surface from a mesh (PolyData or UnstructuredGrid)."""
    if isinstance(mesh, pv.UnstructuredGrid):
        return mesh.extract_surface()
    if isinstance(mesh, pv.PolyData):
        return mesh
    return mesh.extract_surface()


class WorkflowCreateStatisticalModel(PhysioMotion4DBase):
    """Create a PCA statistical shape model from a sample of meshes aligned to a reference.

    Pipeline (mirrors experiments/Heart-Create_Statistical_Model notebooks 1–5):
    1. Extract surfaces from sample meshes and reference mesh (reference surface = alignment target)
    2. ICP (affine) align each sample surface to the reference surface
    3. Deformable (ANTs SyN) registration of each aligned sample to reference
    4. Build corresponded shapes (reference topology) in reference space
    5. Compute PCA and return mean surface, reference mesh, and PCA model dict

    Attributes:
        sample_meshes (list): List of sample mesh DataSets (.vtk/.vtu/.vtp geometry)
        reference_mesh (pv.DataSet): Reference mesh; its surface is used for alignment
        pca_number_of_components (int): Number of PCA components to retain
        reference_spatial_resolution (float): Resolution for reference image from mesh
        reference_buffer_factor (float): Buffer around mesh for reference image
    """

    def __init__(
        self,
        sample_meshes: list[pv.DataSet],
        reference_mesh: pv.DataSet,
        pca_number_of_components: int = 15,
        reference_spatial_resolution: float = 1.0,
        reference_buffer_factor: float = 0.25,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the create-statistical-model workflow.

        Args:
            sample_meshes: List of sample mesh DataSets (PyVista PolyData or UnstructuredGrid).
            reference_mesh: Reference mesh; its surface is used to align all samples.
            pca_number_of_components: Number of PCA components. Default 15.
            reference_spatial_resolution: Isotropic resolution (mm) for reference image. Default 1.0.
            reference_buffer_factor: Buffer factor around mesh for reference image. Default 0.25.
            log_level: Logging level.
        """
        super().__init__(
            class_name="WorkflowCreateStatisticalModel", log_level=log_level
        )
        self.sample_meshes = list(sample_meshes)
        self.reference_mesh = reference_mesh
        self.pca_number_of_components = pca_number_of_components
        self.reference_spatial_resolution = reference_spatial_resolution
        self.reference_buffer_factor = reference_buffer_factor

        self.contour_tools = ContourTools()
        self.transform_tools = TransformTools()

        # Set by pipeline
        self.reference_surface: Optional[pv.PolyData] = None
        self.sample_surfaces: list[pv.PolyData] = []
        self.sample_ids: list[str] = []
        self.aligned_surfaces: list[pv.PolyData] = []
        self.forward_transforms: list = []
        self.inverse_transforms: list = []
        self.pca_input_surfaces: list[pv.PolyData] = []
        self.pca_fitted: Optional[PCA] = None
        self.pca_mean_surface: Optional[pv.PolyData] = None
        self.pca_mean_mesh: Optional[pv.UnstructuredGrid] = None

    def set_pca_number_of_components(self, n: int) -> None:
        """Set number of PCA components to retain."""
        self.pca_number_of_components = n

    def _step1_extract_surfaces(self) -> None:
        """Extract reference surface and all sample surfaces (notebook 1)."""
        self.log_section("Step 1: Extract reference and sample surfaces", width=70)
        if not self.sample_meshes:
            raise ValueError("sample_meshes must not be empty")
        self.reference_surface = _extract_surface(self.reference_mesh)
        self.log_info(
            "Reference surface: %d points",
            self.reference_surface.n_points,
        )
        self.sample_surfaces = []
        self.sample_ids = []
        for i, mesh in enumerate(self.sample_meshes):
            surface = _extract_surface(mesh)
            self.sample_surfaces.append(surface)
            self.sample_ids.append(str(i))
        self.log_info("Extracted %d sample surfaces", len(self.sample_surfaces))

    def _step2_icp_align(self) -> None:
        """ICP (affine) align each sample surface to reference (notebook 2)."""
        self.log_section("Step 2: ICP alignment to reference surface", width=70)
        assert self.reference_surface is not None and self.sample_surfaces
        self.aligned_surfaces = []
        self.forward_transforms = []
        self.inverse_transforms = []

        for i, (sid, moving) in enumerate(zip(self.sample_ids, self.sample_surfaces)):
            self.log_info(
                "ICP aligning %s (%d/%d)", sid, i + 1, len(self.sample_surfaces)
            )
            if isinstance(moving, pv.UnstructuredGrid):
                moving = moving.extract_surface()
            registrar = RegisterModelsICP(fixed_model=self.reference_surface)
            result = registrar.register(
                moving_model=moving,
                transform_type="Affine",
                max_iterations=2000,
            )
            self.aligned_surfaces.append(result["registered_model"])
            self.forward_transforms.append(result["forward_point_transform"])
            self.inverse_transforms.append(result["inverse_point_transform"])

        self.log_info(
            "ICP alignment complete for %d samples", len(self.aligned_surfaces)
        )

    def _step3_deformable_correspondence(self) -> None:
        """Deformable registration of each aligned sample to reference (notebook 3)."""
        self.log_section("Step 3: Deformable registration (correspondence)", width=70)
        assert self.reference_surface is not None and self.aligned_surfaces
        reference_image = self.contour_tools.create_reference_image(
            mesh=self.reference_surface,
            spatial_resolution=self.reference_spatial_resolution,
            buffer_factor=self.reference_buffer_factor,
            ptype=itk.UC,
        )
        self.forward_transforms = []
        self.inverse_transforms = []

        for i, (sid, moving) in enumerate(zip(self.sample_ids, self.aligned_surfaces)):
            self.log_info(
                "Deformable registration %s (%d/%d)",
                sid,
                i + 1,
                len(self.aligned_surfaces),
            )
            registrar = RegisterModelsDistanceMaps(
                moving_model=moving,
                fixed_model=self.reference_surface,
                reference_image=reference_image,
            )
            result = registrar.register(
                transform_type="Deformable",
                use_icon=False,
            )
            self.forward_transforms.append(result["forward_transform"])
            self.inverse_transforms.append(result["inverse_transform"])

        self.log_info(
            "Deformable registration complete for %d samples",
            len(self.forward_transforms),
        )

    def _step4_build_pca_inputs(self) -> None:
        """Build corresponded shapes in reference space (notebook 4).

        For each case, reference_surface is warped by forward (image) deformation
        (= inverse point) transform from step 3, so that we get reference topology
        in ICP-aligned space with residual deformation per subject to be used as PCA
        input.
        """
        self.log_section("Step 4: Build PCA inputs (corresponded shapes)", width=70)
        assert self.reference_surface is not None and self.forward_transforms
        self.pca_input_surfaces = []
        for fwd_tfm in self.forward_transforms:
            pca_input_surface = self.contour_tools.transform_contours(
                self.reference_surface, tfm=fwd_tfm, with_deformation_magnitude=False
            )
            self.pca_input_surfaces.append(pca_input_surface)
        self.log_info(
            "Built %d corresponded surfaces for PCA", len(self.pca_input_surfaces)
        )

    def _step5_compute_pca(self) -> None:
        """Compute PCA and mean surface (notebook 5)."""
        self.log_section("Step 5: Compute PCA model", width=70)
        assert self.reference_surface is not None and self.pca_input_surfaces
        template = self.reference_surface
        n_points = template.n_points

        data_matrix = []
        for i, mesh in enumerate(self.pca_input_surfaces):
            if mesh.n_points != n_points:
                raise ValueError(
                    f"Sample {self.sample_ids[i]} has {mesh.n_points} points, "
                    f"expected {n_points}. Topology must match."
                )
            data_matrix.append(mesh.points.flatten())
        data_matrix = np.array(data_matrix)

        if data_matrix.shape[0] - 1 < 2:
            raise ValueError(
                f"At least 2 samples are required for PCA. Got {data_matrix.shape[0]} samples."
            )
        n_comp = min(self.pca_number_of_components, data_matrix.shape[0] - 1)
        if n_comp < self.pca_number_of_components:
            self.log_warning(
                "Reducing PCA components from %d to %d (n_samples=%d)",
                self.pca_number_of_components,
                n_comp,
                data_matrix.shape[0],
            )
        self.pca_fitted = PCA(n_components=n_comp)
        self.pca_fitted.fit(data_matrix)

        self.pca_mean_surface = template.copy()
        self.pca_mean_surface.points = self.pca_fitted.mean_.reshape(-1, 3)
        self.log_info(
            "PCA complete: %d components, variance explained %.4f",
            len(self.pca_fitted.explained_variance_ratio_),
            self.pca_fitted.explained_variance_ratio_.sum(),
        )

        reference_image = self.contour_tools.create_reference_image(
            mesh=self.pca_mean_surface,
            spatial_resolution=self.reference_spatial_resolution,
            buffer_factor=self.reference_buffer_factor,
            ptype=itk.UC,
        )
        mean_deformation_array = self.pca_mean_surface.points - template.points
        mean_deformation_field = self.contour_tools.create_deformation_field(
            points=template.points,
            point_displacements=mean_deformation_array,
            reference_image=reference_image,
            blur_sigma=2.5,
            ptype=itk.D,
        )
        mean_deformation_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        mean_deformation_transform.SetDisplacementField(mean_deformation_field)
        self.pca_mean_mesh = self.contour_tools.transform_contours(
            self.reference_mesh,
            tfm=mean_deformation_transform,
            with_deformation_magnitude=False,
        )

    def _build_result(self) -> dict[str, Any]:
        """Build result dictionary: surfaces, meshes, and PCA model structure."""
        assert self.pca_mean_surface is not None and self.pca_fitted is not None
        result: dict[str, Any] = {
            "pca_mean_surface": self.pca_mean_surface,
            "pca_mean_mesh": self.pca_mean_mesh,
            "pca_model": {
                "explained_variance_ratio": self.pca_fitted.explained_variance_ratio_.tolist(),
                "eigenvalues": self.pca_fitted.explained_variance_.tolist(),
                "components": [c.tolist() for c in self.pca_fitted.components_],
            },
            "pca_fitted": self.pca_fitted,
        }
        return result

    def run_workflow(self) -> dict[str, Any]:
        """Run the full pipeline and return a dictionary of results (no file I/O).

        Returns:
            dict with keys:
                - pca_mean_surface: pv.PolyData mean shape surface
                - pca_mean_mesh: pv.UnstructuredGrid reference volume mesh, or None if reference was surface-only
                - pca_model: dict with "explained_variance_ratio", "eigenvalues", "components" (same structure as pca_model.json)
                - pca_fitted: fitted sklearn PCA object
        """
        self.log_section("STARTING CREATE STATISTICAL MODEL WORKFLOW", width=70)
        self._step1_extract_surfaces()
        self._step2_icp_align()
        self._step3_deformable_correspondence()
        self._step4_build_pca_inputs()
        self._step5_compute_pca()
        result = self._build_result()
        self.log_section("CREATE STATISTICAL MODEL WORKFLOW COMPLETE", width=70)
        return result
