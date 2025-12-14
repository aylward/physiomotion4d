"""Model-to-image and model-to-model registration for anatomical models.

This module provides the ModelToPatientWorkflow class for registering generic
anatomical models to patient-specific imaging data and surface meshes.
The workflow includes:
1. Rough alignment using ICP (RegisterModelToModelICP)
1.5. Optional PCA-based registration (RegisterModelToImagePCA) if PCA data provided
2. Mask-based deformable registration (RegisterModelToModelMasks)
3. Optional final mask-to-image refinement using Icon

The registration is particularly useful for cardiac modeling where a generic heart model
needs to be fitted to patient-specific imaging data.

Key Features:
    - Automatic mask generation if not provided by user
    - Modular design using RegisterModelToModelICP, RegisterModelToImagePCA, and RegisterModelToModelMasks
    - Multi-stage registration pipeline: ICP → (optional PCA) → mask-to-mask → mask-to-image
    - Optional PCA-based shape fitting with SlicerSALT format support
    - Support for multi-label anatomical structures
    - Optional Icon-based final refinement

Example:
    >>> import itk
    >>> import pyvista as pv
    >>> from physiomotion4d import HeartModelToPatientWorkflow
    >>>
    >>> # Load patient data
    >>> patient_surfaces = [pv.read("lv.stl"), pv.read("mc.stl"), pv.read("rv.stl")]
    >>> reference_image = itk.imread("patient_ct.nii.gz")
    >>>
    >>> # For PCA-based workflow, use a dummy mesh initially (will be replaced)
    >>> dummy_mesh = patient_surfaces[0]  # Placeholder
    >>>
    >>> # Initialize registration
    >>> registrar = HeartModelToPatientWorkflow(
    ...     moving_mesh=dummy_mesh,
    ...     fixed_meshes=patient_surfaces,
    ...     fixed_image=reference_image,
    ... )
    >>>
    >>> # Load PCA model from SlicerSALT format (replaces moving mesh)
    >>> registrar.set_pca_data_from_slicersalt(
    ...     json_filename='path/to/pca.json',
    ...     group_key='All',
    ...     n_pca_modes=10
    ... )
    >>>
    >>> # Run complete workflow
    >>> registered_mesh = registrar.run_workflow()
"""

import logging
from typing import Optional

import itk
import numpy as np
import pyvista as pv
from itk import TubeTK as ttk

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.register_model_to_image_pca import RegisterModelToImagePCA
from physiomotion4d.register_model_to_model_icp import RegisterModelToModelICP
from physiomotion4d.register_model_to_model_masks import RegisterModelToModelMasks
from physiomotion4d.transform_tools import TransformTools


class HeartModelToPatientWorkflow(PhysioMotion4DBase):
    """Register anatomical models using multi-stage ICP, mask-based, and image-based registration.

    This class provides a flexible workflow for registering generic anatomical models
    (e.g., cardiac meshes) to patient-specific surface meshes and images. The
    registration pipeline combines:
    - Initial mesh alignment using RegisterModelToModelICP (centroid + affine ICP)
    - Mask-based deformable registration using RegisterModelToModelMasks (ANTs)
    - Optional final mask-to-image refinement using Icon registration

    **Registration Pipeline:**
        1. **ICP Alignment**: Rough affine alignment using RegisterModelToModelICP
        1.5. **PCA Registration** (optional): If PCA data is provided via set_pca_data(),
            performs PCA-based shape fitting using RegisterModelToImagePCA
        2. **Mask-to-Mask**: Deformable registration using RegisterModelToModelMasks
        3. **Mask-to-Image** (optional): Final refinement using Icon registration

    **Mask Configuration:**
        Masks are automatically generated from meshes if not provided by the user
        via set_masks(). Auto-generated masks use mask_dilation_mm parameter.

    Attributes:
        moving_original_mesh (pv.UnstructuredGrid): Generic anatomical model to be registered
        moving_mesh (pv.PolyData): Surface extracted from moving_mesh
        fixed_meshes (list of pv.PolyData): Patient-specific surface meshes
        fixed_mesh (pv.PolyData): Primary fixed mesh (first in list)
        fixed_image (itk.Image): Reference image providing coordinate frame
        moving_mask_image (itk.Image): Binary/multi-label mask for moving model
        fixed_mask_image (itk.Image): Binary/multi-label mask for fixed model
        moving_mask_roi_image (itk.Image): ROI mask for moving model
        fixed_mask_roi_image (itk.Image): ROI mask for fixed model
        mask_dilation_mm (float): Dilation for mask generation
        roi_dilation_mm (float): Dilation for ROI mask
        transform_tools (TransformTools): Transform utilities
        registrar_icon (RegisterImagesICON): ICON registration instance
        registrar_ants (RegisterImagesANTs): ANTs registration instance
        pca_eigenvectors (np.ndarray): PCA eigenvectors (optional)
        pca_std_deviations (np.ndarray): PCA standard deviations (optional)
        n_pca_modes (int): Number of PCA modes to use
        use_pca (bool): Whether PCA registration is enabled
        icp_phi_FM, icp_phi_MF: ICP transforms
        pca_rigid_transform: PCA rigid transform (if PCA used)
        pca_coefficients: PCA shape coefficients (if PCA used)
        moving_pca_mesh: Mesh after PCA registration (if PCA used)
        m2m_phi_FM, m2m_phi_MF: Mask-to-mask transforms
        m2i_phi_FM, m2i_phi_MF: Mask-to-image transforms
        moving_icp_mesh: Mesh after ICP alignment
        moving_m2m_mesh: Mesh after mask-to-mask registration
        moving_m2i_mesh: Mesh after mask-to-image registration
        moving_registered_mesh: Final registered mesh

    Example:
        >>> # Initialize with minimal parameters
        >>> registrar = HeartModelToPatientWorkflow(
        ...     moving_mesh=generic_heart,
        ...     fixed_meshes=[lv_mesh, mc_mesh, rv_mesh],
        ...     fixed_image=patient_ct,
        ... )
        >>>
        >>> # Optional: Configure parameters (masks auto-generated if not set)
        >>> registrar.set_roi_dilation_mm(20)
        >>>
        >>> # Optional: Enable PCA registration (Method 1 - from SlicerSALT file)
        >>> registrar.set_pca_data_from_slicersalt(
        ...     json_filename='path/to/pca.json',
        ...     group_key='All',
        ...     n_pca_modes=10
        ... )
        >>>
        >>> # Alternative: Enable PCA registration (Method 2 - from arrays)
        >>> # registrar.set_pca_data(
        >>> #     eigenvectors=pca_components,
        >>> #     eigenvalues=pca_eigenvalues,
        >>> #     n_pca_modes=10
        >>> # )
        >>>
        >>> # Run registration
        >>> final_mesh = registrar.run_workflow()
    """

    def __init__(
        self,
        moving_mesh: pv.UnstructuredGrid,
        fixed_meshes: list,
        fixed_image: itk.Image,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the model-to-image-and-model registration pipeline.

        Args:
            moving_mesh: Generic anatomical model mesh to be registered
            fixed_meshes: List of patient-specific surface meshes extracted from imaging
                data. Typically 3 meshes for cardiac applications: LV, myocardium, RV.
            fixed_image: Patient image data providing the target coordinate frame
                (origin, spacing, direction). Used as reference for registration.
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Note:
            The fixed image is median-filtered (radius=2) to reduce noise.
            Masks are auto-generated if not provided via set_masks().
        """
        # Initialize base class with logging
        super().__init__(class_name="HeartModelToPatientWorkflow", log_level=log_level)

        self.moving_original_mesh = moving_mesh
        self.moving_mesh = moving_mesh.extract_surface()

        self.fixed_meshes = fixed_meshes
        fixed_meshes_surfaces = [mesh.extract_surface() for mesh in fixed_meshes]
        combined_fixed_mesh = pv.merge(fixed_meshes_surfaces)
        self.fixed_mesh = combined_fixed_mesh.extract_surface()

        # Apply median filter to fixed image
        median_filter = itk.MedianImageFilter.New(Input=fixed_image)
        median_filter.SetRadius(2)
        median_filter.Update()
        self.fixed_image = median_filter.GetOutput()

        resampler = ttk.ResampleImage.New(Input=self.fixed_image)
        resampler.SetMakeHighResIso(True)
        resampler.Update()
        self.fixed_image = resampler.GetOutput()

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()

        self.registrar_pca = None
        self.n_pca_modes = -1
        self.use_pca = False

        self.registrar_ants = RegisterImagesANTs()
        self.registrar_ants.set_number_of_iterations([5, 2, 5])

        # Icon registration for final mask-to-image step
        self.registrar_icon = RegisterImagesICON()
        self.registrar_icon.set_modality('ct')
        self.registrar_icon.set_mass_preservation(False)
        self.registrar_icon.set_multi_modality(True)
        self.registrar_icon.set_number_of_iterations(50)

        # Mask configuration (to be set by user or auto-generated)
        self.moving_mask_image = None
        self.fixed_mask_image = None
        self.moving_mask_roi_image = None
        self.fixed_mask_roi_image = None

        # Parameters for mask generation and processing
        self.mask_dilation_mm = 5  # For auto-generated mask dilation
        self.roi_dilation_mm = 20  # For ROI mask generation

        # Stage 1: ICP alignment results
        self.icp_phi_FM = None
        self.icp_phi_MF = None
        self.moving_icp_mesh = None
        self.moving_icp_mask_image = None
        self.moving_icp_mask_roi_image = None

        # Stage 1.5: PCA registration results (optional)
        self.pca_rigid_transform = None
        self.pca_coefficients = None
        self.moving_pca_mesh = None

        # Stage 2: Mask-to-mask registration results
        self.m2m_phi_FM = None
        self.m2m_phi_MF = None
        self.moving_m2m_mesh = None
        self.moving_m2m_mask_image = None
        self.moving_m2m_mask_roi_image = None

        # Stage 3: Mask-to-image registration results
        self.m2i_phi_FM = None
        self.m2i_phi_MF = None
        self.moving_m2i_mesh = None
        self.moving_m2i_mask_image = None
        self.moving_m2i_mask_roi_image = None

        # Final result
        self.moving_registered_mesh = None

    def set_masks(
        self,
        moving_mask_image: Optional[itk.Image] = None,
        fixed_mask_image: Optional[itk.Image] = None,
    ):
        """Set user-provided masks for registration.

        Args:
            moving_mask_image: Binary or multi-label mask for moving model
            fixed_mask_image: Binary or multi-label mask for fixed model
        """
        self.moving_mask_image = moving_mask_image
        self.fixed_mask_image = fixed_mask_image

        self.log_info("User-provided masks configured.")

    def _auto_generate_masks(self, meshes: list[pv.UnstructuredGrid]) -> itk.Image:
        """Auto-generate binary masks from meshes.

        Creates binary masks from moving_mesh and fixed_meshes, with dilation
        according to mask_dilation_mm parameter.
        """
        self.log_info(
            f"Auto-generating masks from meshes (dilation: {self.mask_dilation_mm}mm)..."
        )

        # Generate fixed mask (single mesh or multi-label)
        if len(meshes) == 1:
            mask_image = self.contour_tools.create_mask_from_mesh(
                meshes[0],
                self.fixed_image,
                resample_to_reference=True,
            )
        else:
            # Create multi-label mask
            mask_arr = None
            for i, mesh in enumerate(meshes):
                mask = self.contour_tools.create_mask_from_mesh(
                    mesh,
                    self.fixed_image,
                )
                mask_arr = itk.GetArrayFromImage(mask).astype(np.uint8)
                if i == 0:
                    mask_arr = mask_arr * (i + 1)  # Label 1, 2, 3, ...
                else:
                    mask_arr = np.where(mask_arr > 0, (i + 1) * mask_arr, 0)
            mask_image = itk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_image.CopyInformation(self.fixed_image)

        # Apply dilation if requested
        if self.mask_dilation_mm > 0:
            imMath = ttk.ImageMath.New(mask_image)
            dilation_voxels = int(
                self.mask_dilation_mm / self.fixed_image.GetSpacing()[0]
            )
            imMath.Dilate(dilation_voxels, 1, 0)
            mask_image = imMath.GetOutputUChar()

        self.log_info("Masks auto-generated successfully.")

        return mask_image

    def set_roi_masks(
        self,
        moving_mask_roi_image: itk.Image,
        fixed_mask_roi_image: itk.Image,
    ):
        """Set user-provided ROI masks.

        Args:
            moving_mask_roi_image: Binary ROI mask for moving model
            fixed_mask_roi_image: Binary ROI mask for fixed model
        """
        self.moving_mask_roi_image = moving_mask_roi_image
        self.fixed_mask_roi_image = fixed_mask_roi_image

        self.log_info("User-provided ROI masks configured.")

    def _auto_generate_roi_masks(self, mask_image: itk.Image) -> itk.Image:
        """Auto-generate ROI masks from existing masks with dilation.

        Uses self.roi_dilation_mm for dilation amount.

        Note:
            Requires masks to exist (auto-generated or user-provided).
        """
        self.log_info(
            f"Auto-generating ROI masks (dilation: {self.roi_dilation_mm}mm)..."
        )

        # Generate moving ROI mask
        imMath = ttk.ImageMath.New(mask_image)
        dilation_voxels = int(self.roi_dilation_mm / mask_image.GetSpacing()[0])
        imMath.Dilate(dilation_voxels, 1, 0)
        mask_roi_image = imMath.GetOutputUChar()

        self.log_info("ROI masks auto-generated successfully.")
        return mask_roi_image

    def set_mask_dilation_mm(self, mask_dilation_mm: float):
        """Set mask dilation amount for auto-generated masks.

        Args:
            mask_dilation_mm: Dilation amount in millimeters for mask generation.
                Default: 5mm
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_roi_dilation_mm(self, roi_dilation_mm: float):
        """Set ROI mask dilation amount.

        Args:
            roi_dilation_mm: Dilation amount in millimeters for ROI mask generation.
                Default: 20mm
        """
        self.roi_dilation_mm = roi_dilation_mm

    def set_pca_data(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        n_modes: int = -1,
    ):
        """Set PCA eigenvalues and eigenvectors for PCA-based registration.

        When this method is called, the workflow will include PCA registration
        after ICP alignment and before mask-to-mask registration.

        Args:
            eigenvectors: PCA eigenvectors/components array. Shape: (n_pca_modes, n_points*3)
                Each row is a flattened eigenmode with 3D displacements: [x1,y1,z1, x2,y2,z2, ...]
            eigenvalues: PCA eigenvalues array. Shape: (n_pca_modes,)
                These will be converted to standard deviations internally.
            n_pca_modes: Number of PCA modes to use in registration. Default: 10

        Example:
            >>> registrar = HeartModelToPatientWorkflow(...)
            >>> registrar.set_pca_data(
            ...     eigenvectors=pca_components,
            ...     eigenvalues=pca_eigenvalues,
            ...     n_pca_modes=10
            ... )
        """
        self.log_info("Creating PCA registrar.")
        self.log_info(f"  Average mesh n_points: {len(self.moving_mesh.points)}")
        self.log_info(f"  Number of PCA modes: {n_modes}")
        self.log_info(f"  Number of eigenvalues: {len(eigenvalues)}")
        self.log_info(f"  Number of eigenvectors: {eigenvectors.shape[0]}")
        self.log_info(f"  Number of std deviations: {len(np.sqrt(eigenvalues))}")
        self.log_info(f"  Reference image shape: {self.fixed_image.shape}")
        self.registrar_pca = RegisterModelToImagePCA(
            average_mesh=self.moving_mesh,
            eigenvectors=eigenvectors,
            std_deviations=np.sqrt(eigenvalues),
            reference_image=self.fixed_image,
            n_modes=n_modes,
        )

        self.n_pca_modes = self.registrar_pca.n_pca_modes
        self.use_pca = True

        self.log_info(
            f"PCA data configured: {len(eigenvalues)} modes available, "
            f"using {n_pca_modes} modes for registration."
        )

    def set_pca_data_from_slicersalt(
        self,
        json_filename: str,
        group_key: str = 'All',
        n_modes: int = -1,
    ):
        """Load PCA data from SlicerSALT format and configure for registration.

        This convenience method loads PCA statistical shape model data from a
        SlicerSALT JSON file and automatically updates the moving mesh to use
        the PCA mean mesh. The workflow will then include PCA registration
        after ICP alignment and before mask-to-mask registration.

        Args:
            json_filename: Path to the SlicerSALT PCA JSON file (e.g., 'pca.json')
            group_key: Key for the PCA group to extract from JSON. Default: 'All'
            n_pca_modes: Number of PCA modes to use in registration. Default: 10

        Raises:
            FileNotFoundError: If JSON or VTK mesh file not found
            KeyError: If group_key not found in JSON
            ValueError: If data format is invalid

        Example:
            >>> registrar = HeartModelToPatientWorkflow(
            ...     moving_mesh=generic_heart,
            ...     fixed_meshes=[lv_mesh, mc_mesh, rv_mesh],
            ...     fixed_image=patient_ct,
            ... )
            >>> registrar.set_pca_data_from_slicersalt(
            ...     json_filename='path/to/pca.json',
            ...     group_key='All',
            ...     n_pca_modes=10
            ... )
            >>> final_mesh = registrar.run_workflow()

        Note:
            This method expects the SlicerSALT file structure:
            - A JSON file containing eigenvalues and components
            - A corresponding VTK mesh file following the pattern:
              {json_stem}_{group_key}_mean.vtk
        """
        self.log_section("Loading PCA Data from SlicerSALT Format", width=70)

        self.log_info("Creating PCA registrar from SlicerSALT data.")
        self.log_info(f"  Average mesh n_points: {len(self.moving_mesh.points)}")
        self.log_info(f"  Number of PCA modes: {n_modes}")
        self.log_info(f"  Reference image shape: {self.fixed_image.shape}")
        # Load PCA data using RegisterModelToImagePCA
        self.registrar_pca = RegisterModelToImagePCA.from_slicersalt(
            average_mesh=self.moving_mesh,
            json_filename=json_filename,
            group_key=group_key,
            reference_image=self.fixed_image,
            n_modes=n_modes,
        )
        self.use_pca = True

        self.n_pca_modes = self.registrar_pca.n_pca_modes

        self.log_section("SlicerSALT PCA Data Loaded and Configured", width=70)

    def register_mesh_to_mesh_icp(self):
        """Perform ICP alignment of moving mesh to fixed patient mesh.

        Uses RegisterModelToModelICP class for affine ICP alignment.

        Returns:
            dict: Dictionary containing:
                - 'phi_FM': Forward transform (fixed to moving)
                - 'phi_MF': Reverse transform (moving to fixed)
                - 'moving_mesh': Transformed moving mesh
                - 'moving_mask_image': Transformed moving mask image
                - 'moving_mask_roi_image': Transformed moving ROI mask image
        """
        self.log_section("Stage 1: ICP Alignment (RegisterModelToModelICP)", width=70)

        # Create ICP registrar
        icp_registrar = RegisterModelToModelICP(
            moving_mesh=self.moving_mesh,
            fixed_mesh=self.fixed_mesh,
        )

        # Run affine ICP registration
        icp_result = icp_registrar.register(mode='affine', max_iterations=2000)

        # Store results
        self.icp_phi_MF = icp_result['phi_MF']
        self.icp_phi_FM = icp_result['phi_FM']
        self.moving_icp_mesh = icp_result['moving_mesh']

        # Ensure masks exist (auto-generate if needed)
        if self.moving_mask_image is None:
            self.moving_mask_image = self._auto_generate_masks([self.moving_mesh])
        if self.fixed_mask_image is None:
            self.fixed_mask_image = self._auto_generate_masks(self.fixed_meshes)

        if self.moving_mask_roi_image is None:
            self.moving_mask_roi_image = self._auto_generate_roi_masks(
                self.moving_mask_image
            )
        if self.fixed_mask_roi_image is None:
            self.fixed_mask_roi_image = self._auto_generate_roi_masks(
                self.fixed_mask_image
            )

        # Transform moving mask images to fixed space
        self.moving_icp_mask_image = self.transform_tools.transform_image(
            self.moving_mask_image,
            self.icp_phi_MF,
            self.fixed_image,
            interpolation_method="nearest",
        )

        # Now that the moving mask is in the fixed space, we should regenerate the ROI mask
        self.moving_icp_mask_roi_image = self._auto_generate_roi_masks(
            self.moving_icp_mask_image,
        )

        self.log_info("Stage 1 complete: ICP alignment finished.")

        return {
            'phi_FM': self.icp_phi_FM,
            'phi_MF': self.icp_phi_MF,
            'moving_mesh': self.moving_icp_mesh,
            'moving_mask_image': self.moving_icp_mask_image,
            'moving_mask_roi_image': self.moving_icp_mask_roi_image,
        }

    def register_mesh_to_mesh_pca(self):
        """Perform PCA-based registration after ICP alignment.

        Uses RegisterModelToImagePCA class for intensity-based PCA registration.
        This method requires PCA data to be set via set_pca_data().

        Creates a contour distance map from the fixed mesh, clips it to 100,
        and inverts the intensities for use as the target image in PCA registration.

        Returns:
            dict: Dictionary containing:
                - 'pca_rigid_transform': Rigid transform from PCA registration
                - 'pca_coefficients': PCA shape coefficients
                - 'moving_mesh': PCA-registered mesh

        Raises:
            ValueError: If PCA data has not been set via set_pca_data()
        """
        self.log_section(
            "Stage 1.5: PCA-Based Registration (RegisterModelToImagePCA)", width=70
        )

        if not self.use_pca or self.registrar_pca is None:
            raise ValueError(
                "PCA data not set. Call set_pca_data() before using PCA registration."
            )

        # Create contour distance map from fixed mesh
        self.log_info("Creating contour distance map from fixed mesh...")
        fixed_mesh_contour_map = (
            self.contour_tools.create_contour_distance_map_from_mesh(
                mesh=self.fixed_mesh,
                reference_image=self.fixed_image,
                max_distance=100.0,
                invert_distance_map=True,
            )
        )

        self.registrar_pca.set_reference_image(fixed_mesh_contour_map)
        self.registrar_pca.set_average_mesh(self.moving_icp_mesh)

        # Create initial transform (identity, since ICP already aligned the mesh)
        initial_transform = itk.VersorRigid3DTransform[itk.D].New()
        initial_transform.SetIdentity()

        # Run complete PCA registration
        result = self.registrar_pca.register(
            initial_transform=initial_transform,
            n_pca_modes=self.n_pca_modes,
            stage1_max_iterations=10,
            stage2_max_iterations=200,
            pca_coefficient_bounds=3.0,
            rigid_refinement_bounds={'versor': 0.1, 'translation_mm': 10.0},
        )

        # Store results
        self.pca_rigid_transform = result['pre_phi_FM']
        self.pca_coefficients = result['pca_coefficients_FM']
        self.moving_pca_mesh = result['registered_mesh']

        self.moving_pca_mask_image = self._auto_generate_masks([self.moving_pca_mesh])
        self.moving_pca_mask_roi_image = self._auto_generate_roi_masks(
            self.moving_pca_mask_image
        )

        self.log_info("Stage 1.5 complete: PCA registration finished.")

        return {
            'pre_phi_FM': self.pca_rigid_transform,
            'pca_coefficients_FM': self.pca_coefficients,
            'moving_mesh': self.moving_pca_mesh,
            'moving_mask_image': self.moving_pca_mask_image,
            'moving_mask_roi_image': self.moving_pca_mask_roi_image,
        }

    def register_mask_to_mask(self):
        """Perform mask-based deformable registration of moving to fixed mesh.

        Uses RegisterModelToModelMasks class for ANTs deformable registration.
        If PCA registration was performed, uses the PCA-registered mesh as input.

        Returns:
            dict: Dictionary containing:
                - 'phi_FM': Forward transform (fixed to moving)
                - 'phi_MF': Reverse transform (moving to fixed)
                - 'moving_mesh': Transformed moving mesh
                - 'moving_mask_image': Transformed moving mask image
                - 'moving_mask_roi_image': Transformed moving ROI mask image
        """
        self.log_section(
            "Stage 2: Mask-to-Mask Deformable Registration (RegisterModelToModelMasks)",
            width=70,
        )

        # Use PCA mesh if available, otherwise use ICP mesh
        if self.use_pca and self.moving_pca_mesh is not None:
            input_mesh = self.moving_pca_mesh
            input_mask_image = self.moving_pca_mask_image
            input_mask_roi_image = self.moving_pca_mask_roi_image
        else:
            input_mesh = self.moving_icp_mesh
            input_mask_image = self.moving_icp_mask_image
            input_mask_roi_image = self.moving_icp_mask_roi_image

        # Create mask-based registrar
        mask_registrar = RegisterModelToModelMasks(
            moving_mesh=input_mesh,
            fixed_mesh=self.fixed_mesh,
            reference_image=self.fixed_image,
        )

        # Run deformable registration
        mask_result = mask_registrar.register(
            mode='deformable',
            use_icon=False,  # No Icon refinement in this stage
        )

        # Store results
        self.m2m_phi_MF = mask_result['phi_MF']
        self.m2m_phi_FM = mask_result['phi_FM']
        self.moving_m2m_mesh = mask_result['moving_mesh']

        # Transform mask images to fixed space
        self.moving_m2m_mask_image = self.transform_tools.transform_image(
            input_mask_image,
            self.m2m_phi_MF,
            self.fixed_image,
            interpolation_method="nearest",
        )
        self.moving_m2m_mask_roi_image = self.transform_tools.transform_image(
            input_mask_roi_image,
            self.m2m_phi_MF,
            self.fixed_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 2 complete: Mask-to-mask registration finished.")

        return {
            'phi_FM': self.m2m_phi_FM,
            'phi_MF': self.m2m_phi_MF,
            'moving_mesh': self.moving_m2m_mesh,
            'moving_mask_image': self.moving_m2m_mask_image,
            'moving_mask_roi_image': self.moving_m2m_mask_roi_image,
        }

    def register_mask_to_image(self):
        """Perform final mask-to-image refinement using Icon registration.

        Uses Icon registration to align mask to actual image intensities.

        Returns:
            dict: Dictionary containing:
                - 'phi_FM': Forward transform (fixed to moving)
                - 'phi_MF': Reverse transform (moving to fixed)
                - 'moving_mesh': Transformed moving mesh
                - 'moving_mask_image': Transformed moving mask image
                - 'moving_mask_roi_image': Transformed moving ROI mask image
        """
        self.log_section(
            "Stage 3: Mask-to-Image Refinement (Icon Registration)", width=70
        )

        if self.moving_m2m_mask_image is None:
            raise ValueError(
                "Moving mask image not available for mask-to-image registration. "
                "Ensure mask-to-mask registration is run first."
            )

        # Prepare moving mask image for Icon registration (scale to intensity 100)
        mmi_arr = (
            itk.GetArrayFromImage(self.moving_m2m_mask_image).astype(np.float32) * 100
        )
        if mmi_arr.min() == mmi_arr.max():
            raise ValueError(
                "Moving mask image is empty. Ensure mask-to-mask registration is run first."
            )

        mmi = itk.GetImageFromArray(mmi_arr)
        mmi.CopyInformation(self.moving_m2m_mask_image)

        self.registrar_ants.set_fixed_image(self.fixed_image)
        if self.fixed_mask_roi_image is not None:
            self.registrar_ants.set_fixed_image_mask(self.fixed_mask_roi_image)
        result = self.registrar_ants.register(
            moving_image=mmi, moving_image_mask=self.moving_m2m_mask_roi_image
        )
        phi_FM_ants = result["phi_FM"]
        phi_MF_ants = result["phi_MF"]

        # Configure Icon registration
        self.registrar_icon.set_fixed_image(self.fixed_image)
        if self.fixed_mask_roi_image is not None:
            self.registrar_icon.set_fixed_image_mask(self.fixed_mask_roi_image)

        # Perform Icon registration
        result = self.registrar_icon.register(
            moving_image=mmi, moving_image_mask=self.moving_m2m_mask_roi_image
        )
        phi_FM_icon = result["phi_FM"]
        phi_MF_icon = result["phi_MF"]

        # Compose ANTS and Icon transforms
        phi_FM = self.transform_tools.combine_displacement_field_transforms(
            phi_FM_ants, phi_FM_icon, self.fixed_image
        )
        phi_MF = self.transform_tools.combine_displacement_field_transforms(
            phi_MF_icon, phi_MF_ants, self.fixed_image
        )

        self.m2i_phi_FM = phi_FM
        self.m2i_phi_MF = phi_MF

        # Transform mesh with Icon result
        self.moving_m2i_mesh = self.transform_tools.transform_pvcontour(
            self.moving_m2m_mesh, self.m2i_phi_FM, with_deformation_magnitude=True
        )

        # Transform mask images to fixed space
        self.moving_m2i_mask_image = self.transform_tools.transform_image(
            self.moving_m2m_mask_image,
            self.m2i_phi_MF,
            self.fixed_image,
            interpolation_method="nearest",
        )
        self.moving_m2i_mask_roi_image = self.transform_tools.transform_image(
            self.moving_m2m_mask_roi_image,
            self.m2i_phi_MF,
            self.fixed_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 3 complete: Mask-to-image registration finished.")

        return {
            'phi_FM': self.m2i_phi_FM,
            'phi_MF': self.m2i_phi_MF,
            'moving_mesh': self.moving_m2i_mesh,
            'moving_mask_image': self.moving_m2i_mask_image,
            'moving_mask_roi_image': self.moving_m2i_mask_roi_image,
        }

    def apply_transforms_to_original_mesh(self, include_m2i: bool = True):
        """Apply registration transforms to the original mesh.

        Transforms the original mesh through all registration stages.
        Note: If PCA registration was used, the PCA transform is already
        baked into the mesh, so we don't apply it separately.

        Args:
            include_m2i: Whether to include mask-to-image transform. Default: True

        Returns:
            pv.UnstructuredGrid: Registered mesh
        """
        self.log_info("Applying transforms to original mesh...")

        self.moving_registered_mesh = self.moving_original_mesh.copy(deep=True)
        new_points = self.moving_registered_mesh.points

        n_points = new_points.shape[0]
        progress_interval = max(1, n_points // 10)  # Report progress every 10%

        # Transform each point through the complete pipeline
        for i in range(n_points):
            # Report progress
            if i % progress_interval == 0 or i == n_points - 1:
                self.log_progress(i + 1, n_points, prefix="Transforming mesh points")

            p = itk.Point[itk.D, 3]()
            p[0], p[1], p[2] = (
                float(new_points[i, 0]),
                float(new_points[i, 1]),
                float(new_points[i, 2]),
            )

            # Apply ICP transform
            new_p = self.icp_phi_FM.TransformPoint(p)

            # Apply PCA rigid transform (if PCA was used)
            if self.use_pca and self.pca_rigid_transform is not None:
                new_p = self.pca_rigid_transform.TransformPoint(new_p)

            # Apply mask-to-mask transform
            new_p = self.m2m_phi_FM.TransformPoint(new_p)

            # Apply mask-to-image transform (if available and requested)
            if include_m2i and self.m2i_phi_FM is not None:
                new_p = self.m2i_phi_FM.TransformPoint(new_p)

            new_points[i, 0], new_points[i, 1], new_points[i, 2] = (
                new_p[0],
                new_p[1],
                new_p[2],
            )

        self.moving_registered_mesh.points = new_points

        self.log_info("Transform application complete.")

        return self.moving_registered_mesh

    def run_workflow(self, include_mask_to_image: bool = True):
        """Execute the complete multi-stage registration workflow.

        Runs registration stages in sequence:
        1. ICP alignment (RegisterModelToModelICP)
        1.5. Optional PCA registration (if set_pca_data() was called)
        2. Mask-to-mask deformable registration (RegisterModelToModelMasks)
        3. Optional mask-to-image refinement (Icon)

        Masks are automatically generated if not provided via set_masks().

        Args:
            include_mask_to_image: Whether to include mask-to-image registration stage.
                Default: True

        Returns:
            pv.PolyData: Final registered surface mesh

        Note:
            - Masks are auto-generated from meshes if not provided via set_masks().
            - PCA registration is only performed if set_pca_data() was called.
        """
        self.log_section(
            "STARTING COMPLETE MODEL-TO-IMAGE-AND-MODEL REGISTRATION WORKFLOW", width=70
        )

        # Stage 1: ICP alignment
        self.register_mesh_to_mesh_icp()

        # Stage 1.5: Optional PCA registration (if PCA data was set)
        if self.use_pca:
            self.register_pca()

        final_mesh = self.moving_pca_mesh

        # Stage 2: Mask-to-mask deformable registration
        # self.register_mask_to_mask()

        # Stage 3: Optional mask-to-image refinement
        # if include_mask_to_image:
        # self.register_mask_to_image()
        # final_mesh = self.moving_m2i_mesh
        # else:
        # final_mesh = self.moving_m2m_mesh

        self.log_section("REGISTRATION WORKFLOW COMPLETE", width=70)
        self.log_info(f"Final registered mesh: {final_mesh.n_points} points")
        if self.use_pca:
            self.log_info("PCA registration was applied in this workflow.")

        return final_mesh
