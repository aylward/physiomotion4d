"""Model-to-image and model-to-model registration for anatomical models.

This module provides the ModelToPatientWorkflow class for registering generic
anatomical models to patient-specific imaging data and surface models.
The workflow includes:
1. Rough alignment using ICP (RegisterModelsICP)
1.5. Optional PCA-based registration (RegisterModelsPCA) if PCA data provided
2. Mask-based deformable registration (RegisterModelsDistanceMaps)
3. Optional final mask-to-image refinement using Icon

The registration is particularly useful for cardiac modeling where a generic heart model
needs to be fitted to patient-specific imaging data.

Key Features:
    - Automatic mask generation if not provided by user
    - Modular design using RegisterModelsICP, RegisterModelsPCA, and RegisterModelsDistanceMaps
    - Multi-stage registration pipeline: ICP → (optional PCA) → mask-to-mask → mask-to-image
    - Optional PCA-based shape fitting with SlicerSALT format support
    - Support for multi-label anatomical structures
    - Optional Icon-based final refinement
"""

import logging

import itk
import numpy as np
import pyvista as pv
from itk import TubeTK as ttk

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.register_models_distance_maps import RegisterModelsDistanceMaps
from physiomotion4d.register_models_icp import RegisterModelsICP
from physiomotion4d.register_models_pca import RegisterModelsPCA
from physiomotion4d.transform_tools import TransformTools


class HeartModelToPatientWorkflow(PhysioMotion4DBase):
    """Register anatomical models using multi-stage ICP, mask-based, and image-based registration.

    This class provides a flexible workflow for registering generic anatomical models
    (e.g., cardiac models) to patient-specific surface models and images. The
    registration pipeline combines:
    - Initial model alignment using RegisterModelsICP (centroid + affine ICP)
    - Mask-based deformable registration using RegisterModelsDistanceMaps (ANTs)
    - Optional final mask-to-image refinement using Icon registration

    **Registration Pipeline:**
        1. **ICP Alignment**: Rough affine alignment using RegisterModelsICP
        2. **PCA Registration**: Performs PCA-based shape fitting using RegisterModelsPCA
        3. **Mask-to-Mask**: Deformable registration using RegisterModelsDistanceMaps
        4. **Mask-to-Image**: Final refinement

    **Mask Configuration:**
        Masks are automatically generated from models if not provided by the user
        via set_masks(). Auto-generated masks use mask_dilation_mm parameter.

    Attributes:
        template_model (pv.UnstructuredGrid): Generic anatomical model to be registered
        template_model_surface (pv.PolyData): Surface extracted from template_model_surface
        template_model_mask (itk.Image): Binary/multi-label mask for model model
        template_model_roi (itk.Image): ROI mask for model model
        patient_models (list of pv.PolyData): Patient-specific surface models
        patient_model_surface (pv.PolyData): Primary patient model surface (first in list)
        patient_image (itk.Image): Reference image providing coordinate frame
        patient_mask (itk.Image): Binary/multi-label mask for patient model
        patient_roi (itk.Image): ROI mask for patient model
        mask_dilation_mm (float): Dilation for mask generation
        roi_dilation_mm (float): Dilation for ROI mask
        transform_tools (TransformTools): Transform utilities
        registrar_icon (RegisterImagesICON): ICON registration instance
        registrar_ants (RegisterImagesANTs): ANTs registration instance
        pca_json_filename (str): PCA JSON filename (optional)
        pca_group_key (str): PCA group key (optional)
        pca_number_of_modes (int): Number of PCA modes to use
        icp_forward_point_transform : ICP transforms
        icp_inverse_point_transform : ICP inverse transforms
        icp_template_model_surface: template model surface after ICP alignment
        pca_coefficients: PCA shape coefficients (if PCA used)
        pca_template_model_surface: template model surface after PCA registration (if PCA used)
        m2m_forward_transform: Mask-to-mask forward transform
        m2m_inverse_transform: Mask-to-mask inverse transform
        m2m_template_model_surface: template model surface after mask-to-mask registration
        m2i_forward_transform: Mask-to-image forward transform
        m2i_inverse_transform: Mask-to-image inverse transform
        m2i_template_model_surface: template model surface after mask-to-image registration
        m2i_template_labelmap: template labelmap after mask-to-image registration
        registered_template_model: Final registered model
        registered_template_model_surface: Final registered model surface

    Example:
        >>> # Initialize with minimal parameters
        >>> registrar = HeartModelToPatientWorkflow(
        ...     template_model=heart_model,
        ...     patient_models=[lv_model, mc_model, rv_model],
        ...     patient_image=patient_ct,
        ...     pca_json_filename='path/to/pca.json',
        ...     pca_group_key='All',
        ...     pca_number_of_modes=10
        ... )
        >>>
        >>> # Optional: Configure parameters (masks auto-generated if not set)
        >>> registrar.set_roi_dilation_mm(20)
        >>>
        >>> # Run registration
        >>> patient_model = registrar.run_workflow()
    """

    def __init__(
        self,
        template_model: pv.UnstructuredGrid,
        template_labelmap: itk.Image,
        template_labelmap_heart_muscle_ids: list[int],
        template_labelmap_chamber_ids: list[int],
        template_labelmap_background_ids: list[int],
        patient_models: list,
        patient_image: itk.Image,
        pca_json_filename: str | None = None,
        pca_group_key: str = 'All',
        pca_number_of_modes: int = 0,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the model-to-image-and-model registration pipeline.

        Args:
            template_model: Generic anatomical model to be registered
            patient_models: List of patient-specific models extracted from imaging
                data. Typically 3 models for cardiac applications: LV, myocardium, RV.
            patient_image: Patient image data providing the target coordinate frame
                (origin, spacing, direction). Used as reference for registration.
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO
        """
        # Initialize base class with logging
        super().__init__(class_name="HeartModelToPatientWorkflow", log_level=log_level)

        self.template_model = template_model
        self.template_model_surface = template_model.extract_surface()
        self.template_labelmap = template_labelmap
        self.template_labelmap_heart_muscle_ids = template_labelmap_heart_muscle_ids
        self.template_labelmap_chamber_ids = template_labelmap_chamber_ids
        self.template_labelmap_background_ids = template_labelmap_background_ids

        self.patient_models = patient_models
        patient_models_surfaces = [model.extract_surface() for model in patient_models]
        combined_patient_model = pv.merge(patient_models_surfaces)
        self.patient_model_surface = combined_patient_model.extract_surface()

        self.patient_image = patient_image

        resampler = ttk.ResampleImage.New(Input=self.patient_image)
        resampler.SetMakeHighResIso(True)
        resampler.Update()
        self.patient_image = resampler.GetOutput()

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()

        self.registrar_ants = RegisterImagesANTs()
        self.registrar_ants.set_number_of_iterations([5, 2, 5])
        # Icon registration for final mask-to-image step
        self.registrar_icon = RegisterImagesICON()
        self.registrar_icon.set_modality('ct')
        self.registrar_icon.set_mass_preservation(False)
        self.registrar_icon.set_multi_modality(True)
        self.registrar_icon.set_number_of_iterations(50)

        # Mask configuration (auto-generated)
        self.template_model_mask = None
        self.patient_mask = None
        self.template_model_roi = None
        self.patient_roi = None

        # Parameters for mask generation and processing
        self.mask_dilation_mm = 5  # For auto-generated mask dilation
        self.roi_dilation_mm = 20  # For ROI mask generation

        # Stage 1: ICP alignment results
        self.icp_registrar = None
        self.icp_inverse_point_transform = None
        self.icp_forward_point_transform = None
        self.icp_template_model_surface = None
        self.icp_template_labelmap = None

        # Stage 1.5: PCA registration results (optional)
        self.pca_registrar = None
        self.pca_forward_point_transform = None
        self.pca_inverse_point_transform = None
        self.pca_json_filename = pca_json_filename
        self.pca_number_of_modes = pca_number_of_modes
        self.pca_group_key = pca_group_key
        self.pca_coefficients = None
        self.pca_template_model_surface = None
        self.pca_template_labelmap = None

        # Stage 2: Mask-to-mask registration results
        self.use_m2m_registration = True
        self.m2m_inverse_transform = None
        self.m2m_forward_transform = None
        self.m2m_template_model_surface = None
        self.m2m_template_labelmap = None

        # Stage 3: Mask-to-image registration results
        self.use_m2i_registration = True
        self.m2i_inverse_transform = None
        self.m2i_forward_transform = None
        self.m2i_template_model_surface = None
        self.m2i_template_labelmap = None

        self.use_icon_registration_refinement = False

        # Final result
        self.registered_template_model = None
        self.registered_template_model_surface = None

    def _auto_generate_mask(
        self, models: list[pv.UnstructuredGrid], dilate_mm: float | None = None
    ) -> itk.Image:
        """Auto-generate binary masks from models.

        Creates binary masks from list of models, with dilation
        according to mask_dilation_mm parameter.
        """
        self.log_info(
            f"Auto-generating masks from models (dilation: {self.mask_dilation_mm}mm)..."
        )

        # Generate patient mask (single model or multi-label)
        if len(models) == 1:
            mask = self.contour_tools.create_mask_from_mesh(
                models[0],
                self.patient_image,
            )
        else:
            # Create multi-label mask
            mask_arr = None
            for i, model in enumerate(models):
                mask = self.contour_tools.create_mask_from_mesh(
                    model,
                    self.patient_image,
                )
                mask_arr = itk.GetArrayFromImage(mask).astype(np.uint8)
                if i == 0:
                    mask_arr = mask_arr * (i + 1)  # Label 1, 2, 3, ...
                else:
                    mask_arr = np.where(mask_arr > 0, (i + 1) * mask_arr, 0)
            mask = itk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask.CopyInformation(self.patient_image)

        # Apply dilation if requested
        if dilate_mm is None:
            dilate_mm = self.mask_dilation_mm
        if dilate_mm > 0:
            imMath = ttk.ImageMath.New(mask)
            dilation_voxels = int(dilate_mm / self.patient_image.GetSpacing()[0])
            imMath.Dilate(dilation_voxels, 1, 0)
            mask = imMath.GetOutputUChar()

        self.log_info("Masks auto-generated successfully.")

        return mask

    def _auto_generate_roi_mask(
        self, mask: itk.Image, dilate_mm: float | None = None
    ) -> itk.Image:
        """Auto-generate ROI mask from existing masks with dilation.

        Uses self.roi_dilation_mm for dilation amount.

        Note:
            Requires masks to exist (auto-generated or user-provided).
        """
        self.log_info(
            f"Auto-generating ROI masks (dilation: {self.roi_dilation_mm}mm)..."
        )

        if dilate_mm is None:
            dilate_mm = self.roi_dilation_mm

        # Generate model ROI mask
        roi = None
        if dilate_mm > 0:
            imMath = ttk.ImageMath.New(mask)
            dilation_voxels = int(dilate_mm / mask.GetSpacing()[0])
            imMath.Dilate(dilation_voxels, 1, 0)
            roi = imMath.GetOutputUChar()
        else:
            roi = mask

        self.log_info("ROI masks auto-generated successfully.")
        return roi

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

    def set_use_mask_to_mask_registration(self, use_mask_to_mask_registration: bool):
        """Set whether to use mask-to-mask registration.

        Args:
            use_mask_to_mask_registration: Whether to use mask-to-mask registration. Default: True
        """
        self.use_m2m_registration = use_mask_to_mask_registration

    def set_use_mask_to_image_registration(self, use_mask_to_image_registration: bool):
        """Set whether to use mask-to-image registration.

        Args:
            use_m2i: Whether to use mask-to-image registration. Default: True
        """
        self.use_m2i_registration = use_mask_to_image_registration

    def register_model_to_model_icp(self):
        """Perform ICP alignment of template model to patient model.

        Uses RegisterModelsICP class for ICP alignment.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': used to warp an image from model to patient space
                - 'inverse_transform': used to warp an image from patient to model space
                - 'registered_template_model_surface': Transformed model model surface
        """
        self.log_section("Stage 1: ICP Alignment (RegisterModelsICP)", width=70)

        # Create ICP registrar
        self.icp_registrar = RegisterModelsICP(
            moving_model=self.template_model_surface,
            fixed_model=self.patient_model_surface,
        )

        # Run rigid ICP registration
        icp_result = self.icp_registrar.register(mode='rigid', max_iterations=2000)

        # Store results
        # Note: Point transforms are in opposite direction from image transforms
        self.icp_forward_point_transform = icp_result['forward_point_transform']
        self.icp_inverse_point_transform = icp_result['inverse_point_transform']
        self.icp_template_model_surface = icp_result['registered_model']

        self.icp_template_labelmap = self.transform_tools.transform_image(
            self.template_labelmap,
            self.icp_inverse_point_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 1 complete: ICP alignment finished.")

        self.registered_template_model_surface = self.icp_template_model_surface

        return {
            'inverse_point_transform': self.icp_inverse_point_transform,
            'forward_point_transform': self.icp_forward_point_transform,
            'registered_template_model_surface': self.icp_template_model_surface,
            'registered_template_labelmap': self.icp_template_labelmap,
        }

    def register_model_to_model_pca(self):
        """Perform PCA-based registration after ICP alignment.

        Uses RegisterModelsPCA class for intensity-based PCA registration.
        This method requires PCA data to be set via set_pca_data().

        Returns:
            dict: Dictionary containing:
                - 'forward_point_transform': Rigid transform from PCA registration
                - 'pca_coefficients': PCA shape coefficients
                - 'registered_template_model_surface': PCA-registered model surface

        Raises:
            ValueError: If PCA data has not been set
        """
        self.log_section(
            "Stage 2: PCA-Based Registration (RegisterModelsPCA)",
            width=70,
        )

        if self.pca_json_filename is None:
            self.pca_template_model_surface = self.icp_template_model_surface
            return {
                'pca_coefficients': None,
                'registered_model_surface': self.pca_template_model_surface,
            }

        self.pca_registrar = RegisterModelsPCA.from_slicersalt(
            pca_template_model=self.template_model_surface,
            pca_json_filename=self.pca_json_filename,
            pca_group_key=self.pca_group_key,
            pca_number_of_modes=self.pca_number_of_modes,
            post_pca_transform=self.icp_forward_point_transform,
            fixed_model=self.patient_model_surface,
            reference_image=self.patient_image,
        )

        # Run complete PCA registration
        result = self.pca_registrar.register()
        self.pca_coefficients = result['pca_coefficients']
        self.pca_template_model_surface = result['registered_model']

        pca_transforms = self.pca_registrar.compute_pca_transforms(
            reference_image=self.template_labelmap,
        )
        self.pca_forward_point_transform = pca_transforms['forward_point_transform']
        self.pca_inverse_point_transform = pca_transforms['inverse_point_transform']

        # Store results

        self.registered_template_model_surface = self.pca_template_model_surface

        self.pca_template_labelmap = self.transform_tools.transform_image(
            self.template_labelmap,
            self.pca_inverse_point_transform,
            self.template_labelmap,
        )
        self.pca_template_labelmap = self.transform_tools.transform_image(
            self.pca_template_labelmap,
            self.icp_inverse_point_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 2 complete: PCA registration finished.")

        return {
            'pca_coefficients': self.pca_coefficients,
            'forward_point_transform': self.pca_forward_point_transform,
            'inverse_point_transform': self.pca_inverse_point_transform,
            'registered_template_model_surface': self.pca_template_model_surface,
            'registered_template_labelmap': self.pca_template_labelmap,
        }

    def register_mask_to_mask(self, use_icon_refinement: bool = False) -> dict | None:
        """Perform mask-based deformable registration of model to patient model.

        Uses RegisterModelsDistanceMaps class for ANTs deformable registration.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': model to patient space transform
                - 'inverse_transform': patient to model space transform
                - 'registered_template_model_surface': Transformed model model
                - 'registered_template_labelmap': Transformed model labelmap
        """
        self.log_section(
            "Stage 3: Mask-to-Mask Deformable Registration (RegisterModelsDistanceMaps)",
            width=70,
        )

        if not self.use_m2m_registration:
            self.log_info("Mask-to-mask registration is not enabled.")
            return None

        # Create mask-based registrar
        mask_registrar = RegisterModelsDistanceMaps(
            moving_model=self.pca_template_model_surface,
            fixed_model=self.patient_model_surface,
            reference_image=self.patient_image,
            roi_dilation_mm=self.roi_dilation_mm,
        )

        # Run deformable registration
        mask_result = mask_registrar.register(
            transform_type='Deformable',
            use_icon=use_icon_refinement,
        )

        # Store results
        self.m2m_forward_transform = mask_result['forward_transform']
        self.m2m_inverse_transform = mask_result['inverse_transform']
        self.m2m_template_model_surface = mask_result['registered_model']

        self.registered_template_model_surface = self.m2m_template_model_surface

        self.m2m_template_labelmap = self.transform_tools.transform_image(
            self.pca_template_labelmap,
            self.m2m_forward_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 3 complete: Mask-to-mask registration finished.")

        return {
            'forward_transform': self.m2m_forward_transform,
            'inverse_transform': self.m2m_inverse_transform,
            'registered_template_model_surface': self.m2m_template_model_surface,
            'registered_template_labelmap': self.m2m_template_labelmap,
        }

    def register_labelmap_to_image(
        self, use_icon_refinement: bool = False
    ) -> dict | None:
        """Perform labelmap-to-image refinement.

        Uses registration to align labelmap to actual image intensities.

        Returns:
            dict: Dictionary containing:
                - 'inverse_transform': patient to model space transform
                - 'forward_transform': model to patient space transform
                - 'registered_template_model_surface': Transformed model model
                - 'registered_template_labelmap': Transformed model labelmap
        """
        self.log_section(
            "Stage 4: Labelmap-to-Image Refinement (Icon Registration)", width=70
        )

        labelmap_arr = itk.GetArrayFromImage(self.m2m_template_labelmap).astype(
            np.uint16
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_background_ids),
            0,
            labelmap_arr,
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_heart_muscle_ids),
            0,
            labelmap_arr,
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_chamber_ids), 1, labelmap_arr
        )
        labelmap = itk.GetImageFromArray(labelmap_arr)
        labelmap.CopyInformation(self.m2m_template_labelmap)

        labelmap_roi = self._auto_generate_roi_mask(labelmap)

        patient_mask = self._auto_generate_mask(
            [self.patient_model_surface], dilate_mm=0
        )
        patient_roi = self._auto_generate_roi_mask(patient_mask)

        self.registrar_ants.set_fixed_image(self.patient_image)
        self.registrar_ants.set_fixed_mask(patient_roi)

        result = self.registrar_ants.register(
            moving_image=labelmap, moving_mask=labelmap_roi
        )
        self.m2i_inverse_transform = result["inverse_transform"]
        self.m2i_forward_transform = result["forward_transform"]

        if use_icon_refinement:
            # Configure Icon registration
            self.registrar_icon.set_fixed_image(self.patient_image)
            self.registrar_icon.set_fixed_mask(patient_roi)

            # Perform Icon registration
            result = self.registrar_icon.register(
                initial_forward_transform=self.m2i_forward_transform,
                moving_image=labelmap,
                moving_mask=labelmap_roi,
            )
            self.m2i_inverse_transform = result["inverse_transform"]
            self.m2i_forward_transform = result["forward_transform"]

        # Transform model with Icon result
        self.m2i_template_model_surface = self.transform_tools.transform_pvcontour(
            self.m2m_template_model_surface,
            self.m2i_inverse_transform,
            with_deformation_magnitude=True,
        )

        self.m2i_template_labelmap = self.transform_tools.transform_image(
            self.template_labelmap,
            self.m2i_forward_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 4 complete: Mask-to-image registration finished.")

        self.registered_template_model_surface = self.m2i_template_model_surface

        return {
            'inverse_transform': self.m2i_inverse_transform,
            'forward_transform': self.m2i_forward_transform,
            'registered_template_model_surface': self.m2i_template_model_surface,
            'registered_template_labelmap': self.m2i_template_labelmap,
        }

    def transform_model(self, base_model=None) -> pv.UnstructuredGrid | None:
        """Apply registration transforms to the model.

        Transforms the model through all registration stages.

        Args:
            base_model: Base model for generating the new model.
                If None, the template model is used.

        Returns:
            pv.UnstructuredGrid: Registered model
        """
        self.log_info("Applying transforms to model...")

        new_model = None
        if base_model is None:
            self.registered_template_model = self.template_model.copy(deep=True)
            new_points = self.registered_template_model.points
        else:
            new_model = base_model.copy(deep=True)
            new_points = new_model.points

        n_points = new_points.shape[0]
        progress_interval = max(1, n_points // 10)  # Report progress every 10%

        # Transform each point through the complete pipeline
        p = itk.Point[itk.D, 3]()
        for i, point in enumerate(new_points):
            # Report progress
            if i % progress_interval == 0 or i == n_points - 1:
                self.log_progress(i + 1, n_points, prefix="Transforming model points")

            p[0] = float(point[0])
            p[1] = float(point[1])
            p[2] = float(point[2])

            # Apply PCA and ICP transforms
            if self.pca_coefficients is not None:
                p = self.pca_registrar.transform_point(
                    p,
                    include_post_pca_transform=True,
                )

            # Apply mask-to-mask transform
            if self.use_m2m_registration and self.m2m_inverse_transform is not None:
                p = self.m2m_inverse_transform.TransformPoint(p)

            # Apply mask-to-image transform
            if self.use_m2i_registration and self.m2i_inverse_transform is not None:
                p = self.m2i_inverse_transform.TransformPoint(p)

            new_points[i, 0] = p[0]
            new_points[i, 1] = p[1]
            new_points[i, 2] = p[2]

        self.log_info("Transform application complete.")

        if base_model is None:
            self.registered_template_model.points = new_points
            return self.registered_template_model
        else:
            new_model.points = new_points
            return new_model

    def run_workflow(
        self,
        use_mask_to_image_registration: bool = True,
        use_mask_to_mask_registration: bool = True,
        use_icon_registration_refinement: bool = False,
    ) -> dict:
        """Execute the complete multi-stage registration workflow.

        Runs registration stages in sequence:
        1. ICP alignment (RegisterModelsICP)
        2. PCA registration (PCA data was provided)
        3. Mask-to-mask deformable registration (RegisterModelsDistanceMaps)
        4. Optional mask-to-image refinement (Icon)

        Args:
            use_mask_to_image_registration: Whether to include mask-to-image registration stage.
                Default: True
            use_mask_to_mask_registration: Whether to include mask-to-mask registration stage.
                Default: True
            use_icon_registration_refinement: Whether to include icon registration refinement stage.
                Default: False

        Returns:
            pv.UnstructuredGrid: Final registered model
        """
        self.log_section(
            "STARTING COMPLETE MODEL-TO-IMAGE-AND-MODEL REGISTRATION WORKFLOW", width=70
        )

        self.use_m2m_registration = use_mask_to_mask_registration
        self.use_m2i_registration = use_mask_to_image_registration
        self.use_icon_registration_refinement = use_icon_registration_refinement

        # Stage 1: ICP alignment
        self.register_model_to_model_icp()

        # Stage 2: Optional PCA registration (if PCA data was set)
        self.register_model_to_model_pca()

        # Stage 3: Mask-to-mask deformable registration
        if self.use_m2m_registration:
            self.register_mask_to_mask(
                use_icon_refinement=use_icon_registration_refinement
            )

        # Stage 4: Optional mask-to-image refinement
        if self.use_m2i_registration:
            self.register_mask_to_image(
                use_icon_refinement=use_icon_registration_refinement
            )

        _ = self.transform_model()

        self.log_section("REGISTRATION WORKFLOW COMPLETE", width=70)
        self.log_info(
            f"Final registered patient model surface: {self.registered_template_model_surface.n_points} points"
        )

        return {
            'registered_template_model': self.registered_template_model,
            'registered_template_model_surface': self.registered_template_model_surface,
        }
