"""Time series image registration implementation.

This module provides the RegisterTimeSeriesImages class for registering an ordered
sequence of images (time series) to a fixed image. It supports both ANTs
and ICON registration methods and can optionally use prior transforms to initialize
subsequent registrations in the sequence.

The class is particularly useful for 4D medical imaging applications such as cardiac
CT where sequential frames need to be registered to a common frame.
"""

import itk

from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools


class RegisterTimeSeriesImages(RegisterImagesBase):
    """Register a time series of images to a fixed image.

    This class extends RegisterImagesBase to provide sequential registration of
    multiple images (time series) to a fixed image. It supports both
    ANTs and ICON registration methods and can propagate information from prior
    registrations to initialize subsequent ones.

    The registration proceeds in two passes from a starting index:
    1. Forward pass: from starting_index to the end of the series
    2. Backward pass: from starting_index-1 to the beginning

    This bidirectional approach helps maintain temporal coherence in the
    registration results.

    Key features:
    - Sequential registration of ordered image lists
    - Support for both ANTs and ICON registration backends
    - Optional use of prior transforms to initialize next registration
    - Configurable starting point in the time series
    - Returns all transforms and loss values for the entire series

    Attributes:
        registration_method (str): Registration method to use ('ants' or 'icon')
        registrar (RegisterImagesBase): Internal registration object (ANTs or ICON)
        transform_tools (TransformTools): Utility for transform operations

    Example:
        >>> # Register a cardiac CT time series
        >>> registrar = RegisterTimeSeriesImages(registration_method='ants')
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(fixed_image)
        >>> registrar.set_number_of_iterations([40, 20, 10])
        >>>
        >>> # Register all time points to fixed image
        >>> result = registrar.register_time_series(
        ...     moving_images=time_series_images,
        ...     starting_index=5,  # Start from middle of cardiac cycle
        ...     register_start_to_fixed_image=True,
        ...     portion_of_prior_transform_to_init_next_transform=0.5
        ... )
        >>>
        >>> phi_MF_list = result["phi_MF_list"]
        >>> phi_FM_list = result["phi_FM_list"]
        >>> losses = result["losses"]
    """

    def __init__(self, registration_method='ants'):
        """Initialize the time series image registration class.

        Args:
            registration_method (str): Registration method to use.
                Options: 'ants' or 'icon'. Default: 'ants'

        Raises:
            ValueError: If registration_method is not 'ants' or 'icon'
        """
        super().__init__()

        self.registration_method = registration_method.lower()

        self.registrar_ants = RegisterImagesANTs()
        self.registrar_icon = RegisterImagesICON()
        if self.registration_method == 'ants':
            self.number_of_iterations = [40, 20, 10]
        elif self.registration_method == 'icon':
            self.number_of_iterations = 50
        elif self.registration_method == 'ants_icon':
            self.number_of_iterations = [[40, 20, 10], 50]
        else:
            raise ValueError(
                f"registration_method must be 'ants', 'icon' or 'ants_icon', got '{registration_method}'"
            )

        self.transform_tools = TransformTools()

        self.smooth_prior_transform_sigma = 0.5

    def set_number_of_iterations(self, number_of_iterations):
        """Set the number of iterations for registration.

        This passes through to the underlying registration method (ANTs or ICON).

        Args:
            number_of_iterations: Number of iterations to perform.
                For ANTs: list of int (e.g., [40, 20, 10] for multi-resolution)
                For ICON: int (number of fine-tuning steps)
        """
        self.number_of_iterations = number_of_iterations

    def set_smooth_prior_transform_sigma(self, smooth_prior_transform_sigma):
        """Set the sigma for smoothing the prior transform.

        Args:
            smooth_prior_transform_sigma (float): Sigma for smoothing the prior transform.
        """
        self.smooth_prior_transform_sigma = smooth_prior_transform_sigma

    def set_mask_dilation(self, mask_dilation_mm):
        """Set the dilation of the fixed and moving image masks.

        This passes through to the underlying registration method.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_modality(self, modality):
        """Set the imaging modality for registration optimization.

        This passes through to the underlying registration method.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image):
        """Set the fixed image for registration.

        All moving images in the time series will be registered to this
        fixed image.

        Args:
            fixed_image (itk.Image): The 3D fixed image
        """
        self.fixed_image = fixed_image

    def set_fixed_image_mask(self, fixed_image_mask):
        """Set a binary mask for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_image_mask (itk.Image): Binary mask defining ROI
        """
        self.fixed_image_mask = fixed_image_mask

    def register_time_series(
        self,
        moving_images,
        moving_images_masks=None,
        starting_index=0,
        register_start_to_fixed_image=True,
        portion_of_prior_transform_to_init_next_transform=0.0,
        images_are_labelmaps=False,
    ):
        """Register a time series of images to the fixed image.

        This method registers an ordered sequence of images to a common fixed
        frame. Registration proceeds bidirectionally from a starting index:
        forward to the end and backward to the beginning.

        For each image after the starting image, the method can optionally use
        the transform from the previous image to initialize the registration,
        which can improve convergence and temporal coherence.

        Args:
            moving_images (list[itk.Image]): List of 3D images to register
            moving_images_masks (list[itk.Image], optional): List of binary masks,
                one for each moving image. If None, no masks are used. If provided,
                must have the same length as moving_images. Default: None
            starting_index (int, optional): Index of the first image to register.
                Registration proceeds forward from this index to the end, then
                backward from this index to the beginning. Default: 0
            register_start_to_fixed_image (bool, optional): If True, register the
                starting image to the fixed image. If False, use identity transform
                for the starting image. Default: True
            portion_of_prior_transform_to_init_next_transform (float, optional):
                Weight (0.0 to 1.0) for using the prior image's transform to
                initialize the next registration. 0.0 means no prior information
                is used (each registration starts from identity). Higher values
                provide more temporal smoothness but may propagate errors.
                Default: 0.0
            images_are_labelmaps (bool, optional): If True, treat images as label maps
                and use appropriate label-based registration. Default: False

        Returns:
            dict: Dictionary containing:
                - "phi_MF_list" (list[itk.Transform]): Transforms from moving to fixed
                  space for each image in moving_images
                - "phi_FM_list" (list[itk.Transform]): Transforms from fixed to moving
                  space for each image in moving_images
                - "losses" (list[float]): Registration loss value for each image

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If starting_index is out of range
            ValueError: If portion_of_prior_transform_to_init_next_transform not in [0, 1]
            ValueError: If moving_images_masks length doesn't match moving_images length

        Note:
            The method compares registration with identity initialization versus
            prior transform initialization and selects the result with lower loss.
            This helps prevent error propagation in the temporal sequence.

            The fixed image mask can be set using set_fixed_image_mask() before
            calling this method.

        Example:
            >>> registrar = RegisterTimeSeriesImages(registration_method='ants')
            >>> registrar.set_fixed_image(fixed_image)
            >>> registrar.set_fixed_image_mask(fixed_mask)  # Optional
            >>> registrar.set_number_of_iterations([30, 15, 5])
            >>>
            >>> result = registrar.register_time_series(
            ...     moving_images=image_list,
            ...     moving_images_masks=mask_list,  # Optional
            ...     starting_index=5,
            ...     register_start_to_fixed_image=True,
            ...     portion_of_prior_transform_to_init_next_transform=0.0
            ... )
            >>>
            >>> # Access results
            >>> for i, (phi_MF, loss) in enumerate(zip(
            ...     result["phi_MF_list"], result["losses"]
            ... )):
            ...     # Apply transform to image i
            ...     registered = transform_tools.transform_image(
            ...         moving_images[i], phi_MF, fixed_image
            ...     )
        """
        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before registering time series")

        if self.registration_method == 'ants':
            self.registrar_ants.set_fixed_image(self.fixed_image)
            self.registrar_ants.set_modality(self.modality)
            self.registrar_ants.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ants.set_number_of_iterations(self.number_of_iterations)
            self.registrar_ants.set_fixed_image_mask(self.fixed_image_mask)
        elif self.registration_method == 'icon':
            self.registrar_icon.set_fixed_image(self.fixed_image)
            self.registrar_icon.set_modality(self.modality)
            self.registrar_icon.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_icon.set_number_of_iterations(self.number_of_iterations)
            self.registrar_icon.set_fixed_image_mask(self.fixed_image_mask)
        elif self.registration_method == 'ants_icon':
            self.registrar_ants.set_fixed_image(self.fixed_image)
            self.registrar_ants.set_modality(self.modality)
            self.registrar_ants.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ants.set_number_of_iterations(self.number_of_iterations[0])
            self.registrar_ants.set_fixed_image_mask(self.fixed_image_mask)
            self.registrar_icon.set_fixed_image(self.fixed_image)
            self.registrar_icon.set_modality(self.modality)
            self.registrar_icon.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_icon.set_number_of_iterations(self.number_of_iterations[1])
            self.registrar_icon.set_fixed_image_mask(self.fixed_image_mask)

        num_images = len(moving_images)

        if starting_index < 0 or starting_index >= num_images:
            raise ValueError(
                f"starting_index {starting_index} out of range [0, {num_images-1}]"
            )

        if not 0.0 <= portion_of_prior_transform_to_init_next_transform <= 1.0:
            raise ValueError(
                "portion_of_prior_transform_to_init_next_transform must be in [0.0, 1.0]"
            )

        if moving_images_masks is not None and len(moving_images_masks) != num_images:
            raise ValueError(
                f"moving_images_masks length ({len(moving_images_masks)}) must match "
                f"moving_images length ({num_images})"
            )

        # Initialize result lists
        phi_MF_list = [None] * num_images
        phi_FM_list = [None] * num_images
        losses = [0.0] * num_images

        # Create identity transform for fixed image
        identity_tfm = itk.IdentityTransform[itk.D, 3].New()
        identity_tfm = (
            self.transform_tools.convert_transform_to_displacement_field_transform(
                identity_tfm, self.fixed_image
            )
        )

        # Register the starting image
        if register_start_to_fixed_image:
            starting_mask = (
                moving_images_masks[starting_index]
                if moving_images_masks is not None
                else None
            )
            if self.registration_method == 'ants':
                result = self.registrar_ants.register(
                    moving_images[starting_index],
                    moving_image_mask=starting_mask,
                    images_are_labelmaps=images_are_labelmaps,
                )
            elif self.registration_method == 'icon':
                result = self.registrar_icon.register(
                    moving_images[starting_index],
                    moving_image_mask=starting_mask,
                    images_are_labelmaps=images_are_labelmaps,
                )
            elif self.registration_method == 'ants_icon':
                result = self.registrar_ants.register(
                    moving_images[starting_index],
                    moving_image_mask=starting_mask,
                    images_are_labelmaps=images_are_labelmaps,
                )
                phi_MF_ants = result["phi_MF"]
                result = self.registrar_icon.register(
                    moving_images[starting_index],
                    moving_image_mask=starting_mask,
                    initial_phi_MF=phi_MF_ants,
                    images_are_labelmaps=images_are_labelmaps,
                )
            else:
                raise ValueError(
                    f"Invalid registration method: {self.registration_method}"
                )
            phi_MF = result["phi_MF"]
            phi_FM = result["phi_FM"]
            loss = result["loss"]
        else:
            # Use identity transform for starting image
            phi_MF = identity_tfm
            phi_FM = identity_tfm
            loss = 0.0

        phi_MF_list[starting_index] = phi_MF
        phi_FM_list[starting_index] = phi_FM
        losses[starting_index] = loss

        # Compute prior transform for starting image if needed
        prior_phi_MF_ref = None
        if portion_of_prior_transform_to_init_next_transform > 0.0:
            prior_phi_MF_ref = (
                self.transform_tools.combine_displacement_field_transforms(
                    identity_tfm,
                    phi_MF,
                    self.fixed_image,
                    tfm1_weight=1.0,
                    tfm2_weight=portion_of_prior_transform_to_init_next_transform,
                    tfm1_blur_sigma=0.0,
                    tfm2_blur_sigma=0.5,
                    mode="add",
                )
            )

        # Register forward and backward from starting index
        for step, start_idx, end_idx in [
            (1, starting_index + 1, num_images),  # Forward pass
            (-1, starting_index - 1, -1),  # Backward pass
        ]:
            prior_phi_MF = prior_phi_MF_ref

            for img_idx in range(start_idx, end_idx, step):
                moving_image = moving_images[img_idx]
                moving_mask = (
                    moving_images_masks[img_idx]
                    if moving_images_masks is not None
                    else None
                )

                # Try registration with identity initialization
                if self.registration_method == 'ants':
                    result_init_identity = self.registrar_ants.register(
                        moving_image=moving_image,
                        moving_image_mask=moving_mask,
                        images_are_labelmaps=images_are_labelmaps,
                    )
                elif self.registration_method == 'icon':
                    result_init_identity = self.registrar_icon.register(
                        moving_image=moving_image,
                        moving_image_mask=moving_mask,
                        images_are_labelmaps=images_are_labelmaps,
                    )
                elif self.registration_method == 'ants_icon':
                    result_init_identity = self.registrar_ants.register(
                        moving_image=moving_image,
                        moving_image_mask=moving_mask,
                        images_are_labelmaps=images_are_labelmaps,
                    )
                    phi_MF_ants = result_init_identity["phi_MF"]
                    result_init_identity = self.registrar_icon.register(
                        moving_image=moving_image,
                        moving_image_mask=moving_mask,
                        initial_phi_MF=phi_MF_ants,
                        images_are_labelmaps=images_are_labelmaps,
                    )
                else:
                    raise ValueError(
                        f"Invalid registration method: {self.registration_method}"
                    )
                phi_MF_init_identity = result_init_identity["phi_MF"]
                phi_FM_init_identity = result_init_identity["phi_FM"]
                loss_init_identity = result_init_identity["loss"]

                # Select best result based on prior usage
                if portion_of_prior_transform_to_init_next_transform > 0.0:
                    # Try with prior transform initialization
                    if self.registration_method == 'ants':
                        result_init_prior = self.registrar_ants.register(
                            moving_image=moving_image,
                            moving_image_mask=moving_mask,
                            initial_phi_MF=prior_phi_MF,
                            images_are_labelmaps=images_are_labelmaps,
                        )
                    elif self.registration_method == 'icon':
                        result_init_prior = self.registrar_icon.register(
                            moving_image=moving_image,
                            moving_image_mask=moving_mask,
                            initial_phi_MF=prior_phi_MF,
                            images_are_labelmaps=images_are_labelmaps,
                        )
                    elif self.registration_method == 'ants_icon':
                        result_init_prior = self.registrar_ants.register(
                            moving_image=moving_image,
                            moving_image_mask=moving_mask,
                            initial_phi_MF=prior_phi_MF,
                            images_are_labelmaps=images_are_labelmaps,
                        )
                        phi_MF_ants = result_init_prior["phi_MF"]
                        result_init_prior = self.registrar_icon.register(
                            moving_image=moving_image,
                            moving_image_mask=moving_mask,
                            initial_phi_MF=phi_MF_ants,
                            images_are_labelmaps=images_are_labelmaps,
                        )
                    else:
                        raise ValueError(
                            f"Invalid registration method: {self.registration_method}"
                        )
                    phi_MF_init_prior = result_init_prior["phi_MF"]
                    phi_FM_init_prior = result_init_prior["phi_FM"]
                    loss_init_prior = result_init_prior["loss"]

                    # Select result with lower loss
                    if loss_init_identity < loss_init_prior:
                        # Identity initialization was better
                        prior_phi_MF = identity_tfm
                        phi_MF = phi_MF_init_identity
                        phi_FM = phi_FM_init_identity
                        loss = loss_init_identity
                    else:
                        # Prior initialization was better
                        phi_MF = phi_MF_init_prior
                        phi_FM = phi_FM_init_prior
                        loss = loss_init_prior

                    # Update prior for next iteration
                    prior_phi_MF = self.transform_tools.combine_displacement_field_transforms(
                        identity_tfm,
                        phi_MF,
                        self.fixed_image,
                        tfm1_weight=1.0,
                        tfm2_weight=portion_of_prior_transform_to_init_next_transform,
                        tfm1_blur_sigma=0.0,
                        tfm2_blur_sigma=self.smooth_prior_transform_sigma,
                        mode="add",
                    )
                else:
                    # No prior usage, just use identity result
                    phi_MF = phi_MF_init_identity
                    phi_FM = phi_FM_init_identity
                    loss = loss_init_identity

                # Store results
                phi_MF_list[img_idx] = phi_MF
                phi_FM_list[img_idx] = phi_FM
                losses[img_idx] = loss

        return {
            "phi_MF_list": phi_MF_list,
            "phi_FM_list": phi_FM_list,
            "losses": losses,
        }

    def registration_method(
        self,
        moving_image,
        moving_image_mask=None,
        moving_image_pre=None,
        images_are_labelmaps=False,
        initial_phi_MF=None,
    ):
        """Registration method required by RegisterImagesBase.

        This method is not typically called directly. Use register_time_series()
        instead for time series registration.

        Args:
            moving_image (itk.Image): Image to register
            moving_image_mask (itk.Image, optional): Binary mask
            moving_image_pre (itk.Image, optional): Preprocessed image
            images_are_labelmaps (bool, optional): Whether to use label-based registration
            initial_phi_MF (itk.Transform, optional): Initial transform

        Returns:
            dict: Registration result with phi_FM, phi_MF, and loss
        """
        if self.registration_method == 'ants':
            return self.registrar_ants.registration_method(
                moving_image=moving_image,
                moving_image_mask=moving_image_mask,
                moving_image_pre=moving_image_pre,
                images_are_labelmaps=images_are_labelmaps,
                initial_phi_MF=initial_phi_MF,
            )
        elif self.registration_method == 'icon':
            return self.registrar_icon.registration_method(
                moving_image=moving_image,
                moving_image_mask=moving_image_mask,
                moving_image_pre=moving_image_pre,
                images_are_labelmaps=images_are_labelmaps,
                initial_phi_MF=initial_phi_MF,
            )
        elif self.registration_method == 'ants_icon':
            phi_MF_ants = self.registrar_ants.registration_method(
                moving_image=moving_image,
                moving_image_mask=moving_image_mask,
                moving_image_pre=moving_image_pre,
                images_are_labelmaps=images_are_labelmaps,
            )["phi_MF"]
            return self.registrar_icon.registration_method(
                moving_image=moving_image,
                moving_image_mask=moving_image_mask,
                moving_image_pre=moving_image_pre,
                images_are_labelmaps=images_are_labelmaps,
                initial_phi_MF=phi_MF_ants,
            )
        else:
            raise ValueError(f"Invalid registration method: {self.registration_method}")
