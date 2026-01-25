"""Time series image registration implementation.

This module provides the RegisterTimeSeriesImages class for registering an ordered
sequence of images (time series) to a fixed image. It supports both ANTs
and ICON registration methods and can optionally use prior transforms to initialize
subsequent registrations in the sequence.

The class is particularly useful for 4D medical imaging applications such as cardiac
CT where sequential frames need to be registered to a common frame.
"""

import logging
from typing import Optional, TypeAlias, Union, cast

import itk

from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools


class RegisterTimeSeriesImages(RegisterImagesBase):
    NumberOfIterations: TypeAlias = RegisterImagesBase.NumberOfIterations
    """Register a time series of images to a fixed image.

    This class extends RegisterImagesBase to provide sequential registration of
    multiple images (time series) to a fixed image. It supports both
    ANTs and ICON registration methods and can propagate information from prior
    registrations to initialize subsequent ones.

    The registration proceeds in two passes from a reference frame:
    1. Forward pass: from reference_frame to the end of the series
    2. Backward pass: from reference_frame-1 to the beginning

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
        ...     reference_frame=5,  # Start from middle of cardiac cycle
        ...     register_reference=True,
        ...     prior_weight=0.5,
        ... )
        >>>
        >>> forward_tfms = result['forward_transforms']  # Moving → Fixed
        >>> inverse_tfms = result['inverse_transforms']  # Fixed → Moving
        >>> losses = result['losses']
    """

    def __init__(
        self, registration_method: str = "ants", log_level: int | str = logging.INFO
    ) -> None:
        """Initialize the time series image registration class.

        Args:
            registration_method (str): Registration method to use.
                Options: 'ants' or 'icon'. Default: 'ants'
            log_level: Logging level (default: logging.INFO)

        Raises:
            ValueError: If registration_method is not 'ants' or 'icon'
        """
        super().__init__(log_level=log_level)

        self.registration_method_name: str = registration_method.lower()

        self.registrar_ants = RegisterImagesANTs(log_level=log_level)
        self.registrar_icon = RegisterImagesICON(log_level=log_level)
        if self.registration_method_name == "ants":
            self.number_of_iterations = [40, 20, 10]
        elif self.registration_method_name == "icon":
            self.number_of_iterations = 50
        elif self.registration_method_name == "ants_icon":
            iterations: list[int | list[int]] = [[40, 20, 10], 50]
            self.number_of_iterations = iterations
        else:
            raise ValueError(
                f"registration_method must be 'ants', 'icon' or 'ants_icon', got '{registration_method}'"
            )

        self.transform_tools: TransformTools = TransformTools()

        self.smooth_prior_transform_sigma: float = 0.5

    def set_number_of_iterations(
        self, number_of_iterations: NumberOfIterations
    ) -> None:
        """Set the number of iterations for registration.

        This passes through to the underlying registration method (ANTs or ICON).

        Args:
            number_of_iterations: Number of iterations to perform.
                For ANTs: list of int (e.g., [40, 20, 10] for multi-resolution)
                For ICON: int (number of fine-tuning steps)
        """
        self.number_of_iterations = number_of_iterations

    def set_smooth_prior_transform_sigma(
        self, smooth_prior_transform_sigma: float
    ) -> None:
        """Set the sigma for smoothing the prior transform.

        Args:
            smooth_prior_transform_sigma (float): Sigma for smoothing the prior transform.
        """
        self.smooth_prior_transform_sigma = smooth_prior_transform_sigma

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        This passes through to the underlying registration method.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        This passes through to the underlying registration method.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image: itk.Image) -> None:
        """Set the fixed image for registration.

        All moving images in the time series will be registered to this
        fixed image.

        Args:
            fixed_image (itk.Image): The 3D fixed image
        """
        self.fixed_image = fixed_image

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_mask (itk.Image): Binary mask defining ROI
        """
        self.fixed_mask = fixed_mask

    def register_time_series(
        self,
        moving_images: list[itk.Image],
        moving_masks: Optional[list[Optional[itk.Image]]] = None,
        reference_frame: int = 0,
        register_reference: bool = True,
        prior_weight: float = 0.0,
    ) -> dict[str, list[itk.Transform] | list[float]]:
        """Register a time series of images to the fixed image.

        This method registers an ordered sequence of images to a common fixed
        frame. Registration proceeds bidirectionally from a reference frame:
        forward to the end and backward to the beginning.

        For each image after the reference image, the method can optionally use
        the transform from the previous image to initialize the registration,
        which can improve convergence and temporal coherence.

        Args:
            moving_images (list[itk.Image]): List of 3D images to register
            moving_masks (list[itk.Image], optional): List of binary masks,
                one for each moving image. If None, no masks are used. If provided,
                must have the same length as moving_images. Default: None
            reference_frame (int, optional): Index of the reference image to register first.
                Registration proceeds forward from this index to the end, then
                backward from this index to the beginning. Default: 0
            register_reference (bool, optional): If True, register the
                reference image to the fixed image. If False, use identity transform
                for the reference image. Default: True
            prior_weight (float, optional):
                Weight (0.0 to 1.0) for using the prior image's transform to
                initialize the next registration. 0.0 means no prior information
                is used (each registration starts from identity). Higher values
                provide more temporal smoothness but may propagate errors.
                Default: 0.0

        Returns:
            dict: Dictionary containing results:
                - "forward_transforms" (list[itk.Transform]): Transforms from moving to fixed
                  space for each image (warps moving → fixed)
                - "inverse_transforms" (list[itk.Transform]): Transforms from fixed to moving
                  space for each image (warps fixed → moving)
                - "losses" (list[float]): Registration loss value for each image

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If reference_frame is out of range
            ValueError: If prior_weight not in [0, 1]
            ValueError: If moving_masks length doesn't match moving_images length

        Note:
            The method compares registration with identity initialization versus
            prior transform initialization and selects the result with lower loss.
            This helps prevent error propagation in the temporal sequence.

            The fixed image mask can be set using set_fixed_mask() before
            calling this method.

        Example:
            >>> registrar = RegisterTimeSeriesImages(registration_method='ants')
            >>> registrar.set_fixed_image(fixed_image)
            >>> registrar.set_fixed_mask(fixed_mask)  # Optional
            >>> registrar.set_number_of_iterations([30, 15, 5])
            >>>
            >>> # Use new intuitive parameter names
            >>> result = registrar.register_time_series(
            ...     moving_images=image_list,
            ...     moving_masks=mask_list,  # Optional
            ...     reference_frame=5,
            ...     register_reference=True,
            ...     prior_weight=0.5,
            ... )
            >>>
            >>> # Access results using new intuitive names
            >>> for i, (forward_tfm, loss) in enumerate(
            ...     zip(result['forward_transforms'], result['losses'])
            ... ):
            ...     # Apply forward transform to align moving image i to fixed
            ...     registered = transform_tools.transform_image(
            ...         moving_images[i], forward_tfm, fixed_image
            ...     )
        """
        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before registering time series")

        if self.registration_method_name == "ants":
            self.registrar_ants.set_fixed_image(self.fixed_image)
            self.registrar_ants.set_modality(self.modality)
            self.registrar_ants.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ants.set_number_of_iterations(self.number_of_iterations)
            self.registrar_ants.set_fixed_mask(self.fixed_mask)
        elif self.registration_method_name == "icon":
            self.registrar_icon.set_fixed_image(self.fixed_image)
            self.registrar_icon.set_modality(self.modality)
            self.registrar_icon.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_icon.set_number_of_iterations(self.number_of_iterations)
            self.registrar_icon.set_fixed_mask(self.fixed_mask)
        elif self.registration_method_name == "ants_icon":
            self.registrar_ants.set_fixed_image(self.fixed_image)
            self.registrar_ants.set_modality(self.modality)
            self.registrar_ants.set_mask_dilation(self.mask_dilation_mm)
            assert isinstance(self.number_of_iterations, list)
            ants_iterations = self.number_of_iterations[0]
            icon_iterations = self.number_of_iterations[1]
            assert isinstance(ants_iterations, list)
            assert isinstance(icon_iterations, int)
            self.registrar_ants.set_number_of_iterations(ants_iterations)
            self.registrar_ants.set_fixed_mask(self.fixed_mask)
            self.registrar_icon.set_fixed_image(self.fixed_image)
            self.registrar_icon.set_modality(self.modality)
            self.registrar_icon.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_icon.set_number_of_iterations(icon_iterations)
            self.registrar_icon.set_fixed_mask(self.fixed_mask)

        num_images = len(moving_images)

        if reference_frame < 0 or reference_frame >= num_images:
            raise ValueError(
                f"reference_frame {reference_frame} out of range [0, {num_images - 1}]"
            )

        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("prior_weight must be in [0.0, 1.0]")

        if moving_masks is not None and len(moving_masks) != num_images:
            raise ValueError(
                f"moving_masks length ({len(moving_masks)}) must match "
                f"moving_images length ({num_images})"
            )

        # Initialize result lists
        forward_transforms: list[Optional[itk.Transform]] = [None] * num_images
        inverse_transforms: list[Optional[itk.Transform]] = [None] * num_images
        losses = [0.0] * num_images

        # Create identity transform for fixed image
        identity_tfm = itk.IdentityTransform[itk.D, 3].New()
        identity_tfm = (
            self.transform_tools.convert_transform_to_displacement_field_transform(
                identity_tfm, self.fixed_image
            )
        )

        # Register the reference frame image
        if register_reference:
            reference_mask = (
                moving_masks[reference_frame] if moving_masks is not None else None
            )
            if self.registration_method_name == "ants":
                result = self.registrar_ants.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                )
            elif self.registration_method_name == "icon":
                result = self.registrar_icon.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                )
            elif self.registration_method_name == "ants_icon":
                result = self.registrar_ants.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                )
                forward_ants = result["forward_transform"]
                result = self.registrar_icon.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    initial_forward_transform=forward_ants,
                )
            else:
                raise ValueError(
                    f"Invalid registration method: {self.registration_method_name}"
                )
            forward_transform = result["forward_transform"]
            inverse_transform = result["inverse_transform"]
            loss = result["loss"]
        else:
            # Use identity transform for reference frame
            forward_transform = identity_tfm
            inverse_transform = identity_tfm
            loss = 0.0

        forward_transforms[reference_frame] = forward_transform
        inverse_transforms[reference_frame] = inverse_transform
        losses[reference_frame] = loss

        # Compute prior transform for reference frame if needed
        prior_forward_ref = None
        if prior_weight > 0.0:
            prior_forward_ref = (
                self.transform_tools.combine_displacement_field_transforms(
                    identity_tfm,
                    forward_transform,
                    self.fixed_image,
                    tfm1_weight=1.0,
                    tfm2_weight=prior_weight,
                    tfm1_blur_sigma=0.0,
                    tfm2_blur_sigma=0.5,
                    mode="add",
                )
            )

        # Register forward and backward from reference frame
        for step, start_idx, end_idx in [
            (1, reference_frame + 1, num_images),  # Forward pass
            (-1, reference_frame - 1, -1),  # Backward pass
        ]:
            prior_forward = prior_forward_ref

            for img_idx in range(start_idx, end_idx, step):
                moving_image = moving_images[img_idx]
                moving_mask = (
                    moving_masks[img_idx] if moving_masks is not None else None
                )

                # Try registration with identity initialization
                if self.registration_method_name == "ants":
                    result_init_identity = self.registrar_ants.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                    )
                elif self.registration_method_name == "icon":
                    result_init_identity = self.registrar_icon.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                    )
                elif self.registration_method_name == "ants_icon":
                    result_init_identity = self.registrar_ants.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                    )
                    forward_ants = result_init_identity["forward_transform"]
                    result_init_identity = self.registrar_icon.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        initial_forward_transform=forward_ants,
                    )
                else:
                    raise ValueError(
                        f"Invalid registration method: {self.registration_method_name}"
                    )
                forward_init_identity = result_init_identity["forward_transform"]
                inverse_init_identity = result_init_identity["inverse_transform"]
                loss_init_identity = result_init_identity["loss"]

                # Select best result based on prior usage
                if prior_weight > 0.0:
                    # Try with prior transform initialization
                    if self.registration_method_name == "ants":
                        result_init_prior = self.registrar_ants.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            initial_forward_transform=prior_forward,
                        )
                    elif self.registration_method_name == "icon":
                        result_init_prior = self.registrar_icon.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            initial_forward_transform=prior_forward,
                        )
                    elif self.registration_method_name == "ants_icon":
                        result_init_prior = self.registrar_ants.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            initial_forward_transform=prior_forward,
                        )
                        forward_ants = result_init_prior["forward_transform"]
                        result_init_prior = self.registrar_icon.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            initial_forward_transform=forward_ants,
                        )
                    else:
                        raise ValueError(
                            f"Invalid registration method: {self.registration_method_name}"
                        )
                    forward_init_prior = result_init_prior["forward_transform"]
                    inverse_init_prior = result_init_prior["inverse_transform"]
                    loss_init_prior = result_init_prior["loss"]

                    # Select result with lower loss
                    if loss_init_identity < loss_init_prior:
                        # Identity initialization was better
                        prior_forward = identity_tfm
                        forward_transform = forward_init_identity
                        inverse_transform = inverse_init_identity
                        loss = loss_init_identity
                    else:
                        # Prior initialization was better
                        forward_transform = forward_init_prior
                        inverse_transform = inverse_init_prior
                        loss = loss_init_prior

                    # Update prior for next iteration
                    prior_forward = (
                        self.transform_tools.combine_displacement_field_transforms(
                            identity_tfm,
                            forward_transform,
                            self.fixed_image,
                            tfm1_weight=1.0,
                            tfm2_weight=prior_weight,
                            tfm1_blur_sigma=0.0,
                            tfm2_blur_sigma=self.smooth_prior_transform_sigma,
                            mode="add",
                        )
                    )
                else:
                    # No prior usage, just use identity result
                    forward_transform = forward_init_identity
                    inverse_transform = inverse_init_identity
                    loss = loss_init_identity

                # Store results
                forward_transforms[img_idx] = forward_transform
                inverse_transforms[img_idx] = inverse_transform
                losses[img_idx] = loss

        assert all(t is not None for t in forward_transforms)
        assert all(t is not None for t in inverse_transforms)
        return {
            "forward_transforms": [t for t in forward_transforms if t is not None],
            "inverse_transforms": [t for t in inverse_transforms if t is not None],
            "losses": losses,
        }

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Registration method required by RegisterImagesBase.

        This method is not typically called directly. Use register_time_series()
        instead for time series registration.

        Args:
            moving_image (itk.Image): Image to register
            moving_mask (itk.Image, optional): Binary mask
            moving_image_pre (itk.Image, optional): Preprocessed image
            initial_forward_transform (itk.Transform, optional): Initial transform

        Returns:
            dict: Registration result with forward_transform, inverse_transform, and loss
        """
        if self.registration_method_name == "ants":
            res = self.registrar_ants.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            return {
                "forward_transform": cast(itk.Transform, res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, res["inverse_transform"]),
                "loss": float(cast(float, res["loss"])),
            }
        if self.registration_method_name == "icon":
            res = self.registrar_icon.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            return {
                "forward_transform": cast(itk.Transform, res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, res["inverse_transform"]),
                "loss": float(cast(float, res["loss"])),
            }
        if self.registration_method_name == "ants_icon":
            ants_res = self.registrar_ants.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_image_pre=moving_image_pre,
            )
            forward_ants = ants_res["forward_transform"]
            icon_res = self.registrar_icon.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=forward_ants,
            )
            return {
                "forward_transform": cast(itk.Transform, icon_res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, icon_res["inverse_transform"]),
                "loss": float(cast(float, icon_res["loss"])),
            }
        raise ValueError(
            f"Invalid registration method: {self.registration_method_name}"
        )
