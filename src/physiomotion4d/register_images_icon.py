"""Icon-based image registration implementation.

This module provides the RegisterImagesIcon class, a concrete implementation of
RegisterImagesBase that uses the Icon (Inverse Consistent Image Registration)
algorithm with deep learning models. It supports both masked and unmasked
registration for aligning medical images, particularly useful for 4D cardiac CT registration.

The module uses the unigradicon package which provides GPU-accelerated
deformable registration with mass preservation constraints.
"""

import argparse

import icon_registration as icon
import icon_registration.itk_wrapper
import itk
from unigradicon import get_multigradicon, get_unigradicon
from unigradicon import preprocess as unigradicon_preprocess

from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.transform_tools import TransformTools


class RegisterImagesICON(RegisterImagesBase):
    """ICON-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide GPU-accelerated deformable
    image registration using the ICON (Inverse Consistent Image Registration)
    algorithm implemented with deep learning models. It supports both full image
    registration and mask-constrained registration for specific anatomical regions.

    The ICON algorithm ensures inverse consistency, meaning the forward and
    backward transformations are true inverses of each other. This is important
    for maintaining spatial relationships and avoiding registration artifacts.

    ICON-specific features:
    - GPU acceleration using UniGradIcon framework
    - Mass preservation
    - LNCC (Local Normalized Cross Correlation) similarity metric
    - Inverse consistent transformations
    - Fine-tuning with 50 optimization steps per registration

    Inherits from RegisterImagesBase:
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        net (unigradicon model): The ICON deep learning registration network

    Example:
        >>> registrar = RegisterImagesICON()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> phi_FM = result["phi_FM"]
    """

    def __init__(self):
        """Initialize the ICON image registration class.

        Calls the parent RegisterImagesBase constructor to set up common parameters.
        The ICON deep learning network is initialized lazily on first use to avoid
        unnecessary GPU memory allocation.
        """
        super().__init__()

        self.net = None
        self.use_multi_modality = False
        self.use_mass_preservation = False

    def set_multi_modality(self, enable):
        """Enable or disable multi-modality registration.

        Multi-modality registration is useful when aligning images from different
        imaging modalities (e.g., CT to MRI). Enabling this option adjusts the
        registration parameters to better handle differences in intensity
        distributions and contrast between modalities.

        Args:
            enable (bool): True to enable multi-modality registration, False to disable

        Example:
            >>> registrar.set_multi_modality(True)  # Enable for CT to MRI
            >>> registrar.set_multi_modality(False) # Disable for CT to CT
        """
        self.use_multi_modality = enable

    def set_mass_preservation(self, enable):
        """Enable or disable mass preservation constraint.

        Mass preservation is particularly useful for CT images where the
        intensity values correspond to physical tissue densities. Enabling
        this constraint helps maintain realistic intensity distributions
        during registration.

        Args:
            enable (bool): True to enable mass preservation, False to disable

        Example:
            >>> registrar.set_mass_preservation(True)  # Enable for CT
            >>> registrar.set_mass_preservation(False) # Disable for MRI
        """
        self.use_mass_preservation = enable

    def preprocess(self, image, modality):
        """Preprocess the image for ICON registration.

        Applies modality-specific preprocessing steps to prepare the image
        for registration. This may include intensity normalization, bias
        field correction, and resampling.

        Args:
            image (itk.image): The input 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        # Placeholder implementation - override in subclass if needed
        return unigradicon_preprocess(image, modality=modality)

    def registration_method(
        self,
        moving_image,
        moving_image_mask=None,
        moving_image_pre=None,
        images_are_labelmaps=False,
        initial_phi_MF=None,
    ):
        """Register moving image to fixed image using ICON registration algorithm.

        Implementation of the abstract register() method from RegisterImagesBase.
        Performs deformable registration to align the moving image with the
        fixed image using the ICON algorithm. The method automatically handles
        preprocessing, network initialization, and applies the computed transformation.

        Args:
            moving_image (itk.image): The 3D image to be registered/aligned
            moving_image_mask (itk.image, optional): Binary mask defining the
                region of interest in the moving image. If provided along with
                fixed_image_mask, enables mask-constrained registration
            moving_image_pre (itk.image, optional): Pre-processed moving image.
                If None, preprocessing is performed automatically
            images_are_labelmaps (bool, optional): Whether the images are labelmaps.
                Currently not used by ICON implementation
            initial_phi_MF (itk.Transform, optional): Initial transformation from fixed
                to moving. If provided, it is used to transform the moving image before
                registration.

        Returns:
            dict: Dictionary containing:
                - "phi_FM": transform fixed image to moving space
                    - transform goes from moving space into fixed space
                - "phi_MF": transform moving image into fixed space
                    - transform goes from fixed space into moving space
                - "loss": Loss value from the registration

        Note:
            The transformations phi_FM and phi_MF
            are inverse consistent, meaning phi_MF ≈ inverse(
            phi_FM).
            The phi_FM transform is used to warp the fixed image
            to the moving image space. The phi_MF transform is used
            to warp the moving image to the fixed image space.

        Implementation details:
            - Uses UniGradIcon with LNCC loss function
            - Optionally applies mass preservation
            - Performs 50 fine-tuning steps per registration
            - Supports both masked and unmasked registration modes

        Example:
            >>> # Basic registration
            >>> result = registrar.register(moving_image)
            >>> phi_FM = result["phi_FM"]
            >>> phi_MF = result["phi_MF"]
            >>>
            >>> # Masked registration for cardiac structures
            >>> registrar.set_fixed_image_mask(heart_mask_fixed)
            >>> result = registrar.register(
            ...     moving_image, moving_image_mask=heart_mask_moving
            ... )
        """

        tfm_tools = TransformTools()

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(moving_image, self.modality)

        new_moving_image_pre = moving_image_pre
        if initial_phi_MF is not None:
            new_moving_image_pre = tfm_tools.transform_image(
                moving_image_pre,
                initial_phi_MF,
                self.fixed_image,
            )

        if self.net is None:
            if self.use_multi_modality:
                self.net = get_multigradicon(
                    loss_fn=icon.LNCC(sigma=5),
                    # loss_fn=icon.losses.MINDSSC(radius=2, dilation=2),
                    apply_intensity_conservation_loss=self.use_mass_preservation,
                )
            else:
                self.net = get_unigradicon(
                    loss_fn=icon.LNCC(sigma=5),
                    apply_intensity_conservation_loss=self.use_mass_preservation,
                )

        phi_FM = None
        phi_MF = None
        loss_artifacts = None
        if self.fixed_image_mask is not None and moving_image_mask is not None:
            phi_FM, phi_MF, loss_artifacts = (
                icon_registration.itk_wrapper.register_pair_with_mask(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    self.fixed_image_mask,
                    moving_image_mask,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )
        else:
            phi_FM, phi_MF, loss_artifacts = (
                icon_registration.itk_wrapper.register_pair(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )

        loss = loss_artifacts[0]

        if initial_phi_MF is not None:
            phi_MF = tfm_tools.combine_displacement_field_transforms(
                initial_phi_MF,
                phi_MF,
                self.fixed_image,
                tfm1_weight=1.0,
                tfm2_weight=1.0,
                mode="compose",
            )

            dftfm = tfm_tools.convert_transform_to_displacement_field_transform(
                phi_MF,
                self.fixed_image,
            )
            phi_FM = tfm_tools.invert_displacement_field_transform(dftfm)

        return {
            "phi_FM": phi_FM,
            "phi_MF": phi_MF,
            "loss": loss,
        }


def parse_args():
    """Parse command line arguments for image registration.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - fixed_image: Path to fixed/reference image
            - moving_image: Path to moving image to register
            - output_image: Path for registered output image
            - modality: Image modality (e.g., 'ct', 'mri')
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_image", type=str, required=True)
    parser.add_argument("--moving_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """Command line interface for ICON-based image registration.

    Example usage:
        python register_images_icon.py \
            --fixed_image reference.mha \
            --moving_image timepoint_05.mha \
            --output_image registered.mha \
            --modality ct
    """
    args = parse_args()
    registrar = RegisterImagesICON()
    registrar.set_modality(args.modality)
    registrar.set_fixed_image(itk.imread(args.fixed_image))
    moving_image = itk.imread(args.moving_image)
    result = registrar.register(moving_image=moving_image)
    phi_FM = result["phi_FM"]
    phi_MF = result["phi_MF"]
    moving_image_reg = TransformTools().transform_image(
        moving_image, phi_MF, registrar.fixed_image, "sinc"
    )  # Final resampling with sinc
    itk.imwrite(moving_image_reg, args.output_image, compression=True)
