"""Base class for image registration algorithms.

This module provides the RegisterImagesBase class that serves as a foundation
for implementing different image registration algorithms. It defines a common
interface and shared functionality for deformable image registration, particularly
designed for medical imaging applications such as 4D cardiac CT registration.

The base class handles common operations including:
- Fixed and moving image management
- Binary mask processing and dilation
- Modality-specific parameter settings
- Standardized registration interface

Concrete implementations should inherit from RegisterImagesBase and implement
the register() method with their specific algorithm (e.g., Icon, ANTs, etc.).
"""

import itk
import numpy as np
from itk import TubeTK as ttk

from physiomotion4d.transform_tools import TransformTools


class RegisterImagesBase:
    """Base class for deformable image registration algorithms.

    This class provides a common interface and shared functionality for
    implementing different image registration algorithms. It handles standard
    operations like image and mask management, preprocessing, and parameter
    configuration that are common across registration methods.

    The base class is designed to support various registration algorithms
    including deep learning-based methods (Icon, UniGradIcon) and traditional
    methods (ANTs, ITK). Concrete implementations should inherit from this
    class and implement the register() method.

    Key features:
    - Standardized interface for different registration algorithms
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Support for region-of-interest registration

    Attributes:
        net (object): Algorithm-specific network or registration object
        modality (str): Image modality ('ct', 'mri', etc.) for parameter optimization
        fixed_image (itk.image): The target/reference image
        fixed_image_pre (itk.image): Preprocessed fixed image
        fixed_image_mask (itk.image): Binary mask for fixed image ROI
        mask_dilation_mm (float): Mask dilation amount in millimeters

    Example:
        >>> class MyRegistration(RegisterImagesBase):
        ...     def registration_method(self, moving_image, **kwargs):
        ...         # Implement specific registration algorithm
        ...         return {"phi_FM": tfm_forward, "phi_MF": tfm_backward}
        >>>
        >>> registrar = MyRegistration()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> phi_FM = result["phi_FM"]
        >>> phi_MF = result["phi_MF"]
    """

    def __init__(self):
        """Initialize the base image registration class.

        Sets up the common registration parameters with default values. Algorithm-specific
        components (like neural networks or optimization objects) should be initialized
        in the concrete implementation to avoid unnecessary resource allocation.
        """
        self.net = None

        self.modality = 'ct'

        self.fixed_image = None
        self.fixed_image_pre = None
        self.fixed_image_mask = None

        self.moving_image = None
        self.moving_image_pre = None
        self.moving_image_mask = None

        self.mask_dilation_mm = 5

        self.number_of_iterations = 10

    def set_number_of_iterations(self, number_of_iterations):
        """Set the number of iterations for registration.

        Args:
            number_of_iterations (int): Number of iterations to perform during registration.
        """
        self.number_of_iterations = number_of_iterations

    def set_modality(self, modality):
        """Set the imaging modality for registration optimization.

        Different imaging modalities benefit from different registration
        parameters. CT images.

        Args:
            modality (str): The imaging modality.
                Supported values: 'ct', 'mri'

        Example:
            >>> registrar.set_modality('ct')
            >>> registrar.set_modality('mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image):
        """Set the fixed/target image for registration.

        The fixed image serves as the reference coordinate system to which
        all moving images will be aligned. Setting a new fixed image clears
        any preprocessed data to ensure consistency.

        Args:
            fixed_image (itk.image): The 3D reference image that serves as
                the target for registration

        Example:
            >>> registrar.set_fixed_image(reference_frame)
        """
        self.fixed_image = fixed_image
        self.fixed_image_pre = None

    def set_mask_dilation(self, mask_dilation_mm):
        """Set the dilation of the fixed and moving image masks.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_fixed_image_mask(self, fixed_image_mask):
        """Set a binary mask for the fixed image region of interest.

        The mask constrains registration to focus on specific anatomical
        regions, improving accuracy and reducing computation time. The mask
        is automatically converted to binary format. If mask_dilation_mm is set,
        the mask is dilated by the specified amount.

        Args:
            fixed_image_mask (itk.image): Binary or label mask defining the
                region of interest in the fixed image. Non-zero values are
                treated as foreground

        Example:
            >>> # Use heart mask to focus registration on cardiac structures
            >>> registrar.set_fixed_image_mask(heart_mask)
        """
        self.fixed_image_pre = None

        if fixed_image_mask is None:
            self.fixed_image_mask = None
            return

        mask_arr = itk.GetArrayFromImage(fixed_image_mask)
        mask_arr = np.where(mask_arr > 0, 1, 0)
        self.fixed_image_mask = itk.GetImageFromArray(mask_arr.astype(np.uint8))
        self.fixed_image_mask.CopyInformation(self.fixed_image)
        if self.mask_dilation_mm > 0:
            imMath = ttk.ImageMath.New(self.fixed_image_mask)
            imMath.Dilate(
                int(self.fixed_image.GetSpacing()[0] / self.mask_dilation_mm), 1, 0
            )
            self.fixed_image_mask = imMath.GetOutputUChar()

    def preprocess(self, image, modality='ct'):
        """Preprocess the image based on modality-specific requirements.

        This method applies preprocessing steps such as intensity normalization,
        histogram equalization, or noise reduction tailored to the specified
        imaging modality. Preprocessing enhances image quality and improves
        registration accuracy.

        Args:
            image (itk.image): The 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        # Placeholder implementation - override in subclass if needed
        return image

    def registration_method(
        self,
        moving_image,
        moving_image_mask=None,
        moving_image_pre=None,
        images_are_labelmaps=False,
        initial_phi_MF=None,
    ) -> dict:
        """Main registration method to align moving image to fixed image.

        This method serves as the primary interface for performing image
        registration. It takes a moving image and optional mask and
        preprocessed image, and returns the registered image along with
        forward and backward transformations.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_image_mask (itk.image, optional): Binary mask for moving image ROI
            moving_image_pre (itk.image, optional): Preprocessed moving image
            images_are_labelmaps (bool, optional): Whether the images are labelmaps
            initial_phi_MF (itk.Transform, optional): Initial transformation from moving to fixed
        Returns:
            dict: Dictionary containing:
                - "phi_FM": Used to warp fixed image into moving space
                - "phi_MF": Used to warp moving image into fixed space
        Raises:
            ValueError: If fixed image is not set
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def register(
        self,
        moving_image,
        moving_image_mask=None,
        moving_image_pre=None,
        images_are_labelmaps=False,
        initial_phi_MF=None,
    ) -> dict:
        """Register a moving image to the fixed image.

        This is the main registration method that must be implemented by
        concrete subclasses. It should align the moving image to the fixed
        image using the specific algorithm implemented by the subclass.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_image_mask (itk.image, optional): Binary mask for moving image ROI
            moving_image_pre (itk.image, optional): Preprocessed moving image
            images_are_labelmaps (bool, optional): Whether the images are labelmaps for label-based registration
            initial_phi_MF (itk.Transform, optional): Initial transformation from fixed to moving

        Returns:
            dict: Dictionary containing:
                - "phi_FM": Forward transformation from moving to fixed
                - "phi_MF": Backward transformation from fixed to moving

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        if self.fixed_image_pre is None:
            self.fixed_image_pre = self.preprocess(
                self.fixed_image,
                modality=self.modality,
            )

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(
                moving_image,
                modality=self.modality,
            )

        new_moving_image_mask = moving_image_mask
        if moving_image_mask is not None:
            mask_arr = itk.GetArrayFromImage(moving_image_mask)
            mask_arr = np.where(mask_arr > 0, 1, 0)
            new_moving_image_mask = itk.GetImageFromArray(mask_arr.astype(np.uint8))
            new_moving_image_mask.CopyInformation(moving_image)
            if self.mask_dilation_mm > 0:
                imMath = ttk.ImageMath.New(new_moving_image_mask)
                imMath.Dilate(
                    int(moving_image.GetSpacing()[0] / self.mask_dilation_mm), 1, 0
                )
                new_moving_image_mask = imMath.GetOutputUChar()

        self.moving_image = moving_image
        self.moving_image_pre = moving_image_pre
        self.moving_image_mask = new_moving_image_mask

        result = self.registration_method(
            moving_image,
            moving_image_mask=new_moving_image_mask,
            moving_image_pre=moving_image_pre,
            images_are_labelmaps=images_are_labelmaps,
            initial_phi_MF=initial_phi_MF,
        )

        phi_FM = result["phi_FM"]
        phi_MF = result["phi_MF"]
        loss = result["loss"]

        return {
            "phi_FM": phi_FM,
            "phi_MF": phi_MF,
            "loss": loss,
        }
