"""ANTs-based image registration implementation.

This module provides the RegisterImagesANTs class, a concrete implementation of
RegisterImagesBase that uses the Advanced Normalization Tools (ANTs) algorithm
for image registration. It supports both affine and deformable (SyN) registration
for aligning medical images, particularly useful for 4D cardiac CT registration.

The module uses the antspyx package which provides Python bindings to the ANTs
C++ library, offering robust and well-established registration algorithms.
"""

import argparse

import ants
import itk
import numpy as np
from itk import TubeTK as ttk

from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.transform_tools import TransformTools


class RegisterImagesANTs(RegisterImagesBase):
    """ANTs-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide deformable image registration
    using the Advanced Normalization Tools (ANTs) algorithm. It supports various
    registration types including affine, deformable (SyN), and elastic registration
    for aligning medical images.

    ANTs is a well-established image registration framework with proven accuracy
    and robustness for medical imaging applications. The SyN (Symmetric
    Normalization) algorithm provides diffeomorphic registration with inverse
    consistency.

    ANTs-specific features:
    - Multiple transform types: Rigid, Affine, SyN, ElasticSyN
    - Robust optimization algorithms
    - Support for multi-resolution registration
    - Symmetric normalization for inverse consistency
    - Comprehensive metric options (MI, CC, Mattes)

    Inherits from RegisterImagesBase:
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        type_of_transform (str): The registration transform type (default: 'SyN')
        grad_step (float): Gradient step size for SyN (default: 0.2)
        flow_sigma (float): Smoothing parameter for regularization (default: 3.0)
        total_sigma (float): Total field smoothing (default: 0.0)
        syn_metric (str): Similarity metric for SyN (default: 'CC')
        syn_sampling (int): Sampling strategy (default: 2)
        reg_iterations (tuple): Iterations per resolution level (default: (40, 20, 0))

    Example:
        >>> registrar = RegisterImagesANTs()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> phi_FM = result["phi_FM"]
    """

    def __init__(self):
        """Initialize the ANTs image registration class.

        Calls the parent RegisterImagesBase constructor to set up common parameters.
        Default ANTs registration parameters are set to work well for medical images.
        """
        super().__init__()

        self.number_of_iterations = [40, 20, 10]

    def _ants_to_itk_image(self, ants_image):
        """Convert ANTs image back to ITK format.

        Args:
            ants_image (ants.core.ANTsImage): ANTs image to convert
            reference_itk_image (itk.image): Reference ITK image for metadata

        Returns:
            itk.image: Converted ITK image
        """
        data = ants_image.numpy()

        image_dimension = ants_image.dimension
        if image_dimension not in (2, 3, 4):
            raise ValueError(f"Unsupported ANTs image dimension: {image_dimension}")

        is_vector = ants_image.components > 1

        data_reshaped = None
        if is_vector:
            # Vector images: ANTs gives (components, z, y, x) or (components, y, x)
            data_reshaped = data.transpose(
                list(range(image_dimension - 1, -1, -1)) + [image_dimension]
            ).astype(np.float64)
        else:
            data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)))

        img_itk = None
        if is_vector:
            img_itk = itk.GetImageFromArray(data_reshaped, is_vector=True)
        else:
            img_itk = itk.GetImageFromArray(data_reshaped)

        spacing = ants_image.spacing
        origin = ants_image.origin
        direction_reshaped = np.asarray(ants_image.direction).reshape(
            (image_dimension, image_dimension)
        )

        img_itk.SetSpacing(spacing)
        img_itk.SetOrigin(origin)
        img_itk.SetDirection(direction_reshaped)

        return img_itk

    def _itk_to_ants_image(self, itk_image, dtype='float'):
        """Convert ITK image to ANTs format.

        Args:
            itk_image (itk.image): ITK image to convert

        Returns:
            ants.core.ANTsImage: Converted ANTs image
        """
        ndim = itk_image.GetImageDimension()
        if ndim not in (2, 3, 4):
            raise ValueError(f"Unsupported ITK image dimension: {ndim}")

        is_vector = itk_image.GetNumberOfComponentsPerPixel() > 1

        if dtype == 'float':
            data = itk.GetArrayFromImage(itk_image).astype(np.float32)
        elif dtype == 'double':
            data = itk.GetArrayFromImage(itk_image).astype(np.float64)
        elif dtype == 'int':
            data = itk.GetArrayFromImage(itk_image).astype(np.int32)
        elif dtype == 'uint':
            data = itk.GetArrayFromImage(itk_image).astype(np.uint32)
        elif dtype == 'uchar':
            data = itk.GetArrayFromImage(itk_image).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if is_vector:
            # Vector images: ITK gives (z,y,x,components) or (y,x,components)
            spatial_shape = data.shape[:-1]  # drop components
        else:
            spatial_shape = data.shape

        image_dimension = len(spatial_shape)

        direction = np.asarray(itk_image.GetDirection()).reshape(
            (image_dimension, image_dimension)
        )
        spacing = list(itk_image.GetSpacing())
        origin = list(itk_image.GetOrigin())

        # Reshape the array properly for ANTsPy
        if is_vector:
            data_reshaped = data.transpose(
                list(range(image_dimension - 1, -1, -1)) + [image_dimension]
            )
        else:
            data_reshaped = data.transpose(list(range(image_dimension - 1, -1, -1)))

        ants_image: ants.ANTsImage = ants.from_numpy(
            data=data_reshaped,
            origin=origin,
            spacing=spacing,
            direction=direction,
            has_components=is_vector,
        )

        return ants_image

    def _antsfile_to_itk_affine_transform(self, ants_transform_file):
        """Convert ANTs affine transform to ITK affine transform.

        ANTs affine transform has 12 parameters for 3D:
        - parameters[0:9]: 3x3 transformation matrix in row-major order
        - parameters[9:12]: translation vector
        - fixed_parameters[0:3]: center of rotation

        Args:
            ants_transform_file (str): Path to ANTs transform file

        Returns:
            itk.AffineTransform: Converted ITK affine transform
        """
        ants_tfm = ants.read_transform(ants_transform_file)

        params = np.array(ants_tfm.parameters)
        fixed_params = np.array(ants_tfm.fixed_parameters)

        # Parameters structure for 3D affine:
        # params[0:9] = 3x3 matrix in row-major order
        # params[9:12] = translation vector
        # fixed_params[0:3] = center of rotation

        # Create ITK affine transform
        affine_tfm = itk.AffineTransform[itk.D, 3].New()

        # Set the center of rotation (fixed parameters)
        center = itk.Point[itk.D, 3]()
        for i in range(3):
            center[i] = fixed_params[i]
        affine_tfm.SetCenter(center)

        # Set the 3x3 transformation matrix (first 9 parameters in row-major order)
        mat = np.zeros((3, 3), dtype=np.float64)
        for row in range(3):
            for col in range(3):
                mat[row, col] = params[row * 3 + col]
        mat_itk = itk.GetMatrixFromArray(mat)
        affine_tfm.SetMatrix(mat_itk)

        # Set the translation vector (last 3 parameters)
        translation = itk.Vector[itk.D, 3]()
        for i in range(3):
            translation[i] = params[9 + i]
        affine_tfm.SetTranslation(translation)

        return affine_tfm

    def _antsfile_to_itk_displacement_field_transform(
        self, ants_transform_file, ref_image
    ):
        """Create ITK displacement field from ANTs transform.

        Args:
            ants_transform_file (str): Path to ANTs transform file
            reference_image (itk.image): Reference image for field generation

        Returns:
            itk.DisplacementFieldTransform: ITK displacement field transform
        """
        disp_field_tfm_ants = ants.read_transform(
            ants_transform_file, precision='double'
        )
        disp_field_ants = ants.transform_to_displacement_field(
            disp_field_tfm_ants,
            self._itk_to_ants_image(ref_image, dtype='float'),
        )

        disp_field_itk = self._ants_to_itk_image(disp_field_ants)

        disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        disp_tfm.SetDisplacementField(disp_field_itk)

        return disp_tfm

    def itk_transform_to_ants_transform(
        self, itk_tfm: itk.Transform, reference_image: itk.Image
    ):
        """Convert ITK transform to ANTsPy transform object.

        This method converts any ITK transform (Affine, Rigid, DisplacementField, etc.)
        to an ANTsPy transform object that can be used as initial_transform in
        ants.registration() or ants.label_image_registration().

        The conversion process:
        1. Uses TransformTools to convert the ITK transform to a displacement field
        2. Converts the displacement field image from ITK to ANTs format
        3. Creates an ANTsPy transform object from the displacement field

        Args:
            itk_tfm (itk.Transform): Input ITK transform to convert. Can be any
                ITK transform type (AffineTransform, DisplacementFieldTransform,
                CompositeTransform, etc.)
            reference_image (itk.Image): Reference image that defines the spatial
                domain for the displacement field (spacing, size, origin, direction)

        Returns:
            ants.core.ANTsTransform: ANTsPy transform object suitable for use
                as initial_transform parameter in ANTs registration functions

        Example:
            >>> # Convert ITK affine transform to ANTs
            >>> affine_itk = itk.AffineTransform[itk.D, 3].New()
            >>> affine_itk.SetIdentity()
            >>> affine_ants = registrar.itk_transform_to_ants_transform(
            ...     affine_itk, reference_image
            ... )
            >>>
            >>> # Use in registration
            >>> result = ants.registration(
            ...     fixed=fixed_ants,
            ...     moving=moving_ants,
            ...     initial_transform=affine_ants
            ... )
        """
        # Use TransformTools to convert any ITK transform to displacement field
        transform_tools = TransformTools()
        disp_field_itk = transform_tools.convert_transform_to_displacement_field(
            tfm=itk_tfm,
            reference_image=reference_image,
            np_component_type=np.float64,
            use_reference_image_as_mask=False,
        )

        # Convert ITK displacement field to ANTs image format
        disp_field_ants = self._itk_to_ants_image(disp_field_itk, dtype='double')

        # Create ANTs transform object from displacement field
        ants_tfm = ants.transform_from_displacement_field(disp_field_ants)

        return ants_tfm

    def _antsfiles_to_itk_transforms(
        self,
        ants_transforms,
        reference_image,
        inverse=False,
    ):
        phi = itk.CompositeTransform[itk.D, 3].New()
        for ants_tfm_filename in ants_transforms:
            tfm = ants.read_transform(ants_tfm_filename)
            if tfm.transform_type == "AffineTransform":
                affine_tfm_itk = self._antsfile_to_itk_affine_transform(
                    ants_tfm_filename
                )
                if inverse:
                    affine_tfm_itk = affine_tfm_itk.GetInverseTransform()
                phi.AddTransform(affine_tfm_itk)
            elif tfm.transform_type == "DisplacementFieldTransform":
                disp_tfm_itk = self._antsfile_to_itk_displacement_field_transform(
                    ants_tfm_filename, reference_image
                )
                phi.AddTransform(disp_tfm_itk)
            else:
                raise ValueError(
                    f"Unsupported ANTs transform type: {tfm.transform_type}"
                )

        return phi

    def registration_method(
        self,
        moving_image,
        moving_image_mask=None,
        moving_image_pre=None,
        images_are_labelmaps=False,
        initial_phi_MF=None,
    ):
        """Register moving image to fixed image using ANTs registration algorithm.

        Implementation of the abstract register() method from RegisterImagesBase.
        Performs deformable registration to align the moving image with the
        fixed image using ANTs SyN or other specified algorithms.

        Args:
            moving_image (itk.image): The 3D image to be registered/aligned. When
                images_are_labelmaps=True, this should be a label image for label-based
                registration
            moving_image_mask (itk.image, optional): Binary mask defining the
                region of interest in the moving image
            moving_image_pre (ants.core.ANTsImage, optional): Pre-processed moving image
                in ANTs format. If None, preprocessing is performed automatically
            images_are_labelmaps (bool, optional): If True, use label-based registration
                instead of intensity-based registration. In this mode, fixed_image and
                moving_image are treated as label images
            initial_phi_MF (itk.Transform, optional): Initial transform from moving
                to fixed space. Can be any ITK transform type (Affine, Rigid,
                DisplacementField, Composite, etc.). Will be converted to ANTs
                format automatically. The returned transforms will include this
                initial transform composed with the registration result.

        Returns:
            dict: Dictionary containing:
                - "phi_FM": Forward transformation (fixed to moving)
                - "phi_MF": Backward transformation (moving to fixed)
                - "loss": Loss value from the registration

        Note:
            For SyN registration, the transformations are approximately inverse
            consistent. The forward and inverse transforms are stored separately
            by ANTs.

            IMPORTANT: ANTs registration does NOT include the initial_transform
            in its output fwdtransforms/invtransforms. This method automatically
            composes the initial transform with the registration result, so the
            returned phi_MF and phi_FM include both the initial alignment and
            the registration refinement.

        Implementation details:
            - Uses ANTs registration with configurable transform types
            - Supports multi-resolution optimization
            - Handles masked and unmasked registration
            - Returns ITK-compatible displacement field transforms
            - Initial transforms are converted from ITK to ANTs format automatically

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
            >>>
            >>> # Registration with initial transform
            >>> initial_tfm = itk.AffineTransform[itk.D, 3].New()
            >>> result = registrar.register(
            ...     moving_image, initial_phi_MF=initial_tfm
            ... )
        """
        if moving_image is not None:
            self.moving_image = moving_image

        if moving_image_pre is not None:
            self.moving_image_pre = moving_image_pre
        else:
            if self.moving_image is not None:
                self.moving_image_pre = self.preprocess(
                    self.moving_image, self.modality
                )

        if moving_image_mask is not None:
            self.moving_image_mask = moving_image_mask

        if self.fixed_image_pre is None:
            self.fixed_image_pre = self.preprocess(self.fixed_image, self.modality)

        # Convert initial ITK transform to ANTs format if provided
        initial_transform = "identity"
        if initial_phi_MF is not None:
            print("Converting initial ITK transform to ANTs format...")
            initial_transform = self.itk_transform_to_ants_transform(
                itk_tfm=initial_phi_MF, reference_image=self.fixed_image
            )
            print("✓ Initial transform converted successfully")

        if images_are_labelmaps:
            registration_result = ants.label_image_registration(
                fixed_label_images=self._itk_to_ants_image(self.fixed_image),
                moving_label_images=self._itk_to_ants_image(self.moving_image),
                initial_transform=initial_transform,
                verbose=True,
            )
        else:
            if self.fixed_image_mask is not None and self.moving_image_mask is not None:
                registration_result = ants.registration(
                    fixed=self._itk_to_ants_image(self.fixed_image_pre),
                    mask=self._itk_to_ants_image(self.fixed_image_mask),
                    moving=self._itk_to_ants_image(self.moving_image_pre),
                    moving_mask=self._itk_to_ants_image(self.moving_image_mask),
                    initial_transform=initial_transform,
                    type_of_transform="antsRegistrationSyNQuick[so]",
                    use_histogram_matching=False,
                    mask_all_stages=True,
                    verbose=True,
                    reg_iterations=self.number_of_iterations,
                )
            else:
                registration_result = ants.registration(
                    fixed=self._itk_to_ants_image(self.fixed_image_pre),
                    moving=self._itk_to_ants_image(self.moving_image_pre),
                    initial_transform=initial_transform,
                    type_of_transform="antsRegistrationSyNQuick[so]",
                    use_histogram_matching=False,
                    verbose=True,
                    reg_iterations=self.number_of_iterations,
                )

        # Convert ANTs transforms to ITK
        phi_MF_reg = self._antsfiles_to_itk_transforms(
            registration_result['fwdtransforms'],
            inverse=False,
            reference_image=self.fixed_image,
        )
        phi_FM_reg = self._antsfiles_to_itk_transforms(
            registration_result['invtransforms'],
            inverse=True,
            reference_image=self.moving_image,
        )

        # Important: ANTs does NOT include the initial_transform in the output transforms
        # We need to manually compose them
        if initial_phi_MF is not None:
            print("Composing initial transform with registration result...")

            # For phi_MF (Moving -> Fixed): Apply initial_phi_MF first, then registration
            # Transform order: point -> initial_phi_MF -> phi_MF_reg
            phi_MF = itk.CompositeTransform[itk.D, 3].New()
            phi_MF.AddTransform(initial_phi_MF)
            # Add transforms from registration result (may be composite)
            if isinstance(phi_MF_reg, itk.CompositeTransform[itk.D, 3]):
                for i in range(phi_MF_reg.GetNumberOfTransforms()):
                    phi_MF.AddTransform(phi_MF_reg.GetNthTransform(i))
            else:
                phi_MF.AddTransform(phi_MF_reg)

            # For phi_FM (Fixed -> Moving): Apply registration inverse first, then initial inverse
            # Transform order: point -> phi_FM_reg -> initial_phi_MF^(-1)
            phi_FM = itk.CompositeTransform[itk.D, 3].New()
            # Add registration inverse transforms
            if isinstance(phi_FM_reg, itk.CompositeTransform[itk.D, 3]):
                for i in range(phi_FM_reg.GetNumberOfTransforms()):
                    phi_FM.AddTransform(phi_FM_reg.GetNthTransform(i))
            else:
                phi_FM.AddTransform(phi_FM_reg)
            # Invert and add initial transform
            # For displacement field transforms, we need to invert properly
            transform_tools = TransformTools()
            initial_phi_FM = transform_tools.invert_displacement_field_transform(
                transform_tools.convert_transform_to_displacement_field_transform(
                    initial_phi_MF, self.moving_image
                )
            )
            phi_FM.AddTransform(initial_phi_FM)

            print("✓ Transforms composed successfully")
        else:
            # No initial transform, use registration results directly
            phi_MF = phi_MF_reg
            phi_FM = phi_FM_reg
        moving_image_reg = registration_result['warpedmovout']
        loss = ants.image_similarity(
            self._itk_to_ants_image(self.fixed_image),
            moving_image_reg,
        )

        return {"phi_FM": phi_FM, "phi_MF": phi_MF, "loss": loss}


def parse_args():
    """Parse command line arguments for image registration.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - fixed_image: Path to fixed/reference image
            - moving_image: Path to moving image to register
            - output_image: Path for registered output image
            - modality: Image modality (e.g., 'ct', 'mri')
            - transform_type: Type of transform (default: 'SyN')
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_image", type=str, required=True)
    parser.add_argument("--moving_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--transform_type", type=str, default='SyN')
    return parser.parse_args()


if __name__ == "__main__":
    """Command line interface for ANTs-based image registration.

    Example usage:
        python register_images_ants.py \
            --fixed_image reference.mha \
            --moving_image timepoint_05.mha \
            --output_image registered.mha \
            --modality ct \
            --transform_type SyN
    """
    args = parse_args()
    registrar = RegisterImagesANTs()
    registrar.set_modality(args.modality)
    registrar.set_transform_type(args.transform_type)
    registrar.set_fixed_image(itk.imread(args.fixed_image))
    moving_image = itk.imread(args.moving_image)
    result = registrar.register(moving_image=moving_image)
    res_phi_FM = result["phi_FM"]
    res_phi_MF = result["phi_MF"]

    # Apply transform using ANTs
    moving_image_ants = registrar.preprocess(moving_image, args.modality)
    # res_phi_MF contains the forward transform files (moving to fixed)
    moving_image_reg_ants = ants.apply_transforms(
        fixed=registrar.fixed_image_pre,
        moving=moving_image_ants,
        transformlist=res_phi_MF,
    )

    # Convert back to ITK and save
    moving_image_reg = registrar._ants_to_itk_image(moving_image_reg_ants)
    itk.imwrite(moving_image_reg, args.output_image, compression=True)
