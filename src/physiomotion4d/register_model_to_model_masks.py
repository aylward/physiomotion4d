"""Mask-based model-to-model registration for anatomical models.

This module provides the RegisterModelToModelMasks class for aligning anatomical
models using mask-based deformable registration. The workflow includes:
1. Generate binary masks from moving and fixed meshes
2. Generate ROI masks with dilation
4. Progressive registration stages:
   - rigid: ANTs rigid registration
   - affine: ANTs rigid → affine registration
   - deformable: ANTs rigid → affine → deformable (SyN) registration
5. Optional ICON refinement at end

The registration is particularly useful for aligning anatomical models where
shape differences require deformable transformations beyond rigid/affine ICP.

Key Features:
    - Automatic mask generation from PyVista meshes
    - Multi-stage ANTs registration (rigid/affine/deformable)
    - Optional ICON deep learning refinement
    - Automatic transform composition
    - Support for PyVista meshes

Example:
    >>> import itk
    >>> import pyvista as pv
    >>> from physiomotion4d import RegisterModelToModelMasks
    >>>
    >>> # Load meshes and reference image
    >>> moving_mesh = pv.read("generic_model.vtu").extract_surface()
    >>> fixed_mesh = pv.read("patient_surface.stl")
    >>> reference_image = itk.imread("patient_ct.nii.gz")
    >>>
    >>> # Run deformable registration with ICON refinement
    >>> registrar = RegisterModelToModelMasks(
    ...     moving_mesh=moving_mesh,
    ...     fixed_mesh=fixed_mesh,
    ...     reference_image=reference_image,
    ...     roi_dilation_mm=20,
    ... )
    >>> result = registrar.register(
    ...     mode='deformable',
    ...     use_icon=True,
    ...     icon_iterations=50
    ... )
    >>>
    >>> # Access results
    >>> aligned_mesh = result['moving_mesh']
    >>> phi_MF = result['phi_MF']  # Moving to fixed transform
"""

import itk
import numpy as np
import pyvista as pv
from itk import TubeTK as ttk

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools


class RegisterModelToModelMasks:
    """Register anatomical models using mask-based deformable registration.

    This class provides mask-based alignment of 3D surface meshes with support for
    rigid, affine, and deformable transformation modes. The registration pipeline
    generates masks from meshes, applies optional dilation, and uses ANTs for
    progressive multi-stage registration with optional ICON refinement.

    **Registration Pipelines:**
        - **Rigid mode**: ANTs rigid registration
        - **Affine mode**: ANTs rigid → affine registration
        - **Deformable mode**: ANTs rigid → affine → deformable (SyN) registration
        - **Optional**: ICON deep learning refinement after any mode

    **Transform Convention:**
        - phi_MF: Backward transform (moving → fixed space)
        - phi_FM: Forward transform (fixed → moving space)

    Attributes:
        moving_mesh (pv.PolyData): Surface mesh to be aligned
        fixed_mesh (pv.PolyData): Target surface mesh
        reference_image (itk.Image): Reference image for coordinate frame
        roi_dilation_mm (float): Dilation amount in mm for ROI mask
        transform_tools (TransformTools): Transform utility instance
        contour_tools (ContourTools): Mesh utility instance
        registrar_ants (RegisterImagesANTs): ANTs registration instance
        registrar_icon (RegisterImagesICON): ICON registration instance
        phi_MF (itk.CompositeTransform): Optimized backward transform
        phi_FM (itk.CompositeTransform): Optimized forward transform
        registered_mesh (pv.PolyData): Aligned moving mesh

    Example:
        >>> # Initialize with meshes and reference image
        >>> registrar = RegisterModelToModelMasks(
        ...     moving_mesh=model_surface,
        ...     fixed_mesh=patient_surface,
        ...     reference_image=patient_ct,
        ...     roi_dilation_mm=20,
        ... )
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(mode='rigid')
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(mode='affine')
        >>>
        >>> # Or run deformable with ICON refinement
        >>> result = registrar.register(
        ...     mode='deformable',
        ...     use_icon=True,
        ...     icon_iterations=100
        ... )
        >>>
        >>> # Get aligned mesh and transforms
        >>> aligned_mesh = result['moving_mesh']
        >>> phi_MF = result['phi_MF']
    """

    def __init__(
        self,
        moving_mesh: pv.PolyData,
        fixed_mesh: pv.PolyData,
        reference_image: itk.Image,
        roi_dilation_mm: float = 10,
    ):
        """Initialize mask-based model registration.

        Args:
            moving_mesh: PyVista surface mesh to be aligned to fixed mesh
            fixed_mesh: PyVista target surface mesh
            reference_image: ITK image providing coordinate frame (origin, spacing, direction)
                for mask generation. Typically the patient CT/MRI image.
            roi_dilation_mm: Dilation amount in millimeters for ROI mask generation.
                Default: 20mm

        Note:
            The moving_mesh and fixed_mesh are typically extracted from VTU models
            using mesh.extract_surface() before passing to this class.
        """
        self.moving_mesh = moving_mesh
        self.fixed_mesh = fixed_mesh
        self.reference_image = reference_image
        self.roi_dilation_mm = roi_dilation_mm

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()

        # Registration instances
        self.registrar_ants = RegisterImagesANTs()
        self.registrar_icon = RegisterImagesICON()
        self.registrar_icon.set_modality('ct')
        self.registrar_icon.set_multi_modality(True)  # For mask-based registration

        # Generated masks (will be created during registration)
        self.fixed_mask_image: itk.Image = None
        self.fixed_mask_roi_image: itk.Image = None
        self.moving_mask_image: itk.Image = None
        self.moving_mask_roi_image: itk.Image = None

        # Registration results
        self.phi_MF: itk.CompositeTransform = None  # Backward (moving→fixed)
        self.phi_FM: itk.CompositeTransform = None  # Forward (fixed→moving)
        self.registered_mesh: pv.PolyData = None

    def _create_masks_from_meshes(self):
        """Generate binary mask images from moving and fixed meshes.

        Creates:
            - fixed_mask_image: Binary mask from fixed mesh
            - fixed_mask_roi_image: Dilated ROI mask from fixed mesh
            - moving_mask_image: Binary mask from moving mesh
            - moving_mask_roi_image: Dilated ROI mask from moving mesh

        Uses self.reference_image for coordinate frame (origin, spacing, direction).
        """
        print("Generating binary masks from meshes...")

        # Create fixed mask
        self.fixed_mask_image = (
            self.contour_tools.create_contour_distance_map_from_mesh(
                self.fixed_mesh,
                self.reference_image,
                max_distance=100.0,
                invert_distance_map=True,
            )
        )

        # Create fixed ROI mask with dilation
        print(f"  Dilating fixed mask by {self.roi_dilation_mm}mm for ROI...")
        mask = self.contour_tools.create_mask_from_mesh(
            self.fixed_mesh, self.reference_image
        )
        imMath = ttk.ImageMath.New(mask)
        dilation_voxels = int(
            self.roi_dilation_mm / self.reference_image.GetSpacing()[0]
        )
        imMath.Dilate(dilation_voxels, 1, 0)
        self.fixed_mask_roi_image = imMath.GetOutput()

        # Create moving mask
        self.moving_mask_image = (
            self.contour_tools.create_contour_distance_map_from_mesh(
                self.moving_mesh,
                self.reference_image,
                max_distance=100.0,
                invert_distance_map=True,
            )
        )

        # Create moving ROI mask with dilation
        print(f"  Dilating moving mask by {self.roi_dilation_mm}mm for ROI...")
        mask = self.contour_tools.create_mask_from_mesh(
            self.moving_mesh, self.reference_image
        )
        imMath = ttk.ImageMath.New(self.moving_mask_image)
        imMath.Dilate(dilation_voxels, 1, 0)
        self.moving_mask_roi_image = imMath.GetOutputUChar()

        print("  Mask generation complete.")

    def register(
        self,
        mode: str = 'affine',
        use_icon: bool = False,
        icon_iterations: int = 50,
    ) -> dict:
        """Perform mask-based registration of moving mesh to fixed mesh.

        This method executes progressive multi-stage registration:

        **Rigid mode:**
            1. Generate masks from meshes
            3. ANTs rigid registration

        **Affine mode:**
            1. Generate masks from meshes
            3. ANTs affine registration (includes rigid stage)

        **Deformable mode:**
            1. Generate masks from meshes
            3. ANTs SyN deformable registration (includes rigid + affine + deformable stages)

        **Optional ICON refinement** (all modes):
            4. ICON deep learning registration for fine-tuning

        Args:
            mode: Registration mode - 'rigid', 'affine', or 'deformable'. Default: 'affine'
            use_icon: Whether to apply ICON registration refinement after ANTs. Default: False
            icon_iterations: Number of ICON optimization iterations if use_icon=True. Default: 50

        Returns:
            Dictionary containing:
                - 'moving_mesh': Aligned moving mesh (PyVista PolyData)
                - 'phi_MF': Backward transform (moving→fixed, ITK CompositeTransform)
                - 'phi_FM': Forward transform (fixed→moving, ITK CompositeTransform)

        Raises:
            ValueError: If mode is not 'rigid', 'affine', or 'deformable'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(mode='rigid')
            >>>
            >>> # Affine registration
            >>> result = registrar.register(mode='affine')
            >>>
            >>> # Deformable registration with ICON refinement
            >>> result = registrar.register(mode='deformable', use_icon=True, icon_iterations=100)
        """
        if mode not in ['rigid', 'affine', 'deformable']:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'rigid', 'affine', or 'deformable'."
            )

        print(f"Performing {mode.upper()} mask-based registration...")

        # Step 1: Generate masks from meshes
        self._create_masks_from_meshes()

        # Step 3: ANTs registration with appropriate type_of_transform
        if mode == 'rigid':
            transform_type = "Rigid"
        elif mode == 'affine':
            transform_type = "Affine"  # Includes rigid stage
        else:  # deformable
            transform_type = "SyN"  # Includes rigid + affine + deformable stages

        print(f"  Performing ANTs {mode} registration (type: {transform_type})...")

        self.registrar_ants.set_fixed_image(self.fixed_mask_image)
        self.registrar_ants.set_fixed_image_mask(self.fixed_mask_roi_image)

        result_ants = self.registrar_ants.register(
            moving_image=self.moving_mask_image,
            moving_image_mask=self.moving_mask_roi_image,
        )
        phi_FM_ants = result_ants["phi_FM"]
        phi_MF_ants = result_ants["phi_MF"]

        # Initialize composite transforms
        self.phi_MF = phi_MF_ants
        self.phi_FM = phi_FM_ants

        # Step 4: Optional ICON refinement
        if use_icon:
            print(
                f"  Performing ICON refinement registration ({icon_iterations} iterations)..."
            )

            # Transform masks with ANTs result for ICON input
            moving_mask_ants_transformed = self.transform_tools.transform_image(
                self.moving_mask_image,
                phi_FM_ants,
                self.reference_image,
                interpolation_method="linear",
            )

            # Configure ICON
            self.registrar_icon.set_number_of_iterations(icon_iterations)
            self.registrar_icon.set_fixed_image(self.fixed_mask_image)
            self.registrar_icon.set_fixed_image_mask(self.fixed_mask_roi_image)

            # ICON registration
            result_icon = self.registrar_icon.register(
                moving_image=moving_mask_ants_transformed,
                moving_image_mask=self.moving_mask_roi_image,
            )
            phi_FM_icon = result_icon["phi_FM"]
            phi_MF_icon = result_icon["phi_MF"]

            # Compose ANTs and ICON transforms
            composed_phi_MF = itk.CompositeTransform[itk.D, 3].New()
            composed_phi_MF.AddTransform(phi_MF_ants)
            composed_phi_MF.AddTransform(phi_MF_icon)

            composed_phi_FM = itk.CompositeTransform[itk.D, 3].New()
            composed_phi_FM.AddTransform(phi_FM_icon)
            composed_phi_FM.AddTransform(phi_FM_ants)

            self.phi_MF = composed_phi_MF
            self.phi_FM = composed_phi_FM

        # Apply final transform to moving mesh
        print("  Transforming moving mesh...")
        self.registered_mesh = self.transform_tools.transform_pvcontour(
            self.moving_mesh,
            self.phi_MF,
            with_deformation_magnitude=True,
        )

        print(f"  {mode.upper()} mask-based registration complete!")

        # Return results as dictionary
        return {
            'moving_mesh': self.registered_mesh,
            'phi_MF': self.phi_MF,
            'phi_FM': self.phi_FM,
        }


# Example usage
if __name__ == "__main__":
    """Example demonstrating mask-based model-to-model registration."""

    import itk
    import pyvista as pv

    # =========================================================================
    # Setup: Load meshes and reference image
    # =========================================================================
    print("Loading meshes and reference image...")
    moving_mesh = pv.read("generic_model.vtu").extract_surface()
    fixed_mesh = pv.read("patient_surface.stl")
    reference_image = itk.imread("patient_ct.nii.gz")
    print(f"  Moving mesh: {moving_mesh.n_points} points")
    print(f"  Fixed mesh: {fixed_mesh.n_points} points")
    print(f"  Reference image: {reference_image.GetLargestPossibleRegion().GetSize()}")

    # =========================================================================
    # Example 1: Rigid registration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Rigid Mask-based Registration")
    print("=" * 60)

    registrar_rigid = RegisterModelToModelMasks(
        moving_mesh=moving_mesh,
        fixed_mesh=fixed_mesh,
        reference_image=reference_image,
        roi_dilation_mm=20,
    )

    result_rigid = registrar_rigid.register(mode='rigid')

    # Save rigid result
    result_rigid['moving_mesh'].save("aligned_mesh_rigid_masks.vtk")
    print(f"\n  Saved: aligned_mesh_rigid_masks.vtk")

    # =========================================================================
    # Example 2: Affine registration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Affine Mask-based Registration")
    print("=" * 60)

    registrar_affine = RegisterModelToModelMasks(
        moving_mesh=moving_mesh,
        fixed_mesh=fixed_mesh,
        reference_image=reference_image,
        roi_dilation_mm=20,
    )

    result_affine = registrar_affine.register(mode='affine')

    # Save affine result
    result_affine['moving_mesh'].save("aligned_mesh_affine_masks.vtk")
    print(f"\n  Saved: aligned_mesh_affine_masks.vtk")

    # =========================================================================
    # Example 3: Deformable registration with ICON refinement
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Deformable Mask-based Registration with ICON")
    print("=" * 60)

    registrar_deformable = RegisterModelToModelMasks(
        moving_mesh=moving_mesh,
        fixed_mesh=fixed_mesh,
        reference_image=reference_image,
        roi_dilation_mm=20,
    )

    result_deformable = registrar_deformable.register(
        mode='deformable', use_icon=True, icon_iterations=50
    )

    # Save deformable result
    result_deformable['moving_mesh'].save("aligned_mesh_deformable_masks.vtk")
    print(f"\n  Saved: aligned_mesh_deformable_masks.vtk")

    print("\nMask-based registration examples complete!")
