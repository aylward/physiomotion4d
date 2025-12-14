"""ICP-based model-to-model registration for anatomical models.

This module provides the RegisterModelToModelICP class for aligning anatomical
models using Iterative Closest Point (ICP) algorithm. The workflow includes:
1. Initial centroid alignment
2. Rigid or affine ICP alignment

The registration is particularly useful for initial rough alignment of generic
models to patient-specific anatomical data.

Key Features:
    - Centroid-based initial alignment
    - VTK ICP algorithm with rigid or affine transformation modes
    - Three-stage affine pipeline: centroid → rigid ICP → affine ICP
    - Support for PyVista meshes
    - Automatic transform composition

Example:
    >>> import pyvista as pv
    >>> from physiomotion4d import RegisterModelToModelICP
    >>>
    >>> # Load meshes
    >>> moving_mesh = pv.read("generic_model.vtu")
    >>> fixed_mesh = pv.read("patient_surface.stl")
    >>>
    >>> # Run affine registration
    >>> registrar = RegisterModelToModelICP(
    ...     moving_mesh=moving_mesh,
    ...     fixed_mesh=fixed_mesh
    ... )
    >>> result = registrar.register(mode='affine')
    >>>
    >>> # Access results
    >>> aligned_mesh = result['moving_mesh']
    >>> phi_MF = result['phi_MF']  # Moving to fixed transform
"""

import itk
import numpy as np
import pyvista as pv
import vtk

from physiomotion4d.transform_tools import TransformTools


class RegisterModelToModelICP:
    """Register anatomical models using Iterative Closest Point (ICP) algorithm.

    This class provides ICP-based alignment of 3D surface meshes with support for
    both rigid and affine transformation modes. The registration pipeline uses
    centroid alignment for initialization followed by VTK's ICP algorithm.

    **Registration Pipelines:**
        - **Rigid mode**: Centroid alignment → Rigid ICP
        - **Affine mode**: Centroid alignment → Rigid ICP → Affine ICP

    **Transform Convention:**
        - phi_MF: Backward transform (moving → fixed space)
        - phi_FM: Forward transform (fixed → moving space)

    Attributes:
        moving_mesh (pv.PolyData): Surface mesh to be aligned
        fixed_mesh (pv.PolyData): Target surface mesh
        transform_tools (TransformTools): Transform utility instance
        phi_MF (itk.AffineTransform): Optimized backward transform
        phi_FM (itk.AffineTransform): Optimized forward transform
        registered_mesh (pv.PolyData): Aligned moving mesh

    Example:
        >>> # Initialize with meshes
        >>> registrar = RegisterModelToModelICP(
        ...     moving_mesh=model_surface,
        ...     fixed_mesh=patient_surface
        ... )
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(mode='rigid', max_iterations=2000)
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(mode='affine', max_iterations=2000)
        >>>
        >>> # Get aligned mesh and transforms
        >>> aligned_mesh = result['moving_mesh']
        >>> phi_MF = result['phi_MF']
    """

    def __init__(
        self,
        moving_mesh: pv.PolyData,
        fixed_mesh: pv.PolyData,
    ):
        """Initialize ICP-based model registration.

        Args:
            moving_mesh: PyVista surface mesh to be aligned to fixed mesh
            fixed_mesh: PyVista target surface mesh

        Note:
            The moving_mesh is typically extracted from a VTU model using
            mesh.extract_surface() before passing to this class.
        """
        self.moving_mesh = moving_mesh
        self.fixed_mesh = fixed_mesh

        # Transform utilities
        self.transform_tools = TransformTools()

        # Registration results
        self.phi_MF: itk.AffineTransform = None  # Backward (moving→fixed)
        self.phi_FM: itk.AffineTransform = None  # Forward (fixed→moving)
        self.registered_mesh: pv.PolyData = None

    def register(self, mode: str = 'affine', max_iterations: int = 2000) -> dict:
        """Perform ICP alignment of moving mesh to fixed mesh.

        This method executes alignment with either rigid or affine transformations:

        **Rigid mode:**
            1. Centroid alignment: Translate moving mesh to align mass centers
            2. Rigid ICP: Refine with rigid-body transformation (rotation + translation)

        **Affine mode:**
            1. Centroid alignment: Translate moving mesh to align mass centers
            2. Rigid ICP: Refine with rigid-body transformation
            3. Affine ICP: Further refine with affine transformation (includes scaling/shearing)

        Args:
            mode: Registration mode, either 'rigid' or 'affine'. Default: 'affine'
            max_iterations: Maximum number of ICP iterations per stage. Default: 2000

        Returns:
            Dictionary containing:
                - 'moving_mesh': Aligned moving mesh (PyVista PolyData)
                - 'phi_MF': Backward transform (moving→fixed, ITK AffineTransform)
                - 'phi_FM': Forward transform (fixed→moving, ITK AffineTransform)

        Raises:
            ValueError: If mode is not 'rigid' or 'affine'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(mode='rigid', max_iterations=5000)
            >>>
            >>> # Affine registration
            >>> result = registrar.register(mode='affine', max_iterations=2000)
        """
        if mode not in ['rigid', 'affine']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'rigid' or 'affine'.")

        print(
            f"Performing {mode.upper()} ICP alignment of moving mesh to fixed mesh..."
        )

        # Step 1: Centroid alignment (common to both modes)
        self.registered_mesh = self.moving_mesh.copy(deep=True)

        moving_centroid = np.array(self.registered_mesh.center)
        print(f"  Moving mesh centroid: {moving_centroid}")
        fixed_centroid = np.array(self.fixed_mesh.center)
        print(f"  Fixed mesh centroid: {fixed_centroid}")
        translation = fixed_centroid - moving_centroid
        print(f"  Step 1: Translating by {translation} to align centroids...")

        # Create ITK affine transform with translation
        phi_ICP = itk.AffineTransform[itk.D, 3].New()
        phi_ICP.SetIdentity()
        phi_ICP.SetOffset(translation)

        # Apply centroid alignment to mesh
        self.registered_mesh = self.transform_tools.transform_pvcontour(
            self.registered_mesh,
            phi_ICP,
            with_deformation_magnitude=False,
        )

        print(f"  Center after Step 1: {self.registered_mesh.center}")

        # Step 2: Rigid ICP (common to both modes)
        print(f"  Step 2: Performing rigid ICP (max iterations: {max_iterations})...")
        icp_rigid = vtk.vtkIterativeClosestPointTransform()
        icp_rigid.SetSource(self.registered_mesh)
        icp_rigid.SetTarget(self.fixed_mesh)
        icp_rigid.GetLandmarkTransform().SetModeToRigidBody()  # Rigid mode
        icp_rigid.SetMaximumNumberOfIterations(max_iterations)
        icp_rigid.Update()

        # Convert VTK transform to ITK and compose with centroid transform
        rigid_transform = self.transform_tools.convert_vtk_matrix_to_itk_transform(
            icp_rigid.GetMatrix()
        )
        phi_ICP.Compose(rigid_transform)

        # Apply rigid ICP transform to mesh
        self.registered_mesh = self.transform_tools.transform_pvcontour(
            self.registered_mesh,
            rigid_transform,
            with_deformation_magnitude=False,
        )

        print(f"  Center after Step 2: {self.registered_mesh.center}")

        # Step 3: Affine ICP (only if affine mode)
        if mode == 'affine':
            print(
                f"  Step 3: Performing affine ICP (max iterations: {max_iterations})..."
            )
            icp_affine = vtk.vtkIterativeClosestPointTransform()
            icp_affine.SetSource(self.registered_mesh)
            icp_affine.SetTarget(self.fixed_mesh)
            icp_affine.GetLandmarkTransform().SetModeToAffine()  # Affine mode
            icp_affine.SetMaximumNumberOfIterations(max_iterations)
            icp_affine.Update()

            # Convert VTK transform to ITK and compose
            affine_transform = self.transform_tools.convert_vtk_matrix_to_itk_transform(
                icp_affine.GetMatrix()
            )
            phi_ICP.Compose(affine_transform)

            # Apply affine ICP transform to mesh
            self.registered_mesh = self.transform_tools.transform_pvcontour(
                self.registered_mesh,
                affine_transform,
                with_deformation_magnitude=False,
            )

            print(f"  Center after Step 3: {self.registered_mesh.center}")

        # Compute inverse transform
        self.phi_MF = phi_ICP.GetInverseTransform()
        self.phi_FM = phi_ICP

        print(f"  {mode.upper()} ICP registration complete!")

        # Return results as dictionary
        return {
            'moving_mesh': self.registered_mesh,
            'phi_MF': self.phi_MF,
            'phi_FM': self.phi_FM,
        }


# Example usage
if __name__ == "__main__":
    """Example demonstrating ICP-based model-to-model registration."""

    import pyvista as pv

    # =========================================================================
    # Setup: Load meshes
    # =========================================================================
    print("Loading meshes...")
    moving_mesh = pv.read("generic_model.vtu").extract_surface()
    fixed_mesh = pv.read("patient_surface.stl")
    print(f"  Moving mesh: {moving_mesh.n_points} points")
    print(f"  Fixed mesh: {fixed_mesh.n_points} points")

    # =========================================================================
    # Example 1: Rigid ICP registration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Rigid ICP Registration")
    print("=" * 60)

    registrar_rigid = RegisterModelToModelICP(
        moving_mesh=moving_mesh,
        fixed_mesh=fixed_mesh,
    )

    result_rigid = registrar_rigid.register(mode='rigid', max_iterations=2000)

    # Save rigid result
    result_rigid['moving_mesh'].save("aligned_mesh_rigid_icp.vtk")
    print(f"\n  Saved: aligned_mesh_rigid_icp.vtk")

    # =========================================================================
    # Example 2: Affine ICP registration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Affine ICP Registration")
    print("=" * 60)

    registrar_affine = RegisterModelToModelICP(
        moving_mesh=moving_mesh,
        fixed_mesh=fixed_mesh,
    )

    result_affine = registrar_affine.register(mode='affine', max_iterations=2000)

    # Save affine result
    result_affine['moving_mesh'].save("aligned_mesh_affine_icp.vtk")
    print(f"\n  Saved: aligned_mesh_affine_icp.vtk")

    print("\nICP registration examples complete!")
