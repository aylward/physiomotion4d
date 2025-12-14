"""PCA-based model-to-image registration for cardiac anatomical models.

This module provides the RegisterModelToImagePCA class for registering
parametric anatomical models (VTK format with PCA shape variation) to patient-specific
imaging data. The workflow includes:

1. Stage 1: Rigid alignment (rotation + translation) to establish initial pose
2. Stage 2: Joint optimization of rigid parameters + PCA coefficients
   - Allows small refinements to rigid transform
   - Optimizes shape coefficients to maximize mean intensity at model points

The registration is particularly useful for cardiac modeling where a statistical
shape model (mean + PCA modes) needs to be fitted to contrast-enhanced CT images.

Key Features:
    - Two-stage optimization (coarse rigid then joint rigid+shape)
    - PCA-based shape model with eigenmode variation
    - Intensity-based metric (maximize mean intensity at model points)
    - ITK linear interpolation for continuous intensity sampling
    - Quaternion-based rigid transform (VersorRigid3DTransform) avoids gimbal lock
    - Support for VTK unstructured grids and surface meshes

Example:
    >>> import itk
    >>> import numpy as np
    >>> from physiomotion4d import RegisterModelToImagePCA
    >>>
    >>> # Load patient image
    >>> image = itk.imread("patient_ct.nrrd")
    >>>
    >>> # Load PCA model data from SlicerSALT format
    >>> average_mesh, eigenvalues, eigenvectors = (
    ...     RegisterModelToImagePCA.pca_read_slicersalt("pca.json", group_key='All')
    ... )
    >>> std_deviations = np.sqrt(eigenvalues)
    >>>
    >>> # Create initial transform
    >>> initial_transform = itk.VersorRigid3DTransform[itk.D].New()
    >>> initial_transform.SetIdentity()
    >>>
    >>> # Initialize PCA-based registration
    >>> registrar = RegisterModelToImagePCA(
    ...     average_mesh=average_mesh,
    ...     eigenvectors=eigenvectors,
    ...     std_deviations=std_deviations,
    ...     reference_image=image
    ... )
    >>>
    >>> # Run complete two-stage registration
    >>> result = registrar.register(initial_transform=initial_transform)
    >>>
    >>> # Access results
    >>> registered_mesh = result['registered_mesh']
    >>> pca_coefficients = result['pca_coefficients']
    >>> final_intensity = result['final_intensity']
"""

import json
import logging
from pathlib import Path
from typing import Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import itk
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from scipy.spatial import KDTree

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.transform_tools import TransformTools


class RegisterModelToImagePCA(PhysioMotion4DBase):
    """Register PCA-based shape models to medical images using intensity optimization.

    This class implements a two-stage registration pipeline for fitting statistical
    shape models to patient-specific medical images:

    **Stage 1: Coarse Rigid Alignment**
        - Optimizes 6 DOF rigid transform (ITK VersorRigid3DTransform)
        - Uses quaternion representation to avoid gimbal lock
        - Establishes initial pose of the model in image coordinate system
        - Uses Nelder-Mead optimization to maximize mean intensity

    **Stage 2: Joint Rigid + PCA Deformable Registration**
        - Simultaneously optimizes rigid parameters AND PCA coefficients
        - Rigid parameters are allowed small refinements from Stage 1
        - Model equation: P = rigid_transform(mean + Σ(b_i * std_i * eigenvector_i))
        - Maximizes mean intensity at deformed model points

    **Optimization Objective:**
        Maximize the mean intensity of the image sampled at model points using
        ITK's LinearInterpolateImageFunction. This aligns the model with bright
        regions in contrast-enhanced images (e.g., blood pool in cardiac CT).

    Attributes:
        average_mesh (pv.UnstructuredGrid): Mean shape model
        eigenvectors (np.ndarray): PCA eigenvectors/components (modes × n_points*3)
        std_deviations (np.ndarray): Standard deviations per mode (modes,)
        reference_image (itk.Image): Patient image providing coordinate frame and intensity data
        n_points (int): Number of points in the mesh
        n_pca_modes (int): Number of PCA modes available
        rigid_transform (itk.VersorRigid3DTransform): Optimized rigid transformation
        pca_coefficients (np.ndarray): Optimized PCA coefficients
        registered_mesh (pv.UnstructuredGrid): Final registered and deformed mesh

    Example:
        >>> # Load PCA model data
        >>> average_mesh = pv.read("pca_All_mean.vtk")
        >>> with open("pca.json", 'r') as f:
        ...     pca_data = json.load(f)
        >>> group_data = pca_data['All']
        >>> std_deviations = np.sqrt(np.array(group_data['eigenvalues']))
        >>> eigenvectors = np.array(group_data['components'])
        >>>
        >>> # Initialize registrar with loaded data
        >>> registrar = RegisterModelToImagePCA(
        ...     average_mesh=average_mesh,
        ...     eigenvectors=eigenvectors,
        ...     std_deviations=std_deviations,
        ...     reference_image=patient_ct_image
        ... )
        >>>
        >>> # Create initial transform
        >>> initial_transform = itk.VersorRigid3DTransform[itk.D].New()
        >>> initial_transform.SetIdentity()
        >>>
        >>> # Run full registration pipeline
        >>> result = registrar.register(
        ...     initial_transform=initial_transform,
        ...     n_pca_modes=10
        ... )
        >>>
        >>> # Save registered mesh
        >>> result['registered_mesh'].save("registered_heart.vtk")
        >>>
        >>> # Print optimization results
        >>> print(f"Final intensity: {result['final_intensity']:.2f}")
        >>> print(f"PCA coefficients: {result['pca_coefficients']}")
    """

    def __init__(
        self,
        average_mesh: pv.UnstructuredGrid,
        eigenvectors: np.ndarray,
        std_deviations: np.ndarray,
        reference_image: Optional[itk.Image] = None,
        n_modes: int = -1,
        log_level: int | str = logging.INFO,
        point_subsample_step: int = 4,
    ):
        """Initialize the PCA-based model-to-image registration.

        Args:
            average_mesh: PyVista mesh containing the mean 3D shape model
                (unstructured grid or polydata)
            eigenvectors: Numpy array of PCA eigenvectors/components. Shape: (modes, n_points*3)
                Each row is a flattened eigenmode with 3D displacements: [x1,y1,z1, x2,y2,z2, ...]
            std_deviations: Numpy array of standard deviations per PCA mode. Shape: (modes,)
                These are the square roots of eigenvalues
            reference_image: ITK image providing the coordinate frame and intensity values
                for registration. If None, must be set later before registration.
            n_pca_modes: Number of PCA modes to use. Default: -1 (use all)
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Raises:
            ValueError: If eigenvector dimensions don't match mesh points
        """
        # Initialize base class with logging
        super().__init__(class_name="RegisterModelToImagePCA", log_level=log_level)

        # Store model data
        self.average_mesh: pv.UnstructuredGrid = average_mesh
        self.eigenvectors: np.ndarray = eigenvectors
        self.std_deviations: np.ndarray = std_deviations
        self.reference_image = reference_image

        self.n_pca_modes: int = n_modes
        if self.n_pca_modes == -1:
            self.n_pca_modes = len(std_deviations)

        # Working transform (reused to avoid repeated memory allocation)
        self._working_transform: itk.VersorRigid3DTransform = (
            itk.VersorRigid3DTransform[itk.D].New()
        )
        self._transform_center_of_rotation: itk.Point = itk.Point[itk.D, 3]()
        self._transform_center_of_rotation[0] = 0.0
        self._transform_center_of_rotation[1] = 0.0
        self._transform_center_of_rotation[2] = 0.0

        # Registration results - Stage 1 (coarse rigid)
        self.stage1_rigid_transform: Optional[itk.VersorRigid3DTransform] = None

        # Registration results - Stage 2 (refined rigid + PCA)
        self.rigid_transform: Optional[itk.VersorRigid3DTransform] = None
        self.pca_coefficients: Optional[np.ndarray] = None
        self.registered_mesh: Optional[pv.UnstructuredGrid] = None
        self.final_intensity: float = 0.0

        # Transform utilities
        self.transform_tools = TransformTools()

        # Image interpolator (created when needed)
        self._interpolator: Optional[itk.LinearInterpolateImageFunction] = None

        # KDTree for efficient nearest neighbor search (built lazily)
        self._kdtree: Optional[KDTree] = None

        # Distance threshold for transform_point method (in mm)
        self._transform_point_distance_threshold: float = 10.0  # Default: 10mm

        # Store mean and deformed points for computing displacements
        self._average_mesh_points = self.average_mesh.points
        self._average_mesh_points_deformed: Optional[np.ndarray] = None

        self._metric_call_count: int = 0

        # Pre-convert mean shape points to ITK format
        self.point_subsample_step = point_subsample_step
        self._average_mesh_points_itk: Optional[list[itk.Point]] = None
        self._create_itk_points()

    @classmethod
    def from_slicersalt(
        cls,
        average_mesh: pv.UnstructuredGrid,
        json_filename: str,
        group_key: str = 'All',
        reference_image: itk.Image = None,
        n_modes: int = -1,
        log_level: int | str = logging.INFO,
        point_subsample_step: int = 4,
    ) -> Self:
        """Read PCA model data from SlicerSALT format JSON file.

        This method reads PCA statistical shape model data from a JSON file
        created by SlicerSALT, including the mean mesh, eigenvalues, and
        eigenvector components.

        The method expects:
        1. A JSON file (e.g., 'pca.json') containing eigenvalues and components
        2. A corresponding VTK mesh file (e.g., 'pca_All_mean.vtk') in the same
           directory, where the filename follows the pattern:
           pca_{group_key}_mean.vtk

        Args:
            json_filename: Path to the SlicerSALT PCA JSON file
            group_key: Key for the PCA group to extract from JSON.
                Default: 'All'

        Returns:
            Tuple containing:
                - average_mesh: PyVista mesh with mean shape
                - eigenvalues: Numpy array of PCA eigenvalues
                - eigenvectors: Numpy array of PCA eigenvector components
                    Shape: (modes, n_points*3)

        Raises:
            FileNotFoundError: If JSON or VTK mesh file not found
            KeyError: If group_key not found in JSON
            ValueError: If data format is invalid

        Example:
            >>> mesh, eigenvalues, eigenvectors = (
            ...     RegisterModelToImagePCA.pca_read_slicersalt(
            ...         'path/to/pca.json',
            ...         group_key='All'
            ...     )
            ... )
            >>> std_deviations = np.sqrt(eigenvalues)
            >>> registrar = RegisterModelToImagePCA(
            ...     average_mesh=mesh,
            ...     eigenvectors=eigenvectors,
            ...     std_deviations=std_deviations,
            ...     reference_image=patient_image
            ... )
        """
        # Create a logger for the classmethod since superclassclasss hasn'tt
        #      been initialized yet.
        logger = logging.getLogger("PhysioMotion4D")

        json_path = Path(json_filename)

        # Check if JSON file exists
        if not json_path.exists():
            self.log_error(f"PCA JSON file not found: {json_filename}")
            raise FileNotFoundError(f"PCA JSON file not found: {json_filename}")

        logger.info("Loading PCA data from SlicerSALT format...")
        logger.info(f"  JSON file: {json_path}")
        logger.info(f"  Group key: {group_key}")

        # Load PCA data from JSON
        logger.info("Reading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            pca_data = json.load(f)

        # Extract PCA group data
        if group_key not in pca_data:
            available_keys = list(pca_data.keys())
            raise KeyError(
                f"Group key '{group_key}' not found in JSON. "
                f"Available keys: {available_keys}"
            )

        group_data = pca_data[group_key]

        # Extract eigenvalues
        if 'eigenvalues' not in group_data:
            raise ValueError(
                f"'eigenvalues' field not found in group '{group_key}' data"
            )
        eigenvalues = np.array(group_data['eigenvalues'])
        logger.info("  Loaded %d eigenvalues", len(eigenvalues))

        std_deviations = np.sqrt(eigenvalues)

        # Extract eigenvector components
        if 'components' not in group_data:
            raise ValueError(
                f"'components' field not found in group '{group_key}' data"
            )
        eigenvectors = np.array(group_data['components'], dtype=np.float64)
        logger.info(f"  Loaded eigenvectors with shape {eigenvectors.shape}")

        expected_eigenvector_size = average_mesh.n_points * 3
        actual_eigenvector_size = eigenvectors.shape[1]

        if actual_eigenvector_size != expected_eigenvector_size:
            raise ValueError(
                f"Eigenvector dimension mismatch: "
                f"Expected {expected_eigenvector_size} (3 × {average_mesh.n_points} mesh points), "
                f"got {actual_eigenvector_size}"
            )

        logger.info("  ✓ Data validation successful!")
        logger.info("SlicerSALT PCA data loaded successfully!")

        return cls(
            average_mesh=average_mesh,
            eigenvectors=eigenvectors,
            std_deviations=std_deviations,
            reference_image=reference_image,
            n_modes=n_modes,
            log_level=log_level,
        )

    def _create_itk_points(self) -> None:
        """Pre-convert mean shape points to ITK Point format for efficiency.

        This method creates ITK Point objects once at initialization, avoiding
        repeated conversions during optimization iterations.
        """
        self.log_info("Converting mean shape points to ITK format...")

        self._average_mesh_points_itk = []
        for i in range(len(self._average_mesh_points)):
            itk_point = itk.Point[itk.D, 3]()
            itk_point[0] = float(self._average_mesh_points[i, 0])
            itk_point[1] = float(self._average_mesh_points[i, 1])
            itk_point[2] = float(self._average_mesh_points[i, 2])
            self._average_mesh_points_itk.append(itk_point)

        self.log_info(
            f"  Converted {len(self._average_mesh_points_itk)} points to ITK format"
        )

    def set_reference_image(self, reference_image: itk.Image) -> None:
        """Set the reference image for registration.

        Args:
            reference_image: ITK image providing coordinate frame and intensity data
        """
        self.reference_image = reference_image
        # Clear interpolator to force recreation with new image
        self._interpolator = None

    def set_average_mesh(self, average_mesh: pv.UnstructuredGrid) -> None:
        """Set the average mesh for registration.

        Args:
            average_mesh: PyVista mesh containing the mean 3D shape model
                (unstructured grid or polydata)
        """
        self.average_mesh = average_mesh

        self._kdtree = None
        self._average_mesh_points = self.average_mesh.points
        self._average_mesh_points_itk = None
        self._average_mesh_points_deformed = None

        self._create_itk_points()
        self.log_info("  ✓ Average mesh set successfully!")

    def set_transform_point_distance_threshold(self, distance_mm: float) -> None:
        """Set the distance threshold for transform_point method.

        Args:
            distance_mm: Distance threshold in millimeters. Points within this
                distance will be used for weighted averaging when transforming
                arbitrary points. Default: 10.0 mm
        """
        if distance_mm <= 0:
            raise ValueError("Distance threshold must be positive")
        self._transform_point_distance_threshold = distance_mm

    def _evaluate_intensity_metric(
        self,
        pca_deformation: Optional[np.ndarray] = None,
        transform_params: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate the optimization metric (mean intensity) at model points.

        This is the objective function to be MAXIMIZED during optimization.
        Higher values indicate better alignment with bright regions.

        Args:
            pca_deformation: Nx3 numpy array of PCA deformation vectors to add to points.
                If None, no deformation is applied.
            transform_params: 6-element array of rigid transform parameters.
                If None, no rigid transformation is applied.

        Returns:
            Mean intensity value across all points
        """
        # Create interpolator if not already cached (inline creation)
        if self._interpolator is None:
            if self.reference_image is None:
                self.log_error("Reference image is not set")
                raise ValueError(
                    "Reference image must be set before creating interpolator"
                )

            ImageType = type(self.reference_image)
            self._interpolator = itk.LinearInterpolateImageFunction[
                ImageType, itk.D
            ].New()
            self._interpolator.SetInputImage(self.reference_image)
            self.log_debug("   Interpolator created")

        # Update working transform if parameters provided
        if transform_params is not None:
            itk_params = itk.OptimizerParameters[itk.D](6)
            for i in range(6):
                itk_params[i] = transform_params[i]
            self._working_transform.SetParameters(itk_params)

        # Sample intensities at each point
        n_valid_points = 0
        n_invalid_points = 0
        total_intensity = 0.0
        center = np.zeros(3)
        image_size = self.reference_image.GetBufferedRegion().GetSize()
        for i, base_point in enumerate(self._average_mesh_points_itk):
            if i % self.point_subsample_step != 0:
                continue

            # Start with base point
            point = itk.Point[itk.D, 3]()
            point[0] = base_point[0]
            point[1] = base_point[1]
            point[2] = base_point[2]

            # Add PCA deformation if provided
            if pca_deformation is not None:
                point[0] += pca_deformation[i, 0]
                point[1] += pca_deformation[i, 1]
                point[2] += pca_deformation[i, 2]

            # Apply rigid transform if parameters provided
            if transform_params is not None:
                point = self._working_transform.TransformPoint(point)

            # Check if point is inside image bounds
            coord_index = self.reference_image.TransformPhysicalPointToContinuousIndex(
                point
            )
            if (
                0 <= coord_index[0] < image_size[0]
                and 0 <= coord_index[1] < image_size[1]
                and 0 <= coord_index[2] < image_size[2]
            ):
                intensity = self._interpolator.EvaluateAtContinuousIndex(coord_index)
                total_intensity += intensity
                center[0] += point[0]
                center[1] += point[1]
                center[2] += point[2]
                n_valid_points += 1
            else:
                # Point is outside image bounds, skip
                n_invalid_points += 1
                continue

        # Compute mean intensity
        if n_valid_points > 0:
            mean_intensity = total_intensity / n_valid_points
            center /= n_valid_points
        else:
            mean_intensity = 0.0
            self.log_warning("   No valid points found")

        if n_invalid_points > 0:
            self.log_warning("   %d points are outside image bounds", n_invalid_points)
            self.log_warning("   Parameters: %s", transform_params)
            self.log_warning("   Center: %s", center)
            self.log_warning("   Mean intensity: %f", mean_intensity)

        if self.log_level <= logging.DEBUG or self._metric_call_count % 100 == 0:
            self.log_info(
                "   Metric %d: %s -> %f",
                (self._metric_call_count + 1),
                center,
                mean_intensity,
            )
        self._metric_call_count += 1

        return mean_intensity

    def _compute_pca_deformation(
        self, pca_coefficients: np.ndarray, n_pca_modes: Optional[int] = None
    ) -> np.ndarray:
        """Compute PCA deformation vectors for all points.

        Deformation is computed as:
            displacement = Σ(b_i * std_i * eigenvector_i)

        Args:
            pca_coefficients: Array of PCA coefficients b_i (one per mode)
            n_pca_modes: Number of PCA modes to use. Default: use all available modes

        Returns:
            Nx3 array of deformation vectors (displacement from mean shape)
        """
        if n_pca_modes is None:
            n_pca_modes = len(pca_coefficients)

        if n_pca_modes > len(pca_coefficients):
            raise ValueError(
                f"Number of PCA modes to use ({n_pca_modes}) exceeds available modes ({self.n_pca_modes})"
            )

        # Initialize deformation to zero
        deformation = np.zeros((self.average_mesh.n_points, 3), dtype=np.float64)

        # Add contribution from each PCA mode
        for i in range(n_pca_modes):
            # Get eigenvector for this mode (flattened: [x1,y1,z1, x2,y2,z2, ...])
            eigenvector_flat = self.eigenvectors[i, :]

            # Reshape to (N, 3)
            eigenvector_3d = eigenvector_flat.reshape(-1, 3)

            # Add weighted deformation: b_i * std_i * eigenvector_i
            deformation += pca_coefficients[i] * self.std_deviations[i] * eigenvector_3d

        return deformation

    def _rigid_objective_function(self, params: np.ndarray) -> float:
        """Objective function for coarse rigid alignment optimization (Stage 1).

        This function is MINIMIZED by the optimizer, so we return negative mean intensity.

        Args:
            params: 6-element array of transform parameters
                - First 3: Versor rotation (rotation vector)
                - Last 3: Translation

        Returns:
            Negative mean intensity (to be minimized)
        """
        # Evaluate intensity metric (no PCA deformation in Stage 1)
        mean_intensity = self._evaluate_intensity_metric(
            pca_deformation=None, transform_params=params
        )

        # Return negative (optimizer minimizes, we want to maximize intensity)
        return -mean_intensity

    def optimize_rigid_alignment(
        self,
        initial_transform: itk.VersorRigid3DTransform,
        method: str = 'Nelder-Mead',
        max_iterations: int = 500,
    ) -> tuple[itk.VersorRigid3DTransform, float]:
        """Optimize coarse rigid alignment (Stage 1) to maximize mean intensity.

        This method optimizes 6 parameters (versor rotation + translation)
        to align the mean shape model with bright regions in the image.

        Args:
            initial_transform: Initial ITK VersorRigid3DTransform for starting point
            method: Optimization method for scipy.optimize.minimize.
                Default: 'Nelder-Mead'
            max_iterations: Maximum number of optimization iterations.
                Default: 500

        Returns:
            Tuple of (transform, mean_intensity):
                - transform: Optimized ITK VersorRigid3DTransform
                - mean_intensity: Final mean intensity metric value

        Raises:
            ValueError: If reference image is not set
        """
        if self.reference_image is None:
            raise ValueError("Reference image must be set before optimization")

        self.log_section("Stage 1: Coarse Rigid Alignment Optimization", width=60)

        # Get initial parameters from transform
        itk_params = initial_transform.GetParameters()
        initial_params = np.array([itk_params[i] for i in range(len(itk_params))])

        self.log_info(f"Initial parameters: {initial_params}")
        self.log_info(f"Optimization method: {method}")
        self.log_info(f"Max iterations: {max_iterations}")

        # Run optimization
        self.log_info("Running optimization...")
        if self.log_level <= logging.INFO:
            disp = True
        else:
            disp = False

        result_rigid = minimize(
            self._rigid_objective_function,
            initial_params,
            method=method,
            options={'maxiter': max_iterations, 'disp': disp},
        )
        self.log_info(f"Optimization result: {result_rigid.x} -> {result_rigid.fun}")

        # Create optimized transform
        optimized_transform = itk.VersorRigid3DTransform[itk.D].New()
        opt_itk_params = itk.OptimizerParameters[itk.D](6)
        for i in range(6):
            opt_itk_params[i] = result_rigid.x[i]
        optimized_transform.SetParameters(opt_itk_params)

        final_mean_intensity = -result_rigid.fun  # Convert back from negative

        self.log_info("Stage 1 optimization completed!")
        self.log_info(f"Final parameters: {result_rigid.x}")
        self.log_info(f"Final mean intensity: {final_mean_intensity:.2f}")

        # Store Stage 1 result
        self.stage1_rigid_transform = optimized_transform

        return optimized_transform, final_mean_intensity

    def _joint_objective_function(self, params: np.ndarray, n_pca_modes: int) -> float:
        """Objective function for joint rigid + PCA optimization (Stage 2).

        This function is MINIMIZED by the optimizer, so we return negative mean intensity.
        The first 6 parameters are rigid transformation (versor + translation),
        followed by n_pca_modes PCA coefficients.

        Args:
            params: (6 + n_pca_modes)-element array [v1, v2, v3, tx, ty, tz, b1, b2, ..., bn]
                - v1, v2, v3: Versor rotation parameters
                - tx, ty, tz: Translation in physical units
                - b1, ..., bn: PCA coefficients (in units of std deviations)
            n_pca_modes: Number of PCA modes being optimized

        Returns:
            Negative mean intensity (to be minimized)
        """
        # Extract rigid parameters
        rigid_params = params[:6]

        # Extract PCA coefficients and compute deformation
        pca_coefficients = params[6:]
        pca_deformation = self._compute_pca_deformation(
            pca_coefficients, n_pca_modes=n_pca_modes
        )

        # Evaluate intensity metric
        mean_intensity = self._evaluate_intensity_metric(
            pca_deformation=pca_deformation,
            transform_params=rigid_params,
        )

        # Return negative (optimizer minimizes, we want to maximize intensity)
        return -mean_intensity

    def optimize_joint_rigid_and_pca(
        self,
        initial_transform: itk.VersorRigid3DTransform,
        n_pca_modes: int = -1,
        method: str = 'L-BFGS-B',
        pca_coefficient_bounds: float = 3.0,
        rigid_refinement_bounds: Optional[dict[str, float]] = None,
        max_iterations: int = 50,
    ) -> tuple[itk.VersorRigid3DTransform, np.ndarray, float]:
        """Optimize joint rigid parameters + PCA coefficients (Stage 2).

        This method simultaneously optimizes rigid transformation refinements
        and PCA mode coefficients to deform the model to better match bright
        regions in the image. The rigid parameters from Stage 1 are used as
        the initial guess, and bounds constrain them to small refinements.

        Args:
            n_pca_modes: Number of PCA modes to use in optimization. Using fewer
                modes provides smoother deformations. Default: 10
            method: Optimization method for scipy.optimize.minimize.
                Default: 'L-BFGS-B' (supports bounds)
            initial_transform: Initial ITK VersorRigid3DTransform for starting point
            pca_coefficient_bounds: Bound on PCA coefficients in units of std deviations.
                Default: 3.0 (±3 std deviations per mode)
            rigid_refinement_bounds: Dictionary specifying bounds on rigid parameter
                refinements from Stage 1 initial values:
                    - 'versor': Max change in versor parameters (default: 0.2)
                    - 'translation_mm': Max translation change in mm (default: 20mm)
                If None, uses defaults.
            max_iterations: Maximum number of optimization iterations.
                Default: 50

        Returns:
            Tuple of (transform, pca_coefficients, mean_intensity):
                - transform: Final optimized ITK VersorRigid3DTransform
                - pca_coefficients: Optimized PCA coefficients
                - mean_intensity: Final mean intensity metric value

        Raises:
            ValueError: If Stage 1 rigid alignment has not been performed
        """
        self.log_section("Stage 2: Joint Rigid + PCA Deformable Registration", width=60)

        if n_pca_modes == -1:
            n_pca_modes = len(self.eigenvectors)
        if n_pca_modes > len(self.eigenvectors):
            raise ValueError(
                f"Number of PCA modes to use ({n_pca_modes}) exceeds available modes ({len(self.std_deviations)})"
            )
        self.n_pca_modes = n_pca_modes

        # Set default rigid refinement bounds if not provided
        if rigid_refinement_bounds is None:
            rigid_refinement_bounds = {
                'versor': 0.2,  # Max change in versor parameters
                'translation_mm': 20.0,  # ±10 mm
            }

        versor_bound = rigid_refinement_bounds['versor']
        translation_bound_mm = rigid_refinement_bounds['translation_mm']

        self.log_info(f"Number of PCA modes: {n_pca_modes}")
        self.log_info(
            f"PCA coefficient bounds: ±{pca_coefficient_bounds} std deviations"
        )
        self.log_info(f"Rigid versor refinement bounds: ±{versor_bound}")
        self.log_info(
            f"Rigid translation refinement bounds: ±{translation_bound_mm} mm"
        )
        self.log_info(f"Optimization method: {method}")
        self.log_info(f"Max iterations: {max_iterations}")

        # Get Stage 1 rigid parameters
        itk_params = initial_transform.GetParameters()
        initial_rigid_params = np.array([itk_params[i] for i in range(6)])

        # Set initial parameters: Start from Stage 1 rigid + zero PCA coefficients
        initial_params = np.concatenate(
            [
                initial_rigid_params,
                np.zeros(n_pca_modes),
            ]  # Start with mean shape (no deformation)
        )

        # Set bounds: constrained rigid refinement + PCA coefficient bounds
        bounds = []

        # Versor rotation bounds (first 3 parameters - constrained around Stage 1 values)
        for v_rigid in initial_rigid_params[:3]:
            bounds.append((v_rigid - versor_bound, v_rigid + versor_bound))

        # Rigid translation bounds (last 3 rigid parameters - constrained around Stage 1 values)
        for trans_rigid in initial_rigid_params[3:6]:
            bounds.append(
                (
                    trans_rigid - translation_bound_mm,
                    trans_rigid + translation_bound_mm,
                )
            )

        # PCA coefficient bounds (±3 std deviations typically)
        for _ in range(n_pca_modes):
            bounds.append((-pca_coefficient_bounds, pca_coefficient_bounds))

        # Run optimization
        self.log_info("Running joint optimization...")
        result_joint = minimize(
            lambda params: self._joint_objective_function(params, n_pca_modes),
            initial_params,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iterations, 'disp': False},
        )

        # Create optimized transform
        optimized_rigid_params = result_joint.x[:6]
        optimized_transform = itk.VersorRigid3DTransform[itk.D].New()
        opt_itk_params = itk.OptimizerParameters[itk.D](6)
        for i in range(6):
            opt_itk_params[i] = optimized_rigid_params[i]
        optimized_transform.SetParameters(opt_itk_params)

        optimized_pca_coefficients = result_joint.x[6:]

        final_mean_intensity = -result_joint.fun  # Convert back from negative

        # Compute changes from Stage 1
        param_change = optimized_rigid_params - initial_rigid_params

        self.log_info("Stage 2 optimization completed!")
        self.log_info(f"Final rigid parameters: {optimized_rigid_params}")
        self.log_info("Rigid refinement from initial parameters:")
        self.log_info(f"  Versor change: {param_change[:3]}")
        self.log_info(f"  Translation change (mm): {param_change[3:6]}")
        self.log_info(f"Optimized PCA coefficients: {optimized_pca_coefficients}")
        self.log_info(f"Final mean intensity: {final_mean_intensity:.2f}")

        # Store final results
        self.rigid_transform = optimized_transform
        self.pca_coefficients = optimized_pca_coefficients
        self.final_intensity = final_mean_intensity

        return optimized_transform, optimized_pca_coefficients, final_mean_intensity

    def transform_mesh(self) -> pv.UnstructuredGrid:
        """Create the final registered mesh by applying rigid + PCA transformations.

        This method combines the rigid transformation and PCA deformation to
        create the final registered mesh with all point data and cell data
        preserved from the original average mesh.

        Returns:
            Final registered and deformed mesh as PyVista UnstructuredGrid

        Raises:
            ValueError: If registration has not been performed
        """
        if self.rigid_transform is None or self.pca_coefficients is None:
            self.log_error("Must complete registration before creating registered mesh")
            raise ValueError(
                "Must complete registration before creating registered mesh"
            )

        self.log_info("Creating final registered mesh...")

        # Compute PCA deformation
        pca_deformation = self._compute_pca_deformation(
            self.pca_coefficients,
            n_pca_modes=self.n_pca_modes,
        )

        # Apply deformation and rigid transform to each point
        final_points = np.zeros((self.average_mesh.n_points, 3), dtype=np.float64)

        n_points = self.average_mesh.n_points
        progress_interval = max(1, n_points // 10)  # Report progress every 10%

        for i in range(n_points):
            # Report progress
            if i % progress_interval == 0 or i == n_points - 1:
                self.log_progress(i + 1, n_points, prefix="Transforming points")

            # Start with mean shape point
            point = itk.Point[itk.D, 3]()
            point[0] = self._average_mesh_points[i][0]
            point[1] = self._average_mesh_points[i][1]
            point[2] = self._average_mesh_points[i][2]

            # Add PCA deformation
            point[0] += pca_deformation[i, 0]
            point[1] += pca_deformation[i, 1]
            point[2] += pca_deformation[i, 2]

            # Apply rigid transform
            transformed_point = self.rigid_transform.TransformPoint(point)

            # Store result
            final_points[i, 0] = transformed_point[0]
            final_points[i, 1] = transformed_point[1]
            final_points[i, 2] = transformed_point[2]

        # Create new mesh with transformed points
        self.registered_mesh = self.average_mesh.copy(deep=True)
        self.registered_mesh.points = final_points.copy()

        # Store deformed points for transform_point method
        self._average_mesh_points_deformed = final_points.copy()

        # Build KDTree from mean points for efficient nearest neighbor search
        self._kdtree = KDTree(self._average_mesh_points)

        self.log_info(
            f"Registered mesh created with {self.registered_mesh.n_points} points"
        )

        return self.registered_mesh

    def transform_point(
        self, point: itk.Point, distance_threshold: Optional[float] = None
    ) -> itk.Point:
        """Transform an arbitrary point using distance-weighted interpolation.

        Finds all mesh points within a specified distance threshold and applies
        a distance-weighted average of their displacements to transform the input
        point.

        Args:
            point: ITK point to transform (itk.Point[itk.D, 3])
            distance_threshold: Distance threshold in millimeters. If None, uses
                the value set by set_transform_point_distance_threshold()
                (default: 10.0 mm)

        Returns:
            Transformed ITK point

        Raises:
            ValueError: If registration has not been completed yet, or if no
                points are found within the distance threshold

        Example:
            >>> p = itk.Point[itk.D, 3]()
            >>> p[0], p[1], p[2] = 10.0, 20.0, 30.0
            >>> # Use default distance threshold
            >>> transformed_p = registrar.transform_point(p)
            >>> # Or specify a custom distance threshold
            >>> transformed_p = registrar.transform_point(p, distance_threshold=15.0)
        """
        if (
            self._kdtree is None
            or self._average_mesh_points is None
            or self._average_mesh_points_deformed is None
        ):
            self.log_error(
                "Must complete registration and create registered mesh before "
                "calling transform_point(). Call transform_mesh() first."
            )
            raise ValueError(
                "Must complete registration and create registered mesh before "
                "calling transform_point(). Call transform_mesh() first."
            )

        # Use provided distance threshold or default
        if distance_threshold is None:
            distance_threshold = self._transform_point_distance_threshold
        else:
            self._transform_point_distance_threshold = distance_threshold

        # Convert ITK point to numpy array for distance calculations
        point_coords = np.array([float(point[0]), float(point[1]), float(point[2])])

        # Query KDTree for all points within distance threshold
        nn_indices = self._kdtree.query_ball_point(point_coords, r=distance_threshold)

        # Check if any points were found
        if len(nn_indices) == 0:
            raise ValueError(
                f"No mesh points found within distance threshold of {distance_threshold} mm. "
                f"Consider increasing the threshold using set_transform_point_distance_threshold() "
                f"or passing a larger distance_threshold parameter."
            )

        # Compute distances for these points
        nn_distances = np.linalg.norm(
            self._average_mesh_points[nn_indices] - point_coords, axis=1
        )

        # Compute distance weights (inverse distance weighting)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        weights = 1.0 / (nn_distances + epsilon)
        weights = weights / weights.sum()  # Normalize to sum to 1

        # Compute weighted average displacement
        weighted_displacement = np.zeros(3)
        for i, idx in enumerate(nn_indices):
            # Displacement is: deformed_point - mean_point
            displacement = (
                self._average_mesh_points_deformed[idx] - self._average_mesh_points[idx]
            )
            weighted_displacement += weights[i] * displacement

        # Apply displacement to input point
        transformed_point = itk.Point[itk.D, 3]()
        transformed_point[0] = point[0] + weighted_displacement[0]
        transformed_point[1] = point[1] + weighted_displacement[1]
        transformed_point[2] = point[2] + weighted_displacement[2]

        return transformed_point

    def register(
        self,
        initial_transform: itk.VersorRigid3DTransform = None,
        n_pca_modes: int = -1,
        stage1_max_iterations: int = 500,
        stage2_max_iterations: int = 50,
        pca_coefficient_bounds: float = 3.0,
        rigid_refinement_bounds: Optional[dict[str, float]] = None,
    ) -> dict:
        """Execute the complete two-stage registration workflow.

        This method runs both Stage 1 (coarse rigid alignment) and Stage 2
        (joint rigid refinement + PCA deformable registration) in sequence.

        Args:
            initial_transform: Initial ITK VersorRigid3DTransform for starting point
            n_pca_modes: Number of PCA modes to use. Default: 10
            stage1_max_iterations: Max iterations for Stage 1 rigid. Default: 500
            stage2_max_iterations: Max iterations for Stage 2 joint. Default: 50
            pca_coefficient_bounds: PCA coefficient bounds (±std devs). Default: 3.0
            rigid_refinement_bounds: Dictionary with 'versor' and 'translation_mm'
                bounds for Stage 2 rigid refinement.
                Default: {'versor': 0.2, 'translation_mm': 20}

        Returns:
            Dictionary containing:
                - 'registered_mesh': Final registered PyVista mesh
                - 'stage1_transform': Stage 1 ITK VersorRigid3DTransform
                - 'rigid_transform': Final ITK VersorRigid3DTransform
                - 'pca_coefficients': Optimized PCA coefficients
                - 'final_intensity': Final mean intensity metric value

        Raises:
            ValueError: If reference image is not set

        Example:
            >>> initial_tfm = itk.VersorRigid3DTransform[itk.D].New()
            >>> initial_tfm.SetIdentity()
            >>> result = registrar.register(
            ...     initial_transform=initial_tfm,
            ...     n_pca_modes=10
            ... )
            >>> result['registered_mesh'].save("registered_heart.vtk")
        """
        if self.reference_image is None:
            raise ValueError("Reference image must be set before registration")

        if initial_transform is None:
            initial_transform = itk.VersorRigid3DTransform[itk.D].New()
            initial_transform.SetIdentity()

        if n_pca_modes == -1:
            n_pca_modes = self.n_pca_modes

        self.log_section("PCA-BASED MODEL-TO-IMAGE REGISTRATION", width=70)
        self.log_info(f"Number of points: {self.average_mesh.n_points}")
        self.log_info(f"Modes to use: {n_pca_modes}")

        # Stage 1: Coarse rigid alignment
        stage1_rigid_transform, stage1_intensity = self.optimize_rigid_alignment(
            initial_transform=initial_transform, max_iterations=stage1_max_iterations
        )

        # Stage 2: Joint rigid + PCA optimization
        final_rigid_transform, final_pca_coefficients, final_intensity = (
            self.optimize_joint_rigid_and_pca(
                initial_transform=stage1_rigid_transform,
                n_pca_modes=n_pca_modes,
                max_iterations=stage2_max_iterations,
                pca_coefficient_bounds=pca_coefficient_bounds,
                rigid_refinement_bounds=rigid_refinement_bounds,
            )
        )

        # Create final registered mesh
        final_registered_mesh = self.transform_mesh()

        self.log_section("REGISTRATION COMPLETE", width=70)
        self.log_info(f"Stage 1 intensity (coarse rigid): {stage1_intensity:.2f}")
        self.log_info(f"Stage 2 intensity (rigid+PCA): {final_intensity:.2f}")
        intensity_improvement = final_intensity - stage1_intensity
        self.log_info(f"Overall intensity improvement: {intensity_improvement:.2f}")

        # Return results as dictionary
        return {
            'registered_mesh': final_registered_mesh,
            'pre_phi_FM': final_rigid_transform,
            'pca_coefficients_FM': final_pca_coefficients,
            'intensity': final_intensity,
        }


# Example usage
if __name__ == "__main__":
    """Example demonstrating PCA-based model-to-image registration."""

    # This example shows how to use the RegisterModelToImagePCA class
    # to register a statistical shape model to a patient-specific CT image.

    import json

    import itk
    import numpy as np
    import pyvista as pv

    # =========================================================================
    # Setup: Load patient image
    # =========================================================================
    print("Loading patient image...")
    patient_image = itk.imread("patient_cardiac_ct.nrrd")
    print(f"  Image size: {patient_image.GetBufferedRegion().GetSize()}")
    print(f"  Image spacing: {patient_image.GetSpacing()}")

    # =========================================================================
    # Load PCA model data
    # =========================================================================
    print("\nLoading PCA model data...")

    # Load average mesh
    average_mesh = pv.read("pca_All_mean.vtk")
    print(f"  Loaded mesh with {average_mesh.n_points} points")

    # Load PCA data from JSON
    with open("pca.json", 'r') as f:
        pca_data = json.load(f)
    group_data = pca_data['All']

    # Extract eigenvalues and convert to standard deviations
    eigenvalues = np.array(group_data['eigenvalues'])
    std_deviations = np.sqrt(eigenvalues)
    print(f"  Loaded {len(std_deviations)} eigenvalues")

    # Extract eigenvector components
    eigenvectors = np.array(group_data['components'], dtype=np.float64)
    print(f"  Loaded eigenvectors with shape {eigenvectors.shape}")

    # =========================================================================
    # Initialize registration with loaded data
    # =========================================================================
    print("\nInitializing PCA-based registration...")
    registrar = RegisterModelToImagePCA(
        average_mesh=average_mesh,
        eigenvectors=eigenvectors,
        std_deviations=std_deviations,
        reference_image=patient_image,
    )

    # =========================================================================
    # Create initial transform (user-provided)
    # =========================================================================
    print("\nCreating initial transform...")
    initial_transform = itk.VersorRigid3DTransform[itk.D].New()
    initial_transform.SetIdentity()

    # User can set initial rotation and translation here
    # For example:
    # params = itk.OptimizerParameters[itk.D](6)
    # params[0] = 0.1  # Versor component 1
    # params[1] = 0.0  # Versor component 2
    # params[2] = 0.0  # Versor component 3
    # params[3] = 10.0  # Translation X
    # params[4] = 0.0   # Translation Y
    # params[5] = 0.0   # Translation Z
    # initial_transform.SetParameters(params)

    # =========================================================================
    # Run complete registration (Stage 1: coarse rigid, Stage 2: joint rigid+PCA)
    # =========================================================================
    print("\nRunning complete registration...")
    result = registrar.register(
        initial_transform=initial_transform,
        n_pca_modes=10,  # Use first 10 PCA modes
        stage1_max_iterations=500,  # Max iterations for Stage 1 (coarse rigid)
        stage2_max_iterations=50,  # Max iterations for Stage 2 (joint)
        pca_coefficient_bounds=3.0,  # Limit PCA coeffs to ±3 std deviations
        rigid_refinement_bounds={  # Small refinements to rigid in Stage 2
            'versor': 0.2,  # ±0.2 in versor parameters
            'translation_mm': 20.0,  # ±20 mm
        },
    )

    # =========================================================================
    # Access and save results
    # =========================================================================
    print("\nSaving results...")

    # Save registered mesh
    registered_mesh = result['registered_mesh']
    registered_mesh.save("registered_heart_pca.vtk")
    print(f"  Saved registered mesh: registered_heart_pca.vtk")

    # Print Stage 1 rigid transformation
    stage1_itk_params = result['stage1_transform'].GetParameters()
    stage1_params = np.array([stage1_itk_params[i] for i in range(6)])
    print("\nStage 1 (coarse rigid) transformation parameters:")
    print(f"  {stage1_params}")

    # Print final rigid transformation
    final_itk_params = result['final_rigid_transform'].GetParameters()
    final_params = np.array([final_itk_params[i] for i in range(6)])
    print("\nFinal rigid transformation parameters (after Stage 2 refinement):")
    print(f"  {final_params}")

    # Print PCA coefficients
    print("\nPCA coefficients (in units of std deviations):")
    print(f"  {result['final_pca_coefficients']}")

    # Print final metric
    print(f"\nFinal mean intensity: {result['final_intensity']:.2f}")

    print("\nRegistration complete!")
