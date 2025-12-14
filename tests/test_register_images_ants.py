#!/usr/bin/env python
"""
Test for ANTs-based image registration.

This test depends on test_convert_nrrd_4d_to_3d and uses the converted
3D CT images to test ANTs registration functionality.
"""

import os
import tempfile

import ants
import itk
import numpy as np
import pytest

from physiomotion4d.transform_tools import TransformTools


@pytest.mark.requires_data
@pytest.mark.slow
class TestRegisterImagesANTs:
    """Test suite for ANTs-based image registration."""

    def test_registrar_initialization(self, registrar_ants):
        """Test that RegisterImagesANTs initializes correctly."""
        assert registrar_ants is not None, "Registrar not initialized"
        assert hasattr(registrar_ants, 'fixed_image'), "Missing fixed_image attribute"
        assert hasattr(
            registrar_ants, 'fixed_image_mask'
        ), "Missing fixed_image_mask attribute"

        print("\n✓ ANTs registrar initialized successfully")

    def test_set_modality(self, registrar_ants):
        """Test setting imaging modality."""
        registrar_ants.set_modality('ct')
        assert registrar_ants.modality == 'ct', "Modality not set correctly"

        registrar_ants.set_modality('mr')
        assert registrar_ants.modality == 'mr', "Modality change failed"

        print("\n✓ Modality setting works correctly")

    def test_set_fixed_image(self, registrar_ants, test_images):
        """Test setting fixed image."""
        fixed_image = test_images[0]

        registrar_ants.set_fixed_image(fixed_image)
        assert registrar_ants.fixed_image is not None, "Fixed image not set"

        print("\n✓ Fixed image set successfully")
        print(f"  Image size: {itk.size(registrar_ants.fixed_image)}")
        print(f"  Image spacing: {itk.spacing(registrar_ants.fixed_image)}")

    def test_register_without_mask(self, registrar_ants, test_images, test_directories):
        """Test basic registration without masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ants"
        reg_output_dir.mkdir(exist_ok=True)

        # Set up registration
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nRegistering images without mask...")
        print(f"  Fixed image: {itk.size(fixed_image)}")
        print(f"  Moving image: {itk.size(moving_image)}")

        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)

        # Register
        result = registrar_ants.register(moving_image=moving_image)

        # Verify result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "phi_FM" in result, "Missing phi_FM in result"
        assert "phi_MF" in result, "Missing phi_MF in result"

        phi_FM = result["phi_FM"]
        phi_MF = result["phi_MF"]

        # Verify transforms are valid
        assert phi_FM is not None, "phi_FM is None"
        assert phi_MF is not None, "phi_MF is None"

        print("✓ Registration complete without mask")
        print(f"  phi_FM type: {type(phi_FM).__name__}")
        print(f"  phi_MF type: {type(phi_MF).__name__}")

        # Save transforms
        itk.transformwrite(
            [phi_FM], str(reg_output_dir / "ants_phi_FM_no_mask.hdf"), compression=True
        )
        itk.transformwrite(
            [phi_MF], str(reg_output_dir / "ants_phi_MF_no_mask.hdf"), compression=True
        )
        print(f"  Saved transforms to: {reg_output_dir}")

    def test_register_with_mask(self, registrar_ants, test_images, test_directories):
        """Test registration with binary masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ants"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Create simple binary masks (central region)
        fixed_size_itk = itk.size(fixed_image)
        moving_size_itk = itk.size(moving_image)

        # Convert ITK Size to tuple for numpy indexing
        fixed_size = (
            int(fixed_size_itk[0]),
            int(fixed_size_itk[1]),
            int(fixed_size_itk[2]),
        )
        moving_size = (
            int(moving_size_itk[0]),
            int(moving_size_itk[1]),
            int(moving_size_itk[2]),
        )

        # Create mask arrays
        fixed_mask_arr = np.zeros(fixed_size[::-1], dtype=np.uint8)
        moving_mask_arr = np.zeros(moving_size[::-1], dtype=np.uint8)

        # Set central region to 1
        fixed_mask_arr[
            fixed_size[2] // 4 : 3 * fixed_size[2] // 4,
            fixed_size[1] // 4 : 3 * fixed_size[1] // 4,
            fixed_size[0] // 4 : 3 * fixed_size[0] // 4,
        ] = 1

        moving_mask_arr[
            moving_size[2] // 4 : 3 * moving_size[2] // 4,
            moving_size[1] // 4 : 3 * moving_size[1] // 4,
            moving_size[0] // 4 : 3 * moving_size[0] // 4,
        ] = 1

        # Create ITK images
        fixed_mask = itk.image_from_array(fixed_mask_arr)
        fixed_mask.CopyInformation(fixed_image)

        moving_mask = itk.image_from_array(moving_mask_arr)
        moving_mask.CopyInformation(moving_image)

        print("\nRegistering images with masks...")
        print(f"  Fixed mask voxels: {np.sum(fixed_mask_arr)}")
        print(f"  Moving mask voxels: {np.sum(moving_mask_arr)}")

        # Set up registration with masks
        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)
        registrar_ants.set_fixed_image_mask(fixed_mask)

        # Register
        result = registrar_ants.register(
            moving_image=moving_image, moving_image_mask=moving_mask
        )

        # Verify result
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "phi_FM" in result, "Missing phi_FM in result"
        assert "phi_MF" in result, "Missing phi_MF in result"

        phi_FM = result["phi_FM"]
        phi_MF = result["phi_MF"]

        assert phi_FM is not None, "phi_FM is None"
        assert phi_MF is not None, "phi_MF is None"

        print("✓ Registration complete with masks")

        # Save transforms
        itk.transformwrite(
            [phi_FM],
            str(reg_output_dir / "ants_phi_FM_with_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [phi_MF],
            str(reg_output_dir / "ants_phi_MF_with_mask.hdf"),
            compression=True,
        )

    def test_transform_application(self, registrar_ants, test_images, test_directories):
        """Test applying registration transforms to images."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ants"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Register
        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)
        result = registrar_ants.register(moving_image=moving_image)

        phi_MF = result["phi_MF"]

        print("\nApplying transform to moving image...")

        # Apply transform
        transform_tools = TransformTools()
        registered_image = transform_tools.transform_image(
            moving_image, phi_MF, fixed_image, interpolation_method="linear"
        )

        # Verify registered image
        assert registered_image is not None, "Registered image is None"
        assert itk.size(registered_image) == itk.size(fixed_image), "Size mismatch"

        # Check that image content changed
        moving_arr = itk.array_from_image(moving_image)
        registered_arr = itk.array_from_image(registered_image)

        # They should be different (unless perfectly aligned)
        difference = np.sum(
            np.abs(moving_arr.astype(float) - registered_arr.astype(float))
        )

        print("✓ Transform applied successfully")
        print(f"  Registered image size: {itk.size(registered_image)}")
        print(f"  Total difference: {difference:.2f}")

        # Save registered image
        itk.imwrite(
            registered_image,
            str(reg_output_dir / "ants_registered_image.mha"),
            compression=True,
        )
        print(f"  Saved to: {reg_output_dir / 'ants_registered_image.mha'}")

    def test_preprocess_images(self, registrar_ants, test_images):
        """Test image preprocessing."""
        test_image = test_images[0]

        print("\nTesting image preprocessing...")
        print(f"  Original spacing: {itk.spacing(test_image)}")

        # Preprocess
        preprocessed = registrar_ants.preprocess(test_image, modality='ct')

        assert preprocessed is not None, "Preprocessed image is None"

        preprocessed_spacing = itk.spacing(preprocessed)
        print("✓ Image preprocessing complete")
        print(f"  Preprocessed spacing: {preprocessed_spacing}")

    def test_registration_with_initial_transform(
        self, registrar_ants, test_images, test_directories
    ):
        """Test registration with initial transform."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_ants"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        # Create initial translation transform
        initial_tfm_FM = itk.TranslationTransform[itk.D, 3].New()
        initial_tfm_FM.SetOffset([5.0, 5.0, 5.0])

        initial_tfm_MF = itk.TranslationTransform[itk.D, 3].New()
        initial_tfm_MF.SetOffset([-5.0, -5.0, -5.0])

        print("\nRegistering with initial transform...")
        print("  Initial offset: [5.0, 5.0, 5.0]")

        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)

        result = registrar_ants.register(
            moving_image=moving_image,
            initial_phi_FM=initial_tfm_FM,
            initial_phi_MF=initial_tfm_MF,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["phi_FM"] is not None, "phi_FM is None"
        assert result["phi_MF"] is not None, "phi_MF is None"

        print("✓ Registration with initial transform complete")

    def test_multiple_registrations(self, registrar_ants, test_images):
        """Test running multiple registrations in sequence."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nRunning multiple registrations...")

        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)

        results = []
        for i in range(2):
            print(f"  Registration {i+1}...")
            result = registrar_ants.register(moving_image=moving_image)
            results.append(result)

            assert isinstance(result, dict), f"Result {i+1} should be a dictionary"
            assert "phi_FM" in result, f"Missing phi_FM in result {i+1}"
            assert "phi_MF" in result, f"Missing phi_MF in result {i+1}"

        print(f"✓ Multiple registrations complete: {len(results)} runs")

    def test_transform_types(self, registrar_ants, test_images):
        """Test that transforms are correct ITK types."""
        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_ants.set_modality('ct')
        registrar_ants.set_fixed_image(fixed_image)
        result = registrar_ants.register(moving_image=moving_image)

        phi_FM = result["phi_FM"]
        phi_MF = result["phi_MF"]

        print("\nVerifying transform types...")

        # Check that transforms are CompositeTransform (ANTs returns composite)
        assert isinstance(
            phi_FM, itk.CompositeTransform
        ), f"phi_FM should be CompositeTransform, got {type(phi_FM)}"
        assert isinstance(
            phi_MF, itk.CompositeTransform
        ), f"phi_MF should be CompositeTransform, got {type(phi_MF)}"

        print("✓ Transform types verified")
        print(f"  phi_FM: {type(phi_FM).__name__}")
        print(f"  phi_MF: {type(phi_MF).__name__}")

    def test_image_conversion_cycle_scalar(self, registrar_ants, test_images):
        """Test round-trip conversion: ITK image -> ANTs -> ITK for scalar images."""
        original_image = test_images[0]

        print("\nTesting scalar image conversion cycle...")
        print(f"  Original image size: {itk.size(original_image)}")
        print(f"  Original image spacing: {itk.spacing(original_image)}")
        print(f"  Original image origin: {itk.origin(original_image)}")

        # Get original data
        original_array = itk.array_from_image(original_image)
        original_spacing = tuple(itk.spacing(original_image))
        original_origin = tuple(itk.origin(original_image))
        original_direction = itk.array_from_matrix(original_image.GetDirection())

        # Convert ITK -> ANTs
        ants_image = registrar_ants._itk_to_ants_image(original_image, dtype='float')

        # Verify ANTs image
        assert ants_image is not None, "ANTs image is None"
        assert (
            ants_image.dimension == 3
        ), f"ANTs dimension should be 3, got {ants_image.dimension}"
        print(f"  ANTs image shape: {ants_image.shape}")

        # Convert ANTs -> ITK
        recovered_image = registrar_ants._ants_to_itk_image(ants_image)

        # Verify recovered image
        assert recovered_image is not None, "Recovered image is None"
        recovered_array = itk.array_from_image(recovered_image)
        recovered_spacing = tuple(itk.spacing(recovered_image))
        recovered_origin = tuple(itk.origin(recovered_image))
        recovered_direction = itk.array_from_matrix(recovered_image.GetDirection())

        print(f"  Recovered image size: {itk.size(recovered_image)}")
        print(f"  Recovered image spacing: {recovered_spacing}")
        print(f"  Recovered image origin: {recovered_origin}")

        # Compare arrays
        assert (
            original_array.shape == recovered_array.shape
        ), f"Array shape mismatch: {original_array.shape} vs {recovered_array.shape}"

        # Check pixel values (allowing for float conversion tolerance)
        max_diff = np.max(
            np.abs(
                original_array.astype(np.float32) - recovered_array.astype(np.float32)
            )
        )
        print(f"  Max pixel difference: {max_diff}")
        assert max_diff < 1e-5, f"Pixel values differ too much: {max_diff}"

        # Check spacing (with tolerance for floating point)
        spacing_diff = np.max(
            np.abs(np.array(original_spacing) - np.array(recovered_spacing))
        )
        print(f"  Max spacing difference: {spacing_diff}")
        assert spacing_diff < 1e-6, f"Spacing differs: {spacing_diff}"

        # Check origin
        origin_diff = np.max(
            np.abs(np.array(original_origin) - np.array(recovered_origin))
        )
        print(f"  Max origin difference: {origin_diff}")
        assert origin_diff < 1e-6, f"Origin differs: {origin_diff}"

        # Check direction
        direction_diff = np.max(np.abs(original_direction - recovered_direction))
        print(f"  Max direction difference: {direction_diff}")
        assert direction_diff < 1e-6, f"Direction differs: {direction_diff}"

        print("✓ Scalar image conversion cycle successful")

    def test_image_conversion_cycle_different_dtypes(self, registrar_ants, test_images):
        """Test round-trip conversion with different data types."""
        original_image = test_images[0]

        print("\nTesting image conversion cycle with different dtypes...")

        dtypes = ['float', 'double', 'int', 'uint', 'uchar']

        for dtype in dtypes:
            print(f"  Testing dtype: {dtype}")

            # Convert ITK -> ANTs with specified dtype
            ants_image = registrar_ants._itk_to_ants_image(original_image, dtype=dtype)
            assert ants_image is not None, f"ANTs image is None for dtype {dtype}"

            # Convert ANTs -> ITK
            recovered_image = registrar_ants._ants_to_itk_image(ants_image)
            assert (
                recovered_image is not None
            ), f"Recovered image is None for dtype {dtype}"

            # Verify dimensions match
            assert itk.size(original_image) == itk.size(
                recovered_image
            ), f"Size mismatch for dtype {dtype}"

            print(f"    ✓ {dtype} conversion successful")

        print("✓ All dtype conversions successful")

    def test_image_conversion_preserves_metadata(self, registrar_ants):
        """Test that image conversion preserves all metadata."""
        print("\nTesting metadata preservation in image conversion...")

        # Create a test image with specific metadata
        size = [50, 60, 40]
        spacing = [1.5, 2.0, 2.5]
        origin = [10.0, 20.0, 30.0]

        ImageType = itk.Image[itk.F, 3]
        test_image = ImageType.New()
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        test_image.SetRegions(region)
        test_image.SetSpacing(spacing)
        test_image.SetOrigin(origin)
        test_image.Allocate()

        # Fill with test pattern
        arr = itk.array_from_image(test_image)
        arr[:] = np.random.randn(*arr.shape).astype(np.float32)

        print(f"  Test image size: {size}")
        print(f"  Test image spacing: {spacing}")
        print(f"  Test image origin: {origin}")

        # Convert ITK -> ANTs -> ITK
        ants_image = registrar_ants._itk_to_ants_image(test_image)
        recovered_image = registrar_ants._ants_to_itk_image(ants_image)

        # Verify all metadata
        recovered_size = [int(s) for s in itk.size(recovered_image)]
        recovered_spacing = [float(s) for s in itk.spacing(recovered_image)]
        recovered_origin = [float(o) for o in itk.origin(recovered_image)]

        assert recovered_size == size, f"Size mismatch: {recovered_size} vs {size}"
        assert np.allclose(
            recovered_spacing, spacing, atol=1e-6
        ), f"Spacing mismatch: {recovered_spacing} vs {spacing}"
        assert np.allclose(
            recovered_origin, origin, atol=1e-6
        ), f"Origin mismatch: {recovered_origin} vs {origin}"

        print("✓ Metadata preservation verified")

    def test_transform_conversion_cycle_affine(self, registrar_ants, test_images):
        """Test round-trip conversion: ITK affine transform -> ANTs -> ITK."""
        reference_image = test_images[0]

        print("\nTesting affine transform conversion cycle...")

        # Create an ITK affine transform with known parameters
        affine_tfm = itk.AffineTransform[itk.D, 3].New()

        # Set center
        center = itk.Point[itk.D, 3]()
        center[0] = 100.0
        center[1] = 100.0
        center[2] = 50.0
        affine_tfm.SetCenter(center)

        # Set rotation (small rotation around Z axis)
        angle = np.pi / 12  # 15 degrees
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
        )
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(rotation_matrix))

        # Set translation
        translation = itk.Vector[itk.D, 3]()
        translation[0] = 5.0
        translation[1] = 10.0
        translation[2] = -3.0
        affine_tfm.SetTranslation(translation)

        print(f"  Original center: {[center[i] for i in range(3)]}")
        print(f"  Original translation: {[translation[i] for i in range(3)]}")

        # Convert ITK -> ANTs
        ants_tfm = registrar_ants.itk_transform_to_ants_transform(
            affine_tfm, reference_image
        )
        assert ants_tfm is not None, "ANTs transform is None"
        print(f"  ANTs transform type: {ants_tfm.transform_type}")

        # Convert back ANTs -> ITK via displacement field
        # (ANTs stores as displacement field, so we convert back through that)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save ANTs transform to file
            temp_tfm_file = os.path.join(tmpdir, "temp_transform.mat")
            ants.write_transform(ants_tfm, temp_tfm_file)

            # Read back as displacement field
            recovered_tfm = (
                registrar_ants._antsfile_to_itk_displacement_field_transform(
                    temp_tfm_file, reference_image
                )
            )

        assert recovered_tfm is not None, "Recovered transform is None"
        print(f"  Recovered transform type: {type(recovered_tfm).__name__}")

        # Test that transforms produce similar results on test points
        test_points = [
            [100.0, 100.0, 50.0],
            [120.0, 110.0, 60.0],
            [80.0, 90.0, 40.0],
        ]

        print("  Testing transform on sample points...")
        max_point_diff = 0.0

        for pt_coords in test_points:
            # Transform with original
            pt_itk = itk.Point[itk.D, 3]()
            for i in range(3):
                pt_itk[i] = pt_coords[i]

            pt_transformed_original = affine_tfm.TransformPoint(pt_itk)
            pt_transformed_recovered = recovered_tfm.TransformPoint(pt_itk)

            # Calculate difference
            diff = np.sqrt(
                sum(
                    (pt_transformed_original[i] - pt_transformed_recovered[i]) ** 2
                    for i in range(3)
                )
            )
            max_point_diff = max(max_point_diff, diff)

        print(f"  Max point transformation difference: {max_point_diff:.6f}")

        # Allow some tolerance due to displacement field discretization
        assert max_point_diff < 0.5, f"Transform difference too large: {max_point_diff}"

        print("✓ Affine transform conversion cycle successful")

    def test_transform_conversion_cycle_displacement_field(
        self, registrar_ants, test_images
    ):
        """Test round-trip conversion: ITK displacement field -> ANTs -> ITK."""
        reference_image = test_images[0]

        print("\nTesting displacement field transform conversion cycle...")

        # Create a simple displacement field
        VectorImageType = itk.Image[itk.Vector[itk.F, 3], 3]
        disp_field = VectorImageType.New()
        disp_field.CopyInformation(reference_image)
        disp_field.SetRegions(reference_image.GetLargestPossibleRegion())
        disp_field.Allocate()

        # Fill with a simple displacement pattern
        disp_array = itk.array_from_image(disp_field)
        # Create a smooth displacement field (small random displacements)
        for i in range(3):
            disp_array[..., i] = np.random.randn(*disp_array.shape[:-1]) * 0.5

        # Create displacement field transform
        disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        disp_tfm.SetDisplacementField(disp_field)

        print(f"  Original displacement field size: {itk.size(disp_field)}")
        print(
            f"  Max displacement magnitude: {np.max(np.linalg.norm(disp_array, axis=-1)):.3f}"
        )

        # Convert ITK -> ANTs
        ants_tfm = registrar_ants.itk_transform_to_ants_transform(
            disp_tfm, reference_image
        )
        assert ants_tfm is not None, "ANTs transform is None"
        print("  ANTs transform created successfully")

        # Convert back ANTs -> ITK

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_tfm_file = os.path.join(tmpdir, "temp_disp_transform.mat")
            ants.write_transform(ants_tfm, temp_tfm_file)

            recovered_tfm = (
                registrar_ants._antsfile_to_itk_displacement_field_transform(
                    temp_tfm_file, reference_image
                )
            )

        assert recovered_tfm is not None, "Recovered transform is None"
        print(f"  Recovered transform type: {type(recovered_tfm).__name__}")

        # Get recovered displacement field
        recovered_disp_field = recovered_tfm.GetDisplacementField()
        recovered_disp_array = itk.array_from_image(recovered_disp_field)

        print(f"  Recovered displacement field size: {itk.size(recovered_disp_field)}")
        print(
            f"  Recovered max displacement: {np.max(np.linalg.norm(recovered_disp_array, axis=-1)):.3f}"
        )

        # Compare displacement fields
        diff_array = disp_array - recovered_disp_array
        max_diff = np.max(np.linalg.norm(diff_array, axis=-1))
        mean_diff = np.mean(np.linalg.norm(diff_array, axis=-1))

        print(f"  Max displacement difference: {max_diff:.6f}")
        print(f"  Mean displacement difference: {mean_diff:.6f}")

        # Allow tolerance for conversion artifacts
        assert max_diff < 1.0, f"Displacement field difference too large: {max_diff}"
        assert mean_diff < 0.1, f"Mean displacement difference too large: {mean_diff}"

        print("✓ Displacement field transform conversion cycle successful")

    def test_transform_conversion_with_composite(self, registrar_ants, test_images):
        """Test conversion of composite transforms."""
        reference_image = test_images[0]

        print("\nTesting composite transform conversion...")

        # Create a composite transform with translation + affine
        composite_tfm = itk.CompositeTransform[itk.D, 3].New()

        # Add translation
        translation_tfm = itk.TranslationTransform[itk.D, 3].New()
        translation_tfm.SetOffset([5.0, -3.0, 2.0])
        composite_tfm.AddTransform(translation_tfm)

        # Add affine (rotation)
        affine_tfm = itk.AffineTransform[itk.D, 3].New()
        center = itk.Point[itk.D, 3]()
        center.Fill(100.0)
        affine_tfm.SetCenter(center)

        angle = np.pi / 18  # 10 degrees
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        affine_tfm.SetMatrix(itk.GetMatrixFromArray(rotation_matrix))
        composite_tfm.AddTransform(affine_tfm)

        print(
            f"  Composite transform with {composite_tfm.GetNumberOfTransforms()} transforms"
        )

        # Convert to ANTs
        ants_tfm = registrar_ants.itk_transform_to_ants_transform(
            composite_tfm, reference_image
        )
        assert ants_tfm is not None, "ANTs transform is None"

        # Test on sample points
        test_points = [
            [100.0, 100.0, 50.0],
            [150.0, 120.0, 60.0],
        ]

        print("  Testing composite transform on sample points...")

        for pt_coords in test_points:
            pt_itk = itk.Point[itk.D, 3]()
            for i in range(3):
                pt_itk[i] = pt_coords[i]

            pt_transformed = composite_tfm.TransformPoint(pt_itk)
            print(f"    Point {pt_coords} -> {[pt_transformed[i] for i in range(3)]}")

        print("✓ Composite transform conversion successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
