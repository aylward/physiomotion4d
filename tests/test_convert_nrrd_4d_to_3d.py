#!/usr/bin/env python
"""
Test for converting 4D NRRD to 3D time series.

This test depends on test_download_heart_data and replicates the functionality
from cell 3 of the notebook Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb.
"""

import shutil

import pytest

from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D


@pytest.mark.requires_data
class TestConvertNRRD4DTo3D:
    """Test suite for converting 4D NRRD to 3D time series."""

    def test_convert_4d_to_3d(self, download_truncal_valve_data, test_directories):
        """Test conversion of 4D NRRD to 3D time series (replicates notebook cell 3)."""
        data_dir = test_directories["data"]
        output_dir = test_directories["output"]
        input_4d_file = download_truncal_valve_data

        # Convert 4D to 3D time series
        print("\nConverting 4D NRRD to 3D time series...")
        conv = ConvertNRRD4DTo3D()
        conv.load_nrrd_4d(str(input_4d_file))
        conv.save_3d_images(str(data_dir / "slice"))

        # Verify that slice files were created
        slice_007 = data_dir / "slice_007.mha"
        assert slice_007.exists(), f"Expected slice file not created: {slice_007}"

        # Count how many slice files were created
        slice_files = list(data_dir.glob("slice_*.mha"))
        print(f"✓ Created {len(slice_files)} slice files")
        assert len(slice_files) > 0, "No slice files were created"

        # Copy mid-stroke slice as fixed/reference image (as in notebook)
        fixed_image_output = output_dir / "slice_fixed.mha"
        shutil.copyfile(str(slice_007), str(fixed_image_output))

        assert fixed_image_output.exists(), (
            f"Fixed image not created: {fixed_image_output}"
        )
        print(f"✓ Fixed/reference image saved to: {fixed_image_output}")

    def test_slice_files_created(self, download_truncal_valve_data, test_directories):
        """Test that all expected slice files are present after conversion."""
        data_dir = test_directories["data"]

        # Check that slice files exist
        slice_files = list(data_dir.glob("slice_*.mha"))
        assert len(slice_files) > 10, (
            f"Expected more than 10 slice files, found {len(slice_files)}"
        )

        # Verify specific slice file exists (mid-stroke)
        slice_007 = data_dir / "slice_007.mha"
        assert slice_007.exists(), "Expected slice_007.mha not found"

        print(f"\n✓ Found {len(slice_files)} slice files")

    def test_fixed_image_output(self, download_truncal_valve_data, test_directories):
        """Test that fixed/reference image is copied to output directory."""
        output_dir = test_directories["output"]

        # Check that fixed image was copied to output
        fixed_image_path = output_dir / "slice_fixed.mha"
        assert fixed_image_path.exists(), "Fixed image not in output directory"

        # Check file size is reasonable
        file_size = fixed_image_path.stat().st_size
        assert file_size > 100_000, f"Fixed image seems too small: {file_size} bytes"

        print(f"\n✓ Fixed/reference image exists: {fixed_image_path}")
        print(f"  File size: {file_size / 1_000_000:.2f} MB")

    def test_load_nrrd_4d(self, download_truncal_valve_data):
        """Test loading 4D NRRD file."""
        input_4d_file = download_truncal_valve_data

        conv = ConvertNRRD4DTo3D()
        conv.load_nrrd_4d(str(input_4d_file))

        # Verify that data was loaded
        assert conv.nrrd_4d is not None, "4D NRRD data not loaded"
        assert conv.get_number_of_3d_images() > 0, "No time points found in 4D image"

        print(f"\n✓ Loaded 4D NRRD with {conv.get_number_of_3d_images()} time points")

    def test_save_3d_images(self, download_truncal_valve_data, test_directories):
        """Test saving 3D images from 4D NRRD."""
        data_dir = test_directories["data"]
        input_4d_file = download_truncal_valve_data

        conv = ConvertNRRD4DTo3D()
        conv.load_nrrd_4d(str(input_4d_file))

        num_time_points = conv.get_number_of_3d_images()

        # Save to a test subdirectory
        test_output_prefix = data_dir / "test_slice"
        conv.save_3d_images(str(test_output_prefix))

        # Verify files were created
        test_slice_files = list(data_dir.glob("test_slice_*.mha"))
        assert len(test_slice_files) > 0, "No test slice files were created"
        assert len(test_slice_files) == num_time_points, (
            f"Expected {num_time_points} files, found {len(test_slice_files)}"
        )

        print(f"\n✓ Saved {len(test_slice_files)} 3D images")

        # Clean up test files
        for test_file in test_slice_files:
            test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
