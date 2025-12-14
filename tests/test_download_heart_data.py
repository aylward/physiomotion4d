#!/usr/bin/env python
"""
Test for downloading and converting Slicer-Heart-CT data.

This test replicates the functionality from cells 0-2 of the notebook
Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb.
"""

import urllib.request
from pathlib import Path

import pytest


@pytest.mark.requires_data
class TestDownloadHeartData:
    """Test suite for downloading and converting Slicer-Heart-CT data."""

    def test_directories_created(self, test_directories):
        """Test that directories are created successfully."""
        data_dir = test_directories["data"]
        output_dir = test_directories["output"]

        assert data_dir.exists(), f"Data directory not created: {data_dir}"
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        assert data_dir.is_dir(), f"Data path is not a directory: {data_dir}"
        assert output_dir.is_dir(), f"Output path is not a directory: {output_dir}"

    def test_data_downloaded(self, download_truncal_valve_data, test_directories):
        """Test that the TruncalValve 4D CT data file is downloaded."""
        data_file = download_truncal_valve_data

        assert data_file.exists(), f"Data file not found: {data_file}"
        assert data_file.is_file(), f"Data path is not a file: {data_file}"

        # Check file size is reasonable (should be > 1MB)
        file_size = data_file.stat().st_size
        assert file_size > 1_000_000, f"Downloaded file seems too small: {file_size} bytes"

        print(f"\n✓ Data file downloaded successfully: {data_file}")
        print(f"  File size: {file_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
