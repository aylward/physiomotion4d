#!/usr/bin/env python
"""
Tests for heart CT segmentation using SegmentHeartSimpleware (Simpleware Medical ASCardio).

Uses the same input data as experiments/Heart-Simpleware_Segmentation:
  data/CHOP-Valve4D/CT/RVOT28-Dias.nii.gz

Requires Synopsys Simpleware Medical with ASCardio and the test data to run
full segmentation tests. Initialization and path tests run without Simpleware.
"""

import os

import itk
import numpy as np
import pytest


def _simpleware_available(segmenter):
    """Return True if Simpleware Medical executable and script exist."""
    return os.path.exists(segmenter.simpleware_exe_path) and os.path.exists(
        segmenter.simpleware_script_path
    )


@pytest.mark.requires_data
@pytest.mark.slow
class TestSegmentHeartSimpleware:
    """Test suite for SegmentHeartSimpleware (Simpleware Medical ASCardio)."""

    def test_segmenter_initialization(self, segmenter_simpleware):
        """Test that SegmentHeartSimpleware initializes correctly."""
        seg = segmenter_simpleware
        assert seg is not None, "Segmenter not initialized"
        assert seg.target_spacing == 1.0, (
            "Target spacing should be 1.0 mm for Simpleware"
        )

        assert len(seg.heart_mask_ids) > 0, "Heart mask IDs not defined"
        assert len(seg.major_vessels_mask_ids) > 0, "Major vessels mask IDs not defined"
        # ASCardio does not segment lung, bone, soft_tissue (empty dicts)
        assert seg.lung_mask_ids == {}, "ASCardio does not segment lungs"
        assert seg.bone_mask_ids == {}, "ASCardio does not segment bone"
        assert seg.soft_tissue_mask_ids == {}, "ASCardio does not segment soft tissue"

        assert seg.simpleware_exe_path is not None, "Simpleware executable path not set"
        assert seg.simpleware_script_path is not None, "Simpleware script path not set"
        assert "SimplewareScript_heart_segmentation" in seg.simpleware_script_path

        print("\n✓ Segmenter initialized with correct parameters")
        print(f"  Target spacing: {seg.target_spacing} mm")
        print(f"  Heart structures: {len(seg.heart_mask_ids)}")
        print(f"  Major vessels: {len(seg.major_vessels_mask_ids)}")

    def test_set_simpleware_executable_path(self, segmenter_simpleware):
        """Test setting custom Simpleware executable path."""
        seg = segmenter_simpleware
        original = seg.simpleware_exe_path
        custom = "D:/Custom/ConsoleSimplewareMedical.exe"
        seg.set_simpleware_executable_path(custom)
        assert seg.simpleware_exe_path == custom
        seg.set_simpleware_executable_path(original)
        assert seg.simpleware_exe_path == original
        print("\n✓ set_simpleware_executable_path works correctly")

    def test_segment_single_image(
        self,
        segmenter_simpleware,
        heart_simpleware_image,
        heart_simpleware_image_path,
        test_directories,
    ):
        """Test segmentation on the same cardiac CT as the notebook (RVOT28-Dias.nii.gz)."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip(
                "Simpleware Medical not found (executable or script). "
                "Install Simpleware Medical with ASCardio to run this test."
            )

        output_dir = test_directories["output"]
        input_image = heart_simpleware_image

        print(
            "\nSegmenting cardiac CT (same as Heart-Simpleware_Segmentation notebook)..."
        )
        print(f"  Input: {heart_simpleware_image_path.name}")
        print(f"  Image size: {itk.size(input_image)}")

        result = segmenter_simpleware.segment(input_image, contrast_enhanced_study=True)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = [
            "labelmap",
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
            "contrast",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"

        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        print("✓ Segmentation complete")
        print(f"  Unique labels: {len(unique_labels)}")

        seg_output_dir = output_dir / "segmentation_simpleware"
        seg_output_dir.mkdir(exist_ok=True)
        itk.imwrite(
            labelmap,
            str(seg_output_dir / "heart_labelmap_simpleware.nii.gz"),
            compression=True,
        )
        print(f"  Saved to: {seg_output_dir / 'heart_labelmap_simpleware.nii.gz'}")

    def test_anatomy_group_masks(
        self, segmenter_simpleware, heart_simpleware_image, heart_simpleware_image_path
    ):
        """Test that anatomy group masks are created (heart, vessels, etc.)."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = heart_simpleware_image
        result = segmenter_simpleware.segment(input_image, contrast_enhanced_study=True)

        anatomy_groups = [
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
        ]
        for group in anatomy_groups:
            mask = result[group]
            assert mask is not None, f"{group} mask is None"
            mask_arr = itk.array_from_image(mask)
            unique_values = np.unique(mask_arr)
            assert len(unique_values) <= 2, f"{group} mask should be binary"
            assert 0 in unique_values or mask_arr.size == 0
            assert itk.size(mask) == itk.size(input_image), (
                f"{group} mask size mismatch"
            )

        heart_arr = itk.array_from_image(result["heart"])
        vessels_arr = itk.array_from_image(result["major_vessels"])
        print("\n✓ All anatomy group masks created correctly")
        print(f"  heart: {np.sum(heart_arr > 0)} voxels")
        print(f"  major_vessels: {np.sum(vessels_arr > 0)} voxels")

    def test_contrast_detection(
        self, segmenter_simpleware, heart_simpleware_image, heart_simpleware_image_path
    ):
        """Test contrast mask is returned (base class behavior)."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = heart_simpleware_image
        result = segmenter_simpleware.segment(input_image, contrast_enhanced_study=True)
        contrast_mask = result["contrast"]
        assert contrast_mask is not None
        assert itk.size(contrast_mask) == itk.size(input_image)
        print("\n✓ Contrast mask returned")

    def test_postprocessing(
        self, segmenter_simpleware, heart_simpleware_image, heart_simpleware_image_path
    ):
        """Test that output labelmap matches input size and spacing."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = heart_simpleware_image
        result = segmenter_simpleware.segment(input_image, contrast_enhanced_study=True)
        labelmap = result["labelmap"]

        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, (
                f"Spacing mismatch at dimension {i}"
            )
        print("\n✓ Postprocessing: labelmap size and spacing match input")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
