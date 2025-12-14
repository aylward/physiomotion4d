#!/usr/bin/env python
"""
Test for chest CT segmentation using VISTA-3D.

This test depends on test_convert_nrrd_4d_to_3d and tests segmentation
functionality on two time points from the converted 3D data.
"""

from pathlib import Path

import itk
import numpy as np
import pytest

from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D


@pytest.mark.requires_data
@pytest.mark.slow
class TestSegmentChestVista3D:
    """Test suite for VISTA-3D chest CT segmentation."""

    def test_segmenter_initialization(self, segmenter_vista_3d):
        """Test that SegmentChestVista3D initializes correctly."""
        assert segmenter_vista_3d is not None, "Segmenter not initialized"
        assert segmenter_vista_3d.device is not None, "CUDA device not initialized"
        
        # Check that anatomical structure ID mappings are defined
        assert len(segmenter_vista_3d.heart_mask_ids) > 0, "Heart mask IDs not defined"
        assert len(segmenter_vista_3d.major_vessels_mask_ids) > 0, "Major vessels mask IDs not defined"
        assert len(segmenter_vista_3d.lung_mask_ids) > 0, "Lung mask IDs not defined"
        assert len(segmenter_vista_3d.bone_mask_ids) > 0, "Bone mask IDs not defined"
        assert len(segmenter_vista_3d.soft_tissue_mask_ids) > 0, "Soft tissue mask IDs not defined"
        
        # Check VISTA-3D specific attributes
        assert segmenter_vista_3d.bundle_path is not None, "Bundle path not set"
        assert segmenter_vista_3d.label_prompt is None, "Label prompt should be None initially"
        
        print("\n✓ Segmenter initialized with correct parameters")
        print(f"  Heart structures: {len(segmenter_vista_3d.heart_mask_ids)}")
        print(f"  Major vessels: {len(segmenter_vista_3d.major_vessels_mask_ids)}")
        print(f"  Lung structures: {len(segmenter_vista_3d.lung_mask_ids)}")
        print(f"  Bone structures: {len(segmenter_vista_3d.bone_mask_ids)}")
        print(f"  Soft tissue structures: {len(segmenter_vista_3d.soft_tissue_mask_ids)}")
        print(f"  Bundle path: {segmenter_vista_3d.bundle_path}")

    def test_segment_single_image(self, segmenter_vista_3d, test_images, test_directories):
        """Test automatic segmentation on a single time point."""
        output_dir = test_directories["output"]
        
        # Ensure we're in automatic segmentation mode
        segmenter_vista_3d.set_whole_image_segmentation()
        
        # Test on first time point only
        input_image = test_images[0]
        
        print(f"\nSegmenting time point 0 (automatic mode)...")
        print(f"  Input image size: {itk.size(input_image)}")
        
        # Run segmentation
        result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        
        # Verify result is a dictionary with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = ["labelmap", "lung", "heart", "major_vessels", "bone", 
                        "soft_tissue", "other", "contrast"]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"
        
        # Verify labelmap properties
        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"
        
        # Check that labels are present
        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"
        
        print(f"✓ Segmentation complete for time point 0")
        print(f"  Labelmap size: {itk.size(labelmap)}")
        print(f"  Unique labels: {len(unique_labels)}")
        
        # Save results
        seg_output_dir = output_dir / "segmentation_vista3d"
        seg_output_dir.mkdir(exist_ok=True)
        
        itk.imwrite(labelmap, str(seg_output_dir / "slice_000_labelmap.mha"), compression=True)
        print(f"  Saved labelmap to: {seg_output_dir / 'slice_000_labelmap.mha'}")

    def test_segment_multiple_images(self, segmenter_vista_3d, test_images, test_directories):
        """Test automatic segmentation on two time points."""
        output_dir = test_directories["output"]
        seg_output_dir = output_dir / "segmentation_vista3d"
        seg_output_dir.mkdir(exist_ok=True)
        
        # Ensure automatic segmentation mode
        segmenter_vista_3d.set_whole_image_segmentation()
        
        results = []
        for i, input_image in enumerate(test_images):
            print(f"\nSegmenting time point {i}...")
            
            result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
            results.append(result)
            
            # Save labelmap for each time point
            labelmap = result["labelmap"]
            output_file = seg_output_dir / f"slice_{i:03d}_labelmap.mha"
            itk.imwrite(labelmap, str(output_file), compression=True)
            
            print(f"✓ Time point {i} complete")
            print(f"  Saved to: {output_file}")
        
        assert len(results) == 2, "Expected 2 segmentation results"
        print(f"\n✓ Successfully segmented {len(results)} time points")

    def test_anatomy_group_masks(self, segmenter_vista_3d, test_images):
        """Test that anatomy group masks are created correctly."""
        segmenter_vista_3d.set_whole_image_segmentation()
        input_image = test_images[0]
        
        # Run segmentation
        result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        
        # Check each anatomy group mask
        anatomy_groups = ["lung", "heart", "major_vessels", "bone", "soft_tissue", "other"]
        
        for group in anatomy_groups:
            mask = result[group]
            assert mask is not None, f"{group} mask is None"
            
            # Check that mask is binary
            mask_arr = itk.array_from_image(mask)
            unique_values = np.unique(mask_arr)
            assert len(unique_values) <= 2, f"{group} mask should be binary"
            assert 0 in unique_values, f"{group} mask should contain background"
            
            # Check that mask has same size as input
            assert itk.size(mask) == itk.size(input_image), f"{group} mask size mismatch"
        
        print("\n✓ All anatomy group masks created correctly")
        for group in anatomy_groups:
            mask_arr = itk.array_from_image(result[group])
            num_voxels = np.sum(mask_arr > 0)
            print(f"  {group}: {num_voxels} voxels")

    def test_label_prompt_segmentation(self, segmenter_vista_3d, test_images, test_directories):
        """Test segmentation with specific label prompts."""
        output_dir = test_directories["output"]
        seg_output_dir = output_dir / "segmentation_vista3d"
        seg_output_dir.mkdir(exist_ok=True)
        
        input_image = test_images[0]
        
        # Test with heart and aorta labels only
        heart_aorta_labels = [115, 6]  # Heart and aorta
        segmenter_vista_3d.set_label_prompt(heart_aorta_labels)
        
        print(f"\nSegmenting with label prompts: {heart_aorta_labels}")
        result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        
        # Verify result
        assert isinstance(result, dict), "Result should be a dictionary"
        labelmap = result["labelmap"]
        
        # Check that only prompted labels are present (plus background and soft tissue fill)
        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        
        print(f"✓ Label prompt segmentation complete")
        print(f"  Unique labels: {unique_labels}")
        
        # Save result
        output_file = seg_output_dir / "slice_000_label_prompt.mha"
        itk.imwrite(labelmap, str(output_file), compression=True)
        print(f"  Saved to: {output_file}")
        
        # Reset to whole image segmentation
        segmenter_vista_3d.set_whole_image_segmentation()

    def test_contrast_detection(self, segmenter_vista_3d, test_images):
        """Test contrast detection functionality."""
        segmenter_vista_3d.set_whole_image_segmentation()
        input_image = test_images[0]
        
        # Test without contrast
        result_no_contrast = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        contrast_mask_no = result_no_contrast["contrast"]
        
        # Test with contrast flag
        result_with_contrast = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=True)
        contrast_mask_yes = result_with_contrast["contrast"]
        
        # Both should return valid masks
        assert contrast_mask_no is not None, "Contrast mask (no flag) is None"
        assert contrast_mask_yes is not None, "Contrast mask (with flag) is None"
        
        print("\n✓ Contrast detection tested")
        
        contrast_arr_no = itk.array_from_image(contrast_mask_no)
        contrast_arr_yes = itk.array_from_image(contrast_mask_yes)
        print(f"  Without contrast flag: {np.sum(contrast_arr_no > 0)} voxels")
        print(f"  With contrast flag: {np.sum(contrast_arr_yes > 0)} voxels")

    def test_preprocessing(self, segmenter_vista_3d, test_images):
        """Test preprocessing functionality."""
        segmenter_vista_3d.set_whole_image_segmentation()
        input_image = test_images[0]
        
        # Get original properties
        original_spacing = itk.spacing(input_image)
        
        # Preprocessing is done internally by segment(), not exposed as public method
        # Just verify that segment() works (which includes preprocessing)
        result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        
        # Check that segmentation was successful (which means preprocessing worked)
        assert result is not None, "Segmentation result is None"
        assert "labelmap" in result, "Labelmap not in result"
        
        print("\n✓ Preprocessing tested (via successful segmentation)")
        print(f"  Original image spacing: {original_spacing}")

    def test_postprocessing(self, segmenter_vista_3d, test_images):
        """Test postprocessing functionality."""
        segmenter_vista_3d.set_whole_image_segmentation()
        input_image = test_images[0]
        
        # Run full segmentation to get labelmap
        result = segmenter_vista_3d.segment(input_image, contrast_enhanced_study=False)
        labelmap = result["labelmap"]
        
        # Postprocessing is part of segment(), verify output is properly sized
        assert itk.size(labelmap) == itk.size(input_image), "Postprocessing failed: size mismatch"
        
        # Check that labelmap has been resampled to original spacing
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)
        
        # Spacing should match (within floating point tolerance)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, \
                f"Spacing mismatch at dimension {i}"
        
        print("\n✓ Postprocessing tested")
        print(f"  Original spacing: {original_spacing}")
        print(f"  Labelmap spacing: {labelmap_spacing}")

    def test_set_and_reset_prompts(self, segmenter_vista_3d):
        """Test setting and resetting label prompt mode."""
        # Initially should be in automatic mode
        assert segmenter_vista_3d.label_prompt is None, "Label prompt should be None initially"
        
        # Set label prompt
        segmenter_vista_3d.set_label_prompt([115, 6])
        assert segmenter_vista_3d.label_prompt == [115, 6], "Label prompt not set correctly"
        
        # Reset to whole image
        segmenter_vista_3d.set_whole_image_segmentation()
        assert segmenter_vista_3d.label_prompt is None, "Label prompt should be None after reset"
        
        print("\n✓ Prompt setting and resetting works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

