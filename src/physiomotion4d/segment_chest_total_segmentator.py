"""Module for segmenting chest CT images using TotalSegmentator.

This module provides the SegmentChestTotalSegmentator class that implements
chest CT segmentation using the TotalSegmentator deep learning model. It inherits
from SegmentChestBase and defines anatomical structure mappings specific to
TotalSegmentator's output labels.
"""

import argparse
import os
import tempfile

import itk
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator

from physiomotion4d.segment_chest_base import SegmentChestBase


class SegmentChestTotalSegmentator(SegmentChestBase):
    """
    Chest CT segmentation using TotalSegmentator deep learning model.

    This class implements chest CT segmentation using the TotalSegmentator
    neural network, which provides detailed anatomical structure segmentation
    including organs, bones, and vessels. It maps TotalSegmentator's output
    labels to physiological groups for motion analysis.

    TotalSegmentator provides segmentation for 117 anatomical structures
    including detailed organ, bone, and vessel segmentation. This implementation
    combines the 'total' task (main organs and structures) with the 'body' task
    (body outline) to ensure complete coverage.

    The class maintains specific ID mappings for:
    - Heart structures (heart, atrial appendage, heart envelope)
    - Major vessels (aorta, vena cava, carotid arteries, etc.)
    - Lung structures (lung lobes, trachea, esophagus)
    - Bone structures (vertebrae, ribs, sternum, skull)
    - Soft tissue organs (liver, kidneys, spleen, etc.)

    Attributes:
        target_spacing (float): Target spacing set to 1.5mm for TotalSegmentator
        heart_mask_ids (dict): Dictionary mapping heart structure IDs to names
        major_vessels_mask_ids (dict): Dictionary mapping vessel IDs to names
        lung_mask_ids (dict): Dictionary mapping lung structure IDs to names
        bone_mask_ids (dict): Dictionary mapping bone structure IDs to names
        soft_tissue_mask_ids (dict): Dictionary mapping soft tissue IDs to names

    Example:
        >>> segmenter = SegmentChestTotalSegmentator()
        >>> result = segmenter.segment(ct_image, contrast_enhanced_study=True)
        >>> labelmap = result["labelmap"]
        >>> heart_mask = result["heart"]
    """

    def __init__(self):
        """Initialize the TotalSegmentator-based chest segmentation.

        Sets up the TotalSegmentator-specific anatomical structure ID mappings
        and processing parameters. The target spacing is set to 1.5mm which
        provides a good balance between accuracy and processing speed.
        """
        super().__init__()

        self.target_spacing = 1.5

        self.heart_mask_ids = {
            51: "heart",
            61: "atrial_appendage_left",
            140: "heart_envelop",
        }

        self.major_vessels_mask_ids = {
            52: "aorta",
            53: "pulmonary_vein",
            54: "brachiocephalic_trunk",
            55: "right_subclavian_artery",
            56: "left_subclavian_artery",
            57: "common_carotid_artery_right",
            58: "common_carotid_artery_left",
            59: "brachiocephalic_vein_left",
            60: "brachiocephalic_vein_right",
            62: "superior_vena_cava",
            63: "inferior_vena_cava",
        }

        self.lung_mask_ids = {
            10: "lung_upper_lobe_left",
            11: "lung_lower_lobe_left",
            12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right",
            14: "lung_lower_lobe_right",
            15: "esophagus",
            16: "trachea",
        }

        self.bone_mask_ids = {
            26: "vertebra_S1",
            27: "vertebra_L5",
            28: "vertebra_L4",
            29: "vertebrae_L3",
            30: "vertebrae_L2",
            31: "vertebrae_L1",
            32: "vertebrae_T12",
            33: "vertebrae_T11",
            34: "vertebrae_T10",
            35: "vertebrae_T9",
            36: "vertebrae_T8",
            37: "vertebrae_T7",
            38: "vertebrae_T6",
            39: "vertebrae_T5",
            40: "vertebrae_T4",
            41: "vertebrae_T3",
            42: "vertebrae_T2",
            43: "vertebrae_T1",
            44: "vertebrae_C7",
            45: "vertebrae_C6",
            46: "vertebrae_C5",
            47: "vertebrae_C4",
            48: "vertebrae_C3",
            49: "vertebrae_C2",
            50: "vertebrae_C1",
            69: "humerus_left",
            70: "humerus_right",
            71: "scapula_left",
            72: "scapula_right",
            73: "clavicula_left",
            74: "clavicula_right",
            75: "femur_left",
            76: "femur_right",
            77: "hip_left",
            78: "hip_right",
            91: "skull",
            92: "rib_left_1",
            93: "rib_left_2",
            94: "rib_left_3",
            95: "rib_left_4",
            96: "rib_left_5",
            97: "rib_left_6",
            98: "rib_left_7",
            99: "rib_left_8",
            100: "rib_left_9",
            101: "rib_left_10",
            102: "rib_left_11",
            103: "rib_left_12",
            104: "rib_right_1",
            105: "rib_right_2",
            106: "rib_right_3",
            107: "rib_right_4",
            108: "rib_right_5",
            109: "rib_right_6",
            110: "rib_right_7",
            111: "rib_right_8",
            112: "rib_right_9",
            113: "rib_right_10",
            114: "rib_right_11",
            115: "rib_right_12",
            116: "sternum",
            117: "costal_cartilages",
        }

        self.soft_tissue_mask_ids = {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "pancreas",
            8: "adrenal_gland_right",
            9: "adrenal_gland_left",
            17: "thyroid_gland",
            18: "small_bowel",
            19: "duodenum",
            20: "colon",
            21: "urinary_bladder",
            22: "prostate",
            25: "sacrum",
            80: "gluteus_maximus_left",
            81: "gluteus_maximus_right",
            82: "gluteus_medius_left",
            83: "gluteus_medius_right",
            84: "gluteus_minimus_left",
            85: "gluteus_minimus_right",
            90: "brain",
            133: "soft_tissue",
        }

        # From Base Class
        # self.contrast_mask_ids = { 135: "contrast" }

        self.set_other_and_all_mask_ids()

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run TotalSegmentator on the preprocessed image and return result.

        This implementation runs both the 'total' and 'body' tasks from
        TotalSegmentator to ensure comprehensive segmentation. The 'total' task
        segments major organs and structures, while the 'body' task provides
        body outline segmentation to fill gaps.

        The method uses temporary files for coordinate system conversion between
        ITK (LPS) and nibabel (RAS) formats, which is required for proper
        integration with TotalSegmentator.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image with
                isotropic spacing and appropriate intensity scaling

        Returns:
            itk.image: The segmentation labelmap with TotalSegmentator labels.
                Background regions from the 'total' task are filled with
                soft tissue labels from the 'body' task

        Note:
            Requires GPU acceleration (device="gpu:0") for reasonable performance.
            The method automatically handles coordinate system conversions between
            ITK and nibabel formats.

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # ITK and Nibabel use different coordinate systems (LPS vs RAS).
            # The safest conversion is via a temporary file. This approach
            # still reduces I/O compared to the original implementation.
            tmp_file = os.path.join(tmp_dir, "in.nii.gz")
            itk.imwrite(preprocessed_image, tmp_file, compression=True)
            nib_image = nib.load(tmp_file)

            # For higher performance, you can use fast=True, which uses a
            # faster but less accurate model.
            output_nib_image1 = totalsegmentator(nib_image, task="total", device="cuda")
            labelmap_arr1 = output_nib_image1.get_fdata().astype(np.uint8)

            output_nib_image2 = totalsegmentator(nib_image, task="body", device="cuda")
            labelmap_arr2 = output_nib_image2.get_fdata().astype(np.uint8)

            # The data from nibabel is in RAS orientation with xyz axis order.
            # The combination logic can be performed on these numpy arrays.
            mask1 = labelmap_arr1 == 0
            mask2 = labelmap_arr2 > 0
            mask = mask1 & mask2
            final_arr = np.where(
                mask, list(self.soft_tissue_mask_ids.keys())[-1], labelmap_arr1
            )

            # To create an ITK image, we save the result and read it back with
            # ITK. This correctly handles the coordinate system and data
            # layout conversions.
            out_tmp_file = os.path.join(tmp_dir, "out.nii.gz")
            # Use the affine from one of the outputs to preserve spatial info
            result_nib = nib.Nifti1Image(final_arr, output_nib_image1.affine)
            nib.save(result_nib, out_tmp_file)
            labelmap_image = itk.imread(out_tmp_file)
            labelmap_arr = itk.array_from_image(labelmap_image).astype(np.uint8)
            labelmap_image = itk.image_from_array(labelmap_arr)
            labelmap_image.CopyInformation(preprocessed_image)

        return labelmap_image


def parse_args():
    """Parse command line arguments for TotalSegmentator.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - input_image: Path to input CT image
            - output_image: Path for output segmentation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """Command line interface for TotalSegmentator-based chest segmentation.

    Example usage:
        python segment_chest_total_segmentator.py \
            --input_image chest_ct.mha \
            --output_image segmentation.mha
    """
    args = parse_args()
    segmenter = SegmentChestTotalSegmentator()
    result = segmenter.segment(itk.imread(args.input_image))
    itk.imwrite(result["labelmap"], args.output_image, compression=True)
