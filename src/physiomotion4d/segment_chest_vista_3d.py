"""Module for segmenting chest CT images using VISTA3D.

This module provides the SegmentChestVista3D class that implements chest CT
segmentation using the VISTA-3D foundational model from NVIDIA. VISTA-3D is
a versatile segmentation model that can perform both automatic segmentation
and interactive segmentation with point or label prompts.

The module requires the VISTA-3D model weights to be downloaded from Hugging Face
and supports both local inference and NVIDIA NIM deployment modes.
"""

# Please start vista3d docker:
#    docker run --rm -it --name vista3d --runtime=nvidia
#      -e CUDA_VISIBLE_DEVICES=0
#      -e NGC_API_KEY=$NGC_API_KEY
#      --shm-size=8G -p 8000:8000
#      -v /tmp/data:/home/aylward/tmp/data nvcr.io/nim/nvidia/vista3d:latest

import argparse
import logging
import os
import shutil
import tempfile

import itk
import torch
from huggingface_hub import snapshot_download

from physiomotion4d.segment_chest_base import SegmentChestBase


class SegmentChestVista3D(SegmentChestBase):
    """
    Chest CT segmentation using NVIDIA VISTA-3D foundational model.

    This class implements chest CT segmentation using the VISTA-3D model,
    a versatile foundational segmentation model that supports both automatic
    ('everything') segmentation and interactive segmentation with prompts.

    VISTA-3D is a state-of-the-art 3D medical image segmentation model that
    can segment 132+ anatomical structures. It supports two interaction modes:
    1. Everything segmentation: Segments all detectable structures
    2. Label prompts: Segments specific structures by ID

    The class automatically downloads model weights from Hugging Face and
    supports GPU acceleration. It includes additional soft tissue segmentation
    to fill gaps not covered by the base VISTA-3D segmentation.

    Attributes:
        target_spacing (float): Adaptive spacing based on input image
        device (torch.device): GPU device for model inference
        bundle_path (str): Path to VISTA-3D model weights
        hf_pipeline (object): Hugging Face pipeline for inference
        label_prompt (list): Specific anatomical structure IDs to segment

    Example:
        >>> # Automatic segmentation
        >>> segmenter = SegmentChestVista3D()
        >>> result = segmenter.segment(ct_image, contrast_enhanced_study=True)
        >>> labelmap = result["labelmap"]
        >>> heart_mask = result["heart"]
        >>>
        >>> # Segment specific structures
        >>> segmenter.set_label_prompt([115, 6, 28])  # Heart, aorta, left lung
        >>> result = segmenter.segment(ct_image)
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the VISTA-3D based chest segmentation.

        Sets up the VISTA-3D model including downloading weights from Hugging Face,
        configuring GPU device, and initializing anatomical structure mappings
        specific to VISTA-3D's label set.

        The initialization automatically downloads the VISTA-3D model weights
        from the MONAI/VISTA3D-HF repository on Hugging Face if not already present.

        Args:
            log_level: Logging level (default: logging.INFO)

        Raises:
            RuntimeError: If CUDA is not available for GPU acceleration
            ConnectionError: If model weights cannot be downloaded
        """
        super().__init__(log_level=log_level)

        self.target_spacing = 0.0
        self.resale_intensity_range = False
        self.input_percentile_range = None
        self.output_percentile_range = None
        self.output_intensity_range = None

        self.device = torch.device("cuda:0")

        self.bundle_path = os.path.join(
            os.path.dirname(__file__), "network_weights/vista3d"
        )
        os.makedirs(self.bundle_path, exist_ok=True)

        self.model_name = "vista3d"

        self.hf_pipeline_helper = None
        self.hf_pipeline = None

        self.label_prompt = None

        repo_id = "MONAI/VISTA3D-HF"
        snapshot_download(repo_id=repo_id, local_dir=self.bundle_path)

        self.heart_mask_ids = {
            108: "left_atrial_appendage",
            115: "heart",
            140: "heart_envelope",
        }

        self.major_vessels_mask_ids = {
            6: "aorta",
            7: "inferior_vena_cava",
            17: "portal_vein_and_splenic_vein",
            58: "left_iliac_artery",
            59: "right_iliac_artery",
            60: "left_iliac_vena",
            61: "right_iliac_vena",
            110: "left_brachiocephalic_vena",
            111: "right_brachiocephalic_vena",
            112: "left_common_carotid_artery",
            113: "right_common_carotid_artery",
            119: "pulmonary_vein",
            123: "left_subclavian_artery",
            124: "right_subclavian_artery",
            125: "superior_vena_cava",
        }

        self.lung_mask_ids = {
            28: "left_lung_upper_lobe",
            29: "left_lung_lower_lobe",
            30: "right_lung_upper_lobe",
            31: "right_lung_middle_lobe",
            32: "right_lung_lower_lobe",
            57: "trachea",
            132: "airway",
        }

        self.bone_mask_ids = {
            33: "vertebrae_L5",
            34: "vertebrae_L4",
            35: "vertebrae_L3",
            36: "vertebrae_L2",
            37: "vertebrae_L1",
            38: "vertebrae_T12",
            39: "vertebrae_T11",
            40: "vertebrae_T10",
            41: "vertebrae_T9",
            42: "vertebrae_T8",
            43: "vertebrae_T7",
            44: "vertebrae_T6",
            45: "vertebrae_T5",
            46: "vertebrae_T4",
            47: "vertebrae_T3",
            48: "vertebrae_T2",
            49: "vertebrae_T1",
            50: "vertebrae_C7",
            51: "vertebrae_C6",
            52: "vertebrae_C5",
            53: "vertebrae_C4",
            54: "vertebrae_C3",
            55: "vertebrae_C2",
            56: "vertebrae_C1",
            63: "left_rib_1",
            64: "left_rib_2",
            65: "left_rib_3",
            66: "left_rib_4",
            67: "left_rib_5",
            68: "left_rib_6",
            69: "left_rib_7",
            70: "left_rib_8",
            71: "left_rib_9",
            72: "left_rib_10",
            73: "left_rib_11",
            74: "left_rib_12",
            75: "right_rib_1",
            76: "right_rib_2",
            77: "right_rib_3",
            78: "right_rib_4",
            79: "right_rib_5",
            80: "right_rib_6",
            81: "right_rib_7",
            82: "right_rib_8",
            83: "right_rib_9",
            84: "right_rib_10",
            85: "right_rib_11",
            86: "right_rib_12",
            87: "left_humerus",
            88: "right_humerus",
            89: "left_scapula",
            90: "right_scapula",
            91: "left_clavicula",
            92: "right_clavicula",
            93: "left_femur",
            94: "right_femur",
            95: "left_hip",
            96: "right_hip",
            114: "costal_cartilages",
            120: "skull",
            122: "sternum",
            127: "vertebrae_S1",
        }

        self.soft_tissue_mask_ids = {
            121: "spinal_cord",
            118: "prostate",
            126: "thyroid_gland",
            62: "colon",
            19: "small_bowel",
            22: "brain",
            14: "left_kidney",
            15: "bladder",
            12: "stomach",
            13: "duodenum",
            8: "right_adrenal_gland",
            9: "left_adrenal_gland",
            10: "gallbladder",
            1: "liver",
            3: "spleen",
            4: "pancreas",
            5: "right_kidney",
            133: "soft_tissue",
        }

        # From Base Class
        # self.contrast_mask_ids = [135]

        self.set_other_and_all_mask_ids()

    def set_label_prompt(self, label_prompt: list):
        """
        Set specific anatomical structure labels to segment.

        Configures the segmentation to target specific anatomical structures
        by their VISTA-3D label IDs instead of performing automatic segmentation.

        Args:
            label_prompt (list): List of VISTA-3D anatomical structure IDs
                to segment. See class attributes for available IDs

        Example:
            >>> # Segment heart, aorta, and lungs only
            >>> segmenter.set_label_prompt([115, 6, 28, 29, 30, 31, 32])
            >>>
            >>> # Segment all cardiac structures
            >>> heart_ids = list(segmenter.heart_mask_ids.keys())
            >>> segmenter.set_label_prompt(heart_ids)
        """
        self.label_prompt = label_prompt

    def set_whole_image_segmentation(self):
        """
        Configure for automatic whole-image segmentation.

        Resets the segmentation mode to automatic 'everything' segmentation,
        clearing any previously set label prompts.
        This is the default mode that segments all detectable structures.

        Example:
            >>> # Reset to automatic segmentation after using label prompts
            >>> segmenter.set_label_prompt([115, 6])
            >>> # ... perform label-prompted segmentation
            >>> segmenter.set_whole_image_segmentation()  # Reset to automatic
        """
        self.label_prompt = None

    def segment_soft_tissue(
        self, preprocessed_image: itk.image, labelmap_image: itk.image
    ) -> itk.image:
        """
        Add soft tissue segmentation to fill gaps in VISTA-3D output.

        VISTA-3D may not segment all tissue regions, leaving gaps between
        structures. This method identifies soft tissue regions based on
        intensity thresholds and adds them to the labelmap.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image
            labelmap_image (itk.image): Existing VISTA-3D segmentation

        Returns:
            itk.image: Updated labelmap with soft tissue regions filled

        Example:
            >>> filled_labelmap = segmenter.segment_soft_tissue(
            ...     preprocessed_image, vista_labelmap
            ... )
        """
        hole_ids = [0]
        labelmap_plus_soft_tissue_image = self.segment_connected_component(
            preprocessed_image,
            labelmap_image,
            lower_threshold=-150,
            upper_threshold=700,
            labelmap_ids=hole_ids,
            mask_id=list(self.soft_tissue_mask_ids.keys())[-1],
            use_mid_slice=True,
        )

        return labelmap_plus_soft_tissue_image

    def preprocess_input(self, input_image: itk.image) -> itk.image:
        """
        Preprocess the input image for VISTA-3D segmentation.

        Extends the base preprocessing with VISTA-3D specific adaptations:
        - Adaptive spacing calculation based on input image properties
        - No intensity rescaling (VISTA-3D handles raw CT intensities)

        Args:
            input_image (itk.image): The input 3D CT image

        Returns:
            itk.image: Preprocessed image optimized for VISTA-3D

        Note:
            VISTA-3D works best with the original image spacing and intensity
            values, so minimal preprocessing is applied.
        """
        if self.target_spacing == 0.0:
            spacing = input_image.GetSpacing()
            self.target_spacing = (spacing[0] + spacing[1] + spacing[2]) / 3
            if self.target_spacing < 0.5:
                self.target_spacing = 0.5

        preprocessed_image = super().preprocess_input(input_image)

        return preprocessed_image

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run VISTA-3D segmentation on the preprocessed image.

        Performs segmentation using the VISTA-3D model with the configured
        interaction mode (automatic, point prompts, or label prompts). The
        method handles model loading, inference, and post-processing including
        soft tissue gap filling.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image ready
                for VISTA-3D inference

        Returns:
            itk.image: The segmentation labelmap with VISTA-3D labels and
                additional soft tissue segmentation

        Raises:
            ValueError: If no segmentation output is produced
            RuntimeError: If model inference fails

        Note:
            The method automatically selects the interaction mode based on
            the configured prompts:
            - No prompts: Everything segmentation
            - Label prompt set: Specific structure segmentation

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        os.sys.path.append(self.bundle_path)

        from hugging_face_pipeline import HuggingFacePipelineHelper

        if self.hf_pipeline_helper is None:
            self.hf_pipeline_helper = HuggingFacePipelineHelper(self.model_name)
        if self.hf_pipeline is None:
            self.hf_pipeline = self.hf_pipeline_helper.init_pipeline(
                os.path.join(self.bundle_path, "vista3d_pretrained_model"),
                device=self.device,
                resample_spacing=(
                    self.target_spacing,
                    self.target_spacing,
                    self.target_spacing,
                ),
            )

        tmp_dir = tempfile.mkdtemp()
        tmp_input_file_name = os.path.join(tmp_dir, "tmp.nii.gz")
        itk.imwrite(preprocessed_image, tmp_input_file_name, compression=True)

        input = [
            {
                "image": tmp_input_file_name,
            }
        ]
        if self.label_prompt is None:
            input[0].update(
                {
                    "label_prompt": self.hf_pipeline.EVERYTHING_LABEL,
                }
            )
        else:
            input[0].update(
                {
                    "label_prompt": self.label_prompt,
                }
            )

        self.hf_pipeline(input, output_dir=tmp_dir)

        output_itk = None
        for file_name in os.listdir(os.path.join(tmp_dir, "tmp")):
            if file_name.endswith(".nii.gz"):
                output_itk = itk.imread(os.path.join(tmp_dir, "tmp", file_name))
                output_itk.CopyInformation(preprocessed_image)
                break

        if output_itk is None:
            raise ValueError("No output image found")

        shutil.rmtree(tmp_dir)

        output_itk = self.segment_soft_tissue(preprocessed_image, output_itk)

        return output_itk


def parse_args():
    """
    Parse command line arguments for VISTA-3D segmentation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_image: Path to input CT image
            - output_image: Path for output segmentation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    """Command line interface for VISTA-3D based chest segmentation.

    Example usage:
        python segment_chest_vista_3d.py \
            --input_image chest_ct.mha \
            --output_image segmentation.mha
    """
    args = parse_args()
    segmenter = SegmentChestVista3D()
    result = segmenter.segment(itk.imread(args.input_image))
    itk.imwrite(result["labelmap"], args.output_image, compression=True)
