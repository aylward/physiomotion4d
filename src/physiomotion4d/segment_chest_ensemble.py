"""Module for segmenting chest CT images using VISTA3D."""

# Please start vista3d docker:
#    docker run --rm -it --name vista3d --runtime=nvidia
#      -e CUDA_VISIBLE_DEVICES=0
#      -e NGC_API_KEY=$NGC_API_KEY
#      --shm-size=8G -p 8000:8000
#      -v /tmp/data:/home/aylward/tmp/data nvcr.io/nim/nvidia/vista3d:latest

import logging

import itk
import numpy as np

from physiomotion4d.segment_anatomy_base import SegmentAnatomyBase
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D


class SegmentChestEnsemble(SegmentAnatomyBase):
    """
    A class that inherits from physioSegmentChest and implements the
    segmentation method using VISTA3D.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the vista3d class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.target_spacing = 0.0

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
            146: "brachiocephalic_trunk",
        }

        self.lung_mask_ids = {
            28: "left_lung_upper_lobe",
            29: "left_lung_lower_lobe",
            30: "right_lung_upper_lobe",
            31: "right_lung_middle_lobe",
            32: "right_lung_lower_lobe",
            57: "trachea",
            132: "airway",
            147: "esophagus",
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
            120: "skull",
            122: "sternum",
            114: "costal_cartilages",
            127: "vertebrae_S1",
        }

        self.soft_tissue_mask_ids = {
            3: "spleen",
            5: "right_kidney",
            14: "left_kidney",
            10: "gallbladder",
            1: "liver",
            12: "stomach",
            4: "pancreas",
            8: "right_adrenal_gland",
            9: "left_adrenal_gland",
            126: "thyroid_gland",
            19: "small_bowel",
            13: "duodenum",
            62: "colon",
            15: "bladder",
            118: "prostate",
            121: "spinal_cord",
            22: "brain",
            133: "soft_tissue",
            148: "sacrum",
            149: "gluteus_maximus_left",
            150: "gluteus_maximus_right",
            151: "gluteus_medius_left",
            152: "gluteus_medius_right",
            153: "gluteus_minimus_left",
            154: "gluteus_minimus_right",
        }

        self.set_other_and_all_mask_ids()

        self.heart_ids_map = {
            108: 61,  # left_atrial_appendage / atrial_appendage_left
            115: 51,  # heart
            140: 140,  # heart_envelope / heart_envelop
        }

        self.major_vessels_ids_map = {
            6: 52,  # aorta
            7: 63,  # inferior_vena_cava
            17: -1,  # portal_vein_and_splenic_vein
            58: -1,  # left_iliac_artery
            59: -1,  # right_iliac_artery
            60: -1,  # left_iliac_vena
            61: -1,  # right_iliac_vena
            110: 59,  # left_brachiocephalic_vena / brachiocephalic_vein_left
            111: 60,  # right_brachiocephalic_vena / brachiocephalic_vein_right
            112: 58,  # left_common_carotid_artery / common_carotid_artery_left
            113: 57,  # right_common_carotid_artery / common_carotid_artery_right
            119: 53,  # pulmonary_vein
            123: 56,  # left_subclavian_artery
            124: 55,  # right_subclavian_artery
            125: 62,  # superior_vena_cava
            146: 54,  # brachiocephalic_trunk (new key for only-in-2nd-list)
        }

        self.lung_ids_map = {
            28: 10,  # left_lung_upper_lobe / lung_upper_lobe_left
            29: 11,  # left_lung_lower_lobe / lung_lower_lobe_left
            30: 12,  # right_lung_upper_lobe / lung_upper_lobe_right
            31: 13,  # right_lung_middle_lobe / lung_middle_lobe_right
            32: 14,  # right_lung_lower_lobe / lung_lower_lobe_right
            57: 16,  # trachea
            132: -1,  # airway (only in list1)
            147: 15,  # esophagus (only in list2)
        }

        self.bone_ids_map = {
            33: 27,  # vertebrae L5 / vertebra_L5
            34: 28,
            35: 29,
            36: 30,
            37: 31,
            38: 32,
            39: 33,
            40: 34,
            41: 35,
            42: 36,
            43: 37,
            44: 38,
            45: 39,
            46: 40,
            47: 41,
            48: 42,
            49: 43,
            50: 44,
            51: 45,
            52: 46,
            53: 47,
            54: 48,
            55: 49,
            56: 50,
            63: 92,  # left_rib_1 / rib_left_1
            64: 93,
            65: 94,
            66: 95,
            67: 96,
            68: 97,
            69: 98,
            70: 99,
            71: 100,
            72: 101,
            73: 102,
            74: 103,
            75: 104,  # right_rib_1 / rib_right_1
            76: 105,
            77: 106,
            78: 107,
            79: 108,
            80: 109,
            81: 110,
            82: 111,
            83: 112,
            84: 113,
            85: 114,
            86: 115,
            87: 69,  # left_humerus / humerus_left
            88: 70,  # right_humerus / humerus_right
            89: 71,  # left_scapula / scapula_left
            90: 72,  # right_scapula / scapula_right
            91: 73,  # left_clavicula / clavicula_left
            92: 74,  # right_clavicula / clavicula_right
            93: 75,  # left_femur / femur_left
            94: 76,  # right_femur / femur_right
            95: 77,  # left_hip / hip_left
            96: 78,  # right_hip / hip_right
            120: 91,  # skull
            122: 116,  # sternum
            114: 117,  # costal_cartilages
            127: 26,  # vertebrae_S1 / vertebra_S1
        }

        self.soft_tissue_ids_map = {
            3: 1,  # spleen
            5: 2,  # right_kidney / kidney_right
            14: 3,  # left_kidney / kidney_left
            10: 4,  # gallbladder
            1: 5,  # liver
            12: 6,  # stomach
            4: 7,  # pancreas
            8: 8,  # right_adrenal_gland / adrenal_gland_right
            9: 9,  # left_adrenal_gland / adrenal_gland_left
            126: 17,  # thyroid_gland
            19: 18,  # small_bowel
            13: 19,  # duodenum
            62: 20,  # colon
            15: 21,  # bladder / urinary_bladder
            118: 22,  # prostate
            121: -1,  # spinal_cord (only in list1)
            22: 90,  # brain
            133: 133,  # soft_tissue
            # Only-in-second-list below:
            148: 25,  # sacrum (new unique key)
            149: 80,  # gluteus_maximus_left (new unique key)
            150: 81,  # gluteus_maximus_right (new unique key)
            151: 82,  # gluteus_medius_left (new unique key)
            152: 83,  # gluteus_medius_right (new unique key)
            153: 84,  # gluteus_minimus_left (new unique key)
            154: 85,  # gluteus_minimus_right (new unique key)
        }

        self.vista3d_to_totseg_ids_map = {
            **self.heart_ids_map,
            **self.major_vessels_ids_map,
            **self.lung_ids_map,
            **self.bone_ids_map,
            **self.soft_tissue_ids_map,
        }

        self.totseg_to_vista3d_ids_map = {
            v: k for k, v in self.vista3d_to_totseg_ids_map.items() if v != -1
        }

    def ensemble_segmentation(
        self, labelmap_vista: itk.image, labelmap_totseg: itk.image
    ) -> itk.image:
        """
        Combine two segmentation results using label mapping and priority rules.

        Args:
            labelmap_vista (itk.image): The VISTA3D segmentation result.
            labelmap_totseg (itk.image): The TotalSegmentator segmentation result.

        Returns:
            itk.image: The combined segmentation result.
        """

        self.log_info("Running ensemble segmentation: combining results")

        labelmap_vista_arr = itk.GetArrayFromImage(labelmap_vista)
        labelmap_totseg_arr = itk.GetArrayFromImage(labelmap_totseg)
        self.log_info("Segmentations loaded")

        results_arr = np.zeros_like(labelmap_vista_arr)

        self.log_info("Setting interpolators")
        labelmap_vista_interp = itk.LabelImageGaussianInterpolateImageFunction.New(
            labelmap_vista
        )
        labelmap_totseg_interp = itk.LabelImageGaussianInterpolateImageFunction.New(
            labelmap_totseg
        )

        self.log_info("Iterating through labelmaps")
        lastidx0 = -1
        total_slices = labelmap_vista_arr.shape[0]
        for idx in np.ndindex(labelmap_vista_arr.shape):
            if idx[0] != lastidx0:
                if idx[0] % 10 == 0 or idx[0] == total_slices - 1:
                    self.log_progress(
                        idx[0] + 1, total_slices, prefix="Processing slices"
                    )
                lastidx0 = idx[0]
            # Skip if both are zero
            vista_label = labelmap_vista_arr[idx]
            totseg_label = labelmap_totseg_arr[idx]
            if vista_label == 0 and totseg_label == 0:
                continue

            totseg_vista_label = self.totseg_to_vista3d_ids_map.get(totseg_label, 0)
            if vista_label == 0:
                results_arr[idx] = totseg_vista_label
            elif totseg_label == 0:
                results_arr[idx] = vista_label
            else:
                # print("Conflict detected at", idx, vista_label, totseg_label, end="", flush=True)
                # Softtissue label in Vista3D is a catch-all label,
                #  so use the TotalSegmentator label instead
                label = totseg_vista_label
                if vista_label != 133:
                    for sigma in [2, 4, 8]:
                        labelmap_vista_interp.SetSigma([sigma, sigma, sigma])
                        labelmap_totseg_interp.SetSigma([sigma, sigma, sigma])
                        tmp_vista = labelmap_vista_interp.EvaluateAtIndex(idx[::-1])
                        tmp_totseg = labelmap_totseg_interp.EvaluateAtIndex(idx[::-1])
                        # print("   ", tmp_vista, tmp_totseg, end="", flush=True)
                        if tmp_vista == 0 and tmp_totseg == 0:
                            label = 0
                            # print("...agreeing on 0", flush=True)
                            break
                        tmp_totseg_vista = self.totseg_to_vista3d_ids_map.get(
                            tmp_totseg, 0
                        )
                        # print(f"({tmp_totseg_vista})", end="", flush=True)
                        if tmp_vista == tmp_totseg_vista:
                            label = tmp_vista
                            # print("...agreeing on", label, flush=True)
                            break
                        label = totseg_vista_label
                # print("   assigning =", label, flush=True)
                results_arr[idx] = label
                labelmap_vista_arr[idx] = label
                totseg_label = self.vista3d_to_totseg_ids_map.get(vista_label, 0)
                totseg_label = max(totseg_label, 0)
                labelmap_totseg_arr[idx] = totseg_label

        results_arr = results_arr.reshape(labelmap_vista_arr.shape)
        results_image = itk.GetImageFromArray(results_arr)
        results_image.CopyInformation(labelmap_vista)

        return results_image

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run VISTA3D on the preprocessed image and return result.

        Args:
            preprocessed_image (itk.image): The preprocessed image to segment.

        Returns:
            the segmented image.
        """
        vista3d_result = SegmentChestVista3D().segment(preprocessed_image)
        vista3d_labelmap = vista3d_result["labelmap"]

        total_result = SegmentChestTotalSegmentator().segment(preprocessed_image)
        total_labelmap = total_result["labelmap"]

        ensemble_segmentation = self.ensemble_segmentation(
            vista3d_labelmap, total_labelmap
        )

        return ensemble_segmentation
