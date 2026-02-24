"""Module for segmenting heart from chest CT images using Simpleware Medical.

This module provides the SegmentHeartSimpleware class that implements
heart segmentation using Synopsys Simpleware Medical's ASCardio module.
It inherits from SegmentAnatomyBase and provides heart-specific anatomical
structure mappings.
"""

import logging
import os
import subprocess
import sys
import tempfile

import itk
import numpy as np
from itk import TubeTK as tube

from physiomotion4d.segment_anatomy_base import SegmentAnatomyBase


class SegmentHeartSimpleware(SegmentAnatomyBase):
    """
    Heart CT segmentation using Simpleware Medical's ASCardio module.

    This class implements heart segmentation using Synopsys Simpleware Medical,
    a commercial medical image processing platform. It specifically leverages
    the ASCardio module for automated cardiac segmentation. The class handles
    the external process communication with Simpleware Medical and converts
    between ITK and Simpleware image formats.

    Simpleware Medical provides high-quality segmentation for cardiac
    structures including chambers, myocardium, and major vessels. The
    segmentation is performed by launching Simpleware Medical as an external
    process and running a Python script within the Simpleware environment.

    The class maintains specific ID mappings for:
    - Heart structures (left/right atrium, left/right ventricle, myocardium)
    - Major vessels (aorta, pulmonary artery)

    Attributes:
        target_spacing (float): Target spacing set to 1.0mm for Simpleware
        heart_mask_ids (dict): Dictionary mapping heart structure IDs to names
        major_vessels_mask_ids (dict): Dictionary mapping vessel IDs to names
        simpleware_exe_path (str): Path to Simpleware Medical executable
        simpleware_script_path (str): Path to Simpleware Python script

    Example:
        >>> segmenter = SegmentHeartSimpleware()
        >>> result = segmenter.segment(ct_image, contrast_enhanced_study=True)
        >>> labelmap = result['labelmap']
        >>> heart_mask = result['heart']
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the Simpleware Medical based heart segmentation.

        Sets up the Simpleware-specific anatomical structure ID mappings
        and processing parameters. The target spacing is set to 1.0mm which
        is optimal for cardiac segmentation in Simpleware Medical.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.target_spacing = 1.0

        # Heart structure IDs for Simpleware Medical ASCardio output
        # These IDs should match the output from the ASCardio module
        self.heart_mask_ids = {
            1: "left_ventricle",
            2: "right_ventricle",
            3: "left_atrium",
            4: "right_atrium",
            5: "myocardium",
            6: "heart",
        }

        # Major vessel IDs for Simpleware Medical ASCardio output
        self.major_vessels_mask_ids = {
            7: "aorta",
            8: "pulmonary_artery",
            9: "right_coronary_artery",
            10: "left_coronary_artery",
        }

        # Lung structures are not segmented by ASCardio
        self.lung_mask_ids = {}

        # Bone structures are not segmented by ASCardio
        self.bone_mask_ids = {}

        # Soft tissue structures are not segmented by ASCardio
        # (will be filled in by base class 'other' category)
        self.soft_tissue_mask_ids = {}

        # From Base Class
        # self.contrast_mask_ids = {135: "contrast"}

        self._trim_mask = False

        self.set_other_and_all_mask_ids()

        # Path to Simpleware Medical console executable
        self.simpleware_exe_path = "C:/Program Files/Synopsys/Simpleware Medical/X-2025.06/ConsoleSimplewareMedical.exe"

        # Path to the Simpleware Python script for heart segmentation
        self.simpleware_script_path = os.path.join(
            os.path.dirname(__file__),
            "simpleware_medical",
            "SimplewareScript_heart_segmentation.py",
        )

    def set_trim_mask_to_essentials(self, trim_mask: bool) -> None:
        """Set whether to trim mask to common and critical structures.

        Args:
            trim_mask (bool): Whether to reduce to essential.
        """
        self._trim_mask = trim_mask

    def set_simpleware_executable_path(self, path: str) -> None:
        """Set the path to the Simpleware Medical console executable.

        Args:
            path (str): Full path to ConsoleSimplewareMedical.exe

        Example:
            >>> segmenter.set_simpleware_executable_path(
            ...     "C:/Program Files/Synopsys/Simpleware Medical/X-2025.06/ConsoleSimplewareMedical.exe"
            ... )
        """
        self.simpleware_exe_path = path

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run Simpleware Medical ASCardio segmentation on the preprocessed image.

        This implementation calls Simpleware Medical as an external process,
        passing the preprocessed image via a temporary file. The Simpleware
        Python script (SimplewareScript_heart_segmentation.py) runs within the
        Simpleware environment and uses the ASCardio module for heart
        segmentation. The results are written as per-structure MHD mask files and assembled
        into a labelmap, then read back as an ITK image.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image with
                isotropic spacing and appropriate intensity scaling

        Returns:
            itk.image: The segmentation labelmap with heart and vessel labels
                from the ASCardio module

        Raises:
            FileNotFoundError: If Simpleware Medical executable is not found
            RuntimeError: If Simpleware Medical process fails
            ValueError: If output segmentation is not produced

        Note:
            Requires a valid installation of Synopsys Simpleware Medical
            with the ASCardio module. The method creates temporary files
            for input/output communication with Simpleware.

        Example:
            >>> labelmap = segmenter.segmentation_method(preprocessed_ct)
        """
        # Check if Simpleware Medical executable exists
        if not os.path.exists(self.simpleware_exe_path):
            raise FileNotFoundError(
                f"Simpleware Medical executable not found at: {self.simpleware_exe_path}"
            )

        # Check if Simpleware script exists
        if not os.path.exists(self.simpleware_script_path):
            raise FileNotFoundError(
                f"Simpleware script not found at: {self.simpleware_script_path}"
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save preprocessed image to temporary file
            tmp_input_image_file = os.path.join(tmp_dir, "input_image.nii.gz")

            self.log_info("Writing input image to: %s", tmp_input_image_file)
            itk.imwrite(preprocessed_image, tmp_input_image_file, compression=True)

            # Build command line for Simpleware Medical
            # Pass the input NIfTI file path directly as a command-line argument
            # Use --run-script to execute the Python script
            # Use --exit-after-script to close after execution
            cmd = [
                self.simpleware_exe_path,
                "--input-file",  # Use only with ConsoleSimplewareMedical.exe
                tmp_input_image_file,  # Input NIfTI file path as positional argument
                "--input-value",
                tmp_dir,
                "--run-script",
                self.simpleware_script_path,
                "--exit-after-script",
                "--no-progress",  # Use only with ConsoleSimplewareMedical.exe
                # "--no-splash",  # Use only with SimplewareMedical.exe
            ]
            user_input = "y\n"

            self.log_info("Running Simpleware Medical ASCardio segmentation...")
            self.log_info("Command: %s", " ".join(cmd))

            try:
                # Run Simpleware Medical as a subprocess. When the process exits,
                # the OS frees all of its resources (GPU, memory); no extra
                # cleanup is required. Using Popen so we can kill the process
                # tree on timeout and ensure no child processes keep holding GPU.
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=(
                        sys.platform != "win32"
                    ),  # process group on Unix
                )
                try:
                    stdout, stderr = proc.communicate(
                        input=user_input,
                        timeout=600,  # 10 minute timeout
                    )
                except subprocess.TimeoutExpired:
                    # Kill process tree so GPU/memory are released (child may have spawned others)
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                        )
                    else:
                        os.killpg(os.getpgid(proc.pid), 9)
                    proc.wait()
                    raise RuntimeError(
                        "Simpleware Medical segmentation timed out after 600 seconds"
                    )
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode, cmd, stdout, stderr
                    )
                result = type("Result", (), {"stdout": stdout, "stderr": stderr})()

                # Log output from Simpleware
                if result.stdout:
                    self.log_info("Simpleware stdout:\n%s", result.stdout)
                if result.stderr:
                    self.log_warning("Simpleware stderr:\n%s", result.stderr)

            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Simpleware Medical segmentation failed with return code {e.returncode}:\n"
                    f"stdout: {e.stdout}\nstderr: {e.stderr}"
                )

            # Simpleware's right ventricle, left atrium, right atrium correspond to
            #   the interior of those regions.
            mask_ids_of_interior_regions = [1, 2, 3, 4]

            # Check if output file was created
            sz = [s for s in preprocessed_image.GetLargestPossibleRegion().GetSize()]
            sz = sz[::-1]
            labelmap_array = np.zeros(sz, dtype=np.uint8)
            interior_array = np.zeros(sz, dtype=np.uint8)
            for mask_id, mask_name in self.all_mask_ids.items():
                output_file = os.path.join(tmp_dir, f"mask_{mask_name}.mhd")
                if os.path.exists(output_file):
                    mask_image = itk.imread(output_file)
                    mask_array = itk.GetArrayFromImage(mask_image).astype(np.uint8)
                    if mask_id in mask_ids_of_interior_regions:
                        tmp_array = (mask_array > 128).astype(np.uint8)
                        interior_array = np.where(
                            interior_array == 0, tmp_array, interior_array
                        )
                    mask_array = (mask_array > 128) * mask_id
                    labelmap_array = np.where(
                        labelmap_array == 0, mask_array, labelmap_array
                    )

            interior_image = itk.GetImageFromArray(interior_array.astype(np.uint8))
            interior_image.CopyInformation(preprocessed_image)
            imMath = tube.ImageMath.New(interior_image)
            spacing = interior_image.GetSpacing()
            imMath.Dilate(round(7 / spacing[0]), 1, 0)
            imMath.Erode(round(4 / spacing[0]), 1, 0)
            exterior_image = imMath.GetOutputUChar()
            exterior_array = itk.GetArrayFromImage(exterior_image)
            mask_id = 6  # Heart mask id
            exterior_array = exterior_array * mask_id
            labelmap_array = np.where(
                labelmap_array == 0, exterior_array, labelmap_array
            )

            if not np.any(labelmap_array != 0):
                raise ValueError(
                    "Simpleware Medical produced no segmentation output: no mask_*.mhd "
                    "files found or all masks are empty. Check Simpleware logs above and "
                    "ensure the ASCardio module ran successfully."
                )

            labelmap_image = itk.GetImageFromArray(labelmap_array.astype(np.uint8))
            labelmap_image.CopyInformation(preprocessed_image)

            if self._trim_mask:
                labelmap_image = self.trim_mask_to_essentials(labelmap_image)

        return labelmap_image

    def trim_mask_to_essentials(self, labelmap_image: itk.image) -> itk.image:
        """Trim mask to essentials."""

        # Reference code for cropping aorta and pulmonary artery to
        #    portions adjacent to the heart.
        # Trim z-axis
        # z = labelmap_array.shape[2] - 1
        # z_classes = np.unique(labelmap_array[z, :, :])
        # heart_count = np.sum((c in [1, 2, 3, 4, 5]) for c in z_classes)
        # while heart_count < 3 and z > 0:
        #     z -= 1
        #     z_classes = np.unique(labelmap_array[z, :, :])
        #     heart_count = np.sum((c in [1, 2, 3, 4, 5]) for c in z_classes)
        # if z < labelmap_array.shape[2] - 3:
        # labelmap_array[(z + 3) :, :, :] = 0

        # In labelmap,
        #  if pixel is in keep_mask, was left or right atrium, then keep as
        #     left or right atrium

        #  1) Erase Heart and Myo label
        labelmap_arr = itk.array_from_image(labelmap_image)

        heart_arr = itk.array_from_image(labelmap_image)
        heart_arr[heart_arr == 6] = 0
        heart_arr[heart_arr == 5] = 0

        img = itk.image_from_array(heart_arr)
        img.CopyInformation(labelmap_image)
        imMath = tube.ImageMath.New(img)

        #  2) Erode then Dilate Left Atrium label to clip vessels
        spacing = labelmap_image.GetSpacing()
        imMath.Erode(round(7 / spacing[0]), 3, 0)
        imMath.Dilate(round(7 / spacing[0]), 3, 0)

        #  3) Erode then Dilate Right Atrium label to clip vessels
        imMath.Erode(round(7 / spacing[0]), 4, 0)
        imMath.Dilate(round(7 / spacing[0]), 4, 0)
        simple_img = imMath.GetOutput()
        simple_arr = itk.array_from_image(simple_img)

        #  Keep the largest component of the left atrium
        simple_arr_3 = simple_arr.copy()
        simple_arr_3[simple_arr_3 != 3] = 0
        simple_arr_3[simple_arr_3 == 3] = 1
        simple_img_3 = itk.image_from_array(simple_arr_3)
        connComp = tube.SegmentConnectedComponents.New(simple_img_3)
        connComp.SetKeepOnlyLargestComponent(True)
        connComp.Update()
        mask_img_3 = connComp.GetOutput()
        mask_arr_3 = itk.array_from_image(mask_img_3)
        simple_arr_3[mask_arr_3 == 0] = 0

        #  Keep the largest component of the right atrium
        simple_arr_4 = simple_arr.copy()
        simple_arr_4[simple_arr_4 != 4] = 0
        simple_arr_4[simple_arr_4 == 4] = 1
        simple_img_4 = itk.image_from_array(simple_arr_4)
        connComp = tube.SegmentConnectedComponents.New(simple_img_4)
        connComp.SetKeepOnlyLargestComponent(True)
        connComp.Update()
        mask_img_4 = connComp.GetOutput()
        mask_arr_4 = itk.array_from_image(mask_img_4)
        simple_arr_4[mask_arr_4 == 0] = 0

        #  Replace the left and right atrium labels with the largest components
        simple_arr[simple_arr == 3] = 0
        simple_arr[simple_arr == 4] = 0
        simple_arr[simple_arr_3 > 0] = 3
        simple_arr[simple_arr_4 > 0] = 4
        simple_img = itk.image_from_array(simple_arr)
        simple_img.CopyInformation(labelmap_image)

        #  4) Dilate all others = keep_mask
        keep_mask_arr = heart_arr.copy()
        keep_mask_arr[keep_mask_arr == 2] = 1
        keep_mask_arr[keep_mask_arr == 5] = 1
        keep_mask_arr[keep_mask_arr != 1] = 0
        keep_mask = itk.image_from_array(keep_mask_arr)
        keep_mask.CopyInformation(labelmap_image)
        imMath.SetInput(keep_mask)
        imMath.Dilate(round(7 / spacing[0]), 1, 0)
        keep_mask = imMath.GetOutput()
        keep_mask_arr = itk.array_from_image(keep_mask)

        #  Add the left and right atrium labels to the keep_mask
        heart_arr = heart_arr * keep_mask_arr
        heart_arr[simple_arr == 3] = 3
        heart_arr[simple_arr == 4] = 4
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        #  Dilate the keep_mask to simulate 3mm (heart)
        keep_mask_arr = heart_arr.copy()
        keep_mask_arr[keep_mask_arr == 1] = 0
        keep_mask_arr[keep_mask_arr > 0] = 1
        keep_mask = itk.image_from_array(keep_mask_arr)
        keep_mask.CopyInformation(labelmap_image)
        imMath.SetInput(keep_mask)
        imMath.Dilate(round(5 / spacing[0]), 1, 0)
        imMath.Erode(round(2 / spacing[0]), 1, 0)
        heart_mask = imMath.GetOutput()

        #  Insert the heart and myo labels back into the labelmap
        heart_mask_arr = itk.array_from_image(heart_mask)
        heart_mask_arr[heart_arr > 0] = 0
        heart_arr[heart_mask_arr > 0] = 6
        heart_arr_myo = itk.array_from_image(labelmap_image)
        heart_arr[heart_arr_myo == 5] = 5
        heart_arr[heart_arr_myo == 1] = 1
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        #  Add in missing pieces / gaps of the myocardium
        lv_arr = heart_arr.copy()
        lv_arr[lv_arr != 1] = 0
        lv_img = itk.image_from_array(lv_arr)
        lv_img.CopyInformation(labelmap_image)
        imMath.SetInput(lv_img)
        imMath.Dilate(round(2 / spacing[0]), 1, 0)
        lv_img = imMath.GetOutput()
        lv_arr = itk.array_from_image(lv_img)
        lv_arr = lv_arr * 5  # Myocardium label is 5

        #  Add the gap-filled myocardium back into the labelmap
        heart_arr = np.where(heart_arr == 0, lv_arr, heart_arr)
        # Eliminate overlap with other labels
        heart_arr = np.where(labelmap_arr > 6, 0, heart_arr)
        heart_img = itk.image_from_array(heart_arr)
        heart_img.CopyInformation(labelmap_image)

        return heart_img
