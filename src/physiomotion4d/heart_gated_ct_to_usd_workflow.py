"""
Heart-gated CT processor implementing the complete 4D CT to USD workflow.

This module implements the complete pipeline for processing 4D cardiac CT images
as demonstrated in the Heart-GatedCT experiment notebooks.
"""

import os
from typing import List, Optional

import itk
import numpy as np
import pyvista as pv
from pxr import Usd

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.convert_vtk_4d_to_usd import ConvertVTK4DToUSD
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.transform_tools import TransformTools
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools


class HeartGatedCTToUSDWorkflow:
    """
    Complete workflow for Heart-gated CT images to dynamic USD models.

    This class implements the full workflow from 4D CT images to painted USD files
    suitable for visualization in NVIDIA Omniverse.
    """

    def __init__(
        self,
        input_filenames: List[str],
        contrast_enhanced: bool,
        output_directory: str,
        project_name: str,
        reference_image_filename: Optional[str] = None,
        number_of_registration_iterations: Optional[int] = 1,
        registration_method: str = 'icon',
    ):
        """
        Initialize the Heart-gated CT to USD workflow.

        Args:
            input_filenames (List[str]): List of paths to the 3D NRRD files containing cardiac CT data.
                If there is only one file, it will be used as the 4D NRRD file.
            contrast_enhanced (bool): Whether the study uses contrast enhancement
            output_directory (str): Directory path where output files will be stored
            project_name (str): Project name for USD file organization
            reference_image_filename (Optional[str]): Path to reference image file
            number_of_registration_iterations (Optional[int]): Number of registration iterations
            registration_method (str): Registration method to use: 'ants' or 'icon' (default: 'icon')
        """
        self.input_filenames = input_filenames
        self.contrast_enhanced = contrast_enhanced
        self.output_directory = output_directory
        self.project_name = project_name
        self.reference_image_filename = reference_image_filename
        self.number_of_registration_iterations = number_of_registration_iterations

        # Validate registration method
        if registration_method not in ['ants', 'icon']:
            raise ValueError(
                f"Invalid registration_method '{registration_method}'. "
                "Must be 'ants' or 'icon'."
            )
        self.registration_method = registration_method

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Initialize processing components
        self.converter = ConvertNRRD4DTo3D()
        self.segmenter = SegmentChestTotalSegmentator()
        self.segmenter.contrast_threshold = 500

        # Initialize registration method
        if self.registration_method == 'ants':
            print(f"Initializing ANTs registration...")
            self.registrar = RegisterImagesANTs()
            self.registrar.set_modality('ct')
            self.registrar.set_transform_type('SyN')
            if (
                number_of_registration_iterations is not None
                and number_of_registration_iterations > 0
            ):
                self.registrar.set_syn_parameters(
                    reg_iterations=(
                        number_of_registration_iterations,
                        number_of_registration_iterations // 2,
                        0,
                    )
                )
        else:  # icon (default)
            print(f"Initializing ICON registration...")
            self.registrar = RegisterImagesICON()
            self.registrar.set_modality('ct')
            if (
                number_of_registration_iterations is not None
                and number_of_registration_iterations > 0
            ):
                self.registrar.set_number_of_iterations(
                    number_of_registration_iterations
                )

        self.registrar.set_mask_dilation(5)
        self.contour_tools = ContourTools()

        # Data storage for processing pipeline
        self._num_time_points = 0
        self._time_series_images = []
        self._fixed_image = None
        self._fixed_segmentation = None
        self._time_series_transforms = []
        self._reference_contours = {}

    def process(self) -> str:
        """
        Execute the complete workflow from 4D CT to dynamic USD models.

        Returns:
            str: Path to the final dynamic anatomy USD file
        """
        print("Starting Heart-gated CT processing pipeline...")

        # Load and convert data
        self._load_time_series()

        # Segment and register all frames
        self._segment_and_register_frames()

        # Generate reference contours
        self._generate_reference_contours()

        # Transform contours for each time point
        self._transform_all_contours()

        # Create USD files
        self._create_usd_files()

        print("Processing pipeline completed successfully")
        return f"{self.project_name}.dynamic_anatomy_painted.usd"

    def _load_time_series(self):
        """Load and convert 4D data to time series images."""
        print("Loading time series data...")

        if len(self.input_filenames) == 1:
            self.converter.load_nrrd_4d(self.input_filenames[0])
            self.converter.save_3d_images(
                os.path.join(
                    self.output_directory, os.path.basename(self.input_filenames[0])
                )
            )
        else:
            print(f"Loading {len(self.input_filenames)} 3D NRRD files")
            self.converter.load_nrrd_3d(self.input_filenames)

        self._num_time_points = self.converter.get_number_of_3d_images()

        # Load all time series images into memory
        for i in range(self._num_time_points):
            self._time_series_images.append(self.converter.get_3d_image(i))

        # Load reference image
        if self.reference_image_filename:
            self._fixed_image = itk.imread(self.reference_image_filename)
        else:
            # Use 70% frame as reference if none specified
            mid_frame = int(self._num_time_points * 0.7)
            self._fixed_image = self._time_series_images[mid_frame]
            itk.imwrite(
                self._fixed_image,
                os.path.join(self.output_directory, "fixed_image.mha"),
                compression=True,
            )

        print(f"Loaded {self._num_time_points} time points")

    def _segment_and_register_frames(self):
        """Segment each frame and register to reference image."""
        print("Segmenting and registering frames...")

        # Segment reference image
        print("Segmenting reference image...")
        self._fixed_segmentation = self.segmenter.segment(
            self._fixed_image, contrast_enhanced_study=self.contrast_enhanced
        )

        # Create combined masks for registration
        labelmap_mask = self._fixed_segmentation["labelmap"]
        lung_mask = self._fixed_segmentation["lung"]
        heart_mask = self._fixed_segmentation["heart"]
        major_vessels_mask = self._fixed_segmentation["major_vessels"]
        bone_mask = self._fixed_segmentation["bone"]
        soft_tissue_mask = self._fixed_segmentation["soft_tissue"]
        other_mask = self._fixed_segmentation["other"]
        contrast_mask = self._fixed_segmentation["contrast"]
        itk.imwrite(
            labelmap_mask,
            os.path.join(self.output_directory, "fixed_image_mask.mha"),
            compression=True,
        )

        # Create masks for different anatomy types
        heart_arr = itk.GetArrayFromImage(heart_mask)
        contrast_arr = itk.GetArrayFromImage(contrast_mask)
        major_vessels_arr = itk.GetArrayFromImage(major_vessels_mask)
        fixed_dynamic_mask = itk.GetImageFromArray(
            heart_arr + contrast_arr + major_vessels_arr
        )
        fixed_dynamic_mask.CopyInformation(self._fixed_image)

        lung_arr = itk.GetArrayFromImage(lung_mask)
        bone_arr = itk.GetArrayFromImage(bone_mask)
        other_arr = itk.GetArrayFromImage(other_mask)
        fixed_static_mask = itk.GetImageFromArray(lung_arr + bone_arr + other_arr)
        fixed_static_mask.CopyInformation(self._fixed_image)

        # Set up registrar with fixed image
        self.registrar.set_fixed_image(self._fixed_image)
        if self.registration_method == 'icon':
            if self.contrast_enhanced:
                self.registrar.set_mass_preservation(False)
            else:
                self.registrar.set_mass_preservation(True)

        # Process each time point
        self._time_series_transforms = []
        for i in range(self._num_time_points):
            print(f"Processing frame {i+1}/{self._num_time_points}")

            moving_image = self._time_series_images[i]

            # Register without mask first
            self.registrar.set_fixed_image_mask(None)
            result_all = self.registrar.register(moving_image)
            phi_FM_all = result_all["phi_FM"]
            phi_MF_all = result_all["phi_MF"]
            itk.transformwrite(
                phi_FM_all,
                os.path.join(self.output_directory, f"slice_{i:03d}_all_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                phi_MF_all,
                os.path.join(self.output_directory, f"slice_{i:03d}_all_BA.hdf"),
                compression=True,
            )

            # Estimate the moving dynamic mask by the inverse transform of the fixed dynamic mask
            moving_dynamic_mask = TransformTools().transform_image(
                fixed_dynamic_mask, phi_FM_all, moving_image, "nearest"
            )
            itk.imwrite(
                moving_dynamic_mask,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_mask.mha"),
                compression=True,
            )
            self.registrar.set_fixed_image_mask(fixed_dynamic_mask)
            result_dynamic = self.registrar.register(moving_image, moving_dynamic_mask)
            phi_FM_dynamic = result_dynamic["phi_FM"]
            phi_MF_dynamic = result_dynamic["phi_MF"]

            # Estimate the moving static mask by the inverse transform of the fixed static mask
            moving_static_mask = TransformTools().transform_image(
                fixed_static_mask, phi_FM_all, moving_image, "nearest"
            )
            itk.imwrite(
                moving_static_mask,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_mask.mha"),
                compression=True,
            )
            self.registrar.set_fixed_image_mask(fixed_static_mask)
            result_static = self.registrar.register(moving_image, moving_static_mask)
            phi_FM_static = result_static["phi_FM"]
            phi_MF_static = result_static["phi_MF"]

            # Store transforms
            transforms = {
                'dynamic': {'phi_FM': phi_FM_dynamic, 'phi_MF': phi_MF_dynamic},
                'static': {'phi_FM': phi_FM_static, 'phi_MF': phi_MF_static},
                'all': {'phi_FM': phi_FM_all, 'phi_MF': phi_MF_all},
            }
            itk.transformwrite(
                phi_FM_dynamic,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                phi_MF_dynamic,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_BA.hdf"),
                compression=True,
            )
            itk.transformwrite(
                phi_FM_static,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                phi_MF_static,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_BA.hdf"),
                compression=True,
            )
            self._time_series_transforms.append(transforms)

    def _generate_reference_contours(self):
        """Generate contour meshes from reference segmentation."""
        print("Generating reference contours...")

        (
            labelmap_image,
            lung_mask,
            heart_mask,
            major_vessels_mask,
            bone_mask,
            soft_tissue_mask,
            other_mask,
            contrast_mask,
        ) = self._fixed_segmentation

        # Generate all anatomy contours
        all_contours = self.contour_tools.extract_contours(labelmap_image)

        # Generate dynamic anatomy contours
        label_arr = itk.array_from_image(labelmap_image)
        heart_arr = itk.array_from_image(heart_mask)
        contrast_arr = itk.array_from_image(contrast_mask)
        major_vessels_arr = itk.array_from_image(major_vessels_mask)

        dynamic_anatomy_arr = np.maximum(heart_arr, contrast_arr)
        dynamic_anatomy_arr = np.maximum(dynamic_anatomy_arr, major_vessels_arr)
        dynamic_anatomy_arr = np.where(dynamic_anatomy_arr, label_arr, 0)
        dynamic_anatomy_image = itk.image_from_array(
            dynamic_anatomy_arr.astype(np.int16)
        )
        dynamic_anatomy_image.CopyInformation(labelmap_image)

        dynamic_contours = self.contour_tools.extract_contours(dynamic_anatomy_image)

        # Generate static anatomy contours
        lung_arr = itk.array_from_image(lung_mask)
        bone_arr = itk.array_from_image(bone_mask)
        soft_tissue_arr = itk.array_from_image(soft_tissue_mask)
        other_arr = itk.array_from_image(other_mask)

        static_anatomy_arr = lung_arr + bone_arr + soft_tissue_arr + other_arr
        static_anatomy_arr = np.where(static_anatomy_arr, label_arr, 0)
        static_anatomy_image = itk.image_from_array(static_anatomy_arr.astype(np.int16))
        static_anatomy_image.CopyInformation(labelmap_image)

        static_contours = self.contour_tools.extract_contours(static_anatomy_image)

        # Store reference contours
        self._reference_contours = {
            'all': all_contours,
            'dynamic': dynamic_contours,
            'static': static_contours,
        }

    def _transform_all_contours(self):
        """Transform contours for all time points using registration transforms."""
        print("Transforming contours for all time points...")

        self._transformed_contours = {'all': [], 'dynamic': [], 'static': []}

        for i in range(self._num_time_points):
            print(f"Transforming contours for frame {i+1}/{self._num_time_points}")

            frame_contours = {}
            for anatomy_type in ['all', 'dynamic', 'static']:
                # Get the inverse transform for this anatomy type and frame
                phi_MF = self._time_series_transforms[i][anatomy_type]['phi_MF']

                # Transform the reference contours
                transformed_contours = self.contour_tools.transform_contours(
                    self._reference_contours[anatomy_type],
                    phi_MF,
                    with_deformation_magnitude=False,
                )

                frame_contours[anatomy_type] = transformed_contours
                self._transformed_contours[anatomy_type].append(transformed_contours)

    def _create_usd_files(self):
        """Create painted USD files for all anatomy types."""
        print("Creating USD files...")

        # Create USD for each anatomy type
        for anatomy_type in ['all', 'dynamic', 'static']:
            print(f"Creating {anatomy_type} anatomy USD...")

            # Convert VTK contours to USD
            converter = ConvertVTK4DToUSD(
                self.project_name,
                self._transformed_contours[anatomy_type],
                self.segmenter.all_mask_ids,
                os.path.join(
                    self.output_directory, f"{self.project_name}.{anatomy_type}.usd"
                ),
            )
            stage = converter.convert()

            # Paint the USD file
            print(f"Painting {anatomy_type} anatomy USD...")
            output_filename = os.path.join(
                self.output_directory, f"{self.project_name}.{anatomy_type}_painted.usd"
            )
            if os.path.exists(output_filename):
                os.remove(output_filename)
            painter = USDAnatomyTools(stage)
            painter.enhance_meshes(self.segmenter)
            stage.Export(output_filename)


def main():
    """Command-line interface for Heart-gated CT processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process 4D cardiac CT images to dynamic USD models for Omniverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single 4D NRRD file
  physiomotion4d-heart-gated-ct input_4d.nrrd --contrast --output-dir ./results

  # Process multiple 3D NRRD files as time series
  physiomotion4d-heart-gated-ct frame_*.nrrd --output-dir ./results --project-name cardiac

  # Specify reference image and registration iterations
  physiomotion4d-heart-gated-ct input.nrrd --reference-image ref.mha --registration-iterations 50
        """,
    )

    parser.add_argument(
        "input_files", nargs='+', help="Path to 4D NRRD file or list of 3D NRRD files"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--project-name",
        default="cardiac_model",
        help="Project name for USD organization (default: cardiac_model)",
    )
    parser.add_argument(
        "--contrast", action="store_true", help="Indicate if study is contrast enhanced"
    )
    parser.add_argument(
        "--reference-image",
        help="Path to reference image file (default: uses 70%% time point)",
    )
    parser.add_argument(
        "--registration-iterations",
        type=int,
        default=1,
        help="Number of registration iterations (default: 1)",
    )
    parser.add_argument(
        "--registration-method",
        choices=['ants', 'icon'],
        default='icon',
        help="Registration method to use: ants or icon (default: icon)",
    )

    args = parser.parse_args()

    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return 1

    # Initialize processor
    print("Initializing Heart-gated CT processor...")
    processor = HeartGatedCTToUSDWorkflow(
        input_filenames=args.input_files,
        contrast_enhanced=args.contrast,
        output_directory=args.output_dir,
        project_name=args.project_name,
        reference_image_filename=args.reference_image,
        number_of_registration_iterations=args.registration_iterations,
        registration_method=args.registration_method,
    )

    try:
        # Execute complete workflow
        print("\nStarting Heart-gated CT processing pipeline...")
        print("=" * 60)
        final_usd = processor.process()

        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"\nOutput files created in: {args.output_dir}")
        print(f"  - {args.project_name}.dynamic_anatomy_painted.usd")
        print(f"  - {args.project_name}.static_anatomy_painted.usd")
        print(f"  - {args.project_name}.all_anatomy_painted.usd")
        print("\nYou can now open these files in NVIDIA Omniverse.")

        return 0

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
