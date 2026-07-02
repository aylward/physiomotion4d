"""
Image-to-USD workflow implementing the complete 3D/4D image to USD pipeline.

This module implements the complete pipeline for processing 3D or 4D medical
images (e.g. cardiac and respiratory gated CT studies) into dynamic USD
models.  4D image arrays follow the (X, Y, Z, T) axis convention used
throughout PhysioMotion4D.
"""

import logging
import os
from typing import Optional, cast

import itk
import numpy as np
import pyvista as pv

from .contour_tools import ContourTools
from .convert_image_4d_to_3d import ConvertImage4DTo3D
from .convert_vtk_to_usd import ConvertVTKToUSD
from .physiomotion4d_base import PhysioMotion4DBase
from .register_images_base import RegisterImagesBase
from .register_images_icon import RegisterImagesICON
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator import SegmentChestTotalSegmentator
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools


class WorkflowConvertImageToUSD(PhysioMotion4DBase):
    """
    Complete workflow for converting 4D CT images to dynamic USD models.

    This class implements the full workflow from 4D CT images to painted USD files
    suitable for visualization in NVIDIA Omniverse.

    ``segmentation_method`` and ``registration_method`` accept a
    pre-configured :class:`SegmentAnatomyBase` / :class:`RegisterImagesBase`
    instance. Configure backend-specific parameters (iteration counts,
    trim_branches, mass preservation, etc.) on the instance before passing
    it in. Defaults to :class:`SegmentChestTotalSegmentator` /
    :class:`RegisterImagesICON` when omitted.
    """

    def __init__(
        self,
        input_filenames: list,
        contrast_enhanced: bool,
        output_directory: str,
        project_name: str,
        reference_image_filename: Optional[str] = None,
        segmentation_method: Optional[SegmentAnatomyBase] = None,
        registration_method: Optional[RegisterImagesBase] = None,
        times_per_second: float = 24.0,
        log_level: int | str = logging.INFO,
        save_registered_images: bool = True,
        save_registration_transforms: bool = True,
        save_labelmaps: bool = True,
    ):
        """
        Initialize the image-to-USD workflow.

        Args:
            input_filenames (List): One or more image sources for the time
                series.  A single entry may be a 4D image file (NRRD/NIfTI/MHA
                in (X, Y, Z, T) order), a 3D image file, or a directory holding
                a DICOM series (3D or 4D).  Multiple entries are treated as a
                pre-split list of 3D images, one per time point.  All entries
                are routed through :class:`ConvertImage4DTo3D` so any
                ITK-readable format is accepted.
            contrast_enhanced (bool): Whether the study uses contrast enhancement
            output_directory (str): Directory path where output files will be stored
            project_name (str): Project name for USD file organization
            reference_image_filename (Optional[str]): Path to reference image file
            segmentation_method (Optional[SegmentAnatomyBase]): Segmentation
                backend instance. Defaults to a new
                :class:`SegmentChestTotalSegmentator` when None.
            registration_method (Optional[RegisterImagesBase]): Registration
                backend instance. Defaults to a new :class:`RegisterImagesICON`
                when None. A caller-supplied instance is mutated (fixed
                image/mask/modality) during :meth:`process` - pass a fresh
                instance per run unless intentionally reusing state.
            times_per_second: Frames per second for animated USD time series.
                Defaults to 24.0, matching the underlying VTK-to-USD converter.
            log_level: Logging level (default: logging.INFO)
            save_registered_images: Write registered image intermediates to
                output_directory when True
            save_registration_transforms: Write registration transforms to
                output_directory when True
            save_labelmaps: Write segmentation labelmaps and registration masks to
                output_directory when True

        Raises:
            TypeError: If segmentation_method is neither None nor a
                SegmentAnatomyBase instance, or registration_method is
                neither None nor a RegisterImagesBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.input_filenames = input_filenames
        self.contrast_enhanced = contrast_enhanced
        self.output_directory = output_directory
        self.project_name = project_name
        self.reference_image_filename = reference_image_filename
        self.save_registered_images = save_registered_images
        self.save_registration_transforms = save_registration_transforms
        self.save_labelmaps = save_labelmaps
        self.times_per_second = times_per_second

        if segmentation_method is None:
            segmentation_method = SegmentChestTotalSegmentator(log_level=log_level)
            segmentation_method.contrast_threshold = 500
        elif not isinstance(segmentation_method, SegmentAnatomyBase):
            raise TypeError(
                "segmentation_method must be a SegmentAnatomyBase instance or None"
            )
        self.segmenter: SegmentAnatomyBase = segmentation_method

        if registration_method is None:
            registration_method = RegisterImagesICON(log_level=log_level)
            registration_method.set_mass_preservation(not contrast_enhanced)
        elif not isinstance(registration_method, RegisterImagesBase):
            raise TypeError(
                "registration_method must be a RegisterImagesBase instance or None"
            )
        registration_method.set_modality("ct")
        registration_method.set_mask_dilation(5)
        self.registrar: RegisterImagesBase = registration_method

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Initialize processing components
        self.converter = ConvertImage4DTo3D(log_level=log_level)
        self.contour_tools = ContourTools()

        # Data storage for processing pipeline
        self._num_time_points = 0
        self._time_series_images: list[itk.Image] = []
        self._fixed_image: Optional[itk.Image] = None
        self._fixed_segmentation: Optional[dict[str, itk.Image]] = None
        self._time_series_transforms: list[dict[str, dict[str, itk.Transform]]] = []
        self._reference_contours: dict[str, pv.PolyData] = {}

    def _output_path(self, filename: str) -> str:
        """Return an output path inside the workflow output directory."""
        return os.path.join(self.output_directory, filename)

    def _write_image_if_enabled(
        self,
        image: itk.Image,
        filename: str,
        enabled: bool,
    ) -> None:
        """Write an image artifact when its save option is enabled."""
        if enabled:
            itk.imwrite(image, self._output_path(filename), compression=True)

    def _write_transform_if_enabled(
        self,
        transform: itk.Transform,
        filename: str,
    ) -> None:
        """Write a transform artifact when transform saving is enabled."""
        if self.save_registration_transforms:
            itk.transformwrite(
                transform,
                self._output_path(filename),
                compression=True,
            )

    def _write_registered_image_if_enabled(self, filename: str) -> None:
        """Write the current registered moving image when image saving is enabled."""
        if self.save_registered_images:
            itk.imwrite(
                self.registrar.get_registered_image(),
                self._output_path(filename),
                compression=True,
            )

    def process(self) -> str:
        """
        Execute the complete workflow from 4D CT to dynamic USD models.

        Returns:
            str: Path to the final dynamic anatomy USD file
        """
        self.log_section("Image-to-USD Processing Pipeline")

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

        self.log_info("Processing pipeline completed successfully")
        return f"{self.project_name}.dynamic_painted.usd"

    def _load_time_series(self) -> None:
        """Load and convert 4D data to time series images."""
        self.log_info("Loading time series data...")

        self._time_series_images = []
        self._num_time_points = 0

        if len(self.input_filenames) == 1:
            self.converter.load_image_4d(self.input_filenames[0])
            self.converter.save_3d_images(
                self.output_directory,
                os.path.basename(self.input_filenames[0]),
            )
            self._num_time_points = self.converter.get_number_of_3d_images()
            for i in range(self._num_time_points):
                self._time_series_images.append(self.converter.get_3d_image(i))
        else:
            self.log_info("Loading %d 3D image files", len(self.input_filenames))
            self._time_series_images = [
                itk.imread(path) for path in self.input_filenames
            ]
            self._num_time_points = len(self._time_series_images)

        if self._num_time_points <= 0:
            raise ValueError("No time-series images were produced from input data")
        if not self._time_series_images:
            raise ValueError("No time-series images were loaded from input data")

        # Load reference image
        if self.reference_image_filename:
            self._fixed_image = itk.imread(self.reference_image_filename)
        else:
            # Use 70% frame as reference if none specified.
            reference_frame = int(self._num_time_points * 0.7)
            self._fixed_image = self._time_series_images[reference_frame]
            self._write_image_if_enabled(
                self._fixed_image,
                "fixed_image.mha",
                self.save_registered_images,
            )

        self.log_info("Loaded %d time points", self._num_time_points)

    def _optional_mask(self, key: str) -> itk.Image:
        """Return ``self._fixed_segmentation[key]`` or a zero mask if absent.

        Segmenters expose only the anatomy groups they actually produce
        (see ``SegmentAnatomyBase.create_anatomy_group_masks`` —
        ``SegmentHeartSimpleware`` omits ``lung``/``bone``/``soft_tissue``/
        ``contrast``, while ``SegmentChestTotalSegmentator`` provides them).
        Treating missing groups as empty (uint8 zeros matching
        ``self._fixed_image``) lets downstream mask arithmetic run
        uniformly for any ``self.segmenter`` choice.
        """
        assert self._fixed_segmentation is not None, "Fixed segmentation must be set"
        assert self._fixed_image is not None, "Fixed image must be set"
        if key in self._fixed_segmentation:
            return self._fixed_segmentation[key]
        zeros = np.zeros(itk.array_from_image(self._fixed_image).shape, dtype=np.uint8)
        empty = itk.GetImageFromArray(zeros)
        empty.CopyInformation(self._fixed_image)
        return empty

    def _segment_and_register_frames(self) -> None:
        """Segment each frame and register to reference image."""
        self.log_info("Segmenting and registering frames...")

        # Segment reference image
        self.log_info("Segmenting reference image...")
        assert self._fixed_image is not None, "Fixed image must be set"
        self._fixed_segmentation = self.segmenter.segment(
            self._fixed_image, contrast_enhanced_study=self.contrast_enhanced
        )

        # Create combined masks for registration. Optional groups
        # (lung/bone/contrast) are absent from segmenters that do not produce
        # them (e.g., SegmentHeartSimpleware) — fall back to empty masks so
        # the static/dynamic-mask sums below remain well-defined.
        assert self._fixed_segmentation is not None, "Fixed segmentation must be set"
        labelmap_mask = self._fixed_segmentation["labelmap"]
        heart_mask = self._fixed_segmentation["heart"]
        major_vessels_mask = self._fixed_segmentation["major_vessels"]
        other_mask = self._fixed_segmentation["other"]
        lung_mask = self._optional_mask("lung")
        bone_mask = self._optional_mask("bone")
        contrast_mask = self._optional_mask("contrast")
        self._write_image_if_enabled(
            labelmap_mask,
            "fixed_image_mask.mha",
            self.save_labelmaps,
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

        # Process each time point
        self._time_series_transforms = []
        for i in range(self._num_time_points):
            self.log_progress(i + 1, self._num_time_points, prefix="Processing frames")

            moving_image = self._time_series_images[i]

            # Register without mask first
            self.registrar.set_fixed_mask(None)
            result_all = self.registrar.register(moving_image)
            inverse_transform_all = cast(itk.Transform, result_all["inverse_transform"])
            forward_transform_all = cast(itk.Transform, result_all["forward_transform"])
            self._write_transform_if_enabled(
                inverse_transform_all,
                f"slice_{i:03d}_all_AB.hdf",
            )
            self._write_transform_if_enabled(
                forward_transform_all,
                f"slice_{i:03d}_all_BA.hdf",
            )
            self._write_registered_image_if_enabled(f"slice_{i:03d}_registered.mha")

            # Estimate the moving dynamic mask from the fixed dynamic mask.
            moving_dynamic_mask = TransformTools().transform_image(
                fixed_dynamic_mask, inverse_transform_all, moving_image, "nearest"
            )
            self._write_image_if_enabled(
                moving_dynamic_mask,
                f"slice_{i:03d}_dynamic_mask.mha",
                self.save_labelmaps,
            )
            self.registrar.set_fixed_mask(fixed_dynamic_mask)
            result_dynamic = self.registrar.register(moving_image, moving_dynamic_mask)
            inverse_transform_dynamic = cast(
                itk.Transform, result_dynamic["inverse_transform"]
            )
            forward_transform_dynamic = cast(
                itk.Transform, result_dynamic["forward_transform"]
            )
            self._write_registered_image_if_enabled(
                f"slice_{i:03d}_dynamic_registered.mha"
            )

            # Estimate the moving static mask from the fixed static mask.
            moving_static_mask = TransformTools().transform_image(
                fixed_static_mask, inverse_transform_all, moving_image, "nearest"
            )
            self._write_image_if_enabled(
                moving_static_mask,
                f"slice_{i:03d}_static_mask.mha",
                self.save_labelmaps,
            )
            self.registrar.set_fixed_mask(fixed_static_mask)
            result_static = self.registrar.register(moving_image, moving_static_mask)
            inverse_transform_static = cast(
                itk.Transform, result_static["inverse_transform"]
            )
            forward_transform_static = cast(
                itk.Transform, result_static["forward_transform"]
            )
            self._write_registered_image_if_enabled(
                f"slice_{i:03d}_static_registered.mha"
            )

            # Store transforms
            transforms = {
                "dynamic": {
                    "inverse_transform": inverse_transform_dynamic,
                    "forward_transform": forward_transform_dynamic,
                },
                "static": {
                    "inverse_transform": inverse_transform_static,
                    "forward_transform": forward_transform_static,
                },
                "all": {
                    "inverse_transform": inverse_transform_all,
                    "forward_transform": forward_transform_all,
                },
            }
            self._write_transform_if_enabled(
                inverse_transform_dynamic,
                f"slice_{i:03d}_dynamic_AB.hdf",
            )
            self._write_transform_if_enabled(
                forward_transform_dynamic,
                f"slice_{i:03d}_dynamic_BA.hdf",
            )
            self._write_transform_if_enabled(
                inverse_transform_static,
                f"slice_{i:03d}_static_AB.hdf",
            )
            self._write_transform_if_enabled(
                forward_transform_static,
                f"slice_{i:03d}_static_BA.hdf",
            )
            self._time_series_transforms.append(transforms)

    def _generate_reference_contours(self) -> None:
        """Generate contour meshes from reference segmentation."""
        self.log_info("Generating reference contours...")

        # Optional groups (lung/bone/soft_tissue/contrast) are absent from
        # segmenters that do not produce them (e.g., SegmentHeartSimpleware)
        # — fall back to empty masks so the static/dynamic-anatomy sums below
        # remain well-defined.
        assert self._fixed_segmentation is not None, "Fixed segmentation must be set"
        labelmap_image = self._fixed_segmentation["labelmap"]
        heart_mask = self._fixed_segmentation["heart"]
        major_vessels_mask = self._fixed_segmentation["major_vessels"]
        other_mask = self._fixed_segmentation["other"]
        lung_mask = self._optional_mask("lung")
        bone_mask = self._optional_mask("bone")
        soft_tissue_mask = self._optional_mask("soft_tissue")
        contrast_mask = self._optional_mask("contrast")

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
            "all": all_contours,
            "dynamic": dynamic_contours,
            "static": static_contours,
        }

    def _transform_all_contours(self) -> None:
        """Transform contours for all time points using registration transforms."""
        self.log_info("Transforming contours for all time points...")

        self._transformed_contours: dict[str, list[pv.PolyData]] = {
            "all": [],
            "dynamic": [],
            "static": [],
        }

        for i in range(self._num_time_points):
            self.log_progress(
                i + 1, self._num_time_points, prefix="Transforming contours"
            )

            frame_contours = {}
            for anatomy_type in ["all", "dynamic", "static"]:
                # Get the forward transform for this anatomy type and frame
                forward_transform = self._time_series_transforms[i][anatomy_type][
                    "forward_transform"
                ]

                # Transform the reference contours
                transformed_contours = self.contour_tools.transform_contours(
                    self._reference_contours[anatomy_type],
                    forward_transform,
                    with_deformation_magnitude=False,
                )

                frame_contours[anatomy_type] = transformed_contours
                self._transformed_contours[anatomy_type].append(transformed_contours)

    def _create_usd_files(self) -> None:
        """Create painted USD files for all anatomy types."""
        self.log_info("Creating USD files...")

        # Create USD for each anatomy type
        for anatomy_type in ["all", "dynamic", "static"]:
            self.log_info("Creating %s anatomy USD...", anatomy_type)

            # Convert VTK contours to USD. Forwarding the segmenter so labels
            # land under /World/{project}/{type}/{label_name} (and materials
            # under /World/Looks/{type}/{label_name}_material).
            converter = ConvertVTKToUSD(
                self.project_name,
                self._transformed_contours[anatomy_type],
                self.segmenter.taxonomy.all_labels(),
                segmenter=self.segmenter,
                times_per_second=self.times_per_second,
                log_level=self.log_level,
            )
            usd_file = os.path.join(
                self.output_directory, f"{self.project_name}.{anatomy_type}.usd"
            )
            stage = converter.convert(usd_file)

            # Paint the USD file
            self.log_info("Painting %s anatomy USD...", anatomy_type)
            output_filename = os.path.join(
                self.output_directory, f"{self.project_name}.{anatomy_type}_painted.usd"
            )
            if os.path.exists(output_filename):
                os.remove(output_filename)
            painter = USDAnatomyTools(stage)
            painter.enhance_meshes(self.segmenter)
            stage.Export(output_filename)
