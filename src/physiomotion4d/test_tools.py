"""
Test utilities for comparing images in pytest.

Provides TestTools for baseline vs results comparison with configurable
tolerances. All image I/O uses ITK with .mha (compressed); 2D and 3D
images are passed as itk.Image at the API level.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import itk
import numpy as np

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase

# Repo root: src/physiomotion4d/test_tools.py -> parent.parent.parent
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Set by tests/conftest.py from pytest --create-baselines; applies to entire run
_create_baseline_if_missing = False


def set_create_baseline_if_missing(value: bool) -> None:
    """Set whether to create baseline files when missing (used by pytest conftest)."""
    global _create_baseline_if_missing
    _create_baseline_if_missing = value


class TestTools(PhysioMotion4DBase):
    """
    Utilities for pytest image comparison: baseline directory, results directory,
    and comparison with configurable tolerances. Inherits from PhysioMotion4DBase
    for logging. All image I/O uses ITK .mha with compression.
    """

    def __init__(
        self,
        results_dir: Path,
        baselines_dir: Path,
        class_name: str,
        *,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(class_name=class_name, log_level=log_level)

        self._results_dir = results_dir / class_name
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._baselines_dir = baselines_dir / class_name
        self._baselines_dir.mkdir(parents=True, exist_ok=True)

        self._last_image_per_pixdel_absolute_error_tol: float | None = None
        self._last_image_number_of_pixels_above_tol: int | None = None
        self._last_image_max_number_of_pixels_above_tol: int | None = None
        self._last_image_total_absolute_error: float | None = None
        self._last_image_total_absolute_error_tol: float | None = None
        self._last_image_difference_image: Any = (
            None  # itk.Image (type depends on template)
        )

        self._last_transform_per_value_absolute_error_tol: float | None = None
        self._last_transform_number_of_values_above_tol: int | None = None
        self._last_transform_max_number_of_values_above_tol: int | None = None
        self._last_transform_total_absolute_error: float | None = None
        self._last_transform_total_absolute_error_tol: float | None = None
        self._last_transform_difference_transform: Any = (
            None  # itk.Transform (type depends on template)
        )

    def compare_2d_to_3d_slice(
        self,
        image_2d: Any,
        image_3d: Any,
        slice_index: int,
        axis: int = 0,
        *,
        per_pixel_absolute_error_tol: float = 0.0,
        max_number_of_pixels_above_tol: int = 0,
        total_absolute_error_tol: float = 0.0,
    ) -> bool:
        """
        Compare a 2D itk.Image to a slice of a 3D itk.Image. Converts to numpy only for computing differences.
        Stores the difference image and counts for later access via image_difference(),
        image_pass_fail_and_pixels_above_tolerance(), and image_pass_fail_and_total_absolute_error().

        Returns:
            True if number_of_pixels_above_tol <= max_number_of_pixels_above_tol and
            total_absolute_error <= total_absolute_error_tol; False otherwise.
        """
        arr_2d = np.asarray(itk.array_from_image(image_2d), dtype=np.float64)
        arr_3d = np.asarray(itk.array_from_image(image_3d), dtype=np.float64)
        if arr_2d.ndim != 2:
            raise ValueError("image_2d must be 2D")
        if arr_3d.ndim != 3:
            raise ValueError("image_3d must be 3D")

        slice_3d = np.take(arr_3d, slice_index, axis=axis)
        slice_2d = np.squeeze(slice_3d) if slice_3d.ndim == 3 else slice_3d
        if slice_2d.shape != arr_2d.shape:
            raise ValueError(
                f"Shape mismatch: image_2d {arr_2d.shape} vs 3D slice {slice_2d.shape} (axis={axis}, index={slice_index})"
            )

        diff = arr_2d - slice_2d
        diff_magnitude = np.abs(diff)
        total_absolute_error = float(np.sum(diff_magnitude))
        number_of_pixels_above_tol = int(
            np.sum(diff_magnitude > per_pixel_absolute_error_tol)
        )

        self._last_image_number_of_pixels_above_tol = number_of_pixels_above_tol
        self._last_image_total_absolute_error = total_absolute_error

        self._last_image_per_pixel_absolute_error_tol = per_pixel_absolute_error_tol
        self._last_image_max_number_of_pixels_above_tol = max_number_of_pixels_above_tol
        self._last_image_total_absolute_error_tol = total_absolute_error_tol

        self._last_image_difference_image = itk.image_from_array(
            diff_magnitude.astype(np.float64)
        )

        passed = (
            number_of_pixels_above_tol <= max_number_of_pixels_above_tol
            and total_absolute_error <= total_absolute_error_tol
        )
        if passed:
            self.log_info(
                "PASS: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )
        else:
            self.log_error(
                "FAIL: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )
        return passed

    def image_pass_fail_and_pixels_above_tolerance(self) -> tuple[bool, int]:
        """
        Return (pass, value) for number of pixels above tolerance from the most recent compare_2d_to_3d_slice call.
        pass is True if value <= max_pixels_above_tol that was used in that call.
        """
        if (
            self._last_image_number_of_pixels_above_tol is None
            or self._last_image_max_number_of_pixels_above_tol is None
        ):
            raise RuntimeError("No previous compare_2d_to_3d_slice call")
        val = self._last_image_number_of_pixels_above_tol
        passed = val <= self._last_image_max_number_of_pixels_above_tol
        return (passed, val)

    def image_pass_fail_and_total_absolute_error(self) -> tuple[bool, float]:
        """
        Return (pass, value) for total absolute error from the most recent compare_2d_to_3d_slice call.
        pass is True if value <= total_absolute_error_tol that was used in that call.
        """
        if (
            self._last_image_total_absolute_error is None
            or self._last_image_total_absolute_error_tol is None
        ):
            raise RuntimeError("No previous compare_2d_to_3d_slice call")
        val = self._last_image_total_absolute_error
        passed = val <= self._last_image_total_absolute_error_tol
        return (passed, val)

    def image_difference(self) -> Any:
        """Return the difference image (itk.Image) from the most recent compare_2d_to_3d_slice call."""
        if self._last_image_difference_image is None:
            raise RuntimeError("No previous compare_2d_to_3d_slice call")
        return self._last_image_difference_image

    def transform_pass_fail_and_number_of_values_above_tolerance(
        self,
    ) -> tuple[bool, int]:
        """
        Return (pass, value) for number of values above tolerance from the most recent compare_result_to_baseline_transform call.
        pass is True if value <= max_number_of_values_above_tol that was used in that call.
        """
        if (
            self._last_transform_number_of_values_above_tol is None
            or self._last_transform_max_number_of_values_above_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        val = self._last_transform_number_of_values_above_tol
        passed = val <= self._last_transform_max_number_of_values_above_tol
        return (passed, val)

    def transform_pass_fail_and_total_absolute_error(self) -> tuple[bool, float]:
        """
        Return (pass, value) for total absolute error from the most recent compare_result_to_baseline_transform call.
        pass is True if value <= total_absolute_error_tol that was used in that call.
        """
        if (
            self._last_transform_total_absolute_error is None
            or self._last_transform_total_absolute_error_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        val = self._last_transform_total_absolute_error
        passed = val <= self._last_transform_total_absolute_error_tol
        return (passed, val)

    def transform_difference(self) -> Any:
        """Return the difference transform (itk.Transform) from the most recent compare_result_to_baseline_transform call."""
        if self._last_transform_difference_transform is None:
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        return self._last_transform_difference_transform

    def write_result_image(self, image: Any, filename: str) -> None:
        """Write the image to the results directory."""
        itk.imwrite(image, str(self._results_dir / filename), compression=True)

    def write_result_transform(self, transform: Any, filename: str) -> None:
        """Write the transform to the results directory."""
        itk.transformwrite(
            transform, str(self._results_dir / filename), compression=True
        )

    def compare_result_to_baseline_transform(
        self,
        filename: str,
        *,
        per_value_absolute_error_tol: float = 0.0,
        max_number_of_values_above_tol: int = 0,
        total_absolute_error_tol: float = 0.0,
    ) -> bool:
        """Compare the transform to the baseline transform."""
        results_path = Path(self._results_dir / filename)
        baseline_path = Path(self._baselines_dir / filename)
        if not results_path.exists():
            raise FileNotFoundError(f"Results transform not found: {results_path}")
        if not baseline_path.exists():
            if not _create_baseline_if_missing:
                self.log_error(
                    "Baseline transform missing: %s (run pytest with --create-baselines to create from current output)",
                    baseline_path,
                )
                return False
            shutil.copy(str(results_path), str(baseline_path))
            self.log_warning(
                "Baseline transform did not exist; copied results transform: %s",
                results_path,
            )
        transform = itk.transformread(str(results_path))
        transform_params = np.array(transform[0].GetParameters())

        baseline_transform = itk.transformread(str(baseline_path))
        baseline_transform_params = np.array(baseline_transform[0].GetParameters())

        diff = transform_params - baseline_transform_params
        absolute_err = np.abs(diff)
        number_of_values_above_tol = int(
            np.sum(absolute_err > per_value_absolute_error_tol)
        )
        total_absolute_error = float(np.sum(absolute_err))

        difference_transform = baseline_transform[0]
        difference_transform_params = baseline_transform[0].GetParameters()
        for i in range(len(difference_transform_params)):
            difference_transform_params[i] = diff[i]
        difference_transform.SetParameters(difference_transform_params)

        self._last_transform_difference_transform = difference_transform
        self._last_transform_per_value_absolute_error_tol = per_value_absolute_error_tol
        self._last_transform_number_of_values_above_tol = number_of_values_above_tol
        self._last_transform_max_number_of_values_above_tol = (
            max_number_of_values_above_tol
        )
        self._last_transform_total_absolute_error = total_absolute_error
        self._last_transform_total_absolute_error_tol = total_absolute_error_tol

        passed = (
            self._last_transform_number_of_values_above_tol
            <= self._last_transform_max_number_of_values_above_tol
            and self._last_transform_total_absolute_error
            <= self._last_transform_total_absolute_error_tol
        )

        if passed:
            self.log_info(
                "PASS: number_of_values_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                self._last_transform_number_of_values_above_tol,
                self._last_transform_max_number_of_values_above_tol,
                self._last_transform_total_absolute_error,
                self._last_transform_total_absolute_error_tol,
            )
        else:
            self.log_error(
                "FAIL: number_of_values_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                self._last_transform_number_of_values_above_tol,
                self._last_transform_max_number_of_values_above_tol,
                self._last_transform_total_absolute_error,
                self._last_transform_total_absolute_error_tol,
            )
        return passed

    def compare_result_to_baseline_image(
        self,
        filename: str,
        slice_index: int | None = None,
        axis: int = 0,
        *,
        per_pixel_absolute_error_tol: float = 0.0,
        max_number_of_pixels_above_tol: int = 0,
        total_absolute_error_tol: float = 0.0,
    ) -> bool:
        """
        Load 3D image from results_filename and 2D baseline from baseline_filename (.mha), compare the given slice to baseline,
        save the difference image with \"_diff\" in the name (.mha), and log pass/fail and values (INFO/ERROR).
        If slice_index is not given, the middle slice is used.
        If the baseline file does not exist, it is created from the corresponding slice of the 3D result and a warning is logged.
        Returns True if comparison passed (pixels and total absolute error within tolerance).
        """
        results_path = Path(self._results_dir / filename)
        baseline_path = Path(self._baselines_dir / filename)
        if not results_path.exists():
            raise FileNotFoundError(f"Results image not found: {results_path}")

        image_3d = itk.imread(str(results_path))
        arr_3d = np.asarray(itk.array_from_image(image_3d), dtype=np.float64)
        if slice_index is None:
            slice_index = arr_3d.shape[axis] // 2

        if not baseline_path.exists():
            if not _create_baseline_if_missing:
                self.log_error(
                    "Baseline image missing: %s (run pytest with --create-baselines to create from current output)",
                    baseline_path,
                )
                return False
            # Create baseline from the corresponding slice of the 3D result
            slice_3d = np.take(arr_3d, slice_index, axis=axis)
            slice_2d = np.squeeze(slice_3d) if slice_3d.ndim == 3 else slice_3d
            slice_itk = itk.image_from_array(slice_2d.astype(np.float64))
            baseline_path = (
                (self._baselines_dir / filename).with_suffix(".mha")
                if baseline_path.suffix.lower() != ".mha"
                else baseline_path
            )
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            itk.imwrite(slice_itk, str(baseline_path), compression=True)
            self.log_warning(
                "Baseline file did not exist; created from 3D result slice: %s",
                baseline_path,
            )
            image_2d = slice_itk
        else:
            image_2d = itk.imread(str(baseline_path))

        passed = self.compare_2d_to_3d_slice(
            image_2d,
            image_3d,
            slice_index,
            axis=axis,
            per_pixel_absolute_error_tol=per_pixel_absolute_error_tol,
            max_number_of_pixels_above_tol=max_number_of_pixels_above_tol,
            total_absolute_error_tol=total_absolute_error_tol,
        )

        # Save difference image to results dir
        stem = Path(filename).stem
        if "." in stem:
            stem = stem.split(".")[0]
        diff_filename = stem + "_diff.mha"
        diff_path = self._results_dir / diff_filename
        itk.imwrite(self._last_image_difference_image, str(diff_path), compression=True)

        # Log pass/fail and values
        _, number_of_pixels_above_tol = (
            self.image_pass_fail_and_pixels_above_tolerance()
        )
        _, total_absolute_error = self.image_pass_fail_and_total_absolute_error()
        if passed:
            self.log_info(
                "PASS: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )
        else:
            self.log_error(
                "FAIL: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )

        return passed
