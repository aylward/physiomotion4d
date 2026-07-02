"""Tests for the image-to-USD workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.workflow_convert_image_to_usd import WorkflowConvertImageToUSD
import physiomotion4d.workflow_convert_image_to_usd as workflow_module


def _make_workflow(**overrides: Any) -> WorkflowConvertImageToUSD:
    """Construct a WorkflowConvertImageToUSD with minimal required args,
    overridable via keyword (e.g. segmentation_method=..., output_directory=...)."""
    kwargs: dict[str, Any] = {
        "input_filenames": ["input.nrrd"],
        "contrast_enhanced": False,
        "output_directory": str(overrides.pop("output_directory", "results")),
        "project_name": "patient",
        "log_level": logging.CRITICAL,
    }
    kwargs.update(overrides)
    return WorkflowConvertImageToUSD(**kwargs)


def test_default_segmentation_and_registration_methods(tmp_path: Path) -> None:
    """Omitting segmentation_method/registration_method defaults to
    SegmentChestTotalSegmentator (contrast_threshold=500) and
    RegisterImagesICON, matching this workflow's historical string defaults."""
    workflow = _make_workflow(output_directory=tmp_path)

    assert isinstance(workflow.segmenter, SegmentChestTotalSegmentator)
    assert workflow.segmenter.contrast_threshold == 500
    assert isinstance(workflow.registrar, RegisterImagesICON)


def test_segmentation_method_rejects_wrong_type(tmp_path: Path) -> None:
    """A non-SegmentAnatomyBase segmentation_method raises TypeError."""
    with pytest.raises(TypeError, match="segmentation_method must be"):
        _make_workflow(
            output_directory=tmp_path, segmentation_method="ChestTotalSegmentator"
        )


def test_registration_method_rejects_wrong_type(tmp_path: Path) -> None:
    """A non-RegisterImagesBase registration_method raises TypeError."""
    with pytest.raises(TypeError, match="registration_method must be"):
        _make_workflow(output_directory=tmp_path, registration_method="ICON")


def test_caller_supplied_instances_are_used_as_is(tmp_path: Path) -> None:
    """A caller-supplied segmenter/registrar instance is stored unmodified
    (beyond the documented shared setters): the workflow must not apply its
    default-only contrast_threshold=500 tuning to a caller-supplied segmenter."""
    segmenter = SegmentChestTotalSegmentator()
    original_contrast_threshold = segmenter.contrast_threshold
    registrar: RegisterImagesBase = RegisterImagesICON()

    workflow = _make_workflow(
        output_directory=tmp_path,
        segmentation_method=segmenter,
        registration_method=registrar,
    )

    assert workflow.segmenter is segmenter
    assert workflow.registrar is registrar
    assert workflow.segmenter.contrast_threshold == original_contrast_threshold


def test_create_usd_files_passes_times_per_second(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Workflow forwards FPS to VTK-to-USD for shape (X, Y, Z, T) outputs."""
    times_per_second_values: list[float] = []

    class FakeStage:
        def Export(self, _output_filename: str) -> None:
            return None

    class FakeConvertVTKToUSD:
        def __init__(
            self,
            _project_name: str,
            _input_polydata: list[Any],
            _mask_ids: dict[int, str],
            *,
            times_per_second: float,
            **_kwargs: Any,
        ) -> None:
            times_per_second_values.append(times_per_second)

        def convert(self, _usd_file: str) -> FakeStage:
            return FakeStage()

    class FakeUSDAnatomyTools:
        def __init__(self, _stage: FakeStage) -> None:
            return None

        def enhance_meshes(self, _segmenter: Any) -> None:
            return None

    class FakeTaxonomy:
        def all_labels(self) -> dict[int, str]:
            return {1: "heart"}

    class FakeSegmenter:
        taxonomy = FakeTaxonomy()

    monkeypatch.setattr(workflow_module, "ConvertVTKToUSD", FakeConvertVTKToUSD)
    monkeypatch.setattr(workflow_module, "USDAnatomyTools", FakeUSDAnatomyTools)

    workflow = WorkflowConvertImageToUSD.__new__(WorkflowConvertImageToUSD)
    PhysioMotion4DBase.__init__(
        workflow,
        class_name=WorkflowConvertImageToUSD.__name__,
        log_level=logging.CRITICAL,
    )
    workflow.project_name = "patient"
    workflow.output_directory = str(tmp_path)
    # FakeSegmenter is a minimal test double, not a real SegmentAnatomyBase.
    workflow.segmenter = FakeSegmenter()  # type: ignore[assignment]
    workflow.times_per_second = 12.5
    workflow._transformed_contours = {
        "all": [],
        "dynamic": [],
        "static": [],
    }

    workflow._create_usd_files()

    assert times_per_second_values == [12.5, 12.5, 12.5]
