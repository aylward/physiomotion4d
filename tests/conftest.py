#!/usr/bin/env python
"""
Shared pytest fixtures for PhysioMotion4D tests.

This file defines fixtures that are available to all test modules
in the tests directory via pytest's automatic fixture discovery.
"""

import shutil
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import itk
import pytest
from itk import TubeTK as ttk

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_greedy import RegisterImagesGreedy
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D
from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware
from physiomotion4d.transform_tools import TransformTools

# ============================================================================
# Pytest Configuration - Command Line Options
# ============================================================================

# Module-level variable to store config for access in hooks
_pytest_config = None


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-experiments",
        action="store_true",
        default=False,
        help="Run experiment tests (extremely long-running notebook tests)",
    )
    parser.addoption(
        "--create-baselines",
        action="store_true",
        default=False,
        help="Create baseline files from current test outputs when missing (otherwise missing baseline fails)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    global _pytest_config
    _pytest_config = config

    from physiomotion4d import test_tools as _test_tools

    _test_tools.set_create_baseline_if_missing(
        config.getoption("--create-baselines", default=False)
    )

    config.addinivalue_line(
        "markers",
        "experiment: marks tests that run experiment notebooks (extremely slow, manual only)",
    )
    # Initialize test timing storage
    config._test_timings = {
        "tests": [],
        "total_time": 0.0,
        "start_time": datetime.now(),
    }


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip experiment tests unless --run-experiments is passed.

    This ensures that experiment tests are opt-in only and won't run
    accidentally when running the normal test suite.
    """
    if config.getoption("--run-experiments"):
        # User explicitly requested experiment tests, let them run
        return

    # Skip all tests marked with @pytest.mark.experiment
    skip_experiments = pytest.mark.skip(
        reason="Experiment tests require --run-experiments flag to run"
    )
    for item in items:
        if "experiment" in item.keywords:
            item.add_marker(skip_experiments)


def pytest_runtest_logreport(report):
    """
    Collect test timing information after each test completes.

    This hook is called for each phase of test execution (setup, call, teardown).
    We only collect timing from the 'call' phase which is the actual test execution.
    """
    if report.when == "call":
        # Use the module-level config reference
        if _pytest_config is None:
            return

        # Store test timing information
        test_info = {
            "nodeid": report.nodeid,
            "duration": report.duration,
            "outcome": report.outcome,
            "is_experiment": "experiment" in report.keywords,
        }

        _pytest_config._test_timings["tests"].append(test_info)
        _pytest_config._test_timings["total_time"] += report.duration


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Print comprehensive test timing report after all tests complete.

    This hook is called at the end of the test session to display
    timing statistics for all tests, including experiment tests.
    """
    timings = config._test_timings
    tests = timings["tests"]

    if not tests:
        return

    # Calculate session duration
    session_duration = datetime.now() - timings["start_time"]

    # Separate regular and experiment tests
    regular_tests = [t for t in tests if not t["is_experiment"]]
    experiment_tests = [t for t in tests if t["is_experiment"]]

    # Write the timing report
    terminalreporter.write_sep("=", "TEST TIMING REPORT", bold=True)
    terminalreporter.write_line("")

    # Session summary
    terminalreporter.write_line(f"Session Duration: {session_duration}")
    terminalreporter.write_line(
        f"Total Test Time: {timedelta(seconds=int(timings['total_time']))}"
    )
    terminalreporter.write_line(f"Total Tests: {len(tests)}")
    terminalreporter.write_line("")

    # Regular tests section
    if regular_tests:
        terminalreporter.write_sep("-", "Regular Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(regular_tests)}")

        # Sort by duration (longest first)
        sorted_regular = sorted(
            regular_tests, key=lambda x: x["duration"], reverse=True
        )

        # Calculate total time
        regular_total = sum(t["duration"] for t in regular_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(regular_total))}"
        )
        terminalreporter.write_line("")

        # Show all regular tests with timing
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_regular:
            outcome_symbol = "✓" if test["outcome"] == "passed" else "✗"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Experiment tests section
    if experiment_tests:
        terminalreporter.write_sep("-", "Experiment Tests", bold=True)
        terminalreporter.write_line(f"Count: {len(experiment_tests)}")

        # Sort by duration (longest first)
        sorted_experiments = sorted(
            experiment_tests, key=lambda x: x["duration"], reverse=True
        )

        # Calculate total time
        experiment_total = sum(t["duration"] for t in experiment_tests)
        terminalreporter.write_line(
            f"Total Time: {timedelta(seconds=int(experiment_total))}"
        )
        terminalreporter.write_line("")

        # Show all experiment tests with timing
        terminalreporter.write_line("Individual Test Times:")
        for test in sorted_experiments:
            outcome_symbol = "✓" if test["outcome"] == "passed" else "✗"
            duration_str = _format_duration(test["duration"])
            terminalreporter.write_line(
                f"  {outcome_symbol} {duration_str:>10s}  {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Top 10 slowest tests overall
    if len(tests) > 10:
        terminalreporter.write_sep("-", "Top 10 Slowest Tests", bold=True)
        sorted_all = sorted(tests, key=lambda x: x["duration"], reverse=True)[:10]

        for i, test in enumerate(sorted_all, 1):
            outcome_symbol = "✓" if test["outcome"] == "passed" else "✗"
            duration_str = _format_duration(test["duration"])
            test_type = "[EXP]" if test["is_experiment"] else "[REG]"
            terminalreporter.write_line(
                f"  {i:2d}. {outcome_symbol} {duration_str:>10s} {test_type} {test['nodeid']}"
            )
        terminalreporter.write_line("")

    # Statistics by outcome
    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    skipped = sum(1 for t in tests if t["outcome"] == "skipped")

    terminalreporter.write_sep("-", "Test Outcomes", bold=True)
    terminalreporter.write_line(f"Passed:  {passed}")
    terminalreporter.write_line(f"Failed:  {failed}")
    terminalreporter.write_line(f"Skipped: {skipped}")
    terminalreporter.write_line("")


def _format_duration(seconds):
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# Directory and Data Download Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_directories():
    """Set up test directories for data and results."""
    data_dir = Path("tests/data/Slicer-Heart-CT")
    output_dir = Path("tests/results")
    baselines_dir = Path("tests/baselines")

    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    return {"data": data_dir, "output": output_dir, "baselines": baselines_dir}


@pytest.fixture(scope="session")
def download_truncal_valve_data(test_directories):
    """Download TruncalValve 4D CT data."""
    data_dir = test_directories["data"]
    input_image_filename = data_dir / "TruncalValve_4DCT.seq.nrrd"

    # Check if file already exists in test data directory
    if input_image_filename.exists():
        print(f"\nData file already exists: {input_image_filename}")
        return input_image_filename

    # Check if file exists in main data directory (one level up from project root)
    main_data_file = Path("data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd")
    if main_data_file.exists():
        print(f"\nCopying data from main data directory: {main_data_file}")
        import shutil

        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(main_data_file), str(input_image_filename))
        print(f"Copied to {input_image_filename}")
        return input_image_filename

    # Try to download if not found locally
    input_image_url = "https://github.com/Slicer-Heart-CT/Slicer-Heart-CT/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd"
    print(f"\nDownloading TruncalValve 4D CT data from {input_image_url}...")

    try:
        urllib.request.urlretrieve(input_image_url, str(input_image_filename))
        print(f"Downloaded to {input_image_filename}")
    except urllib.error.HTTPError as e:
        pytest.skip(
            f"Could not download test data: {e}. Please manually place TruncalValve_4DCT.seq.nrrd in {data_dir}"
        )

    return input_image_filename


# ============================================================================
# Image Conversion Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def converted_3d_images(download_truncal_valve_data, test_directories):
    """Convert 4D NRRD to 3D time series and return slice files."""
    data_dir = test_directories["data"]
    output_dir = test_directories["output"]
    input_4d_file = download_truncal_valve_data

    # Check if conversion already done
    slice_000 = data_dir / "slice_000.mha"
    slice_007 = data_dir / "slice_007.mha"

    if not slice_000.exists() or not slice_007.exists():
        # Convert 4D to 3D time series
        print("\nConverting 4D NRRD to 3D time series...")
        conv = ConvertNRRD4DTo3D()
        conv.load_nrrd_4d(str(input_4d_file))
        conv.save_3d_images(str(data_dir / "slice"))

        # Copy mid-stroke slice as fixed/reference image
        fixed_image_output = output_dir / "slice_fixed.mha"
        shutil.copyfile(str(slice_007), str(fixed_image_output))
        print(f"Conversion complete, saved fixed image to: {fixed_image_output}")
    else:
        print("\n3D slice files already exist")

    return data_dir


@pytest.fixture(scope="session")
def test_images(converted_3d_images):
    """Load time points from the converted 3D data for testing."""
    data_dir = converted_3d_images

    # Load time points
    slice_000 = data_dir / "slice_000.mha"
    slice_001 = data_dir / "slice_001.mha"
    slice_002 = data_dir / "slice_002.mha"
    slice_003 = data_dir / "slice_003.mha"
    slice_004 = data_dir / "slice_004.mha"
    slice_005 = data_dir / "slice_005.mha"

    # Ensure the files exist
    if not slice_000.exists() or not slice_001.exists() or not slice_002.exists():
        pytest.skip("Converted 3D slice files not found. Run conversion test first.")

    images = [
        itk.imread(str(slice_000)),
        itk.imread(str(slice_001)),
        itk.imread(str(slice_002)),
        itk.imread(str(slice_003)),
        itk.imread(str(slice_004)),
        itk.imread(str(slice_005)),
    ]

    for i, img in enumerate(images):
        resampler = ttk.ResampleImage.New(Input=img)
        resampler.SetResampleFactor([0.5, 0.5, 0.5])
        resampler.Update()
        images[i] = resampler.GetOutput()

    print(f"\nLoaded {len(images)} time points for testing")
    return images


# ============================================================================
# Segmentation Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def segmenter_total_segmentator():
    """Create a SegmentChestTotalSegmentator instance."""
    return SegmentChestTotalSegmentator()


@pytest.fixture(scope="session")
def segmenter_vista_3d():
    """Create a SegmentChestVista3D instance."""
    return SegmentChestVista3D()


@pytest.fixture(scope="session")
def segmenter_simpleware():
    """Create a SegmentHeartSimpleware instance."""
    return SegmentHeartSimpleware()


@pytest.fixture(scope="session")
def heart_simpleware_image_path():
    """Path to cardiac CT image used by experiments/Heart-Simpleware_Segmentation notebook."""
    # Heart-Simpleware_Segmentation uses same data as the notebook: data/CHOP-Valve4D/CT/RVOT28-Dias.nii.gz
    image_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "CHOP-Valve4D"
        / "CT"
        / "RVOT28-Dias.nii.gz"
    )
    if not image_path.exists():
        pytest.skip(
            f"Heart Simpleware test data not found: {image_path}. "
            "Place RVOT28-Dias.nii.gz there or run from repo with data/CHOP-Valve4D/CT/ populated."
        )
    return image_path


@pytest.fixture(scope="session")
def heart_simpleware_image(heart_simpleware_image_path):
    """Load cardiac CT image for SegmentHeartSimpleware tests (same as notebook)."""
    return itk.imread(str(heart_simpleware_image_path))


@pytest.fixture(scope="session")
def segmentation_results(segmenter_total_segmentator, test_images, test_directories):
    """
    Get or create segmentation results using TotalSegmentator.
    Used by multiple tests (contour, USD conversion, etc.)
    """
    output_dir = test_directories["output"]
    seg_output_dir = output_dir / "segmentation_total_segmentator"

    # Check if segmentation files exist
    labelmap_000 = seg_output_dir / "slice_000_labelmap.mha"
    labelmap_001 = seg_output_dir / "slice_001_labelmap.mha"

    if not labelmap_000.exists() or not labelmap_001.exists():
        # Run segmentation if results don't exist
        print("\nSegmentation results not found, generating them...")
        seg_output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, input_image in enumerate(test_images):
            result = segmenter_total_segmentator.segment(
                input_image, contrast_enhanced_study=False
            )
            results.append(result)

            # Save labelmap
            labelmap = result["labelmap"]
            output_file = seg_output_dir / f"slice_{i:03d}_labelmap.mha"
            itk.imwrite(labelmap, str(output_file), compression=True)

        return results
    # Load existing segmentation results
    print("\nLoading existing segmentation results...")
    results = []
    for i in range(2):
        labelmap_file = seg_output_dir / f"slice_{i:03d}_labelmap.mha"
        labelmap = itk.imread(str(labelmap_file))

        # Create anatomy group masks from labelmap
        masks = segmenter_total_segmentator.create_anatomy_group_masks(labelmap)

        result = {
            "labelmap": labelmap,
            "lung": masks["lung"],
            "heart": masks["heart"],
            "major_vessels": masks["major_vessels"],
            "bone": masks["bone"],
            "soft_tissue": masks["soft_tissue"],
            "other": masks["other"],
            "contrast": masks["contrast"],
        }
        results.append(result)

    return results


# ============================================================================
# Contour Tool Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def contour_tools():
    """Create a ContourTools instance."""
    return ContourTools()


# ============================================================================
# Registration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def registrar_ants():
    """Create a RegisterImagesANTs instance."""
    return RegisterImagesANTs()


@pytest.fixture(scope="session")
def registrar_greedy():
    """Create a RegisterImagesGreedy instance."""
    return RegisterImagesGreedy()


@pytest.fixture(scope="session")
def registrar_icon():
    """Create a RegisterImagesICON instance."""
    return RegisterImagesICON()


@pytest.fixture(scope="session")
def ants_registration_results(registrar_ants, test_images, test_directories):
    """
    Perform ANTs registration and return results.
    Generates them if not already present, otherwise loads from disk.
    """
    output_dir = test_directories["output"]
    reg_output_dir = output_dir / "registration_ants"
    reg_output_dir.mkdir(exist_ok=True)

    inverse_transform_path = reg_output_dir / "ants_inverse_transform_no_mask.hdf"
    forward_transform_path = reg_output_dir / "ants_forward_transform_no_mask.hdf"

    if inverse_transform_path.exists() and forward_transform_path.exists():
        print("\nLoading existing ANTs registration results...")
        try:
            inverse_transform = itk.transformread(str(inverse_transform_path))
            forward_transform = itk.transformread(str(forward_transform_path))
            return {
                "inverse_transform": inverse_transform,
                "forward_transform": forward_transform,
            }
        except (RuntimeError, Exception) as e:
            print(f"Error loading transforms: {e}")
            print("Regenerating registration results...")
            # Delete corrupt files
            inverse_transform_path.unlink(missing_ok=True)
            forward_transform_path.unlink(missing_ok=True)

    # Perform registration if files don't exist or loading failed
    print("\nPerforming ANTs registration...")
    fixed_image = test_images[0]
    moving_image = test_images[1]

    registrar_ants.set_fixed_image(fixed_image)
    result = registrar_ants.register(moving_image=moving_image)

    inverse_transform = result["inverse_transform"]
    forward_transform = result["forward_transform"]

    itk.transformwrite(inverse_transform, str(inverse_transform_path), compression=True)
    itk.transformwrite(forward_transform, str(forward_transform_path), compression=True)
    return {
        "inverse_transform": inverse_transform,
        "forward_transform": forward_transform,
    }


# ============================================================================
# Transform Tool Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def transform_tools():
    """Create a TransformTools instance."""
    return TransformTools()
