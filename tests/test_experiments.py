#!/usr/bin/env python
"""
Test suite for running experiment notebooks.

These tests execute Jupyter notebooks in the experiments/ directory. Each subdirectory
in experiments/ gets its own test that runs all notebooks in that subdirectory in
alphanumeric order.

WARNING: These are EXTREMELY long-running tests that may take hours to complete.
They are NOT part of CI/CD and should only be run manually.

Usage:
    # Run all experiment tests
    pytest tests/test_experiments.py -v -m experiment

    # Run a specific experiment subdirectory
    pytest tests/test_experiments.py::test_experiment_colormap_vtk_to_usd -v

    # Run with detailed output
    pytest tests/test_experiments.py -v -s -m experiment

Note: These tests require all dependencies installed and GPU/CUDA support for
many of the experiments.
"""

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

# Base directories
REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Experiment subdirectories to test (in order of complexity/dependencies)
EXPERIMENT_SUBDIRS = [
    "Colormap-VTK_To_USD",
    "Convert_VTK_To_USD",
    # 'DisplacementField_To_USD',  # Disabled - notebooks not ready
    "Reconstruct4DCT",
    "Heart-VTKSeries_To_USD",
    "Heart-GatedCT_To_USD",
    "Heart-Create_Statistical_Model",
    "Heart-Statistical_Model_To_Patient",
    "Lung-GatedCT_To_USD",
    # 'Lung-VesselsAirways',  # Disabled - notebooks not ready
]


def get_notebooks_in_subdir(subdir_name: str) -> list[Path]:
    """
    Get all Jupyter notebooks in a subdirectory, sorted alphanumerically.

    Args:
        subdir_name: Name of the subdirectory in experiments/

    Returns:
        List of Path objects for notebook files, sorted alphanumerically
    """
    subdir = EXPERIMENTS_DIR / subdir_name
    if not subdir.exists():
        return []

    notebooks = sorted(subdir.glob("*.ipynb"))
    return notebooks


def clear_notebook_outputs(notebook_path: Path) -> bool:
    """
    Clear all cell outputs from a Jupyter notebook.

    This removes execution outputs, execution counts, and metadata from all cells
    to keep the repository clean and avoid committing large output data.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if outputs were successfully cleared, False otherwise
    """
    try:
        print(f"Clearing outputs from: {notebook_path.name}")

        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Clear outputs and execution counts from all cells
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        # Clear notebook-level metadata
        if "execution" in notebook.metadata:
            del notebook.metadata["execution"]

        # Write the cleaned notebook back
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        print(f"✅ Cleared outputs from: {notebook_path.name}")
        return True

    except Exception as e:
        print(f"⚠️ Failed to clear outputs from {notebook_path.name}: {e}")
        return False


def execute_notebook(notebook_path: Path, timeout: int = 3600) -> dict:
    """
    Execute a Jupyter notebook using nbconvert.

    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum execution time in seconds (default: 1 hour)

    Returns:
        Dictionary with execution results:
            - success: bool
            - stdout: str
            - stderr: str
            - returncode: int

    Raises:
        subprocess.TimeoutExpired: If notebook execution exceeds timeout
    """
    print(f"\n{'=' * 80}")
    print(f"Executing notebook: {notebook_path.name}")
    print(f"Path: {notebook_path}")
    print(f"Timeout: {timeout} seconds ({timeout // 60} minutes)")
    print(f"{'=' * 80}\n")

    # Use nbconvert to execute the notebook in place
    # --execute: Execute the notebook
    # --to notebook: Output as notebook (not HTML/PDF)
    # --inplace: Overwrite the original notebook with execution results
    # --ExecutePreprocessor.timeout: Set timeout per cell
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(notebook_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * 1.5,  # Give extra time for nbconvert overhead
            cwd=notebook_path.parent,  # Run in notebook's directory
            check=False,
        )

        success = result.returncode == 0

        if success:
            print(f"✅ Successfully executed: {notebook_path.name}")

            # Clear outputs from the notebook after successful execution
            print("Clearing cell outputs to keep repository clean...")
            clear_notebook_outputs(notebook_path)
        else:
            print(f"❌ Failed to execute: {notebook_path.name}")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout executing: {notebook_path.name}")
        print(f"Exceeded: {timeout} seconds")
        raise


def run_experiment_notebooks(subdir_name: str, timeout_per_notebook: int = 3600):
    """
    Run all notebooks in an experiment subdirectory in alphanumeric order.

    IMPORTANT: Notebooks are executed SEQUENTIALLY in alphanumeric order within
    this function. This ensures proper dependency handling even when running
    tests with multiple pytest workers (e.g., pytest -n 2).

    The sequential execution is enforced by:
    1. Using a standard Python for loop (not parallelized)
    2. Each notebook must complete before the next begins
    3. Failures in earlier notebooks prevent later ones from running

    Args:
        subdir_name: Name of the subdirectory in experiments/
        timeout_per_notebook: Timeout in seconds for each notebook (default: 1 hour)

    Raises:
        AssertionError: If any notebook fails to execute successfully
    """
    notebooks = get_notebooks_in_subdir(subdir_name)

    if not notebooks:
        pytest.skip(f"No notebooks found in experiments/{subdir_name}")

    print(f"\n{'#' * 80}")
    print(f"# Experiment: {subdir_name}")
    print(f"# Found {len(notebooks)} notebook(s)")
    print("# Sequential execution enforced (notebooks run in order)")
    print(f"{'#' * 80}\n")

    failed_notebooks = []
    successful_notebooks = []

    for i, notebook in enumerate(notebooks, 1):
        print(f"\n--- Notebook {i}/{len(notebooks)} ---")
        print(f"Sequential execution: notebook {i} must complete before {i + 1} starts")

        try:
            result = execute_notebook(notebook, timeout=timeout_per_notebook)

            if result["success"]:
                successful_notebooks.append(notebook.name)
            else:
                failed_notebooks.append(
                    {
                        "name": notebook.name,
                        "returncode": result["returncode"],
                        "stderr": result["stderr"],
                    }
                )
                # Stop execution on first failure to maintain dependencies
                print(f"\n⚠️ Stopping execution: {notebook.name} failed")
                print("Remaining notebooks in this experiment will not run.")
                break

        except subprocess.TimeoutExpired:
            failed_notebooks.append(
                {
                    "name": notebook.name,
                    "returncode": -1,
                    "stderr": f"Timeout after {timeout_per_notebook} seconds",
                }
            )
            # Stop execution on timeout
            print(f"\n⚠️ Stopping execution: {notebook.name} timed out")
            print("Remaining notebooks in this experiment will not run.")
            break

        except Exception as e:
            failed_notebooks.append(
                {"name": notebook.name, "returncode": -2, "stderr": str(e)}
            )
            # Stop execution on exception
            print(f"\n⚠️ Stopping execution: {notebook.name} raised exception")
            print("Remaining notebooks in this experiment will not run.")
            break

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Experiment Summary: {subdir_name}")
    print(f"{'=' * 80}")
    print(f"Total notebooks: {len(notebooks)}")
    print(f"Successful: {len(successful_notebooks)}")
    print(f"Failed: {len(failed_notebooks)}")

    if successful_notebooks:
        print("\n✅ Successful notebooks:")
        for name in successful_notebooks:
            print(f"  - {name}")

    if failed_notebooks:
        print("\n❌ Failed notebooks:")
        for failure in failed_notebooks:
            print(f"  - {failure['name']}")
            print(f"    Return code: {failure['returncode']}")
            if failure["stderr"]:
                # Print first few lines of error
                error_lines = failure["stderr"].split("\n")[:10]
                for line in error_lines:
                    print(f"    {line}")

    print(f"{'=' * 80}\n")

    # Assert all notebooks succeeded
    assert not failed_notebooks, (
        f"{len(failed_notebooks)} notebook(s) failed in {subdir_name}: "
        f"{[f['name'] for f in failed_notebooks]}"
    )


# ============================================================================
# Test Functions - One per Experiment Subdirectory
# ============================================================================


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.timeout(7200)  # 2 hours total timeout
@pytest.mark.xdist_group(
    name="experiment_colormap"
)  # Prevent parallel execution within group
def test_experiment_colormap_vtk_to_usd():
    """
    Test Colormap-VTK_To_USD experiment notebooks.

    This experiment demonstrates converting VTK files with colormaps to USD format.

    EXECUTION MODEL:
    - Notebooks run SEQUENTIALLY in alphanumeric order within this test
    - This test function is atomic - pytest-xdist treats it as a single unit
    - Multiple experiment tests CAN run in parallel (different subdirectories)
    - Notebooks within THIS experiment CANNOT run in parallel or out of order
    """
    run_experiment_notebooks("Colormap-VTK_To_USD", timeout_per_notebook=3600)


# DISABLED - Notebooks not ready
# @pytest.mark.experiment
# @pytest.mark.slow
# @pytest.mark.timeout(7200)  # 2 hours total timeout
# def test_experiment_displacement_field_to_usd():
#     """
#     Test DisplacementField_To_USD experiment notebooks.
#
#     This experiment demonstrates converting registration displacement fields to USD
#     format for visualization in PhysicsNeMo and Omniverse.
#     """
#     run_experiment_notebooks('DisplacementField_To_USD', timeout_per_notebook=3600)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.timeout(14400)  # 4 hours total timeout
@pytest.mark.xdist_group(name="experiment_reconstruct4dct")
def test_experiment_reconstruct_4dct():
    """
    Test Reconstruct4DCT experiment notebooks.

    This experiment demonstrates 4D CT reconstruction techniques.

    EXECUTION MODEL:
    - Notebooks run SEQUENTIALLY in alphanumeric order within this test
    - Each notebook must complete before the next begins
    - Failure in one notebook stops execution of remaining notebooks
    """
    run_experiment_notebooks("Reconstruct4DCT", timeout_per_notebook=7200)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.timeout(10800)  # 3 hours total timeout
@pytest.mark.xdist_group(name="experiment_heart_vtk")
def test_experiment_heart_vtk_series_to_usd():
    """
    Test Heart-VTKSeries_To_USD experiment notebooks.

    This experiment converts heart VTK time series data to USD format.

    EXECUTION ORDER (ENFORCED):
    1. 0-download_and_convert_4d_to_3d.ipynb (downloads data)
    2. 1-heart_vtkseries_to_usd.ipynb (uses downloaded data)

    Each notebook must complete successfully before the next begins.
    """
    run_experiment_notebooks("Heart-VTKSeries_To_USD", timeout_per_notebook=5400)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.requires_data
@pytest.mark.timeout(21600)  # 6 hours total timeout
@pytest.mark.xdist_group(name="experiment_heart_gated_ct")
def test_experiment_heart_gated_ct_to_usd():
    """
    Test Heart-GatedCT_To_USD experiment notebooks.

    This is the main cardiac imaging pipeline experiment with strict dependencies.

    EXECUTION ORDER (STRICTLY ENFORCED):
    1. 0-download_and_convert_4d_to_3d.ipynb (downloads and converts data)
    2. 1-register_images.ipynb (registers converted images)
    3. 2-generate_segmentation.ipynb (segments registered images)
    4. 3-transform_dynamic_and_static_contours.ipynb (transforms segmentations)
    5. 4-merge_dynamic_and_static_usd.ipynb (merges into final USD)
    6. test_vista3d_class.ipynb (tests segmentation class)
    7. test_vista3d_inMem.ipynb (tests in-memory segmentation)

    Each notebook depends on outputs from previous notebooks.
    Execution stops on first failure to prevent cascading errors.
    """
    run_experiment_notebooks("Heart-GatedCT_To_USD", timeout_per_notebook=5400)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.timeout(7200)  # 2 hours total timeout
@pytest.mark.xdist_group(name="experiment_convert_vtk_to_usd")
def test_experiment_convert_vtk_to_usd():
    """
    Test Convert_VTK_To_USD experiment notebooks.

    This experiment demonstrates VTK to USD conversion using the library classes.

    EXECUTION ORDER (ENFORCED):
    1. convert_chop_valve_to_usd.ipynb (converts CHOP valve data)
    2. convert_vtk_to_usd_using_class.ipynb (demonstrates library usage)

    Sequential execution ensures examples build on each other.
    """
    run_experiment_notebooks("Convert_VTK_To_USD", timeout_per_notebook=3600)


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.timeout(10800)  # 3 hours total timeout
@pytest.mark.xdist_group(name="experiment_create_statistical_model")
def test_experiment_create_statistical_model():
    """
    Test Heart-Create_Statistical_Model experiment notebooks.

    This experiment demonstrates creating a PCA statistical shape model from the
    KCL Heart Model dataset.

    EXECUTION ORDER (ENFORCED):
    1. 1-input_meshes_to_input_surfaces.ipynb (convert meshes to surfaces)
    2. 2-input_surfaces_to_surfaces_aligned.ipynb (align surfaces)
    3. 3-registration_based_correspondence.ipynb (establish point correspondence)
    4. 4-surfaces_aligned_correspond_to_pca_inputs.ipynb (prepare PCA inputs)
    5. 5-compute_pca_model.ipynb (compute PCA model using sklearn)

    Sequential execution ensures data dependencies are met.
    """
    run_experiment_notebooks(
        "Heart-Create_Statistical_Model", timeout_per_notebook=5400
    )


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.requires_data
@pytest.mark.timeout(14400)  # 4 hours total timeout
@pytest.mark.xdist_group(name="experiment_heart_statistical_model")
def test_experiment_heart_statistical_model_to_patient():
    """
    Test Heart-Statistical_Model_To_Patient experiment notebooks.

    This experiment demonstrates heart model to patient registration using
    statistical shape models (PCA).

    ⚠️ PREREQUISITE: Complete Heart-Create_Statistical_Model experiment first to generate
    the PCA model data required for this experiment.

    EXECUTION ORDER (ENFORCED):
    1. heart_model_to_model_icp_itk.ipynb (ICP registration)
    2. heart_model_to_model_registration_pca.ipynb (PCA-based registration)
    3. heart_model_to_patient.ipynb (applies registration to patient)

    Sequential execution ensures registration results are available for subsequent steps.
    """
    run_experiment_notebooks(
        "Heart-Statistical_Model_To_Patient", timeout_per_notebook=7200
    )


@pytest.mark.experiment
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.requires_data
@pytest.mark.timeout(21600)  # 6 hours total timeout
@pytest.mark.xdist_group(name="experiment_lung_gated_ct")
def test_experiment_lung_gated_ct_to_usd():
    """
    Test Lung-GatedCT_To_USD experiment notebooks.

    This is the lung imaging pipeline experiment using DirLab 4DCT data.

    EXECUTION ORDER (STRICTLY ENFORCED):
    1. 0-register_dirlab_4dct.ipynb (registers lung 4DCT data)
    2. 1-make_dirlab_models.ipynb (creates 3D models from registered data)
    3. 2-paint_dirlab_models.ipynb (applies textures/materials to models)
    4. Experiment_ArrangeOnStage.ipynb (arranges models in USD scene)
    5. Experiment_CombineModels.ipynb (combines models into single USD)
    6. Experiment_SegReg.ipynb (segmentation and registration experiments)
    7. Experiment_SubSurfaceScatter.ipynb (applies advanced materials)

    Each notebook depends on outputs from previous notebooks.
    Execution is sequential and stops on first failure.
    """
    run_experiment_notebooks("Lung-GatedCT_To_USD", timeout_per_notebook=5400)


# DISABLED - Notebooks not ready
# @pytest.mark.experiment
# @pytest.mark.slow
# @pytest.mark.requires_gpu
# @pytest.mark.requires_data
# @pytest.mark.timeout(7200)  # 2 hours total timeout
# def test_experiment_lung_vessels_airways():
#     """
#     Test Lung-VesselsAirways experiment notebooks.
#
#     This experiment demonstrates specialized vessel and airway segmentation
#     using deep learning models.
#     Expected notebooks (in order):
#     - 0-GenData.ipynb
#     """
#     run_experiment_notebooks('Lung-VesselsAirways', timeout_per_notebook=3600)


# ============================================================================
# Discovery Test - Validate Experiment Structure
# ============================================================================


@pytest.mark.experiment
def test_experiment_structure():
    """
    Validate the structure of the experiments directory.

    This test checks that:
    1. The experiments directory exists
    2. Each expected subdirectory exists
    3. Each subdirectory contains at least one notebook
    """
    assert EXPERIMENTS_DIR.exists(), (
        f"Experiments directory not found: {EXPERIMENTS_DIR}"
    )

    missing_subdirs = []
    empty_subdirs = []

    for subdir_name in EXPERIMENT_SUBDIRS:
        subdir = EXPERIMENTS_DIR / subdir_name

        if not subdir.exists():
            missing_subdirs.append(subdir_name)
            continue

        notebooks = list(subdir.glob("*.ipynb"))
        if not notebooks:
            empty_subdirs.append(subdir_name)

    # Report findings
    if missing_subdirs:
        print(f"\n⚠️ Missing subdirectories: {missing_subdirs}")

    if empty_subdirs:
        print(f"\n⚠️ Empty subdirectories (no notebooks): {empty_subdirs}")

    # Print discovered notebooks
    print("\n📓 Discovered Notebooks:")
    for subdir_name in EXPERIMENT_SUBDIRS:
        notebooks = get_notebooks_in_subdir(subdir_name)
        if notebooks:
            print(f"\n{subdir_name}/ ({len(notebooks)} notebook(s)):")
            for nb in notebooks:
                print(f"  - {nb.name}")

    assert not missing_subdirs, f"Missing subdirectories: {missing_subdirs}"
    assert not empty_subdirs, f"Empty subdirectories: {empty_subdirs}"


# ============================================================================
# Helper Test - Notebook Discovery
# ============================================================================


@pytest.mark.experiment
@pytest.mark.parametrize("subdir_name", EXPERIMENT_SUBDIRS)
def test_list_notebooks_in_subdir(subdir_name):
    """
    List all notebooks in each experiment subdirectory.

    This helper test can be used to preview what notebooks will be run
    without actually executing them.

    Usage:
        pytest tests/test_experiments.py::test_list_notebooks_in_subdir -v -s
    """
    notebooks = get_notebooks_in_subdir(subdir_name)

    print(f"\n{subdir_name}/ - {len(notebooks)} notebook(s):")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {nb.name}")

    assert notebooks, f"No notebooks found in {subdir_name}"


# ============================================================================
# Notes for Future Enhancements
# ============================================================================

# TODO: Consider adding these features:
# 1. Parallel execution of independent experiments
# 2. HTML report generation from executed notebooks
# 3. Automatic artifact collection (generated USD files, images, etc.)
# 4. Smoke tests that run only first cell of each notebook
# 5. Integration with papermill for parameterized notebook execution
# 6. Checkpointing to resume failed experiment runs
# 7. Resource usage monitoring (memory, GPU, disk)
# 8. Comparison of outputs with baseline results
