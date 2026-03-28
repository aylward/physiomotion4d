# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Python launcher:** Use `py` on this Windows system (not `python`).

```bash
# Install in editable mode (preferred)
uv pip install -e .

# Lint and format
ruff check . --fix
ruff format .

# Type checking
mypy src/

# All pre-commit hooks
pre-commit run --all-files

# Run fast tests (recommended for development)
py -m pytest tests/ -m "not slow and not requires_data" -v

# Run a single test file
py -m pytest tests/test_contour_tools.py -v

# Run a single test by name
py -m pytest tests/test_contour_tools.py::test_extract_surface -v

# Run tests without GPU-dependent tests
py -m pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
              --ignore=tests/test_segment_chest_vista_3d.py \
              --ignore=tests/test_register_images_icon.py

# Run with coverage
py -m pytest tests/ --cov=src/physiomotion4d --cov-report=html

# Run experiment notebook tests (opt-in, very slow)
py -m pytest tests/ --run-experiments

# Create baseline files when missing
py -m pytest tests/ --create-baselines
```

**Version bumping:** `bumpver update --patch` (or `--minor`, `--major`)

## Architecture

### Pipeline Overview

PhysioMotion4D converts 4D CT scans (cardiac or pulmonary) into animated USD models for NVIDIA Omniverse. The pipeline flows:

```
4D CT → Segmentation → Registration → Contour Extraction → USD Export
```

### Class Hierarchy

All major classes inherit from `PhysioMotion4DBase` (`src/physiomotion4d/physiomotion4d_base.py`), which provides a shared logger named `"PhysioMotion4D"`. Use `self.log_info()`, `self.log_debug()`, etc. — never `print()`. Use `PhysioMotion4DBase.set_log_classes([...])` to filter output to specific classes.

### Workflow Classes (entry points)

- **`WorkflowConvertHeartGatedCTToUSD`**: Full 4D cardiac CT → USD pipeline. Orchestrates: 4D→3D conversion → segmentation (TotalSegmentator) → registration (ICON or ANTs) → contour extraction → USD generation.
- **`WorkflowCreateStatisticalModel`**: Builds a PCA statistical shape model (sklearn) from a population of aligned meshes. Outputs `pca_model.json`, `pca_mean_surface.vtp`.
- **`WorkflowFitStatisticalModelToPatient`**: Multi-stage model-to-patient registration: (1) ICP rough alignment → (2) optional PCA shape fitting → (3) mask-to-mask deformable registration → (4) optional Icon final refinement.
- **`WorkflowReconstructHighres4DCT`**: Reconstructs high-resolution 4D CT from sparse time samples via deformable registration.

### Segmentation Classes

All segment methods return anatomy group masks (heart, lung, major_vessels, bone, soft_tissue, contrast, other, dynamic). The `SegmentAnatomyBase` abstract class defines the interface.

- `SegmentChestTotalSegmentator` — default, CPU-capable
- `SegmentChestVista3D` — GPU-accelerated MONAI VISTA-3D model
- `SegmentChestVista3DNIM` — NIM cloud API version (requires `pip install physiomotion4d[nim]`)
- `SegmentChestEnsemble` — combines multiple methods
- `SegmentHeartSimpleware` — wraps Simpleware ScanIP SDK (requires Simpleware installation)

### Registration Classes

**Image-to-image:**
- `RegisterImagesICON` — deep learning, GPU, preferred for 4D CT
- `RegisterImagesANTs` — classical deformable, CPU-capable
- `RegisterTimeSeriesImages` — wraps ICON or ANTs for 4D time series; handles reference frame selection

All image registerers follow the interface: `set_fixed_image()` → `register(moving_image)` → returns `{"forward_transform": ..., "inverse_transform": ...}` (ITK composite transforms).

**Model-to-model/image:**
- `RegisterModelsICP` — centroid + affine ICP using VTK/PyVista
- `RegisterModelsICPITK` — ICP using ITK
- `RegisterModelsPCA` — PCA shape space fitting; requires `pca_model.json`
- `RegisterModelsDistanceMaps` — deformable registration via distance map matching (uses ANTs or ICON internally)

### USD Pipeline

Two APIs exist for VTK→USD conversion:

1. **`ConvertVTKToUSD`** (`convert_vtk_to_usd.py`) — high-level, operates on PyVista objects in memory. Supports colormap overlays, multi-label anatomy, and animated time series.
2. **`vtk_to_usd/`** subpackage — file-based, modular. Core: `VTKToUSDConverter`, `ConversionSettings`, `MaterialData`. Use `convert_vtk_file()` for simple cases.

`USDTools` and `USDAnatomyTools` handle USD stage merging, time-varying data preservation, and applying surgical materials from a materials library.

### Key Data Conventions

- Medical images use ITK (`itk.Image`); surfaces use PyVista (`pv.PolyData`, `pv.UnstructuredGrid`)
- Coordinate system: RAS (medical) internally; converted to Y-up for USD/Omniverse export
- Masks are ITK images with integer labels; anatomy groups use consistent label IDs across segmenters
- Transforms stored as ITK composite transforms in `.hdf` files

### Testing

- Test baselines are stored in `tests/baselines/` via **Git LFS** — run `git lfs pull` after cloning
- `tests/conftest.py` provides session-scoped fixtures that chain (download → convert → segment → register); most tests depend on upstream fixtures
- Test markers: `slow`, `requires_gpu`, `requires_data`, `experiment` (skipped by default; use `--run-experiments`)
- `test_tools.py` (`src/physiomotion4d/test_tools.py`) provides baseline comparison utilities

### Reference Code

API documentation and examples for advanced third-party libraries (ITK, VTK, PyVista, Omniverse, PhysicsNeMo, Simpleware, MONAI, OpenUSD) are in the `reference_code/` directory.

## File Operations

Use `git mv` / `git rm` for moving or deleting tracked files — not `mv` / `rm` — to preserve git history.

## Documentation Policy

Do **not** create new `.md` files unless explicitly requested. Document via docstrings and inline comments. A `README.md` may be created for new submodules that lack one.

## Code Style

- Single quotes for strings (`'...'`), double quotes for docstrings (`"""..."""`)
- Full type hints required (`mypy` is strict; `disallow_untyped_defs = true`)
- `Optional[X]` not `X | None` for ITK compatibility (ruff `UP007` is suppressed)
- Backward compatibility is **not** a priority — breaking changes are acceptable
