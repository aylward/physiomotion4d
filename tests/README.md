# PhysioMotion4D Tests

This directory contains comprehensive test suites for the PhysioMotion4D package, validating the complete medical imaging to Omniverse pipeline.

## 📚 Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing guide with setup, troubleshooting, and best practices
- **[GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)** - CI/CD documentation and GitHub Actions workflow details
- **[EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md)** - Guide for running experiment notebook tests
- **[PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md)** - How parallel execution works with experiment tests
- **[EXPERIMENT_FLAG_USAGE.md](EXPERIMENT_FLAG_USAGE.md)** - Details on the --run-experiments flag
- **[TEST_FIXES_SUMMARY.md](TEST_FIXES_SUMMARY.md)** - Recent bug fixes and known issues

## 🧪 Test Categories

### Data Pipeline Tests
- **`test_download_heart_data.py`** - Automatic data download with fallback logic
- **`test_convert_nrrd_4d_to_3d.py`** - 4D NRRD to 3D time series conversion

### Segmentation Tests (GPU Required)
- **`test_segment_chest_total_segmentator.py`** - TotalSegmentator chest CT segmentation
- **`test_segment_chest_vista_3d.py`** - NVIDIA VISTA-3D segmentation (requires 20GB+ RAM)

### Registration Tests (Slow ~5-10 min)
- **`test_register_images_ants.py`** - ANTs deformable registration
- **`test_register_images_icon.py`** - ICON deep learning registration

### Geometry & Visualization Tests
- **`test_contour_tools.py`** - PyVista mesh extraction and manipulation
- **`test_transform_tools.py`** - ITK transform operations and visualization
- **`test_convert_vtk_to_usd_polymesh.py`** - VTK to USD conversion

### USD Utility Tests
- **`test_usd_merge.py`** - USD file merging with material preservation
- **`test_usd_time_preservation.py`** - Time-varying data validation

### Experiment Tests (EXTREMELY SLOW - Manual Only)
- **`test_experiments.py`** - End-to-end experiment notebook execution (hours to complete)
  - 🔒 **Opt-in only** - Requires `--run-experiments` flag to run
  - ⚠️ **NOT included in CI/CD** - Never runs in automated workflows
  - ⚠️ **Automatically skipped** - Won't run with `pytest tests/` unless flag is set
  - Runs all notebooks in `experiments/` subdirectories
  - Each subdirectory gets its own test
  - Notebooks run in alphanumeric order
  - Requires GPU, CUDA, and all dependencies installed
  - 📖 **See [EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md) for detailed usage instructions**

## 📂 Directory Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── data/                    # Downloaded/input test data
│   └── Slicer-Heart-CT/    # 4D cardiac CT dataset
├── results/                 # Test outputs (organized by module)
├── baseline/                # Regression test baselines
└── *.py                     # Test modules
```

## 🚀 Quick Start

### Install Dependencies
```bash
uv pip install -e ".[test]"
```

### Run Tests
```bash
# Fast tests only (recommended for development)
pytest tests/ -m "not slow" -v

# Specific test module
pytest tests/test_usd_merge.py -v

# All tests (including slow registration tests)
# Note: Experiment tests are automatically skipped
pytest tests/ -v

# Run experiment tests (EXTREMELY SLOW - hours to complete)
# NOTE: Requires --run-experiments flag!
pytest tests/test_experiments.py -v --run-experiments

# Run a specific experiment
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s --run-experiments
```

### Common Test Commands
```bash
# Skip GPU-dependent tests
pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
              --ignore=tests/test_segment_chest_vista_3d.py

# Run with coverage
pytest tests/ --cov=src/physiomotion4d --cov-report=html

# Run specific test class or method
pytest tests/test_usd_merge.py::TestUSDMerge::test_merge_usd_files_copy_method -v
```

> 💡 **For detailed instructions**, see [TESTING_GUIDE.md](TESTING_GUIDE.md)

## ⚙️ Test Configuration

### Global Settings
- **Timeout**: 900 seconds (15 minutes) per test
- **Data Management**: Automatic download with intelligent fallback
- **Output Organization**: Results saved to `tests/results/` by module

### Test Markers
- `@pytest.mark.slow` - Tests taking >30 seconds (registration, segmentation)
- `@pytest.mark.requires_data` - Tests requiring external data download
- `@pytest.mark.integration` - Integration tests vs unit tests
- `@pytest.mark.experiment` - **Experiment tests (EXTREMELY SLOW, manual only, NOT in CI/CD)**
- `@pytest.mark.timeout(seconds)` - Per-test timeout override

### Test Dependencies

Tests are organized hierarchically - some tests depend on outputs from earlier tests:

```
test_download_heart_data
    ↓
test_convert_nrrd_4d_to_3d
    ↓                    ↓
    ↓                    ├─→ test_register_images_ants ──→ test_transform_tools
    ↓                    ├─→ test_register_images_icon
    ↓                    ↓
test_segment_chest_total_segmentator ────→ test_contour_tools
    ↓                                           ↓
test_segment_chest_vista_3d                test_convert_vtk_to_usd_polymesh
```

Fixtures in `conftest.py` automatically manage these dependencies.

## 🔄 Continuous Integration

Tests automatically run on pull requests via GitHub Actions. The CI workflow:

- ✅ **Runs fast tests** - USD utilities, data conversion, basic validation
- ❌ **Skips slow tests** - Registration and segmentation (too slow for CI)
- ❌ **Automatically skips experiment tests** - Protected by `--run-experiments` flag requirement
- ✅ **Caches test data** - Speeds up subsequent runs
- ✅ **Generates coverage** - Reports uploaded to Codecov

**CI Configuration:**
- Platforms: Ubuntu, Windows, macOS
- Python versions: 3.10, 3.11, 3.12
- Target coverage: >70%
- Protection: Experiment tests require `--run-experiments` flag (never used in CI/CD)

> 📖 **For detailed CI/CD information**, see [GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)

## 🛠️ Troubleshooting

### Common Issues

**Problem: HTTP 404 when downloading data**
- ✅ **Fixed!** Tests now check `data/Slicer-Heart-CT/` first
- Place `TruncalValve_4DCT.seq.nrrd` there to avoid download

**Problem: Out of memory errors**
- VISTA-3D requires 20GB+ RAM
- Skip with: `pytest tests/ --ignore=tests/test_segment_chest_vista_3d.py`

**Problem: Test timeout**
- Global timeout: 900 seconds (15 minutes)
- Registration tests need GPU for reasonable speed
- Override with: `@pytest.mark.timeout(1800)` decorator or use `-o timeout=1800`

**Problem: Fixture naming errors**
- ✅ **Fixed!** Use correct fixture names from `conftest.py`
- Don't import from other test files

> 🔍 **For complete troubleshooting guide**, see [TESTING_GUIDE.md](TESTING_GUIDE.md#troubleshooting)

## 📝 Adding New Tests

When creating new tests:

1. **Name properly**: Use `test_` prefix for files and methods
2. **Add markers**: `@pytest.mark.slow`, `@pytest.mark.requires_data`, etc.
3. **Use fixtures**: Define shared fixtures in `conftest.py`
4. **Document well**: Clear docstrings explaining what's validated
5. **Organize outputs**: Save results to `tests/results/<module_name>/`
6. **Update docs**: Add test description to this README

> 💡 **For best practices**, see [TESTING_GUIDE.md](TESTING_GUIDE.md#best-practices)

## 📊 Test Data

### Slicer-Heart-CT Dataset
- **Source**: [Slicer-Heart-CT GitHub](https://github.com/Slicer-Heart-CT/Slicer-Heart-CT)
- **File**: `TruncalValve_4DCT.seq.nrrd` (~1.2 GB)
- **Content**: 21-phase pediatric cardiac CT
- **Usage**: Tests use first 2 time points for speed

### Data Management
Tests automatically:
1. Check `tests/data/Slicer-Heart-CT/`
2. Check `data/Slicer-Heart-CT/` (copies if found)
3. Download from GitHub if needed
4. Skip gracefully if unavailable

## 📚 Additional Resources

- **Detailed Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Experiment Tests Guide**: [EXPERIMENT_TESTS_GUIDE.md](EXPERIMENT_TESTS_GUIDE.md)
- **Parallel Execution Guide**: [PARALLEL_EXECUTION_GUIDE.md](PARALLEL_EXECUTION_GUIDE.md)
- **Experiment Flag Usage**: [EXPERIMENT_FLAG_USAGE.md](EXPERIMENT_FLAG_USAGE.md)
- **CI/CD Documentation**: [GITHUB_WORKFLOWS.md](GITHUB_WORKFLOWS.md)
- **Recent Fixes**: [TEST_FIXES_SUMMARY.md](TEST_FIXES_SUMMARY.md)
- **Main Project**: [../README.md](../README.md)
- **Example Workflows**: [../experiments/Heart-GatedCT_To_USD/](../experiments/Heart-GatedCT_To_USD/)

---

**Need help?** Check the [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive documentation and FAQ.