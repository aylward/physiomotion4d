# GitHub Workflows for Tests

This document describes the GitHub Actions workflows configured for running tests on pull requests.

## Workflow Configuration

File: `.github/workflows/tests.yml`

## Workflow Jobs

### 1. Unit Tests (`test` job)
**Trigger**: All pull requests and pushes to main
**Platforms**: Ubuntu, Windows, macOS
**Python versions**: 3.10, 3.11, 3.12

**What runs**:
- Tests that don't require external data
- Tests that aren't marked as slow
- Coverage reporting to Codecov

**Command**:
```bash
pytest tests/ -v -m "not requires_data and not slow" --cov=src/physiomotion4d --cov-report=xml
```

**Tests included**:
- Basic unit tests
- Fast integration tests with mocked data

**Tests excluded**:
- Data-dependent tests (marked with `@pytest.mark.requires_data`)
- Slow/GPU tests (marked with `@pytest.mark.slow`)

---

### 2. Integration Tests with Data (`test-with-data` job)
**Trigger**: Pull requests only
**Platform**: Ubuntu only
**Python version**: 3.10

**What runs**:
Tests that require downloading external data, executed in sequence with caching.

#### Test Steps:

1. **Data Download Tests**
   ```bash
   pytest tests/test_download_heart_data.py -v --cov=src/physiomotion4d --cov-report=xml
   ```
   - Downloads TruncalValve 4D CT data
   - Sets up test directories
   - Cached for subsequent runs

2. **Data Conversion Tests**
   ```bash
   pytest tests/test_convert_nrrd_4d_to_3d.py -v --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Converts 4D NRRD to 3D time series
   - Creates slice files
   - Depends on downloaded data

3. **Contour Tools Tests**
   ```bash
   pytest tests/test_contour_tools.py -v -m "not slow" --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Tests contour extraction
   - Tests mesh manipulation
   - Uses segmentation results (creates if missing)
   - Excludes slow GPU-dependent segmentation

4. **USD Conversion Tests**
   ```bash
   pytest tests/test_convert_vtk_4d_to_usd_polymesh.py -v -m "not slow" --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - Tests VTK to USD conversion
   - Uses contour data
   - Multiple conversion scenarios

5. **Existing USD Tests**
   ```bash
   pytest tests/test_usd_merge.py tests/test_usd_time_preservation.py -v --cov=src/physiomotion4d --cov-append --cov-report=xml
   ```
   - USD file merging
   - Time preservation validation

#### Caching Strategy:
- **Cached directories**: `tests/data/`, `tests/results/`
- **Cache key**: `test-data-${{ hashFiles('tests/test_*.py') }}-v2`
- **Restore keys**: `test-data-`
- **Cache invalidation**: Changes to test files or manual version bump

#### Error Handling:
All data-dependent test steps use `continue-on-error: true` to prevent CI failures due to:
- Network issues during data download
- Transient failures
- Missing dependencies

Coverage is still uploaded even if tests fail.

---

## Tests Excluded from CI

### Slow and GPU-Dependent Tests

These tests are **NOT** run in CI because they are slow or require CUDA-enabled GPUs:

1. **ANTs Registration Tests** (`test_register_images_ants.py`)
   - Requires: ANTsPy library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_register_images_ants.py -v -s`
   - Why excluded: Computationally intensive, slow execution time

2. **ICON Registration Tests** (`test_register_images_icon.py`)
   - Requires: CUDA GPU, ICON library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_register_images_icon.py -v -s`
   - Why excluded: Requires GPU, deep learning inference

3. **Transform Tools Tests** (`test_transform_tools.py`)
   - Requires: Registration results from ANTs tests
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_transform_tools.py -v -s`
   - Why excluded: Depends on slow registration tests

4. **TotalSegmentator Tests** (`test_segment_chest_total_segmentator.py`)
   - Requires: CUDA GPU, TotalSegmentator library
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_segment_chest_total_segmentator.py -v -s`
   - Why excluded: Requires GPU, model inference

5. **VISTA-3D Tests** (`test_segment_chest_vista_3d.py`)
   - Requires: CUDA GPU, VISTA-3D model weights
   - Markers: `@pytest.mark.requires_data`, `@pytest.mark.slow`
   - Run locally: `pytest tests/test_segment_chest_vista_3d.py -v -s`
   - Why excluded: Requires GPU, model inference

**Why excluded**:
- GitHub Actions runners don't have CUDA-enabled GPUs
- Model inference requires significant GPU memory and time
- Registration algorithms are computationally intensive
- Tests would timeout or fail without GPU/sufficient compute
- CI runtime would exceed reasonable limits

**Local testing**:
```bash
# Run all slow tests (registration + segmentation)
pytest tests/ -v -m "slow"

# Run only registration tests
pytest tests/test_register_images_ants.py tests/test_register_images_icon.py -v -s

# Run only segmentation tests
pytest tests/test_segment_chest_total_segmentator.py tests/test_segment_chest_vista_3d.py -v -s
```

---

## Coverage Reporting

### Unit Tests Coverage
- **Flag**: `unittests`
- **When**: Ubuntu + Python 3.10 only
- **Upload**: After unit tests complete

### Integration Tests Coverage
- **Flag**: `integration-tests`
- **When**: After all data-dependent tests complete
- **Upload**: Even if tests fail (`fail_ci_if_error: false`)

### Viewing Coverage
Coverage reports are uploaded to Codecov and can be viewed at:
- Repository Codecov dashboard
- Pull request comments (automatic)
- Coverage badge in README

---

## Test Execution Flow

```
Pull Request Created
    ↓
┌───────────────────────────────────────────────┐
│ Unit Tests (test job)                         │
│ - Runs on Ubuntu, Windows, macOS              │
│ - Python 3.10, 3.11, 3.12                    │
│ - Fast, no external data                      │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│ Integration Tests (test-with-data job)        │
│ - Runs on Ubuntu only                         │
│ - Python 3.10                                 │
│                                               │
│ 1. Cache Check (tests/data/, tests/results/) │
│    ↓                                          │
│ 2. Download Data (if cache miss)             │
│    ↓                                          │
│ 3. Convert 4D to 3D                          │
│    ↓                                          │
│ 4. Test Contour Tools (no GPU)              │
│    ↓                                          │
│ 5. Test USD Conversion                       │
│    ↓                                          │
│ 6. Test USD Utilities                        │
│    ↓                                          │
│ 7. Upload Coverage                           │
└───────────────────────────────────────────────┘
    ↓
Tests Complete (Pass/Fail reported on PR)
```

---

## Manual Workflow Dispatch

Currently not configured, but can be added for:
- Running GPU tests on self-hosted runners
- Running full test suite on-demand
- Testing with different configurations

---

## Monitoring and Debugging

### Check Workflow Status
1. Go to repository → Actions tab
2. Select workflow run
3. View job logs

### Common Issues

**Data Download Fails**:
- Check network connectivity
- Verify data URL is accessible
- Review cache restoration logs

**Cache Miss**:
- Normal on first run
- Check cache key matches pattern
- Verify test files haven't changed significantly

**Test Timeouts**:
- Increase timeout in workflow
- Optimize slow tests
- Consider parallel execution

### Workflow Files
- Main workflow: `.github/workflows/tests.yml`
- Test configuration: `tests/conftest.py`
- Test documentation: `tests/TEST_ORGANIZATION.md`

---

## Future Enhancements

Potential improvements for the workflow:

1. **Self-hosted GPU Runner**
   - Add GPU-enabled runner for segmentation tests
   - Run full test suite including GPU tests

2. **Parallel Test Execution**
   - Run independent tests in parallel
   - Reduce total CI time

3. **Matrix Testing for Data Tests**
   - Test with different data sources
   - Test with different model versions

4. **Scheduled Runs**
   - Daily/weekly full test runs
   - Regression testing with baseline data

5. **Manual Workflow Dispatch**
   - On-demand test execution
   - Custom parameter inputs

