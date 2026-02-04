# Experiment Tests Implementation Summary

## Overview

This document summarizes the implementation of automated testing for PhysioMotion4D experiment notebooks.

**Created:** 2026-01-25

## What Was Implemented

A comprehensive test suite (`test_experiments.py`) that automatically executes all Jupyter notebooks in the `experiments/` directory, organized by subdirectory. Each subdirectory gets its own dedicated test function.

### Key Features

1. **Automated notebook execution** - Uses `nbconvert` to execute notebooks in place
2. **Alphanumeric ordering** - Notebooks run in order (0-, 1-, 2-, etc.)
3. **One test per subdirectory** - Each experiment gets its own test function
4. **Long timeouts** - Tests accommodate hours-long execution times
5. **Detailed reporting** - Progress updates, summaries, and failure diagnostics
6. **CI/CD exclusion** - Tests marked to NEVER run in automated workflows
7. **Flexible execution** - Run all experiments or individual ones

### Covered Experiment Subdirectories

1. **Colormap-VTK_To_USD** - VTK colormap conversion to USD ✅ Active
2. **Convert_VTK_To_USD** - VTK to USD using library classes ✅ Active
3. ~~**DisplacementField_To_USD**~~ - Registration displacement field visualization 🚫 **Disabled (notebooks not ready)**
4. **Reconstruct4DCT** - 4D CT reconstruction techniques ✅ Active
5. **Heart-VTKSeries_To_USD** - Heart VTK time series conversion ✅ Active
6. **Heart-GatedCT_To_USD** - Complete cardiac imaging pipeline ✅ Active
7. **Heart-Create_Statistical_Model** - PCA statistical shape model creation ✅ Active
8. **Heart-Statistical_Model_To_Patient** - Heart model to patient registration with PCA ✅ Active
9. **Lung-GatedCT_To_USD** - Lung imaging pipeline with DirLab data ✅ Active
10. ~~**Lung-VesselsAirways**~~ - Vessel and airway segmentation 🚫 **Disabled (notebooks not ready)**

**8 active experiments, 2 disabled**

## Files Created

### 1. `tests/test_experiments.py` (435 lines)

Main test module containing:
- Notebook discovery and execution functions
- One test function per experiment subdirectory
- Helper tests for listing notebooks and validating structure
- Comprehensive docstrings with usage examples

**Key functions:**
- `get_notebooks_in_subdir()` - Find notebooks in a subdirectory
- `execute_notebook()` - Execute a single notebook with timeout
- `run_experiment_notebooks()` - Execute all notebooks in a subdirectory
- `test_experiment_*()` - Individual test functions (8 total)
- `test_experiment_structure()` - Validate experiment directory structure
- `test_list_notebooks_in_subdir()` - Helper to preview notebooks

### 2. `tests/EXPERIMENT_TESTS_GUIDE.md` (480 lines)

Comprehensive usage guide containing:
- Quick start examples
- Detailed command reference
- System and software requirements
- Troubleshooting section
- Advanced usage patterns
- FAQ

### 3. `tests/EXPERIMENT_TESTS_SUMMARY.md` (This file)

Implementation summary and change log.

## Files Modified

### 1. `pyproject.toml`

Added new pytest marker:
```python
markers = [
    # ... existing markers ...
    "experiment: marks tests that run experiment notebooks (extremely slow, manual only)"
]
```

### 2. `tests/README.md`

- Added experiment tests section
- Updated test commands to exclude experiments by default
- Added marker documentation
- Updated CI/CD section
- Added link to EXPERIMENT_TESTS_GUIDE.md

### 3. `.github/workflows/ci.yml`

Updated all pytest commands to exclude experiment tests:
```bash
pytest tests/ -v -m "not slow and not requires_data and not experiment"
```

Changes in:
- Unit tests job
- Integration tests job
- GPU tests job
- Notes section

### 4. `.github/workflows/test-slow.yml`

Updated slow test command:
```bash
pytest tests/ -v -m "slow and not experiment"
```

### 5. `experiments/README.md`

Added **Automated Testing** section with:
- Usage examples
- Test features description
- Requirements
- Important warnings

## Usage Examples

### Run All Experiments

```bash
pytest tests/test_experiments.py -v -m experiment
```

### Run Single Experiment

```bash
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s
```

### List Notebooks (No Execution)

```bash
pytest tests/test_experiments.py::test_list_notebooks_in_subdir -v -s
```

### Validate Structure

```bash
pytest tests/test_experiments.py::test_experiment_structure -v
```

## Test Characteristics

### Timeouts

| Test                               | Per-Notebook   | Total Test     | Status     |
| ---------------------------------- | -------------- | -------------- | ---------- |
| Colormap VTK to USD                | 3600s (1h)     | 7200s (2h)     | ✅ Active   |
| Convert VTK to USD                 | 3600s (1h)     | 7200s (2h)     | ✅ Active   |
| ~~Displacement Field~~             | ~~3600s (1h)~~ | ~~7200s (2h)~~ | 🚫 Disabled |
| Reconstruct 4DCT                   | 7200s (2h)     | 14400s (4h)    | ✅ Active   |
| Heart VTK Series                   | 5400s (1.5h)   | 10800s (3h)    | ✅ Active   |
| Heart Gated CT                     | 5400s (1.5h)   | 21600s (6h)    | ✅ Active   |
| Create Statistical Model           | 5400s (1.5h)   | 10800s (3h)    | ✅ Active   |
| Heart Statistical Model to Patient | 7200s (2h)     | 14400s (4h)    | ✅ Active   |
| Lung Gated CT                      | 5400s (1.5h)   | 21600s (6h)    | ✅ Active   |
| ~~Lung Vessels Airways~~           | ~~3600s (1h)~~ | ~~7200s (2h)~~ | 🚫 Disabled |

**Total for active experiments: ~32 hours** (2 experiments disabled)

### Markers

All experiment tests have:
- `@pytest.mark.experiment` (always)
- `@pytest.mark.slow` (always)
- `@pytest.mark.requires_gpu` (when applicable)
- `@pytest.mark.requires_data` (when applicable)
- `@pytest.mark.timeout(seconds)` (specific to test)

## CI/CD Integration

**IMPORTANT:** Experiment tests are **NEVER** run in CI/CD workflows.

All GitHub Actions workflows explicitly exclude them:
- `ci.yml` - Unit, integration, and GPU tests
- `test-slow.yml` - Slow tests workflow
- `docs.yml` - Documentation building (doesn't run tests)

## Design Decisions

### 1. Execution Method: nbconvert

**Chosen:** `nbconvert --execute --inplace`

**Why:**
- Standard tool, comes with JupyterLab
- Executes notebooks with real kernel
- Preserves outputs in notebook file
- Good timeout and error handling

**Alternatives considered:**
- papermill - More features, but extra dependency
- nbclient - Lower level, more complex
- Manual kernel execution - Too complex

### 2. One Test Per Subdirectory

**Why:**
- Clear organization and naming
- Independent execution
- Easy to run individual experiments
- Matches mental model of experiments

**Alternative:** Single test with parametrization
- Pros: Less code duplication
- Cons: Harder to run specific experiments, less clear output

### 3. In-Place Execution

**Why:**
- Notebooks retain execution results
- Easier debugging (open notebook to see outputs)
- Matches interactive development workflow

**Downside:** Modifies notebooks (consider using nbstripout)

### 4. Alphanumeric Ordering

**Why:**
- Respects numbered prefixes (0-, 1-, 2-)
- Matches dependency order in experiments
- Intuitive and predictable

**Implementation:** `sorted(subdir.glob('*.ipynb'))`

### 5. Long Timeouts

**Why:**
- Experiments are genuinely long-running
- GPU operations can be slow
- Data downloads can take time
- Better to have generous timeouts than false failures

**Trade-off:** Hung notebooks take long to fail

## Testing the Tests

To verify the implementation works:

1. **Structure validation:**
   ```bash
   pytest tests/test_experiments.py::test_experiment_structure -v
   ```

2. **Notebook discovery:**
   ```bash
   pytest tests/test_experiments.py::test_list_notebooks_in_subdir -v -s
   ```

3. **Run fastest experiment:**
   ```bash
   pytest tests/test_experiments.py::test_experiment_colormap_vtk_to_usd -v -s
   ```

4. **Verify CI/CD exclusion:**
   ```bash
   # Should NOT include test_experiments.py in output
   pytest tests/ --collect-only -m "not slow and not experiment"
   ```

## Future Enhancements

Potential improvements for future iterations:

1. **Parallel execution** - Run independent experiments simultaneously
2. **HTML reports** - Generate browsable report from executed notebooks
3. **Artifact collection** - Automatically gather generated USD files, images
4. **Smoke tests** - Quick mode that only runs first cell of each notebook
5. **Parameterization** - Use papermill to parameterize notebook execution
6. **Checkpointing** - Resume from failed notebook in sequence
7. **Resource monitoring** - Track memory, GPU, disk usage during execution
8. **Baseline comparison** - Compare outputs with known-good results
9. **Partial execution** - Run only specific notebooks within a subdirectory
10. **Cleanup hooks** - Automatically remove temporary files after tests

## Maintenance Notes

### Adding New Experiments

When adding a new experiment subdirectory:

1. Add to `EXPERIMENT_SUBDIRS` list in `test_experiments.py`
2. Create a test function following the naming pattern
3. Choose appropriate markers and timeout
4. Document expected notebooks in docstring
5. Update this summary and the guide

### Modifying Timeouts

If experiments become faster/slower:

1. Update `timeout_per_notebook` argument
2. Update `@pytest.mark.timeout()` decorator
3. Update documentation (guide and summary)

### Handling Notebook Changes

If notebook dependencies or order changes:

- Tests automatically use alphanumeric ordering
- No code changes needed unless notebooks are added/removed
- Consider updating test docstrings if notebook purposes change

## Known Limitations

1. **Not deterministic** - Some experiments use random seeds or external data
2. **No output validation** - Tests only check execution success, not correctness
3. **Resource intensive** - Requires significant compute, GPU, memory, disk
4. **Long runtime** - Hours to complete all tests
5. **External dependencies** - Requires data downloads, internet connectivity
6. **Modifies files** - Notebooks executed in-place with outputs saved
7. **Limited parallelism** - Tests run sequentially by default
8. **No cleanup** - Generated files not automatically removed

## Documentation References

- **User Guide:** `tests/EXPERIMENT_TESTS_GUIDE.md`
- **Main Test Docs:** `tests/README.md`
- **Experiment Docs:** `experiments/README.md`
- **CI/CD Config:** `.github/workflows/ci.yml`
- **Pytest Config:** `pyproject.toml` (markers, testpaths)

## Support and Troubleshooting

For issues:
1. Consult `EXPERIMENT_TESTS_GUIDE.md`
2. Check test output and notebook execution results
3. Verify system requirements (GPU, memory, disk)
4. Try running notebook manually in JupyterLab
5. Check GitHub issues or open new one

## Conclusion

This implementation provides a solid foundation for automated validation of PhysioMotion4D experiment notebooks. The test suite is:

- ✅ Comprehensive - Covers all 10 experiment subdirectories (8 active, 2 disabled)
- ✅ Well-documented - Multiple documentation files and guides
- ✅ Excluded from CI/CD - Properly marked and filtered
- ✅ Flexible - Run all experiments or individual ones
- ✅ Maintainable - Clear structure and extensible design
- ✅ Production-ready - Error handling, timeouts, reporting

The tests serve as both validation tools and living documentation of the experiment workflows.

---

**Last Updated:** 2026-01-25
**Implementation By:** AI Assistant (Claude Sonnet 4.5)
**Review Status:** Pending human review

## Change Log

### 2026-01-25 - Initial Implementation
- Created comprehensive test suite for 8 experiment subdirectories
- Implemented automated notebook execution with nbconvert
- Added extensive documentation and guides

### 2026-01-25 - Disabled Non-Ready Experiments
- Disabled `DisplacementField_To_USD` (notebooks not ready)
- Disabled `Lung-VesselsAirways` (notebooks not ready)
- Tests commented out but can be easily re-enabled
- Updated documentation to reflect active status
- 6 experiments remain active (~25 hours total runtime)

### 2026-01-25 - Added Opt-In Protection
- Added `--run-experiments` flag to pytest via conftest.py
- Experiment tests now automatically skipped unless flag is provided
- Prevents accidental execution with `pytest tests/` or similar commands
- Enhanced protection against CI/CD execution
- Updated all documentation to reflect new flag requirement

### 2026-01-25 - Enforced Sequential Notebook Execution
- Added explicit sequential execution enforcement within each experiment test
- Added `@pytest.mark.xdist_group` markers for pytest-xdist compatibility
- Modified `run_experiment_notebooks()` to stop on first failure
- Added detailed execution order documentation to each test function
- Added pytest-xdist>=3.0.0 to test dependencies
- Ensures notebooks run in strict alphanumeric order even with parallel workers
- Different experiment subdirectories CAN run in parallel safely
- Notebooks within a subdirectory CANNOT run in parallel or out of order

### 2026-01-30 - Added Heart-Create_Statistical_Model and Renamed Heart-Model_To_Patient
- Added `Heart-Create_Statistical_Model` experiment (PCA shape model creation with SlicerSALT)
- Renamed `Heart-Model_To_Patient` to `Heart-Statistical_Model_To_Patient` for clarity
- Added `Convert_VTK_To_USD` experiment test
- Updated all documentation to reflect experiment reorganization
- Created comprehensive README.md for Heart-Create_Statistical_Model experiment
- Updated data/KCL-Heart-Model/README.md to reference new experiment
- Total active experiments: 8 (up from 6)
- Total estimated runtime: ~32 hours (up from ~25 hours)
