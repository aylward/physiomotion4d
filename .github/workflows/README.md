# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated testing and CI/CD.

## Workflows

### `test.yml` - Main Test Suite

Runs on every push and pull request to main branches. Includes:

- **test-cpu**: Unit tests on CPU across Python 3.10, 3.11, and 3.12
  - Uses PyTorch CPU version to avoid GPU dependencies
  - Runs tests marked with `unit` and excludes GPU-requiring tests
  - Generates coverage reports

- **test-gpu**: Tests on self-hosted GPU runners (if available)
  - Uses PyTorch with CUDA 12.6 support
  - Runs all tests except those marked as slow
  - Requires self-hosted runner with `[self-hosted, linux, gpu]` labels

- **test-integration**: Integration tests on CPU
  - Runs after CPU tests pass
  - Tests marked with `integration` marker

- **code-quality**: Static code analysis
  - Black, isort, ruff, flake8 checks
  - Does not fail the build (continue-on-error: true)

### `test-slow.yml` - Long-Running Tests

Runs nightly or on manual trigger. Includes:

- **test-slow-gpu**: Slow tests requiring GPU
  - Tests marked with `slow` marker
  - Extended timeout (3600 seconds)
  - Uses self-hosted GPU runners

## Caching Strategy

The workflows use multiple caching layers to speed up builds:

1. **Python package cache** via `setup-python` action
   - Caches pip packages based on `pyproject.toml` hash

2. **Additional pip cache** via `actions/cache`
   - Caches `~/.cache/pip` directory
   - Separate keys for CPU, GPU, and integration tests
   - Hierarchical restore keys for fallback

## GPU Support

### Self-Hosted Runners

GPU tests require self-hosted runners with:
- Linux OS
- NVIDIA GPU with CUDA 12.6+ support
- Runner labels: `[self-hosted, linux, gpu]`

### Setting Up Self-Hosted GPU Runners

1. **Install GitHub Actions Runner**:
   ```bash
   # Download and configure runner from GitHub repository Settings > Actions > Runners
   ```

2. **Install NVIDIA Drivers and CUDA**:
   ```bash
   # Install NVIDIA drivers
   sudo apt-get install nvidia-driver-535
   
   # Install CUDA toolkit 12.6
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-12-6
   ```

3. **Configure Runner Labels**:
   - Add labels: `self-hosted`, `linux`, `gpu`
   - Verify GPU is accessible: `nvidia-smi`

4. **Start the Runner**:
   ```bash
   ./run.sh
   ```

### GitHub-Hosted Runners

GitHub-hosted runners do **not** have GPU support. GPU tests will be skipped automatically if no self-hosted runners are available (`continue-on-error: true`).

## Test Dependencies

Test dependencies are installed from `pyproject.toml`:

```bash
pip install -e ".[test]"
```

This installs:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-xdist >= 3.0.0
- pytest-timeout >= 2.0.0
- coverage[toml] >= 7.0.0

## Test Markers

Tests should be marked appropriately:

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    """Fast unit test"""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test"""
    pass

@pytest.mark.slow
def test_long_running():
    """Long-running test"""
    pass

@pytest.mark.requires_gpu
def test_gpu_function():
    """Test requiring GPU"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    pass
```

## Running Tests Locally

### CPU Tests
```bash
# Install dependencies
pip install -e ".[test]"

# Run unit tests
pytest tests/ -m "unit and not requires_gpu"

# Run with coverage
pytest tests/ -m "unit and not requires_gpu" --cov=physiomotion4d
```

### GPU Tests
```bash
# Install with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[test]"

# Run all tests (including GPU)
pytest tests/ -m "not slow"

# Run slow tests
pytest tests/ -m "slow"
```

## Coverage Reports

Coverage reports are:
1. Uploaded to Codecov (if configured)
2. Stored as artifacts for 7 days
3. Available as HTML reports in the `htmlcov/` directory

## Troubleshooting

### GPU Tests Not Running

If GPU tests are not running:
1. Verify self-hosted runner is online: Settings > Actions > Runners
2. Check runner labels include `gpu`
3. Verify `nvidia-smi` works on the runner
4. Check workflow logs for runner assignment

### Cache Not Working

If builds are slow:
1. Check cache hit/miss in workflow logs
2. Verify `pyproject.toml` hasn't changed unexpectedly
3. Try clearing caches: Settings > Actions > Caches

### Test Failures

For test failures:
1. Check individual test logs in the workflow run
2. Run tests locally to reproduce
3. Use `pytest -v --tb=long` for detailed error traces
4. Check if tests are marked correctly (unit/integration/slow/requires_gpu)

