# PhysioMotion4D Documentation Setup

## Overview

### Configuration Files

1. **`.readthedocs.yaml`** - ReadTheDocs build configuration
   - Uses `pyproject.toml` for dependencies (no separate requirements.txt)
   - Configures Python 3.11, Sphinx build, PDF/ePub formats
   - Includes CPU-only PyTorch for documentation builds

2. **`docs/conf.py`** - Sphinx configuration
   - Theme: sphinx_rtd_theme (ReadTheDocs theme)
   - Extensions: autodoc, napoleon, intersphinx, myst-parser, etc.
   - Mock imports for heavy dependencies
   - Custom CSS styling

3. **`docs/Makefile`** and **`docs/make.bat`** - Build scripts
   - Cross-platform documentation building
   - Standard Sphinx make commands

4. **`.github/workflows/docs.yml`** - GitHub Actions workflow
   - Automatic documentation build on push/PR
   - Deploy to GitHub Pages on main branch
   - Artifact upload for review

### Documentation Structure

```
docs/
├── index.rst                      # Main entry point
├── installation.rst               # Installation guide
├── quickstart.rst                 # Quick start guide
├── examples.rst                   # Code examples
│
├── api/                           # API Reference
│   ├── core.rst                   # Core workflow classes
│   ├── segmentation.rst           # Segmentation methods
│   ├── registration.rst           # Registration methods
│   └── utilities.rst              # Utility functions
│
├── tutorials/                     # Detailed Tutorials
│   ├── basic_workflow.rst         # Complete workflow tutorial
│   ├── custom_segmentation.rst    # Segmentation customization
│   ├── image_registration.rst     # Registration guide
│   ├── vtk_to_usd.rst            # USD conversion
│   ├── colormap_rendering.rst     # Colormap visualization
│   └── model_to_image_registration.rst  # Model fitting
│
├── user_guide/                    # User Guides
│   ├── heart_gated_ct.rst        # Cardiac CT guide
│   ├── lung_4dct.rst             # Lung 4D-CT guide
│   ├── segmentation.rst          # Segmentation overview
│   ├── registration.rst          # Registration overview
│   ├── usd_conversion.rst        # USD export guide
│   └── visualization.rst         # Visualization guide
│
├── contributing.rst               # Contribution guidelines
├── architecture.rst               # System architecture
├── testing.rst                    # Testing guide
├── changelog.rst                  # Version history
├── faq.rst                       # Frequently asked questions
├── troubleshooting.rst           # Common issues
├── references.rst                # Citations and links
│
├── _static/
│   └── custom.css                # Custom styling
│
└── README.md                     # Documentation build guide
```

## Key Features

### API Documentation

- **Complete API reference** with autodoc
- **Four main modules**:
  - Core: Workflow processors and converters
  - Segmentation: TotalSegmentator, VISTA-3D, Ensemble
  - Registration: ICON, ANTs
  - Utilities: Transform, USD, and Contour tools

### Tutorials

- **Basic workflow tutorial**: Step-by-step cardiac CT processing
- **Specialized tutorials**: Segmentation, registration, USD export
- **Code examples**: Practical use cases
- **Troubleshooting**: Common issues and solutions

### User Guides

- **Application-specific**: Heart-gated CT, Lung 4D-CT
- **Component guides**: Segmentation, registration, visualization
- **Best practices**: Parameter selection, quality settings

### Developer Documentation

- **Contributing guide**: Code style, workflow, testing
- **Architecture**: System design and components
- **Testing guide**: Running and writing tests

## Setup for GitHub/ReadTheDocs

### 1. GitHub Repository Setup

The documentation is ready to work with GitHub. When you push to GitHub:

```bash
git add .
git commit -m "Add comprehensive ReadTheDocs documentation"
git push origin main
```

### 2. ReadTheDocs Setup

To enable automatic documentation building:

1. Go to https://readthedocs.org/
2. Sign in with your GitHub account
3. Import your PhysioMotion4D repository
4. ReadTheDocs will automatically:
   - Detect `.readthedocs.yaml`
   - Install dependencies from `pyproject.toml[docs]`
   - Build documentation on every push
   - Create preview builds for PRs

### 3. GitHub Actions

The workflow at `.github/workflows/docs.yml` will:

- Build documentation on every push/PR
- Check for warnings
- Deploy to GitHub Pages (optional)
- Upload artifacts for review

## Building Locally

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Build Documentation

```bash
cd docs
make html
```

### View Documentation

```bash
# Linux
xdg-open _build/html/index.html

# macOS
open _build/html/index.html

# Windows
start _build/html/index.html
```

## Updating Documentation

### Adding New Pages

1. Create new `.rst` file in appropriate directory
2. Add to `toctree` in parent file
3. Build locally to test
4. Commit and push

### Updating API Documentation

API docs are auto-generated from docstrings. To update:

1. Update docstrings in source code
2. Rebuild documentation
3. Commit and push

### Updating pyproject.toml

If you add documentation dependencies, update `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    # ... add new dependencies here
]
```

## Documentation URLs

Once set up on ReadTheDocs:

- **Latest**: https://physiomotion4d.readthedocs.io/en/latest/
- **Stable**: https://physiomotion4d.readthedocs.io/en/stable/
- **Specific version**: https://physiomotion4d.readthedocs.io/en/v2025.05.0/

## Features

### Automatic Updates

- ✅ Documentation builds on every merge to main
- ✅ PR previews for documentation changes
- ✅ Version-specific documentation for releases
- ✅ Search functionality
- ✅ PDF and ePub downloads
- ✅ Mobile-responsive theme

### Quality Assurance

- ✅ Lint checks via GitHub Actions
- ✅ Link checking
- ✅ Warning-as-error option
- ✅ Code example testing (doctest)

### SEO and Discoverability

- ✅ Comprehensive index
- ✅ Cross-references throughout
- ✅ Google-style docstrings
- ✅ Keywords and metadata

## Maintenance

### Regular Updates

- Keep API documentation in sync with code
- Update examples with new features
- Refresh tutorials as workflow changes
- Add FAQ entries for common questions

### Version Management

ReadTheDocs automatically builds documentation for:

- Latest commit on main branch
- All tagged releases
- Pull request previews

## Next Steps

1. **Push to GitHub**: Commit all documentation files
2. **Enable ReadTheDocs**: Import project on readthedocs.org
3. **Update URLs**: Add ReadTheDocs URL to README.md and pyproject.toml
4. **Review Build**: Check first build for any issues
5. **Announce**: Update project description with documentation link

## Support

For issues with documentation:

- Check build logs on ReadTheDocs
- Run `make html` locally to test
- Check `.readthedocs.yaml` configuration
- Verify `pyproject.toml[docs]` dependencies

## Files Modified

### Updated Files

- `pyproject.toml` - Added enhanced docs dependencies
- `.readthedocs.yaml` - Updated to use pyproject.toml only
- `docs/PYPI_RELEASE_GUIDE.md` - Moved from `doc/` directory

### New Files

- All `.rst` documentation files
- `docs/conf.py` - Sphinx configuration
- `docs/_static/custom.css` - Custom styling
- `docs/Makefile` and `docs/make.bat` - Build scripts
- `.github/workflows/docs.yml` - CI workflow
- This file: `docs/DOCUMENTATION_SETUP.md`

## Summary

PhysioMotion4D now has production-ready documentation that will:

✅ Build automatically on GitHub merges
✅ Host on ReadTheDocs with version management
✅ Include comprehensive API reference
✅ Provide detailed tutorials and guides
✅ Support PDF/ePub downloads
✅ Enable search and indexing
✅ Work on mobile devices
✅ Integrate with GitHub Actions

The documentation is complete and ready for deployment!

