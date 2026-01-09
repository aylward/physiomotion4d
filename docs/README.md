# PhysioMotion4D Documentation

This directory contains the source files for PhysioMotion4D's documentation, which is built using [Sphinx](https://www.sphinx-doc.org/) and hosted on [ReadTheDocs](https://readthedocs.org/).

## 🎉 Recently Updated!

The documentation has been restructured with:
- ✅ Modern table-of-contents sidebar with search
- ✅ Separate pages for each module (33 new API files)
- ✅ Navigation widgets (prev/next/up) on all pages
- ✅ Comprehensive API reference organized by functionality
- ✅ Enhanced search capabilities
- ✅ Modern custom CSS styling

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

**Linux/macOS:**

```bash
cd docs
make html
```

**Windows:**

```bash
cd docs
make.bat html
```

### View Documentation

The built documentation will be in `docs/_build/html/`. Open `index.html` in your browser:

**Linux:**
```bash
xdg-open _build/html/index.html
```

**macOS:**
```bash
open _build/html/index.html
```

**Windows:**
```bash
start _build/html/index.html
```

## Documentation Structure

```
docs/
├── index.rst                    # Main entry point
├── installation.rst             # Installation guide
├── quickstart.rst               # Quick start guide
├── examples.rst                 # Code examples
│
├── api/                         # 📚 API Reference (NEW STRUCTURE!)
│   ├── index.rst               # Main API hub
│   ├── base.rst                # Core base class
│   ├── workflows.rst           # Workflow classes
│   ├── segmentation/           # Segmentation (6 files)
│   │   ├── index.rst
│   │   ├── base.rst
│   │   ├── totalsegmentator.rst
│   │   ├── vista3d.rst
│   │   ├── vista3d_nim.rst
│   │   └── ensemble.rst
│   ├── registration/           # Image registration (5 files)
│   │   ├── index.rst
│   │   ├── base.rst
│   │   ├── ants.rst
│   │   ├── icon.rst
│   │   └── time_series.rst
│   ├── model_registration/     # Model registration (5 files)
│   ├── usd/                    # USD generation (6 files)
│   └── utilities/              # Utilities (5 files)
│
├── developer/                   # 👨‍💻 Developer guides
│   ├── architecture.rst
│   ├── extending.rst
│   ├── workflows.rst
│   └── core.rst
│
├── cli_scripts/                 # 🔧 CLI documentation
├── contributing.rst
├── testing.rst
└── ...
```

**Total: 33 new API documentation files!**

## Building Other Formats

### PDF

```bash
make latexpdf
```

### ePub

```bash
make epub
```

### Check for Errors

```bash
make linkcheck  # Check for broken links
make doctest    # Run documentation code examples
```

## ReadTheDocs

Documentation is automatically built and deployed to ReadTheDocs on:

- Push to `main` branch
- Pull requests (preview builds)
- Tagged releases

Configuration is in `.readthedocs.yaml` at the repository root.

## Contributing to Documentation

1. Make changes to `.rst` files
2. Build locally to test: `make html`
3. Check for warnings: `make clean && make html`
4. Commit and push changes
5. Documentation will be built automatically by ReadTheDocs

### Documentation Style

- Use reStructuredText (`.rst`) format
- Follow existing structure and formatting
- Include code examples with syntax highlighting
- Add cross-references with `:doc:` and `:class:` roles
- Keep line length reasonable (~80-100 characters)

### Adding New Pages

1. Create new `.rst` file in appropriate directory
2. Add to `toctree` in `index.rst` or parent file
3. Follow existing naming conventions (lowercase, underscores)

## Cleaning Build Files

```bash
make clean
```

This removes the `_build` directory with all generated documentation.

## Help

For more information:

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [ReadTheDocs Guide](https://docs.readthedocs.io/)

