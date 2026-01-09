# How to Build and Preview Documentation

## Quick Start (Windows)

### 1. One-Time Setup

Install documentation dependencies (only needed once):

```bash
# From project root
cd c:/src/Projects/PhysioMotion/physiomotion4d
source venv/Scripts/activate
pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints sphinx-copybutton sphinx-tabs myst-parser
```

### 2. Build Documentation

Every time you make changes to documentation files:

```bash
# From project root
cd c:/src/Projects/PhysioMotion/physiomotion4d
source venv/Scripts/activate
cd docs
sphinx-build -b html . _build/html
```

### 3. Preview in Browser

```bash
# From docs directory
start _build/html/index.html
```

Or simply navigate to:
```
c:\src\Projects\PhysioMotion\physiomotion4d\docs\_build\html\index.html
```

## Complete Workflow

```bash
# 1. Activate virtual environment
cd c:/src/Projects/PhysioMotion/physiomotion4d
source venv/Scripts/activate

# 2. Make your documentation changes
# Edit files in docs/ directory

# 3. Build documentation
cd docs
sphinx-build -b html . _build/html

# 4. Preview in browser
start _build/html/index.html

# 5. If satisfied, commit and push
git add docs/
git commit -m "Update documentation"
git push
```

## Fast Rebuild (After Making Changes)

```bash
# Quick command to rebuild and open
cd c:/src/Projects/PhysioMotion/physiomotion4d && source venv/Scripts/activate && cd docs && sphinx-build -b html . _build/html && start _build/html/index.html
```

## Common Commands

### Clean Build (Remove Old Files)

```bash
cd docs
rm -rf _build
sphinx-build -b html . _build/html
```

### Check for Errors/Warnings Only

```bash
cd docs
sphinx-build -b html . _build/html -W
```
The `-W` flag treats warnings as errors.

### Build with Verbose Output

```bash
cd docs
sphinx-build -b html . _build/html -v
```

### Check for Broken Links

```bash
cd docs
sphinx-build -b linkcheck . _build/linkcheck
```

## What to Preview

After building, you should check:

### ✅ Navigation
- [ ] Sidebar TOC shows all sections
- [ ] Sidebar is expandable/collapsible
- [ ] Current page is highlighted
- [ ] Search box is visible and functional

### ✅ Page Content
- [ ] Previous/Next buttons appear at top and bottom
- [ ] Navigation links at bottom work
- [ ] Code examples are properly formatted
- [ ] Tables display correctly
- [ ] Cross-references (links) work

### ✅ Module Pages
- [ ] All API module pages load correctly
- [ ] `api/segmentation/index.rst` and subpages
- [ ] `api/registration/index.rst` and subpages
- [ ] `api/model_registration/index.rst` and subpages
- [ ] `api/usd/index.rst` and subpages
- [ ] `api/utilities/index.rst` and subpages

### ✅ Search
- [ ] Search box finds relevant pages
- [ ] Module index is complete
- [ ] General index is complete

### ✅ Cross-References
- [ ] Links between pages work
- [ ] "See Also" sections link correctly
- [ ] Quick links in index pages work

## Troubleshooting

### Build Errors

If you see errors during build:

1. **Check file syntax**: Ensure `.rst` files have correct reStructuredText syntax
2. **Check indentation**: RST is indentation-sensitive
3. **Check references**: Ensure all `:doc:` and `:class:` references are valid

### Missing Content

If pages are missing:

1. **Check toctree**: Ensure page is listed in a `.. toctree::` directive
2. **Check file location**: Ensure file is in correct directory
3. **Rebuild from scratch**: `rm -rf _build && sphinx-build -b html . _build/html`

### Sphinx Not Found

If you get "sphinx-build: command not found":

1. Ensure virtual environment is activated: `source venv/Scripts/activate`
2. Reinstall sphinx: `pip install sphinx sphinx_rtd_theme`

### Import Errors During Build

If you see Python import errors:

- These are expected! The `conf.py` file has mock imports for heavy dependencies
- The documentation will still build correctly
- These are just warnings, not actual errors

## Tips

### Live Reload

For continuous preview while editing, you can use `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```

This will:
- Build documentation automatically when files change
- Serve docs at `http://localhost:8000`
- Auto-refresh browser when changes are detected

### Faster Builds

For faster iteration during development:

```bash
# Only rebuild changed files
sphinx-build -b html . _build/html

# Use multiple processes
sphinx-build -b html . _build/html -j auto
```

### Preview Specific Page

To quickly jump to a specific page:

```bash
# Build and open specific page
cd docs
sphinx-build -b html . _build/html
start _build/html/api/segmentation/index.html
```

## File Locations

After building, find files at:

```
docs/_build/html/
├── index.html                     # Main page
├── api/
│   ├── index.html                # API reference hub
│   ├── segmentation/
│   │   ├── index.html
│   │   ├── totalsegmentator.html
│   │   └── ...
│   ├── registration/
│   ├── usd/
│   └── ...
├── _static/                       # CSS, JS, images
├── _sources/                      # RST source files
└── genindex.html                  # General index
```

## Before Committing

Always preview before committing:

1. ✅ Build documentation: `sphinx-build -b html . _build/html`
2. ✅ Check for errors/warnings in build output
3. ✅ Preview in browser: `start _build/html/index.html`
4. ✅ Navigate through changed pages
5. ✅ Test navigation links
6. ✅ Test search functionality
7. ✅ Check code examples render correctly

Then commit:

```bash
git add docs/
git commit -m "Update documentation: [describe changes]"
git push
```

## ReadTheDocs

When you push to GitHub, ReadTheDocs will automatically:
- Build the documentation
- Deploy to the documentation site
- Create a preview for pull requests

You can preview the ReadTheDocs build at:
- Production: https://physiomotion4d.readthedocs.io/
- PR previews: Available in PR checks

## Summary

**Quick Build & Preview:**
```bash
cd c:/src/Projects/PhysioMotion/physiomotion4d && source venv/Scripts/activate && cd docs && sphinx-build -b html . _build/html && start _build/html/index.html
```

**Location of Built Docs:**
```
docs/_build/html/index.html
```

**What to Check:**
- Navigation works
- Links are correct
- Content displays properly
- Search functions
- No build errors

Happy documenting! 📚
