# Documentation Structure Update

## Overview

The PhysioMotion4D documentation has been restructured to provide a modern, searchable interface with a comprehensive table-of-contents panel and improved navigation.

## Key Changes

### 1. Enhanced Configuration (`conf.py`)

**Search & Navigation Improvements:**
- Enabled `prev_next_buttons_location: 'both'` for navigation at top and bottom of pages
- Increased `navigation_depth: 5` to show full module hierarchy
- Added search language configuration
- Enabled page source links and breadcrumbs
- Configured split index for better organization

**Theme Options:**
- Kept sidebar expanded (`collapse_navigation: False`)
- Sticky navigation enabled for better UX
- Full TOC hierarchy visible (`titles_only: False`)

### 2. New API Reference Structure

Created a comprehensive API reference organized by functionality:

```
docs/api/
├── index.rst                    # Main API reference index
├── base.rst                     # PhysioMotion4DBase class
├── workflows.rst                # Workflow classes
├── segmentation/
│   ├── index.rst               # Segmentation overview
│   ├── base.rst                # SegmentChestBase
│   ├── totalsegmentator.rst    # TotalSegmentator
│   ├── vista3d.rst             # VISTA-3D
│   ├── vista3d_nim.rst         # VISTA-3D NIM
│   └── ensemble.rst            # Ensemble segmentation
├── registration/
│   ├── index.rst               # Registration overview
│   ├── base.rst                # RegisterImagesBase
│   ├── ants.rst                # ANTs registration
│   ├── icon.rst                # Icon registration
│   └── time_series.rst         # Time series registration
├── model_registration/
│   ├── index.rst               # Model registration overview
│   ├── icp.rst                 # ICP
│   ├── icp_itk.rst            # ICP-ITK
│   ├── distance_maps.rst       # Distance map registration
│   └── pca.rst                 # PCA registration
├── usd/
│   ├── index.rst               # USD generation overview
│   ├── tools.rst               # USD tools
│   ├── anatomy_tools.rst       # Anatomy tools
│   ├── vtk_conversion.rst      # VTK conversion
│   ├── polymesh.rst            # PolyMesh
│   └── tetmesh.rst             # TetMesh
└── utilities/
    ├── index.rst               # Utilities overview
    ├── image_tools.rst         # Image utilities
    ├── transform_tools.rst     # Transform utilities
    ├── contour_tools.rst       # Contour utilities
    └── nrrd_conversion.rst     # NRRD conversion
```

### 3. Module Organization

Each module section now includes:

**Index Pages:**
- Overview of the module category
- Quick links to all classes/functions
- Comparison tables (where applicable)
- Quick start examples
- Navigation to related modules

**Individual Module Pages:**
- Complete class/function documentation with autodoc
- Usage examples
- Best practices
- Common patterns
- Performance tips
- Navigation widgets (prev/next/up)

### 4. Navigation Widgets

Every page includes navigation links at the bottom:
- Previous page
- Up to index
- Next page

Example:
```rst
.. rubric:: Navigation

:doc:`previous_page` | :doc:`index` | :doc:`next_page`
```

### 5. Updated Main Index

The main `docs/index.rst` now has clearer sections:

**Before:**
```rst
.. toctree::
   :caption: Developer & API Reference
   
   developer/architecture
   developer/core
   developer/workflows
   ...
```

**After:**
```rst
.. toctree::
   :caption: API Reference
   
   api/index
   api/base
   api/workflows
   api/segmentation/index
   api/registration/index
   api/model_registration/index
   api/usd/index
   api/utilities/index

.. toctree::
   :caption: Developer Guides
   
   developer/architecture
   developer/extending
   developer/workflows
   developer/core
```

### 6. Developer Guide Updates

All developer guides now:
- Start with a clear purpose statement
- Link to corresponding API documentation
- Focus on development patterns rather than API details
- Cross-reference related modules

## Benefits

### For Users:

1. **Better Discoverability**: Modules organized by functionality
2. **Easier Navigation**: Clear hierarchy in sidebar, navigation widgets on every page
3. **Improved Search**: Enhanced search configuration finds relevant content faster
4. **Quick Reference**: Index pages provide quick links and comparison tables
5. **Contextual Help**: Each page links to related modules

### For Developers:

1. **Clear Separation**: API reference vs. development guides
2. **Consistent Structure**: All modules follow same organization pattern
3. **Easy Extension**: Template structure for adding new modules
4. **Better Maintenance**: Smaller, focused files easier to update

## Building the Documentation

To build the updated documentation:

```bash
# Activate virtual environment
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install documentation dependencies (if needed)
pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints sphinx-copybutton sphinx-tabs myst-parser

# Build HTML documentation
cd docs
sphinx-build -b html . _build/html

# Or use make (if available)
make html

# View the documentation
# Open _build/html/index.html in your browser
```

## Search Functionality

The documentation now includes:

1. **Full-text search**: Search across all documentation
2. **Module index**: Searchable index of all classes and functions
3. **General index**: Searchable index of all terms
4. **Quick navigation**: Jump to any module from sidebar

## Navigation Features

### Sidebar (Left Panel):

- **Hierarchical TOC**: Shows full documentation structure
- **Expandable sections**: Click to expand/collapse sections
- **Sticky navigation**: Follows as you scroll
- **Search box**: Quick search at top of sidebar
- **Current location highlighted**: Easy to see where you are

### Page Navigation:

- **Breadcrumbs**: Show path from home to current page
- **Previous/Next buttons**: At top and bottom of each page
- **Related links**: See Also sections link to related modules
- **Quick links**: Index pages provide shortcuts to common tasks

## Future Enhancements

Potential improvements for future updates:

1. **Version selector**: Add dropdown to switch between documentation versions
2. **Dark mode**: Add theme toggle for dark/light modes
3. **Interactive examples**: Add executable code examples
4. **Video tutorials**: Embed tutorial videos in relevant sections
5. **PDF generation**: Add PDF export of complete documentation
6. **Multi-language**: Add translations for international users

## Migration Notes

If you have bookmarks to old documentation:

- `developer/core.rst` → Now links to `api/base.rst`
- `developer/segmentation.rst` → Now links to `api/segmentation/index.rst`
- `developer/registration_images.rst` → Now links to `api/registration/index.rst`
- `developer/registration_models.rst` → Now links to `api/model_registration/index.rst`
- `developer/usd_generation.rst` → Now links to `api/usd/index.rst`
- `developer/utilities.rst` → Now links to `api/utilities/index.rst`

Developer guides remain but now focus on development patterns and link to the API reference for detailed documentation.
