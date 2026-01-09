# PhysioMotion4D Documentation Quick Reference

## Finding What You Need

### By Task

| I want to... | Go to... |
|-------------|----------|
| Get started quickly | [Quickstart](quickstart.rst) |
| See code examples | [Examples](examples.rst) |
| Use command-line tools | [CLI Scripts Overview](cli_scripts/overview.rst) |
| Find a specific class | [API Reference Index](api/index.rst) |
| Learn about architecture | [Architecture Guide](developer/architecture.rst) |
| Extend PhysioMotion4D | [Extending Guide](developer/extending.rst) |

### By Module

| Module | Purpose | API Docs | Developer Guide |
|--------|---------|----------|-----------------|
| **Base** | Core functionality | [api/base.rst](api/base.rst) | [developer/core.rst](developer/core.rst) |
| **Workflows** | Complete pipelines | [api/workflows.rst](api/workflows.rst) | [developer/workflows.rst](developer/workflows.rst) |
| **Segmentation** | AI segmentation | [api/segmentation/](api/segmentation/index.rst) | [developer/segmentation.rst](developer/segmentation.rst) |
| **Registration** | Image alignment | [api/registration/](api/registration/index.rst) | [developer/registration_images.rst](developer/registration_images.rst) |
| **Model Registration** | 3D model alignment | [api/model_registration/](api/model_registration/index.rst) | [developer/registration_models.rst](developer/registration_models.rst) |
| **USD Generation** | Omniverse export | [api/usd/](api/usd/index.rst) | [developer/usd_generation.rst](developer/usd_generation.rst) |
| **Utilities** | Helper functions | [api/utilities/](api/utilities/index.rst) | [developer/utilities.rst](developer/utilities.rst) |

### By Use Case

#### Cardiac Imaging
1. [Heart Gated CT to USD](cli_scripts/heart_gated_ct.rst)
2. [Heart Model to Patient Registration](cli_scripts/heart_model_to_patient.rst)
3. [Cardiac Segmentation](api/segmentation/index.rst)
4. [Time Series Registration](api/registration/time_series.rst)

#### Pulmonary Imaging
1. [Lung Gated CT to USD](cli_scripts/lung_gated_ct.rst)
2. [Lung Segmentation](api/segmentation/totalsegmentator.rst)
3. [4D CT Reconstruction](cli_scripts/4dct_reconstruction.rst)

#### Custom Development
1. [Extending PhysioMotion4D](developer/extending.rst)
2. [Creating Custom Workflows](developer/workflows.rst)
3. [Custom Segmentation](api/segmentation/base.rst)
4. [Custom Registration](api/registration/base.rst)

## Common Classes

### Core Classes
- `PhysioMotion4DBase` - [api/base.rst](api/base.rst)

### Workflows
- `WorkflowConvertHeartGatedCTToUSD` - [api/workflows.rst](api/workflows.rst)
- `WorkflowRegisterHeartModelToPatient` - [api/workflows.rst](api/workflows.rst)

### Segmentation
- `SegmentChestBase` - [api/segmentation/base.rst](api/segmentation/base.rst)
- `SegmentChestTotalSegmentator` - [api/segmentation/totalsegmentator.rst](api/segmentation/totalsegmentator.rst)
- `SegmentChestVISTA3D` - [api/segmentation/vista3d.rst](api/segmentation/vista3d.rst)
- `SegmentChestEnsemble` - [api/segmentation/ensemble.rst](api/segmentation/ensemble.rst)

### Registration
- `RegisterImagesBase` - [api/registration/base.rst](api/registration/base.rst)
- `RegisterImagesANTs` - [api/registration/ants.rst](api/registration/ants.rst)
- `RegisterImagesIcon` - [api/registration/icon.rst](api/registration/icon.rst)
- `RegisterTimeSeriesImages` - [api/registration/time_series.rst](api/registration/time_series.rst)

### Model Registration
- `RegisterModelsICP` - [api/model_registration/icp.rst](api/model_registration/icp.rst)
- `RegisterModelsICPITK` - [api/model_registration/icp_itk.rst](api/model_registration/icp_itk.rst)
- `RegisterModelsDistanceMaps` - [api/model_registration/distance_maps.rst](api/model_registration/distance_maps.rst)
- `RegisterModelsPCA` - [api/model_registration/pca.rst](api/model_registration/pca.rst)

### USD Generation
- `ConvertVTK4DToUSD` - [api/usd/vtk_conversion.rst](api/usd/vtk_conversion.rst)
- `ConvertVTK4DToUSDPolyMesh` - [api/usd/polymesh.rst](api/usd/polymesh.rst)
- `ConvertVTK4DToUSDTetMesh` - [api/usd/tetmesh.rst](api/usd/tetmesh.rst)

## Navigation Tips

### Using the Sidebar
1. **Expand/Collapse**: Click section headers to show/hide subsections
2. **Search**: Use the search box at the top of the sidebar
3. **Current Location**: Your current page is highlighted
4. **Sticky**: Sidebar follows as you scroll

### Using Page Navigation
1. **Breadcrumbs**: Top of page shows path from home
2. **Previous/Next**: Buttons at top and bottom of pages
3. **See Also**: Links to related modules at bottom of pages
4. **Quick Links**: Index pages have shortcuts to common tasks

### Using Search
1. **Full-text Search**: Search box finds content across all pages
2. **Module Index**: Click "Module Index" for alphabetical class list
3. **General Index**: Click "Index" for alphabetical term list
4. **Quick Jump**: Type class name to jump directly to documentation

## Quick Start Examples

### Segment a CT Scan
```python
from physiomotion4d import SegmentChestTotalSegmentator

segmentator = SegmentChestTotalSegmentator(fast=True, verbose=True)
labelmap = segmentator.segment("ct_scan.nrrd")
```
→ [Full Documentation](api/segmentation/totalsegmentator.rst)

### Register Two Images
```python
from physiomotion4d import RegisterImagesIcon

registrar = RegisterImagesIcon(device="cuda:0", verbose=True)
displacement = registrar.register("reference.nrrd", "moving.nrrd")
```
→ [Full Documentation](api/registration/icon.rst)

### Convert to USD
```python
from physiomotion4d import ConvertVTK4DToUSD

converter = ConvertVTK4DToUSD("output.usd", colormap="rainbow")
converter.convert(vtk_files=["phase0.vtk", "phase1.vtk"])
```
→ [Full Documentation](api/usd/vtk_conversion.rst)

### Complete Workflow
```python
from physiomotion4d import WorkflowConvertHeartGatedCTToUSD

workflow = WorkflowConvertHeartGatedCTToUSD(
    input_filenames=["phase0.nrrd", "phase1.nrrd"],
    contrast_enhanced=True,
    output_directory="./results",
    verbose=True
)
result = workflow.process()
```
→ [Full Documentation](api/workflows.rst)

## Getting Help

### Documentation Sections
- **Getting Started**: Installation, quickstart, examples
- **CLI Scripts**: Command-line tool usage
- **API Reference**: Complete class and function documentation
- **Developer Guides**: Development patterns and best practices
- **Contributing**: How to contribute to the project
- **Additional Resources**: FAQ, troubleshooting, references

### External Resources
- [GitHub Repository](https://github.com/aylward/PhysioMotion4d)
- [PyPI Package](https://pypi.org/project/physiomotion4d/)
- [Issue Tracker](https://github.com/aylward/PhysioMotion4d/issues)

### Support
- Check [FAQ](faq.rst) for common questions
- See [Troubleshooting](troubleshooting.rst) for common issues
- Search documentation using the search box
- File an issue on GitHub for bugs or feature requests
