# PhysioMotion4D Experiments

This directory contains research and design experiments that informed the development of the PhysioMotion4D library. **These are not examples of how to use the library**.

## Purpose

The code in this folder and its subfolders represents:

- **Research prototypes** - Early explorations of algorithms and approaches
- **Design experiments** - Testing different implementation strategies
- **Development iterations** - Code that evolved into the final library components
- **Proof-of-concept work** - Demonstrations that guided architectural decisions

## Structure

Each subdirectory represents a different experimental domain:

### Cardiac Imaging
- `Heart-GatedCT_To_USD/` - Complete cardiac 4D CT pipeline from images to animated USD models
- `Heart-VTKSeries_To_USD/` - Direct VTK time series to USD conversion for cardiac data
- `Heart-Model_To_Patient/` - Advanced model-to-patient registration with ICP, mask-based, and PCA methods

### Pulmonary Imaging
- `Lung-GatedCT_To_USD/` - Respiratory motion analysis using DirLab 4D-CT benchmark data
- `Lung-VesselsAirways/` - Specialized vessel and airway segmentation with deep learning models

### Advanced Visualization
- `Colormap-VTK_To_USD/` - Time-varying colormap rendering for scalar field visualization in Omniverse
- `DisplacementField_To_USD/` - Convert registration displacement fields to USD for PhysicsNeMo visualization

### Data Processing
- `Reconstruct4DCT/` - 4D CT reconstruction from sparse temporal samples using deformable registration

## Important Notes

⚠️ **These experiments are not production code** - They may contain:
- Hardcoded paths and parameters
- Incomplete error handling
- Experimental APIs that have since changed
- Code that doesn't follow the final library conventions

⚠️ **Do not use these as usage examples** - For proper library usage, refer to:
- The main library documentation
- Official examples in the `scripts/` directory
- Unit tests in the `tests/` directory
- The command-line tools provided by the package

## Development History

This experimental code was instrumental in:
1. Defining the final library architecture
2. Testing registration algorithms (Icon, SyN, LDDMM)
3. Evaluating segmentation approaches (TotalSegmentator, VISTA-3D)
4. Developing the USD export pipeline
5. Optimizing the complete 4D CT → USD workflow

The lessons learned from these experiments led to the creation of the unified `HeartGatedCTProcessor` class and other production components in the main library.