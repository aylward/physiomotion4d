# PhysioMotion4D Scripts

Example scripts demonstrating how to use PhysioMotion4D for processing CT images into dynamic USD models for NVIDIA Omniverse visualization.

## Overview

The CLI provides a streamlined interface to the PhysioMotion4D pipeline for converting 4D cardiac CT images into animated anatomical models. The workflow handles segmentation, registration, mesh generation, and USD file creation automatically.

## Installation

Ensure PhysioMotion4D is installed in your environment:

```bash
# From PyPI
pip install physiomotion4d

# Or from source
pip install -e .
```

## Example Scripts

### process_heart_gated_ct_example.py

Processes cardiac gated CT images through the complete workflow from input images to painted USD files.

### example_colormap_usage.py

Demonstrates the colormap features of `ConvertVTK4DToUSD` for visualizing point data arrays in NVIDIA Omniverse. This script shows how to:
- Apply pre-defined colormaps (plasma, viridis, rainbow, heat, coolwarm, grayscale, random)
- Control intensity ranges for value-to-color mapping
- Visualize scalar data on 3D meshes with time-varying animations
- Merge USD files while preserving materials and animation data

**Usage:**
```bash
python scripts/example_colormap_usage.py
```

**Output:** Creates example USD files in `./output/` directory demonstrating different colormap options and the USD merge functionality.

## Running Tests

PhysioMotion4D includes comprehensive tests for USD merging, time preservation, and the complete processing pipeline. Tests are located in the `tests/` directory and use pytest.

**Run all tests:**
```bash
pytest tests/
```

**Run specific test modules:**
```bash
# Test USD merge functionality
pytest tests/test_usd_merge.py -v

# Test time-varying data preservation
pytest tests/test_usd_time_preservation.py -v

# Skip slow/data-dependent tests
pytest tests/ -m "not slow and not requires_data"
```

**Note:** Some tests require experimental data. See individual test files for details.

## Command-Line Interface

#### Basic Usage

**Single 4D NRRD file:**
```bash
# Using the installed CLI tool (recommended)
physiomotion4d-heart-gated-ct input_4d.nrrd --contrast

# Or using the example script
python scripts/process_heart_gated_ct_example.py input_4d.nrrd --contrast
```

**Multiple 3D files (time series):**
```bash
physiomotion4d-heart-gated-ct frame_00.nrrd frame_01.nrrd frame_02.nrrd --contrast
```

**Using wildcard patterns:**
```bash
physiomotion4d-heart-gated-ct frame_*.nrrd --contrast --project-name my_cardiac_study
```

#### Command-Line Arguments

**Required:**
- `input_files` - Path to 4D NRRD file OR list of 3D NRRD/NII files (one per cardiac phase)

**Optional:**
- `--output-dir DIR` - Output directory for intermediate files and results (default: `./results`)
- `--project-name NAME` - Project name for USD organization (default: `cardiac_model`)
- `--contrast` - Flag indicating the study is contrast-enhanced (recommended for cardiac CTA)
- `--reference-image PATH` - Path to custom reference image for registration (default: uses 70% cardiac phase)
- `--registration-iterations NUM` - Number of registration iterations (default: 1)

#### Examples

**Basic contrast-enhanced cardiac CT:**
```bash
physiomotion4d-heart-gated-ct my_cardiac_4d.nrrd \
    --contrast \
    --output-dir ./cardiac_results \
    --project-name patient_001
```

**Non-contrast study with custom reference:**
```bash
physiomotion4d-heart-gated-ct phase_*.nrrd \
    --reference-image reference_phase.mha \
    --output-dir ./results \
    --project-name non_contrast_cardiac
```

**Complete example with all options:**
```bash
physiomotion4d-heart-gated-ct \
    /data/cardiac/phase_00.nrrd \
    /data/cardiac/phase_01.nrrd \
    /data/cardiac/phase_02.nrrd \
    --contrast \
    --output-dir /output/patient_123 \
    --project-name Patient123Cardiac \
    --reference-image /data/cardiac/phase_01.nrrd \
    --registration-iterations 50
```

## Data Workflow

### Input Data

The CLI accepts two input formats:

1. **4D NRRD file** - Single file containing all temporal phases
   - Format: `.nrrd` or `.seq.nrrd`
   - Example: `TruncalValve_4DCT.seq.nrrd`

2. **Multiple 3D files** - Separate file for each cardiac phase
   - Formats: `.nrrd`, `.nii`, `.mha`
   - Example: `phase_00.nrrd`, `phase_10.nrrd`, `phase_20.nrrd`, etc.

### Processing Pipeline

The workflow executes the following steps automatically:

#### 1. Load Time Series Data
- Converts 4D data to 3D time series or loads multiple 3D files
- Selects reference image (70% cardiac phase by default, or user-specified)

#### 2. Segmentation
- Segments reference image using ensemble methods (TotalSegmentator + VISTA-3D)
- Identifies anatomical structures:
  - Heart chambers and myocardium
  - Major vessels (aorta, pulmonary arteries, vena cava)
  - Lungs and airways
  - Bones (ribs, spine, sternum)
  - Contrast-enhanced regions
  - Soft tissues

#### 3. Registration
- Registers each temporal frame to the reference image
- Uses Icon-based deformable registration with mass preservation
- Creates separate registrations for:
  - **Dynamic anatomy** - Heart, vessels, contrast (moving structures)
  - **Static anatomy** - Lungs, bones, tissues (relatively stationary)
  - **All anatomy** - Combined registration

#### 4. Contour Generation
- Generates VTK mesh contours from reference segmentation
- Creates smooth surface representations for each anatomical structure

#### 5. Transform Contours
- Applies registration transforms to contours for each time point
- Creates animated mesh sequences showing cardiac motion

#### 6. USD File Creation
- Generates painted USD files for Omniverse:
  - `{project_name}.dynamic_anatomy_painted.usd` - Animated moving anatomy
  - `{project_name}.static_anatomy_painted.usd` - Static background anatomy
  - `{project_name}.all_anatomy_painted.usd` - Complete anatomical model

### Output Files

#### Primary Outputs
- **Dynamic anatomy USD** - Heart and vessels with cardiac motion animation
- **Static anatomy USD** - Lungs, bones, and soft tissues (stationary reference)
- **All anatomy USD** - Combined model with all structures

#### Intermediate Files (in output directory)
- `slice_*.mha` - Individual 3D images for each time point
- `slice_*.labelmap.mha` - Segmentation masks
- `slice_*.reg_*.phi_FM.hdf` - Forward transformation files
- `slice_*.reg_*.phi_MF.hdf` - Backward transformation files
- `slice_max.reg_*.mha` - Maximum intensity projection images
- `*.vtk` - VTK mesh files for contours
- `*_4d.vtk` - Time series VTK files

## Use Cases

### Cardiac Motion Visualization
Process cardiac CTA studies to visualize heart motion through the cardiac cycle in Omniverse:
```bash
physiomotion4d-heart-gated-ct cardiac_cta_4d.nrrd --contrast --project-name heart_motion
```

### Multi-Phase Analysis
Process non-gated multi-phase CT for temporal analysis:
```bash
physiomotion4d-heart-gated-ct arterial.nrrd venous.nrrd delayed.nrrd \
    --project-name multiphase_study
```

### Research Data Processing
Batch process research datasets with custom reference frames:
```bash
for case in case_*; do
    physiomotion4d-heart-gated-ct ${case}/images/*.nrrd \
        --contrast \
        --reference-image ${case}/reference.mha \
        --output-dir ${case}/results \
        --project-name ${case}
done
```

## Tips and Best Practices

### Choosing a Reference Image
- **Default (70% phase)**: Mid-diastole, good for most cardiac studies
- **Custom reference**: Use the phase with best image quality or least motion artifact
- **Contrast studies**: Choose phase with optimal vessel enhancement

### Performance Optimization
- Use GPU-accelerated environment for faster registration
- Ensure sufficient RAM (16GB+ recommended for large datasets)
- Process on SSD storage for faster I/O operations

### Output Organization
- Use meaningful `--project-name` for easy identification
- Keep `--output-dir` separate for each study to avoid file conflicts
- Project name is used for all USD file naming

### Data Quality
- Ensure input images have consistent spacing and orientation
- Verify cardiac phases are properly ordered temporally
- Check that contrast enhancement is uniform across phases (for CTA studies)

## Troubleshooting

**Error: Input file not found**
- Verify file paths are correct and accessible
- Use absolute paths if relative paths fail
- Check file permissions

**Segmentation quality issues**
- Use `--contrast` flag for contrast-enhanced studies
- Ensure image quality is sufficient (no severe artifacts)
- Verify cardiac structures are visible in the field of view

**Registration failures**
- Check that reference image is appropriate (good quality, mid-cardiac phase)
- Ensure sufficient overlap between temporal phases
- Verify image spacing and dimensions are reasonable

**Memory errors**
- Reduce image resolution if needed
- Process fewer time points
- Close other applications to free RAM

## Additional Resources

- Cardiac experiments: `../experiments/Heart-GatedCT_To_USD/`
- Lung experiments: `../experiments/Lung-GatedCT_To_USD/`
- Main documentation: `../README.md`
- API documentation: `../src/physiomotion4d/`