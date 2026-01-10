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

### convert_heart_gated_ct_to_usd.py

Processes cardiac gated CT images through the complete workflow from input images to painted USD files.

### register_heart_model_to_patient.py

Registers a generic heart model to patient-specific imaging data and surface models using multi-stage registration (ICP, PCA, mask-based, and optional image-based refinement).

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

### Heart Model to Patient Registration

#### Basic Usage

Register a generic heart model to patient-specific data:

```bash
python scripts/register_heart_model_to_patient.py \
  --template-model heart_model.vtu \
  --template-labelmap heart_labelmap.nii.gz \
  --patient-models lv.vtp rv.vtp myo.vtp \
  --patient-image patient_ct.nii.gz \
  --output-dir ./results
```

#### With PCA Shape Fitting

```bash
python scripts/register_heart_model_to_patient.py \
  --template-model heart_model.vtu \
  --template-labelmap heart_labelmap.nii.gz \
  --patient-models lv.vtp rv.vtp myo.vtp \
  --patient-image patient_ct.nii.gz \
  --pca-json pca_model.json \
  --pca-number-of-modes 10 \
  --output-dir ./results
```

#### With ICON Refinement

```bash
python scripts/register_heart_model_to_patient.py \
  --template-model heart_model.vtu \
  --template-labelmap heart_labelmap.nii.gz \
  --patient-models lv.vtp rv.vtp \
  --patient-image patient_ct.nii.gz \
  --use-icon-refinement \
  --output-dir ./results
```

#### Command-Line Arguments

**Required:**
- `--template-model PATH` - Path to template/generic heart model (.vtu, .vtk, .stl)
- `--template-labelmap PATH` - Path to template labelmap image (.nii.gz, .nrrd, .mha)
- `--patient-models PATH [PATH ...]` - Paths to patient-specific surface models (e.g., lv.vtp rv.vtp myo.vtp)
- `--patient-image PATH` - Path to patient CT/MRI image (.nii.gz, .nrrd, .mha)
- `--output-dir DIR` - Output directory for results

**Template Labelmap Configuration:**
- `--template-labelmap-muscle-ids ID [ID ...]` - Label IDs for heart muscle (default: 1)
- `--template-labelmap-chamber-ids ID [ID ...]` - Label IDs for heart chambers (default: 2)
- `--template-labelmap-background-ids ID [ID ...]` - Label IDs for background (default: 0)

**PCA Registration Options:**
- `--pca-json PATH` - Path to PCA JSON file for shape-based registration (optional)
- `--pca-group-key KEY` - PCA group key in JSON file (default: All)
- `--pca-number-of-modes NUM` - Number of PCA modes to use (default: 0, uses all if PCA enabled)

**Registration Configuration:**
- `--use-mask-to-mask` / `--no-mask-to-mask` - Enable/disable mask-to-mask deformable registration (default: enabled)
- `--use-mask-to-image` / `--no-mask-to-image` - Enable/disable mask-to-image refinement (default: enabled)
- `--use-icon-refinement` - Enable ICON registration refinement (default: disabled)

**Output Options:**
- `--output-prefix PREFIX` - Prefix for output files (default: registered)

#### Output Files

The registration workflow produces the following output files in the specified output directory:

**Final Results:**
- `{prefix}_model.vtu` - Final registered volumetric model
- `{prefix}_model_surface.vtp` - Final registered surface model
- `{prefix}_labelmap.nii.gz` - Final registered labelmap

**Intermediate Results (if generated):**
- `{prefix}_icp_surface.vtp` - Result after ICP alignment
- `{prefix}_pca_surface.vtp` - Result after PCA shape fitting (if PCA used)
- `{prefix}_m2m_surface.vtp` - Result after mask-to-mask deformable registration

### Heart Gated CT to USD Conversion

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
- `slice_*.reg_*.inverse_transform.hdf` - Backward transformation files
- `slice_*.reg_*.forward_transform.hdf` - Forward transformation files
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