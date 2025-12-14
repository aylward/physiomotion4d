# PhysioMotion4D

**Generate anatomic models in Omniverse with physiological motion derived from 4D medical images.**

PhysioMotion4D is a comprehensive medical imaging package that converts 4D CT scans (particularly heart and lung gated CT data) into dynamic 3D models for visualization in NVIDIA Omniverse. The package provides state-of-the-art deep learning-based image processing, segmentation, registration, and USD file generation capabilities.

## 🚀 Key Features

- **Complete 4D Medical Imaging Pipeline**: End-to-end processing from 4D CT data to animated USD models
- **Multiple AI Segmentation Methods**: TotalSegmentator, VISTA-3D, and ensemble approaches
- **Deep Learning Registration**: GPU-accelerated image registration using Icon algorithm
- **NVIDIA Omniverse Integration**: Direct USD file export for medical visualization
- **Physiological Motion Analysis**: Capture and visualize cardiac and respiratory motion
- **Flexible Workflow Control**: Step-based processing with checkpoint management

## 📋 Supported Applications

- **Cardiac Imaging**: Heart-gated CT processing with cardiac motion analysis
- **Pulmonary Imaging**: Lung 4D-CT processing with respiratory motion tracking
- **Medical Education**: Interactive 3D anatomical models with physiological motion
- **Research Visualization**: Advanced medical imaging research in Omniverse
- **Clinical Planning**: Dynamic anatomical models for treatment planning

## 🛠️ Installation

### Prerequisites

- Python 3.10+ (Python 3.10, 3.11, or 3.12 recommended)
- NVIDIA GPU with CUDA 12.6+ (for AI models and registration)
- 16GB+ RAM (32GB+ recommended for large datasets)
- NVIDIA Omniverse (for USD visualization)

### Installation from PyPI

```bash
pip install physiomotion4d
```

For development with NVIDIA NIM cloud services:
```bash
pip install physiomotion4d[nim]
```

### Installation from Source

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PhysioMotion4D
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install uv package manager** (recommended):
   ```bash
   pip install uv
   ```

4. **Install PhysioMotion4D**:
   ```bash
   uv pip install -e .
   ```

   Or with pip:
   ```bash
   pip install -e .
   ```

### Verify Installation

```python
import physiomotion4d
from physiomotion4d import ProcessHeartGatedCT

print(f"PhysioMotion4D version: {physiomotion4d.__version__}")
```

## 🏗️ Package Architecture

### Core Components

- **Workflow Classes**: Complete end-to-end pipeline processors
  - `HeartGatedCTToUSDWorkflow`: Heart-gated CT to USD processing workflow
  - `HeartModelToPatientWorkflow`: Model-to-patient registration workflow
- **Segmentation Classes**: Multiple AI-based chest segmentation implementations
  - `SegmentChestTotalSegmentator`: TotalSegmentator-based segmentation
  - `SegmentChestVista3D`: VISTA-3D model-based segmentation
  - `SegmentChestVista3DNIM`: NVIDIA NIM version of VISTA-3D
  - `SegmentChestEnsemble`: Ensemble segmentation combining multiple methods
  - `SegmentChestBase`: Base class for custom segmentation methods
- **Registration Classes**: Multiple registration methods for different use cases
  - Image-to-Image Registration:
    - `RegisterImagesICON`: Deep learning-based registration using Icon algorithm
    - `RegisterImagesANTs`: Classical deformable registration using ANTs
    - `RegisterTimeSeriesImages`: Specialized time series registration for 4D CT
  - Model-to-Image/Model Registration:
    - `RegisterModelToImagePCA`: PCA-based statistical shape model registration
    - `RegisterModelToModelICP`: ICP-based surface registration
    - `RegisterModelToModelMasks`: Mask-based deformable model registration
  - `RegisterImagesBase`: Base class for custom registration methods
- **Base Classes**: Foundation classes providing common functionality
  - `PhysioMotion4DBase`: Base class providing standardized logging and debug settings
- **Utility Classes**: Tools for data manipulation and conversion
  - `TransformTools`: Comprehensive transform manipulation utilities
  - `USDTools`: USD file manipulation for Omniverse integration
  - `ImageTools`: Medical image processing utilities
  - `ContourTools`: Mesh extraction and contour manipulation

### Key Dependencies

- **Medical Imaging**: ITK, TubeTK, MONAI, nibabel, PyVista
- **AI/ML**: PyTorch (CUDA 12.6), transformers, MONAI
- **Registration**: icon-registration, unigradicon
- **Visualization**: USD-core, PyVista
- **Segmentation**: TotalSegmentator, VISTA-3D models

## 🎯 Quick Start

### Command-Line Interface

After installation, PhysioMotion4D provides a command-line tool for heart-gated CT processing:

```bash
# Process a single 4D cardiac CT file
physiomotion4d-heart-gated-ct cardiac_4d.nrrd --contrast --output-dir ./results

# Process multiple time frames
physiomotion4d-heart-gated-ct frame_*.nrrd --contrast --project-name patient_001

# With custom settings
physiomotion4d-heart-gated-ct cardiac.nrrd \
    --contrast \
    --reference-image ref.mha \
    --registration-iterations 50 \
    --output-dir ./output
```

See the [scripts](scripts/) directory for more CLI usage examples.

### Python API - Basic Heart-Gated CT Processing

```python
from physiomotion4d import HeartGatedCTToUSDWorkflow

# Initialize processor
processor = HeartGatedCTToUSDWorkflow(
    input_filenames=["path/to/cardiac_4d_ct.nrrd"],
    contrast_enhanced=True,
    output_directory="./results",
    project_name="cardiac_model",
    registration_method='icon'  # or 'ants'
)

# Run complete workflow
final_usd = processor.process()
```

### Python API - Model to Patient Registration

```python
from physiomotion4d import HeartModelToPatientWorkflow
import pyvista as pv
import itk

# Load generic model and patient data
model_mesh = pv.read("generic_heart_model.vtu")
patient_surfaces = [pv.read("lv.stl"), pv.read("rv.stl")]
reference_image = itk.imread("patient_ct.nii.gz")

# Initialize and run workflow
workflow = HeartModelToPatientWorkflow(
    moving_mesh=model_mesh,
    fixed_meshes=patient_surfaces,
    fixed_image=reference_image
)

# Run complete three-stage registration
registered_mesh = workflow.run_workflow()
```

### Custom Segmentation

```python
from physiomotion4d import SegmentChestVista3D
import itk

# Initialize VISTA-3D segmentation
segmenter = SegmentChestVista3D()

# Load and segment image
image = itk.imread("chest_ct.nrrd")
masks = segmenter.segment(image, contrast_enhanced_study=True)

# Extract individual anatomy masks
heart_mask, vessels_mask, lungs_mask, bones_mask, soft_tissue_mask, \
contrast_mask, all_mask, dynamic_mask = masks
```

### Image Registration

```python
from physiomotion4d import RegisterImagesICON, RegisterImagesANTs, RegisterTimeSeriesImages
import itk

# Option 1: Icon deep learning registration (GPU-accelerated)
registerer = RegisterImagesICON()
registerer.set_modality('ct')
registerer.set_fixed_image(itk.imread("reference_frame.mha"))
results = registerer.register(itk.imread("target_frame.mha"))

# Option 2: ANTs classical registration
registerer = RegisterImagesANTs()
registerer.set_fixed_image(itk.imread("reference_frame.mha"))
results = registerer.register(itk.imread("target_frame.mha"))

# Option 3: Time series registration for 4D CT
time_series_reg = RegisterTimeSeriesImages(
    reference_index=0,
    registration_method='icon'  # or 'ants'
)
transforms = time_series_reg.register_time_series(
    image_filenames=["time00.mha", "time01.mha", "time02.mha"]
)

# Get forward and inverse displacement fields
phi_FM = results["phi_FM"]  # Fixed to moving
phi_MF = results["phi_MF"]  # Moving to fixed
```

### Logging and Debug Control

PhysioMotion4D provides standardized logging through the `PhysioMotion4DBase` class, which is inherited by workflow and registration classes.

```python
import logging
from physiomotion4d import HeartModelToPatientWorkflow, PhysioMotion4DBase

# Control logging level globally for all classes
PhysioMotion4DBase.set_log_level(logging.DEBUG)

# Or filter to show logs from specific classes only
PhysioMotion4DBase.set_log_classes(["HeartModelToPatientWorkflow", "RegisterModelToImagePCA"])

# Show all classes again
PhysioMotion4DBase.set_log_all_classes()

# Query which classes are currently filtered
filtered = PhysioMotion4DBase.get_log_classes()
```

Classes that inherit from `PhysioMotion4DBase` provide:
- Standard log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Progress reporting for long-running operations
- Class-based log filtering
- Unified logging interface across the package

## 📊 Experiments and Examples

The `experiments/` directory contains comprehensive Jupyter notebooks demonstrating the complete PhysioMotion4D pipeline:

### 🫀 Heart-Gated CT (`experiments/Heart-GatedCT_To_USD/`)

Complete cardiac imaging workflow with step-by-step tutorials:

- **`0-download_and_convert_4d_to_3d.ipynb`**: Data preparation and 4D to 3D conversion
- **`1-register_images.ipynb`**: Image registration between cardiac phases
- **`2-generate_segmentation.ipynb`**: AI-based cardiac segmentation
- **`3-transform_dynamic_and_static_contours.ipynb`**: Dynamic contour transformation
- **`4-merge_dynamic_and_static_usd.ipynb`**: Final USD model creation and merging

**Sample Data**: The notebooks include instructions for downloading cardiac CT datasets from Slicer-Heart-CT.

### 🫁 Lung-Gated CT (`experiments/Lung-GatedCT_To_USD/`)

Respiratory motion analysis using DirLab 4D-CT benchmark data:

- **`0-register_dirlab_4dct.ipynb`**: Registration of respiratory phases
- **`1-make_dirlab_models.ipynb`**: 3D model generation from lung segmentation
- **`2-paint_dirlab_models.ipynb`**: USD material and visualization enhancement

**Sample Data**: Uses the standard DirLab 4D-CT benchmark datasets. The notebooks include automatic download scripts for:
- Case 1-10 respiratory 4D-CT data
- Landmark point validation data
- Pre-processed segmentation masks

### 🎨 Colormap Visualization (`experiments/Colormap-VTK_To_USD/`)

Time-varying colormap rendering for scalar data visualization in Omniverse:

- **`colormap_vtk_to_usd.ipynb`**: Convert VTK meshes with scalar data to USD with colormaps
- Demonstrates plasma, viridis, rainbow, heat, coolwarm, grayscale, and custom colormaps

### 🫀 Heart VTK Series (`experiments/Heart-VTKSeries_To_USD/`)

Direct VTK time series to USD conversion for cardiac data:

- **`0-download_and_convert_4d_to_3d.ipynb`**: Data preparation
- **`1-heart_vtkseries_to_usd.ipynb`**: VTK series to USD conversion

### 🧠 Heart Model to Patient (`experiments/Heart-Model_To_Patient/`)

Advanced registration between generic anatomical models and patient-specific data:

- **`heart_model_to_patient.ipynb`**: Complete model-to-patient registration workflow
- **`heart_model_to_model_registration_pca.ipynb`**: PCA-based statistical shape model registration

Uses the `HeartModelToPatientWorkflow` class for three-stage registration:
1. ICP-based rough alignment
2. Mask-to-mask deformable registration
3. Optional mask-to-image refinement

### 🔬 4D CT Reconstruction (`experiments/Reconstruct4DCT/`)

Reconstruct 4D CT from sparse time samples using deformable registration:

- **`reconstruct_4d_ct.ipynb`**: Temporal interpolation and 4D reconstruction
- **`reconstruct_4d_ct_class.ipynb`**: Class-based reconstruction approach

### 🫁 Vessel and Airway Segmentation (`experiments/Lung-VesselsAirways/`)

Specialized deep learning for pulmonary vessel and airway segmentation:

- **`0-GenData.ipynb`**: Training data generation for vessel segmentation models
- Includes trained ResNet18 models for vessel segmentation
- Supporting branch structure test data

### 🌊 Displacement Field Visualization (`experiments/DisplacementField_To_USD/`)

Convert image registration displacement fields to USD for advanced visualization:

- **`displacement_field_to_usd.ipynb`**: Convert displacement fields to time-varying USD
- **`displacement_field_converter.py`**: DisplacementFieldToUSD class implementation
- Integration with PhysicsNeMo for flow visualization in Omniverse
- Supports streamlines, vector glyphs, and particle advection

## 📥 Sample Data Sources

### Cardiac Data
- **Slicer-Heart-CT**: Cardiac gating examples from 3D Slicer
- **Duke CardiacCT**: Research cardiac datasets (requires institutional access)

### Lung Data
- **DirLab 4D-CT**: Public benchmark for respiratory motion
  - Automatic download via: `DirLab4DCT.download_case(case_number)`
  - 10 cases with respiratory motion and landmark validation

### Download Scripts

Each experiment directory contains data download utilities:

```python
# Download DirLab case
from physiomotion4d import DirLab4DCT
downloader = DirLab4DCT()
downloader.download_case(1)  # Downloads Case 1 to ./data/

# Download Slicer-Heart-CT cardiac data
# See experiments/Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb
```

## 🔧 Development

### Code Quality Tools

- **Formatting**: `black` (line length: 100)
- **Import sorting**: `isort` (profile: black)
- **Linting**: `flake8` (max line length: 100)
- **Static analysis**: `pylint`
- **Pre-commit hooks**: `pre-commit`

### Running Quality Checks

```bash
# Format code
black src/
isort src/

# Check code quality
flake8 src/
pylint src/

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing

PhysioMotion4D includes comprehensive tests covering the complete pipeline from data download to USD generation.

```bash
# Run all tests
pytest tests/

# Run fast tests only (recommended for development)
pytest tests/ -m "not slow and not requires_data" -v

# Run specific test categories
pytest tests/test_usd_merge.py -v                           # USD merge functionality
pytest tests/test_usd_time_preservation.py -v               # Time-varying data preservation
pytest tests/test_register_images_ants.py -v                # ANTs registration
pytest tests/test_register_images_icon.py -v                # Icon registration
pytest tests/test_register_time_series_images.py -v         # Time series registration
pytest tests/test_segment_chest_total_segmentator.py -v     # TotalSegmentator
pytest tests/test_segment_chest_vista_3d.py -v              # VISTA-3D segmentation
pytest tests/test_contour_tools.py -v                       # Mesh and contour tools
pytest tests/test_image_tools.py -v                         # Image processing utilities
pytest tests/test_transform_tools.py -v                     # Transform operations

# Skip GPU-dependent tests (segmentation and registration)
pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
              --ignore=tests/test_segment_chest_vista_3d.py \
              --ignore=tests/test_register_images_icon.py

# Run with coverage report
pytest tests/ --cov=src/physiomotion4d --cov-report=html
```

**Test Categories:**
- **Data Pipeline**: Download, conversion, and preprocessing
- **Segmentation**: TotalSegmentator and VISTA-3D (GPU required)
- **Registration**: ANTs, Icon, and time series methods (slow, ~5-10 min)
- **Geometry & Visualization**: Contour tools, transform tools, VTK to USD
- **USD Utilities**: Merging, time preservation, material handling

Tests automatically run on pull requests via GitHub Actions. See `tests/README.md` for detailed testing guide.

## 📖 Documentation

- **API Documentation**: Comprehensive docstrings for all classes and methods
- **Tutorial Notebooks**: Step-by-step examples in `experiments/`
- **CLAUDE.MD**: Development guidelines and architecture overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run code quality checks (`black src/ && flake8 src/`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NVIDIA Omniverse** team for USD format and visualization platform
- **MONAI** community for medical imaging AI tools
- **DirLab** for providing the 4D-CT benchmark datasets
- **TotalSegmentator** and **VISTA-3D** teams for segmentation models
- **Icon Registration** team for deep learning registration methods

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Refer to docstrings and tutorial notebooks
- **Examples**: Explore comprehensive examples in `experiments/` directory

---

**Get started with the tutorial notebooks in `experiments/` to see PhysioMotion4D in action! 🚀**