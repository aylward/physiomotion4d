# PhysioMotion4D Experiments

This directory contains research and design experiments that informed the development
of the PhysioMotion4D library.

**These are not examples of how to use the library**.

## ⭐ For Production Workflows: See the `scripts/` Directory

**The `scripts/` directory is the definitive source for:**
- Production-ready workflows and usage patterns
- Proper class instantiation and method usage
- Library capabilities and best practices
- Command-line tools and parameter specifications

The experiments here serve as **conceptual references** showing the research process
and design explorations that can inform adaptation to new digital twin models and tasks.

## Purpose

The code in this folder and its subfolders represents:

- **Research prototypes** - Early explorations of algorithms and approaches
- **Design experiments** - Testing different implementation strategies
- **Development iterations** - Code that evolved into the final library components
- **Proof-of-concept work** - Demonstrations that guided architectural decisions

## Experiment Highlights

These experiments demonstrate key digital twin workflows that can be adapted to new
anatomical regions, imaging modalities, and physiological motion tasks.

> **Note:** For production implementations of these workflows, consult the `scripts/`
> directory for proper class usage and parameter specifications.

### `Reconstruct4DCT` - High-Resolution 4D Reconstruction

**What it demonstrates:** Generates high-resolution 4D sequences by registering a
high-resolution 3D scan with a low-resolution 4D sequence and upsampling. Critical for
obtaining finer details in 4D data used to train AI motion simulations.

**Key technologies:**
- GPU-accelerated ICON registration (inverse consistent)
- Traditional ANTs registration library
- CT-appropriate tissue mass-preserving metric (optional)

**Adaptation potential:** This approach can be fine-tuned for:
- Different anatomical regions (abdomen, pelvis, extremities)
- Alternative imaging modalities (MRI, ultrasound time series)
- Custom registration metrics for specific tissue types
- Multi-phase imaging protocols beyond respiratory/cardiac gating

### `Lung-GatedCT_To_USD` and `Heart-GatedCT_To_USD` - Complete 4D CT Pipelines

**What they demonstrate:** End-to-end pipelines from 4D CT reconstruction through
segmentation, meshing, USD conversion, and tissue property assignment for Omniverse
visualization and manipulation.

**Pipeline stages:**
1. 4D CT reconstruction (using methods from `Reconstruct4DCT`)
2. AI segmentation (TotalSegmentator, Vista3D/Clara Segment Open Model as NIM or local)
3. Per-organ mesh generation
4. OpenUSD conversion with animation
5. Tissue property mapping (subsurface scatter, color, etc.)

**Adaptation potential:** Highly modular design enables:
- New anatomical regions (brain, spine, joints, abdominal organs)
- Custom segmentation models and label schemas
- Alternative meshing strategies (higher/lower resolution, topology optimization)
- Domain-specific tissue property mappings for different visualization needs
- Integration with Isaac for Healthcare "Bring-Your-Own-Data" workflows
- Multi-organ system digital twins for surgical planning

### `Heart-VTKSeries_To_USD` - VTK to USD Conversion

**What it demonstrates:** Direct conversion from VTK time series representations to
OpenUSD, preserving colormaps and visualization properties where possible.

**Use cases:**
- Leveraging existing VTK-based medical imaging applications in Omniverse
- Accessing Omniverse connectors (PyAnsys, ROS/ROS2, Isaac for Healthcare)
- Enhanced AR/VR visualization beyond traditional medical imaging viewers
- Integration with physics simulation engines
- Surgical robot training environments

**Adaptation potential:**
- Custom VTK property to USD material mappings
- Support for additional VTK data structures (polydata, unstructured grids)
- Pipeline integration for real-time or near-real-time visualization
- Domain-specific colormap standards (ultrasound Doppler, functional imaging)

**Note:** VTK-to-USD mapping is complex and may require refinement for specific use cases.
This experiment provides a foundation to build upon.

### `Heart-Model_To_Patient` - Foundation for Patient-Specific Digital Twins

**What it demonstrates:** Fitting a simulation-ready or AI-ready anatomical model to
patient-specific imaging data, creating a consistent architecture across subjects.

**Key benefit:** Using consistent model topology across patients dramatically improves
cross-patient robustness of AI models for physiological motion, enabling a single model
to handle diverse cases.

**Technologies:**
- ICP (Iterative Closest Point) registration
- Mask-based deformable registration
- PCA (Principal Component Analysis) shape modeling
- Computationally intensive (>1 hour on typical PC)

**Adaptation potential:** This foundational approach can be extended to:
- Any anatomical structure with consistent topology (liver, brain, vessels)
- Multi-organ system models with coupled motion patterns
- Patient-specific biomechanics simulation preparation
- AI training datasets with normalized anatomical coordinates
- Population-based statistical shape modeling
- Real-time deformable registration for interventional guidance
- Integration with simulation engines (PyAnsys, FEniCS, etc.)

**Output:** Patient-specific USD models ready for physics simulation or AI training.  

## Adapting to New Digital Twin Models and Tasks

These experiments illustrate architectural patterns and workflows that can be adapted
to new anatomical regions, physiological processes, and digital twin applications.

### General Adaptation Strategy

1. **Study the experiment conceptually** - Understand the workflow stages and data transformations
2. **Consult the `scripts/` directory** - Identify production classes and methods that implement similar functionality
3. **Identify customization points:**
   - Registration parameters (metrics, transforms, optimization)
   - Segmentation models (custom training, fine-tuning, label mappings)
   - Meshing parameters (resolution, smoothing, topology)
   - Tissue property mappings (visualization, physics parameters)
4. **Leverage modular design** - Swap components (registration algorithms, segmentation models) as needed
5. **Validate on your data** - Test with representative samples before production deployment

### Key Takeaway

These experiments are **starting points for exploration**, not copy-paste solutions.
The **`scripts/` directory contains the production-quality implementations** you should
use and extend for real-world digital twin projects.

## Automated Testing

A comprehensive test suite is available to validate all experiment notebooks:

```bash
# Run all experiment tests (EXTREMELY SLOW - may take hours)
# NOTE: Requires --run-experiments flag!
pytest tests/test_experiments.py -v --run-experiments

# Run a specific experiment subdirectory
pytest tests/test_experiments.py::test_experiment_heart_gated_ct_to_usd -v -s --run-experiments

# List all notebooks that would be run (without executing)
pytest tests/test_experiments.py::test_list_notebooks_in_subdir -v -s --run-experiments

# Validate experiment directory structure
pytest tests/test_experiments.py::test_experiment_structure -v --run-experiments
```

🔒 **IMPORTANT:** Experiment tests require the `--run-experiments` flag. Without this flag, they are automatically skipped, even if you run `pytest tests/` or target the test file directly.

### Test Features

- **One test per subdirectory** - Each experiment subdirectory gets its own test function
- **Alphanumeric ordering** - Notebooks execute in alphanumeric order (e.g., `0-`, `1-`, `2-`)
- **Long timeouts** - Each notebook has up to 1 hour execution time, tests have multi-hour timeouts
- **Detailed output** - Progress reporting, execution summaries, and failure diagnostics
- **Opt-in only** - Requires `--run-experiments` flag; automatically skipped otherwise
- **Protected from CI/CD** - NEVER runs in automated workflows

### Requirements

These tests require:
- All dependencies installed (see `pyproject.toml`)
- GPU/CUDA support for most experiments
- Large amounts of disk space and memory
- External data downloads (see individual experiment notebooks)

### Important Notes

⚠️ **These tests are extremely long-running** - Plan for multiple hours of execution time

⚠️ **Not part of CI/CD** - These tests are excluded from all automated workflows

⚠️ **Resource intensive** - Requires GPU, significant memory, and disk space

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
- Dataset-specific assumptions and configurations

⚠️ **Do not use these as usage examples** - These are research artifacts, not tutorials.

### ✅ For Production Use, Consult:

1. **`scripts/` directory** ⭐ **PRIMARY RESOURCE**
   - Production-ready workflow implementations
   - Proper class usage patterns and parameter specifications
   - Complete error handling and validation
   - Command-line interface examples
   - Tested on diverse datasets

2. **Main library documentation**
   - API references and class documentation
   - Architecture explanations
   - Performance considerations

3. **`tests/` directory**
   - Unit tests demonstrating correct API usage
   - Edge case handling
   - Expected input/output formats

4. **Command-line tools**
   - Package-provided CLI utilities
   - Production parameter defaults
   - Integration with standard medical imaging formats

### Using Experiments as Inspiration

**DO:** Study these experiments to understand:
- Overall workflow architecture and data flow
- Problem decomposition strategies
- Technology choices and trade-offs
- Potential adaptation approaches for your domain

**DON'T:** Copy experiment code directly into production:
- Experiments may use outdated APIs
- Error handling may be minimal or missing
- Paths and parameters are often hardcoded for specific datasets
- Performance may not be optimized

**INSTEAD:** Use the production implementations in `scripts/` and extend them for your needs.

## Development History

This experimental code was instrumental in:
1. Defining the final library architecture
2. Testing registration algorithms (ICON, SyN, LDDMM)
3. Evaluating segmentation approaches (TotalSegmentator, VISTA-3D)
4. Developing the USD export pipeline
5. Optimizing the complete 4D CT → USD workflow
6. Identifying modular extension points for new anatomical regions and tasks

The lessons learned from these experiments led to the creation of production components
like the unified `HeartGatedCTProcessor` class and other reusable classes in the main
library.

### Evolution from Experiment to Production

The typical evolution path was:
1. **Experiment** - Proof of concept with specific dataset (this directory)
2. **Refinement** - Generalization and parameterization
3. **Testing** - Validation across diverse datasets
4. **Production** - Documented, tested implementation in `scripts/` directory ⭐

When exploring new digital twin applications, you can follow a similar path:
- Start by understanding relevant experiments here as conceptual references
- Examine production implementations in `scripts/` for proper library usage
- Extend and adapt production code with your domain-specific requirements
- Contribute back improvements and new capabilities to the community

**Remember:** The `scripts/` directory is your source of truth for how to properly use
PhysioMotion4D classes, workflows, and capabilities in production environments.