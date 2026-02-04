# PhysioMotion4D Data Directory

This directory contains sample datasets used for experiments, testing, and development of the PhysioMotion4D library. Each subdirectory contains a specific medical imaging dataset.

## 📁 Directory Structure

```
data/
├── Slicer-Heart-CT/          # 4D cardiac CT with gated cardiac phases (AUTO-DOWNLOAD)
├── DirLab-4DCT/              # 4D lung CT benchmark dataset (MANUAL)
├── KCL-Heart-Model/          # Statistical shape model of the heart (MANUAL)
├── CHOP-Valve4D/             # 4D valve models (MANUAL)
```

## 📥 Data Download Methods

### Automatic Download (Only Slicer-Heart-CT)
Only the **Slicer-Heart-CT** dataset can be automatically downloaded by running the appropriate notebook.

### Manual Download (All Others)
The following datasets must be **manually downloaded and preprocessed** by the user:
- **DirLab-4DCT**: Respiratory motion benchmark data
- **KCL-Heart-Model**: Statistical cardiac shape models
- **CHOP-Valve4D**: Time-varying valve reconstructions

See individual dataset sections below for download instructions and preprocessing requirements.

---

## 🫀 Slicer-Heart-CT ✅ AUTO-DOWNLOAD

### Description
4D cardiac CT dataset with temporal gating showing complete cardiac cycle motion. Pediatric cardiac CT with truncal valve visualization.

### Specifications
- **Format**: `.seq.nrrd` (4D NRRD sequence file)
- **Phases**: 21 temporal cardiac phases
- **Size**: ~1.2 GB
- **Content**: Contrast-enhanced cardiac CT
- **Anatomy**: Heart, great vessels, thoracic structures

### Acknowledgement
Data provided by Jolley Lab at CHOP (Children's Hospital of Philadelphia):
- https://www.linkedin.com/company/jolleylab
- https://github.com/Slicer-Heart-CT/Slicer-Heart-CT

### Downloading the Data

**Automatic download** (recommended):
```python
# The data will be automatically downloaded when running:
# experiments/Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb
```

**Manual download** (alternative):
```bash
# Direct download link:
wget https://github.com/Slicer-Heart-CT/Slicer-Heart-CT/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd -P data/Slicer-Heart-CT/
```

### Usage
- Primary dataset for `Heart-GatedCT_To_USD` experiments
- Used in test suite (`tests/test_download_heart_data.py`)
- Example data for cardiac motion visualization in Omniverse

---

## 🫁 DirLab-4DCT ⚠️ MANUAL DOWNLOAD

### Description
Benchmark dataset for 4D CT respiratory motion analysis. Contains 10 cases of lung CT scans at different respiratory phases with annotated landmark points for registration validation.

### Specifications
- **Format**: `.mhd/.raw` (MetaImage format)
- **Cases**: 10 patient cases (Case 1-10)
- **Phases**: 10 respiratory phases per case (T00-T90)
- **Content**: Non-contrast lung CT
- **Anatomy**: Lungs, airways, thoracic structures
- **Landmarks**: 300+ annotated points per case for validation

### Acknowledgement
Data provided by the DIR-Lab at MD Anderson Cancer Center:
- **Project**: COPDGene 4D-CT dataset
- **Publication**: Castillo et al., "A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGene study archive"
- **Website**: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html

### Downloading the Data

⚠️ **MANUAL DOWNLOAD REQUIRED**

Users must manually download and preprocess this dataset. Follow these steps:

**Step 1: Manual Download**
```python
# Using provided utilities in experiments notebooks
# See: experiments/Lung-GatedCT_To_USD/0-register_dirlab_4dct.ipynb
# The notebook includes download utilities but requires manual execution
```

**Step 2: User Preprocessing**
Users are responsible for:
- Downloading data from DIR-Lab website
- Extracting and organizing files in the proper directory structure
- Running preprocessing notebooks if needed

### Directory Structure
```
DirLab-4DCT/
├── Case1Pack/
│   ├── Images/              # T00-T50 phase images
│   ├── ExtremePhases/       # T00 and T50 (max inhale/exhale)
│   └── Sampled4D/           # Sampled time points
├── Case1Pack_T00.mhd        # Extracted phase files
├── Case1Pack_T10.mhd
...
├── Case10Pack/
└── dirlab_password.txt      # Password for access (if needed)
```

### Usage
- Primary dataset for `Lung-GatedCT_To_USD` experiments
- Registration algorithm validation
- Respiratory motion analysis
- Benchmark for deformable registration accuracy

---

## 🧠 KCL-Heart-Model ⚠️ MANUAL DOWNLOAD

### Description
Statistical shape model (SSM) of the human heart derived from cardiac imaging data. Includes principal component analysis (PCA) modes of shape variation.

### Specifications
- **Format**: `.vtk`, `.vtp` (VTK PolyData formats)
- **Content**:
  - Average heart surface and mesh
  - Individual heart models
  - PCA eigenvectors
  - Mode standard deviations
  - Variance explained by each mode
- **Components**: Full heart mesh with chambers and vessels

### Files
- `average_surface.vtp` - Mean heart surface (PolyData)
- `average_mesh.vtk` - Mean heart volume mesh (UnstructuredGrid)
- `Full_Heart_Mesh_1.vtk` - Example individual heart
- `Eigenvectors.csv` - PCA eigenvectors (shape modes)
- `Mode_standard_deviations.csv` - Standard deviation for each mode
- `Normalized_explained_variance.csv` - Variance explained by each PCA mode
- `data_description.pdf` - Detailed model documentation
- `publication.pdf` - Associated research publication
- `citation.txt` - Citation information

### Acknowledgement
Data from King's College London (KCL):
- **Repository**: Cardiac imaging research group
- **License**: Check `citation.txt` for proper attribution

### Downloading the Data

⚠️ **MANUAL DOWNLOAD REQUIRED**

Users must manually obtain and place this data:
1. Obtain data from published research repositories or contact authors
2. Place files in `data/KCL-Heart-Model/` directory
3. Required files: `average_surface.vtp`, `average_mesh.vtk`

Check the included PDFs (if available) for source information and proper citation.

### Usage
- **Statistical shape model creation** (`experiments/Heart-Create_Statistical_Model/`) ⭐ **Primary use case**
- **Model-to-patient registration** (`experiments/Heart-Statistical_Model_To_Patient/`)
- VTK to USD conversion experiments (`experiments/Convert_VTK_To_USD/`)
- Shape-based cardiac analysis
- Atlas-based segmentation initialization
- Population-based statistical analysis

---

## 🫀 CHOP-Valve4D ⚠️ MANUAL DOWNLOAD

### Description
Time-varying 4D valve reconstruction models showing valve motion over the cardiac cycle. These datasets represent dynamic valve geometries reconstructed from medical imaging data.

### Specifications
- **Format**: `.vtk` (VTK PolyData files)
- **Content**: Time series of valve surface meshes
- **Valves**: Alterra, TPV25, and other valve types
- **Phases**: Multiple time points per cardiac cycle (200+ frames)
- **Resolution**: High-resolution surface meshes with anatomical features

### Directory Structure
```
CHOP-Valve4D/
├── Alterra/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ... (232 frames)
├── TPV25/
│   ├── frame_0000.vtk
│   ├── frame_0001.vtk
│   └── ... (265 frames)
```

### Acknowledgement
Data provided by Jolley Lab at CHOP (Children's Hospital of Philadelphia):
- https://www.linkedin.com/company/jolleylab

### Downloading the Data

⚠️ **MANUAL DOWNLOAD REQUIRED**

**Availability**: This dataset will soon be publicly available for download from the **FEBio website** under the **Creative Commons Attribution (CC-BY) license**.

- **Source**: https://febio.org/ (coming soon)
- **License**: CC-BY (Creative Commons Attribution)
- **Citation**: Please cite the Jolley Lab and FEBio when using this data

**Setup Instructions**:
1. Download valve reconstruction data from FEBio website when available
2. Place files in `data/CHOP-Valve4D/` with proper subdirectory structure
3. Ensure files are named sequentially for time-series processing (e.g., `frame_0000.vtk`, `frame_0001.vtk`, ...)
4. Organize by valve type in subdirectories (e.g., `Alterra/`, `TPV25/`)

### Usage
- Time-series VTK to USD conversion (`experiments/convert_vtk_to_usd_lib/valve4d_time_series.ipynb`)
- 4D valve motion visualization in NVIDIA Omniverse
- Temporal cardiac mechanics analysis
- Valve dynamics studies and surgical planning

### Related Resources
- **FEBio**: Finite Element Biomechanics software suite (https://febio.org/)
- **Jolley Lab**: Cardiac imaging and computational modeling research

---

## 📝 Data Usage Guidelines

### For Testing
- Tests automatically use cached data when available
- Download occurs only if data is missing
- Tests use subsets (e.g., first 2 time points) for speed

### For Experiments
- Full datasets used for complete analysis
- Results saved to respective `experiments/*/results/` directories
- Original data remains unmodified

### For Development
- Use small subsets for rapid iteration
- Full datasets for validation and benchmarking
- Cache intermediate results to avoid reprocessing

---

## 🔒 Data Access and Licensing

- **Slicer-Heart-CT** ✅: Public release from GitHub (auto-download available)
- **DirLab-4DCT** ⚠️: Public benchmark dataset (manual download required, may require registration)
- **KCL-Heart-Model** ⚠️: Requires manual download from research repositories
- **CHOP-Valve4D** ⚠️: Soon available from FEBio website under CC-BY license (manual download)

⚠️ **Important**: Always cite the original data sources in publications and respect any usage restrictions.

### Summary of Download Methods
| Dataset         | Auto-Download | Manual Required | License         | Source              | Used in Tests            |
| --------------- | ------------- | --------------- | --------------- | ------------------- | ------------------------ |
| Slicer-Heart-CT | ✅ Yes         | No              | Public          | GitHub              | Yes                      |
| DirLab-4DCT     | ❌ No          | Yes             | Public/Academic | DIR-Lab             | No                       |
| KCL-Heart-Model | ❌ No          | Yes             | Check citation  | Zenodo/KCL          | Yes (skipped if missing) |
| CHOP-Valve4D    | ❌ No          | Yes             | CC-BY           | FEBio (coming soon) | No                       |

---

## 📚 References

### Slicer-Heart-CT
- Jolley Lab: https://www.linkedin.com/company/jolleylab
- GitHub: https://github.com/Slicer-Heart-CT/Slicer-Heart-CT

### DirLab-4DCT
- Castillo et al., "A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGene study archive"
- DIR-Lab: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/

### KCL-Heart-Model
- Rodero et al. (2021), "Linking statistical shape models and simulated function in the healthy adult human heart", *PLOS Computational Biology*
- DOI: [10.1371/journal.pcbi.1008851](https://doi.org/10.1371/journal.pcbi.1008851)
- Zenodo: https://zenodo.org/records/4590294

### CHOP-Valve4D
- Jolley Lab (CHOP): https://www.linkedin.com/company/jolleylab
- FEBio Project: https://febio.org/ (dataset coming soon)
- License: Creative Commons Attribution (CC-BY)
- Citation: Please acknowledge Jolley Lab at CHOP and the FEBio Project

---

## 💡 Tips

1. **Storage**: Ensure adequate disk space (~10-20GB for all datasets)
2. **Download Time**: Initial downloads can be slow; be patient
3. **Organization**: Keep data organized; don't modify original files
4. **Backups**: Consider backing up processed results separately
5. **Documentation**: Update this README when adding new datasets
