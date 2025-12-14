# PhysioMotion4D Data Directory

This directory contains sample datasets used for experiments, testing, and development of the PhysioMotion4D library. Each subdirectory contains a specific medical imaging dataset.

## 📁 Directory Structure

```
data/
├── Slicer-Heart-CT/          # 4D cardiac CT with gated cardiac phases
├── DirLab-4DCT/              # 4D lung CT benchmark dataset (respiratory motion)
├── KCL-Heart-Model/          # Statistical shape model of the heart
```

---

## 🫀 Slicer-Heart-CT

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

**Manual download**:
```bash
# Direct download link:
wget https://github.com/Slicer-Heart-CT/Slicer-Heart-CT/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd -P data/Slicer-Heart-CT/
```

### Usage
- Primary dataset for `Heart-GatedCT_To_USD` experiments
- Used in test suite (`tests/test_download_heart_data.py`)
- Example data for cardiac motion visualization in Omniverse

---

## 🫁 DirLab-4DCT

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

**Python API** (recommended):
```python
from physiomotion4d.lung_gatedct_to_usd import data_dirlab_4d_ct

# Download specific case
downloader = data_dirlab_4d_ct.DirLab4DCT()
downloader.download_case(1)  # Downloads Case 1

# Or download multiple cases
for case_num in [1, 2, 3]:
    downloader.download_case(case_num)
```

**Automatic download via notebooks**:
```python
# See: experiments/Lung-GatedCT_To_USD/0-register_dirlab_4dct.ipynb
# The notebook includes automatic download functionality
```

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

## 🧠 KCL-Heart-Model

### Description
Statistical shape model (SSM) of the human heart derived from cardiac imaging data. Includes principal component analysis (PCA) modes of shape variation.

### Specifications
- **Format**: `.vtk` (VTK PolyData format)
- **Content**: 
  - Average heart mesh
  - Individual heart models
  - PCA eigenvectors
  - Mode standard deviations
  - Variance explained by each mode
- **Components**: Full heart mesh with chambers and vessels

### Files
- `average.vtk` - Mean heart shape
- `Full_Heart_Mesh_1.vtk` - Example individual heart
- `Eigenvectors.csv` - PCA eigenvectors (shape modes)
- `Mode_standard_deviations.csv` - Standard deviation for each mode
- `Normalized_explained_variance.csv` - Variance explained by each PCA mode
- `data_description.pdf` - Detailed model documentation
- `publication.pdf` - Associated research publication
- `citation.txt` - Citation information

### Acknowledgement
Data from King's College London (KCL):
- **Repository**: Likely from cardiac imaging research group
- **License**: Check `citation.txt` for proper attribution

### Downloading the Data
Data is typically obtained from published research repositories or upon request from the authors. Check the included PDFs for source information.

### Usage
- Model-to-image registration experiments (`Heart-Model_To_Image_Registration/`)
- Shape-based cardiac analysis
- Atlas-based segmentation initialization
- Statistical shape analysis

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

- **Slicer-Heart-CT**: Public release from GitHub
- **DirLab-4DCT**: Public benchmark dataset (may require registration)
- **KCL-Heart-Model**: Check included citation and license files

⚠️ **Important**: Always cite the original data sources in publications and respect any usage restrictions.

---

## 📚 References

### Slicer-Heart-CT
- Jolley Lab: https://www.linkedin.com/company/jolleylab
- GitHub: https://github.com/Slicer-Heart-CT/Slicer-Heart-CT

### DirLab-4DCT
- Castillo et al., "A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGene study archive"
- DIR-Lab: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/

### KCL-Heart-Model
- See included PDFs and citation.txt for proper attribution

---

## 💡 Tips

1. **Storage**: Ensure adequate disk space (~10-20GB for all datasets)
2. **Download Time**: Initial downloads can be slow; be patient
3. **Organization**: Keep data organized; don't modify original files
4. **Backups**: Consider backing up processed results separately
5. **Documentation**: Update this README when adding new datasets
