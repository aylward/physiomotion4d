# KCL Heart Model Dataset

## Overview

This directory contains data from the King's College London (KCL) four-chamber heart model dataset, which provides a virtual cohort of adult healthy heart meshes derived from CT images.

## Dataset Access

The KCL heart model data is publicly available through Zenodo:

**🔗 [Virtual cohort of adult healthy four-chamber heart meshes from CT images](https://zenodo.org/records/4590294)**

### Dataset Details

- **20 four-chamber heart models** generated from end-diastolic CT scans
- Tetrahedral meshes with average edge length of 1 mm
- Includes ventricular fibers and Universal Ventricular Coordinates (UVC)
- Contains labels for all cardiac chambers, valves, and major vessels
- Provides statistical shape model (SSM) weights and simulation outputs

### Citation

If you use this dataset, please cite:

> Rodero et al. (2021), "Linking statistical shape models and simulated function in the healthy adult human heart". *PLOS Computational Biology*. DOI: [10.1371/journal.pcbi.1008851](https://doi.org/10.1371/journal.pcbi.1008851)

## PCA Shape Statistics with SlicerSALT

[**SlicerSALT**](https://salt.slicer.org/) (Shape AnaLysis Toolbox) can be used to compute Principal Component Analysis (PCA) shape statistics from the KCL heart model data.

### About SlicerSALT

SlicerSALT is a powerful open-source shape analysis toolbox built on 3D Slicer that provides:

- **Point Distributed Models (PDM)** computation using Spherical Harmonic Representation (SPHARM-PDM)
- **Correspondence optimization** for study-wise shape analysis
- **PCA analysis** for shape space exploration
- **4D regression** for time-varying shape data
- **Advanced shape statistics** for hypothesis testing
- Command-line batch processing capabilities

### Using SlicerSALT with KCL Data

1. **Download SlicerSALT**: Get the latest version from [https://salt.slicer.org/](https://salt.slicer.org/)

2. **Download KCL Data**: Obtain the heart meshes from [Zenodo](https://zenodo.org/records/4590294)

3. **Import Surfaces**: SlicerSALT works on surfaces, not tetmeshes, so convert the KCL data using the jupyter notebook in this directory, and then load the VTK mesh files into SlicerSALT

4. **Compute Correspondence**: Use SlicerSALT's correspondence optimization tools to establish point correspondences across the heart meshes

5. **Run PCA**: Perform Principal Component Analysis to identify the main modes of shape variation

6. **Analyze Results**: Explore the shape space and extract statistical shape features

### Files in This Directory

- `pca/`: PCA results and shape statistics computed using SlicerSALT
- `citation.txt`: Citation information for the dataset

## Additional Resources

- **SlicerSALT Documentation**: Available through the SlicerSALT website
- **SlicerSALT GitHub**: [https://github.com/KitwareMedical/SlicerSALT](https://github.com/KitwareMedical/SlicerSALT)
- **Related Datasets**:
  - [Virtual cohort of extreme and average four-chamber heart meshes](https://zenodo.org/records/4593739)
  - [Virtual cohort of 1000 synthetic heart meshes](https://zenodo.org/records/4506930)

## Contact

For questions about the dataset, contact the original authors:
- Cristobal Rodero, King's College London
- Steven A. Niederer, King's College London
- Pablo Lamata, King's College London

For questions about SlicerSALT, contact: beatriz.paniagua@kitware.com

