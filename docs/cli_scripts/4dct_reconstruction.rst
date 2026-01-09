====================================
4D-CT Reconstruction
====================================

.. note::
   This script is planned for a future release. Documentation will be updated when available.

Overview
========

The ``physiomotion4d-4dct-reconstruction`` script will reconstruct 4D CT volumes from multiple 3D acquisitions taken at different physiological phases or time points.

Planned Features
================

Input Data
----------

* **Multiple 3D CT Scans**: Acquired at different phases
* **Phase Information**: Cardiac or respiratory phase labels
* **Optional**: ECG or respiratory signals for phase assignment

Reconstruction Methods
----------------------

1. **Phase-Based Sorting**
   * Sort slices/acquisitions by physiological phase
   * Group into temporal bins
   * Handle irregular sampling

2. **Motion Estimation**
   * Estimate motion between phases
   * Build temporal motion model
   * Interpolate missing data

3. **4D Volume Assembly**
   * Reconstruct complete 4D dataset
   * Apply motion compensation
   * Ensure temporal smoothness

4. **Quality Control**
   * Detect artifacts and inconsistencies
   * Generate confidence maps
   * Provide reconstruction metrics

Expected Outputs
----------------

* Complete 4D NRRD volume
* Motion vector fields
* Phase-assignment maps
* Quality metrics and reports

Use Cases
=========

* **Cardiac Imaging**: Retrospective cardiac CT reconstruction
* **Respiratory Imaging**: Build 4D-CT from helical acquisitions
* **Research**: Study temporal resolution effects
* **Clinical**: Improve motion characterization

Workflow Class
==============

For Python API access, see :class:`physiomotion4d.WorkflowReconstructFourDCT` in :doc:`../developer/workflows`.

Related Scripts
===============

* :doc:`heart_gated_ct` - Process reconstructed cardiac 4D-CT
* :doc:`lung_gated_ct` - Process reconstructed lung 4D-CT
