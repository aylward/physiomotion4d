.. PhysioMotion4D documentation master file

=====================================
PhysioMotion4D Documentation
=====================================

**Generate anatomic models in Omniverse with physiological motion derived from 4D medical images.**

PhysioMotion4D is a comprehensive medical imaging package that converts 4D CT scans (particularly heart and lung gated CT data) into dynamic 3D models for visualization in NVIDIA Omniverse. The package provides state-of-the-art deep learning-based image processing, segmentation, registration, and USD file generation capabilities.

.. image:: https://img.shields.io/pypi/v/physiomotion4d.svg
   :target: https://pypi.org/project/physiomotion4d/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/physiomotion4d.svg
   :target: https://pypi.org/project/physiomotion4d/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue.svg
   :target: https://github.com/aylward/PhysioMotion4d/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/actions/workflow/status/aylward/PhysioMotion4d/ci.yml?branch=main&label=CI%20Tests
   :target: https://github.com/aylward/PhysioMotion4d/actions/workflows/ci.yml
   :alt: CI Tests

.. image:: https://img.shields.io/badge/tests-Windows%20%7C%20Linux%20%7C%20Python%203.10--3.12-blue
   :target: https://github.com/aylward/PhysioMotion4d/actions/workflows/ci.yml
   :alt: Test Matrix: Windows, Linux, Python 3.10-3.12

.. image:: https://codecov.io/gh/aylward/PhysioMotion4d/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/aylward/PhysioMotion4d
   :alt: Test Coverage

🚀 Key Features
===============

* **Complete 4D Medical Imaging Pipeline**: End-to-end processing from 4D CT data to animated USD models
* **Multiple AI Segmentation Methods**: TotalSegmentator, VISTA-3D, and ensemble approaches
* **Deep Learning Registration**: GPU-accelerated image registration using Icon algorithm
* **NVIDIA Omniverse Integration**: Direct USD file export for medical visualization
* **Physiological Motion Analysis**: Capture and visualize cardiac and respiratory motion
* **Flexible Workflow Control**: Step-based processing with checkpoint management

📋 Supported Applications
==========================

* **Cardiac Imaging**: Heart-gated CT processing with cardiac motion analysis
* **Pulmonary Imaging**: Lung 4D-CT processing with respiratory motion tracking
* **Medical Education**: Interactive 3D anatomical models with physiological motion
* **Research Visualization**: Advanced medical imaging research in Omniverse
* **Clinical Planning**: Dynamic anatomical models for treatment planning

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: CLI & Scripts Guide

   cli_scripts/overview
   cli_scripts/heart_gated_ct
   cli_scripts/heart_model_to_patient
   cli_scripts/lung_gated_ct
   cli_scripts/4dct_reconstruction
   cli_scripts/vtk_to_usd
   cli_scripts/brain_vessel_modeling
   cli_scripts/best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/base
   api/workflows
   api/segmentation/index
   api/registration/index
   api/model_registration/index
   api/usd/index
   api/utilities/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guides

   developer/architecture
   developer/extending
   developer/workflows
   developer/core
   developer/segmentation
   developer/registration_images
   developer/registration_models
   developer/usd_generation
   developer/utilities

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing
   testing

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   troubleshooting
   references
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
