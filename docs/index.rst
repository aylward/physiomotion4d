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
   :target: https://github.com/NVIDIA/PhysioMotion4D/blob/main/LICENSE
   :alt: License

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
   :caption: User Guide

   user_guide/heart_gated_ct
   user_guide/lung_4dct
   user_guide/segmentation
   user_guide/registration
   user_guide/usd_conversion
   user_guide/visualization
   user_guide/logging

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_workflow
   tutorials/custom_segmentation
   tutorials/image_registration
   tutorials/vtk_to_usd
   tutorials/colormap_rendering
   tutorials/model_to_image_registration
   ants_initial_transform_guide

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/segmentation
   api/registration
   api/utilities

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   architecture
   testing
   changelog
   README
   DOCUMENTATION_SETUP
   LOGGING_API_REFERENCE
   PYPI_RELEASE_GUIDE

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   troubleshooting
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

