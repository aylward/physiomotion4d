========
Testing
========

Guide to running and writing tests for PhysioMotion4D.

Running Tests
=============

Basic Test Execution
--------------------

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with verbose output
   pytest tests/ -v

   # Run specific test file
   pytest tests/test_usd_merge.py -v

   # Run specific test
   pytest tests/test_usd_merge.py::test_merge_basic -v

Test Categories
---------------

PhysioMotion4D uses pytest markers to categorize tests:

.. code-block:: bash

   # Fast tests only (recommended for development)
   pytest tests/ -m "not slow and not requires_data" -v

   # Unit tests only
   pytest tests/ -m unit

   # Integration tests only
   pytest tests/ -m integration

   # Skip slow tests (registration and segmentation)
   pytest tests/ -m "not slow"

   # Skip tests requiring external data
   pytest tests/ -m "not requires_data"

   # Skip GPU-dependent tests
   pytest tests/ --ignore=tests/test_segment_chest_total_segmentator.py \
                 --ignore=tests/test_segment_chest_vista_3d.py \
                 --ignore=tests/test_register_images_icon.py

Specific Test Modules
----------------------

.. code-block:: bash

   # USD utilities (fast)
   pytest tests/test_usd_merge.py -v
   pytest tests/test_usd_time_preservation.py -v

   # Data conversion (fast)
   pytest tests/test_convert_nrrd_4d_to_3d.py -v
   pytest tests/test_convert_vtk_4d_to_usd_polymesh.py -v

   # Image tools (fast)
   pytest tests/test_image_tools.py -v
   pytest tests/test_contour_tools.py -v
   pytest tests/test_transform_tools.py -v

   # Registration (slow, ~5-10 minutes each)
   pytest tests/test_register_images_ants.py -v
   pytest tests/test_register_images_icon.py -v
   pytest tests/test_register_time_series_images.py -v

   # Segmentation (GPU required, ~2-5 minutes each)
   pytest tests/test_segment_chest_total_segmentator.py -v
   pytest tests/test_segment_chest_vista_3d.py -v

Coverage Reports
----------------

.. code-block:: bash

   # Generate coverage report
   pytest tests/ --cov=src/physiomotion4d --cov-report=html

   # View report
   open htmlcov/index.html

Test Structure
==============

Tests are organized by functionality:

.. code-block:: text

   tests/
   ├── Data Pipeline Tests
   │   ├── test_download_heart_data.py           # Data download with fallback logic
   │   └── test_convert_nrrd_4d_to_3d.py         # 4D to 3D conversion
   │
   ├── Segmentation Tests (GPU Required)
   │   ├── test_segment_chest_total_segmentator.py  # TotalSegmentator
   │   └── test_segment_chest_vista_3d.py           # VISTA-3D segmentation
   │
   ├── Registration Tests (Slow ~5-10 min)
   │   ├── test_register_images_ants.py          # ANTs registration
   │   ├── test_register_images_icon.py          # Icon registration  
   │   └── test_register_time_series_images.py   # Time series registration
   │
   ├── Geometry & Visualization Tests
   │   ├── test_contour_tools.py                 # Mesh extraction and manipulation
   │   ├── test_transform_tools.py               # Transform operations
   │   ├── test_image_tools.py                   # Image processing utilities
   │   └── test_convert_vtk_4d_to_usd_polymesh.py # VTK to USD conversion
   │
   └── USD Utility Tests
       ├── test_usd_merge.py                     # USD file merging
       └── test_usd_time_preservation.py         # Time-varying data validation

Writing Tests
=============

See :doc:`contributing` for guidelines on writing tests.

Continuous Integration
======================

Tests run automatically on:

* Pull requests
* Merges to main branch
* Tagged releases

See the `.github/workflows/` directory for CI configuration.

For more details, see :doc:`tests/TESTING_GUIDE`.

