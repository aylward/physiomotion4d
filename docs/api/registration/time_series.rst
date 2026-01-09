====================================
Time Series Registration
====================================

.. currentmodule:: physiomotion4d

Register 4D medical image sequences (cardiac-gated, respiratory-gated).

Class Reference
===============

.. autoclass:: RegisterTimeSeriesImages
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
========

Specialized registration for 4D sequences, tracking motion through time with temporal consistency constraints.

**Key Features**:
   * Handles multi-phase cardiac or respiratory cycles
   * Temporal smoothness constraints
   * Efficient batch processing
   * Supports both ANTs and Icon backends

Usage Examples
==============

Register Cardiac Sequence
--------------------------

.. code-block:: python

   from physiomotion4d import RegisterTimeSeriesImages
   
   registrar = RegisterTimeSeriesImages(
       method="icon",
       device="cuda:0",
       temporal_smoothing=True,
       verbose=True
   )
   
   transforms = registrar.register_sequence(
       image_files=[
           "cardiac_phase_00.nrrd",
           "cardiac_phase_01.nrrd",
           "cardiac_phase_02.nrrd",
           # ... more phases
       ],
       reference_index=0  # Use first phase as reference
   )

See Also
========

* :doc:`index` - Registration overview
* :doc:`icon` - Icon backend
* :doc:`../workflows` - Complete 4D workflows

.. rubric:: Navigation

:doc:`icon` | :doc:`index` | :doc:`../model_registration/index`
