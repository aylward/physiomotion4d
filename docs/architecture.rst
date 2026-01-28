============
Architecture
============

Overview of PhysioMotion4D's system architecture and design principles.

.. note::

   **Implementation References:**

   * **src/physiomotion4d/cli/** - See production implementations of the architectural patterns described here
   * **experiments/** - See research prototypes showing evolution of these design principles

System Overview
===============

PhysioMotion4D is designed as a modular pipeline for medical imaging processing:

.. code-block:: text

   Input (4D CT) → Preprocessing → Registration → Segmentation
                                                     ↓
   USD (Omniverse) ← USD Export ← Transform ← Contour Extraction

Core Components
===============

1. **Workflow Processors**

   * :class:`ProcessHeartGatedCT`: Complete pipeline orchestration

2. **Segmentation Module**

   * Base class: :class:`SegmentChestBase`
   * Implementations: TotalSegmentator, VISTA-3D, Ensemble

3. **Registration Module**

   * Base class: :class:`RegisterImagesBase`
   * Implementations: ICON, ANTs

4. **Transform Module**

   * :class:`TransformTools`: Apply deformation fields
   * :class:`ContourTools`: Surface extraction and processing

5. **USD Export Module**

   * :class:`ConvertVTKToUSD`: VTK to USD conversion
   * :class:`USDTools`: USD manipulation
   * :class:`USDAnatomyTools`: Material application

Design Principles
=================

* **Modularity**: Components can be used independently
* **Extensibility**: Easy to add new methods
* **GPU Acceleration**: Leverage CUDA when available
* **Type Safety**: Type hints throughout
* **Error Handling**: Graceful degradation

For implementation details, see the source code and :doc:`api/base`.
