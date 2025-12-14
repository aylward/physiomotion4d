================================
Logging Guide
================================

Overview
========

The :class:`~physiomotion4d.PhysioMotion4DBase` class provides standardized logging functionality 
for all PhysioMotion4D classes. It replaces scattered ``print()`` statements with a professional, 
configurable logging system based on Python's standard ``logging`` module.

**All PhysioMotion4D classes share a single logger called "PhysioMotion4D"**, but each class name 
is included in the log messages for identification. This allows for unified log management and 
selective filtering by class name.

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   import logging

   class MyClass(PhysioMotion4DBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(
               logger_name="MyClass",
               log_level=log_level
           )
       
       def process(self):
           self.log_info("Processing started")
           self.log_debug("Detailed debug information")
           self.log_warning("Warning message")
           self.log_error("Error message")

Key Features
============

Shared Logger with Class Identification
-----------------------------------------

- All classes use the shared "PhysioMotion4D" logger
- Each message includes the class name in brackets: ``[ClassName]``
- Unified log management across all PhysioMotion4D classes

Example output:

.. code-block:: text

   2025-12-13 11:24:49 - PhysioMotion4D - INFO - [RegisterModelToImagePCA] Processing started
   2025-12-13 11:24:49 - PhysioMotion4D - DEBUG - [HeartModelToPatientWorkflow] Detailed debug info

Multiple Log Levels
-------------------

- ``log_debug()`` - Detailed diagnostic information
- ``log_info()`` - General informational messages
- ``log_warning()`` - Warning messages
- ``log_error()`` - Error messages
- ``log_critical()`` - Critical errors

Flexible Log Level Control
----------------------------

- **DEBUG level** (``logging.DEBUG``): Shows all messages including detailed diagnostics
- **INFO level** (``logging.INFO``): Shows informational messages and above (default)
- **WARNING level** (``logging.WARNING``): Shows only warnings, errors, and critical messages

Class Filtering
---------------

Filter to show logs from only specific classes:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase

   # Show only RegisterModelToImagePCA logs
   PhysioMotion4DBase.set_log_classes(["RegisterModelToImagePCA"])

   # Show logs from multiple classes
   PhysioMotion4DBase.set_log_classes([
       "RegisterModelToImagePCA",
       "HeartModelToPatientWorkflow"
   ])

   # Show all classes again
   PhysioMotion4DBase.set_log_all_classes()

   # Query which classes are filtered
   classes = PhysioMotion4DBase.get_log_classes()
   print(classes)  # [] if all enabled, or list of class names

Professional Formatting
-----------------------

All log messages include:

- Timestamp
- Logger name ("PhysioMotion4D")
- Log level
- Class name in brackets
- Message

Convenience Methods
-------------------

Section Headers
^^^^^^^^^^^^^^^

.. code-block:: python

   self.log_section("Stage 1: Initialization", width=70)

Outputs:

.. code-block:: text

   ======================================================================
   Stage 1: Initialization
   ======================================================================

Progress Tracking
^^^^^^^^^^^^^^^^^

.. code-block:: python

   for i in range(100):
       if i % 10 == 0:
           self.log_progress(i+1, 100, prefix="Processing")

Outputs:

.. code-block:: text

   Processing: 10/100 (10.0%)
   Processing: 20/100 (20.0%)
   Processing: 30/100 (30.0%)
   ...

Usage Patterns
==============

Creating a New Class
--------------------

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   import logging

   class NewRegistration(PhysioMotion4DBase):
       def __init__(self, param1, param2, log_level=logging.INFO):
           # Initialize base class first
           super().__init__(
               logger_name="NewRegistration",
               log_level=log_level
           )
           
           # Your initialization
           self.log_info("Initializing registration")
           self.param1 = param1
           self.param2 = param2
           self.log_debug(f"Parameters: param1={param1}, param2={param2}")

Converting Existing Classes
----------------------------

**Before:**

.. code-block:: python

   class MyClass:
       def __init__(self):
           print("Initializing...")
           print(f"Parameters: {self.params}")

**After:**

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase

   class MyClass(PhysioMotion4DBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(logger_name="MyClass", log_level=log_level)
           self.log_info("Initializing...")
           self.log_debug(f"Parameters: {self.params}")

Dynamic Log Level Control
--------------------------

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase

   # Create with INFO level output
   obj = MyClass(log_level=logging.INFO)

   # Change log level for ALL PhysioMotion4D classes
   PhysioMotion4DBase.set_log_level(logging.WARNING)
   obj.expensive_operation()  # Won't show INFO messages

   # Change back to INFO level
   PhysioMotion4DBase.set_log_level(logging.INFO)
   obj.log_info("Operation complete!")

   # Enable DEBUG level
   PhysioMotion4DBase.set_log_level(logging.DEBUG)
   obj.log_debug("Now showing debug messages")

   # Can also use strings
   PhysioMotion4DBase.set_log_level('WARNING')

Class Filtering for Selective Debugging
----------------------------------------

.. code-block:: python

   from physiomotion4d import (
       PhysioMotion4DBase,
       RegisterModelToImagePCA,
       HeartModelToPatientWorkflow
   )
   import logging

   # Create multiple objects
   registrar1 = RegisterModelToImagePCA(..., log_level=logging.INFO)
   registrar2 = HeartModelToPatientWorkflow(..., log_level=logging.INFO)

   # Show logs from all classes (default)
   registrar1.log_info("Message from PCA")
   registrar2.log_info("Message from Workflow")
   # Both messages are shown

   # Filter to show only RegisterModelToImagePCA
   PhysioMotion4DBase.set_log_classes(["RegisterModelToImagePCA"])
   registrar1.log_info("This is shown")
   registrar2.log_info("This is hidden")

   # Show multiple specific classes
   PhysioMotion4DBase.set_log_classes([
       "RegisterModelToImagePCA",
       "HeartModelToPatientWorkflow"
   ])

   # Show all classes again
   PhysioMotion4DBase.set_log_all_classes()

   # Check which classes are currently filtered
   filtered_classes = PhysioMotion4DBase.get_log_classes()
   print(filtered_classes)  # [] if all enabled, or list of class names

File Logging
------------

.. code-block:: python

   import logging

   # Log to both console and file
   obj = MyClass(log_level=logging.DEBUG, log_to_file="registration.log")

Best Practices
==============

Choose Appropriate Log Levels
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Level
     - When to Use
     - Examples
   * - DEBUG
     - Detailed diagnostic info
     - Variable values, loop iterations, internal state
   * - INFO
     - Normal operation messages
     - "Loading data", "Stage 1 started", "Complete"
   * - WARNING
     - Potential issues
     - "Using default value", "Parameter out of range"
   * - ERROR
     - Serious problems
     - "Failed to load file", "Invalid parameter"
   * - CRITICAL
     - Fatal errors
     - "Cannot continue", "Critical failure"

Use Section Headers for Major Steps
------------------------------------

.. code-block:: python

   self.log_section("Stage 1: Rigid Alignment")
   # ... stage 1 code ...

   self.log_section("Stage 2: Deformable Registration")
   # ... stage 2 code ...

Add Debug Information Generously
---------------------------------

.. code-block:: python

   self.log_debug(f"Input dimensions: {data.shape}")
   self.log_debug(f"Processing with {n_iterations} iterations")
   self.log_debug(f"Intermediate result: {intermediate_value}")

Debug messages won't clutter normal output unless debug mode is enabled.

Use Hierarchical Indentation
-----------------------------

.. code-block:: python

   self.log_info("Main operation:")
   self.log_info("  Sub-operation 1")
   self.log_info("  Sub-operation 2")
   self.log_info("    Detail of sub-operation 2")

Examples
========

Example 1: Basic Registration Class
------------------------------------

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase

   class SimpleRegistration(PhysioMotion4DBase):
       def __init__(self, image, mesh, log_level=logging.INFO):
           super().__init__(logger_name="SimpleRegistration", log_level=log_level)
           
           self.log_info("Initializing registration")
           self.image = image
           self.mesh = mesh
           self.log_info(f"Image size: {image.shape}")
           self.log_info(f"Mesh points: {mesh.n_points}")
       
       def register(self):
           self.log_section("Registration")
           self.log_info("Starting rigid alignment...")
           # ... registration code ...
           self.log_info("Registration complete!")

Example 2: With Progress Tracking
----------------------------------

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase

   class IterativeProcess(PhysioMotion4DBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(logger_name="IterativeProcess", log_level=log_level)
       
       def process(self, n_iterations=100):
           self.log_section("Iterative Processing")
           
           for i in range(n_iterations):
               # Process iteration
               objective = self.compute_objective()
               self.log_debug(f"Iteration {i}: objective = {objective}")
               
               # Log progress every 10 iterations
               if (i + 1) % 10 == 0:
                   self.log_progress(i + 1, n_iterations, prefix="Processing")
           
           self.log_info("Processing complete!")

Example 3: Error Handling
--------------------------

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase

   class SafeProcessor(PhysioMotion4DBase):
       def __init__(self, log_level=logging.INFO):
           super().__init__(logger_name="SafeProcessor", log_level=log_level)
       
       def process_file(self, filename):
           self.log_info(f"Processing file: {filename}")
           
           try:
               # ... file processing ...
               self.log_info("File processed successfully")
           except FileNotFoundError:
               self.log_error(f"File not found: {filename}")
               raise
           except ValueError as e:
               self.log_error(f"Invalid file format: {e}")
               raise

Benefits
========

1. **Professional output** - Clean, timestamped, structured logging
2. **Flexible control** - Users can adjust verbosity without code changes
3. **Better debugging** - Debug messages can be toggled on/off
4. **File logging** - Easy to save logs for later analysis
5. **Standard practice** - Uses Python's standard logging module
6. **Maintainable** - Consistent logging across all classes
7. **User-friendly** - Clear, hierarchical output structure

See Also
========

* :class:`~physiomotion4d.PhysioMotion4DBase` - Base class API reference
* :doc:`../api/utilities` - Utilities API reference
* :doc:`../api/core` - Core API reference


