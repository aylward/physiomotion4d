====================================
Base Class (PhysioMotion4DBase)
====================================

.. currentmodule:: physiomotion4d

The :class:`PhysioMotion4DBase` class provides foundational functionality for all PhysioMotion4D components.

Overview
========

All major PhysioMotion4D classes inherit from :class:`PhysioMotion4DBase`, which provides:

* Structured logging with configurable verbosity
* Parameter validation and management
* File path handling and organization
* Common image I/O operations
* Error handling patterns

This inheritance ensures consistent behavior across the package.

Class Reference
===============

.. autoclass:: PhysioMotion4DBase
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Key Features
============

Logging Infrastructure
----------------------

The base class provides sophisticated logging with multiple levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) and outputs to both console and log files.

**Example:**

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   
   class MyProcessor(PhysioMotion4DBase):
       def __init__(self):
           super().__init__(verbose=True)
       
       def process(self):
           self.log("Starting processing", level="INFO")
           self.log("Detailed info", level="DEBUG")
           self.log("Warning message", level="WARNING")

Parameter Management
--------------------

Parameters are validated and stored systematically with built-in validation methods.

**Example:**

.. code-block:: python

   class MyWorkflow(PhysioMotion4DBase):
       def __init__(self, input_file, output_dir="./results"):
           super().__init__()
           
           self.input_file = self.validate_file_exists(input_file)
           self.output_dir = self.ensure_directory(output_dir)

File Path Utilities
-------------------

Consistent file path handling with automatic directory creation and path generation.

**Example:**

.. code-block:: python

   class MyWorkflow(PhysioMotion4DBase):
       def save_results(self, data, name):
           output_path = self.get_output_path(name, ".mha")
           self.save_image(data, output_path)
           self.log(f"Saved to {output_path}")

Best Practices
==============

1. **Always call super().__init__()** in derived class constructors
2. **Use logging, not print statements** for all output
3. **Validate inputs early** in __init__ or method entry points
4. **Document parameters clearly** using docstrings

See Also
========

* :doc:`workflows` - Workflow classes that build on this base
* :doc:`../developer/extending` - Guide to creating new classes
* :doc:`../developer/architecture` - Overall system architecture

.. rubric:: Navigation

:doc:`index` | :doc:`workflows`
