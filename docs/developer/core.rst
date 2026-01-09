====================================
Core Base Class
====================================

The :class:`PhysioMotion4DBase` class provides foundational functionality for all PhysioMotion4D components, including logging, parameter management, and common utilities.

Overview
========

All major PhysioMotion4D classes inherit from :class:`PhysioMotion4DBase`, which provides:

* Structured logging with configurable verbosity
* Parameter validation and management
* File path handling and organization
* Common image I/O operations
* Error handling patterns

This inheritance ensures consistent behavior across the package.

PhysioMotion4DBase Class
=========================

.. autoclass:: physiomotion4d.PhysioMotion4DBase
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
============

Logging Infrastructure
----------------------

The base class provides sophisticated logging:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   
   class MyProcessor(PhysioMotion4DBase):
       def __init__(self):
           super().__init__(verbose=True)
       
       def process(self):
           self.log("Starting processing", level="INFO")
           self.log("Detailed info", level="DEBUG")
           self.log("Warning message", level="WARNING")
           self.log("Error occurred", level="ERROR")

**Log Levels**:
   * ``DEBUG``: Detailed diagnostic information
   * ``INFO``: General informational messages
   * ``WARNING``: Warning messages for unusual situations
   * ``ERROR``: Error messages for failures
   * ``CRITICAL``: Critical errors requiring immediate attention

**Output**:
   * Console output (colored, formatted)
   * Log file (``physiomotion4d.log``)
   * Timestamps and context automatically included

Parameter Management
--------------------

Parameters are validated and stored systematically:

.. code-block:: python

   class MyWorkflow(PhysioMotion4DBase):
       def __init__(self, input_file, output_dir="./results"):
           super().__init__()
           
           # Validate and store parameters
           self.input_file = self.validate_file_exists(input_file)
           self.output_dir = self.ensure_directory(output_dir)

**Validation Methods**:
   * ``validate_file_exists(path)``: Check file exists
   * ``validate_directory(path)``: Check directory exists
   * ``ensure_directory(path)``: Create directory if needed
   * ``validate_parameter(value, allowed)``: Check parameter value

File Path Utilities
-------------------

Consistent file path handling:

.. code-block:: python

   class MyWorkflow(PhysioMotion4DBase):
       def __init__(self, output_dir):
           super().__init__()
           self.output_dir = output_dir
       
       def save_results(self, data, name):
           # Generate output path
           output_path = self.get_output_path(name, ".mha")
           # Save data
           self.save_image(data, output_path)
           self.log(f"Saved to {output_path}")

**Path Methods**:
   * ``get_output_path(basename, ext)``: Generate output file path
   * ``get_temp_path()``: Get temporary file path
   * ``organize_outputs()``: Create output directory structure

Common Use Patterns
===================

Creating a New Class
--------------------

Inherit from :class:`PhysioMotion4DBase`:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   
   class CustomProcessor(PhysioMotion4DBase):
       """My custom processor."""
       
       def __init__(self, param1, param2, verbose=False):
           """Initialize processor.
           
           Args:
               param1: First parameter
               param2: Second parameter
               verbose: Enable verbose logging
           """
           super().__init__(verbose=verbose)
           
           self.param1 = param1
           self.param2 = param2
           
           self.log("CustomProcessor initialized", level="INFO")
       
       def process(self):
           """Execute processing."""
           self.log("Starting processing...")
           
           # Processing steps here
           result = self._internal_method()
           
           self.log("Processing complete", level="INFO")
           return result
       
       def _internal_method(self):
           """Internal helper method."""
           self.log("Internal processing", level="DEBUG")
           # Implementation
           pass

Error Handling
--------------

Use base class error handling patterns:

.. code-block:: python

   def process_image(self, image_path):
       """Process an image with error handling."""
       try:
           # Validate input
           image_path = self.validate_file_exists(image_path)
           
           # Load and process
           image = self.load_image(image_path)
           result = self.apply_processing(image)
           
           return result
           
       except FileNotFoundError as e:
           self.log(f"File not found: {e}", level="ERROR")
           raise
       except Exception as e:
           self.log(f"Processing failed: {e}", level="ERROR")
           raise

Configuration Management
------------------------

Store and access configuration:

.. code-block:: python

   class ConfigurableWorkflow(PhysioMotion4DBase):
       def __init__(self, config_dict):
           super().__init__()
           
           # Store configuration
           self.config = config_dict
           
           # Extract parameters with defaults
           self.iterations = config_dict.get('iterations', 100)
           self.learning_rate = config_dict.get('lr', 0.001)
           
           self.log(f"Configuration: {self.config}", level="DEBUG")
       
       def get_config(self):
           """Return current configuration."""
           return {
               'iterations': self.iterations,
               'learning_rate': self.learning_rate
           }

Advanced Features
=================

Progress Tracking
-----------------

Track progress through long operations:

.. code-block:: python

   def process_time_series(self, frames):
       """Process multiple frames with progress tracking."""
       total = len(frames)
       
       for i, frame in enumerate(frames):
           self.log(f"Processing frame {i+1}/{total}", level="INFO")
           
           # Process frame
           result = self.process_frame(frame)
           
           # Calculate progress
           progress = (i + 1) / total * 100
           self.log(f"Progress: {progress:.1f}%", level="DEBUG")

Checkpoint System
-----------------

Save and resume from checkpoints:

.. code-block:: python

   def process_with_checkpoints(self):
       """Process with checkpoint support."""
       checkpoint_file = self.get_output_path("checkpoint", ".pkl")
       
       # Check for existing checkpoint
       if os.path.exists(checkpoint_file):
           self.log("Resuming from checkpoint", level="INFO")
           state = self.load_checkpoint(checkpoint_file)
       else:
           state = {'step': 0, 'data': None}
       
       # Process from checkpoint
       if state['step'] < 1:
           state['data'] = self.step1()
           state['step'] = 1
           self.save_checkpoint(state, checkpoint_file)
       
       if state['step'] < 2:
           result = self.step2(state['data'])
           state['step'] = 2
           self.save_checkpoint(state, checkpoint_file)
       
       return result

Integration Examples
====================

Using Base Class in Applications
---------------------------------

Integrate PhysioMotion4D classes into your application:

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   def my_application():
       """Application using PhysioMotion4D."""
       
       # Create workflow instance
       workflow = WorkflowConvertHeartGatedCTToUSD(
           input_filenames=["cardiac_4d.nrrd"],
           contrast_enhanced=True,
           output_directory="./results",
           verbose=True  # Enable logging from base class
       )
       
       # All logging comes through base class
       # Access logs programmatically if needed
       log_file = workflow.get_log_file_path()
       
       # Execute workflow
       result = workflow.process()
       
       return result

Custom Logging Handlers
------------------------

Add custom log handlers:

.. code-block:: python

   import logging
   from physiomotion4d import PhysioMotion4DBase
   
   class CustomLogger(PhysioMotion4DBase):
       def __init__(self):
           super().__init__()
           
           # Add custom handler (e.g., to database, cloud)
           custom_handler = logging.StreamHandler()
           custom_handler.setFormatter(
               logging.Formatter('%(asctime)s - %(message)s')
           )
           self.logger.addHandler(custom_handler)

Best Practices
==============

Inheritance Guidelines
----------------------

1. **Always call super().__init__()**
   
   .. code-block:: python
   
      def __init__(self, *args, **kwargs):
          super().__init__(verbose=kwargs.get('verbose', False))

2. **Use logging, not print statements**
   
   .. code-block:: python
   
      # Good
      self.log("Processing started", level="INFO")
      
      # Avoid
      print("Processing started")

3. **Validate inputs early**
   
   .. code-block:: python
   
      def __init__(self, input_file):
          super().__init__()
          self.input_file = self.validate_file_exists(input_file)

4. **Document parameters clearly**
   
   Use docstrings with parameter descriptions

Parameter Validation
--------------------

Validate all user-provided parameters:

.. code-block:: python

   def set_parameter(self, value):
       """Set parameter with validation."""
       if not isinstance(value, int):
           raise TypeError("Parameter must be integer")
       if value < 1:
           raise ValueError("Parameter must be positive")
       
       self.parameter = value
       self.log(f"Parameter set to {value}", level="DEBUG")

See Also
========

* :doc:`workflows` - Workflow classes building on base class
* :doc:`extending` - Creating new classes from base
* :doc:`architecture` - Overall system architecture
