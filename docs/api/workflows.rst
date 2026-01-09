====================
Workflow Classes
====================

.. currentmodule:: physiomotion4d

High-level workflow classes that orchestrate complete processing pipelines.

Overview
========

PhysioMotion4D provides workflow classes that combine multiple processing steps into complete pipelines. Each workflow handles a specific medical imaging task from input to USD output.

Available Workflows
===================

Heart Gated CT to USD
---------------------

.. autoclass:: WorkflowConvertHeartGatedCTToUSD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Purpose**: Convert cardiac-gated CT sequences to animated USD models.

**Features**:
   * Automatic segmentation of cardiac structures
   * Motion field computation using deep learning registration
   * USD timeline animation generation
   * Support for both contrast-enhanced and non-contrast CT

**Example:**

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=["cardiac_phase_00.nrrd", "cardiac_phase_01.nrrd"],
       contrast_enhanced=True,
       output_directory="./heart_results",
       verbose=True
   )
   
   result = workflow.process()

Heart Model to Patient Registration
------------------------------------

.. autoclass:: WorkflowRegisterHeartModelToPatient
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

**Purpose**: Register a template heart model to patient-specific imaging data.

**Features**:
   * Multi-scale registration pipeline
   * PCA-based shape matching
   * Distance map optimization
   * Physiological constraint preservation

**Example:**

.. code-block:: python

   from physiomotion4d import WorkflowRegisterHeartModelToPatient
   
   workflow = WorkflowRegisterHeartModelToPatient(
       model_file="heart_template.vtk",
       patient_image="patient_ct.nrrd",
       output_directory="./registration_results",
       verbose=True
   )
   
   registered_model = workflow.process()

Common Workflow Patterns
========================

Step-based Processing
---------------------

All workflows support step-based execution with checkpointing:

.. code-block:: python

   # Create workflow
   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=file_list,
       output_directory="./results"
   )
   
   # Run specific steps
   workflow.run_step(1)  # Segmentation
   workflow.run_step(2)  # Registration
   workflow.run_step(3)  # USD generation
   
   # Or run all steps
   workflow.process()

Progress Monitoring
-------------------

Monitor workflow progress with logging:

.. code-block:: python

   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=file_list,
       output_directory="./results",
       verbose=True  # Enable detailed logging
   )
   
   # Workflow provides detailed progress updates
   result = workflow.process()
   
   # Access log file
   log_file = workflow.get_log_file_path()

Parameter Customization
-----------------------

Customize workflow parameters:

.. code-block:: python

   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=file_list,
       output_directory="./results",
       
       # Segmentation options
       segmentation_method="vista3d",  # or "totalsegmentator"
       contrast_enhanced=True,
       
       # Registration options
       registration_device="cuda:0",
       registration_iterations=100,
       
       # USD options
       usd_format="polymesh",  # or "tetmesh"
       colormap="rainbow",
       
       verbose=True
   )
   
   result = workflow.process()

Error Handling
--------------

Workflows include comprehensive error handling:

.. code-block:: python

   try:
       workflow = WorkflowConvertHeartGatedCTToUSD(
           input_filenames=file_list,
           output_directory="./results"
       )
       result = workflow.process()
   except FileNotFoundError as e:
       print(f"Input file not found: {e}")
   except RuntimeError as e:
       print(f"Processing failed: {e}")
       # Workflow automatically saves checkpoint
       # Can resume from last successful step

Advanced Usage
==============

Custom Workflow Creation
------------------------

Create custom workflows by inheriting from base classes:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   
   class CustomWorkflow(PhysioMotion4DBase):
       """Custom medical imaging workflow."""
       
       def __init__(self, input_files, output_dir, **kwargs):
           super().__init__(verbose=kwargs.get('verbose', False))
           
           self.input_files = input_files
           self.output_dir = output_dir
       
       def process(self):
           """Execute workflow steps."""
           self.log("Starting custom workflow", level="INFO")
           
           # Step 1: Load data
           data = self.load_data()
           
           # Step 2: Process
           result = self.process_data(data)
           
           # Step 3: Save results
           self.save_results(result)
           
           self.log("Workflow complete", level="INFO")
           return result

Workflow Composition
--------------------

Combine multiple workflows:

.. code-block:: python

   # First workflow: Process heart CT
   heart_workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=ct_files,
       output_directory="./heart_output"
   )
   heart_result = heart_workflow.process()
   
   # Second workflow: Register model
   registration_workflow = WorkflowRegisterHeartModelToPatient(
       model_file=heart_result['model'],
       patient_image=patient_ct,
       output_directory="./registration_output"
   )
   final_result = registration_workflow.process()

Best Practices
==============

1. **Always validate inputs** before starting workflow
2. **Use verbose mode** during development and debugging
3. **Check intermediate results** after each step
4. **Save checkpoints** for long-running workflows
5. **Handle exceptions** appropriately for production use

Performance Tips
================

* Use GPU acceleration when available (set ``device="cuda:0"``)
* Enable fast mode for segmentation (``fast=True``)
* Process multiple time points in batch
* Monitor memory usage for large datasets

See Also
========

* :doc:`../cli_scripts/overview` - Command-line workflow execution
* :doc:`../developer/workflows` - Workflow development guide
* :doc:`segmentation/index` - Segmentation modules
* :doc:`registration/index` - Registration modules
* :doc:`usd/index` - USD generation modules

.. rubric:: Navigation

:doc:`base` | :doc:`index` | :doc:`segmentation/index`
