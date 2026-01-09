====================================
Workflows
====================================

Workflow classes orchestrate complete processing pipelines, from raw medical images to final USD outputs. Each workflow corresponds to a CLI script and provides programmatic access to the same functionality.

Overview
========

Workflows in PhysioMotion4D:

* Coordinate multiple processing steps (segmentation, registration, conversion)
* Manage intermediate data and checkpoints
* Provide high-level interfaces for common tasks
* Support both programmatic and CLI usage

All workflow classes inherit from :class:`PhysioMotion4DBase` and follow consistent patterns.

Workflow to Script Mapping
===========================

Each CLI script wraps a corresponding workflow class:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - CLI Script
     - Workflow Class
   * - ``physiomotion4d-heart-gated-ct``
     - :class:`WorkflowConvertHeartGatedCTToUSD`
   * - ``physiomotion4d-heart-model-to-patient``
     - :class:`WorkflowRegisterHeartModelToPatient` *(planned)*
   * - ``physiomotion4d-lung-gated-ct``
     - :class:`LungGatedCTToUSDWorkflow` *(planned)*
   * - ``physiomotion4d-4dct-reconstruction``
     - :class:`FourDCTReconstructionWorkflow` *(planned)*
   * - ``physiomotion4d-vtk-to-usd``
     - :class:`VTKToUSDWorkflow` *(planned)*
   * - ``physiomotion4d-brain-vessel-modeling``
     - :class:`BrainVesselModelingWorkflow` *(planned)*

This design allows:

* **CLI users**: Simple command-line interface
* **Developers**: Full programmatic control with Python API
* **Consistency**: Same functionality available both ways

Available Workflows
===================

WorkflowConvertHeartGatedCTToUSD
---------------------------------

Process 4D cardiac CT to animated USD models.

.. autoclass:: physiomotion4d.WorkflowConvertHeartGatedCTToUSD
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods**:
   * ``process()``: Execute complete workflow
   * ``load_time_series()``: Load and organize 4D data
   * ``segment_reference()``: Segment reference frame
   * ``register_time_series()``: Register all frames
   * ``generate_contours()``: Extract surface meshes
   * ``transform_contours()``: Apply transforms to meshes
   * ``create_usd_files()``: Generate USD outputs

**Example Usage**:

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   # Initialize workflow
   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=["cardiac_4d_ct.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="patient_001",
       registration_method='icon',
       number_of_registration_iterations=1
   )
   
   # Execute complete pipeline
   usd_files = workflow.process()
   
   print(f"Generated USD files: {usd_files}")

**CLI Equivalent**:

.. code-block:: bash

   physiomotion4d-heart-gated-ct cardiac_4d_ct.nrrd \
       --contrast \
       --output-dir ./results \
       --project-name patient_001 \
       --registration-method icon \
       --registration-iterations 1

See :doc:`../cli_scripts/heart_gated_ct` for CLI usage.

WorkflowRegisterHeartModelToPatient
------------------------------------

.. note::
   Planned for future release.

Register population heart models to patient images.

**Planned Usage**:

.. code-block:: python

   from physiomotion4d import WorkflowRegisterHeartModelToPatient
   
   workflow = WorkflowRegisterHeartModelToPatient(
       model_file="population_heart_model.vtk",
       patient_image="patient_ct.nrrd",
       output_directory="./results"
   )
   
   registered_model = workflow.process()

See :doc:`../cli_scripts/heart_model_to_patient` for planned CLI usage.

Common Workflow Patterns
========================

Step-Based Execution
--------------------

Workflows execute in discrete steps with checkpointing:

.. code-block:: python

   workflow = WorkflowConvertHeartGatedCTToUSD(...)
   
   # Execute step-by-step for debugging
   images = workflow.load_time_series()
   segmentation = workflow.segment_reference()
   transforms = workflow.register_time_series()
   contours = workflow.generate_contours()
   contours_4d = workflow.transform_contours(contours, transforms)
   usd_files = workflow.create_usd_files(contours_4d)
   
   # Or execute all at once
   usd_files = workflow.process()

Custom Configuration
--------------------

Configure workflow behavior programmatically:

.. code-block:: python

   workflow = WorkflowConvertHeartGatedCTToUSD(
       input_filenames=["cardiac.nrrd"],
       contrast_enhanced=True,
       output_directory="./results",
       project_name="custom_project",
       
       # Registration configuration
       registration_method='ants',  # Use ANTs instead of Icon
       number_of_registration_iterations=50,
       reference_image_filename="custom_ref.mha",
       
       # Segmentation configuration
       segmentation_method='ensemble',  # Use ensemble
       
       # Output configuration
       verbose=True  # Enable detailed logging
   )

Workflow Components
===================

Internal Processing Steps
-------------------------

Workflows coordinate these components:

1. **Data Loading**
   
   * Convert 4D to 3D time series
   * Organize temporal data
   * Select reference frame

2. **Segmentation**
   
   * Call segmentation classes (:doc:`segmentation`)
   * Generate anatomical labelmaps
   * Post-process segmentations

3. **Registration**
   
   * Call registration classes (:doc:`registration_images`)
   * Generate transform fields
   * Apply transforms

4. **Mesh Processing**
   
   * Extract surface contours
   * Apply transforms to meshes
   * Generate 4D VTK files

5. **USD Generation**
   
   * Call USD converters (:doc:`usd_generation`)
   * Apply anatomical materials
   * Create time-varying geometry

Component Selection
-------------------

Workflows select components based on parameters:

.. code-block:: python

   # In WorkflowConvertHeartGatedCTToUSD
   
   if self.registration_method == 'icon':
       from physiomotion4d import RegisterImagesIcon
       registrator = RegisterImagesIcon(...)
   elif self.registration_method == 'ants':
       from physiomotion4d import RegisterImagesANTs
       registrator = RegisterImagesANTs(...)
   
   # Execute registration
   transform = registrator.register(fixed_image, moving_image)

Extending Workflows
===================

Creating Custom Workflows
-------------------------

Create new workflows by inheriting from :class:`PhysioMotion4DBase`:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   from physiomotion4d import SegmentChestTotalSegmentator
   from physiomotion4d import RegisterImagesIcon
   from physiomotion4d import ConvertVTK4DToUSD
   
   class MyCustomWorkflow(PhysioMotion4DBase):
       """Custom medical imaging workflow."""
       
       def __init__(self, input_file, output_dir):
           super().__init__(verbose=True)
           self.input_file = input_file
           self.output_dir = output_dir
       
       def process(self):
           """Execute custom workflow."""
           # Step 1: Load data
           self.log("Loading data...")
           image = self.load_image(self.input_file)
           
           # Step 2: Custom processing
           self.log("Processing...")
           processed = self.custom_processing(image)
           
           # Step 3: Generate output
           self.log("Generating output...")
           output = self.create_output(processed)
           
           self.log("Workflow complete!", level="INFO")
           return output
       
       def custom_processing(self, image):
           """Custom processing logic."""
           # Implement your processing
           pass
       
       def create_output(self, data):
           """Create output files."""
           # Implement output generation
           pass

Modifying Existing Workflows
-----------------------------

Extend existing workflows by inheritance:

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   class EnhancedHeartWorkflow(WorkflowConvertHeartGatedCTToUSD):
       """Enhanced cardiac workflow with custom features."""
       
       def __init__(self, *args, custom_param=None, **kwargs):
           super().__init__(*args, **kwargs)
           self.custom_param = custom_param
       
       def segment_reference(self):
           """Override segmentation step."""
           self.log("Using custom segmentation...")
           
           # Call parent implementation
           segmentation = super().segment_reference()
           
           # Add custom post-processing
           segmentation = self.custom_post_process(segmentation)
           
           return segmentation
       
       def custom_post_process(self, segmentation):
           """Custom segmentation post-processing."""
           # Implement custom logic
           return segmentation

Creating CLI Scripts for Custom Workflows
------------------------------------------

Wrap your workflow in a CLI script following the pattern:

.. code-block:: python

   #!/usr/bin/env python
   """CLI for custom workflow."""
   
   import argparse
   from my_package import MyCustomWorkflow
   
   def main():
       parser = argparse.ArgumentParser(
           description="My custom medical imaging workflow"
       )
       parser.add_argument("input_file", help="Input image file")
       parser.add_argument("--output-dir", default="./results")
       parser.add_argument("--custom-param", type=int, default=10)
       
       args = parser.parse_args()
       
       # Initialize workflow
       workflow = MyCustomWorkflow(
           input_file=args.input_file,
           output_dir=args.output_dir,
           custom_param=args.custom_param
       )
       
       # Execute
       result = workflow.process()
       print(f"Complete! Output: {result}")
       return 0
   
   if __name__ == "__main__":
       exit(main())

Integration Patterns
====================

Batch Processing
----------------

Process multiple cases with workflows:

.. code-block:: python

   from pathlib import Path
   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   def batch_process(data_dir, output_root):
       """Batch process multiple patients."""
       
       # Find all input files
       input_files = list(Path(data_dir).glob("patient_*/cardiac.nrrd"))
       
       for input_file in input_files:
           patient_id = input_file.parent.name
           output_dir = Path(output_root) / patient_id
           
           print(f"Processing {patient_id}...")
           
           try:
               workflow = WorkflowConvertHeartGatedCTToUSD(
                   input_filenames=[str(input_file)],
                   contrast_enhanced=True,
                   output_directory=str(output_dir),
                   project_name=patient_id
               )
               
               result = workflow.process()
               print(f"  Success: {result}")
               
           except Exception as e:
               print(f"  Failed: {e}")
               continue
       
       print("Batch processing complete!")

Pipeline Integration
--------------------

Integrate workflows into larger pipelines:

.. code-block:: python

   def medical_imaging_pipeline(input_data):
       """Complete medical imaging pipeline."""
       
       # Step 1: Preprocessing
       preprocessed = preprocess_images(input_data)
       
       # Step 2: PhysioMotion4D workflow
       workflow = WorkflowConvertHeartGatedCTToUSD(
           input_filenames=[preprocessed],
           output_directory="./temp"
       )
       usd_files = workflow.process()
       
       # Step 3: Post-processing
       final_output = postprocess_usd(usd_files)
       
       # Step 4: Upload to cloud/database
       upload_results(final_output)
       
       return final_output

Parallel Execution
------------------

Process multiple workflows in parallel:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   
   def process_single(input_file, output_dir):
       """Process single case."""
       workflow = WorkflowConvertHeartGatedCTToUSD(
           input_filenames=[input_file],
           output_directory=output_dir
       )
       return workflow.process()
   
   def parallel_process(cases):
       """Process cases in parallel."""
       with ProcessPoolExecutor(max_workers=4) as executor:
           futures = [
               executor.submit(process_single, case['input'], case['output'])
               for case in cases
           ]
           
           results = [future.result() for future in futures]
       
       return results

Best Practices
==============

Workflow Design
---------------

1. **Keep workflows focused**: One workflow per clinical/research task
2. **Make steps resumable**: Save intermediate results
3. **Validate inputs early**: Check all parameters in ``__init__``
4. **Log progress clearly**: Use informative log messages
5. **Handle errors gracefully**: Provide useful error messages

Parameter Management
--------------------

1. **Use sensible defaults**: Most parameters should be optional
2. **Validate parameter combinations**: Check for incompatible settings
3. **Document parameters**: Clear docstrings with examples
4. **Support configuration files**: Allow loading from JSON/YAML

Performance
-----------

1. **Leverage GPU**: Use GPU-accelerated methods when available
2. **Cache expensive operations**: Save intermediate results
3. **Monitor memory**: Be aware of large image sizes
4. **Profile bottlenecks**: Identify and optimize slow steps

Troubleshooting
===============

Common Issues
-------------

**Workflow fails partway through**
   * Check log files for error messages
   * Verify intermediate files exist
   * Resume from checkpoint if supported

**Out of memory errors**
   * Reduce image resolution
   * Process fewer time points
   * Enable disk caching

**Slow performance**
   * Enable GPU acceleration
   * Check CPU/GPU utilization
   * Consider parallel processing for multiple cases

See Also
========

* :doc:`../cli_scripts/overview` - CLI usage for workflows
* :doc:`architecture` - Overall system architecture
* :doc:`extending` - Creating custom workflows
* :doc:`core` - Base class documentation
