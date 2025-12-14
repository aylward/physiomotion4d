#!/usr/bin/env python
"""
Example usage of the ProcessHeartGatedCT class.

This script demonstrates how to use the ProcessHeartGatedCT to process
a 4D cardiac CT image through the complete workflow.
"""

import os
import argparse
from physiomotion4d import ProcessHeartGatedCT


def main():
    """Example usage of ProcessHeartGatedCT."""
    parser = argparse.ArgumentParser(description="Process 4D cardiac CT to dynamic USD")
    parser.add_argument("input_files", nargs='+', help="Path to 4D NRRD file or list of 3D NRRD files")
    parser.add_argument("--output-dir", default="./results",
        help="Output directory for files")
    parser.add_argument("--output-name", default="cardiac_model",
        help="Base name for output USD files")
    parser.add_argument("--project-name", 
        help="Project name for USD organization (defaults to output-name)")
    parser.add_argument("--contrast", action="store_true",
        help="Indicate if study is contrast enhanced")
    parser.add_argument("--reference-image",
        help="Path to reference image file (if not specified, uses 70% frame)")
    parser.add_argument("--registration_iterations", default=-1,
        help="Number of iterations to run registration")

    args = parser.parse_args()

    # Check if input files exist
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return 1

    # Initialize processor
    processor = ProcessHeartGatedCT(
        input_filenames=args.input_files,
        contrast_enhanced=args.contrast,
        output_directory=args.output_dir,
        project_name=args.output_name,
        reference_image_filename=args.reference_image,
        number_of_registration_iterations=int(args.registration_iterations)
    )

    try:
        # Execute complete workflow
        print("Starting Heart-gated CT processing...")
        final_usd = processor.process()

        print(f"\nProcessing completed successfully!")
        print(f"Final USD files created:")
        print(f"  Dynamic anatomy: {processor.project_name}.dynamic_anatomy_painted.usd")
        print(f"  Static anatomy: {processor.project_name}.static_anatomy_painted.usd") 
        print(f"  All anatomy: {processor.project_name}.all_anatomy_painted.usd")

    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


def example_basic_usage():
    """Example of basic usage with default settings."""
    
    # Example parameters
    input_file = "../../data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd"
    output_dir = "./results/heart_gated_ct"
    output_name = "example_cardiac"

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please download the example data first")
        return

    # Initialize processor with simple parameters
    processor = ProcessHeartGatedCT(
        input_filenames=[input_file],
        contrast_enhanced=True,
        output_directory=output_dir,
        output_usd_name=output_name
    )

    # Process complete workflow
    print("Processing cardiac CT data...")
    final_usd = processor.process()
    print(f"Completed! Final USD: {final_usd}")


def example_multiple_files():
    """Example of processing multiple 3D NRRD files as a time series."""
    
    # Example with multiple 3D files
    input_files = [
        "../../data/cardiac_series/frame_000.nrrd",
        "../../data/cardiac_series/frame_001.nrrd", 
        "../../data/cardiac_series/frame_002.nrrd",
        # ... more files
    ]
    
    # Check if files exist (skip if not)
    existing_files = [f for f in input_files if os.path.exists(f)]
    if not existing_files:
        print("No input files found for multiple file example")
        return
    
    processor = ProcessHeartGatedCT(
        input_filenames=existing_files,
        contrast_enhanced=False,  # Non-contrast study
        output_directory="./multi_file_results",
        output_usd_name="multi_cardiac",
        project_name="MultiFileCardiac"
    )
    
    final_usd = processor.process()
    print(f"Multi-file processing completed: {final_usd}")


def example_with_reference_image():
    """Example showing how to specify a custom reference image."""
    
    input_file = "../../data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd"
    reference_image = "../../data/reference/cardiac_reference.mha"
    
    if not os.path.exists(input_file):
        print("Input file not found for reference image example")
        return
        
    processor = ProcessHeartGatedCT(
        input_filenames=[input_file],
        contrast_enhanced=True,
        output_directory="./reference_example_results",
        project_name="reference_cardiac",
        reference_image_filename=reference_image if os.path.exists(reference_image) else None
    )
    
    final_usd = processor.process()
    print(f"Processing with reference image completed: {final_usd}")


def print_workflow_description():
    """Print description of the processing workflow."""
    print("Heart-gated CT Processing Workflow:")
    print("="*50)
    
    workflow_steps = [
        "1. Load time series data from 4D NRRD or multiple 3D files",
        "2. Segment reference image and all time frames", 
        "3. Register each frame to reference using different anatomy masks:",
        "   - Dynamic anatomy (heart, vessels, contrast)",
        "   - Static anatomy (lungs, bones, other tissues)",
        "   - All anatomy combined",
        "4. Generate contour meshes from reference segmentation",
        "5. Transform contours for each time frame using registration transforms",
        "6. Create and paint USD files for visualization in Omniverse"
    ]
    
    for step in workflow_steps:
        print(f"  {step}")
    
    print("\nOutput files:")
    print("  - <project_name>.dynamic_anatomy_painted.usd (moving structures)")
    print("  - <project_name>.static_anatomy_painted.usd (stationary structures)")
    print("  - <project_name>.all_anatomy_painted.usd (all structures)")
    
    print("\nUsage examples:")
    print("  # Basic usage with 4D file:")
    print("  python heart_gated_ct_example.py input_4d.nrrd --contrast")
    print("")
    print("  # With custom output directory and name:")  
    print("  python heart_gated_ct_example.py input.nrrd --output-dir /path/to/output --output-name my_heart")
    print("")
    print("  # Multiple 3D files:")
    print("  python heart_gated_ct_example.py frame_*.nrrd --output-name time_series")


if __name__ == "__main__":
    # Run the main command-line interface
    # Uncomment other examples to try them:
    
    main()
    
    # Other examples (uncomment to run):
    # example_basic_usage() 
    # example_multiple_files()
    # example_with_reference_image()
    # print_workflow_description()