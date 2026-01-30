"""
PhysioMotion4D - Medical imaging package for generating anatomic models with
    physiological motion.

This package converts 4D CT scans (particularly heart and lung gated CT data) into
    dynamic 3D models
for visualization in NVIDIA Omniverse. It provides comprehensive tools for image
    processing,
segmentation, registration, and USD file generation.

Main Components:
    - HeartGatedCTProcessor: Complete workflow processor for heart-gated CT data
    - Segmentation classes: Multiple AI-based chest segmentation implementations
    - Registration tools: Deep learning-based image registration
    - Transform utilities: Tools for image and contour transformations
    - USD tools: Utilities for Omniverse integration
    - PhysioMotion4DBase: Base class with standardized logging and debug settings
"""

__version__ = "2025.05.0"

# VTK to USD library
# VTK to USD library (new modular implementation)
from . import vtk_to_usd
from .contour_tools import ContourTools

# Data processing utilities
from .convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from .convert_vtk_to_usd import ConvertVTKToUSD

# Utility classes
from .image_tools import ImageTools

# Base classes
from .physiomotion4d_base import PhysioMotion4DBase
from .register_images_ants import RegisterImagesANTs

# Registration classes
from .register_images_base import RegisterImagesBase
from .register_images_icon import RegisterImagesICON
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP
from .register_models_icp_itk import RegisterModelsICPITK
from .register_models_pca import RegisterModelsPCA
from .register_time_series_images import RegisterTimeSeriesImages

# Segmentation classes
from .segment_chest_base import SegmentChestBase
from .segment_chest_ensemble import SegmentChestEnsemble
from .segment_chest_total_segmentator import SegmentChestTotalSegmentator
from .segment_chest_vista_3d import SegmentChestVista3D
from .segment_chest_vista_3d_nim import SegmentChestVista3DNIM
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools

# Core workflow processor
from .workflow_convert_heart_gated_ct_to_usd import WorkflowConvertHeartGatedCTToUSD
from .workflow_reconstruct_highres_4d_ct import WorkflowReconstructHighres4DCT
from .workflow_register_heart_model_to_patient import (
    WorkflowRegisterHeartModelToPatient,
)

__all__ = [
    # Workflow classes
    "WorkflowConvertHeartGatedCTToUSD",
    "WorkflowReconstructHighres4DCT",
    "WorkflowRegisterHeartModelToPatient",
    # Segmentation classes
    "SegmentChestBase",
    "SegmentChestEnsemble",
    "SegmentChestTotalSegmentator",
    "SegmentChestVista3D",
    "SegmentChestVista3DNIM",
    # Registration classes
    "RegisterImagesBase",
    "RegisterImagesICON",
    "RegisterImagesANTs",
    "RegisterTimeSeriesImages",
    "RegisterModelsPCA",
    "RegisterModelsICP",
    "RegisterModelsICPITK",
    "RegisterModelsDistanceMaps",
    # Base classes
    "PhysioMotion4DBase",
    # Utility classes
    "ImageTools",
    "TransformTools",
    "USDTools",
    "ContourTools",
    "USDAnatomyTools",
    # Data processing utilities
    "ConvertNRRD4DTo3D",
    "ConvertVTKToUSD",
    # VTK to USD library
    "vtk_to_usd",
]
