"""
This module contains the USDTools class for manipulating USD objects and files.

This module provides utilities for working with Universal Scene Description (USD)
files in the context of medical visualization. It includes functions for merging
USD files, arranging objects in grids, computing bounding boxes, and preserving
materials and animations for visualization in NVIDIA Omniverse.

The tools are specifically designed for medical imaging workflows where multiple
anatomical structures need to be organized and visualized together.
"""

import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils


class USDTools:
    """
    Utilities for manipulating Universal Scene Description (USD) files.

    This class provides tools for working with USD files in medical visualization
    contexts, including merging multiple USD files, arranging objects in spatial
    grids, computing bounding boxes, and preserving materials and animations.

    USD (Universal Scene Description) is the foundation for 3D content in
    NVIDIA Omniverse and other modern 3D pipelines. This class facilitates
    the creation of complex medical visualizations by organizing anatomical
    structures from multiple sources.

    Key capabilities:
    - Merge multiple USD files while preserving hierarchy and materials
    - Arrange objects in spatial grids for comparison or overview
    - Compute bounding boxes for spatial layout
    - Preserve time-varying animation data
    - Handle material bindings and shader networks

    The class is designed to work with USD files generated from medical
    imaging data, particularly anatomical structures extracted from CT
    and MR images.

    Example:
        >>> usd_tools = USDTools()
        >>> # Merge multiple anatomical USD files
        >>> usd_tools.merge_usd_files(
        ...     "combined_anatomy.usd",
        ...     ["heart.usd", "lungs.usd", "bones.usd"]
        ... )
        >>> # Create grid arrangement for comparison
        >>> usd_tools.save_usd_file_arrangement(
        ...     "comparison_grid.usd",
        ...     ["patient1.usd", "patient2.usd", "patient3.usd"]
        ... )
    """

    def __init__(self):
        """Initialize the USDTools class.

        No parameters are required for initialization as all methods
        operate on provided USD files and stages.
        """
        pass

    def get_subtree_bounding_box(
        self, prim: UsdGeom.Xform
    ) -> tuple[Gf.Vec3f, Gf.Vec3f]:
        """
        Compute the axis-aligned bounding box of a USD primitive subtree.

        Recursively traverses a USD primitive hierarchy and computes the
        combined bounding box of all mesh geometry within the subtree.
        This is useful for spatial layout and positioning operations.

        Args:
            prim (UsdGeom.Xform): The root primitive of the subtree to analyze.
                Should be a UsdGeom.Xform or similar transformable primitive

        Returns:
            tuple[Gf.Vec3f, Gf.Vec3f]: Tuple containing:
                - bbox_min: Minimum corner of the bounding box
                - bbox_max: Maximum corner of the bounding box

        Example:
            >>> stage = Usd.Stage.Open("anatomy.usd")
            >>> heart_prim = stage.GetPrimAtPath("/World/Heart")
            >>> bbox_min, bbox_max = usd_tools.get_subtree_bounding_box(heart_prim)
            >>> center = (bbox_min + bbox_max) / 2
        """
        first_bbox = True
        bbox_min = np.array([0, 0, 0])
        bbox_max = np.array([0, 0, 0])

        def traverse_prim(current_prim):
            nonlocal bbox_min, bbox_max, first_bbox
            if current_prim.IsA(UsdGeom.Mesh):
                bbox = UsdGeom.Boundable.ComputeExtentFromPlugins(
                    UsdGeom.Boundable(current_prim), Usd.TimeCode.Default()
                )
                if first_bbox:
                    bbox_min = bbox[0]
                    bbox_max = bbox[1]
                    first_bbox = False
                else:
                    bbox_min = np.minimum(bbox_min, bbox[0])
                    bbox_max = np.maximum(bbox_max, bbox[1])

            # Recursively traverse all children
            for child in current_prim.GetAllChildren():
                traverse_prim(child)

        traverse_prim(prim)

        return bbox_min, bbox_max

    def save_usd_file_arrangement(self, new_stage_name: str, usd_file_names: list[str]):
        """
        Create a spatial grid arrangement of objects from multiple USD files.

        Takes a list of USD files and arranges them in a regular grid pattern
        for comparison or overview visualization. Each USD file is referenced
        into the new stage and positioned to avoid overlap. This is useful
        for comparing anatomical structures from different patients or time
        points.

        The grid layout is automatically computed based on the number of
        input files, creating approximately square arrangements. Objects
        are centered at their computed positions and spaced to avoid overlap.

        Args:
            new_stage_name (str): Path for the output USD file containing
                the arranged objects
            usd_file_names (list[str]): List of paths to USD files to arrange.
                Each file should contain anatomical structures under /World

        Note:
            The method preserves material bindings from the source files and
            applies spatial transforms to position objects in the grid.
            The first USD file in the list is used as the template for the
            new stage structure.

        Example:
            >>> # Create comparison grid of cardiac models
            >>> usd_tools.save_usd_file_arrangement(
            ...     "cardiac_comparison.usd",
            ...     ["patient_001_heart.usd", "patient_002_heart.usd",
            ...      "patient_003_heart.usd", "patient_004_heart.usd"]
            ... )
        """
        new_stage = Usd.Stage.Open(usd_file_names[0])

        n_objects = len(usd_file_names)
        n_rows = int(np.floor(np.sqrt(n_objects)))
        n_cols = int(np.ceil(n_objects / n_rows))
        print(f"n_rows: {n_rows}, n_cols: {n_cols}")
        x_spacing = 400.0
        y_spacing = 400.0
        x_offset = -x_spacing * (n_cols - 1) / 2
        y_offset = -y_spacing * (n_rows - 1) / 2

        for i, usd_file_name in enumerate(usd_file_names):

            source_stage = Usd.Stage.Open(usd_file_name, Usd.Stage.LoadAll)

            source_root = source_stage.GetPrimAtPath("/World")
            children = source_root.GetChildren()
            for child in children:
                print(f"Copying {usd_file_name}:{child.GetPrimPath()}")
                new_stage.DefinePrim(child.GetPrimPath()).GetReferences().AddReference(
                    assetPath=usd_file_name,
                    primPath=child.GetPrimPath(),
                )
                # Apply translation to t
                for grandchild in child.GetAllChildren():
                    print(f"   Bounding box of {grandchild.GetPrimPath()}")
                    bbox_min, bbox_max = self.get_subtree_bounding_box(grandchild)
                    bbox_center = (bbox_min + bbox_max) / 2
                    print(f"   Bounding box center: {bbox_center}")

                    xform = UsdGeom.Xformable(grandchild)
                    if not xform.GetOrderedXformOps():
                        xform.AddTranslateOp()
                    xform_op = xform.GetOrderedXformOps()[
                        -1
                    ]  # Get the last transform op

                    # Calculate translation to position object center at grid position
                    grid_x = (i % n_cols) * x_spacing + x_offset
                    grid_y = (i // n_cols) * y_spacing + y_offset
                    translate = (
                        grid_x - bbox_center[0],
                        grid_y - bbox_center[1],
                        -bbox_center[2],
                    )
                    print(f"   Translating {grandchild.GetPrimPath()} to {translate}")
                    xform_op.Set(translate, Usd.TimeCode.Default())

            for prim in source_stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    bindingAPI = UsdShade.MaterialBindingAPI(prim)
                    mesh_material = bindingAPI.ComputeBoundMaterial()
                    if bool(mesh_material):
                        material_path = (
                            str(mesh_material[0].GetPath())
                            if isinstance(mesh_material, tuple)
                            and len(mesh_material) > 0
                            else str(mesh_material.GetPath())
                        )
                        print(
                            f"   Mesh {prim.GetPrimPath()} has material {material_path}"
                        )
                        new_prim = new_stage.GetPrimAtPath(prim.GetPrimPath())
                        material = UsdShade.Material.Get(new_stage, material_path)
                        if new_prim is not None and new_prim.IsValid():
                            binding_api = UsdShade.MaterialBindingAPI.Apply(new_prim)
                            binding_api.Bind(material)
                        else:
                            print(
                                f"      Cannot bind. No new prim found for {prim.GetPrimPath()}"
                            )

        print("Exporting stage...")
        new_stage.Export(new_stage_name)

    def merge_usd_files(self, output_filename: str, input_filenames_list: list[str]):
        """
        Merge multiple USD files into a single comprehensive USD file.

        Combines multiple USD files while preserving all essential data
        including object hierarchies, transforms, materials, shaders,
        and time-varying animation data. This is useful for creating
        complete anatomical scenes from individually processed structures.

        The merging process:
        1. Creates a new USD stage with proper metadata
        2. Copies all primitive hierarchies from input files
        3. Preserves all attributes including time-sampled data
        4. Maintains material bindings and shader networks
        5. Handles coordinate system and units consistently

        Args:
            output_filename (str): Path for the merged output USD file.
                Should have .usd or .usda extension
            input_filenames_list (list[str]): List of input USD file paths
                to merge. Each should be a valid USD file with compatible
                coordinate systems

        Note:
            The merged file uses meters as the base unit (0.01 scale factor)
            and Y-up axis orientation, which are standard for Omniverse.
            Time-varying data (animations) are preserved across all time samples.

        Example:
            >>> # Merge anatomical components into complete scene
            >>> usd_tools.merge_usd_files(
            ...     "complete_anatomy.usd",
            ...     ["heart_dynamic.usd", "lungs_static.usd", "skeleton.usd"]
            ... )
        """
        # Create new stage with meters as units (standard USD configuration)
        stage = Usd.Stage.CreateNew(output_filename)
        stage.SetMetadata('metersPerUnit', 0.01)
        stage.SetMetadata('upAxis', 'Y')

        # Define root prim for organization
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)

        # Track time range across all input files for stage metadata
        global_start_time = float('inf')
        global_end_time = float('-inf')
        time_codes_per_second = None
        frames_per_second = None

        for i, input_path in enumerate(input_filenames_list):
            # Open input stage with time-sampling enabled
            input_stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)

            # Track time range from this input file
            start_time = input_stage.GetStartTimeCode()
            end_time = input_stage.GetEndTimeCode()
            global_start_time = min(global_start_time, start_time)
            global_end_time = max(global_end_time, end_time)

            # Capture time codes per second from first file
            if time_codes_per_second is None:
                time_codes_per_second = input_stage.GetTimeCodesPerSecond()
                frames_per_second = input_stage.GetFramesPerSecond()

            # Copy all root prims from input
            for prim in input_stage.GetPseudoRoot().GetAllChildren():
                new_path = "/" + prim.GetName()
                print(f"Copying {prim.GetPrimPath()} to {new_path}")

                # Recursively copy prim hierarchy with all attributes and time samples
                def _copy_prim(src_prim, target_path):
                    # Create new prim with same type
                    new_prim = stage.DefinePrim(target_path, src_prim.GetTypeName())

                    # Copy properties and metadata
                    for attr in src_prim.GetAttributes():
                        if attr.GetName() == "deformationMagnitude":
                            continue

                        new_attr = new_prim.CreateAttribute(
                            attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()
                        )

                        # Copy default value if it exists
                        if attr.HasValue():
                            new_attr.Set(attr.Get())

                        # Copy all time samples for time-varying attributes
                        time_samples = attr.GetTimeSamples()
                        if time_samples:
                            for time in time_samples:
                                new_attr.Set(attr.Get(time), time)

                        # Copy attribute connections (critical for material shader networks)
                        connections = attr.GetConnections()
                        if connections:
                            new_attr.SetConnections(connections)

                    # Copy relationships (important for material connections)
                    for rel in src_prim.GetRelationships():
                        new_rel = new_prim.CreateRelationship(
                            rel.GetName(), custom=rel.IsCustom()
                        )
                        new_rel.SetTargets(rel.GetTargets())

                    # Copy transforms if applicable
                    if src_prim.IsA(UsdGeom.Xformable):
                        xform = UsdGeom.Xformable(new_prim)
                        xform_op = xform.GetTransformOp()
                        if src_prim.HasAttribute("xformOp:transform"):
                            src_xform = UsdGeom.Xformable(src_prim)
                            xform_op.Set(src_xform.GetLocalTransformation())

                    # Recurse through children
                    for child in src_prim.GetChildren():
                        child_path = f"{target_path}/{child.GetName()}"
                        _copy_prim(child, child_path)

                _copy_prim(prim, new_path)

            # Copy material bindings from source stage to target stage
            for prim in input_stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    bindingAPI = UsdShade.MaterialBindingAPI(prim)
                    mesh_material = bindingAPI.ComputeBoundMaterial()
                    if bool(mesh_material):
                        # Get material path from source
                        material_path = (
                            str(mesh_material[0].GetPath())
                            if isinstance(mesh_material, tuple)
                            and len(mesh_material) > 0
                            else str(mesh_material.GetPath())
                        )
                        print(
                            f"   Binding material {material_path} to {prim.GetPrimPath()}"
                        )
                        # Get corresponding mesh prim and material in target stage
                        new_prim = stage.GetPrimAtPath(prim.GetPrimPath())
                        material = UsdShade.Material.Get(stage, material_path)
                        if new_prim is not None and new_prim.IsValid():
                            if material and material.GetPrim().IsValid():
                                binding_api = UsdShade.MaterialBindingAPI.Apply(new_prim)
                                binding_api.Bind(material)
                            else:
                                print(
                                    f"      Warning: Material not found at {material_path} in target stage"
                                )
                        else:
                            print(
                                f"      Warning: Cannot bind material. No mesh prim found at {prim.GetPrimPath()}"
                            )

        # Set stage time range metadata for animation playback
        if global_start_time != float('inf') and global_end_time != float('-inf'):
            stage.SetStartTimeCode(global_start_time)
            stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                stage.SetTimeCodesPerSecond(time_codes_per_second)
            if frames_per_second is not None:
                stage.SetFramesPerSecond(frames_per_second)
            print(f"\nSet stage time range: {global_start_time} to {global_end_time}")
            print(f"Time codes per second: {time_codes_per_second}, Frames per second: {frames_per_second}")

        # Save with USDA format
        # stage.GetRootLayer().Export(output_path, args=['--usdFormat', 'usda'])
        stage.Export(output_filename)

    def merge_usd_files_flattened(
        self, output_filename: str, input_filenames_list: list[str]
    ):
        """
        Merge multiple USD files using references and flattening.

        This method uses USD's native composition system (references) and then flattens
        the result into a self-contained file. This approach is simpler (~50 lines vs
        ~150 lines) and leverages USD's built-in composition engine.

        The method properly preserves:
            - All materials and MDL shader networks
            - Time-varying animation data with correct time codes
            - Material bindings to geometry
            - Stage metadata (TimeCodesPerSecond, time range, etc.)

        Args:
            output_filename (str): Path for the merged output USD file.
                Should have .usd or .usda extension
            input_filenames_list (list[str]): List of input USD file paths
                to merge. Each should be a valid USD file with compatible
                coordinate systems

        Comparison to merge_usd_files():
            - **merge_usd_files()**: More control, can skip specific attributes
            - **merge_usd_files_flattened()**: Simpler, faster, USD-native approach

        Both methods produce equivalent results for most use cases. Use the flattened
        method unless you need fine-grained control over what gets copied.

        Example:
            >>> usd_tools = USDTools()
            >>> usd_tools.merge_usd_files_flattened(
            ...     "complete_anatomy.usd",
            ...     ["heart_dynamic.usd", "lungs_static.usd"]
            ... )
        """
        # Create temporary in-memory stage for composition
        temp_stage = Usd.Stage.CreateInMemory()

        # Set standard metadata (meters and Y-up for Omniverse)
        temp_stage.SetMetadata('metersPerUnit', 0.01)
        temp_stage.SetMetadata('upAxis', 'Y')

        # Define root prim for organization
        root_prim = temp_stage.DefinePrim("/World", "Xform")
        temp_stage.SetDefaultPrim(root_prim)

        # Track time range across all input files for stage metadata
        global_start_time = float('inf')
        global_end_time = float('-inf')
        time_codes_per_second = None
        frames_per_second = None

        # Add references to all input files
        for input_path in input_filenames_list:
            print(f"Referencing {input_path}")
            input_stage = Usd.Stage.Open(input_path, Usd.Stage.LoadAll)

            # Track time range from this input file
            start_time = input_stage.GetStartTimeCode()
            end_time = input_stage.GetEndTimeCode()
            global_start_time = min(global_start_time, start_time)
            global_end_time = max(global_end_time, end_time)

            # Capture time codes per second from first file
            if time_codes_per_second is None:
                time_codes_per_second = input_stage.GetTimeCodesPerSecond()
                frames_per_second = input_stage.GetFramesPerSecond()

            # Reference each top-level prim from the input file
            for prim in input_stage.GetPseudoRoot().GetAllChildren():
                new_path = "/" + prim.GetName()
                print(f"  Adding reference: {prim.GetPrimPath()} -> {new_path}")

                # Create prim and add reference to source file
                temp_stage.DefinePrim(new_path).GetReferences().AddReference(
                    assetPath=input_path,
                    primPath=prim.GetPrimPath()
                )

        # Set time range metadata on temporary stage before flattening
        if global_start_time != float('inf') and global_end_time != float('-inf'):
            temp_stage.SetStartTimeCode(global_start_time)
            temp_stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                temp_stage.SetTimeCodesPerSecond(time_codes_per_second)
            if frames_per_second is not None:
                temp_stage.SetFramesPerSecond(frames_per_second)
            print(f"Time range: {global_start_time} to {global_end_time}")
            print(f"Time codes per second: {time_codes_per_second}, Frames per second: {frames_per_second}")

        # Flatten the composed stage into a single layer
        # This resolves all references and bakes everything into one file
        print("Flattening composed stage...")
        flattened_layer = temp_stage.Flatten()

        # Create output stage from flattened layer
        output_stage = Usd.Stage.Open(flattened_layer)

        # Set time metadata on the output stage (must be done AFTER flattening)
        # This is critical - the flattened layer doesn't inherit metadata from temp_stage
        if global_start_time != float('inf') and global_end_time != float('-inf'):
            output_stage.SetStartTimeCode(global_start_time)
            output_stage.SetEndTimeCode(global_end_time)
            if time_codes_per_second is not None:
                output_stage.SetTimeCodesPerSecond(time_codes_per_second)
                print(f"Set output TimeCodesPerSecond: {time_codes_per_second}")
            if frames_per_second is not None:
                output_stage.SetFramesPerSecond(frames_per_second)
                print(f"Set output FramesPerSecond: {frames_per_second}")

        # Export the flattened layer with corrected metadata
        print(f"Exporting to {output_filename}")
        output_stage.Export(output_filename)
