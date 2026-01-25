#!/usr/bin/env python
"""
Test for VTK to USD PolyMesh conversion.

This test depends on test_contour_tools and uses the extracted contours
to test USD conversion functionality.
"""

import itk
import pytest
import pyvista as pv
from pxr import UsdGeom

from physiomotion4d.convert_vtk_4d_to_usd_polymesh import ConvertVTK4DToUSDPolyMesh


@pytest.mark.requires_data
@pytest.mark.slow
class TestConvertVTK4DToUSDPolyMesh:
    """Test suite for VTK to USD PolyMesh conversion."""

    @pytest.fixture(scope="class")
    def contour_meshes(self, contour_tools, segmentation_results, test_directories):
        """Extract or load contour meshes for USD conversion testing."""
        output_dir = test_directories["output"]
        contour_output_dir = output_dir / "contour_tools"

        # Check if contour files exist
        heart_contour_000 = contour_output_dir / "heart_contours_slice000.vtp"
        heart_contour_001 = contour_output_dir / "heart_contours_slice001.vtp"

        if not heart_contour_000.exists() or not heart_contour_001.exists():
            # Extract contours if they don't exist
            print("\nContour files not found, extracting them...")
            contour_output_dir.mkdir(parents=True, exist_ok=True)

            meshes = []
            for i, result in enumerate(segmentation_results):
                heart_mask = result["heart"]
                contours = contour_tools.extract_contours(heart_mask)
                meshes.append(contours)

                # Save contours
                output_file = contour_output_dir / f"heart_contours_slice{i:03d}.vtp"
                contours.save(str(output_file))

            return meshes
        # Load existing contours
        print("\nLoading existing contour files...")
        meshes = [
            pv.read(str(contour_output_dir / "heart_contours_slice000.vtp")),
            pv.read(str(contour_output_dir / "heart_contours_slice001.vtp")),
        ]
        return meshes

    def test_converter_initialization(self):
        """Test that ConvertVTK4DToUSDPolyMesh initializes correctly."""
        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="TestModel", input_polydata=[], mask_ids=None
        )

        assert converter is not None, "Converter not initialized"
        assert converter.data_basename == "TestModel", "Data basename not set correctly"

        print("\nConverter initialized successfully")

    def test_supports_mesh_type(self, contour_meshes):
        """Test that converter correctly identifies supported mesh types."""
        mesh = contour_meshes[0]

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="TestModel", input_polydata=[mesh], mask_ids=None
        )

        # PolyData should be supported
        assert converter.supports_mesh_type(mesh), "PolyData should be supported"

        print("\nMesh type support check passed")

    def test_convert_single_time_point(self, contour_meshes, test_directories):
        """Test converting a single time point to USD."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Use first time point only
        mesh = contour_meshes[0]

        print("\nConverting single time point to USD...")
        print(f"  Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartSingle", input_polydata=[mesh], mask_ids=None
        )

        output_file = usd_output_dir / "heart_single_time.usd"
        stage = converter.convert(str(output_file))

        # Verify USD stage was created
        assert stage is not None, "USD stage not created"
        assert output_file.exists(), f"USD file not created: {output_file}"

        # Verify stage contents (actual path includes /World prefix)
        prim = stage.GetPrimAtPath("/World/HeartSingle")
        assert prim.IsValid(), "Root prim not found at /World/HeartSingle"

        print("Single time point converted to USD")
        print(f"  Output: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")

    def test_convert_multiple_time_points(self, contour_meshes, test_directories):
        """Test converting multiple time points to USD."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        print("\nConverting multiple time points to USD...")
        print(f"  Time points: {len(contour_meshes)}")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartMulti", input_polydata=contour_meshes, mask_ids=None
        )

        output_file = usd_output_dir / "heart_multi_time.usd"
        stage = converter.convert(str(output_file))

        # Verify USD stage
        assert stage is not None, "USD stage not created"
        assert output_file.exists(), f"USD file not created: {output_file}"

        # Verify time samples (actual path includes /World prefix)
        prim = stage.GetPrimAtPath("/World/HeartMulti")
        assert prim.IsValid(), "Root prim not found at /World/HeartMulti"

        # Check that mesh exists (checking the Transform group)
        transform_path = "/World/HeartMulti/Transform_heart_multi_time"
        transform_prim = stage.GetPrimAtPath(transform_path)
        assert transform_prim.IsValid(), f"Transform not found at {transform_path}"

        print("Multiple time points converted to USD")
        print(f"  Output: {output_file}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")

    def test_convert_with_deformation(
        self, contour_tools, segmentation_results, test_directories
    ):
        """Test converting meshes with deformation magnitude."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Extract contours
        heart_mask = segmentation_results[0]["heart"]
        contours = contour_tools.extract_contours(heart_mask)

        # Add deformation magnitude (simulate with random values)
        import numpy as np

        deformation = np.random.uniform(0, 5, contours.n_points)
        contours["DeformationMagnitude"] = deformation

        print("\nConverting mesh with deformation magnitude...")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartDeformation", input_polydata=[contours], mask_ids=None
        )

        output_file = usd_output_dir / "heart_with_deformation.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Check that transform was created (actual path includes /World prefix)
        transform_path = "/World/HeartDeformation/Transform_heart_with_deformation"
        transform_prim = stage.GetPrimAtPath(transform_path)
        assert transform_prim.IsValid(), f"Transform not found at {transform_path}"

        print("Mesh with deformation converted to USD")
        print(f"  Output: {output_file}")

    def test_convert_with_colormap(self, contour_meshes, test_directories):
        """Test converting meshes with colormap visualization."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Add scalar field to mesh for colormapping
        import numpy as np

        mesh = contour_meshes[0]
        scalar_values = np.random.uniform(0, 100, mesh.n_points)
        mesh["pressure"] = scalar_values

        print("\nConverting mesh with colormap...")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartColormap", input_polydata=[mesh], mask_ids=None
        )

        # Set colormap
        converter.set_colormap(color_by_array="pressure", colormap="plasma")

        output_file = usd_output_dir / "heart_with_colormap.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Verify transform was created (actual path includes /World prefix)
        transform_path = "/World/HeartColormap/Transform_heart_with_colormap"
        transform_prim = stage.GetPrimAtPath(transform_path)
        assert transform_prim.IsValid(), f"Transform not found at {transform_path}"

        print("Mesh with colormap converted to USD")
        print("  Colormap: plasma")
        print(f"  Output: {output_file}")

    def test_convert_unstructured_grid_to_surface(self, test_directories):
        """Test converting UnstructuredGrid to surface mesh."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Create a simple UnstructuredGrid (cube)
        import numpy as np
        import pyvista as pv

        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        ).astype(np.float32)
        cells = [8, 0, 1, 2, 3, 4, 5, 6, 7]
        cell_types = [12]  # VTK_HEXAHEDRON

        ugrid = pv.UnstructuredGrid(cells, cell_types, points)

        print("\nConverting UnstructuredGrid to USD...")
        print(f"  Grid: {ugrid.n_points} points, {ugrid.n_cells} cells")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="CubeSurface",
            input_polydata=[ugrid],
            mask_ids=None,
            convert_to_surface=True,
        )

        output_file = usd_output_dir / "cube_surface.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        print("UnstructuredGrid converted to surface USD")
        print(f"  Output: {output_file}")

    def test_usd_file_structure(self, contour_meshes, test_directories):
        """Test the structure of generated USD file."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartStructure",
            input_polydata=[contour_meshes[0]],
            mask_ids=None,
        )

        output_file = usd_output_dir / "heart_structure_test.usd"
        stage = converter.convert(str(output_file))

        print("\nVerifying USD file structure...")

        # Check root prim (actual path includes /World prefix)
        root_prim = stage.GetPrimAtPath("/World/HeartStructure")
        assert root_prim.IsValid(), "Root prim not found at /World/HeartStructure"
        assert UsdGeom.Xform(root_prim), "Root should be an Xform"

        # Check transform/mesh structure
        transform_prim = stage.GetPrimAtPath(
            "/World/HeartStructure/Transform_heart_structure_test"
        )
        assert transform_prim.IsValid(), (
            "Transform prim not found at /World/HeartStructure/Transform_heart_structure_test"
        )

        print("USD file structure verified")
        print(f"  Root: {root_prim.GetPath()}")
        print(f"  Transform: {transform_prim.GetPath()}")

    def test_time_varying_topology(self, contour_meshes, test_directories):
        """Test handling of time-varying topology."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Modify one mesh to have different topology
        mesh1 = contour_meshes[0].copy()
        mesh2 = contour_meshes[1].copy()

        # Decimate second mesh to change topology
        mesh2 = mesh2.decimate(0.5)

        print("\nConverting meshes with varying topology...")
        print(f"  Mesh 1: {mesh1.n_points} points, {mesh1.n_cells} cells")
        print(f"  Mesh 2: {mesh2.n_points} points, {mesh2.n_cells} cells")

        converter = ConvertVTK4DToUSDPolyMesh(
            data_basename="HeartVarying", input_polydata=[mesh1, mesh2], mask_ids=None
        )

        output_file = usd_output_dir / "heart_varying_topology.usd"
        stage = converter.convert(str(output_file))

        assert stage is not None, "USD stage not created"
        assert output_file.exists(), "USD file not created"

        # Check for time-varying meshes (separate mesh prims)
        parent_path = "/HeartVarying/default"
        parent_prim = stage.GetPrimAtPath(parent_path)

        # Should have child meshes for each time step
        children = parent_prim.GetChildren() if parent_prim.IsValid() else []

        print("Time-varying topology handled")
        print(f"  Parent prim: {parent_path}")
        print(f"  Child prims: {len(children)}")
        print(f"  Output: {output_file}")

    def test_batch_conversion(
        self, contour_tools, segmentation_results, test_directories
    ):
        """Test converting multiple anatomy structures in batch."""
        output_dir = test_directories["output"]
        usd_output_dir = output_dir / "usd_polymesh"
        usd_output_dir.mkdir(exist_ok=True)

        # Extract contours from multiple anatomies
        anatomy_groups = ["lung", "heart"]
        meshes_dict = {}

        for group in anatomy_groups:
            mask = segmentation_results[0][group]
            mask_arr = itk.array_from_image(mask)

            import numpy as np

            if np.sum(mask_arr > 0) > 100:
                contours = contour_tools.extract_contours(mask)
                meshes_dict[group] = contours

        if len(meshes_dict) >= 2:
            print(f"\nConverting {len(meshes_dict)} anatomy structures...")

            # Convert each anatomy separately
            for anatomy, mesh in meshes_dict.items():
                converter = ConvertVTK4DToUSDPolyMesh(
                    data_basename=f"{anatomy.capitalize()}",
                    input_polydata=[mesh],
                    mask_ids=None,
                )

                output_file = usd_output_dir / f"{anatomy}_anatomy.usd"
                stage = converter.convert(str(output_file))

                assert stage is not None, f"USD stage not created for {anatomy}"
                assert output_file.exists(), f"USD file not created for {anatomy}"

                print(f"  {anatomy}: {output_file}")

            print("Batch conversion complete")
        else:
            pytest.skip("Not enough anatomies with sufficient voxels")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
