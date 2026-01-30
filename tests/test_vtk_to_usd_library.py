#!/usr/bin/env python
"""
Tests for the vtk_to_usd library module.

This test suite validates the new modular vtk_to_usd library including:
- VTK file reading (VTP, VTK, VTU formats)
- Data structure conversions
- USD conversion
- Material handling
- Time-series support

Note: These tests require manually downloaded data:
- KCL-Heart-Model: Must be manually downloaded and placed in data/KCL-Heart-Model/
- CHOP-Valve4D: Must be manually downloaded and placed in data/CHOP-Valve4D/
"""

from pathlib import Path

import pytest
from pxr import UsdGeom, UsdShade

from physiomotion4d.vtk_to_usd import (
    ConversionSettings,
    MaterialData,
    VTKToUSDConverter,
    read_vtk_file,
)


# Helper to get data paths
def get_data_dir():
    """Get the data directory path."""
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    return project_root / "data"


def check_kcl_heart_data():
    """Check if KCL Heart Model data is available."""
    data_dir = get_data_dir() / "KCL-Heart-Model"
    vtp_file = data_dir / "average_surface.vtp"
    vtk_file = data_dir / "average_mesh.vtk"
    return vtp_file.exists() and vtk_file.exists()


def check_valve4d_data():
    """Check if CHOP Valve4D data is available."""
    data_dir = get_data_dir() / "CHOP-Valve4D"
    alterra_dir = data_dir / "Alterra"
    return alterra_dir.exists() and any(alterra_dir.glob("*.vtk"))


@pytest.mark.requires_data
class TestVTKReader:
    """Test VTK file reading capabilities."""

    def test_read_vtp_file(self):
        """Test reading VTP (PolyData) files."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        assert vtp_file.exists(), f"VTP file not found: {vtp_file}"

        # Read the file
        mesh_data = read_vtk_file(vtp_file)

        # Verify mesh data structure
        assert mesh_data is not None
        assert mesh_data.points is not None
        assert len(mesh_data.points) > 0
        assert mesh_data.face_vertex_counts is not None
        assert mesh_data.face_vertex_indices is not None

        print(f"\n✓ Read VTP file: {vtp_file.name}")
        print(f"  Points: {len(mesh_data.points):,}")
        print(f"  Faces: {len(mesh_data.face_vertex_counts):,}")
        print(f"  Data arrays: {len(mesh_data.generic_arrays)}")

    def test_read_legacy_vtk_file(self):
        """Test reading legacy VTK files."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtk_file = data_dir / "average_mesh.vtk"

        assert vtk_file.exists(), f"VTK file not found: {vtk_file}"

        # Read the file with surface extraction
        mesh_data = read_vtk_file(vtk_file, extract_surface=True)

        # Verify mesh data structure
        assert mesh_data is not None
        assert mesh_data.points is not None
        assert len(mesh_data.points) > 0
        assert mesh_data.face_vertex_counts is not None
        assert mesh_data.face_vertex_indices is not None

        print(f"\n✓ Read legacy VTK file: {vtk_file.name}")
        print(f"  Points: {len(mesh_data.points):,}")
        print(f"  Faces: {len(mesh_data.face_vertex_counts):,}")
        print(f"  Data arrays: {len(mesh_data.generic_arrays)}")

    def test_generic_arrays_preserved(self):
        """Test that generic data arrays are preserved during reading."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        mesh_data = read_vtk_file(vtp_file)

        # Verify generic arrays
        assert len(mesh_data.generic_arrays) > 0, "No data arrays found"

        # Check array structure
        for array in mesh_data.generic_arrays:
            assert array.name is not None
            assert array.data is not None
            assert array.num_components > 0
            assert array.interpolation in ["vertex", "uniform", "constant"]

        print("\n✓ Generic arrays preserved:")
        for array in mesh_data.generic_arrays:
            print(
                f"  - {array.name}: {array.num_components} components, {len(array.data):,} values"
            )


@pytest.mark.requires_data
class TestVTKToUSDConversion:
    """Test VTK to USD conversion capabilities."""

    def test_single_file_conversion(self, test_directories):
        """Test converting a single VTK file to USD."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get test data
        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Convert to USD
        output_usd = output_dir / "heart_surface.usd"
        converter = VTKToUSDConverter()
        stage = converter.convert_file(
            vtp_file,
            output_usd,
            mesh_name="HeartSurface",
        )

        # Verify USD file
        assert output_usd.exists()
        assert stage is not None

        # Check mesh exists in stage
        mesh_prim = stage.GetPrimAtPath("/World/Meshes/HeartSurface")
        assert mesh_prim.IsValid()
        assert mesh_prim.IsA(UsdGeom.Mesh)

        # Check mesh has geometry
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        assert len(points) > 0

        print("\n✓ Converted single file to USD")
        print(f"  Input: {vtp_file.name}")
        print(f"  Output: {output_usd}")
        print(f"  Points: {len(points):,}")

    def test_conversion_with_material(self, test_directories):
        """Test conversion with custom material."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Create custom material
        material = MaterialData(
            name="heart_tissue",
            diffuse_color=(0.9, 0.3, 0.3),
            roughness=0.4,
            metallic=0.0,
        )

        # Convert with material
        output_usd = output_dir / "heart_with_material.usd"
        converter = VTKToUSDConverter()
        stage = converter.convert_file(
            vtp_file,
            output_usd,
            mesh_name="HeartSurface",
            material=material,
        )

        # Verify material exists
        material_path = f"/World/Looks/{material.name}"
        material_prim = stage.GetPrimAtPath(material_path)
        assert material_prim.IsValid()
        assert material_prim.IsA(UsdShade.Material)

        # Verify material is bound to mesh
        mesh_prim = stage.GetPrimAtPath("/World/Meshes/HeartSurface")
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        bound_material = binding_api.ComputeBoundMaterial()[0]
        assert bound_material.GetPrim().IsValid()

        print("\n✓ Converted with custom material")
        print(f"  Material: {material.name}")
        print(f"  Color: {material.diffuse_color}")

    def test_conversion_settings(self, test_directories):
        """Test conversion with custom settings."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Create custom settings
        settings = ConversionSettings(
            triangulate_meshes=True,
            compute_normals=True,
            preserve_point_arrays=True,
            preserve_cell_arrays=True,
            meters_per_unit=0.001,  # mm to meters
            up_axis="Y",
        )

        # Convert with settings
        output_usd = output_dir / "heart_custom_settings.usd"
        converter = VTKToUSDConverter(settings)
        stage = converter.convert_file(vtp_file, output_usd)

        # Verify stage metadata
        assert UsdGeom.GetStageMetersPerUnit(stage) == 0.001
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y

        print("\n✓ Converted with custom settings")
        print(f"  Meters per unit: {settings.meters_per_unit}")
        print(f"  Up axis: {settings.up_axis}")
        print(f"  Compute normals: {settings.compute_normals}")

    def test_primvar_preservation(self, test_directories):
        """Test that VTK data arrays are preserved as USD primvars."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Read to check arrays
        mesh_data = read_vtk_file(vtp_file)
        array_names = [arr.name for arr in mesh_data.generic_arrays]

        # Convert to USD
        output_usd = output_dir / "heart_with_primvars.usd"
        converter = VTKToUSDConverter()
        stage = converter.convert_file(vtp_file, output_usd)

        # Check primvars exist
        mesh_prim = stage.GetPrimAtPath("/World/Meshes/Mesh")
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvars = primvars_api.GetPrimvars()

        primvar_names = [pv.GetPrimvarName() for pv in primvars]

        # Verify at least some arrays were converted to primvars
        assert len(primvar_names) > 0

        print("\n✓ Primvars preserved:")
        print(f"  Source arrays: {len(array_names)}")
        print(f"  USD primvars: {len(primvar_names)}")
        for name in primvar_names[:5]:  # Show first 5
            print(f"    - {name}")


@pytest.mark.requires_data
class TestTimeSeriesConversion:
    """Test time-series conversion capabilities."""

    def test_time_series_conversion(self, test_directories):
        """Test converting multiple VTK files as time series."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Use same file multiple times to simulate time series
        vtk_files = [vtp_file] * 3
        time_codes = [0.0, 1.0, 2.0]

        # Convert time series
        output_usd = output_dir / "heart_time_series.usd"
        converter = VTKToUSDConverter()
        stage = converter.convert_sequence(
            vtk_files=vtk_files,
            output_usd=output_usd,
            time_codes=time_codes,
        )

        # Verify time range
        assert stage.GetStartTimeCode() == 0.0
        assert stage.GetEndTimeCode() == 2.0

        # Verify mesh has time samples
        mesh_prim = stage.GetPrimAtPath("/World/Meshes/Mesh")
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()

        # Check time samples exist
        time_samples = points_attr.GetTimeSamples()
        assert len(time_samples) == 3
        assert time_samples == time_codes

        print("\n✓ Converted time series")
        print(f"  Frames: {len(vtk_files)}")
        print(f"  Time codes: {time_codes}")
        print(
            f"  Stage time range: {stage.GetStartTimeCode()} - {stage.GetEndTimeCode()}"
        )


@pytest.mark.slow
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_end_to_end_conversion(self, test_directories):
        """Test complete conversion workflow with all features."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtp_file = data_dir / "average_surface.vtp"

        # Configure everything
        settings = ConversionSettings(
            triangulate_meshes=True,
            compute_normals=True,
            preserve_point_arrays=True,
            meters_per_unit=0.001,
            times_per_second=24.0,
        )

        material = MaterialData(
            name="cardiac_muscle",
            diffuse_color=(0.85, 0.2, 0.2),
            roughness=0.5,
            metallic=0.0,
        )

        # Convert
        output_usd = output_dir / "heart_complete.usd"
        converter = VTKToUSDConverter(settings)
        stage = converter.convert_file(
            vtp_file,
            output_usd,
            mesh_name="CardiacModel",
            material=material,
        )

        # Comprehensive verification
        assert output_usd.exists()
        assert stage is not None

        # Check structure
        mesh_prim = stage.GetPrimAtPath("/World/Meshes/CardiacModel")
        assert mesh_prim.IsValid()

        # Check geometry
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        assert len(points) > 0

        # Check material
        material_prim = stage.GetPrimAtPath(f"/World/Looks/{material.name}")
        assert material_prim.IsValid()

        # Check primvars
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvars = primvars_api.GetPrimvars()
        assert len(primvars) > 0

        print("\n✓ End-to-end conversion complete")
        print(f"  Output: {output_usd}")
        print(f"  Size: {output_usd.stat().st_size / 1024:.1f} KB")
        print(f"  Points: {len(points):,}")
        print(f"  Primvars: {len(primvars)}")
