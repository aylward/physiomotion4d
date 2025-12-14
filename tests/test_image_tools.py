#!/usr/bin/env python
"""
Tests for ImageTools functionality.

Tests conversion between ITK and SimpleITK image formats for both
scalar and vector images.
"""

import itk
import numpy as np
import pytest
import SimpleITK as sitk

from physiomotion4d.image_tools import ImageTools


class TestImageTools:
    """Test suite for ImageTools conversions."""
    
    @pytest.fixture
    def image_tools(self):
        """Create ImageTools instance."""
        return ImageTools()
    
    def test_itk_to_sitk_scalar_image(self, image_tools):
        """Test conversion of scalar ITK image to SimpleITK."""
        # Create a simple 3D scalar ITK image
        size = [10, 20, 30]
        spacing = [1.0, 2.0, 3.0]
        origin = [0.0, 0.0, 0.0]
        
        # Create ITK image with known values
        ImageType = itk.Image[itk.F, 3]
        itk_image = ImageType.New()
        
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image.SetRegions(region)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.Allocate()
        
        # Fill with test pattern
        itk_image.FillBuffer(42.0)
        
        # Convert to SimpleITK
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image)
        
        # Verify metadata
        assert sitk_image.GetSize() == tuple(reversed(size))
        assert sitk_image.GetSpacing() == tuple(reversed(spacing))
        assert sitk_image.GetOrigin() == tuple(reversed(origin))
        
        # Verify data
        array_sitk = sitk.GetArrayFromImage(sitk_image)
        assert np.allclose(array_sitk, 42.0)
        
        print("✓ ITK to SimpleITK scalar conversion successful")
    
    def test_sitk_to_itk_scalar_image(self, image_tools):
        """Test conversion of scalar SimpleITK image to ITK."""
        # Create a simple 3D scalar SimpleITK image
        size = [10, 20, 30]
        spacing = [1.0, 2.0, 3.0]
        origin = [0.0, 0.0, 0.0]
        
        # Create SimpleITK image
        sitk_image = sitk.Image(size, sitk.sitkFloat32)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)
        
        # Fill with test pattern
        array = np.ones((size[2], size[1], size[0]), dtype=np.float32) * 99.0
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)
        
        # Convert to ITK
        itk_image = image_tools.convert_sitk_image_to_itk(sitk_image)
        
        # Verify metadata
        assert itk.size(itk_image) == tuple(reversed(size))
        assert itk.spacing(itk_image) == tuple(spacing)
        assert itk.origin(itk_image) == tuple(origin)
        
        # Verify data
        array_itk = itk.array_from_image(itk_image)
        assert np.allclose(array_itk, 99.0)
        
        print("✓ SimpleITK to ITK scalar conversion successful")
    
    def test_roundtrip_scalar_image(self, image_tools):
        """Test roundtrip conversion: ITK -> SimpleITK -> ITK."""
        # Create ITK image
        size = [15, 25, 35]
        spacing = [0.5, 1.5, 2.5]
        origin = [10.0, 20.0, 30.0]
        
        ImageType = itk.Image[itk.F, 3]
        itk_image_original = ImageType.New()
        
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image_original.SetRegions(region)
        itk_image_original.SetSpacing(spacing)
        itk_image_original.SetOrigin(origin)
        itk_image_original.Allocate()
        
        # Fill with test data
        array_original = np.random.rand(size[2], size[1], size[0]).astype(np.float32)
        itk.array_view_from_image(itk_image_original)[:] = array_original
        
        # Roundtrip conversion
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image_original)
        itk_image_final = image_tools.convert_sitk_image_to_itk(sitk_image)
        
        # Verify metadata preserved
        assert itk.size(itk_image_final) == tuple(size)
        assert np.allclose(itk.spacing(itk_image_final), spacing)
        assert np.allclose(itk.origin(itk_image_final), origin)
        
        # Verify data preserved
        array_final = itk.array_from_image(itk_image_final)
        assert np.allclose(array_original, array_final)
        
        print("✓ Roundtrip scalar conversion successful")
    
    def test_itk_to_sitk_vector_image(self, image_tools):
        """Test conversion of vector ITK image to SimpleITK."""
        # Create a 3D vector ITK image (like a displacement field)
        size = [8, 12, 16]
        spacing = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]
        
        # Create vector image with 3 components
        VectorImageType = itk.Image[itk.Vector[itk.F, 3], 3]
        itk_image = VectorImageType.New()
        
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image.SetRegions(region)
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin(origin)
        itk_image.Allocate()
        
        # Fill with test vector data
        array = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)
        itk.array_view_from_image(itk_image)[:] = array
        
        # Convert to SimpleITK
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image)
        
        # Verify it's a vector image
        assert sitk_image.GetNumberOfComponentsPerPixel() == 3
        
        # Verify metadata
        assert sitk_image.GetSize() == tuple(reversed(size))
        assert sitk_image.GetSpacing() == tuple(reversed(spacing))
        
        # Verify data
        array_sitk = sitk.GetArrayFromImage(sitk_image)
        assert np.allclose(array, array_sitk)
        
        print("✓ ITK to SimpleITK vector conversion successful")
    
    def test_sitk_to_itk_vector_image(self, image_tools):
        """Test conversion of vector SimpleITK image to ITK."""
        # Create a 3D vector SimpleITK image
        size = [8, 12, 16]
        spacing = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]
        
        # Create vector data
        array = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)
        
        # Create SimpleITK vector image
        sitk_image = sitk.GetImageFromArray(array, isVector=True)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)
        
        # Convert to ITK
        itk_image = image_tools.convert_sitk_image_to_itk(sitk_image)
        
        # Verify it's a vector image
        assert itk_image.GetNumberOfComponentsPerPixel() == 3
        
        # Verify metadata
        assert itk.size(itk_image) == tuple(reversed(size))
        assert itk.spacing(itk_image) == tuple(spacing)
        
        # Verify data
        array_itk = itk.array_from_image(itk_image)
        assert np.allclose(array, array_itk)
        
        print("✓ SimpleITK to ITK vector conversion successful")
    
    def test_roundtrip_vector_image(self, image_tools):
        """Test roundtrip conversion for vector images: ITK -> SimpleITK -> ITK."""
        # Create ITK vector image
        size = [10, 15, 20]
        spacing = [0.8, 1.2, 1.6]
        origin = [5.0, 10.0, 15.0]
        
        VectorImageType = itk.Image[itk.Vector[itk.F, 3], 3]
        itk_image_original = VectorImageType.New()
        
        region = itk.ImageRegion[3]()
        region.SetSize(size)
        itk_image_original.SetRegions(region)
        itk_image_original.SetSpacing(spacing)
        itk_image_original.SetOrigin(origin)
        itk_image_original.Allocate()
        
        # Fill with test vector data
        array_original = np.random.rand(size[2], size[1], size[0], 3).astype(np.float32)
        itk.array_view_from_image(itk_image_original)[:] = array_original
        
        # Roundtrip conversion
        sitk_image = image_tools.convert_itk_image_to_sitk(itk_image_original)
        itk_image_final = image_tools.convert_sitk_image_to_itk(sitk_image)
        
        # Verify metadata preserved
        assert itk.size(itk_image_final) == tuple(size)
        assert np.allclose(itk.spacing(itk_image_final), spacing)
        assert np.allclose(itk.origin(itk_image_final), origin)
        assert itk_image_final.GetNumberOfComponentsPerPixel() == 3
        
        # Verify data preserved
        array_final = itk.array_from_image(itk_image_final)
        assert np.allclose(array_original, array_final)
        
        print("✓ Roundtrip vector conversion successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

