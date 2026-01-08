# Using ITK Transforms as Initial Transforms in ANTs Registration

## Overview

The `RegisterImagesANTs` class now supports passing any ITK transform as an initial transform to ANTs registration functions (`ants.registration()` and `ants.label_image_registration()`). This feature enables you to:

- Initialize registration with results from previous registrations
- Use transforms from other registration frameworks (e.g., ICON)
- Compose multiple transforms as a starting point
- Refine existing transforms with ANTs algorithms

## Implementation

### New Method: `itk_transform_to_ants_transform()`

This method converts any ITK transform to an ANTsPy transform object without requiring file I/O.

**Signature:**
```python
def itk_transform_to_ants_transform(
    self, 
    itk_tfm: itk.Transform, 
    reference_image: itk.Image
) -> ants.core.ANTsTransform
```

**Process:**
1. Uses `TransformTools.convert_transform_to_displacement_field()` to convert any ITK transform to a dense displacement field
2. Converts the displacement field from ITK to ANTs format using `_itk_to_ants_image()`
3. Creates an ANTsPy transform object using `ants.transform_from_displacement_field()`
4. Returns the ANTsPy transform object (no disk I/O)

### Updated Method: `register()`

The registration method now accepts an `initial_forward_transform` parameter:

**New Parameter:**
- `initial_forward_transform` (itk.Transform, optional): Initial transform from moving to fixed space. Can be any ITK transform type.

**Supported Transform Types:**
- `itk.AffineTransform`
- `itk.VersorRigid3DTransform`
- `itk.Euler3DTransform`
- `itk.DisplacementFieldTransform`
- `itk.CompositeTransform`
- Any other ITK transform

## Usage Examples

### Basic Usage

```python
from physiomotion4d.register_images_ants import RegisterImagesANTs
import itk

# Load images
fixed_image = itk.imread("fixed.mha")
moving_image = itk.imread("moving.mha")

# Create initial transform
initial_tfm = itk.AffineTransform[itk.D, 3].New()
initial_tfm.SetIdentity()
# ... set transform parameters ...

# Register with initial transform
registrar = RegisterImagesANTs()
registrar.set_modality('ct')
registrar.set_fixed_image(fixed_image)

result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=initial_tfm  # Pass ITK transform directly!
)
```

### Using Previous Registration Results

```python
# First registration (rigid)
registrar_rigid = RegisterImagesANTs()
registrar_rigid.set_fixed_image(fixed_image)
result_rigid = registrar_rigid.register(moving_image)

# Use rigid result as initial transform for deformable registration
registrar_deform = RegisterImagesANTs()
registrar_deform.set_fixed_image(fixed_image)
result_deform = registrar_deform.register(
    moving_image=moving_image,
    initial_forward_transform=result_rigid["forward_transform"]  # Use previous result
)
```

### Using Displacement Field from File

```python
# Load existing displacement field
disp_field_image = itk.imread("previous_deformation.mha")

# Create displacement field transform
disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
disp_tfm.SetDisplacementField(disp_field_image)

# Use as initial transform
registrar = RegisterImagesANTs()
registrar.set_fixed_image(fixed_image)
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=disp_tfm
)
```

### Using with Label-Based Registration

```python
# Set the label images as the fixed and moving images
registrar = RegisterImagesANTs()
registrar.set_fixed_image(fixed_labels)

result = registrar.register(
    moving_image=moving_labels,
    initial_forward_transform=initial_transform
)
```

### Composing Multiple Transforms

```python
# Create composite transform
composite = itk.CompositeTransform[itk.D, 3].New()

# Add affine component
affine = itk.AffineTransform[itk.D, 3].New()
# ... configure affine ...
composite.AddTransform(affine)

# Add deformation component
disp_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
# ... configure displacement field ...
composite.AddTransform(disp_tfm)

# Use composite as initial transform
registrar = RegisterImagesANTs()
registrar.set_fixed_image(fixed_image)
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=composite  # Composite is automatically converted
)
```

## Technical Details

### Transform Composition (IMPORTANT!)

**ANTs does NOT include the initial_transform in its output transforms.** 

When you pass an `initial_transform` to `ants.registration()`:
- ANTs uses it as a starting point for optimization
- The returned `fwdtransforms` contain only the **refinement** from that starting point
- They do **NOT** include the initial transform itself

**This implementation handles composition automatically:**
```python
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=initial_transform
)

# The returned transforms INCLUDE the initial transform!
forward_transform = result["forward_transform"]
inverse_transform = result["inverse_transform"]
```

The composition is done as follows:
- **forward_transform**: `initial_forward_transform` → `registration_result` (applied in sequence)
- **inverse_transform**: `registration_result_inverse` → `initial_forward_transform_inverse` (applied in sequence)

### Coordinate Systems

- The initial forward transform maps points from **moving space to fixed space**
- The displacement field represents the displacement at each voxel in the **fixed image space**
- ANTs internally handles the transform composition with its optimization

### Performance Considerations

- **Memory**: Converting transforms to displacement fields requires memory proportional to the reference image size
- **Computation**: Conversion is performed once at the start of registration
- **No Disk I/O**: The entire conversion happens in memory (no temporary files)

### Transform Direction

The parameter `initial_forward_transform` represents:
- **forward_transform**: Transform from Moving → Fixed space
- This aligns with ANTs' expected initial transform direction

### Interpolation

When converting parametric transforms (Affine, Rigid) to displacement fields:
- The displacement is computed at each voxel of the reference image
- No interpolation artifacts are introduced
- The conversion is exact within floating-point precision

## Integration with ANTsPy

The converted transform can also be used directly with ANTsPy functions:

```python
import ants

registrar = RegisterImagesANTs()

# Convert ITK to ANTs transform
ants_tfm = registrar.itk_transform_to_ants_transform(
    itk_tfm=my_itk_transform,
    reference_image=fixed_image
)

# Use with ANTsPy directly
fixed_ants = registrar._itk_to_ants_image(fixed_image)
moving_ants = registrar._itk_to_ants_image(moving_image)

result = ants.registration(
    fixed=fixed_ants,
    moving=moving_ants,
    initial_transform=ants_tfm,  # ANTsPy transform object
    type_of_transform="SyN"
)
```

## Error Handling

The implementation handles several edge cases:

1. **Null transforms**: If `initial_forward_transform=None`, uses identity transform (default behavior)
2. **List transforms**: If ITK returns a list with one transform, extracts the single transform
3. **Transform composition**: Composite transforms are flattened to a single displacement field
4. **Type checking**: Validates that inputs are proper ITK transforms

## Transform Composition Example

Here's a concrete example showing how the automatic composition works:

```python
# Scenario: You have an affine pre-alignment and want to refine with deformable registration

# Step 1: Load images
fixed_image = itk.imread("fixed.mha")
moving_image = itk.imread("moving.mha")

# Step 2: Create initial affine transform (e.g., from rigid registration)
initial_affine = itk.AffineTransform[itk.D, 3].New()
# ... configure initial_affine with translation/rotation ...

# Step 3: Register with initial transform
registrar = RegisterImagesANTs()
registrar.set_fixed_image(fixed_image)
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=initial_affine  # Pass initial alignment
)

# Step 4: The returned transform includes BOTH the initial affine AND the deformation
forward_transform = result["forward_transform"]
# phi_MF is a CompositeTransform containing:
#   1. initial_affine (applied first)
#   2. deformation from ANTs registration (applied second)

# When you transform a point, it goes through both:
point_moving = itk.Point[itk.D, 3]()
point_moving[0], point_moving[1], point_moving[2] = 10.0, 20.0, 30.0
point_fixed = phi_MF.TransformPoint(point_moving)
# point_moving -> initial_affine -> deformation -> point_fixed
```

### Without Automatic Composition (Raw ANTsPy)

If you use ANTsPy directly, you'd need to compose manually:

```python
import ants

# Registration with initial transform
result = ants.registration(
    fixed=fixed_ants,
    moving=moving_ants,
    initial_transform=initial_tfm_ants,
    type_of_transform='SyN'
)

# ❌ WRONG: This only contains the deformation refinement!
transforms_incomplete = result['fwdtransforms']

# ✅ CORRECT: Must manually prepend the initial transform
transforms_complete = [initial_tfm_path] + result['fwdtransforms']

# Apply complete transform
warped = ants.apply_transforms(
    fixed=fixed_ants,
    moving=moving_ants,
    transformlist=transforms_complete  # Must include initial_tfm!
)
```

### With RegisterImagesANTs (Automatic Composition)

```python
# ✅ Composition is automatic!
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=initial_tfm
)

# The returned transform already includes everything
forward_transform = result["forward_transform"]  # Complete transform ready to use
```

## Comparison with File-Based Approach

### Old Approach (File-Based + Manual Composition)
```python
# Save ITK transform to disk
itk.transformwrite([itk_tfm], "temp_transform.mat")

# Load with ANTs
ants_tfm = ants.read_transform("temp_transform.mat")

# Use in registration
result = ants.registration(
    ...,
    initial_transform="temp_transform.mat"  # File path
)

# ❌ Must manually compose!
complete_transforms = ["temp_transform.mat"] + result['fwdtransforms']

# Clean up
os.remove("temp_transform.mat")
```

### New Approach (In-Memory + Automatic Composition)
```python
# ✅ No file I/O, automatic composition!
result = registrar.register(
    moving_image=moving_image,
    initial_forward_transform=itk_tfm  # ITK transform object
)

# Transform is complete and ready to use
forward_transform = result["forward_transform"]
```

## References

- [ANTsPy Documentation](https://antspyx.readthedocs.io/)
- [ANTs GitHub Repository](https://github.com/ANTsX/ANTs)
- [ITK Transform Documentation](https://itk.org/Doxygen/html/group__ITKTransform.html)
- TransformTools implementation in `physiomotion4d.transform_tools`

## See Also

- `examples/ants_initial_transform_example.py` - Complete working examples
- `src/physiomotion4d/transform_tools.py` - Transform conversion utilities
- `src/physiomotion4d/register_images_ants.py` - ANTs registration implementation

