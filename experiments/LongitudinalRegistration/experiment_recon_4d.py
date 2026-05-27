# %% [markdown]
# # 4D CT Reconstruction Using RegisterTimeSeriesImages Class
#
# This script demonstrates the use of the `RegisterTimeSeriesImages` class
# to register a time series of CT images to a common reference frame.
#
# This is a refactored version of `reconstruct_4d_ct.ipynb` that uses
# the new class-based approach,including:
# - Registration of time series images using ANTs, Greedy, ICON, or combined
#   ANTs/Greedy + ICON methods
# - Reconstruction of time series using the `reconstruct_time_series()` method
# - Optional upsampling to fixed image resolution while preserving spatial positioning
#

# %%
# Import necessary libraries
########################################################

import os

import itk
import numpy as np
from physiomotion4d import RegisterTimeSeriesImages

# %%
# Identify reference images
########################################################

ref_data_dir = "d:/PhysioMotion4D/duke_data/ref_images"
src_data_dir_base = "d:/PhysioMotion4D/duke_data/gated_nii"
dest_data_dir_base = "d:/PhysioMotion4D/duke_data/recon4d"

ref_files = [
    os.path.join(ref_data_dir, f)
    for f in sorted(os.listdir(ref_data_dir))
    if f.startswith("pm00") and f.endswith(".nii.gz")
]

print(f"Found {len(ref_files)} reference images")

# %%
# Identify source data directories and files using reference image names
########################################################


print(os.path.basename(ref_files[0])[:6])
src_data_dirs = []
src_data_files = []
for ref_file in ref_files:
    src_dir = os.path.join(src_data_dir_base, os.path.basename(ref_file)[:6])
    src_data_dirs.append(src_dir)

    file_list = sorted(os.listdir(src_dir))
    valid_file_list = [
        f
        for f in file_list
        if "dia" not in f
        and "nop" not in f
        and "sys" not in f
        and f.endswith(".nii.gz")
    ]
    src_data_files.append(valid_file_list)

print(f"Found {len(src_data_dirs)} source data directories")
for d, fs in zip(src_data_dirs, src_data_files):
    print(f"{d}: {len(fs)} files")
    for f in fs:
        print(f"  {f}")

# %%
# Define registration function
########################################################


def register_time_series(
    reference_image_file: str,
    source_image_dir: str,
    source_image_files: list[str],
    registration_method: str,
) -> None:
    # ANTs registration
    if registration_method in ["ANTS", "greedy"]:
        number_of_iterations = [30, 15, 7, 3]
    elif registration_method == "ICON":
        number_of_iterations = 20
    elif registration_method in ["ANTS_ICON", "greedy_ICON"]:
        number_of_iterations = [[30, 15, 7, 3], 20]
    else:
        raise ValueError(f"Invalid registration method: {registration_method}")

    # Create output dir
    output_dir = os.path.join(
        dest_data_dir_base, registration_method, os.path.basename(source_image_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    # Read the reference image as the fixed image
    fixed_image = itk.imread(reference_image_file, pixel_type=itk.F)

    images = []
    for file in source_image_files:
        img = itk.imread(os.path.join(source_image_dir, file), pixel_type=itk.F)
        images.append(img)

    reference_image_num = 7
    register_start_to_reference = True
    if reference_image_file in source_image_files:
        reference_image_num = source_image_files.index(reference_image_file)
        register_start_to_reference = False

    portion_of_prior_transform_to_init_next_transform = 0.0

    # Register the time series
    registrar = RegisterTimeSeriesImages(registration_method=registration_method)
    registrar.set_modality("ct")
    registrar.set_fixed_image(fixed_image)
    if registration_method == "ANTS":
        registrar.set_number_of_iterations_ANTS(number_of_iterations)
    elif registration_method == "greedy":
        registrar.set_number_of_iterations_greedy(number_of_iterations)
    elif registration_method == "ICON":
        registrar.set_number_of_iterations_ICON(number_of_iterations)
    elif registration_method == "ANTS_ICON":
        registrar.set_number_of_iterations_ANTS(number_of_iterations[0])
        registrar.set_number_of_iterations_ICON(number_of_iterations[1])
    elif registration_method == "greedy_ICON":
        registrar.set_number_of_iterations_greedy(number_of_iterations[0])
        registrar.set_number_of_iterations_ICON(number_of_iterations[1])
    else:
        raise ValueError(f"Invalid registration method: {registration_method}")

    result = registrar.register_time_series(
        moving_images=images,
        reference_frame=reference_image_num,
        register_reference=register_start_to_reference,
        prior_weight=portion_of_prior_transform_to_init_next_transform,
    )

    upsampled_images = registrar.reconstruct_time_series(
        moving_images=images,
        inverse_transforms=result["inverse_transforms"],
        upsample_to_fixed_resolution=True,
    )

    losses = result["losses"]
    print("Registration complete!")
    print(f"  Average loss: {np.mean(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")
    print("")
    print("Saving results...")
    output_file_basename = os.path.basename(reference_image_file)[:6]
    for i, fwd_transform in enumerate(result["forward_transforms"]):
        time_point_index = source_image_files[i].index("_g") + 2
        time_point = source_image_files[i][time_point_index : time_point_index + 3]

        output_file = f"{output_file_basename}_{time_point}_fwd.hdf"
        itk.transformwrite(
            fwd_transform,
            os.path.join(output_dir, output_file),
            compression=True,
        )

        inv_transform = result["inverse_transforms"][i]
        output_file = f"{output_file_basename}_{time_point}_inv.hdf"
        itk.transformwrite(
            inv_transform,
            os.path.join(output_dir, output_file),
            compression=True,
        )

        output_file = f"{output_file_basename}_{time_point}_hrr.mha"
        itk.imwrite(
            upsampled_images[i],
            os.path.join(output_dir, output_file),
            compression=True,
        )


# %%
# Register time series
########################################################

for ref_file, src_dir, src_files in zip(ref_files, src_data_dirs, src_data_files):
    register_time_series(ref_file, src_dir, src_files, "ANTS")
    register_time_series(ref_file, src_dir, src_files, "greedy")
    register_time_series(ref_file, src_dir, src_files, "ICON")
    register_time_series(ref_file, src_dir, src_files, "ANTS_ICON")
    register_time_series(ref_file, src_dir, src_files, "greedy_ICON")

# %%
