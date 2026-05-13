#!/usr/bin/env python
# %%
import os
import shutil

from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.data_download_tools import DataDownloadTools

# %%
data_dir = "."
output_dir = "."

if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

input_image_filename = DataDownloadTools.DownloadSlicerHeartCTData(data_dir)

# %%
conv = ConvertNRRD4DTo3D()
conv.load_nrrd_4d(str(input_image_filename))
conv.save_3d_images(output_dir, "slice")

# Save the mid-stroke slice as the fixed/reference image
shutil.copyfile(f"{output_dir}/slice_007.mha", f"{output_dir}/slice_fixed.mha")
