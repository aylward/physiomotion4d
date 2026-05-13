#!/usr/bin/env python
# %%
import os

from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.data_download_tools import DataDownloadTools

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
output_dir = os.path.join(_HERE, "results")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_image_filename = DataDownloadTools.DownloadSlicerHeartCTData(data_dir)

# %%
if not os.path.exists(f"{data_dir}/slice_000.mha"):
    conv = ConvertNRRD4DTo3D()
    conv.load_nrrd_4d(str(input_image_filename))
    conv.save_3d_images(data_dir, "slice")
