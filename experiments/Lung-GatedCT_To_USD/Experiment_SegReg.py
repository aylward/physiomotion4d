#!/usr/bin/env python
# %%
import os

import itk

from data_dirlab_4d_ct import DataDirLab4DCT

from physiomotion4d import RegisterImagesICON
from physiomotion4d import SegmentChestTotalSegmentator

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "..", "data", "DirLab-4DCT")
_RESULTS_DIR = os.path.join(_HERE, "results_SegReg")

# %%
fixed_image = DataDirLab4DCT().fix_image(
    itk.imread(os.path.join(_DATA_DIR, "Case1Pack_T30.mhd"))
)
moving_image = DataDirLab4DCT().fix_image(
    itk.imread(os.path.join(_DATA_DIR, "Case1Pack_T00.mhd"))
)

# %%
# Register images
reg_images = RegisterImagesICON()
reg_images.set_fixed_image(fixed_image)
_ = reg_images.register(moving_image)
moving_image_registered = reg_images.get_registered_image()
os.makedirs(_RESULTS_DIR, exist_ok=True)
itk.imwrite(
    moving_image_registered,
    os.path.join(_RESULTS_DIR, "Experiment_reg.mha"),
    compression=True,
)

# %%
img = itk.imread(os.path.join(_RESULTS_DIR, "Experiment_reg.mha"))
tot_seg = SegmentChestTotalSegmentator()
seg_results = tot_seg.segment(img, contrast_enhanced_study=False)
itk.imwrite(
    seg_results["labelmap"],
    os.path.join(_RESULTS_DIR, "Experiment_totseg.mha"),
    compression=True,
)
