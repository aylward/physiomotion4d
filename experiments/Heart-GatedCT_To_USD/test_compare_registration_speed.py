#!/usr/bin/env python
# %% [markdown]
# # Compare registration speed: Greedy vs ANTs vs ICON
#
# This notebook times **Greedy**, **ANTs**, and **ICON** when registering two time points of CT from the Slicer-Heart-CT data (TruncalValve 4D CT).
#
# **Prerequisites:** Run `0-download_and_convert_4d_to_3d.py` first so that `data/Slicer-Heart-CT/` contains the 4D NRRD and the 3D slice series (`slice_000.mha`, `slice_001.mha`, ...), and `results/slice_fixed.mha` exists.

# %%
import os
import time

import itk
import matplotlib.pyplot as plt
import pandas as pd
from itk import TubeTK as ttk

from physiomotion4d.test_tools import TestTools
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_greedy import RegisterImagesGreedy
from physiomotion4d.register_images_icon import RegisterImagesICON

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
output_dir = os.path.join(_HERE, "results")
os.makedirs(output_dir, exist_ok=True)

# Fixed = reference time point; moving = time point to align to fixed
fixed_image_path = os.path.join(output_dir, "slice_fixed.mha")
moving_image_path = os.path.join(data_dir, "slice_000.mha")

if not os.path.exists(fixed_image_path):
    raise FileNotFoundError(
        f"Fixed image not found: {fixed_image_path}. "
        "Run 0-download_and_convert_4d_to_3d.py first."
    )
if not os.path.exists(moving_image_path):
    raise FileNotFoundError(
        f"Moving image not found: {moving_image_path}. "
        "Run 0-download_and_convert_4d_to_3d.py first."
    )

fixed_image = itk.imread(fixed_image_path)
moving_image = itk.imread(moving_image_path)
print(f"Fixed image: {itk.size(fixed_image)}, spacing {itk.spacing(fixed_image)}")
print(f"Moving image: {itk.size(moving_image)}, spacing {itk.spacing(moving_image)}")

# %% [markdown]
# ## Optional: downsample for faster comparison
#
# Set `downsample_factor = 1.0` to use full resolution (slower). Use e.g. `0.5` to halve each dimension for a quicker run.

# %%
downsample_factor = 0.5  # 1.0 = full resolution

if downsample_factor != 1.0:
    resampler_f = ttk.ResampleImage.New(Input=fixed_image)
    resampler_f.SetResampleFactor([downsample_factor] * 3)
    resampler_f.Update()
    fixed_image = resampler_f.GetOutput()

    resampler_m = ttk.ResampleImage.New(Input=moving_image)
    resampler_m.SetResampleFactor([downsample_factor] * 3)
    resampler_m.Update()
    moving_image = resampler_m.GetOutput()
    print(f"Downsampled to factor {downsample_factor}")
    print(f"  Fixed: {itk.size(fixed_image)}")
    print(f"  Moving: {itk.size(moving_image)}")
else:
    print("Using full resolution.")

# %% [markdown]
# ## Run each method and record time
#
# All three use **deformable** registration (Greedy: affine + deformable; ANTs: SyN; ICON: deep learning). Settings are chosen for a fair comparison with reduced iterations so the notebook runs in a few minutes.

# %%
results_list = []

# --- Greedy (deformable) ---
try:
    reg_g = RegisterImagesGreedy()
    reg_g.set_modality("ct")
    reg_g.set_transform_type("Deformable")
    reg_g.set_number_of_iterations([10, 5, 2])
    reg_g.set_fixed_image(fixed_image)

    t0 = time.perf_counter()
    out_g = reg_g.register(moving_image)
    elapsed_g = time.perf_counter() - t0

    loss_g = out_g.get("loss")
    results_list.append(
        {
            "method": "Greedy",
            "time_sec": round(elapsed_g, 2),
            "loss": float(loss_g) if loss_g is not None else None,
        }
    )
    print(f"Greedy:  {elapsed_g:.2f} s")
except Exception as e:
    results_list.append({"method": "Greedy", "time_sec": None, "loss": None})
    print(f"Greedy:  failed - {e}")

# --- ANTs (deformable SyN) ---
try:
    reg_a = RegisterImagesANTs()
    reg_a.set_modality("ct")
    reg_a.set_transform_type("Deformable")
    reg_a.set_number_of_iterations([10, 5, 2])  # reduced for speed
    reg_a.set_fixed_image(fixed_image)

    t0 = time.perf_counter()
    out_a = reg_a.register(moving_image)
    elapsed_a = time.perf_counter() - t0

    loss_a = out_a.get("loss")
    results_list.append(
        {
            "method": "ANTs",
            "time_sec": round(elapsed_a, 2),
            "loss": float(loss_a) if loss_a is not None else None,
        }
    )
    print(f"ANTs:   {elapsed_a:.2f} s")
except Exception as e:
    results_list.append({"method": "ANTs", "time_sec": None, "loss": None})
    print(f"ANTs:   failed - {e}")

# --- ICON (deformable, GPU) ---
try:
    reg_i = RegisterImagesICON()
    reg_i.set_modality("ct")
    reg_i.set_number_of_iterations(50)
    reg_i.set_fixed_image(fixed_image)

    t0 = time.perf_counter()
    out_i = reg_i.register(moving_image)
    elapsed_i = time.perf_counter() - t0

    loss_i = out_i.get("loss")
    results_list.append(
        {
            "method": "ICON",
            "time_sec": round(elapsed_i, 2),
            "loss": float(loss_i) if loss_i is not None else None,
        }
    )
    print(f"ICON:   {elapsed_i:.2f} s")
except Exception as e:
    results_list.append({"method": "ICON", "time_sec": None, "loss": None})
    print(f"ICON:   failed - {e}")

df = pd.DataFrame(results_list)

# %%
print(df)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
valid = df["time_sec"].notna()
if valid.any():
    methods = df.loc[valid, "method"]
    times = df.loc[valid, "time_sec"]
    ax.bar(methods, times, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Registration time: two time points (Slicer-Heart-CT)")
    plt.tight_layout()
    if not TestTools.running_as_test():
        plt.show()
else:
    print("No successful runs to plot.")

# %% [markdown]
# ## Notes
#
# - **Greedy**: CPU-based, often faster than ANTs for comparable quality; see [Greedy](https://greedy.readthedocs.io/) and [picsl-greedy](https://pypi.org/project/picsl-greedy/).
# - **ANTs**: CPU-based, very widely used; typically slower than Greedy for similar settings.
# - **ICON**: GPU-based (UniGradIcon); speed depends on GPU. Loss values are not directly comparable across methods.
# - For a quicker comparison, use `downsample_factor = 0.5` or reduce `number_of_iterations` further.
