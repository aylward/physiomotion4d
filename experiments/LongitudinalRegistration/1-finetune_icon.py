# %% [markdown]
# # Fine-tune uniGradICON on Duke 4D Gated CT Data
#
# Discovers per-patient gated CT images and their precomputed
# SegmentHeartSimpleware labelmaps and applies the project-wide fixed 80/20
# train/test split (sort patients in ``ref_data_dir`` by filename; the first
# 80% are train, the last 20% are test).  The train cohort is handed to
# :class:`WorkflowFineTuneICONRegistration`, which builds the paired dataset
# JSON, YAML config, and derived loss-function masks, then launches
# ``unigradicon.finetuning.finetune`` as a subprocess.
#
# ``2-recon_4d_icon_eval.py`` re-derives the same split from the same sorted
# patient list — no cached split file is needed.
#
# Each patient directory under ``src_data_dir_base`` is one ``subject_id``;
# all of that patient's gated time-point frames form a paired training group.
# Frames whose labelmap is missing on disk are dropped from the dataset.

# %%
import os
from pathlib import Path

import itk

from physiomotion4d import WorkflowFineTuneICONRegistration
from physiomotion4d.register_images_icon import RegisterImagesICON

# %% [markdown]
# ## 1. Configure data, output locations, and the train/test split

# %%
ref_data_dir = Path("d:/PhysioMotion4D/duke_data/ref_images")
src_data_dir_base = Path("d:/PhysioMotion4D/duke_data/gated_nii")
segmentation_dir_base = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")

# Where the workflow writes the dataset JSON, YAML config, derived masks, and
# the uniGradICON ``checkpoints/`` tree.  experiment_dir resolves to
# ``output_dir / fine_tune_name``.
output_dir = Path("./results")
fine_tune_name = "icon_finetuned"

# Fixed train/test split: sort patients in ``ref_data_dir`` by filename;
# first 80% are train, last 20% are test.  ``2-recon_4d_icon_eval.py`` applies
# the same rule so the two scripts agree without a cached split record.
train_fraction = 0.8

# Local clone of uniGradICON (feat-add-finetuning branch) — prepended to
# PYTHONPATH so the subprocess picks up the local source instead of the
# installed package.  Set to ``None`` to use the pip-installed unigradicon.
unigradicon_src_path: Path | None = Path(__file__).parent / "uniGradICON" / "src"

# %% [markdown]
# ## 2. Enumerate patients and apply the fixed 80/20 split
#
# Sort ``ref_data_dir`` by filename to produce the canonical patient order.
# The first 80% become the train cohort; the last 20% are the held-out test
# cohort that ``2-recon_4d_icon_eval.py`` will evaluate.

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
print(f"Found {len(all_patient_ids)} patients under {ref_data_dir}")

if len(all_patient_ids) < 2:
    raise FileNotFoundError(
        f"Need at least 2 patients to form a train/test split; "
        f"discovered {len(all_patient_ids)} under {ref_data_dir}"
    )

n_train = max(
    1,
    min(len(all_patient_ids) - 1, round(train_fraction * len(all_patient_ids))),
)
train_subjects = all_patient_ids[:n_train]
test_subjects = all_patient_ids[n_train:]
print(f"  Train (first {n_train}): {train_subjects}")
print(f"  Test  (last {len(test_subjects)}): {test_subjects}")

# %% [markdown]
# ## 3. Gather the train cohort's gated frames and labelmaps
#
# For each train-cohort patient, list gated frames in
# ``src_data_dir_base / <patient_id>`` (excluding ``"nop"`` non-gated
# references) and pair each frame with its
# ``<stem>_labelmap.nii.gz`` under ``segmentation_dir_base / <patient_id>``.
# Patients with no source directory or no valid frames are skipped here only
# — they remain part of the canonical train list above, but contribute no
# training data.  Missing labelmaps are recorded as ``None`` so the workflow
# skips just that frame.

# %%
train_image_files: list[list[str]] = []
train_segmentation_files: list[list[str | None]] = []
valid_train_subjects: list[str] = []

for patient_id in train_subjects:
    src_dir = src_data_dir_base / patient_id
    seg_dir = segmentation_dir_base / patient_id

    if not src_dir.is_dir():
        print(f"  Skipping {patient_id}: source dir {src_dir} not found")
        continue

    frame_names = sorted(
        f for f in os.listdir(src_dir) if "nop" not in f and f.endswith(".nii.gz")
    )
    if not frame_names:
        print(f"  Skipping {patient_id}: no valid frames in {src_dir}")
        continue

    image_paths = [str(src_dir / f) for f in frame_names]
    seg_paths: list[str | None] = []
    for f in frame_names:
        labelmap = seg_dir / f.replace(".nii.gz", "_labelmap.nii.gz")
        seg_paths.append(str(labelmap) if labelmap.exists() else None)

    train_image_files.append(image_paths)
    train_segmentation_files.append(seg_paths)
    valid_train_subjects.append(patient_id)

    n_seg = sum(1 for s in seg_paths if s is not None)
    print(f"  {patient_id}: {len(image_paths)} frames, {n_seg} with labelmap")

# %% [markdown]
# ## 4. Pre-compute loss-function masks next to each labelmap
#
# Use :meth:`RegisterImagesICON.create_mask` (``>0`` threshold + 5 mm
# physical-radius dilation) to derive each frame's binary heart-ROI mask and
# write it as ``<labelmap_stem>_mask.nii.gz`` in the labelmap's own directory.
# Pre-computing here means the workflow does not have to re-derive masks
# during ``run_fine_tuning`` and the same masks are reused by downstream
# evaluation scripts.

# %%
mask_dilation_mm = 5.0
train_mask_files: list[list[str | None]] = []
for image_paths, seg_paths in zip(
    train_image_files, train_segmentation_files, strict=True
):
    mask_paths: list[str | None] = []
    for seg_path in seg_paths:
        if seg_path is None:
            mask_paths.append(None)
            continue
        seg_p = Path(seg_path)
        stem = seg_p.name
        stem = stem[:-7] if stem.endswith(".nii.gz") else seg_p.stem
        mask_p = seg_p.parent / f"{stem}_mask.nii.gz"
        if not mask_p.exists():
            mask = RegisterImagesICON.create_mask(
                itk.imread(str(seg_p)), dilation_mm=mask_dilation_mm
            )
            itk.imwrite(mask, str(mask_p), compression=True)
        mask_paths.append(str(mask_p))
    train_mask_files.append(mask_paths)

# %% [markdown]
# ## 5. Fine-tune uniGradICON on the train cohort
#
# The workflow consumes both the labelmaps (for paired-with-seg training and
# ``use_label``) and the pre-computed masks (for ``loss_function_masking``)
# and launches ``unigradicon.finetuning.finetune`` as a subprocess.  The
# final checkpoint lands at
# :meth:`WorkflowFineTuneICONRegistration.expected_weights_path`, which is
# the default ``--finetuned-weights-path`` read by ``2-recon_4d_icon_eval.py``.

# %%
workflow = WorkflowFineTuneICONRegistration(
    subject_image_files=train_image_files,
    output_dir=output_dir,
    fine_tune_name=fine_tune_name,
    subject_ids=valid_train_subjects,
    subject_segmentation_files=train_segmentation_files,
    subject_mask_files=train_mask_files,
    mask_dilation_mm=mask_dilation_mm,
    unigradicon_src_path=unigradicon_src_path,
    epochs=100,
)

weights_path = workflow.run_fine_tuning()
print(f"\nFine-tuning complete. Expected weights at: {weights_path}")
print(f"Held-out test cohort (for 2-recon_4d_icon_eval.py): {test_subjects}")
