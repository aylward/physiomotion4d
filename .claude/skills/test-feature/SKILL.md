---
description: Inspect a PhysioMotion4D implementation and its existing tests, propose a synthetic-data test plan, then create or update pytest tests. Explains how to run them.
---

Write or update tests for the following in PhysioMotion4D:

$ARGUMENTS

Instructions:
1. Read the implementation file(s) to understand the public interface.
2. Read the existing test file for this module if one exists (e.g. `tests/test_<module>.py`).
3. Propose a test plan: list the behaviors to cover and the synthetic data to create.
4. Implement tests using synthetic `itk.Image` objects (32–64 voxels/side) or small
   `pv.PolyData` surfaces — not real patient data.
5. State image shape and axis order in every test docstring.
6. Mark any test that genuinely requires real data with `@pytest.mark.requires_data`.
7. Show the exact command to run the new tests:
   `py -m pytest tests/test_<module>.py -v`

## Examples

Synthetic volumes must be ≤64 voxels/side. State shape and axes in every test docstring.

Good invocations:

```
/test-feature SegmentChestTotalSegmentator with synthetic (64,64,32) ITK image in RAS —
              no GPU required; use CPU fallback mode

/test-feature RegisterImagesICON.register returns dict with forward_transform and
              inverse_transform keys pointing to .hdf files

/test-feature tutorial_01_heart_gated_ct_to_usd.run_tutorial with synthetic
              single-frame NRRD and a temporary output directory
```

Anti-example (no class, no data shape, scope is unclear):

```
/test-feature segmentation   # which class? what behaviors? what synthetic data?
```
