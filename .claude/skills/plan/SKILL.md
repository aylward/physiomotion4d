---
description: Inspect PhysioMotion4D source files, summarize the current design, and produce a numbered implementation plan with open questions. Does not write code unless explicitly asked.
---

Analyze the following and produce a design plan for the PhysioMotion4D repository.

Task: $ARGUMENTS

Instructions:
1. Use `docs/API_MAP.md` to locate relevant classes and methods, then read those source files.
2. Summarize current behavior in 3–5 bullet points.
3. Produce a numbered implementation plan with enough detail to act on.
4. List every file that will change.
5. Call out any image-shape, axis-order, or coordinate-system implications explicitly.
6. List open questions that need user input before coding starts.
7. Do not modify any files unless the task explicitly asks you to.

## Examples

Pre-read the relevant `docs/developer/` guide before planning structural changes.

Good invocations:

```
/plan add SegmentChestNNUNet following the TotalSegmentator pattern
      (pre-read: docs/developer/segmentation.rst, docs/developer/extending.rst)

/plan redesign the ITK↔PyVista boundary in contour_tools.py to support mesh
      decimation before surface extraction
      (pre-read: docs/developer/architecture.rst)

/plan add tutorial_07_lung_gated_ct_to_usd.py following the tutorial_01 pattern
      (pre-read: tutorials/README.md, tutorials/tutorial_01_heart_gated_ct_to_usd.py)
```

Anti-example (too vague — /plan cannot scope this):

```
/plan improve segmentation   # which segmenter? what improvement? what files?
```
