---
description: Read relevant PhysioMotion4D source files, summarize current behavior, propose a brief plan, then implement the requested feature or refactor in small diffs. Calls out breaking changes.
---

Implement the following in the PhysioMotion4D repository:

$ARGUMENTS

Instructions:
1. Use `docs/API_MAP.md` to locate relevant files, then read them in full.
2. Summarize current behavior in 2–4 sentences.
3. State the implementation plan in numbered steps. For non-trivial changes, pause and confirm before proceeding.
4. Implement in the smallest reviewable diff possible.
5. Update docstrings and type hints for every changed public method.
6. Run `ruff check . --fix && ruff format .` after editing Python files.
7. Explicitly note any breaking changes introduced.
8. Do not add features beyond what was requested.

## Examples

Pre-read `docs/developer/extending.rst` for the class template and interface contract
before implementing anything that touches `PhysioMotion4DBase` subclasses.

Good invocations:

```
/impl add RegisterImagesGreedy to src/physiomotion4d/register_images_greedy.py
      following the RegisterImagesICON interface (set_fixed_image → register → dict)
      (pre-read: docs/developer/registration_images.rst)

/impl fix the RAS-to-Y-up transform being applied twice in vtk_to_usd/usd_utils.py

/impl add tutorials/tutorial_07_lung_gated_ct_to_usd.py following the pattern of
      tutorials/tutorial_01_heart_gated_ct_to_usd.py — same run_tutorial() signature,
      argparse CLI, and screenshot helpers
```

Anti-example (too vague — no file, no interface contract):

```
/impl add greedy registration   # which file? which base class? what interface?
```
