---
description: Inspect changed PhysioMotion4D code and existing docstrings, update docstrings and inline comments with accurate shape/axis information, and regenerate docs/API_MAP.md if public APIs changed.
---

Update documentation for the following in PhysioMotion4D:

$ARGUMENTS

Instructions:
1. Read the changed source file(s) in full.
2. Read existing docstrings for every public method or class that changed.
3. Update docstrings to reflect current behavior using NumPy docstring style.
   State image/tensor shape and axis order wherever arrays are involved.
4. Add inline comments only for non-obvious logic (coordinate transforms, shape permutations).
5. Do not create new `.md` files unless explicitly asked.
6. If any public class, method, or function signature changed, regenerate the API map:
   `python utils/generate_api_map.py`
7. Do not paraphrase the method name as the docstring — explain what it does and why.

## Examples

Pre-read `docs/developer/workflows.rst` for docstring style examples before
updating workflow classes.

Good invocations:

```
/doc-feature update docstrings for RegisterImagesICON after the mask_image
             parameter was added — state image shape (X, Y, Z) in RAS

/doc-feature update WorkflowConvertHeartGatedCTToUSD.run() — return type changed
             to Path; follow the style in docs/developer/workflows.rst

/doc-feature update the Purpose/Inputs/Outputs block in
             tutorials/tutorial_01_heart_gated_ct_to_usd.py after adding a new
             screenshot output; regenerate docs/API_MAP.md if public API changed
```

Anti-example (no class, no change context — nothing to anchor the update):

```
/doc-feature update docs   # which file? what changed?
```
