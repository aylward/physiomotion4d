# AGENTS.md

**Who reads this file:** Two audiences.

- **Contributors and developers** — scientists adding new segmenters or registerers,
  ML practitioners extending the pipeline. You read this to understand the role-based
  rules that Claude Code follows when you invoke a slash skill (`/plan`, `/impl`,
  `/test-feature`, `/doc-feature`).
- **AI agents** — Claude Code subagents activated via `.claude/agents/`. This file is
  injected as system context for every agent session in this repository.

If you are a developer invoking a skill for the first time, start with the
`## Developer tool prerequisites` section below, then read `CLAUDE.md`.
Before extending the pipeline, read the relevant guide in `docs/developer/`:
`extending.rst` for new workflows, `segmentation.rst` for segmenters,
`registration_images.rst` or `registration_models.rst` for registerers.

PhysioMotion4D converts 4D CT scans into animated USD models for NVIDIA Omniverse.
It is an **early-alpha** scientific Python library. Clarity beats premature optimization.
Breaking changes are acceptable. Backward compatibility is not a goal.

## Developer tool prerequisites

Two non-Python tools are required for contributor workflows.
(For the Python package itself, see `docs/installation.rst`.)

### Claude Code CLI (`claude`)

Powers all slash skills and `utils/claude_github_reviews.py`.

| Platform | Install |
|----------|---------|
| Windows  | `winget install Anthropic.ClaudeCode` |
| macOS    | `brew install claude` or download from claude.ai/code |
| Linux    | Download AppImage from claude.ai/code; place on `$PATH` |

### gh CLI (`gh`)

Required by `utils/claude_github_reviews.py`. Not installable via pip/uv.

| Platform | Install |
|----------|---------|
| Windows  | `winget install GitHub.cli` |
| macOS    | `brew install gh` |
| Linux    | See cli.github.com/manual/installation |

After installing on any platform: `gh auth login`

## Which skill should I use?

```text
I want to...                              Skill             Pre-read
──────────────────────────────────────────────────────────────────────────────
Understand scope before touching code     /plan             —
Add a feature or fix a bug               /impl             docs/developer/extending.rst
Write a new tutorial script              /impl             tutorials/README.md +
                                                           an existing tutorial for pattern
Write tests for a module                 /test-feature     docs/developer/core.rst
Update docstrings after an API change    /doc-feature      —
Design a new segmenter                   /plan             docs/developer/segmentation.rst
Design a new registerer                  /plan             docs/developer/registration_images.rst
                                                           or registration_models.rst
Make a structural design decision        /plan             docs/developer/architecture.rst
Review PR comments automatically         py utils/claude_github_reviews.py --pr <N>
```

`/plan` never modifies files. `/impl` always proposes a numbered plan before diffing.

Typical sequencing for a new feature:

```text
/plan add a new registration method
    ↓ confirm the plan
/impl add RegisterImagesGreedy to register_images_greedy.py
    ↓ implementation done
/test-feature RegisterImagesGreedy with synthetic ITK images
    ↓ tests pass
/doc-feature update docstrings for RegisterImagesGreedy
```

## Universal rules

- Read the relevant source files before proposing changes.
- Runtime classes (workflow, segmentation, registration, USD tools) inherit from
  `PhysioMotion4DBase`; new runtime classes must too. Standalone utility scripts
  and data/container/helper classes do not.
- In classes that inherit from `PhysioMotion4DBase`, use `self.log_info()` /
  `self.log_debug()` — never `print()`. Standalone scripts may use `print()`.
- Single quotes for strings; double quotes for docstrings. 88-char line limit.
- Full type hints (`mypy` strict). Use `Optional[X]` not `X | None`.
- Run `py -m pytest tests/ -m "not slow and not requires_data" -v` to verify changes.
- Consult `docs/API_MAP.md` to locate classes and methods before searching manually.

## Implementation role

- Summarize current behavior → propose numbered plan → implement.
- Keep diffs small and reviewable. Call out breaking changes explicitly.
- Prefer editing existing modules over creating new ones.
- No backward-compat shims: just change the code.

## Testing role

- Prefer synthetic `itk.Image` and small `pv.PolyData` surfaces — not real patient data.
- State image shape and axis order in every test docstring: e.g. `shape (X, Y, Z, T)`.
- Keep synthetic volumes ≤64 voxels per side for speed.
- Mark tests that genuinely need real data with `@pytest.mark.requires_data`.
- Use `test_tools.py` baseline utilities for surface and image regression checks.

## Documentation role

- Update docstrings for every changed public method. Keep claims factual.
- Do not create new `.md` files unless explicitly requested.
- Regenerate `docs/API_MAP.md` after any public API change:
  `py utils/generate_api_map.py`

## Architecture role

- Propose a numbered design plan with trade-offs before structural changes.
- Identify every file that will change and how the class hierarchy is affected.
- Flag changes at the ITK↔PyVista boundary or the RAS→Y-up coordinate transform as high-risk.
