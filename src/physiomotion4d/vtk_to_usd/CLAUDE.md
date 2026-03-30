# CLAUDE.md — vtk_to_usd

This subpackage is an **advanced internal library**. Understand this before touching it.

## Preferred API

**Do not add calls to `vtk_to_usd` from outside this subpackage.**

The correct entry point for all VTK→USD conversion in PhysioMotion4D is:

```python
from physiomotion4d.convert_vtk_to_usd import ConvertVTKToUSD
```

`ConvertVTKToUSD` operates on in-memory PyVista objects, handles colormap overlays,
multi-label anatomy, and animated time series, and is the only API that external users
or other PhysioMotion4D modules should call.

## Role of this subpackage

`vtk_to_usd/` is called internally by `ConvertVTKToUSD`. It provides:
- File-based VTK→USD conversion (reads `.vtk`, `.vtp`, `.vtu` from disk)
- Low-level data structures: `MeshData`, `ConversionSettings`, `MaterialData`
- USD primitive writing: normals, primvars, time samples, materials

Other PhysioMotion4D modules (workflow, segmentation, registration) should never
import directly from `physiomotion4d.vtk_to_usd`; they must go through
`ConvertVTKToUSD`. External library users may use the file-based API.

## When to edit this subpackage

Only edit code here when:
1. `ConvertVTKToUSD` cannot expose the needed behavior through its own API, **and**
2. The change is to the file-based conversion layer itself (readers, USD writers, data structures).

Always check whether the fix belongs in `convert_vtk_to_usd.py` first.

## Module responsibilities

| File                  | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `data_structures.py`  | Data containers: `MeshData`, `MaterialData`, etc.   |
| `vtk_reader.py`       | Read `.vtk`, `.vtp`, `.vtu` files into `MeshData`  |
| `usd_utils.py`        | Coordinate conversion (RAS→Y-up), primvar helpers  |
| `material_manager.py` | `UsdPreviewSurface` creation and binding            |
| `usd_mesh_converter.py` | Write `MeshData` to a USD prim                   |
| `converter.py`        | `VTKToUSDConverter` — high-level file-based API    |

## Coordinate system

RAS→Y-up conversion: `USD(x, y, z) = RAS(x, z, -y)`

This conversion happens inside `usd_utils.ras_to_usd()` / `ras_points_to_usd()`.
It must not be applied more than once. If you add a code path that produces USD
geometry, verify the transform is applied exactly once.

## What not to do

- Do not expose new public symbols in `__init__.py` without a clear reason.
- Do not call `vtk_to_usd` internals from `workflow_*.py` or any other top-level module.
- Do not duplicate coordinate conversion logic outside `usd_utils.py`.
