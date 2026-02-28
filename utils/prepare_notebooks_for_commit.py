#!/usr/bin/env python
"""
Clear cell outputs and widget state in every Jupyter notebook in the project.

Use this script before committing (or as a pre-commit hook) to keep notebook
diffs small and avoid committing large output blobs, execution metadata, and
ipywidget/PyVista widget state (application/vnd.jupyter.widget-state+json).

Usage:
    python prepare_notebooks_for_commit.py [root_dir]

If root_dir is omitted, uses the parent of the directory containing this script
(i.e. the physiomotion4d project root).
"""

from pathlib import Path
import argparse
import json
import sys


def clear_cell_outputs(cell: dict) -> None:
    """Clear outputs and execution state for a single cell."""
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        # Remove execution metadata (timestamps, etc.) for cleaner diffs
        if "metadata" in cell and "execution" in cell["metadata"]:
            del cell["metadata"]["execution"]


def strip_widget_state(nb: dict) -> bool:
    """
    Remove Jupyter widget state from notebook metadata (ipywidgets, PyVista, etc.).
    Returns True if widget state was present and removed, False otherwise.
    """
    meta = nb.get("metadata")
    if not isinstance(meta, dict) or "widgets" not in meta:
        return False
    del meta["widgets"]
    return True


def clear_notebook(path: Path) -> bool:
    """
    Clear all cell outputs and strip widget state in a notebook file in place.
    Returns True if the file was modified, False otherwise.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"  Error reading {path}: {e}", file=sys.stderr)
        return False

    try:
        nb = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Invalid JSON in {path}: {e}", file=sys.stderr)
        return False

    if "cells" not in nb:
        return False

    modified = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        had_output = (
            bool(cell.get("outputs")) or cell.get("execution_count") is not None
        )
        had_exec_meta = (
            "metadata" in cell
            and isinstance(cell["metadata"], dict)
            and "execution" in cell["metadata"]
        )
        if had_output or had_exec_meta:
            clear_cell_outputs(cell)
            modified = True

    if strip_widget_state(nb):
        modified = True

    if not modified:
        return False

    try:
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"  Error writing {path}: {e}", file=sys.stderr)
        return False

    return True


def find_notebooks(root: Path) -> list[Path]:
    """Return all .ipynb files under root, excluding hidden and common ignore dirs."""
    notebooks = []
    for path in root.rglob("*.ipynb"):
        # Skip hidden dirs and typical ignore dirs
        if any(part.startswith(".") for part in path.parts):
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        notebooks.append(path)
    return sorted(notebooks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear cell outputs in all Jupyter notebooks under a directory."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Root directory to search (default: project root, i.e. parent of utils/).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only list notebooks that would be modified; do not write.",
    )
    args = parser.parse_args()

    if args.root_dir is not None:
        root = Path(args.root_dir).resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 1
    else:
        # Default: parent of the directory containing this script
        root = Path(__file__).resolve().parent.parent

    notebooks = find_notebooks(root)
    if not notebooks:
        print(f"No notebooks found under {root}")
        return 0

    print(f"Found {len(notebooks)} notebook(s) under {root}")
    modified_count = 0

    for path in notebooks:
        rel = path.relative_to(root)
        if args.dry_run:
            # In dry-run we still need to check if it would be modified
            try:
                nb = json.loads(path.read_text(encoding="utf-8"))
                would_modify = False
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code" and (
                        cell.get("outputs")
                        or cell.get("execution_count") is not None
                        or (cell.get("metadata") or {}).get("execution")
                    ):
                        would_modify = True
                        break
                if not would_modify and isinstance(nb.get("metadata"), dict):
                    would_modify = "widgets" in nb["metadata"]
                if would_modify:
                    print(f"  Would clear: {rel}")
                    modified_count += 1
            except (OSError, json.JSONDecodeError):
                pass
        else:
            if clear_notebook(path):
                print(f"  Cleared: {rel}")
                modified_count += 1

    if args.dry_run:
        print(f"\nDry run: {modified_count} notebook(s) would be modified.")
    else:
        print(f"\nCleared outputs in {modified_count} notebook(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
