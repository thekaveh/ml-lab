#!/usr/bin/env python3
"""Markdown-cells-only notebook editor.

Tier-C safety: never modifies, deletes, or re-orders `cell_type == "code"`
cells; never alters outputs or execution_count. Used by the cleanup-and-
standardization loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def replace_markdown_cell(nb_path: Path, index: int, new_text: str) -> None:
    doc = nbformat.read(nb_path, as_version=4)
    if index < 0 or index >= len(doc.cells):
        raise SystemExit(f"cell index {index} out of range (0..{len(doc.cells)-1})")
    cell = doc.cells[index]
    if cell.cell_type != "markdown":
        raise SystemExit(
            f"refusing to edit cell {index}: cell_type={cell.cell_type!r}, "
            f"only markdown cells may be edited"
        )
    cell.source = new_text
    nbformat.write(doc, nb_path)


def insert_markdown_cell(nb_path: Path, index: int, new_text: str) -> None:
    doc = nbformat.read(nb_path, as_version=4)
    if index < 0 or index > len(doc.cells):
        raise SystemExit(f"insert index {index} out of range (0..{len(doc.cells)})")
    new_cell = nbformat.v4.new_markdown_cell(new_text)
    doc.cells.insert(index, new_cell)
    nbformat.write(doc, nb_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Edit markdown cells of a Jupyter notebook without touching code cells.",
    )
    p.add_argument("--notebook", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", type=int, help="Replace markdown cell at this index.")
    group.add_argument("--insert-at", type=int, help="Insert a new markdown cell at this index.")
    p.add_argument("--text", required=True, help="New markdown source.")
    args = p.parse_args(argv)

    if not args.notebook.exists():
        raise SystemExit(f"notebook not found: {args.notebook}")

    if args.cell is not None:
        replace_markdown_cell(args.notebook, args.cell, args.text)
    else:
        insert_markdown_cell(args.notebook, args.insert_at, args.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
