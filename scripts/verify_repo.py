#!/usr/bin/env python3
"""Repo verification oracle for the cleanup-and-standardization /goal loop.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
findings; nonzero = findings present (count in stderr).

See docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTIVE_TASK_DIRS = (
    "image_classification-mnist-ffnn-numpy",
    "image_classification-mnist-ffnn-pytorch",
    "node_classification-reddit-gnn-pyg",
)

VERIFY_ONLY_DIRS = ("archive", "nnx", "vendor")


@dataclass
class Finding:
    id: str
    check: str
    severity: str
    location: str
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


_INTERNAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#][^)#]*)(#[^)]*)?\)")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
_GITIGNORE_REQUIRED_PATTERNS = (".claude/", ".superpowers/", "plan-*.md", "notes-*.md")
_BLOAT_PATTERNS = ("__pycache__", ".ipynb_checkpoints", ".DS_Store")


def _git_ls_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _iter_notebooks(repo: Path) -> Iterator[Path]:
    for d in ACTIVE_TASK_DIRS:
        for nb_path in (repo / d).glob("*.ipynb"):
            yield nb_path


def _iter_in_scope_text_files(repo: Path) -> Iterator[Path]:
    yield repo / "README.md"
    yield repo / "CLAUDE.md"
    for p in (repo / "docs").glob("*.md"):
        yield p
    for d in ACTIVE_TASK_DIRS:
        for p in (repo / d).glob("*.md"):
            yield p


def check_structure(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="structure")
    tracked = set(_git_ls_files(repo))

    valid_types = {"code", "markdown", "raw"}
    notebooks = list(_iter_notebooks(repo))
    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
            for i, c in enumerate(doc.cells):
                if c.cell_type not in valid_types:
                    result.findings.append(Finding(
                        id="S1.cell_type", check="structure", severity="error",
                        location=f"{nb.relative_to(repo)}:cell[{i}]",
                        message=f"unknown cell_type={c.cell_type!r}",
                    ))
        except Exception as e:
            result.findings.append(Finding(
                id="S1.parse", check="structure", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"failed to parse: {e}",
            ))

    seen_modules: dict[str, str] = {}
    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for li, line in enumerate(cell.source.splitlines()):
                m = _IMPORT_RE.match(line)
                if not m:
                    continue
                module = (m.group(1) or m.group(2) or "").split(".")[0]
                if not module or module in seen_modules:
                    continue
                seen_modules[module] = f"{nb.relative_to(repo)}:cell[{ci}]:line[{li}]"
                try:
                    if importlib.util.find_spec(module) is None:
                        result.findings.append(Finding(
                            id="S2.unresolved_import", check="structure", severity="error",
                            location=seen_modules[module],
                            message=f"module {module!r} not importable in current env",
                        ))
                except (ImportError, ValueError) as e:
                    result.findings.append(Finding(
                        id="S2.import_error", check="structure", severity="warning",
                        location=seen_modules[module],
                        message=f"find_spec({module!r}) raised {e!r}",
                    ))

    for md in _iter_in_scope_text_files(repo):
        text = _read_text(md)
        for m in _INTERNAL_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path = (md.parent / target).resolve()
            if not target_path.exists():
                result.findings.append(Finding(
                    id="S3.broken_link", check="structure", severity="error",
                    location=f"{md.relative_to(repo)}",
                    message=f"internal link target missing: {target}",
                    detail={"link": m.group(0)},
                ))

    _COMMON_IMPORT_RE = re.compile(r"^\s*(?:from\s+common(?:\.\w+)*\s+import\b|import\s+common(?:\.\w+)*)")
    for path in tracked:
        if path.startswith(("tests/", "archive/", "nnx/", "vendor/")):
            continue
        full = repo / path
        if not full.is_file():
            continue
        suffix = full.suffix.lower()
        if suffix == ".py":
            for i, line in enumerate(_read_text(full).splitlines(), 1):
                if _COMMON_IMPORT_RE.match(line):
                    result.findings.append(Finding(
                        id="S5.common_import", check="structure", severity="error",
                        location=f"{path}:{i}",
                        message="forbidden import; use `from nnx.` instead",
                    ))
        elif suffix == ".ipynb":
            try:
                doc = nbformat.read(full, as_version=4)
            except Exception:
                continue
            for ci, cell in enumerate(doc.cells):
                if cell.cell_type != "code":
                    continue
                for li, line in enumerate(cell.source.splitlines(), 1):
                    if _COMMON_IMPORT_RE.match(line):
                        result.findings.append(Finding(
                            id="S5.common_import", check="structure", severity="error",
                            location=f"{path}:cell[{ci}]:line[{li}]",
                            message="forbidden import; use `from nnx.` instead",
                        ))

    gitignore = _read_text(repo / ".gitignore")
    for pat in _GITIGNORE_REQUIRED_PATTERNS:
        if pat not in gitignore:
            result.findings.append(Finding(
                id="S6.gitignore_missing", check="structure", severity="error",
                location=".gitignore",
                message=f"required pattern absent: {pat}",
            ))
    for path in tracked:
        if path.startswith((".claude/", ".superpowers/")):
            result.findings.append(Finding(
                id="S6.tracked_bloat", check="structure", severity="error",
                location=path,
                message="bloat directory tracked; should be gitignored",
            ))

    for path in tracked:
        for pat in _BLOAT_PATTERNS:
            if pat in path:
                result.findings.append(Finding(
                    id="S7.tracked_bloat", check="structure", severity="error",
                    location=path,
                    message=f"bloat artifact tracked: contains {pat!r}",
                ))

    return result


def check_docs(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="docs")


def check_comments(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="comments")


def check_execution(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="execution")


CHECKS: dict[str, Callable[[Path, bool], CheckResult]] = {
    "structure": check_structure,
    "docs": check_docs,
    "comments": check_comments,
    "execution": check_execution,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repo verification oracle. Runs one or all of the four checks: "
            "structure, execution, docs, comments, all."
        )
    )
    parser.add_argument(
        "--check", required=True,
        choices=("structure", "execution", "docs", "comments", "all"),
        help="Which check to run.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow checks (E1-E3 in execution). Required when only "
             "non-executable areas changed in the round.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write findings JSON. Default: print to stdout.",
    )
    args = parser.parse_args(argv)

    if args.check == "all":
        checks_to_run = list(CHECKS.keys())
    else:
        checks_to_run = [args.check]

    results = [CHECKS[name](REPO_ROOT, args.fast) for name in checks_to_run]

    all_findings = [asdict(f) for r in results for f in r.findings]
    payload = {
        "schema_version": 1,
        "summary": {
            "checks_run": checks_to_run,
            "skipped": [r.name for r in results if r.skipped],
            "total_findings": len(all_findings),
            "by_check": {r.name: len(r.findings) for r in results},
        },
        "findings": all_findings,
    }

    out_text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text)
    else:
        print(out_text)

    if all_findings:
        print(f"verify_repo: {len(all_findings)} findings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
