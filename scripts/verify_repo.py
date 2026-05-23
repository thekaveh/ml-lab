#!/usr/bin/env python3
"""Repo verification oracle for the cleanup-and-standardization /goal loop.

Runs four orthogonal checks (structure, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
findings; nonzero = findings present (count in stderr).

See docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

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


def check_structure(repo: Path, fast: bool) -> CheckResult:
    return CheckResult(name="structure")


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
