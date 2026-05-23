---
date: 2026-05-22
title: Repo cleanup and documentation standardization
status: approved
owner: kaveh
---

# Repo cleanup and documentation standardization

A design for a `/goal`-driven, iterative verify-and-fix loop that brings the entire `ml` repo to a polished, conventional data-science-portfolio state: all documentation standardized to a hierarchical numbered template, all code free of "states-the-what" noise, all dangling references resolved, and all bloat purged — with zero remaining findings across four orthogonal verification checks.

---

## 1. Goal & scope

### 1.1 Goal

Reach a state where running one verification pass produces zero findings across all four checks (structural & link sanity, executable correctness, documentation conformance, comment hygiene), while the repo presents as a polished, professionally-conventional data-science portfolio with explicit room for the planned roadmap tasks.

### 1.2 In scope for edits

- Root: `README.md`, `CLAUDE.md`, `.gitignore`, `Makefile`, `pyproject.toml` metadata, `Dockerfile` comments.
- `docs/` — `env-setup.md`, `jupyterhub-integration.md`, `vscode-remote-access.md`.
- `scripts/` — `.py` and `.sh` files for comment hygiene; new files `verify_repo.py` and `edit_notebook_markdown.py` added in round 0.
- The three active task folders, including their notebooks. Markdown cells freely; code cells edited only for comment hygiene. Tier-A notebooks may be re-executed; Tier-B/Tier-C notebooks have markdown cells edited via the `edit_notebook_markdown.py` helper and are validated by smoke runs that write to `/tmp/`.
- The three active task `README.md` files.

### 1.3 Verify-only, never modify

- `archive/` — read-only per `CLAUDE.md`.
- `nnx/` — submodule; issues become findings in `docs/FINDINGS-NNX.md`, not fixes.
- `vendor/genai-vanilla/` — vendored; issues become findings in `docs/FINDINGS-VENDOR.md`, not fixes.

### 1.4 Definition of "bloat"

- `.claude/`, `.superpowers/`, and any local plan files outside `docs/superpowers/specs/`.
- The loop confirms none are tracked and adds patterns to `.gitignore` where missing.
- The README-referenced `docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md` does not exist; the dangling link is replaced by an inline paragraph in round 0 (see §4.2).

---

## 2. Documentation standard

The target shapes the verification check D1–D4 enforces. Three artifact types, each with a fixed hierarchy.

### 2.1 Notebook shape

The 6-section hierarchy is the contract. Markdown cells only — code cells are touched only for comment hygiene.

```
# <Notebook title — matches task/phase>

# 1. Overview
## 1.1 Task & motivation
## 1.2 Dataset summary
## 1.3 Approach in one paragraph
## 1.4 Libraries used (with versions)

# 2. Environment & Setup
## 2.1 Imports
## 2.2 Configuration / hyperparameters
## 2.3 Reproducibility (seed, device)

# 3. Data
## 3.1 Loading
## 3.2 Inspection / EDA
## 3.3 Preprocessing & splits

# 4. Model
## 4.1 Architecture
## 4.2 Loss & optimizer
## 4.3 Why this design

# 5. Training
## 5.1 Training loop
## 5.2 Metrics tracked
## 5.3 Run-time notes

# 6. Evaluation & Results
## 6.1 Test-set evaluation
## 6.2 Visualizations
## 6.3 Discussion
```

Rules:

- Every active-task notebook MUST contain every `#`-level heading (1–6) and every `##` subheading listed above, in this order.
- Sections that don't naturally apply to a specific notebook still appear with a one-line "Not applicable — see [other notebook]" pointer. Empty placeholders are not allowed.
- Multi-phase notebooks get scoped variants. The mapping below is the source of truth for verify check D1; the verify script embeds this table.
- Notebook headings are the single source of truth for the per-task README's "What's in the notebook" section.

#### Required-sections lookup table

| Notebook path | Required top-level sections | Notes |
|---|---|---|
| `image_classification-mnist-ffnn-numpy/notebook.ipynb` | §1–§6 (full) | All subsections per §2.1 |
| `image_classification-mnist-ffnn-pytorch/notebook.ipynb` | §1–§6 (full) | All subsections per §2.1 |
| `node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb` | §1, §2, §3 (renamed "Dataset deep-dive") | §4–§6 omitted; §3 has additional subsections §3.4 "Graph structure" and §3.5 "Class distribution" |
| `node_classification-reddit-gnn-pyg/phase2-model-selection-notebook{1,2,3,4}.ipynb` | §1, §2, §3, §4 (expanded), §5, §6 (abbreviated to §6.1, §6.3 only) | §4 has one §4.x per architecture being selected |
| `node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook{,2,3,4}.ipynb` | §1–§6 (full) | All subsections per §2.1 |

Adding a future task means adding a row to this table or accepting the default 6-section requirement.

### 2.2 Per-task README shape

```
# <task-folder-name>

## 1. Task summary
- Task, Dataset, Model, Framework (one line each)

## 2. Why this exists
(2–4 sentences, intent + how it relates to siblings)

## 3. What's in the notebook(s)
(one bullet per top-level §-heading from the notebook)

## 4. How to run
(steps for genai-vanilla; one-liner for venv; tier callout)

## 5. Dependencies
(libraries with major versions; pointer to root requirements)

## 6. Known issues / caveats

## 7. Future work
(optional — bullets only if relevant)
```

### 2.3 Root README shape

```
# ml — personal ML lab

## 1. Overview
## 2. Repository layout
## 3. Quick start
  ### 3.1 genai-vanilla jupyterhub (recommended)
  ### 3.2 Local Docker
  ### 3.3 Local venv
## 4. Tasks
  (the existing table, plus a "Planned" sub-table)
## 5. Notebook re-execution policy (tier table)
## 6. NNx library (submodule + how to update)
## 7. Repository conventions (link to CLAUDE.md highlights)
## 8. Roadmap
## 9. License
```

### 2.4 Comment-hygiene policy

Applies to `.py` files in scope and to notebook code cells in scope.

A comment is **allowed** only if it does at least one of:
1. Explains *why* a non-obvious choice was made.
2. Notes a hidden constraint, invariant, or workaround.
3. Cites an external reference (paper, ticket) needed to understand the code.

A comment is **deleted** if it restates what self-explanatory code already says (e.g., `# loop over batches` over `for batch in loader`, `# import numpy as np` over the import, type-restating docstrings on trivially-typed functions).

Docstrings on public functions/classes stay if they document parameters or contracts; they go if they merely paraphrase the signature.

---

## 3. Verification matrix

Each round runs all four checks. A round is **green** iff every check produces an empty findings list. All four checks are dispatched by a single new entry point: `scripts/verify_repo.py` (added in round 0).

### 3.1 Structural & link sanity (`check-structure`)

Programmatic, fast (<30s).

| # | Check | Mechanic |
|---|---|---|
| S1 | All `.ipynb` parse as JSON, every cell has a valid `cell_type` | `nbformat.read` |
| S2 | All notebook imports resolve in the smoke env | parse `import` lines, `importlib.util.find_spec` |
| S3 | Every internal markdown link `[...](relative/path)` resolves to an existing file/anchor | regex + `os.path.exists` + anchor scan |
| S4 | No reference to deleted/non-existent files | dangling-link sweep over README, CLAUDE.md, `docs/` |
| S5 | No `from common.` imports anywhere (only `from nnx.`) | grep |
| S6 | `.gitignore` covers `.claude/`, `.superpowers/`, plan files; none of these are tracked | `git ls-files` cross-check |
| S7 | No `.DS_Store`, no `__pycache__`, no `.ipynb_checkpoints` tracked | `git ls-files` |

### 3.2 Executable correctness (`check-execution`)

Slow (10–20 min for Tier-A).

| # | Check | Mechanic |
|---|---|---|
| E1 | `make run-tier-a` succeeds end-to-end | subprocess, exit 0 |
| E2 | `make smoke-tier-b` succeeds (writes to `/tmp/`) | subprocess, exit 0 |
| E3 | `make smoke-tier-c` succeeds (writes to `/tmp/`) | subprocess, exit 0 |
| E4 | Tier-A notebooks left with cells executed; no cell errored | `nbformat` scan for outputs containing `"ename"` |
| E5 | Tier-C code-cell content (source + outputs) unchanged from `pre-cleanup-baseline` tag | For each phase3 notebook, parse both `HEAD` and `pre-cleanup-baseline` revisions via `nbformat`; assert that for every cell with `cell_type == "code"`, the `source`, `outputs`, and `execution_count` fields are byte-equal. Markdown-cell diffs are permitted. |
| E6 | `scripts/setup-in-jupyter.sh` and `scripts/start-jupyterhub.sh` shellcheck-clean | `shellcheck` |

Rounds that only touch markdown / READMEs can skip E1–E3 by passing `--fast`. The **final** round before exit MUST run the full executable suite.

### 3.3 Documentation conformance (`check-docs`)

Programmatic, structural.

| # | Check | Mechanic |
|---|---|---|
| D1 | Every active-task notebook contains the required §1–§6 headings in order (per §2.1, with phase variants from the lookup table) | parse markdown cells, ordered-set match |
| D2 | Every notebook has a top markdown cell stating purpose+dataset (per CLAUDE.md) | first-cell type & content scan |
| D3 | Every per-task README matches the §2.2 shape (all required H2s present, in order) | regex match |
| D4 | Root README matches the §2.3 shape | regex match |
| D5 | Root README's task table row count = number of active task folders | filesystem cross-check |
| D6 | "Planned" / Roadmap section in root README is non-empty | regex |
| D7 | `docs/jupyterhub-integration.md`, `docs/env-setup.md`, `docs/vscode-remote-access.md` exist and contain at least one H2 | filesystem + regex |
| D8 | Terminology consistency: pick one canonical spelling for `genai-vanilla`, `JupyterHub`, `NumPy`, `PyTorch`, `PyG`; flag deviations | grep with allow-list |

### 3.4 Comment-hygiene scan (`check-comments`)

Two phases.

**Phase A (heuristic, deterministic):** For each `.py` file and notebook code cell in scope, tokenize and for every comment line look at the next non-blank code line. If the comment line, stripped and lowercased, is a verb-phrase that the next code line obviously implements (whitelist of patterns: `# import …` → `import …`; `# loop over …` → `for …`; `# return …` → `return …`; `# define …` → `def …`; `# create …` / `# initialize …` → assignment), flag. Also flag comments that are just the function name in English (`# do_thing` over `def do_thing`).

**Phase B (judge, on the survivors):** Each surviving comment is fed to a subagent with the next 5 code lines and asked: "Does this comment explain WHY, note a constraint, or cite a reference? Yes/no." The judge runs once per round, on changed files only.

Findings from both phases land in the round report. The fix step does NOT auto-delete judge-flagged comments; it presents them as a batch suggestion the loop must confirm against the §2.4 rule. A single round may delete at most 50 comments — above that, halt.

### 3.5 Findings artifact

Each round writes:
- `/tmp/ml-verify/round-N-findings.json` — machine-readable; the loop reads this.
- `/tmp/ml-verify/round-N-report.md` — human-readable.

---

## 4. Cleanup catalog (round 0)

Pre-loop mechanical sweep that runs once before the verify/fix cycle begins.

### 4.1 Untracked-bloat sweep

| Item | Action |
|---|---|
| `.claude/` (currently untracked) | Confirm untracked; ensure `.gitignore` covers it |
| `.superpowers/` | Add pattern to `.gitignore` |
| `plan-*.md`, `*-plan.md`, `notes-*.md` at repo root | Add patterns to `.gitignore`; surface any existing matches for user confirmation before deletion |
| `docs/superpowers/specs/` | Keep; only commit canonical specs. Add ignore for `*-draft.md`, `*-scratch.md` |
| `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/` | Already gitignored; `git rm --cached` if any are tracked |

### 4.2 Dangling-reference catalog

| Reference | Where | Decision |
|---|---|---|
| `docs/superpowers/specs/2026-05-16-ml-repo-revival-design.md` | `README.md` Roadmap section | Remove the link; replace with a one-paragraph rationale inline. This new spec replaces it. |
| `Phase 1 merge cb4d8f4` SHA | `image_classification-mnist-ffnn-pytorch/README.md` | Verify the SHA still exists; if rewritten / dropped, replace with a plain dependency statement |
| `(private)` annotation on NNx submodule | `README.md` NNx section | Confirm with `git ls-remote` whether NNx is actually private; correct the label |

### 4.3 Explicitly NOT cleaned up

- The 23 archived codexglue experiments — kept as-is per CLAUDE.md.
- The `vendor/genai-vanilla/` submodule — vendored, not ours.
- The `nnx/` submodule — issues become findings, not fixes.
- Notebook code cells of Tier-B/Tier-C — markdown cells get standardized, code cells untouched (output preservation).

### 4.4 Round-0 commit

Round 0 produces exactly one commit: `chore: pre-cleanup and dangling-ref purge`. Round 0 also creates a git tag `pre-cleanup-baseline` to make Tier-C output recovery a single command (see §6.1).

---

## 5. Loop architecture

### 5.1 Round structure

```
Round 0 (pre-loop, runs once):
  1. Run §4 cleanup sweep
  2. Add scripts/verify_repo.py and scripts/edit_notebook_markdown.py
  3. Commit: "chore: pre-cleanup and dangling-ref purge"
  4. Tag: pre-cleanup-baseline
  5. Initialize /tmp/ml-verify/state.json with empty findings history

Round N (N = 1..8):
  1. PLAN STEP
     - Read /tmp/ml-verify/round-(N-1)-findings.json (empty on N=1 → full verify first)
     - Group findings by area (docs, code, scripts, structure)
     - Decide round budget per §5.3 (single-area-per-round)
     - Write /tmp/ml-verify/round-N-plan.md

  2. FIX STEP
     - Apply edits per the plan
     - If a finding is in nnx/ or vendor/, log to docs/FINDINGS-NNX.md /
       docs/FINDINGS-VENDOR.md — do NOT fix

  3. VERIFY STEP
     - Run scripts/verify_repo.py
     - Rounds 1..N-1: --fast permitted if no code changes this round
     - Round N final: full suite, no shortcuts

  4. COMMIT STEP
     - If findings reduced from previous round OR doc/comment cleanup occurred:
         git commit -m "round N: <area> — fixed X findings, Y remain"
     - If findings unchanged (stall detector): see §5.2

  5. EXIT CHECK
     - If verify is fully green: commit "round N: final verify green", HALT success
     - If N == 8: HALT, report remaining findings
     - Else: increment N, return to PLAN STEP
```

### 5.2 Halt-on-stall

A round is **stalled** if its findings JSON is a (possibly reordered) superset of the previous round's. The loop halts on the second consecutive stalled round (one stall tolerated for retry, two means stuck).

When halted, the loop writes `/tmp/ml-verify/HALT.md` describing why and what remains. The user resumes manually.

### 5.3 One-area-per-round rule

Each round is one of:
- **D-round**: docs / markdown cells / READMEs only. Skips E1–E3 in verify.
- **C-round**: code + scripts + comment hygiene. Runs full verify.
- **S-round**: structure / `.gitignore` / dangling refs. Skips E1–E3.

Why: keeps each commit reviewable and isolates causality if E1–E3 break. Worst case this means doc-fixes finish in 2 D-rounds before a single C-round addresses code.

### 5.4 The verify script

`scripts/verify_repo.py` (added in round 0). Single entry point:

```
python scripts/verify_repo.py --check {structure|execution|docs|comments|all} \
                              [--fast] [--out /tmp/ml-verify/round-N-findings.json]
```

Exit code 0 = no findings; nonzero = findings present (count in stderr).

### 5.5 What the user does during the loop

Nothing, unless the loop halts. The /goal directive runs autonomously up to 8 rounds. The user inspects `git log` and `/tmp/ml-verify/round-*-report.md`.

---

## 6. Risks & tradeoffs

### 6.1 Tier-C re-execution risk

**Risk:** The loop edits markdown cells in `phase3-main-model-training-and-eval-notebook*.ipynb`. `nbformat.write` rewrites the whole file, so even a pure-markdown edit changes the file's bytes. If the COMMIT step ever runs the wrong papermill target, Aug-2023 training outputs are destroyed.

**Mitigations:**
- E5 (`git diff --quiet` on phase3 outputs) is asserted after every C/D round, not only at the end.
- The Tier-C smoke target is the only papermill path the loop is allowed to use on phase3 files; the loop NEVER runs `papermill phase3-*.ipynb phase3-*.ipynb`.
- Markdown-cell edits on Tier-C use `scripts/edit_notebook_markdown.py`, which mutates only `cell_type == "markdown"` cells and leaves all `code` cells (including outputs) byte-identical. E5 then validates that code-cell bytes are unchanged.
- Belt-and-braces: round 0 creates the git tag `pre-cleanup-baseline` so any misstep is recoverable via `git checkout pre-cleanup-baseline -- node_classification-reddit-gnn-pyg/phase3-*.ipynb`.

### 6.2 Comment-hygiene false positives

**Risk:** The Phase-B judge agent over-flags valid WHY comments, or under-flags state-the-what comments, causing over-deletion or stall.

**Mitigations:**
- Judge runs only on Phase-A survivors → smaller surface, less noise.
- Each judge verdict is logged with reasoning to the round report. The fix step does not auto-delete judge-flagged comments; it presents them as a batch and applies the §2.4 deterministic rule.
- Hard cap: a single round deletes at most 50 comments. Above that, halt and ask.

### 6.3 Loop runtime cost

**Risk:** 8 rounds × full Tier-A papermill (10–20 min) + judge agent = significant time and tokens.

**Mitigations:**
- Most rounds are D-rounds → skip E1–E3 → fast.
- Judge runs only on changed files per round → bounded.
- Realistic budget: 1 round 0 (~1 min) + ~3 D-rounds (~2 min each) + 1–2 C-rounds (~15 min each for full verify) + final full-verify round. Total ~45 min of compute, mostly idle.

### 6.4 NNx submodule findings get ignored forever

**Risk:** The loop only logs nnx findings to `docs/FINDINGS-NNX.md` without fixing.

**Mitigation:** `FINDINGS-NNX.md` is committed and surfaced in the final report. The success criterion does NOT require nnx findings to be empty — only ml-repo findings. nnx is a follow-up via an upstream PR, captured in a clear artifact.

### 6.5 Doc template forces synthetic structure

**Risk:** Some active notebooks (e.g., `phase1-dataset-exploration-notebook.ipynb`) genuinely don't have a model or training phase. Forcing §4–§6 produces filler.

**Mitigation:** §2.1 permits scoped variants for multi-phase notebooks. The verify check D1 carries a small lookup table mapping notebook path → required-sections-list. Adding a new task means adding a row to that table or accepting the default 6-section requirement.

### 6.6 Explicitly out of scope

- Rewriting `nnx` library code.
- Touching `archive/`.
- Adding new abstractions to `scripts/` beyond `verify_repo.py` and `edit_notebook_markdown.py`.
- 100% code coverage, perfect type hints, or any polish that isn't on the four checks.

---

## 7. Deliverables

- This spec doc (`docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md`), committed.
- An implementation plan (next step via `writing-plans`) — concrete checklist the loop executes round-by-round.
- The literal `/goal` directive in §8 below.

---

## 8. How to run

The /goal directive to paste:

```
/goal Run repo cleanup & doc standardization loop per
docs/superpowers/specs/2026-05-22-repo-cleanup-and-doc-standardization-design.md.

Exit when: scripts/verify_repo.py --check all returns exit code 0
           AND git status --short is empty
           AND the last commit message is "round N: final verify green".

Hard cap: 8 rounds.

Per-round procedure: §5.1 of the spec.
Halt conditions: §5.2 of the spec.
Edit boundaries: §1 of the spec (ml-repo only; nnx & vendor verify-only).
Tier-C output preservation: §6.1 of the spec (mandatory).
```

The loop runs in this worktree on the current branch (`main` of the worktree). It commits round-by-round. The user merges or opens a PR at the end.
