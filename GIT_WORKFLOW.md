# Git Workflow Documentation — Lab 3

**Project:** churn-prediction (Telco Customer Churn MLOps Pipeline)
**Lab:** Lab 3 — Git-Based Version Control and Collaborative ML Workflows

This document records the version-control workflows demonstrated in this
repository: repository initialization, modular branching, simulated
collaborative development, merge-conflict resolution, and faulty-commit
identification with rollback.

---

## 1. Repository Initialization

The repository was initialized with a baseline preprocessing and modeling
workflow for the Telco Customer Churn dataset.

| Commit | Message |
|---|---|
| `d35d492` | Initial commit: Completed Lab 1 and Lab 2 baseline model |

---

## 2. Branching Strategy

Two feature branches were created off `main` to isolate independent lines
of work, following the modular-component principle (preprocessing, training,
evaluation kept as separate concerns):

- **`feature/modular-scripts`** — split the monolithic notebook logic into
  reusable `src/preprocess.py`, `src/train.py`, and `src/evaluate.py`
  modules.
- **`feature/fill-median`** — an experimental branch created to try an
  alternative missing-value strategy for `TotalCharges` (see Section 3).

`feature/modular-scripts` was merged back into `main` as a fast-forward
merge once the modular refactor was complete:

| Commit | Message |
|---|---|
| `b43eb22` | Added modular preprocessing, training and evaluation scripts |

---

## 3. Simulated Collaborative Development & Merge Conflict

To demonstrate a realistic multi-engineer workflow, two independent,
conflicting changes were made to the same line of `src/preprocess.py` —
how to handle missing `TotalCharges` values:

| Branch | Commit | Change |
|---|---|---|
| `feature/fill-median` | `0b36a77` | "Engineer 1: Fill missing TotalCharges with median" |
| `main` | `ddf7bb5` | "Engineer 2: Fill missing TotalCharges with mean" |

Merging `feature/fill-median` into `main` produced a genuine conflict on
the `TotalCharges` imputation line, since both branches had modified it
differently. The conflict was resolved manually — neither the median nor
the mean approach was kept; instead, filling with `0` was chosen, since
all blank `TotalCharges` values correspond to customers with zero tenure
(confirmed during EDA), making `0` the semantically correct value rather
than an imputed statistic.

| Commit | Message |
|---|---|
| `ea5f0f2` (merge commit) | "Resolve merge conflict: Fill missing TotalCharges with 0" |

```
*   ea5f0f2 Resolve merge conflict: Fill missing TotalCharges with 0
|\
| * 0b36a77 Engineer 1: Fill missing TotalCharges with median
* | ddf7bb5 Engineer 2: Fill missing TotalCharges with mean
|/
* b43eb22 Added modular preprocessing, training and evaluation scripts
```

Repository consistency after the merge was validated by re-running
`src/preprocess.py` and confirming no errors and no missing values
remained in the processed output.

---

## 4. Faulty Commit Identification and Rollback

To demonstrate fault detection and recovery — a distinct skill from
merging — a regression was intentionally introduced into
`src/preprocess.py`, committed, detected via the project's own validation
tooling, and reverted.

**4.1 — The faulty change.** The `.fillna(0)` call was removed from the
`TotalCharges` cleaning step, so blank values were coerced to `NaN` but
never filled:

```diff
- df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
+ df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

This was committed under an innocuous, refactor-style message —
representative of how faulty commits actually slip into a real history:

| Commit | Message |
|---|---|
| `<FAULTY_COMMIT_HASH>` | "Refactor: streamline TotalCharges numeric conversion" |

**4.2 — Detecting the fault.** Running the pipeline did **not** raise any
error — the script printed "Preprocessing completed successfully!" — but
running the existing Lab 5 validation script (`src/validate_outputs.py`)
against the resulting arrays caught the corruption immediately:

```
[INFO] Validating Preprocessing Outputs...
[ERROR] Output Validation FAILED!
  - NaNs detected in X_train after preprocessing.
  - NaNs detected in X_test after preprocessing.
```

This is the intended failure mode being demonstrated: a commit that looks
harmless and doesn't crash anything, but silently corrupts data
downstream — exactly the kind of regression version control and
validation tooling together are meant to catch.

**4.3 — Identifying the faulty commit.** `git log` and `git show` were
used to locate the exact change responsible:

```bash
git log --oneline -5
git show <FAULTY_COMMIT_HASH>
```

**4.4 — Reverting.** Rather than `git reset --hard` (which rewrites
history and would delete the record of the mistake), `git revert` was
used to create a new commit that undoes the faulty change while keeping
full, honest history — the safer choice for a shared or graded repository:

```bash
git revert --no-edit <FAULTY_COMMIT_HASH>
```

| Commit | Message |
|---|---|
| `<REVERT_COMMIT_HASH>` | "Revert 'Refactor: streamline TotalCharges numeric conversion'" |

**4.5 — Confirming the fix.** The pipeline and validation script were
re-run after the revert:

```
Preprocessing completed successfully!
[INFO] Validating Preprocessing Outputs...
[SUCCESS] Output Validation PASSED.
[INFO] Features are clean, scaled, encoded, and dimensionally consistent.
```

Resulting history:

```
<REVERT_COMMIT_HASH> Revert "Refactor: streamline TotalCharges numeric conversion"
<FAULTY_COMMIT_HASH> Refactor: streamline TotalCharges numeric conversion
ea0c329 Added Lab 6 orchestration pipeline for model registry and lifecycle management
...
```

---

## 5. Summary

| Workflow demonstrated | Evidence |
|---|---|
| Repository initialization | `d35d492` |
| Modular branching | `feature/modular-scripts` → `b43eb22` |
| Independent collaborative changes | `0b36a77` (Engineer 1), `ddf7bb5` (Engineer 2) |
| Merge conflict creation & manual resolution | `ea5f0f2` |
| Faulty commit introduction | `<FAULTY_COMMIT_HASH>` |
| Fault detection via validation tooling | `src/validate_outputs.py` output |
| Rollback via `git revert` | `<REVERT_COMMIT_HASH>` |
| Post-revert validation | `src/validate_outputs.py` → PASSED |

This workflow reflects standard practice for collaborative ML engineering:
isolate work in branches, resolve conflicts deliberately rather than by
default preference, and treat "the pipeline ran without crashing" as
insufficient proof of correctness — validation tooling and version
control together are what make a faulty change catchable and reversible.
