# Analysis scripts

INDICON 2026 (Paper ID 5232). Cache-only: each script reads `../frozen_predictions/` and writes
JSON + Markdown to `../results/`. No API calls, no database, no test-set fitting. Deterministic
(exact permutation enumeration where feasible; fixed seed 20260815 for bootstrap and Monte-Carlo).

## Run
```bash
cd analysis
python exp9_significance.py
python exp10_revision_fixes.py
python exp11_robustness.py
```
Requires only `numpy` (see `../requirements.txt`).

## What each reproduces
| Script | Source | Output (in ../results) | Paper items |
|---|---|---|---|
| `exp9_significance.py` | `experiments/exp9_significance.py` | `experiment9_significance_testing_data.{json,md}` | McNemar p-values; F1 95% CIs (trial/patient/pair cluster bootstrap, B=5000, seed 20260815); dev-only advisory justification |
| `exp10_revision_fixes.py` | `experiments/exp10_revision_fixes.py` | `experiment10_revision_fixes_testing_data.{json,md}` | Table III accuracies (0.878 / 0.919, N=270 convention); Table IV cluster-robust paired tests; per-system confusion matrices |
| `exp11_robustness.py` | `experiments/exp11_robustness.py` | `experiment11_robustness_testing_data.{json,md}` | Leave-one-trial-out robustness (advisory beats hard veto 18/18; advisory never beats zero-shot 0/18) |

Only the input (`CACHE`) and output (`RES`) paths were changed from the originals to point at
`../frozen_predictions` and `../results`; all computation is unchanged.

The scripts also write a `.md` companion next to each JSON when run.
