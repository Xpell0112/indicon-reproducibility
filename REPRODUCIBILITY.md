# REPRODUCIBILITY — INDICON 2026 (Paper ID 5232)

Reproducing the reported statistics needs only NumPy and the frozen predictions. No API, no
database, no model calls, no test-set fitting.

## 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy==2.2.1 ; tested on CPython 3.13.1
```

## 2. Run the analysis and the exact numbers each reproduces
```bash
cd analysis
python exp9_significance.py
python exp10_revision_fixes.py
python exp11_robustness.py
```
Outputs are written to `../results/` and match the included JSONs.

### `exp10_revision_fixes.py` → `results/experiment10_revision_fixes_testing_data.json`
- **Table III** (same-run, N=270 accuracy convention, refusals counted incorrect):
  - Hard veto: Acc **0.878**, Prec **0.882**, Rec **0.682**, F1 **0.769**, **FN 21**.
  - Advisory: Acc **0.919**, Prec **0.866**, Rec **0.879**, F1 **0.872**, **FN 8**.
- **Table IV** (paired significance):
  - Hard veto vs zero-shot: McNemar **.0015**, trial-cluster **.018**, Acc diff **−.063 [−.103, −.024]**.
  - Advisory vs zero-shot: **.057 / .156**.
  - Advisory vs hard veto: **.019 / .016**.
- **Patient-cluster sensitivity:** advisory vs hard veto **0.018**; advisory vs zero-shot **0.055**.
- **Confusion matrices (TP/FP/FN/TN):** zero-shot 64/9/5/192; few-shot 65/9/4/192; RAG 61/6/8/195;
  structural 52/95/17/106; hard veto (full) 45/4/21/194; hard veto (same-run) 45/6/21/192;
  advisory 58/9/8/190.

### `exp9_significance.py` → `results/experiment9_significance_testing_data.json`
- **Table I** F1 (from the per-system block): zero-shot **0.901**, few-shot **0.909**, RAG **0.897**,
  structural **0.481**, hard veto **0.783**; hard-veto coverage **0.978**.
- **McNemar** exact: hard-veto vs zero-shot **0.0015**; advisory vs zero-shot **0.057**;
  advisory vs hard-veto **0.019**.
- **F1 95% CIs** (trial-cluster bootstrap, B=5000, seed 20260815), e.g. zero-shot [0.838, 0.963],
  hard veto [0.623, 0.885], advisory [0.824, 0.922].

### `exp11_robustness.py` → `results/experiment11_robustness_testing_data.json`
- **Leave-one-trial-out:** advisory beats hard veto **18/18**; advisory never beats zero-shot
  **0/18**; hard veto below zero-shot **18/18**.

### `results/gate_failure_groupAB_testing_data.json` (precomputed)
- **Table II** structural verifier: development folds **0 FN**; held-out **17 FN**, recall **0.754**.
- The 17 structural false negatives split **Group A / Group B = 9 / 8**.

## 3. Seeds
- Data split: **2026** (80/20 stratified dev split).
- Bootstrap / permutation: **20260815**.
- Patient generation: **20260718** (generator default).

## 4. LLM non-determinism
Serving runs at temperature 0, but two temperature-0 passes over the 270 test pairs agreed on
**264/270** — live regeneration is not bit-identical. Frozen predictions are therefore provided so
the statistics above reproduce **exactly**; do not expect a live re-run of the pipeline to match to
the last digit.
