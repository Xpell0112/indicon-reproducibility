# Data (synthetic patients + public trial criteria)

INDICON 2026 (Paper ID 5232). Patient profiles are **synthetic** (template-generated, no real
patient data); trial inclusion/exclusion criteria derive from the **public TREC 2022 Clinical
Trials corpus** (ClinicalTrials.gov).

## Files and columns
| File | Rows (pairs) | Columns |
|---|---|---|
| `dev_train.csv` | 104 | `inclusion_criteria, exclusion_criteria, patient_id, patient_profiles, target` |
| `dev_validation.csv` | 26 | same as above |
| `testing_data.csv` | 270 | `pair_id, patient_id, trial_id, patient_profile, inclusion_criteria, exclusion_criteria, label_final` |
| `split_report.txt` | — | Human-readable split integrity report. |

`target` / `label_final` is the binary gold label (`Eligible` / `Not Eligible`). The gold labels
come from independent blind annotation; no per-annotator or provisional labels are included here.

## Counts (from `split_report.txt`)
- **Total:** 400 pairs, 71 patients, 40 trials.
- **Development:** 130 pairs, 26 patients, 22 trials (41 Eligible / 89 Not Eligible). Further split
  80/20, stratified by label, **seed 2026**, into dev-train (104) and dev-validation (26).
- **Held-out test:** 270 pairs, 45 patients, 18 trials — **69 Eligible / 201 Not Eligible**.
- **No overlap:** development and test share **0 patients and 0 trials** (trial-level disjoint).

Patients are synthetic; trial criteria are from the public TREC 2022 corpus — no real patient data.
