# Frozen predictions (the exact inputs the analysis scripts load)

INDICON 2026 (Paper ID 5232). These are the frozen per-pair model outputs. All reported
statistics reproduce from these files alone — no API, no database, no model calls. Filenames are
kept as the analysis scripts expect them.

Source of every file: `experiments/results/_cache/<same name>` in the original repository.

## Test set (270 pairs) — used for all reported test numbers
| File | Condition(s) | Rows | Key fields |
|---|---|---|---|
| `experiment1_testing_data_zero_shot.jsonl` | Zero-shot LLM | 270 | `prediction, confidence, brier, llm_calls, pair_id, target, trial_id` |
| `experiment1_testing_data_few_shot.jsonl` | Few-shot LLM | 270 | same |
| `experiment1_testing_data_rag_only.jsonl` | Retrieval-augmented LLM | 270 | same |
| `experiment1_testing_data_symbolic_only.jsonl` | Structural-only (Type-A gate) | 270 | same (`prediction` from the gate) |
| `experiment1_testing_data_full.jsonl` | Full framework (hard veto) | 270 | `… confidence_blocked …` + same |
| `experiment8_testing_data_advisory.jsonl` | **Advisory AND hard-veto, same run** | **270** | `veto_pred, veto_conf, adv_pred, adv_conf, ensemble_pred, gate_pass, target, pair_id, trial_id` |

The same-run policy comparison (Table III, Table IV) reads **both** `veto_pred` and `adv_pred` from
`experiment8_testing_data_advisory.jsonl`. This is the **270-pair TEST** file — verify by its 270
rows. It is NOT the 130-dev file below.

## Development set (130 pairs) — used only for the exp9 dev-justification section
| File | Condition | Rows | Key fields |
|---|---|---|---|
| `experiment8_dev130_advisory.jsonl` | Advisory vs hard-veto on development | 130 | same keys as the test advisory file |

`prediction`/`*_pred` values are `"Eligible"`, `"Not Eligible"`, or `null` (a refusal). A refusal is
counted as incorrect for accuracy and excluded from precision/recall/F1.
