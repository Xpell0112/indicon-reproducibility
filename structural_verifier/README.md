# Structural verifier (transparency)

INDICON 2026 (Paper ID 5232). These modules implement the Type-A structural path that
produces the gate decision g(P, T) ∈ {0, 1}. They are included **for transparency**; running
them live requires the full serving stack (LLM API + PostgreSQL + Pinecone) and is **not**
needed to reproduce the reported statistics (those reproduce from `../frozen_predictions`).

## Files
| File | Source | Content |
|---|---|---|
| `classifier.py` | `src/agents/classifier.py` | `CONDITION_TYPE_MAP` (Type-A vs Type-B tags) and `classify_conditions` (field classification). |
| `validation_agent.py` | `src/agents/validation_agent.py` | `CRITICAL_FIELDS`, `EVIDENCE_PATTERNS` (deterministic regex), the confidence tiers, and `_llm_reextract`. |
| `sql_gate_logic.py` | `app.py` (verbatim excerpt) | The actual Type-A **gate rule logic**: `build_condition_clauses` (per-condition NULL-guarded SQL clauses), `build_current_patient_cte`, and `sql_eligibility_check` (produces g(P, T) ∈ {0, 1}). |

## How it works
- **Deterministic text patterns → Type-A fields.** `EVIDENCE_PATTERNS` (regex) extract patient
  attributes; `CONDITION_TYPE_MAP` labels each parsed trial condition Type-A (structural,
  SQL-checkable) or Type-B (semantic/temporal — left to the LLM).
- **Five critical attributes** (`CRITICAL_FIELDS`): performance status, cancer stage (`stage_group`),
  prior chemotherapy, prior radiotherapy, active infection. A Tier-3 result on any of these blocks
  the pipeline.
- **Confidence tiers** (`validation_agent.py`): Tier-1 `≥ 0.80` accept the regex value (no LLM);
  Tier-2 `0.30–0.80` LLM re-read (`_llm_reextract`); Tier-3 `< 0.30` review/refuse.
- **SQL gate (rule logic in `sql_gate_logic.py`):** `build_condition_clauses` turns each parsed
  Type-A condition into a NULL-guarded boolean SQL clause (disease family, stage subsumption,
  M-status, FEV1, performance-status ceiling, prior-therapy exclusions, histology restriction,
  conditional IIIB-requires-effusion, line-of-therapy `requires_prior_chemo`, neuropathy grade);
  `build_current_patient_cte` builds a one-row patient; `sql_eligibility_check` ANDs the clauses and
  evaluates them in PostgreSQL, producing `g(P, T) ∈ {0, 1}` (1 = passes all applicable Type-A
  checks). This is the full structural path, not just the field classification.

No secrets are present in these files (all credentials come from environment variables).
