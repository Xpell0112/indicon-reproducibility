"""
src/agents/classifier.py
=========================
Criteria classifier (Type A / Type B).

Deterministic, no LLM call. Tags each parsed trial condition as:
  Type A — structural/exact, directly SQL-checkable (disease, stage, FEV1,
           age, performance status, and the flat yes/no prior-therapy /
           infection exclusion flags).
  Type B — semantic/temporal/judgment (neuropathy grade boundary comparison,
           cardiovascular event recency, free-text catch-all conditions,
           and the inclusion-side negation-based "no prior treatment"
           phrasing). These stay on the LLM/ensemble side of the decision.

CONDITION_TYPE_MAP is the single source of truth for condition typing —
app.py's build_condition_clauses() (SQL clause building) and the Gradio
"Type A / Type B" panel both read from it so the two never drift apart.
"""

CONDITION_TYPE_MAP: dict[str, str] = {
    # Inclusion
    "disease": "A",
    "stage": "A",
    "M_status": "A",
    "FEV1": "A",
    "age_min": "A",
    "performance_status": "A",
    # Matches the proposal's own worked example (§5.2, Mr. Anderson): prior
    # chemotherapy/radio/targeted therapy is classified Type A there, not B —
    # only cardiac timing and neuropathy are called out as Type B.
    "prior_treatment_history": "A",   # inclusion-side "no prior chemotherapy" flag
    "requires_prior_chemo": "A",      # second-line trial: prior chemo REQUIRED
    "requires_non_squamous": "A",     # non-squamous-only histology restriction
    "requires_squamous": "A",         # squamous-only histology restriction
    "iiib_requires_effusion": "A",    # IIIB admitted only WITH malignant effusion
    "additional_inclusion": "B",      # free-text catch-all (organ function, labs, ...)
    # Exclusion
    "prior_systemic_chemotherapy": "A",
    "prior_radiotherapy": "A",
    "prior_targeted_therapy": "A",
    "active_pulmonary_infection": "A",
    "cardiovascular_event": "B",      # clinically time-bound even though extraction is a flat boolean
    "neuropathy_grade": "B",          # boundary comparison ("not strictly > Grade 1")
    "other_exclusion_terms": "B",     # free-text catch-all
}


def _is_yes(value) -> bool:
    return str(value or "No").strip().lower() == "yes"


def classify_conditions(parsed_criteria: dict) -> dict:
    """
    Returns {"Type A": {tag: value_repr, ...}, "Type B": {tag: value_repr, ...}}
    restricted to conditions this trial actually specifies (falsy/empty
    values are skipped so the panel only shows what's actually being tested).
    """
    inclusion = parsed_criteria.get("parsed_inclusion_conditions", {}) or {}
    exclusion = parsed_criteria.get("parsed_exclusion_conditions", {}) or {}

    fev1_present = (
        inclusion.get("FEV1_cc_min") is not None
        or inclusion.get("FEV1_pct_min") is not None
    )

    candidates = {
        "disease": inclusion.get("disease"),
        "stage": inclusion.get("stage"),
        "M_status": inclusion.get("M_status"),
        "FEV1": "present" if fev1_present else None,
        "age_min": inclusion.get("age_min"),
        "performance_status": inclusion.get("performance_status"),
        "prior_treatment_history": inclusion.get("prior_treatment_history"),
        "requires_prior_chemo": inclusion.get("requires_prior_chemo"),
        "requires_non_squamous": inclusion.get("requires_non_squamous"),
        "requires_squamous": inclusion.get("requires_squamous"),
        "iiib_requires_effusion": inclusion.get("iiib_requires_effusion"),
        "additional_inclusion": inclusion.get("additional_inclusion"),
        "prior_systemic_chemotherapy": (
            exclusion.get("prior_systemic_chemotherapy")
            if _is_yes(exclusion.get("prior_systemic_chemotherapy"))
            else None
        ),
        "prior_radiotherapy": (
            exclusion.get("prior_radiotherapy")
            if _is_yes(exclusion.get("prior_radiotherapy"))
            else None
        ),
        "prior_targeted_therapy": (
            exclusion.get("prior_targeted_therapy")
            if _is_yes(exclusion.get("prior_targeted_therapy"))
            else None
        ),
        "active_pulmonary_infection": (
            exclusion.get("active_pulmonary_infection")
            if _is_yes(exclusion.get("active_pulmonary_infection"))
            else None
        ),
        "cardiovascular_event": (
            exclusion.get("cardiovascular_event")
            if _is_yes(exclusion.get("cardiovascular_event"))
            else None
        ),
        "neuropathy_grade": exclusion.get("neuropathy_grade"),
        "other_exclusion_terms": exclusion.get("other_exclusion_terms"),
    }

    result = {"Type A": {}, "Type B": {}}
    for tag, value in candidates.items():
        if not value:
            continue
        type_label = CONDITION_TYPE_MAP.get(tag, "B")
        bucket = "Type A" if type_label == "A" else "Type B"
        result[bucket][tag] = value

    return result


def format_classification(classification: dict) -> str:
    """Human-readable panel text for the Gradio UI / logs."""
    lines = ["Type A (structural — SQL-checkable):"]
    if classification.get("Type A"):
        for tag, value in classification["Type A"].items():
            lines.append(f"  • {tag}: {value}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Type B (semantic / temporal / judgment — LLM reasoning):")
    if classification.get("Type B"):
        for tag, value in classification["Type B"].items():
            lines.append(f"  • {tag}: {value}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)
