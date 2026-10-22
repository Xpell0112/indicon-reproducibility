"""
sql_gate_logic.py — Type-A gate RULE LOGIC (transparency excerpt)
=================================================================
INDICON 2026 (Paper ID 5232).

This is the actual implemented Type-A gate rule logic, extracted VERBATIM from the
full pipeline (app.py). It turns the parsed trial criteria + the validated patient
attributes into the structural decision g(P, T) in {0, 1}:

    build_condition_clauses()   -> per-condition NULL-guarded SQL clause fragments,
                                   restricted to Type-A conditions (see classifier.py)
    build_current_patient_cte() -> a one-row `current_patient` CTE from the validated
                                   attributes
    sql_eligibility_check()     -> ANDs the Type-A clauses and evaluates them as real
                                   PostgreSQL; returns g(P, T)

TRANSPARENCY ONLY — not standalone-runnable. The following helpers are defined
elsewhere in the full codebase and are referenced here as-is:
  - CONDITION_TYPE_MAP        (structural_verifier/classifier.py)
  - escape_sql_value()        (SQL literal escaping, app.py)
  - expand_allowed_stages()   (substage subsumption + TN<->group equivalence, app.py)
  - execute_boolean_check()   (runs the one-row boolean query in PostgreSQL, src/db/repository.py)
`sql_eligibility_check` is a method of AgenticTrialMatcher in app.py; it is shown here as a
function (the `self` receiver carries no state used by the gate).

Confidence tiers that feed `validated_attrs` (Tier-1 >= 0.80 accept / Tier-2 0.30-0.80 LLM
re-read / Tier-3 < 0.30 review-refuse) and the Type-A/Type-B field map live in
validation_agent.py and classifier.py in this same folder.
"""

import re


def build_condition_clauses(parsed_criteria, only_types=None):
    """
    Builds the same tagged SQL clause fragments sql_generation_agent has
    always produced, each labelled with the condition it came from so
    callers can filter by Type A / Type B (see src.agents.classifier).

    only_types=None returns every clause in the original order — this is
    exactly what sql_generation_agent used to build inline, so the cohort/
    informational query's behavior is unchanged.
    only_types={"A"} restricts to Type-A (structural) clauses only — used
    by sql_eligibility_check() for the per-patient gate.

    Returns: list[(tag, clause_sql)]
    """
    inclusion = parsed_criteria.get("parsed_inclusion_conditions", {}) or {}
    exclusion = parsed_criteria.get("parsed_exclusion_conditions", {}) or {}

    tagged = []

    def add(tag, clause):
        if only_types is None or CONDITION_TYPE_MAP.get(tag, "B") in only_types:
            tagged.append((tag, clause))

    # Disease gate is NULL-tolerant: a patient whose disease wasn't confidently
    # extracted (disease IS NULL / '') is NOT failed here — only a patient with
    # a confidently-different disease is. This keeps the Type-A gate from
    # rejecting on missing data and defers ambiguous cases to the ensemble.
    # Lung subtypes (NSCLC/SCLC/LUNG) are treated as a compatible family so a
    # patient tagged "LUNG" isn't failed against an "NSCLC" trial on tokenization
    # alone (mirrors clinical_reasoning_agent's disease check).
    trial_dz = inclusion.get("disease")
    if trial_dz:
        lung_family = {"NSCLC", "SCLC", "LUNG"}
        accepted = lung_family if trial_dz in lung_family else {trial_dz}
        in_list = ", ".join(f"'{escape_sql_value(d)}'" for d in sorted(accepted))
        add("disease", f"(disease IN ({in_list}) OR disease IS NULL OR disease = '')")

    stage = inclusion.get("stage", [])
    if isinstance(stage, str):
        stage = [stage] if stage else []   # backward-compat guard

    if stage:
        # Expand with substage subsumption + TN↔group equivalence (RC2) so the
        # gate matches "IIIB" against an allowed "III", and "IA" against an
        # allowed "T1N0", exactly like the deterministic reasoner does.
        expanded = sorted(expand_allowed_stages(stage))
        tn_stages = [s for s in expanded if re.match(r"^T\d+[A-Z]?N\d+$", s)]
        group_stages = [s for s in expanded if not re.match(r"^T\d+[A-Z]?N\d+$", s)]
        sub_clauses = []
        if tn_stages:
            tn_list = ", ".join(f"'{escape_sql_value(s)}'" for s in tn_stages)
            sub_clauses.append(f"stage IN ({tn_list})")
            sub_clauses.append(f"stage_group IN ({tn_list})")
        if group_stages:
            grp_list = ", ".join(f"'{escape_sql_value(s)}'" for s in group_stages)
            sub_clauses.append(f"stage_group IN ({grp_list})")
        if sub_clauses:
            add("stage", "(" + " OR ".join(sub_clauses) + ")")

    # M_status is now "M0" | "M1" | None (no longer hardcoded to "M0").
    # None → emit NO M clause (trial has no M-stage requirement, e.g. many
    # Stage IV trials carry "metastatic" via the stage field instead). This
    # is what stops the SQL gate from rejecting genuinely-metastatic (M1c)
    # patients on a fabricated M0 requirement.
    m_status = (inclusion.get("M_status") or "").upper()
    if m_status == "M0":
        add("M_status", "(M_stage = '0' OR M_stage IS NULL OR M_stage = '')")
    elif m_status == "M1":
        # Metastatic required — NULL-tolerant so a patient with no recorded
        # M-stage is not rejected on missing data.
        add("M_status", "(M_stage IS NULL OR M_stage <> '0')")

    fev1_cc_min = inclusion.get("FEV1_cc_min")
    fev1_pct_min = inclusion.get("FEV1_pct_min")
    if fev1_cc_min is not None and fev1_pct_min is not None:
        add("FEV1", f"(FEV1_cc >= {int(fev1_cc_min)} OR FEV1_pct >= {int(fev1_pct_min)} "
                     f"OR (FEV1_cc IS NULL AND FEV1_pct IS NULL))")
    elif fev1_cc_min is not None:
        add("FEV1", f"(FEV1_cc >= {int(fev1_cc_min)} OR FEV1_cc IS NULL)")
    elif fev1_pct_min is not None:
        add("FEV1", f"(FEV1_pct >= {int(fev1_pct_min)} OR FEV1_pct IS NULL)")

    perf = str(inclusion.get("performance_status", "")).strip().lower()
    perf = perf.replace("–", "-").replace("—", "-")

    # Any "lo-hi" ECOG range → ceiling is hi (handles "0-1", "0-2", and also
    # non-zero-based ranges like "1-2"). Falls through to KPS / single below.
    # NULL-tolerant like the other structural clauses: a patient whose PS
    # wasn't extracted (performance_status_int IS NULL) must NOT poison the
    # AND with a SQL NULL (which yields "no boolean result" and was silently
    # failing the gate closed). Missing PS defers to the ensemble instead.
    def _ps_clause(threshold):
        return (f"(performance_status_int <= {int(threshold)} "
                f"OR performance_status_int IS NULL)")

    _ps_range = re.search(r"(\d)\s*(?:-|to)\s*(\d)", perf)
    if _ps_range and "kps" not in perf:
        add("performance_status", _ps_clause(_ps_range.group(2)))
    elif perf.startswith("kps "):
        kps_m = re.search(r"kps\s*(\d+)", perf)
        if kps_m:
            kps_val = int(kps_m.group(1))
            if kps_val >= 90:
                add("performance_status", _ps_clause(0))
            elif kps_val >= 70:
                add("performance_status", _ps_clause(1))
            elif kps_val >= 50:
                add("performance_status", _ps_clause(2))
            else:
                add("performance_status", _ps_clause(3))
    elif re.search(r"\b[0-4]\b", perf):
        threshold = int(re.search(r"\b([0-4])\b", perf).group(1))
        add("performance_status", _ps_clause(threshold))

    if "no prior" in inclusion.get("prior_treatment_history", "").lower():
        add("prior_treatment_history", "prior_chemotherapy = 0")
        add("prior_treatment_history", "prior_radiotherapy = 0")
        add("prior_treatment_history", "prior_surgery = 0")

    if exclusion.get("prior_systemic_chemotherapy", "NOT_AN_EXCLUSION") == "EXCLUDED_IF_PRESENT":
        add("prior_systemic_chemotherapy", "prior_chemotherapy = 0")

    if exclusion.get("prior_radiotherapy", "NOT_AN_EXCLUSION") == "EXCLUDED_IF_PRESENT":
        add("prior_radiotherapy", "prior_radiotherapy = 0")

    if exclusion.get("prior_targeted_therapy", "NOT_AN_EXCLUSION") == "EXCLUDED_IF_PRESENT":
        add("prior_targeted_therapy", "prior_targeted_therapy = 0")

    if exclusion.get("active_pulmonary_infection", "NOT_AN_EXCLUSION") == "EXCLUDED_IF_PRESENT":
        add("active_pulmonary_infection", "active_infection = 0")

    if exclusion.get("cardiovascular_event", "NOT_AN_EXCLUSION") == "EXCLUDED_IF_PRESENT":
        add("cardiovascular_event", "cardiovascular_event = 0")

    # Histology gate (RC4): trials restricted to non-squamous histology (or
    # explicitly excluding squamous) reject a squamous patient. NULL/unknown
    # histology is tolerated (deferred to the ensemble), not failed.
    if inclusion.get("requires_non_squamous"):
        add("requires_non_squamous", "(histology <> 'squamous' OR histology IS NULL OR histology = '')")

    # Mirror restriction: squamous-only trial rejects a non-squamous patient.
    if inclusion.get("requires_squamous"):
        add("requires_squamous", "(histology <> 'non-squamous' OR histology IS NULL OR histology = '')")

    # Conditional Stage-IIIB: when a trial admits IIIB only WITH a malignant
    # effusion, a patient who is IIIB/III must actually have one; Stage-IV (and
    # any non-IIIB) patients are unaffected. NULL-safe on has_effusion.
    if inclusion.get("iiib_requires_effusion"):
        add("iiib_requires_effusion",
            "(stage_group NOT IN ('IIIB','III') OR has_effusion = 1)")

    # Line-of-therapy gate (RC3): a second-line trial whose inclusion REQUIRES
    # prior chemotherapy ("failed ≥1 front-line regimen") rejects a chemo-naive
    # patient. This is the mirror image of the prior-chemo *exclusion* — here
    # prior chemo is mandatory, so prior_chemotherapy must be 1.
    if inclusion.get("requires_prior_chemo"):
        add("requires_prior_chemo", "prior_chemotherapy = 1")

    neuropathy = exclusion.get("neuropathy_grade", "")
    if neuropathy.startswith("<="):
        try:
            threshold = int(re.sub(r"[^0-9]", "", neuropathy))
            add("neuropathy_grade", f"(neuropathy_grade <= {threshold} OR neuropathy_grade IS NULL)")
        except ValueError:
            pass
    elif neuropathy.startswith(">"):
        try:
            threshold = int(re.sub(r"[^0-9]", "", neuropathy))
            add("neuropathy_grade", f"(neuropathy_grade <= {threshold} OR neuropathy_grade IS NULL)")
        except ValueError:
            pass

    return tagged


def _sql_literal(value, cast):
    """Render a Python value as a typed SQL literal, NULL-safe."""
    if value is None or value == "":
        return f"NULL::{cast}"
    if cast == "int":
        try:
            return f"{int(value)}::int"
        except (ValueError, TypeError):
            return "NULL::int"
    return f"'{escape_sql_value(value)}'::text"


def build_current_patient_cte(validated_attrs):
    """
    Synthesises a one-row `current_patient` CTE from this run's own
    Phase-4-validated attributes. Lets the exact same Type-A clause
    fragments build_condition_clauses() produces for the cohort query
    against `patients` also run as real Postgres SQL for a genuinely new
    patient who isn't (and may never be) a row in that table.
    """
    validated_attrs = validated_attrs or {}

    # Defensive: strip a stray leading letter (e.g. an LLM-corrected value
    # of "N2" instead of "2") before re-prefixing below — otherwise this
    # silently doubles up into "T2NN2", matching no allowed stage at all.
    t_stage = str(validated_attrs.get("T_stage") or "").upper().removeprefix("T") or None
    n_stage = str(validated_attrs.get("N_stage") or "").upper().removeprefix("N") or None
    stage = f"T{t_stage}N{n_stage}" if t_stage and n_stage else None

    ecog = validated_attrs.get("performance_status_ecog")
    if ecog is None:
        raw = validated_attrs.get("performance_status")
        if isinstance(raw, (int, str)) and str(raw).strip().lstrip("-").isdigit():
            raw_int = int(raw)
            ecog = raw_int if 0 <= raw_int <= 4 else None

    columns = {
        "disease": ("text", validated_attrs.get("disease")),
        "histology": ("text", validated_attrs.get("histology")),
        "has_effusion": ("int", int(bool(validated_attrs.get("has_effusion")))),
        "stage": ("text", stage),
        "stage_group": ("text", validated_attrs.get("stage_group")),
        "M_stage": ("text", validated_attrs.get("M_stage")),
        "FEV1_cc": ("int", validated_attrs.get("FEV1_cc")),
        "FEV1_pct": ("int", validated_attrs.get("FEV1_pct")),
        "performance_status_int": ("int", ecog),
        "prior_chemotherapy": ("int", int(bool(validated_attrs.get("prior_chemotherapy")))),
        "prior_radiotherapy": ("int", int(bool(validated_attrs.get("prior_radiotherapy")))),
        "prior_targeted_therapy": ("int", int(bool(validated_attrs.get("prior_targeted_therapy")))),
        "prior_surgery": ("int", int(bool(validated_attrs.get("prior_surgery")))),
        "active_infection": ("int", int(bool(validated_attrs.get("active_infection")))),
        "cardiovascular_event": ("int", int(bool(validated_attrs.get("cardiovascular_event")))),
        "neuropathy_grade": ("int", validated_attrs.get("neuropathy_grade")),
    }

    select_list = ", ".join(
        f"{_sql_literal(value, cast)} AS {col}" for col, (cast, value) in columns.items()
    )
    return f"WITH current_patient AS (SELECT {select_list})"


def sql_eligibility_check(parsed_criteria, validated_attrs):
    """
    Real per-patient SQL Type-A gate. Builds the exact same NULL-guarded
    clause fragments sql_generation_agent uses for the cohort/
    informational query, restricted to Type-A conditions (see
    src.agents.classifier), and evaluates them as genuine Postgres SQL
    against a synthetic one-row `current_patient` CTE built from this
    run's own validated attributes — so the check is real even for a
    patient who was never a row in the `patients` table.

    Age is intentionally NOT part of this gate: the `patients` table has
    no age column, and build_condition_clauses() must keep producing the
    identical clause set sql_generation_agent has always used. Age is
    already checked deterministically in clinical_reasoning_agent, whose
    result feeds the ensemble's Type-B verdict.

    Returns: (is_eligible: Optional[bool], sql_query: str, error: Optional[str])
    g(P, T) = 1 when is_eligible is True; 0 otherwise.

    NOTE: a method of AgenticTrialMatcher in app.py; shown here as a function.
    """
    tagged_clauses = build_condition_clauses(parsed_criteria, only_types={"A"})
    clauses = [clause for _, clause in tagged_clauses]
    if not clauses:
        # No Type-A conditions parsed for this trial — vacuously
        # satisfied, consistent with sql_generation_agent's own
        # "1=1" fallback for an empty clause list.
        clauses = ["1=1"]

    cte = build_current_patient_cte(validated_attrs)
    boolean_expr = " AND ".join(clauses)
    sql_query = f"{cte}\nSELECT ({boolean_expr}) AS is_eligible FROM current_patient;"

    is_eligible, error = execute_boolean_check(sql_query)
    return is_eligible, sql_query, error
