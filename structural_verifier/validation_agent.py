"""
src/agents/validation_agent.py
================================
Phase 4 — Clinical Attribute Validation Agent

RESEARCH CONTRIBUTION:
    Introduces a verification-driven architecture between attribute extraction
    and clinical reasoning. Replaces the implicit trust model (regex output =
    ground truth) with a three-tier confidence-gated pipeline:

        Tier 1 (conf ≥ 0.80) → Accept regex value directly
        Tier 2 (conf 0.30–0.79) → Targeted LLM re-extraction for this field
        Tier 3 (conf < 0.30) → Flag for human review, block pipeline

    Confidence scoring is hybrid:
        • Deterministic component: range validity + evidence span presence +
          cross-field consistency check → base_confidence (0.0–1.0)
        • LLM adjustment: only invoked when base_confidence is in Tier 2;
          the LLM returns its own confidence which is averaged with the
          deterministic score to form the final decision.

    This design avoids unnecessary LLM calls (Tier 1 fields pass through
    with zero LLM cost), corrects common extraction errors automatically
    (Tier 2), and refuses to fabricate uncertain clinical data (Tier 3).

PERFORMANCE SCALE HANDLING:
    Clinical profiles may use either:
        • ECOG/Zubrod scale: 0–4 (integer)
        • Karnofsky Performance Status (KPS): 0–100 (multiples of 10)
    These are not directly interchangeable numerically. The agent detects
    the scale, stores the raw value + scale type, and normalises to ECOG
    for all downstream SQL and reasoning logic using the standard mapping:
        KPS 90-100 → ECOG 0
        KPS 70-80  → ECOG 1
        KPS 50-60  → ECOG 2
        KPS 30-40  → ECOG 3
        KPS 10-20  → ECOG 4

OUTPUTS:
    validated_attrs dict  — corrected attributes for downstream agents
    validation_report     — per-field detail for Gradio panel + Postgres log
    pipeline_blocked      — True if any critical field fell into Tier 3

PERSISTENCE:
    All validation decisions are written to the `validated_attributes` Postgres
    table (via repository.log_validation_result). The original patients table
    is NEVER modified — both regex and LLM values are preserved for ablation
    analysis in Phase 9.
"""

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KPS → ECOG conversion table (WHO standard mapping)
# ─────────────────────────────────────────────────────────────────────────────
KPS_TO_ECOG: dict[int, int] = {
    100: 0, 90: 0,
    80: 1,  70: 1,
    60: 2,  50: 2,
    40: 3,  30: 3,
    20: 4,  10: 4,
}

# Word-form KPS values ("sixty percent Karnofsky") — the regex-based
# extractors only ever look for digits, so a KPS score spelled out as a
# word was silently unresolvable, driving base_confidence to 0.0 for this
# critical field and blocking the pipeline (Tier 3) even though the value
# is unambiguous to a human reader.
KPS_WORD_TO_NUM: dict[str, int] = {
    "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def kps_to_ecog(kps_value: int) -> int:
    """
    Convert a KPS score (0–100, multiples of 10) to ECOG equivalent.
    Rounds to nearest valid KPS bucket before lookup.
    """
    rounded = round(kps_value / 10) * 10
    rounded = max(10, min(100, rounded))
    return KPS_TO_ECOG.get(rounded, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical validity ranges  (deterministic component of confidence scoring)
# These encode domain knowledge that regex cannot express.
# ─────────────────────────────────────────────────────────────────────────────
FIELD_CONSTRAINTS: dict[str, dict] = {
    "performance_status_ecog": {
        "type": "int_range", "min": 0, "max": 4,
        "description": "ECOG performance status (0–4)"
    },
    "performance_status_kps": {
        "type": "int_range", "min": 0, "max": 100,
        "description": "Karnofsky Performance Status (0–100)"
    },
    "FEV1_cc": {
        "type": "int_range", "min": 500, "max": 6000,
        "description": "FEV1 in cc (clinically: 500–6000 mL)"
    },
    "FEV1_pct": {
        "type": "int_range", "min": 1, "max": 150,
        "description": "FEV1 % predicted (1–150%)"
    },
    "T_stage": {
        "type": "enum",
        "valid": {"0", "1", "1A", "1B", "1C", "2", "2A", "2B",
                  "3", "4", "X"},
        "description": "TNM T-stage"
    },
    "N_stage": {
        "type": "enum", "valid": {"0", "1", "2", "3", "X"},
        "description": "TNM N-stage"
    },
    "M_stage": {
        "type": "enum", "valid": {"0", "1", "1A", "1B", "1C", "X"},
        "description": "TNM M-stage"
    },
    "stage_group": {
        "type": "enum",
        "valid": {"I", "IA", "IB", "II", "IIA", "IIB",
                  "III", "IIIA", "IIIB", "IV", "IVA", "IVB",
                  "T1N0", "T1N1", "T1N2", "T2N0", "T2N1", "T2N2",
                  "T3N0", "T3N1", "T3N2", "T4N0", "T4N1", "T4N2",
                  "T1AN0", "T1BN0", "T2AN0", "T2BN1"},
        "description": "Overall stage group"
    },
    "neuropathy_grade": {
        "type": "int_range", "min": 0, "max": 4,
        "description": "CTCAE neuropathy grade (0–4)"
    },
    "prior_chemotherapy":      {"type": "bool", "description": "Prior chemotherapy flag"},
    "prior_radiotherapy":      {"type": "bool", "description": "Prior radiotherapy flag"},
    "prior_targeted_therapy":  {"type": "bool", "description": "Prior targeted therapy flag"},
    "prior_surgery":           {"type": "bool", "description": "Prior surgery flag"},
    "active_infection":        {"type": "bool", "description": "Active infection flag"},
    "cardiovascular_event":    {"type": "bool", "description": "Cardiovascular event flag"},
}

# Which fields are "critical" — a Tier 3 result on these blocks the pipeline.
#
# RESEARCH NOTE: Only fields whose absence makes eligibility *undecidable*
# are critical blockers. T_stage/N_stage/M_stage are sub-staging detail —
# when stage_group is confirmed (e.g. "IIIA") the pipeline can proceed;
# sub-stages are flagged as warnings only. FEV1 is optional — profiles that
# omit it simply skip the FEV1 threshold check downstream.
#
# True blockers: performance_status (direct eligibility criterion),
# stage_group (primary staging criterion), prior_chemotherapy /
# prior_radiotherapy / active_infection (direct exclusion criteria).
CRITICAL_FIELDS = {
    "performance_status",
    "stage_group",
    "prior_chemotherapy",
    "prior_radiotherapy",
    "active_infection",
}

# Evidence patterns per field (used to find supporting text spans)
EVIDENCE_PATTERNS: dict[str, list[str]] = {
    "performance_status": [
        # ECOG/Zubrod — handles: "ECOG 0", "ECOG: 0", "ECOG=0",
        # "ECOG score is 0", "ECOG score of 0", "ECOG performance status 1"
        r"ecog\s*(?:performance\s*status\s*)?(?:score\s*)?(?:is\s*|of\s*|:?\s*|=\s*)(\d)",
        r"zubrod\s*(?:performance\s*status\s*)?(?:score\s*)?(?:is\s*|of\s*|:?\s*|=\s*)(\d)",
        # KPS / Karnofsky
        r"(?:karnofsky|kps)\s*(?:performance\s*status\s*)?(?:of\s*|:?\s*|score\s*(?:of\s*)?|=\s*)?(\d{2,3})",
        # Generic "performance status X" or "performance status is X"
        r"performance\s*status\s*(?:of\s*|:?\s*|score\s*(?:of\s*)?|is\s*|=\s*)?(\d+)",
        # "PS 0", "PS: 1", "PS=2"
        r"\bps\s*(?:of\s*|:?\s*|=\s*)(\d)\b",
        # WHO performance status
        r"who\s*(?:performance\s*)?(?:status\s*)?(?:of\s*|:?\s*|=\s*)(\d)",
    ],
    "T_stage": [r"\bT(\d+[A-Za-z]?)\b"],
    "N_stage": [r"\bN(\d+)\b"],
    "M_stage": [r"\bM(\d+)\b"],
    "stage_group": [
        r"stage\s+([ivxIVX]+[abAB]?)",
        r"\bT\d+[A-Za-z]?N\d+\b",
    ],
    "FEV1_cc":  [r"fev1[^\d\n\r]*?(\d{3,4})\s*(?:cc|ml|mL)"],
    "FEV1_pct": [r"fev1[^\d\n\r]*?(\d{1,3})\s*(?:%|percent|predicted)"],
    "prior_chemotherapy": [
        r"received\s+chemotherapy",
        r"underwent\s+chemotherapy",
        r"prior\s+chemotherapy",
        r"no\s+(?:prior\s+)?(?:history\s+of\s+)?(?:systemic\s+)?chemotherapy",
        r"no\s+history\s+of\s+(?:lung\s+cancer\s+)?treatment",
    ],
    "prior_radiotherapy": [
        r"received\s+(?:thoracic\s+)?(?:radiotherapy|radiation)",
        r"prior\s+(?:thoracic\s+)?(?:radiotherapy|radiation)",
        r"no\s+prior\s+(?:thoracic\s+)?(?:radiotherapy|radiation)",
        r"no\s+history\s+of\s+(?:lung\s+cancer\s+)?treatment",
        r"radiation\s+boost\s+field",           # planning constraint (NOT prior treatment)
    ],
    "active_infection": [
        r"active\s+(?:pulmonary\s+)?infection",
        # r"no\s+active\s+infection",
        r"no\s+active\s+(?:pulmonary\s+)?(?:bacterial\s+)?infection"
        r"pneumonitis",
        r"tuberculosis",
    ],
    "cardiovascular_event": [
        r"myocardial\s+infarction",
        r"heart\s+failure",
        r"no\s+cardiac",
        r"arrhythmia",
    ],
    "neuropathy_grade": [
        r"grade\s*([0-4])\s*neuropathy",
        r"neuropathy\s*grade\s*([0-4])",
    ],
}

# Cross-field consistency rules
# Each rule is a tuple: (field_a, field_b, check_fn, description)
CONSISTENCY_RULES = [
    # If both T and N stage are present, stage_group should be derivable
    ("T_stage", "N_stage", "stage_group",
     "T+N stage combination should be consistent with stage_group"),
    # No prior therapy flags should not contradict has_prior_therapy
    ("prior_chemotherapy", "prior_radiotherapy", "has_prior_therapy",
     "Individual therapy flags should be consistent with has_prior_therapy"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldValidationResult:
    """Validation result for a single extracted attribute."""
    field_name: str
    regex_value: Any                        # raw value from patient_extractor.py
    evidence_span: Optional[str] = None    # supporting text found in profile
    evidence_found: bool = False

    base_confidence: float = 0.0           # deterministic score component
    llm_confidence: Optional[float] = None # LLM adjustment (Tier 2 only)
    final_confidence: float = 0.0          # authoritative score used for decision

    range_check_passed: Optional[bool] = None
    consistency_check_passed: Optional[bool] = None
    consistency_note: str = ""

    tier: int = 1                          # 1=Accept, 2=Re-extract, 3=HumanReview
    decision: str = "ACCEPT"              # ACCEPT | REEXTRACT | HUMAN_REVIEW

    llm_value: Any = None                  # LLM re-extraction result (Tier 2)
    llm_evidence: Optional[str] = None    # evidence quoted by LLM
    final_value: Any = None               # value actually used downstream

    scale_detected: Optional[str] = None  # "ECOG" | "KPS" | None
    ecog_normalized: Optional[int] = None # normalised ECOG value (if KPS detected)

    is_critical: bool = False
    pipeline_blocked: bool = False
    block_reason: str = ""


@dataclass
class ValidationReport:
    """Complete validation report for one patient pipeline run."""
    run_id: str
    patient_id: Optional[str]
    fields: list[FieldValidationResult] = field(default_factory=list)
    pipeline_blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    total_fields: int = 0
    accepted_direct: int = 0
    reextracted: int = 0
    human_review_required: int = 0

    def summary_text(self) -> str:
        lines = [
            f"{'='*60}",
            f"VALIDATION REPORT  (run_id: {self.run_id})",
            f"{'='*60}",
            f"Total fields validated : {self.total_fields}",
            f"Accepted directly      : {self.accepted_direct}",
            f"Re-extracted by LLM    : {self.reextracted}",
            f"Flagged for review     : {self.human_review_required}",
            f"Pipeline blocked       : {'YES ⛔' if self.pipeline_blocked else 'NO ✅'}",
        ]
        if self.block_reasons:
            lines.append(f"Block reasons          : {'; '.join(self.block_reasons)}")
        lines.append("")
        for r in self.fields:
            lines.append(self._field_line(r))
        return "\n".join(lines)

    @staticmethod
    def _field_line(r: FieldValidationResult) -> str:
        tier_icon = {1: "✅", 2: "🔄", 3: "⛔"}[r.tier]
        scale_info = f" [{r.scale_detected}→ECOG {r.ecog_normalized}]" if r.ecog_normalized is not None else ""
        evidence_display = f'  Evidence: "{r.evidence_span}"' if r.evidence_span else "  Evidence: (none found)"

        lines = [
            f"{tier_icon} {r.field_name}",
            f"  Regex value     : {r.regex_value}",
        ]
        if r.llm_value is not None:
            lines.append(f"  LLM value       : {r.llm_value}")
        lines.append(f"  Final value     : {r.final_value}{scale_info}")
        lines.append(f"  Confidence      : {r.final_confidence:.2f} "
                     f"(base={r.base_confidence:.2f}"
                     + (f", llm={r.llm_confidence:.2f}" if r.llm_confidence is not None else "")
                     + ")")
        lines.append(evidence_display)
        if r.consistency_note:
            lines.append(f"  Consistency     : {r.consistency_note}")
        if r.pipeline_blocked:
            lines.append(f"  ⛔ BLOCKED: {r.block_reason}")
        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Validation Agent class
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalAttributeValidationAgent:
    """
    Clinical Attribute Validation Agent — Phase 4 core component.

    Usage in pipeline:
        validator = ClinicalAttributeValidationAgent(
            call_llm_fn=matcher.call_llm,  # pass the existing LLM caller
            run_id=run_id,
            patient_id=patient_id,
        )
        validated_attrs, report = validator.validate(
            raw_attrs=extracted_attrs,      # from patient_extractor.py
            patient_profile=patient_profile  # original free text
        )
        # validated_attrs flows into clinical_reasoning_agent and sql_generation_agent
        # report is logged to Postgres and displayed in Gradio panel

    Confidence thresholds:
        TIER1_THRESHOLD = 0.80   (accept, no LLM call)
        TIER2_LOWER     = 0.30   (below this → human review directly)

    These are module-level constants so they appear in the experiment config
    table in the journal's methods section.
    """

    TIER1_THRESHOLD: float = 0.80
    TIER2_LOWER:     float = 0.30

    def __init__(
        self,
        call_llm_fn,          # callable: (system_prompt, user_prompt) -> str
        run_id: str,
        patient_id: Optional[str] = None,
    ):
        self.call_llm = call_llm_fn
        self.run_id = run_id
        self.patient_id = patient_id

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def validate(
        self,
        raw_attrs: dict,
        patient_profile: str,
        trial_criteria: Optional[dict] = None,
    ) -> tuple[dict, ValidationReport]:
        """
        Validate all extracted attributes against the original patient profile.

        Args:
            raw_attrs:       extracted attributes from patient_extractor.py
            patient_profile: original free text
            trial_criteria:  parsed_criteria from parsing_agent — used to determine
                             which fields are actually tested by this trial so
                             irrelevant fields skip Tier 2 LLM calls.

        Returns:
            validated_attrs  — corrected dict, safe to pass downstream
            report           — ValidationReport for logging + Gradio display
        """
        report = ValidationReport(
            run_id=self.run_id,
            patient_id=self.patient_id,
        )

        validated = dict(raw_attrs)  # start with regex output, correct in place

        profile_lower = patient_profile.lower()

        # ── Determine which fields are tested by this trial ───────────────────
        # Fields not mentioned in the trial criteria cannot affect eligibility,
        # so Tier 2 LLM re-extraction for them is wasteful. We derive a set of
        # "relevant fields" from parsed_criteria and skip LLM calls for the rest.
        relevant_fields = self._get_relevant_fields(trial_criteria)

        # ── 1. Performance status — most complex field (scale detection) ──────
        perf_result = self._validate_performance_status(
            raw_attrs, patient_profile, profile_lower,
            skip_llm=not self._is_relevant("performance_status", relevant_fields)
        )
        report.fields.append(perf_result)
        if perf_result.final_value is not None:
            # performance_status: store ECOG-normalised int for SQL range comparisons
            validated["performance_status"] = str(perf_result.ecog_normalized
                                                   if perf_result.ecog_normalized is not None
                                                   else perf_result.final_value)
            # performance_status_raw: store original scale value for KPS threshold checks
            # (e.g. KPS 60 — needed by clinical_reasoning_agent KPS path)
            validated["performance_status_raw"] = str(perf_result.final_value)
        if perf_result.scale_detected:
            validated["performance_scale"] = perf_result.scale_detected
            if perf_result.ecog_normalized is not None:
                validated["performance_status_ecog"] = perf_result.ecog_normalized

        # ── 2. Stage fields ───────────────────────────────────────────────────
        for field_name in ["T_stage", "N_stage", "M_stage", "stage_group"]:
            result = self._validate_generic_field(
                field_name, raw_attrs.get(field_name), patient_profile, profile_lower,
                skip_llm=not self._is_relevant(field_name, relevant_fields)
            )
            report.fields.append(result)
            if result.final_value is not None:
                validated[field_name] = result.final_value

        # Consistency check: T+N → stage_group
        consistency_result = self._check_tn_stage_consistency(validated)
        if consistency_result:
            # Attach note to stage_group field result
            for r in report.fields:
                if r.field_name == "stage_group":
                    r.consistency_note = consistency_result
                    if "INCONSISTENT" in consistency_result:
                        r.final_confidence = max(0.0, r.final_confidence - 0.2)
                    break

        # ── 3. FEV1 ───────────────────────────────────────────────────────────
        for field_name in ["FEV1_cc", "FEV1_pct"]:
            result = self._validate_generic_field(
                field_name, raw_attrs.get(field_name), patient_profile, profile_lower,
                skip_llm=not self._is_relevant(field_name, relevant_fields)
            )
            report.fields.append(result)
            if result.final_value is not None:
                validated[field_name] = result.final_value

        # ── 4. Boolean clinical flags ─────────────────────────────────────────
        bool_fields = [
            "prior_chemotherapy", "prior_radiotherapy",
            "prior_targeted_therapy", "prior_surgery",
            "active_infection", "cardiovascular_event",
        ]
        for field_name in bool_fields:
            result = self._validate_boolean_field(
                field_name, raw_attrs.get(field_name, False),
                patient_profile, profile_lower,
                skip_llm=not self._is_relevant(field_name, relevant_fields)
            )
            report.fields.append(result)
            validated[field_name] = result.final_value

        # ── 5. Neuropathy grade ───────────────────────────────────────────────
        result = self._validate_generic_field(
            "neuropathy_grade", raw_attrs.get("neuropathy_grade"),
            patient_profile, profile_lower,
            skip_llm=not self._is_relevant("neuropathy_grade", relevant_fields)
        )
        report.fields.append(result)
        if result.final_value is not None:
            validated["neuropathy_grade"] = result.final_value

        # ── 6. Re-derive has_prior_therapy from validated flags ───────────────
        validated["has_prior_therapy"] = any([
            validated.get("prior_chemotherapy", False),
            validated.get("prior_radiotherapy", False),
            validated.get("prior_targeted_therapy", False),
            validated.get("prior_surgery", False),
        ])

        # ── 7. Finalise report stats ──────────────────────────────────────────
        report.total_fields = len(report.fields)
        report.accepted_direct = sum(1 for r in report.fields if r.tier == 1)
        report.reextracted = sum(1 for r in report.fields if r.tier == 2)
        report.human_review_required = sum(1 for r in report.fields if r.tier == 3)

        blocked_fields = [r for r in report.fields if r.pipeline_blocked]
        if blocked_fields:
            report.pipeline_blocked = True
            report.block_reasons = [f.block_reason for f in blocked_fields]

        return validated, report

    # ─────────────────────────────────────────────────────────────────────────
    # Performance status validation (special — scale detection required)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Trial-relevance: which fields does this trial actually test?
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_relevant_fields(trial_criteria: Optional[dict]) -> set:
        """
        Derive which patient attributes are actually tested by this trial's criteria.
        Fields not mentioned in the criteria cannot affect eligibility and should
        skip Tier 2 LLM re-extraction to avoid unnecessary API calls.

        Returns a set of field names that are relevant, or None if criteria
        are unavailable (in which case all fields are treated as relevant).
        """
        if trial_criteria is None:
            return None  # None means "all fields relevant" — safe default

        relevant = set()
        inclusion = trial_criteria.get("parsed_inclusion_conditions", {})
        exclusion = trial_criteria.get("parsed_exclusion_conditions", {})

        # Stage — relevant if trial specifies any stage criteria
        if inclusion.get("stage"):
            relevant.update({"stage_group", "T_stage", "N_stage", "M_stage"})

        # Performance status — relevant if trial specifies it
        if inclusion.get("performance_status"):
            relevant.add("performance_status")

        # FEV1 — relevant only if trial has an FEV1 threshold
        if inclusion.get("FEV1_cc_min") is not None or inclusion.get("FEV1_pct_min") is not None:
            relevant.update({"FEV1_cc", "FEV1_pct"})

        # Prior therapy — relevant if trial excludes prior treatments
        if exclusion.get("prior_systemic_chemotherapy", "No").lower() == "yes":
            relevant.add("prior_chemotherapy")
        if exclusion.get("prior_radiotherapy", "No").lower() == "yes":
            relevant.add("prior_radiotherapy")
        if exclusion.get("prior_targeted_therapy", "No").lower() == "yes":
            relevant.add("prior_targeted_therapy")
        if inclusion.get("prior_treatment_history", ""):
            relevant.update({"prior_chemotherapy", "prior_radiotherapy",
                             "prior_targeted_therapy", "prior_surgery"})

        # Active infection — relevant if trial excludes it
        if exclusion.get("active_pulmonary_infection", "No").lower() == "yes":
            relevant.add("active_infection")

        # Cardiovascular — relevant if trial excludes it
        if exclusion.get("cardiovascular_event", "No").lower() == "yes":
            relevant.add("cardiovascular_event")

        # Neuropathy — relevant if trial has a neuropathy threshold
        if exclusion.get("neuropathy_grade", ""):
            relevant.add("neuropathy_grade")

        # Always validate performance_status and stage_group — they are
        # the most fundamental eligibility fields in NSCLC trials.
        relevant.update({"performance_status", "stage_group"})

        return relevant

    def _is_relevant(self, field_name: str, relevant_fields: Optional[set]) -> bool:
        """
        Returns True if this field should be fully validated (including Tier 2 LLM),
        False if the field is not tested by this trial and can be passed through as-is.
        """
        if relevant_fields is None:
            return True  # all fields relevant when no criteria provided
        # Map field name variants to the set keys
        base_name = field_name.replace("_ecog", "").replace("_kps", "")
        return base_name in relevant_fields or field_name in relevant_fields

    # ─────────────────────────────────────────────────────────────────────────
    # Performance status validation (special — scale detection required)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_performance_status(
        self, raw_attrs: dict, patient_profile: str, profile_lower: str,
        skip_llm: bool = False,
    ) -> FieldValidationResult:
        """
        Detects whether the profile uses ECOG (0–4) or KPS (0–100) scale.
        Normalises to ECOG int for downstream logic.
        The KPS=60 → ECOG=6 bug is caught here by the range check.
        """
        raw_value = raw_attrs.get("performance_status", "")
        result = FieldValidationResult(
            field_name="performance_status",
            regex_value=raw_value,
            is_critical=True,
        )

        # ── Step 1: Parse the raw value ───────────────────────────────────────
        numeric_match = re.search(r"(\d+)", str(raw_value)) if raw_value else None
        raw_int = int(numeric_match.group(1)) if numeric_match else None

        # ── Step 2: Detect scale from profile text ────────────────────────────
        kps_match = re.search(
            r"(?:karnofsky|kps)\s*(?:performance\s*status\s*)?(?:of\s*|:?\s*|score\s*(?:of\s*)?)(\d{2,3})",
            profile_lower,
        )
        kps_word_match = re.search(
            r"(?:karnofsky|kps)\s*(?:performance\s*status\s*)?(?:score\s*)?"
            r"(?:is\s*|of\s*|:?\s*)?(" + "|".join(KPS_WORD_TO_NUM.keys()) + r")\b",
            profile_lower,
        )
        ecog_match = re.search(
            # Handles: "ECOG 0", "ECOG: 0", "ECOG=0", "ECOG score is 0",
            # "ECOG score of 0", "ECOG performance status 1", "Zubrod 1"
            r"(?:ecog|zubrod)\s*(?:performance\s*status\s*)?(?:score\s*)?"
            r"(?:is\s*|of\s*|:?\s*|=\s*)(\d)",
            profile_lower,
        )
        generic_ps_match = re.search(
            r"performance\s*status\s*(?:of\s*|:?\s*|score\s*(?:of\s*)?|is\s*|=\s*)?(\d+)",
            profile_lower,
        )

        # ── Step 3: Determine scale and true value ────────────────────────────
        if kps_match:
            result.scale_detected = "KPS"
            true_kps = int(kps_match.group(1))
            result.evidence_span = self._find_evidence_span(patient_profile, "performance_status")
            result.evidence_found = True

            # This is where KPS=60→ECOG=6 is caught:
            # raw_int might be 6 (regex captured first digit of "60")
            # but we now know scale is KPS, so true value is true_kps
            result.ecog_normalized = kps_to_ecog(true_kps)
            result.final_value = true_kps  # store raw KPS

            # Confidence: high if KPS value is a valid multiple of 10
            if true_kps % 10 == 0 and 0 <= true_kps <= 100:
                result.base_confidence = 0.92
                result.range_check_passed = True
            else:
                result.base_confidence = 0.55
                result.range_check_passed = False

        elif kps_word_match:
            # "sixty percent Karnofsky" / "Karnofsky score is sixty" — same
            # unambiguous KPS mention as kps_match, just spelled out in words.
            result.scale_detected = "KPS"
            true_kps = KPS_WORD_TO_NUM[kps_word_match.group(1)]
            result.evidence_span = self._find_evidence_span(patient_profile, "performance_status")
            result.evidence_found = True

            result.ecog_normalized = kps_to_ecog(true_kps)
            result.final_value = true_kps
            result.base_confidence = 0.92
            result.range_check_passed = True

        elif ecog_match:
            result.scale_detected = "ECOG"
            true_ecog = int(ecog_match.group(1))
            result.evidence_span = self._find_evidence_span(patient_profile, "performance_status")
            result.evidence_found = True

            if 0 <= true_ecog <= 4:
                result.base_confidence = 0.95
                result.range_check_passed = True
            else:
                result.base_confidence = 0.20   # impossible ECOG value
                result.range_check_passed = False

            result.ecog_normalized = true_ecog
            result.final_value = true_ecog

        elif generic_ps_match:
            # Scale ambiguous — use value but flag for LLM verification
            raw_ps_int = int(generic_ps_match.group(1))
            result.evidence_span = self._find_evidence_span(patient_profile, "performance_status")
            result.evidence_found = bool(result.evidence_span)

            if 0 <= raw_ps_int <= 4:
                # Likely ECOG
                result.scale_detected = "ECOG"
                result.ecog_normalized = raw_ps_int
                result.final_value = raw_ps_int
                result.base_confidence = 0.65   # plausible but scale not explicit
                result.range_check_passed = True
            elif raw_ps_int in range(10, 101, 10):
                # Likely KPS
                result.scale_detected = "KPS"
                result.ecog_normalized = kps_to_ecog(raw_ps_int)
                result.final_value = raw_ps_int
                result.base_confidence = 0.60
                result.range_check_passed = True
            else:
                # Impossible value (e.g. regex captured wrong digit)
                result.scale_detected = "UNKNOWN"
                result.final_value = raw_int
                result.base_confidence = 0.15   # very suspicious
                result.range_check_passed = False

        elif raw_int is not None:
            # Regex found something but no context phrase matched
            result.evidence_found = False
            if 0 <= raw_int <= 4:
                result.scale_detected = "ECOG"
                result.ecog_normalized = raw_int
                result.final_value = raw_int
                result.base_confidence = 0.45  # value OK but no evidence span
            else:
                result.scale_detected = "UNKNOWN"
                result.final_value = raw_int
                result.base_confidence = 0.20
                result.range_check_passed = False
        else:
            # Not found at all
            result.evidence_found = False
            result.final_value = None
            result.base_confidence = 0.0

        # ── Step 4: Apply confidence tier and possibly call LLM ──────────────
        result = self._apply_tier_logic(result, patient_profile, "performance_status",
                                        skip_llm=skip_llm)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Boolean flag validation (prior therapies, infections, cardiac)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_boolean_field(
        self,
        field_name: str,
        raw_value: bool,
        patient_profile: str,
        profile_lower: str,
        skip_llm: bool = False,
    ) -> FieldValidationResult:
        """
        For boolean fields, confidence is derived from:
        - Whether a clear positive evidence span exists (for True values)
        - Whether a clear negative evidence span exists (for False values)
        - Whether both positive AND negative spans exist (contradiction → low confidence)

        Special case: "radiation boost field" context (treatment planning)
        should not set prior_radiotherapy = True.
        """
        result = FieldValidationResult(
            field_name=field_name,
            regex_value=raw_value,
            is_critical=(field_name in CRITICAL_FIELDS),
        )

        evidence_span = self._find_evidence_span(patient_profile, field_name)
        result.evidence_span = evidence_span
        result.evidence_found = bool(evidence_span)

        # ── Detect planning-context false positives (radiotherapy specific) ───
        if field_name == "prior_radiotherapy" and raw_value:
            planning_patterns = [
                r"radiation\s+boost\s+field",
                r"radiation\s+dose\s+constraint",
                r"radiation\s+treatment\s+plan",
                r"planned\s+radiation",
                r"radiation\s+field\s+(?:is|was|within)",
                r"boost\s+volume",
            ]
            for pat in planning_patterns:
                if re.search(pat, profile_lower):
                    # Treatment planning term found — suspicious, but route to Tier 2
                    # (Tier 2 not Tier 3: LLM re-extraction can resolve this confidently)
                    m = re.search(pat, profile_lower, re.I)
                    result.evidence_span = m.group(0) if m else evidence_span
                    result.base_confidence = 0.38  # above TIER2_LOWER → triggers LLM
                    result.consistency_note = (
                        "⚠️ Radiation planning term detected — may not be prior treatment history"
                    )
                    result.final_value = raw_value
                    result = self._apply_tier_logic(result, patient_profile, field_name,
                                                    skip_llm=skip_llm)
                    return result

        # ── Standard boolean confidence ───────────────────────────────────────
        if raw_value:
            # Claimed True — check evidence quality
            positive_patterns = EVIDENCE_PATTERNS.get(field_name, [])
            positive_found = any(
                re.search(p, profile_lower) for p in positive_patterns
                if "no" not in p and "not" not in p
            )
            negative_found = any(
                re.search(p, profile_lower) for p in positive_patterns
                if "no" in p or "not" in p
            )

            if positive_found and not negative_found:
                # result.base_confidence = 0.88    # To strict
                result.base_confidence = 0.65 
                result.range_check_passed = True
            elif positive_found and negative_found:
                # Contradiction in text — low confidence
                result.base_confidence = 0.35
                result.consistency_note = "⚠️ Both positive and negative signals found in profile"
                result.range_check_passed = False
            else:
                result.base_confidence = 0.40
                result.range_check_passed = False
        else:
            # Claimed False — check no evidence slipped through negation guard
            positive_patterns = EVIDENCE_PATTERNS.get(field_name, [])
            any_mention = any(
                re.search(p.replace(r"\s*no\s*", r"\s*").replace(r"\bnot\b", ""),
                          profile_lower)
                for p in positive_patterns
                if "no" in p or "not" in p
            )
            # If the field isn't mentioned at all, False is reasonable
            mentioned_at_all = bool(evidence_span)

            if not mentioned_at_all:
                result.base_confidence = 0.75  # absence of evidence is reasonable
            else:
                result.base_confidence = 0.82  # negation explicitly confirmed

            result.range_check_passed = True

        result.final_value = raw_value
        result = self._apply_tier_logic(result, patient_profile, field_name,
                                        skip_llm=skip_llm)

        # After tier logic: if LLM corrected a boolean, coerce to bool
        if result.tier == 2 and result.llm_value is not None:
            if isinstance(result.llm_value, bool):
                result.final_value = result.llm_value
            elif isinstance(result.llm_value, str):
                result.final_value = result.llm_value.lower() in ("true", "1", "yes")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Generic field validation (stage, FEV1, neuropathy)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_generic_field(
        self,
        field_name: str,
        raw_value: Any,
        patient_profile: str,
        profile_lower: str,
        skip_llm: bool = False,
    ) -> FieldValidationResult:
        """
        Validates numeric and enum fields against FIELD_CONSTRAINTS.
        Confidence = range_weight + evidence_weight + (consistency if applicable)

        skip_llm: when True, Tier 2 LLM re-extraction is skipped because this
                  field is not tested by the current trial's criteria. The field
                  is still scored and flagged but will not trigger an API call.
        """
        result = FieldValidationResult(
            field_name=field_name,
            regex_value=raw_value,
            is_critical=(field_name in CRITICAL_FIELDS),
        )

        constraint = FIELD_CONSTRAINTS.get(field_name)
        evidence_span = self._find_evidence_span(patient_profile, field_name)
        result.evidence_span = evidence_span
        result.evidence_found = bool(evidence_span)

        # ── Confidence component 1: range / validity check (weight 0.4) ──────
        if raw_value is None or raw_value == "":
            # Field genuinely absent from profile — not an impossible value,
            # just not mentioned. Give a neutral base score so Tier 2 LLM
            # re-extraction is attempted before any blocking decision.
            # If the LLM also cannot find it, the field stays None and
            # non-critical fields pass through as "Cannot Evaluate" downstream.
            result.range_check_passed = None
            range_score = 0.15   # neutral (replaces hard 0.0 that caused false blocking)
        elif constraint:
            ctype = constraint["type"]
            if ctype == "int_range":
                try:
                    val_int = int(str(raw_value).strip())
                    in_range = constraint["min"] <= val_int <= constraint["max"]
                    result.range_check_passed = in_range
                    range_score = 0.4 if in_range else 0.0
                except (ValueError, TypeError):
                    result.range_check_passed = False
                    range_score = 0.0
            elif ctype == "enum":
                normalized_val = str(raw_value).strip().upper()
                in_valid_set = normalized_val in constraint["valid"]
                result.range_check_passed = in_valid_set
                range_score = 0.4 if in_valid_set else 0.1
            elif ctype == "bool":
                result.range_check_passed = isinstance(raw_value, bool)
                range_score = 0.4 if result.range_check_passed else 0.0
            else:
                range_score = 0.2
        else:
            range_score = 0.2  # unknown field — neutral

        # ── Confidence component 2: evidence span present (weight 0.35) ──────
        evidence_score = 0.35 if result.evidence_found else 0.0

        # ── Confidence component 3: consistency (weight 0.25) ─────────────────
        # (basic — full cross-field consistency runs at the end of validate())
        consistency_score = 0.25  # default: assume consistent until checked

        result.base_confidence = min(1.0, range_score + evidence_score + consistency_score)
        result.final_value = raw_value

        # ── Apply tier logic ──────────────────────────────────────────────────
        result = self._apply_tier_logic(result, patient_profile, field_name, skip_llm=skip_llm)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Tier routing + LLM re-extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_tier_logic(
        self,
        result: FieldValidationResult,
        patient_profile: str,
        field_name: str,
        skip_llm: bool = False,
    ) -> FieldValidationResult:
        """
        Routes each field to the correct tier based on base_confidence:
            Tier 1 (≥ 0.80): Accept directly — no LLM call
            Tier 2 (0.30–0.79): Targeted LLM re-extraction
            Tier 3 (< 0.30): Human review required

        skip_llm: when True, Tier 2 LLM re-extraction is bypassed.
                  The field is accepted at its base_confidence, or flagged
                  as NOT_APPLICABLE if the field is not tested by this trial.
        """
        conf = result.base_confidence

        if conf >= self.TIER1_THRESHOLD:
            # ── Tier 1: Accept ────────────────────────────────────────────────
            result.tier = 1
            result.decision = "ACCEPT"
            result.final_confidence = conf

        elif conf >= self.TIER2_LOWER:
            # ── Tier 2: Targeted LLM re-extraction ───────────────────────────
            if skip_llm:
                # Field not tested by this trial — accept at base confidence,
                # no LLM call needed. Mark as ACCEPT_NOT_APPLICABLE so the
                # Gradio panel and research logs can distinguish this from a
                # normal Tier 1 accept.
                result.tier = 1
                result.decision = "ACCEPT_NOT_APPLICABLE"
                result.final_confidence = conf
                return result

            result.tier = 2
            result.decision = "REEXTRACT"
            llm_value, llm_confidence, llm_evidence = self._llm_reextract(
                field_name, patient_profile, result.regex_value
            )
            result.llm_value = llm_value
            result.llm_confidence = llm_confidence
            result.llm_evidence = llm_evidence

            # Weighted combination: LLM weight increases when it is highly confident.
            # Rationale: the deterministic base score reflects extraction suspicion;
            # a highly-confident LLM that cites specific evidence should dominate.
            #
            # Special case: LLM confidence = 0.0 means the LLM actively found
            # nothing / field is genuinely absent from the profile. This is a
            # valid "absent" signal — do NOT let it drag the final score down
            # below the regex base. Keep the base_confidence as final instead.
            #
            # Weight schedule:
            #   LLM conf = 0.0          → keep base only (field absent)
            #   LLM conf 0.01–0.84      → 50/50 blend
            #   LLM conf ≥ 0.85         → 30/70 (trust LLM more)
            if llm_confidence is not None and llm_confidence < 0.0:
                # Infrastructure failure (rate-limit/401/JSON) — NOT a clinical
                # signal. Accept the regex value at its base confidence and do
                # not let this escalate to a critical-field block. Return early.
                result.llm_confidence = None
                result.final_confidence = conf
                result.decision = "ACCEPT_LLM_UNAVAILABLE"
                result.final_value = result.regex_value
                result.consistency_note = (
                    (result.consistency_note + " | ") if result.consistency_note else ""
                ) + "LLM re-extraction unavailable (infra) — kept regex value, not blocking"
                return result
            if llm_confidence == 0.0:
                result.final_confidence = conf   # field absent — don't penalise
            elif llm_confidence >= 0.85:
                result.final_confidence = 0.30 * conf + 0.70 * llm_confidence
            else:
                result.final_confidence = 0.50 * conf + 0.50 * llm_confidence

            if result.final_confidence >= 0.70:
                # LLM correction accepted — coerce type to match field expectation
                result.decision = "ACCEPT_CORRECTED"
                if isinstance(result.regex_value, bool) or field_name.startswith("prior_") \
                        or field_name in ("active_infection", "cardiovascular_event"):
                    # Boolean field: normalise LLM response
                    if isinstance(llm_value, bool):
                        result.final_value = llm_value
                    elif isinstance(llm_value, str):
                        result.final_value = llm_value.lower() in ("true", "1", "yes")
                    else:
                        result.final_value = bool(llm_value) if llm_value is not None else result.regex_value
                else:
                    # Never downgrade to None: if LLM returned None but regex had a
                    # valid value, the field is likely genuinely present but the LLM
                    # couldn't find it in this call — keep the regex value.
                    # This prevents neuropathy_grade=1 (regex correct) being overwritten
                    # by LLM None when the profile doesn't use standard neuropathy phrasing.
                    if llm_value is None and result.regex_value is not None                             and result.regex_value != "":
                        result.final_value = result.regex_value
                        result.decision = "ACCEPT"   # treat as accepted regex value
                    else:
                        # T_stage/N_stage/M_stage: the LLM naturally answers
                        # with its own letter prefix ("N2", "M0") since that's
                        # how staging is normally written, but downstream code
                        # (clinical_reasoning_agent's f"T{T}N{N}", and
                        # build_current_patient_cte's SQL gate) re-prefixes
                        # the letter itself — "T" + "2" + "N" + "N2" would
                        # silently become "T2NN2", matching nothing. Strip a
                        # leading matching letter so the stored value is
                        # always the bare digit(s), like the regex extractor
                        # already produces.
                        stage_prefix = {"T_stage": "T", "N_stage": "N", "M_stage": "M"}.get(field_name)
                        if stage_prefix and isinstance(llm_value, str) and llm_value.upper().startswith(stage_prefix):
                            llm_value = llm_value[1:]
                        result.final_value = llm_value
                # Re-apply KPS normalisation if this was a performance status fix
                if field_name == "performance_status" and result.scale_detected == "KPS":
                    try:
                        result.ecog_normalized = kps_to_ecog(int(llm_value))
                    except (ValueError, TypeError):
                        pass
            else:
                # Still uncertain after LLM — escalate
                result.tier = 3
                result.decision = "HUMAN_REVIEW"
                if result.is_critical:
                    result.pipeline_blocked = True
                    result.block_reason = (
                        f"Field '{field_name}' confidence {result.final_confidence:.2f} "
                        f"remains below 0.70 after LLM re-extraction. "
                        f"Regex={result.regex_value}, LLM={llm_value}"
                    )

        else:
            # ── Tier 3: Human review ──────────────────────────────────────────
            result.tier = 3
            result.decision = "HUMAN_REVIEW"
            result.final_confidence = conf
            if result.is_critical:
                result.pipeline_blocked = True
                result.block_reason = (
                    f"Field '{field_name}' base confidence {conf:.2f} < {self.TIER2_LOWER}. "
                    f"Extracted value '{result.regex_value}' requires human review."
                )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # LLM targeted re-extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_reextract(
        self,
        field_name: str,
        patient_profile: str,
        regex_value: Any,
    ) -> tuple[Any, float, Optional[str]]:
        """
        Calls the LLM with a field-specific prompt to re-extract a single attribute.
        Returns: (value, confidence_0_to_1, evidence_quote)

        Prompt design principles for research reproducibility:
        - Field-specific context (clinical meaning + valid range)
        - Explicit instruction to return JSON only
        - Ask for evidence quote so we can audit the decision
        - Ask for self-reported confidence (0.0–1.0)
        """
        constraint = FIELD_CONSTRAINTS.get(field_name, {})
        field_desc = constraint.get("description", field_name)

        system_prompt = (
            "You are a clinical NLP specialist extracting structured data from "
            "oncology patient profiles. You must return ONLY a valid JSON object — "
            "no preamble, no markdown, no explanation."
        )

        # Build a field-specific instruction
        field_instructions = {
            "performance_status": (
                "Extract the patient's performance status score. "
                "IMPORTANT: Determine which scale is used:\n"
                "  - ECOG/Zubrod scale: integer 0–4\n"
                "  - Karnofsky (KPS) scale: integer 0–100 in multiples of 10\n"
                "Return the raw value as stated AND the scale type.\n"
                "Return JSON: {\"value\": <number>, \"scale\": \"ECOG\" or \"KPS\", "
                "\"ecog_equivalent\": <0-4>, \"evidence\": \"<quote>\", \"confidence\": <0.0-1.0>}"
            ),
            "prior_radiotherapy": (
                "Determine whether the patient has PREVIOUSLY RECEIVED radiotherapy "
                "as part of their cancer treatment history.\n"
                "CRITICAL: Do NOT confuse treatment planning language (e.g. 'radiation boost "
                "field within limits', 'planned radiation dose') with prior treatment receipt.\n"
                "Return JSON: {\"value\": true or false, \"evidence\": \"<direct quote>\", "
                "\"reasoning\": \"<brief>\", \"confidence\": <0.0-1.0>}"
            ),
            "prior_chemotherapy": (
                "Determine whether the patient has PREVIOUSLY RECEIVED systemic chemotherapy.\n"
                "Return JSON: {\"value\": true or false, \"evidence\": \"<direct quote>\", "
                "\"confidence\": <0.0-1.0>}"
            ),
            "stage_group": (
                "Extract the cancer stage group. Valid values: I, IA, IB, II, IIA, IIB, "
                "III, IIIA, IIIB, IV, IVA, IVB, or TN notation like T2N1.\n"
                "Return JSON: {\"value\": \"<stage>\", \"evidence\": \"<quote>\", "
                "\"confidence\": <0.0-1.0>}"
            ),
            "FEV1_cc": (
                "Extract FEV1 in cubic centimetres (cc or mL). Valid range: 500–6000.\n"
                "Return JSON: {\"value\": <integer or null>, \"evidence\": \"<quote>\", "
                "\"confidence\": <0.0-1.0>}"
            ),
            "FEV1_pct": (
                "Extract FEV1 as percentage of predicted. Valid range: 1–150%.\n"
                "Return JSON: {\"value\": <integer or null>, \"evidence\": \"<quote>\", "
                "\"confidence\": <0.0-1.0>}"
            ),
        }

        default_instruction = field_instructions.get(
            field_name,
            f"Extract the value for: {field_desc}.\n"
            f"Previously extracted value (may be wrong): {regex_value}\n"
            f"Return JSON: {{\"value\": <value or null>, \"evidence\": \"<quote>\", "
            f"\"confidence\": <0.0-1.0>}}"
        )

        # Training-mode refinement (app.py's refine_validation_field_prompt)
        # may have saved an improved version of this field's instruction to
        # Postgres — read it back if present, else fall back to the default
        # above. A DB hiccup here must never block validation.
        specific_instruction = default_instruction
        # EXP_NO_REFINEMENT (ablation only, default off): skip the refined
        # instruction so the initial default is used. Normal runs unaffected.
        if not os.environ.get("EXP_NO_REFINEMENT"):
            try:
                from src.db.repository import get_refined_prompt
                refined = get_refined_prompt(f"validation_field_{field_name}")
                if refined:
                    specific_instruction = refined
            except Exception:
                pass

        user_prompt = (
            f"PATIENT PROFILE:\n{patient_profile}\n\n"
            f"TASK:\n{specific_instruction}\n\n"
            f"Previously extracted value (may be incorrect): {regex_value}\n\n"
            "Return ONLY the JSON object."
        )

        try:
            raw_response = self.call_llm(system_prompt, user_prompt)

            # Strip markdown fences defensively
            clean = re.sub(r"```(?:json)?|```", "", raw_response, flags=re.DOTALL).strip()
            parsed = json.loads(clean)

            value = parsed.get("value")
            confidence = float(parsed.get("confidence", 0.5))
            evidence = parsed.get("evidence", "")

            # For performance_status: if LLM returned scale info, use ecog_equivalent
            if field_name == "performance_status" and "ecog_equivalent" in parsed:
                value = parsed.get("ecog_equivalent", value)
                # We return the raw KPS value; caller handles normalisation
                value = parsed.get("value", value)

            confidence = max(0.0, min(1.0, confidence))
            return value, confidence, evidence

        except (json.JSONDecodeError, KeyError, ValueError, Exception) as e:
            logger.warning(f"LLM re-extraction failed for {field_name}: {e}")
            # Signal an INFRASTRUCTURE failure (rate-limit/401/JSON error) with a
            # sentinel confidence of -1.0, distinct from a genuine low-confidence
            # result. _apply_tier_logic accepts the regex value and does NOT
            # block the pipeline on a mere API failure — an infra fault must not
            # masquerade as an unresolvable clinical field (which would wrongly
            # trigger a critical-field pipeline block, as happened when parallel
            # eval runs exhausted the Groq key pool).
            return regex_value, -1.0, None

    # ─────────────────────────────────────────────────────────────────────────
    # Evidence span finder
    # ─────────────────────────────────────────────────────────────────────────

    def _find_evidence_span(
        self, patient_profile: str, field_name: str
    ) -> Optional[str]:
        """
        Finds and returns the first matching evidence span in the patient profile
        for the given field. Returns None if no pattern matches.
        This span is stored in the validation_log for audit traceability.
        """
        patterns = EVIDENCE_PATTERNS.get(field_name, [])
        profile_lower = patient_profile.lower()

        for pattern in patterns:
            m = re.search(pattern, profile_lower, re.I)
            if m:
                # Return the surrounding sentence (up to 120 chars) for context
                start = max(0, m.start() - 20)
                end = min(len(patient_profile), m.end() + 40)
                return patient_profile[start:end].strip()
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-field consistency checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_tn_stage_consistency(self, attrs: dict) -> str:
        """
        Validates that T_stage + N_stage combination is consistent with stage_group.
        Returns a human-readable note (empty string if consistent).
        """
        t = str(attrs.get("T_stage") or "")
        n = str(attrs.get("N_stage") or "")
        sg = str(attrs.get("stage_group") or "")

        if not t or not n or not sg:
            return ""

        derived = f"T{t}N{n}"

        # Check if stage_group is TN notation that matches
        if sg == derived:
            return f"✅ T{t}N{n} consistent with stage_group={sg}"

        # Check if stage_group is Roman numeral — do basic plausibility check
        # (Full AJCC staging table would be required for exact validation)
        roman_to_tn_ranges = {
            "I":    [("1", "0"), ("1A", "0"), ("1B", "0")],
            "IA":   [("1A", "0"), ("1B", "0")],
            "IB":   [("2A", "0"), ("2B", "0")],
            "II":   [("2", "1"), ("2A", "1"), ("2B", "1")],
            "IIA":  [("2", "1")],
            "IIB":  [("3", "0"), ("3", "1")],
            "III":  [("3", "2"), ("4", "0"), ("4", "1"), ("4", "2"), ("3", "1")],
            "IIIA": [("3", "1"), ("3", "2"), ("4", "0"), ("4", "1")],
            "IIIB": [("3", "2"), ("4", "2")],
        }

        sg_upper = sg.upper()
        if sg_upper in roman_to_tn_ranges:
            valid_pairs = roman_to_tn_ranges[sg_upper]
            if any(t.upper() == vt and n == vn for vt, vn in valid_pairs):
                return f"✅ T{t}N{n} plausibly consistent with {sg}"
            else:
                return (
                    f"⚠️ INCONSISTENT: T{t}N{n} is unusual for Stage {sg}. "
                    f"Expected T/N combinations: {valid_pairs}"
                )

        return f"ℹ️ Stage group {sg} uses TN notation — cross-check: {derived}"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for pipeline integration
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    raw_attrs: dict,
    patient_profile: str,
    call_llm_fn,
    run_id: str,
    patient_id: Optional[str] = None,
    trial_criteria: Optional[dict] = None,
) -> tuple[dict, ValidationReport]:
    """
    Top-level function called from run_agentic_pipeline() in app.py.

    Args:
        raw_attrs:       dict from extract_patient_attributes()
        patient_profile: raw profile text
        call_llm_fn:     matcher.call_llm — used for Tier 2 re-extraction
        run_id:          UUID4 pipeline run identifier
        patient_id:      resolved after ground-truth lookup; None is fine here
        trial_criteria:  parsed_criteria dict from parsing_agent(). When provided,
                         fields that are not tested by this trial's criteria are
                         marked as "not applicable" and skip Tier 2 LLM calls,
                         avoiding unnecessary API cost and latency.

    Example:
        from src.agents.validation_agent import run_validation
        validated_attrs, report = run_validation(
            raw_attrs=extract_patient_attributes(patient_profile, ...),
            patient_profile=patient_profile,
            call_llm_fn=matcher.call_llm,
            run_id=run_id,
            patient_id=None,
            trial_criteria=parsed_criteria,
        )
    """
    agent = ClinicalAttributeValidationAgent(
        call_llm_fn=call_llm_fn,
        run_id=run_id,
        patient_id=patient_id,
    )
    return agent.validate(
        raw_attrs=raw_attrs,
        patient_profile=patient_profile,
        trial_criteria=trial_criteria,
    )