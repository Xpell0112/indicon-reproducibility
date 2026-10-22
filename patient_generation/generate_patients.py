"""
generate_patients.py

Generates synthetic NSCLC patient profiles for the eligibility dataset.

DESIGN PRINCIPLE (integrity-critical):
  Patient attributes are sampled from a realistic NSCLC clinical distribution
  WITHOUT any reference to trial criteria. The narrative rendering step never
  sees any trial's criteria. Each patient is then assigned a mechanism-stratified
  spread of trials. This decouples the patient from any single trial, so labels
  are a consequence of the (patient, trial) combination -- determined by BLIND
  ANNOTATORS, not baked in at generation.

The "provisional_label" this script derives is a deterministic best-guess used
ONLY for balance monitoring during the pilot. It is written to a SEPARATE key
file that annotators must never see. It is NOT ground truth. Ground truth comes
from annotation.

Outputs (three files):
  1. annotation_<batch>.csv    -- BLIND: patient_id, trial_id, patient_profile,
                                   inclusion_criteria, exclusion_criteria
                                   (this is what annotators receive)
  2. internal_key_<batch>.csv  -- SEPARATE, NOT for annotators: patient_id,
                                   trial_id, provisional_label, patient attributes
  3. patients_<batch>.jsonl    -- full structured patient records (attributes)

Usage:
    python generate_patients.py --trials trials_test.jsonl --n 20 \
        --trials-per-patient 6 --seed 20260718 --batch pilot --out-dir ./patients_out
"""

import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# CLINICAL ATTRIBUTE SCHEMA — realistic NSCLC distributions.
# Weights are set BEFORE any trial matching. Tune these (not labels) if the
# pilot balance is off. Distributions loosely reflect NSCLC epidemiology but
# are deliberately spread to produce a mix of eligible/ineligible outcomes.
# ---------------------------------------------------------------------------

def weighted(rng, options):
    """options: list of (value, weight)."""
    vals, wts = zip(*options)
    return rng.choices(vals, weights=wts, k=1)[0]


def sample_patient_attributes(rng, pid):
    """Sample one structured NSCLC patient. No trial criteria consulted."""
    sex = weighted(rng, [("male", 55), ("female", 45)])
    age = int(rng.gauss(64, 10))
    age = max(30, min(88, age))  # clamp to plausible NSCLC range

    histology = weighted(rng, [
        ("adenocarcinoma", 50), ("squamous cell carcinoma", 30),
        ("large cell carcinoma", 8), ("NSCLC not otherwise specified", 12),
    ])

    # Stage: this trial set enrolls predominantly advanced/metastatic disease.
    # Early stage kept as a small minority (they'll fail the advanced-disease trials,
    # producing genuine Not-Eligible cases).
    stage = weighted(rng, [
        ("IIIA", 10), ("IIIB", 28), ("IV", 62),
    ])
    stage_tnm = {
        "IIIA": rng.choice(["T3N1M0", "T2N2M0", "T3N2M0"]),
        "IIIB": rng.choice(["T3N2M0", "T4N2M0", "T2N3M0"]),
        "IV": rng.choice(["T2N2M1a", "T3N2M1b", "T4N3M1c", "T2N3M1a"]),
    }[stage]

    # Measurable disease per RECIST — most of these trials require it
    measurable_disease = weighted(rng, [(True, 78), (False, 22)])

    # Disease course: many of these are 2nd-line trials requiring progression on prior tx
    disease_status = weighted(rng, [
        ("newly diagnosed, treatment-naive", 22),
        ("progressed after prior therapy", 48),
        ("recurrent disease", 18),
        ("stable on current therapy", 12),
    ])

    # Malignant pleural effusion — 5 trials specifically require it
    pleural_effusion = weighted(rng, [(True, 30), (False, 70)])

    # Performance status — mostly expressed as ECOG, a subset as KPS (scale test)
    ps_scale = weighted(rng, [("ECOG", 75), ("KPS", 25)])
    if ps_scale == "ECOG":
        ps_value = weighted(rng, [(0, 30), (1, 45), (2, 18), (3, 6), (4, 1)])
    else:
        ps_value = weighted(rng, [(100, 15), (90, 30), (80, 30), (70, 15), (60, 7), (50, 3)])

    # FEV1 — often unstated (drives Insufficient-Information cases). Raised
    # unstated rate so trials with an FEV1 threshold produce genuine
    # under-specified cases that exercise the abstention tier.
    fev1_stated = weighted(rng, [(True, 42), (False, 58)])
    fev1_pct = rng.randint(35, 95) if fev1_stated else None

    # Biomarkers
    # More actionable mutations so a fair share match the biomarker-gated TKI trials
    biomarker = weighted(rng, [
        ("EGFR exon 19 deletion", 18), ("EGFR L858R", 12), ("ALK rearrangement", 12),
        ("ROS1 fusion", 4), ("KRAS G12C", 6), ("RET fusion", 4),
        ("no actionable mutation", 32), ("not tested", 12),
    ])

    brain_mets = weighted(rng, [
        ("none", 62), ("treated/stable", 18), ("untreated symptomatic", 20),
    ])

    # Prior therapies — must be consistent with disease_status. A patient who
    # "progressed after prior therapy" or "recurrent" MUST have had prior therapy.
    requires_prior = disease_status in ("progressed after prior therapy", "recurrent disease")
    if requires_prior:
        prior_chemo = weighted(rng, [(True, 62), (False, 38)])
        prior_tki = weighted(rng, [(True, 35), (False, 65)])
        prior_io = weighted(rng, [(True, 28), (False, 72)])
        prior_radio = weighted(rng, [(True, 30), (False, 70)])
        # guarantee at least one prior therapy for progressed/recurrent patients
        if not (prior_chemo or prior_tki or prior_io or prior_radio):
            prior_chemo = True
    else:
        prior_chemo = weighted(rng, [(True, 20), (False, 80)])
        prior_radio = weighted(rng, [(True, 12), (False, 88)])
        prior_tki = weighted(rng, [(True, 8), (False, 92)])
        prior_io = weighted(rng, [(True, 8), (False, 92)])
    # timing of most recent prior therapy (for temporal exclusions)
    prior_timing = weighted(rng, [
        ("within 1 month", 15), ("1-3 months ago", 25),
        ("3-6 months ago", 25), ("over 6 months ago", 35),
    ]) if (prior_chemo or prior_radio or prior_tki or prior_io) else None

    neuropathy = weighted(rng, [
        ("none", 68), ("Grade 1", 16), ("Grade 2", 11), ("Grade 3", 5),
    ])

    cardiovascular = weighted(rng, [
        ("none", 68), ("MI within 6 months", 8), ("MI over 6 months ago", 7),
        ("controlled arrhythmia", 9), ("unstable angina recent", 8),
    ])

    active_infection = weighted(rng, [(True, 12), (False, 88)])

    # Labs — sometimes unstated (raised unstated rate for more II cases)
    labs_stated = weighted(rng, [(True, 48), (False, 52)])
    if labs_stated:
        creatinine = round(rng.uniform(0.6, 2.2), 1)
        anc = round(rng.uniform(1.2, 6.0), 1)
        platelets = rng.randint(80, 380)
        hemoglobin = round(rng.uniform(8.5, 15.5), 1)
        bilirubin = round(rng.uniform(0.3, 2.5), 1)
    else:
        creatinine = anc = platelets = hemoglobin = bilirubin = None

    autoimmune = weighted(rng, [(True, 10), (False, 90)])

    pregnancy_applicable = (sex == "female" and age <= 50)
    pregnant = weighted(rng, [(True, 8), (False, 92)]) if pregnancy_applicable else False

    return {
        "patient_id": pid,
        "sex": sex, "age": age, "histology": histology,
        "stage": stage, "tnm": stage_tnm,
        "measurable_disease": measurable_disease,
        "disease_status": disease_status,
        "pleural_effusion": pleural_effusion,
        "ps_scale": ps_scale, "ps_value": ps_value,
        "fev1_stated": fev1_stated, "fev1_pct": fev1_pct,
        "biomarker": biomarker, "brain_mets": brain_mets,
        "prior_chemo": prior_chemo, "prior_radio": prior_radio,
        "prior_tki": prior_tki, "prior_io": prior_io, "prior_timing": prior_timing,
        "neuropathy": neuropathy, "cardiovascular": cardiovascular,
        "active_infection": active_infection,
        "labs_stated": labs_stated, "creatinine": creatinine, "anc": anc,
        "platelets": platelets, "hemoglobin": hemoglobin, "bilirubin": bilirubin,
        "autoimmune": autoimmune,
        "pregnancy_applicable": pregnancy_applicable, "pregnant": pregnant,
    }


# ---------------------------------------------------------------------------
# NARRATIVE RENDERING — varied free text. NO trial criteria consulted.
# Multiple templates + shuffled optional clauses to avoid a leakable structure.
# ---------------------------------------------------------------------------

def render_narrative(rng, a):
    name_prefix = "Mr." if a["sex"] == "male" else "Ms."
    pron = "He" if a["sex"] == "male" else "She"
    pron_l = pron.lower()
    poss = "his" if a["sex"] == "male" else "her"

    # Opening sentence — a few variants
    openers = [
        f"A {a['age']}-year-old, has histologically "
        f"confirmed {a['histology']} of the lung, staged as {a['stage']} ({a['tnm']}).",
        f"A {a['age']}-year-old {a['sex']} patient presents with {a['histology']}, "
        f"clinical stage {a['stage']} ({a['tnm']}) NSCLC.",
        f"A {a['age']}-year-old with stage "
        f"{a['stage']} ({a['tnm']}) {a['histology']} of the lung.",
    ]
    sentences = [rng.choice(openers)]

    # Disease course
    status_txt = {
        "newly diagnosed, treatment-naive": "The disease is newly diagnosed and the patient is treatment-naive.",
        "progressed after prior therapy": "The disease has progressed following prior systemic therapy.",
        "recurrent disease": "The patient has recurrent disease after previous treatment.",
        "stable on current therapy": "The disease is currently stable on ongoing therapy.",
    }[a["disease_status"]]
    sentences.append(status_txt)

    # Measurable disease
    if a["measurable_disease"]:
        sentences.append("Imaging shows measurable disease per RECIST v1.1.")

    # Pleural effusion
    if a["pleural_effusion"]:
        sentences.append("A malignant pleural effusion is present.")

    # Performance status
    if a["ps_scale"] == "ECOG":
        sentences.append(f"{pron} has an ECOG performance status of {a['ps_value']}.")
    else:
        sentences.append(f"{poss.capitalize()} Karnofsky Performance Status is {a['ps_value']}.")

    # Biomarker
    if a["biomarker"] == "not tested":
        sentences.append("Molecular testing has not been performed.")
    elif a["biomarker"] == "no actionable mutation":
        sentences.append("Molecular profiling showed no actionable driver mutation.")
    else:
        sentences.append(f"Molecular testing revealed {a['biomarker']}.")

    # Optional clauses — shuffled so structure varies
    optional = []
    if a["fev1_stated"]:
        optional.append(f"Pulmonary function testing showed an FEV1 of {a['fev1_pct']}% predicted.")
    if a["brain_mets"] != "none":
        bm = {"treated/stable": "treated, currently stable brain metastases",
              "untreated symptomatic": "untreated symptomatic brain metastases"}[a["brain_mets"]]
        optional.append(f"Imaging demonstrated {bm}.")
    priors = []
    if a["prior_chemo"]: priors.append("chemotherapy")
    if a["prior_radio"]: priors.append("radiotherapy")
    if a["prior_tki"]: priors.append("TKI therapy")
    if a["prior_io"]: priors.append("immunotherapy")
    if priors:
        timing = f" ({a['prior_timing']})" if a["prior_timing"] else ""
        optional.append(f"{pron} previously received {', '.join(priors)}{timing}.")
    else:
        optional.append(f"{pron} has received no prior systemic or radiation therapy.")
    if a["neuropathy"] != "none":
        optional.append(f"{pron} has peripheral neuropathy {a['neuropathy']}.")
    if a["cardiovascular"] != "none":
        cv = {"MI within 6 months": "a myocardial infarction within the past 6 months",
              "MI over 6 months ago": "a myocardial infarction more than 6 months ago",
              "controlled arrhythmia": "a controlled cardiac arrhythmia",
              "unstable angina recent": "recent unstable angina"}[a["cardiovascular"]]
        optional.append(f"Cardiac history includes {cv}.")
    if a["active_infection"]:
        optional.append(f"{pron} currently has an active infection requiring treatment.")
    if a["autoimmune"]:
        optional.append(f"{pron} has an active autoimmune condition on systemic therapy.")
    if a["labs_stated"]:
        optional.append(
            f"Laboratory values: creatinine {a['creatinine']} mg/dL, ANC {a['anc']} x10^3, "
            f"platelets {a['platelets']}k, hemoglobin {a['hemoglobin']} g/dL, "
            f"bilirubin {a['bilirubin']} mg/dL.")
    if a["pregnant"]:
        optional.append(f"{pron} is currently pregnant.")

    rng.shuffle(optional)
    sentences.extend(optional)
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# PROVISIONAL LABEL — deterministic best-guess, BALANCE MONITORING ONLY.
# This is a heuristic, NOT ground truth. Only checks the most common, clearly
# checkable criteria. Returns Eligible / Not Eligible / Insufficient Information.
# ---------------------------------------------------------------------------

def provisional_label(a, trial):
    """Strict binary provisional label for balance monitoring. NOT authoritative.
    Convention: a required field that the profile never states -> Not Eligible
    (can't verify -> can't enroll). Also returns would_defer: whether the real
    framework would likely BLOCK/defer this case (critical field unverifiable or
    low confidence), which is scored separately from wrong predictions."""
    inc = (trial["inclusion_criteria"] or "").lower()
    exc = (trial["exclusion_criteria"] or "").lower()
    would_defer = False

    # --- clear exclusion triggers ---
    if a["brain_mets"] == "untreated symptomatic" and ("brain metasta" in exc or "cns" in exc):
        return "Not Eligible", would_defer
    if a["active_infection"] and "infection" in exc:
        return "Not Eligible", would_defer
    if a["cardiovascular"] in ("MI within 6 months", "unstable angina recent") and \
       ("myocardial" in exc or "angina" in exc or "cardiovascular" in exc):
        return "Not Eligible", would_defer
    if a["neuropathy"] in ("Grade 2", "Grade 3") and "neuropathy" in exc:
        return "Not Eligible", would_defer
    if a["pregnant"] and "pregnan" in exc:
        return "Not Eligible", would_defer
    if a["autoimmune"] and "autoimmune" in exc:
        return "Not Eligible", would_defer

    # --- prior therapy exclusions ---
    if a["prior_chemo"] and ("prior" in exc and "chemo" in exc):
        return "Not Eligible", would_defer
    if a["prior_io"] and ("prior" in exc and ("immuno" in exc or "pd-1" in exc or "pd-l1" in exc)):
        return "Not Eligible", would_defer
    if a["prior_tki"] and ("prior" in exc and ("tki" in exc or "tyrosine kinase" in exc or "inhibitor" in exc)):
        return "Not Eligible", would_defer

    # --- performance status inclusion check (ECOG normalized) ---
    ecog = a["ps_value"] if a["ps_scale"] == "ECOG" else \
           (0 if a["ps_value"] >= 90 else 1 if a["ps_value"] >= 70 else 2 if a["ps_value"] >= 50 else 3)
    if "performance status" in inc or "ecog" in inc or "zubrod" in inc:
        cap = 2 if ("0, 1, or 2" in inc or "0-2" in inc or "0 to 2" in inc or "0, 1 or 2" in inc) else 1
        if ecog > cap:
            return "Not Eligible", would_defer

    # --- biomarker inclusion check ---
    requires_egfr = "egfr" in inc and ("mutation" in inc or "sensitizing" in inc or "exon" in inc)
    requires_alk = "alk" in inc and ("rearrangement" in inc or "positive" in inc or "fusion" in inc)
    requires_ret = "ret" in inc and "fusion" in inc
    requires_any_biomarker = requires_egfr or requires_alk or requires_ret
    patient_egfr = "egfr" in a["biomarker"].lower()
    patient_alk = "alk" in a["biomarker"].lower()
    patient_ret = "ret" in a["biomarker"].lower()
    # trial needs a biomarker but patient never tested -> framework would DEFER
    # (critical field unverifiable), gold label strict = Not Eligible
    if requires_any_biomarker and a["biomarker"] == "not tested":
        return "Not Eligible", True
    if requires_egfr and not patient_egfr:
        return "Not Eligible", would_defer
    if requires_alk and not patient_alk:
        return "Not Eligible", would_defer
    if requires_ret and not patient_ret:
        return "Not Eligible", would_defer

    # --- stage inclusion check ---
    wants_advanced = ("stage iv" in inc or "metastatic" in inc or "advanced" in inc or "stage iiib" in inc)
    wants_early = ("stage i" in inc and "stage iv" not in inc) or "resectable" in inc or "early stage" in inc
    if wants_advanced and a["stage"] in ("I", "II"):
        return "Not Eligible", would_defer
    if wants_early and a["stage"] in ("IIIB", "IV"):
        return "Not Eligible", would_defer

    # --- pleural effusion requirement (5 trials REQUIRE it as core inclusion) ---
    if "pleural effusion" in inc and not a["pleural_effusion"]:
        return "Not Eligible", would_defer

    # --- measurable disease requirement ---
    if "measurable" in inc and not a["measurable_disease"]:
        return "Not Eligible", would_defer

    # --- progression / prior-treatment requirement ---
    needs_progression = ("progress" in inc or "failed" in inc or "relapsed" in inc
                         or "refractory" in inc or "after treatment" in inc
                         or "previously treated" in inc or "second-line" in inc)
    if needs_progression and a["disease_status"] == "newly diagnosed, treatment-naive":
        return "Not Eligible", would_defer
    if "recurrent" in inc and a["disease_status"] not in ("recurrent disease", "progressed after prior therapy"):
        return "Not Eligible", would_defer

    # --- strict convention: required field unstated -> Not Eligible + framework DEFERS ---
    # (framework blocks on unverifiable critical field; gold label is strict NE)
    if ("fev1" in inc or "fev 1" in inc) and not a["fev1_stated"]:
        return "Not Eligible", True
    labs_required = any(k in inc for k in
                        ["creatinine", "renal function", "anc", "neutrophil",
                         "platelet", "hemoglobin", "bilirubin", "hepatic function",
                         "adequate organ", "adequate bone marrow"])
    if labs_required and not a["labs_stated"]:
        return "Not Eligible", True

    # default: passes all checks we can evaluate
    return "Eligible", would_defer


# ---------------------------------------------------------------------------
# MECHANISM-STRATIFIED TRIAL ASSIGNMENT
# ---------------------------------------------------------------------------

def assign_trials(rng, trials, trials_per_patient):
    """Assign a mechanism-stratified spread: draw from distinct mechanism
    classes first, so each patient sees varied treatment modalities."""
    by_mech = defaultdict(list)
    for t in trials:
        by_mech[t["mechanism_class"]].append(t)
    mechs = list(by_mech.keys())
    rng.shuffle(mechs)

    assigned = []
    # round-robin across mechanism classes until we have enough
    mi = 0
    used = set()
    while len(assigned) < trials_per_patient and len(used) < len(trials):
        mech = mechs[mi % len(mechs)]
        pool = [t for t in by_mech[mech] if t["trial_id"] not in used]
        if pool:
            choice = rng.choice(pool)
            assigned.append(choice)
            used.add(choice["trial_id"])
        mi += 1
    return assigned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--trials-per-patient", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--batch", type=str, default="pilot")
    ap.add_argument("--out-dir", type=Path, default=Path("patients_out"))
    ap.add_argument("--id-start", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trials = [json.loads(l) for l in open(args.trials)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = []
    annotation_rows = []
    key_rows = []

    for i in range(args.n):
        pid = f"P{args.id_start + i:04d}"
        attrs = sample_patient_attributes(rng, pid)
        narrative = render_narrative(rng, attrs)
        attrs["narrative"] = narrative
        patients.append(attrs)

        assigned = assign_trials(rng, trials, args.trials_per_patient)
        for t in assigned:
            annotation_rows.append({
                "pair_id": f"{pid}__{t['trial_id']}",
                "patient_id": pid,
                "trial_id": t["trial_id"],
                "patient_profile": narrative,
                "inclusion_criteria": t["inclusion_criteria"],
                "exclusion_criteria": t["exclusion_criteria"],
            })
            prov_label, would_defer = provisional_label(attrs, t)
            key_rows.append({
                "pair_id": f"{pid}__{t['trial_id']}",
                "patient_id": pid,
                "trial_id": t["trial_id"],
                "mechanism_class": t["mechanism_class"],
                "provisional_label": prov_label,
                "would_defer": would_defer,
                "stage": attrs["stage"], "ecog_kps": f"{attrs['ps_scale']} {attrs['ps_value']}",
                "biomarker": attrs["biomarker"], "brain_mets": attrs["brain_mets"],
                "fev1_stated": attrs["fev1_stated"], "labs_stated": attrs["labs_stated"],
            })

    # write blind annotation file
    ann_path = args.out_dir / f"annotation_{args.batch}.csv"
    with open(ann_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(annotation_rows[0].keys()))
        w.writeheader(); w.writerows(annotation_rows)

    # write SEPARATE internal key (never given to annotators)
    key_path = args.out_dir / f"internal_key_{args.batch}.csv"
    with open(key_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)

    # write full structured patients
    pat_path = args.out_dir / f"patients_{args.batch}.jsonl"
    with open(pat_path, "w") as f:
        for p in patients:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # balance report (from provisional labels)
    from collections import Counter
    bal = Counter(r["provisional_label"] for r in key_rows)
    defer_count = sum(1 for r in key_rows if r["would_defer"])
    total = len(key_rows)
    print(f"Generated {len(patients)} patients, {total} pairs (batch: {args.batch})")
    print(f"\nProvisional label balance (BALANCE MONITORING ONLY, not ground truth):")
    print(f"  (strict convention: unstated required field -> Not Eligible)")
    for label in ["Eligible", "Not Eligible"]:
        c = bal.get(label, 0)
        print(f"  {label:14s}: {c:3d}  ({100*c/total:.1f}%)")
    print(f"\n  Pairs the framework would likely DEFER (block, scored separately): "
          f"{defer_count} ({100*defer_count/total:.1f}%)")
    print(f"    -- these are unverifiable-critical-field cases, not wrong predictions")
    print(f"\nFiles written to {args.out_dir}/:")
    print(f"  {ann_path.name}      <- BLIND, give to annotators")
    print(f"  {key_path.name}    <- SEPARATE, DO NOT show annotators")
    print(f"  {pat_path.name}")


if __name__ == "__main__":
    main()