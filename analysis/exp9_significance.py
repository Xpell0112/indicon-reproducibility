#!/usr/bin/env python3
"""
experiments/exp9_significance.py — paired significance, bootstrap CIs, clustering
=================================================================================
Free / cache-only. NO new API calls, NO pipeline re-run, NO test-set fitting.

Closes reviewer concern MODERATE-5 (no statistical inference) and documents the
clean test protocol (advisory justified on DEV, not chosen on test):

  * Per-system TEST metrics (accuracy on all 270 with refusal=incorrect; precision/
    recall/F1 on the Eligible class over committed predictions; coverage).
  * Bootstrap 95% CIs three ways — pair-level, patient-cluster, trial-cluster —
    to show how CIs widen once the non-independence of 270 pairs (= 45 patients x
    18 trials, ~6 trials/patient) is respected.
  * McNemar's exact paired test between systems on the identical 270 pairs.
  * DEV-only justification for the advisory variant (advisory vs veto on the 130
    dev pairs + McNemar) so selecting advisory is a development decision.

Outputs (NEW files only, kept separate from results.md / experiment3):
  experiments/results/experiment9_significance_testing_data.{json,md}
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(os.path.dirname(HERE), "frozen_predictions")
RNG = np.random.default_rng(20260815)
B = 5000
POS = "eligible"


def norm(v):
    return None if v is None else str(v).strip().lower()


def load(path, pred_key="prediction"):
    """pair_id -> {pred, target, patient, trial}. pred None => refusal."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["pair_id"]
        out[pid] = {"pred": norm(r.get(pred_key)), "target": norm(r.get("target")),
                    "patient": pid.split("__")[0], "trial": r.get("trial_id")}
    return out


def confusion(items):
    """items: list of dicts with pred/target. Refusals (pred None) excluded from
    the confusion cells but counted for accuracy_full and coverage."""
    N = len(items)
    committed = [x for x in items if x["pred"] is not None]
    tp = sum(1 for x in committed if x["pred"] == POS and x["target"] == POS)
    fp = sum(1 for x in committed if x["pred"] == POS and x["target"] != POS)
    fn = sum(1 for x in committed if x["pred"] != POS and x["target"] == POS)
    tn = sum(1 for x in committed if x["pred"] != POS and x["target"] != POS)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc_full = sum(1 for x in items if x["pred"] is not None and x["pred"] == x["target"]) / N
    return {"n": N, "committed": len(committed), "coverage": round(len(committed) / N, 4),
            "refused": N - len(committed), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy_full": round(acc_full, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def _metric(items, which):
    m = confusion(items)
    return m[which]


def boot_ci(items, which, unit="pair"):
    """Percentile 95% CI. unit: 'pair' | 'patient' | 'trial' (cluster bootstrap)."""
    if unit == "pair":
        arr = items
        vals = []
        idx = np.arange(len(arr))
        for _ in range(B):
            s = [arr[i] for i in RNG.choice(idx, len(idx), replace=True)]
            vals.append(_safe(s, which))
    else:
        groups = {}
        for x in items:
            groups.setdefault(x[unit], []).append(x)
        keys = list(groups.keys())
        vals = []
        for _ in range(B):
            chosen = RNG.choice(len(keys), len(keys), replace=True)
            s = []
            for k in chosen:
                s.extend(groups[keys[k]])
            vals.append(_safe(s, which))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _safe(items, which):
    try:
        return _metric(items, which)
    except ZeroDivisionError:
        return 0.0


def mcnemar(a, b, common):
    """Exact two-sided McNemar on paired correctness over 'common' pair_ids.
    Returns discordant counts and exact binomial p."""
    b01 = b10 = 0  # b01: A wrong, B correct ; b10: A correct, B wrong
    for pid in common:
        ca = a[pid]["pred"] is not None and a[pid]["pred"] == a[pid]["target"]
        cb = b[pid]["pred"] is not None and b[pid]["pred"] == b[pid]["target"]
        if ca and not cb:
            b10 += 1
        elif cb and not ca:
            b01 += 1
    n = b01 + b10
    k = min(b01, b10)
    # exact two-sided binomial p at theta=0.5
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)) if n > 0 else 1.0
    # continuity-corrected chi-square (reported alongside)
    chi = ((abs(b01 - b10) - 1) ** 2) / n if n > 0 else 0.0
    return {"A_correct_B_wrong": b10, "A_wrong_B_correct": b01, "discordant": n,
            "p_exact": round(p, 5), "chi2_cc": round(chi, 3),
            "significant_0.05": bool(p < 0.05)}


def system_block(name, items):
    m = confusion(items)
    m["name"] = name
    m["ci95"] = {
        "f1_pair": boot_ci(items, "f1", "pair"),
        "f1_patient_cluster": boot_ci(items, "f1", "patient"),
        "f1_trial_cluster": boot_ci(items, "f1", "trial"),
        "accuracy_pair": boot_ci(items, "accuracy_full", "pair"),
        "accuracy_trial_cluster": boot_ci(items, "accuracy_full", "trial"),
    }
    return m


def to_items(d, ids):
    return [d[i] for i in ids]


def mdtab(headers, rows):
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    # ---------- TEST systems (align on the common pair set) ----------
    sysdefs = {
        "Zero-shot LLM": ("experiment1_testing_data_zero_shot.jsonl", "prediction"),
        "Few-shot LLM": ("experiment1_testing_data_few_shot.jsonl", "prediction"),
        "RAG-only LLM": ("experiment1_testing_data_rag_only.jsonl", "prediction"),
        "Symbolic-only gate": ("experiment1_testing_data_symbolic_only.jsonl", "prediction"),
        "Full framework (hard-veto)": ("experiment1_testing_data_full.jsonl", "prediction"),
        "Advisory variant": ("experiment8_testing_data_advisory.jsonl", "adv_pred"),
        "Hard-veto (exp8 run)": ("experiment8_testing_data_advisory.jsonl", "veto_pred"),
    }
    loaded = {k: load(os.path.join(CACHE, f), key) for k, (f, key) in sysdefs.items()}
    common = sorted(set.intersection(*[set(d.keys()) for d in loaded.values()]))
    n_patients = len({loaded["Zero-shot LLM"][i]["patient"] for i in common})
    n_trials = len({loaded["Zero-shot LLM"][i]["trial"] for i in common})

    blocks = {k: system_block(k, to_items(v, common)) for k, v in loaded.items()}

    # ---------- key paired McNemar tests on TEST ----------
    pairs = [
        ("Full framework (hard-veto)", "Zero-shot LLM"),
        ("Advisory variant", "Zero-shot LLM"),
        ("Advisory variant", "Hard-veto (exp8 run)"),
        ("Full framework (hard-veto)", "Symbolic-only gate"),
        ("Zero-shot LLM", "Few-shot LLM"),
        ("Zero-shot LLM", "RAG-only LLM"),
    ]
    mc = {f"{a} vs {b}": mcnemar(loaded[a], loaded[b], common) for a, b in pairs}

    # ---------- DEV justification for advisory (dev-only) ----------
    dev = load(os.path.join(CACHE, "experiment8_dev130_advisory.jsonl"), "adv_pred")
    dev_veto = load(os.path.join(CACHE, "experiment8_dev130_advisory.jsonl"), "veto_pred")
    dcommon = sorted(set(dev) & set(dev_veto))
    dev_adv_m = confusion(to_items(dev, dcommon))
    dev_veto_m = confusion(to_items(dev_veto, dcommon))
    dev_mcnemar = mcnemar(dev, dev_veto, dcommon)

    out = {
        "integrity": {"new_api_calls": 0, "pipeline_rerun": False,
                      "test_used_for_fitting": False, "bootstrap_resamples": B,
                      "rng_seed": 20260815},
        "test_meta": {"n_pairs": len(common), "n_patients": n_patients,
                      "n_trials": n_trials, "avg_pairs_per_patient": round(len(common) / n_patients, 2)},
        "test_systems": blocks,
        "test_mcnemar": mc,
        "dev_advisory_justification": {
            "advisory": dev_adv_m, "veto": dev_veto_m, "mcnemar_adv_vs_veto": dev_mcnemar},
    }
    with open(os.path.join(RES, "experiment9_significance_testing_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---------- markdown ----------
    md = []
    md.append("# Experiment 9 — Paired significance, bootstrap CIs, clustering (TEST)\n")
    md.append("> **Separate, un-merged results.** Cache-only: no new API calls, no pipeline "
              "re-run, no test-set fitting. Bootstrap B=%d, seed 20260815.\n\n" % B)
    md.append(f"**Test set:** {len(common)} pairs = **{n_patients} patients x {n_trials} trials** "
              f"(~{len(common)/n_patients:.1f} trials/patient). The 270 pairs are **not independent** — "
              "they are repeated measures clustered by patient and by trial. Pair-level CIs therefore "
              "*understate* uncertainty; the patient- and trial-cluster bootstraps below are the honest ones.\n\n")

    md.append("## 1. Per-system TEST metrics with 95% bootstrap CIs\n")
    md.append("Accuracy is over all 270 (refusal counted incorrect); precision/recall/F1 are for the "
              "**Eligible** (minority, clinically-critical) class over committed predictions.\n\n")
    rows = []
    order = ["Zero-shot LLM", "Few-shot LLM", "RAG-only LLM", "Symbolic-only gate",
             "Full framework (hard-veto)", "Advisory variant"]
    for k in order:
        m = blocks[k]
        rows.append([k, m["accuracy_full"], m["f1"], m["recall"], m["precision"],
                     f"{m['coverage']} ({m['refused']} ref)"])
    md.append(mdtab(["System", "Acc(270)", "F1(elig)", "Recall", "Precision", "Coverage"], rows))
    md.append("\n### F1 CIs — pair vs patient-cluster vs trial-cluster (note the widening)\n\n")
    rows = []
    for k in order:
        c = blocks[k]["ci95"]
        rows.append([k, blocks[k]["f1"], c["f1_pair"], c["f1_patient_cluster"], c["f1_trial_cluster"]])
    md.append(mdtab(["System", "F1", "CI pair", "CI patient-cluster", "CI trial-cluster"], rows))

    md.append("\n## 2. McNemar's exact paired test (TEST, identical 270 pairs)\n\n")
    rows = []
    for name, r in mc.items():
        rows.append([name, f"{r['A_correct_B_wrong']}/{r['A_wrong_B_correct']}", r["discordant"],
                     r["p_exact"], "YES" if r["significant_0.05"] else "no"])
    md.append(mdtab(["Comparison (A vs B)", "A>B / B>A", "discordant", "p (exact)", "sig p<.05"], rows))
    md.append("\n*A>B / B>A* = pairs where only A / only B was correct. p is the two-sided exact "
              "binomial McNemar p-value.\n")

    md.append("\n## 3. Clean test protocol — advisory justified on DEV (not chosen on test)\n\n")
    md.append("To remove the researcher-degree-of-freedom of picking the method after seeing test "
              "results, the advisory variant is selected on the **development set only**; the test set "
              "is then a single confirmatory report.\n\n")
    md.append(mdtab(["DEV (130 pairs)", "Accuracy", "F1(elig)", "Recall", "Precision", "FN"],
                    [["Hard-veto", dev_veto_m["accuracy_full"], dev_veto_m["f1"], dev_veto_m["recall"],
                      dev_veto_m["precision"], dev_veto_m["FN"]],
                     ["Advisory", dev_adv_m["accuracy_full"], dev_adv_m["f1"], dev_adv_m["recall"],
                      dev_adv_m["precision"], dev_adv_m["FN"]]]))
    dm = dev_mcnemar
    md.append(f"\nDEV McNemar advisory vs veto: advisory-only-correct {dm['A_correct_B_wrong']}, "
              f"veto-only-correct {dm['A_wrong_B_correct']}, discordant {dm['discordant']}, "
              f"exact p = {dm['p_exact']} ({'significant' if dm['significant_0.05'] else 'not significant'}). "
              "The direction of improvement (recall/FN) is established on dev; test is confirmatory.\n")

    md.append("\n## 4. Honest reading\n")
    md.append("- Report every headline number as **F1 [trial-cluster CI]**, not a bare point value.\n")
    md.append("- If the LLM-vs-hybrid McNemar is significant, say the hybrid is significantly *worse*; "
              "if advisory-vs-LLM is *not* significant, that is the honest basis for the “LLM-parity” claim.\n")
    md.append("- With only %d trials, the trial-cluster CI is wide by construction; do not over-claim "
              "small gaps.\n" % n_trials)

    with open(os.path.join(RES, "experiment9_significance_testing_data.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))

    # console summary
    print(f"TEST: {len(common)} pairs | {n_patients} patients | {n_trials} trials")
    for k in order:
        m = blocks[k]
        print(f"  {k:32s} acc={m['accuracy_full']:.3f} F1={m['f1']:.3f} "
              f"F1CI(trial)={m['ci95']['f1_trial_cluster']} refused={m['refused']}")
    print("\nMcNemar (exact p):")
    for name, r in mc.items():
        print(f"  {name:48s} disc={r['discordant']:2d} p={r['p_exact']} sig={r['significant_0.05']}")
    print(f"\nDEV advisory vs veto: F1 {dev_veto_m['f1']:.3f}->{dev_adv_m['f1']:.3f}, "
          f"recall {dev_veto_m['recall']:.3f}->{dev_adv_m['recall']:.3f}, "
          f"FN {dev_veto_m['FN']}->{dev_adv_m['FN']}, McNemar p={dev_mcnemar['p_exact']}")
    print("\nWrote experiment9_significance_testing_data.{json,md}")


if __name__ == "__main__":
    main()
