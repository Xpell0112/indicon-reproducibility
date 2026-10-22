#!/usr/bin/env python3
"""
experiments/exp10_revision_fixes.py — INDICON 2026 major-revision analysis fixes
================================================================================
Free / cache-only. NO new API calls, NO pipeline re-run, NO test-set fitting.
Addresses two reviewer points that are pure analysis issues:

  R2.1 (Reviewer #2): "accuracy in the policy comparison appears to exclude
        refusals, contrary to the stated method." -> recompute EVERY system's
        accuracy under ONE convention: refusal counted incorrect, denominator
        N = 270 (the stated method). This repairs the Table I vs Table III
        inconsistency (Table III previously used decided-only denominators).

  R2.2 (Reviewer #2): "significance tests should account for repeated patients
        and trials." -> add trial-cluster (and patient-cluster) robust paired
        tests next to the naive per-pair McNemar:
          (a) cluster sign-flip permutation test on net paired correctness
              (trial clusters: exact over all 2^18 sign flips; patient
               clusters: Monte-Carlo), and
          (b) cluster bootstrap 95% CI + two-sided p of the paired accuracy
              difference (resample clusters with replacement).

Outputs (NEW files, separate from results.md / experiment9):
  experiments/results/experiment10_revision_fixes_testing_data.{json,md}

Every number here is derived from already-cached per-pair predictions in
experiments/results/_cache/.
"""
import json, os, math, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(os.path.dirname(HERE), "frozen_predictions")
RNG = np.random.default_rng(20260815)
B = 5000
MC_PERM = 200_000          # Monte-Carlo sign flips for large (patient) clusters
POS = "eligible"


# ----------------------------------------------------------------------------- loading
def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in ("none", "null", "") else s


def load(path, pred_key="prediction"):
    """pair_id -> {pred, target, patient, trial}. pred None => refusal."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["pair_id"]
        out[pid] = {
            "pred": norm(r.get(pred_key)),
            "target": norm(r.get("target")),
            "patient": pid.split("__")[0],
            "trial": r.get("trial_id"),
        }
    return out


# ----------------------------------------------------------------------------- metrics
def confusion(items):
    """Accuracy computed BOTH ways so the convention is explicit and auditable."""
    N = len(items)
    committed = [x for x in items if x["pred"] is not None]
    tp = sum(1 for x in committed if x["pred"] == POS and x["target"] == POS)
    fp = sum(1 for x in committed if x["pred"] == POS and x["target"] != POS)
    fn = sum(1 for x in committed if x["pred"] != POS and x["target"] == POS)
    tn = sum(1 for x in committed if x["pred"] != POS and x["target"] != POS)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    correct = tp + tn
    acc_full = correct / N if N else 0.0            # refusal = incorrect  (STATED METHOD)
    acc_decided = correct / len(committed) if committed else 0.0  # decided-only (old Table III)
    return {
        "n": N, "committed": len(committed), "refused": N - len(committed),
        "coverage": round(len(committed) / N, 4) if N else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "acc_full": round(acc_full, 4), "acc_decided": round(acc_decided, 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }


def correct_vec(d, ids):
    """1 if system correct on a committed pair, else 0 (refusal => 0 = incorrect)."""
    return np.array([1 if (d[i]["pred"] is not None and d[i]["pred"] == d[i]["target"]) else 0
                     for i in ids], dtype=np.int64)


# ----------------------------------------------------------------------------- McNemar (naive, per-pair)
def mcnemar(a, b, ids):
    b10 = b01 = 0
    for pid in ids:
        ca = a[pid]["pred"] is not None and a[pid]["pred"] == a[pid]["target"]
        cb = b[pid]["pred"] is not None and b[pid]["pred"] == b[pid]["target"]
        if ca and not cb:
            b10 += 1
        elif cb and not ca:
            b01 += 1
    n = b01 + b10
    k = min(b01, b10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)) if n else 1.0
    return {"A_correct_B_wrong": b10, "A_wrong_B_correct": b01, "discordant": n,
            "p_exact": round(p, 5), "significant_0.05": bool(p < 0.05)}


# ----------------------------------------------------------------------------- cluster-robust paired tests
def cluster_sign_flip(a, b, ids, cluster_key, exact_max_clusters=22):
    """Cluster sign-flip permutation test on NET paired correctness.

    d_i = correct_A(i) - correct_B(i) in {-1,0,+1}; per-cluster sum S_c.
    Observed T = sum_c S_c = (#A correct - #B correct) overall.
    Null: each cluster's contribution sign-flips independently (clusters are the
    exchangeable units, so repeated pairs within a trial/patient are respected).
    p = P(|sum_c eps_c S_c| >= |T|).  Exact enumeration when #clusters is small,
    else Monte-Carlo.
    """
    ca = correct_vec(a, ids)
    cb = correct_vec(b, ids)
    d = ca - cb
    clusters = {}
    for i, pid in enumerate(ids):
        clusters.setdefault(a[pid][cluster_key], []).append(i)
    S = np.array([int(d[idx].sum()) for idx in clusters.values()], dtype=np.int64)
    m = len(S)
    T_obs = int(S.sum())
    absT = abs(T_obs)
    if m <= exact_max_clusters:
        # exact: all 2^m sign vectors
        total = 1 << m
        ge = 0
        # iterate in chunks to bound memory
        chunk = 1 << min(m, 16)
        base = np.arange(chunk, dtype=np.int64)
        for start in range(0, total, chunk):
            idx = base[: min(chunk, total - start)] + start
            bits = ((idx[:, None] >> np.arange(m)) & 1).astype(np.int64)
            signs = 1 - 2 * bits
            Tperm = signs @ S
            ge += int((np.abs(Tperm) >= absT).sum())
        p = ge / total
        method = f"exact sign-flip over 2^{m}={total} flips"
    else:
        signs = RNG.integers(0, 2, size=(MC_PERM, m)) * 2 - 1
        Tperm = signs @ S
        p = float((np.abs(Tperm) >= absT).mean())
        method = f"Monte-Carlo sign-flip, {MC_PERM} flips, seed 20260815"
    return {"n_clusters": m, "net_correct_diff_T": T_obs,
            "p_cluster_signflip": round(p, 5), "significant_0.05": bool(p < 0.05),
            "method": method}


def cluster_bootstrap_accdiff(a, b, ids, cluster_key):
    """Cluster bootstrap of the paired accuracy difference (A - B).

    Resample whole clusters with replacement; within a resample, accuracy diff =
    mean over its pairs of (correct_A - correct_B). Returns the point diff, a 95%
    percentile CI, and a two-sided bootstrap p = 2*min(frac<=0, frac>=0).
    """
    ca = correct_vec(a, ids)
    cb = correct_vec(b, ids)
    d = ca - cb
    clusters = {}
    for i, pid in enumerate(ids):
        clusters.setdefault(a[pid][cluster_key], []).append(i)
    keys = list(clusters.keys())
    point = float(d.mean())
    diffs = np.empty(B)
    for bi in range(B):
        chosen = RNG.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([clusters[keys[k]] for k in chosen])
        diffs[bi] = d[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_le = float((diffs <= 0).mean())
    frac_ge = float((diffs >= 0).mean())
    p = min(1.0, 2 * min(frac_le, frac_ge))
    return {"acc_diff_A_minus_B": round(point, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_cluster_bootstrap": round(p, 5), "significant_0.05": bool(p < 0.05)}


def mdtab(headers, rows):
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


# ----------------------------------------------------------------------------- main
def main():
    # Table-I systems come from the exp1 run; the same-run policy comparison
    # (Table III) comes from the single exp8 pass (veto_pred + adv_pred).
    sysdefs = {
        "Symbolic-only": ("experiment1_testing_data_symbolic_only.jsonl", "prediction"),
        "Zero-shot LLM": ("experiment1_testing_data_zero_shot.jsonl", "prediction"),
        "Few-shot LLM": ("experiment1_testing_data_few_shot.jsonl", "prediction"),
        "RAG + LLM": ("experiment1_testing_data_rag_only.jsonl", "prediction"),
        "Hard veto (exp1 run)": ("experiment1_testing_data_full.jsonl", "prediction"),
        "Hard veto (exp8 run)": ("experiment8_testing_data_advisory.jsonl", "veto_pred"),
        "Advisory (exp8 run)": ("experiment8_testing_data_advisory.jsonl", "adv_pred"),
    }
    loaded = {k: load(os.path.join(CACHE, f), key) for k, (f, key) in sysdefs.items()}
    common = sorted(set.intersection(*[set(d.keys()) for d in loaded.values()]))
    patients = {loaded["Zero-shot LLM"][i]["patient"] for i in common}
    trials = {loaded["Zero-shot LLM"][i]["trial"] for i in common}

    # Majority-class ("Not Eligible") baseline over the SAME common ids
    maj = {i: {"pred": "not eligible", "target": loaded["Zero-shot LLM"][i]["target"],
               "patient": loaded["Zero-shot LLM"][i]["patient"],
               "trial": loaded["Zero-shot LLM"][i]["trial"]} for i in common}
    loaded["Majority"] = maj

    conf = {k: confusion([loaded[k][i] for i in common]) for k in loaded}

    # ---- R2.1 : corrected tables ------------------------------------------------
    table1_order = ["Majority", "Symbolic-only", "Zero-shot LLM", "Few-shot LLM",
                    "RAG + LLM", "Hard veto (exp1 run)"]
    table3_order = ["Hard veto (exp8 run)", "Advisory (exp8 run)"]

    # ---- R2.2 : cluster-robust paired tests ------------------------------------
    key_pairs = [
        ("Hard veto (exp1 run)", "Zero-shot LLM"),      # the headline "hybrid worse" claim
        ("Advisory (exp8 run)", "Zero-shot LLM"),       # advisory vs zero-shot (post-hoc exploratory)
        ("Advisory (exp8 run)", "Hard veto (exp8 run)"),  # same-run authority effect
        ("Few-shot LLM", "Zero-shot LLM"),              # null control
        ("RAG + LLM", "Zero-shot LLM"),                 # null control
    ]
    tests = {}
    for a, b in key_pairs:
        tests[f"{a} vs {b}"] = {
            "mcnemar_pairlevel": mcnemar(loaded[a], loaded[b], common),
            "trial_cluster_signflip": cluster_sign_flip(loaded[a], loaded[b], common, "trial"),
            "patient_cluster_signflip": cluster_sign_flip(loaded[a], loaded[b], common, "patient"),
            "trial_cluster_bootstrap": cluster_bootstrap_accdiff(loaded[a], loaded[b], common, "trial"),
        }

    out = {
        "integrity": {"new_api_calls": 0, "pipeline_rerun": False,
                      "test_used_for_fitting": False, "bootstrap_resamples": B,
                      "mc_permutations": MC_PERM, "rng_seed": 20260815},
        "test_meta": {"n_pairs": len(common), "n_patients": len(patients),
                      "n_trials": len(trials)},
        "R2_1_accuracy_convention": {"stated_method": "refusal = incorrect, denominator N",
                                     "systems": conf},
        "R2_2_cluster_significance": tests,
    }
    with open(os.path.join(RES, "experiment10_revision_fixes_testing_data.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---------------- markdown report -------------------------------------------
    md = []
    md.append("# Experiment 10 — INDICON revision analysis fixes (TEST)\n")
    md.append("> Cache-only: no new API calls, no pipeline re-run, no test-set fitting. "
              f"Cluster bootstrap B={B}, seed 20260815; patient-cluster permutation "
              f"{MC_PERM:,} flips; trial-cluster permutation exact.\n\n")
    md.append(f"**Test set:** {len(common)} pairs = {len(patients)} patients x {len(trials)} trials.\n\n")

    md.append("## R2.1 — One accuracy convention (refusal = incorrect, denominator N=270)\n\n")
    md.append("The stated method counts a refusal as an incorrect classification for accuracy. "
              "Table I already followed it; the Table III (policy-comparison) accuracies had used a "
              "decided-only denominator. Below, `Acc(270)` is the corrected, consistent figure; "
              "`Acc(decided)` is the old decided-only value, shown only to document the discrepancy.\n\n")
    rows = []
    for k in table1_order:
        c = conf[k]
        pr = "—" if k == "Majority" else c["precision"]
        rc = "—" if k == "Majority" else c["recall"]
        f1 = "—" if k == "Majority" else c["f1"]
        rows.append([k, c["acc_full"], c["acc_decided"], pr, rc, f1,
                     f"{c['refused']}", c["coverage"]])
    md.append("**Corrected Table I (held-out, 270 pairs):**\n\n")
    md.append(mdtab(["System", "Acc(270)", "Acc(decided)", "Prec", "Rec", "F1", "Ref", "Cov"], rows))

    rows = []
    for k in table3_order:
        c = conf[k]
        rows.append([k.replace(" (exp8 run)", ""), c["acc_full"], c["acc_decided"],
                     c["precision"], c["recall"], c["f1"], c["FN"],
                     f"TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}", f"{c['refused']}"])
    md.append("\n**Corrected Table III (same-run policy comparison, exp8 pass):**\n\n")
    md.append(mdtab(["Policy", "Acc(270)", "Acc(decided)", "Prec", "Rec", "F1", "FN",
                     "Confusion", "Ref"], rows))
    md.append("\n*Fix:* report `Acc(270)` in both tables. Table III's previously-printed "
              ".898 / .936 were decided-only; the consistent values are above.\n")

    md.append("\n## R2.2 — Significance that respects repeated patients & trials\n\n")
    md.append("The 270 pairs are 45 patients x 18 trials, not 270 independent observations. Beside the "
              "naive per-pair McNemar we add two cluster-robust paired tests on net correctness: a "
              "**sign-flip permutation test** over trial clusters (exact, 2^18 flips) and over patient "
              "clusters (Monte-Carlo), and a **trial-cluster bootstrap** 95% CI + p of the paired "
              "accuracy difference (A−B).\n\n")
    rows = []
    for name, t in tests.items():
        mc = t["mcnemar_pairlevel"]
        tr = t["trial_cluster_signflip"]
        pa = t["patient_cluster_signflip"]
        bo = t["trial_cluster_bootstrap"]
        rows.append([
            name,
            f"{mc['p_exact']}{'*' if mc['significant_0.05'] else ''}",
            f"{tr['p_cluster_signflip']}{'*' if tr['significant_0.05'] else ''}",
            f"{pa['p_cluster_signflip']}{'*' if pa['significant_0.05'] else ''}",
            f"{bo['acc_diff_A_minus_B']} {bo['ci95']}",
            f"{bo['p_cluster_bootstrap']}{'*' if bo['significant_0.05'] else ''}",
        ])
    md.append(mdtab(["Comparison (A vs B)", "McNemar p (pair)", "Trial-cluster p",
                     "Patient-cluster p", "Acc diff (A−B) [95% CI]", "Boot p"], rows))
    md.append("\n`*` = significant at 0.05. Trial-cluster p is the primary test (the trial is the unit "
              "of generalization). A CI that excludes 0 agrees with a significant cluster p.\n")

    with open(os.path.join(RES, "experiment10_revision_fixes_testing_data.md"), "w",
              encoding="utf-8") as f:
        f.write("".join(md))

    # ---------------- console -----------------------------------------------------
    print(f"TEST {len(common)} pairs | {len(patients)} patients | {len(trials)} trials\n")
    print("R2.1 corrected accuracy (Acc270 | old decided | F1):")
    for k in table1_order + table3_order:
        c = conf[k]
        print(f"  {k:24s} Acc270={c['acc_full']:.4f}  decided={c['acc_decided']:.4f}  "
              f"F1={c['f1']:.4f}  ref={c['refused']}  TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}")
    print("\nR2.2 paired significance (McNemar | trial-cluster | patient-cluster | boot CI, p):")
    for name, t in tests.items():
        mc = t["mcnemar_pairlevel"]; tr = t["trial_cluster_signflip"]
        pa = t["patient_cluster_signflip"]; bo = t["trial_cluster_bootstrap"]
        print(f"  {name:42s} McN={mc['p_exact']:.4f}  trial={tr['p_cluster_signflip']:.4f}  "
              f"pat={pa['p_cluster_signflip']:.4f}  d={bo['acc_diff_A_minus_B']:+.4f} "
              f"{bo['ci95']} pboot={bo['p_cluster_bootstrap']:.4f}")
    print("\nWrote experiment10_revision_fixes_testing_data.{json,md}")


if __name__ == "__main__":
    main()
