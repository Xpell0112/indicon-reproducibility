#!/usr/bin/env python3
"""
experiments/exp11_robustness.py — leave-one-trial-out robustness (TEST)
=======================================================================
Free / cache-only. NO new API calls, NO pipeline re-run, NO test-set fitting.

Strengthens the INDICON R2.4 response (advisory policy chosen after inspecting
test errors) WITHOUT a fresh annotated set, by showing the key effects are not
driven by a few trials. For each of the 18 held-out trials we drop it and
recompute F1 / recall over the remaining 17 trials, for:

  * Advisory vs Hard-veto (does advisory still beat veto when any trial is removed?)
  * Advisory vs Zero-shot (does non-superiority hold under every drop?)
  * Hard-veto vs Zero-shot (is the headline "hybrid worse" stable?)

We report, per comparison, the observed full-set ΔF1 and the min/max/mean ΔF1 and
sign-stability across the 18 leave-one-trial-out folds. Stable signs => the effect
is a property of the trial population, not an artifact of one or two trials.

Outputs (NEW files): experiments/results/experiment11_robustness_testing_data.{json,md}
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(os.path.dirname(HERE), "frozen_predictions")
POS = "eligible"


def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in ("none", "null", "") else s


def load(path, pred_key="prediction"):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["pair_id"]
        out[pid] = {"pred": norm(r.get(pred_key)), "target": norm(r.get("target")),
                    "trial": r.get("trial_id")}
    return out


def f1_recall(items):
    c = [x for x in items if x["pred"] is not None]
    tp = sum(1 for x in c if x["pred"] == POS and x["target"] == POS)
    fp = sum(1 for x in c if x["pred"] == POS and x["target"] != POS)
    fn = sum(1 for x in c if x["pred"] != POS and x["target"] == POS)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, rec


def loto(a, b, ids):
    """Leave-one-trial-out: drop each trial, recompute ΔF1 (A−B) and Δrecall over the rest."""
    trials = sorted({a[i]["trial"] for i in ids})
    full_fa, full_ra = f1_recall([a[i] for i in ids])
    full_fb, full_rb = f1_recall([b[i] for i in ids])
    full_df1 = full_fa - full_fb
    full_drec = full_ra - full_rb
    df1s, drecs = [], []
    per = []
    for t in trials:
        keep = [i for i in ids if a[i]["trial"] != t]
        fa, ra = f1_recall([a[i] for i in keep])
        fb, rb = f1_recall([b[i] for i in keep])
        df1s.append(fa - fb)
        drecs.append(ra - rb)
        per.append({"dropped_trial": t, "dF1": round(fa - fb, 4), "dRecall": round(ra - rb, 4)})
    df1s = np.array(df1s); drecs = np.array(drecs)
    same_sign_f1 = int(np.sum(np.sign(df1s) == np.sign(full_df1))) if full_df1 != 0 else None
    return {
        "full": {"F1_A": round(full_fa, 4), "F1_B": round(full_fb, 4),
                 "dF1": round(full_df1, 4), "dRecall": round(full_drec, 4)},
        "loto_dF1": {"min": round(float(df1s.min()), 4), "max": round(float(df1s.max()), 4),
                     "mean": round(float(df1s.mean()), 4)},
        "loto_dRecall": {"min": round(float(drecs.min()), 4), "max": round(float(drecs.max()), 4),
                         "mean": round(float(drecs.mean()), 4)},
        "n_folds": len(trials), "folds_same_sign_as_full_dF1": same_sign_f1,
        "per_trial": per,
    }


def mdtab(headers, rows):
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    zs = load(os.path.join(CACHE, "experiment1_testing_data_zero_shot.jsonl"), "prediction")
    adv = load(os.path.join(CACHE, "experiment8_testing_data_advisory.jsonl"), "adv_pred")
    veto = load(os.path.join(CACHE, "experiment8_testing_data_advisory.jsonl"), "veto_pred")
    ids = sorted(set(zs) & set(adv) & set(veto))

    comps = {
        "Advisory vs Hard-veto": loto(adv, veto, ids),
        "Advisory vs Zero-shot": loto(adv, zs, ids),
        "Hard-veto vs Zero-shot": loto(veto, zs, ids),
    }
    out = {"integrity": {"new_api_calls": 0, "pipeline_rerun": False, "test_used_for_fitting": False},
           "n_pairs": len(ids), "n_trials": len({zs[i]["trial"] for i in ids}),
           "leave_one_trial_out": comps}
    with open(os.path.join(RES, "experiment11_robustness_testing_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = ["# Experiment 11 — Leave-one-trial-out robustness (TEST)\n",
          "> Cache-only: no new API calls, no pipeline re-run, no test-set fitting. "
          "Drop each of the 18 held-out trials in turn; recompute ΔF1 / Δrecall (A−B) over the "
          "remaining 17 trials.\n\n",
          f"**Test set:** {len(ids)} pairs over {out['n_trials']} trials.\n\n",
          "## Stability of each effect across trials\n\n"]
    rows = []
    for name, r in comps.items():
        ss = r["folds_same_sign_as_full_dF1"]
        rows.append([name, f"{r['full']['dF1']:+.4f}", f"{r['full']['dRecall']:+.4f}",
                     f"[{r['loto_dF1']['min']:+.3f}, {r['loto_dF1']['max']:+.3f}]",
                     f"{ss}/{r['n_folds']}"])
    md.append(mdtab(["Comparison (A vs B)", "Full ΔF1", "Full Δrecall",
                     "LOTO ΔF1 range", "Folds keeping sign"], rows))
    md.append("\nIf every leave-one-trial-out fold keeps the sign of the full-set ΔF1, the effect "
              "is a property of the 18-trial population, not an artifact of one or two trials. "
              "Source: `experiment11_robustness_testing_data.json`.\n")
    with open(os.path.join(RES, "experiment11_robustness_testing_data.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"TEST {len(ids)} pairs | {out['n_trials']} trials | leave-one-trial-out\n")
    for name, r in comps.items():
        print(f"  {name:26s} full_dF1={r['full']['dF1']:+.4f} full_drec={r['full']['dRecall']:+.4f} "
              f"LOTO_dF1[{r['loto_dF1']['min']:+.3f},{r['loto_dF1']['max']:+.3f}] "
              f"sign-stable {r['folds_same_sign_as_full_dF1']}/{r['n_folds']}")
    print("\nWrote experiment11_robustness_testing_data.{json,md}")


if __name__ == "__main__":
    main()
