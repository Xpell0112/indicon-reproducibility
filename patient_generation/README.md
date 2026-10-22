# Patient generation (transparency)

INDICON 2026 (Paper ID 5232).

## File
| File | Source |
|---|---|
| `generate_patients.py` | `dataset_files/generate_patients.py` |

## How it works
- **Template-based narrative generation (NO LLM).** Profiles are rendered by `render_narrative`
  using multiple templates with shuffled optional clauses (a seeded `random.Random`); no language
  model is called at any point.
- **Attributes sampled from a fixed NSCLC clinical schema WITHOUT any trial criteria.** The
  narrative step never sees a trial's criteria, so a profile is not written to pass or fail any
  specific trial.
- **Eligibility follows from the patient–trial pairing.** Each patient is matched against several
  trials; the gold label is a consequence of the (patient, trial) pair, determined by blind
  annotation — not baked in at generation.
- **The `provisional_label` is NOT ground truth** and is **excluded from this release** (it is a
  generation-time, balance-monitoring value written to a separate key file that annotators never
  see).
- **Generation seed:** the script default is `20260718` (`--seed`, `random.Random(args.seed)`).

This module is included for transparency; the released dataset in `../data/` already contains the
frozen synthetic profiles.
