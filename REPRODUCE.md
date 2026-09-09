# Reproducing this project

Every number in the manuscript comes from a file in `results/`, written by a script in `scripts/`,
with a run manifest recording the seed, the scale, the dataset, the predictor and the code state.
Nothing is typed by hand. This document is the path from a clean checkout to those files.

If a command below produces a different number from the one in the paper, that is a finding and we
would like to hear about it. The most likely causes, in order: a different `river` version changing
the ELEC2 / Insects row order, a UCI archive changed upstream (check the SHA-256s, below), or a
different PyTorch build changing floating-point reduction order on the GPU.

---

## 1. Environment

```bash
conda env create -f environment.yml && conda activate prismv
```

Python 3.11. `requirements.txt` pins major versions only.

**On version records, precisely.** Manifests written from this commit onward record the interpreter
path and the resolved versions of numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, river,
lifelines and matplotlib, under `hardware.packages`. **Manifests written earlier do not** — they
carry the Python and torch versions and the hardware, and nothing more. So for the gate results
produced before this commit, the package set is not recoverable from the manifest, and we say so
rather than implying a completeness the files do not have.

One consequence worth naming: this project was run from two interpreters at different times, a
system Python and the project `.venv`, because only the latter has `river`. Where the two overlap
they agree — `insects` reproduces to four decimals across both — but use `.venv` for everything, or
the `river`-backed loaders will fail at import.

The full experiment set runs on CPU. A GPU shortens the Yearbook CNN and the GRU fits but is not
required and does not change any verdict.

## 2. Data

Nothing is committed. Fetch it:

```bash
python scripts/22_fetch_mechanism_datasets.py
```

That writes `results/processed/mechanism_downloads.json` with a SHA-256 for each archive. Compare
against the values already in that file in this repository. **A mismatch means the archive changed
upstream and the results in the manuscript no longer describe the file you have.**

`elec2` and `insects` are fetched by `river` on first use. Yearbook and HuffPost come from the
Wild-Time distribution and land in `data/raw/wildtime/`.

Licences differ and some are restrictive. The UCI gas-sensor archives and Air Quality are
**research use only, commercial use excluded by the provider**. Every licence is recorded per
dataset in `configs/drift_mechanism.yaml` and in each stream's metadata.

## 3. Build the monitored streams

State construction is the expensive step and is cached on disk, keyed by everything that could
change it (dataset, seed, window size, label delay, epsilon rule, pre-deployment fraction).

```bash
python scripts/_build_streams.py                 # elec2, insects, yearbook
python scripts/_build_one.py huffpost 0          # one (dataset, seed) at a time
python scripts/_build_one.py gas_temp_mod 0      # ... repeat for seeds 0-4
python scripts/_build_one.py cmapss 0
python scripts/_build_one.py metropt3 0
python scripts/_build_one.py news_popularity 0
python scripts/_build_one.py gas_drift 0
python scripts/_build_one.py air_quality 0
```

`hydraulic` is expected to **fail here**, refused by the windowing rule with a message explaining
why. That refusal is a recorded outcome, not a bug — see `configs/drift_mechanism.yaml`, which
predicted it before the archive was downloaded.

Rough costs on the reference machine: `gas_temp_mod` and `metropt3` about 80 s per seed,
`cmapss` about 12 s, `yearbook` about 2 min, the rest under a minute.

## 4. The gates, in the order they were decided

```bash
python scripts/09_run_baselines.py  --scale full   # Gate 1  -> results/e1_gate1/
python scripts/16_run_residual.py   --scale full   # E16, the anchored follow-up
python scripts/19_gate0_negative_control.py        # Gate 0  -> results/gate0/
python scripts/21_gate2_survival.py --scale full   # Gate 2  -> results/e2_gate2/
python scripts/20_transfer.py       --scale full   # Gate 5  -> results/e9e10_transfer/
python scripts/18_scope_conditions.py              # scope conditions
```

**All four of these gates FAIL, and they are supposed to.** The verdict files say so
(`GATE1_VERDICT.json`, `GATE0_VERDICT.json`, `GATE2_VERDICT.json`, and the transfer summary). If
one of them passes on your machine, something has changed and we would want to know.

Only `--scale full` numbers may be quoted. `smoke` and `dev` exist for wiring checks and their
output must never reach a manuscript; the scale is stamped into every manifest so this is checkable.

## 5. The drift-mechanism study (PR-H1, PR-H2)

Read the pre-registration first — it is the point of this part:

* `proposal/PREREGISTRATION_drift_mechanism.md` (commit `32af9fd`)
* `configs/drift_mechanism.yaml` (commit `a1af1bd`) — classes fixed from provider documentation
* `proposal/PREREGISTRATION_drift_mechanism_ADDENDUM_1.md` (commit `41cf66f`) — the one amendment

The commit order matters and is the evidence: each of those was committed **before** the data it
governs was downloaded or run. `git log --format='%h %ad %s' --date=iso` shows it.

```bash
python scripts/25_usability_screen.py                   # which streams can support the estimand
```

```bash
python scripts/24_age_hazard_diagnostic.py              # why an age-blind marginal is a bad floor
```

```bash
python scripts/23_mechanism_boundary.py --scale full    # -> results/mechanism/PRH1_VERDICT.json
```

Run them in that order. The usability screen decides which datasets may enter the confirmatory
test, and it reads only origin counts and censoring rates — never any method's accuracy — which is
what makes it safe to run before fitting anything. `hydraulic` is expected to appear in its output
as refused before monitoring.

**The verdict is `UNDECIDED`, and that is the correct output.** Only two physical-drift datasets met
the usability threshold the pre-registration set, against the three it required, so the rule returns
undecided regardless of the numbers. If your run reports `SUPPORTED` or `REFUTED`, the dataset
roster differs from ours and the comparison is not the one in the paper.

## 6. The manuscript

Not in this repository. The manuscript, its figures and tables, and the scripts that build and audit
it are kept separately; this repository is the implementation and the result artefacts it produces.

Every number the paper reports is read from a file under `results/`, so the stages above are what a
reader needs to regenerate them.

## 7. Tests

```bash
pytest tests/ -q
```

The ones that matter most for trusting the results:

* the predictor-isolation test — poisons the deployment segment and checks the fitted base
  predictor is bit-identical, so the pre-deployment/deployment boundary is structural rather than
  conventional.
* the temporal-leakage suite — recomputes an early window inside a truncated stream and requires it
  to equal the value it had in the full stream, which any state feature that peeks would fail.

## 8. What you should not expect to reproduce

`insects` is reported throughout but **decides nothing** in the drift-mechanism study: it generated
the hypothesis, so using it as evidence for that hypothesis would be circular. It sits in a
secondary tier along with every dataset that failed the usability screen. This is stated in the
pre-registration, not decided afterwards.
