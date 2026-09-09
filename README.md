# Time-to-Risk-Violation — implementation

Reference implementation for **Time-to-Risk-Violation (TTRV)**: forecasting how long a frozen,
deployed predictor will keep meeting its risk requirement under temporal distribution shift.

This repository holds the code and the frozen result artefacts it produces. It does not hold the
manuscript.

## The problem

A predictor is frozen after commissioning and its environment moves. Monitoring tells you what its
risk is **now**. The operational question is different: given that the model still meets its risk
requirement today, how long will that remain true?

TTRV is that quantity, as a right-censored first-passage time:

```
T_t = min{ h >= 1 : R_{t+h} > eps },    defined only at origins where R_t <= eps
```

`R_t` is the realised risk of the frozen predictor over evaluation window `t`, and `eps` is the
operational threshold. Conditioning on validity at the origin is what makes the task prospective
rather than a restatement of drift detection.

## Layout

| Path | Contents |
|---|---|
| `src/` | The framework: monitoring-state construction (seven views), the temporal encoder, risk-path heads, the first-passage survival readout, survival metrics and the statistical protocol |
| `scripts/` | The experimental pipeline, numbered in run order |
| `configs/` | Dataset, predictor and experiment configuration, and the smoke/dev/full scales |
| `tests/` | Leakage and isolation tests, including the ones that make the pre-deployment boundary structural rather than conventional |
| `results/` | Frozen result artefacts from the full-scale runs, to compare your own against |
| `data/metadata/` | Per-dataset provenance: archive version, licence and SHA-256 for every file used |
| `proposal/` | The pre-registration, its dated addendum, the experimental protocol, the go/no-go gates and the implementation plan |
| `REPRODUCE.md` | Stage by stage, from a clean checkout to the result files |

## What is deliberately not here

* **The datasets.** All eleven are third-party public archives. None was collected by us and none is
  redistributed. `data/metadata/` records the exact version, licence and SHA-256 of each file so the
  same inputs can be obtained from the original providers; two of them (UCI gas-sensor drift and UCI
  Air Quality) exclude commercial use. `REPRODUCE.md` section 2 fetches them.
* **Intermediate caches and raw per-run tensors.** Large and regenerable; the scripts rebuild them.
* **The manuscript**, its figures and tables, and the scripts that produce and audit it.

## Running it

```bash
conda env create -f environment.yml && conda activate prismv
pytest tests/ -q
```

Then follow `REPRODUCE.md`. Use the project environment throughout: several stream loaders need
`river`. The full experiment set runs on CPU; a GPU shortens the image-stream CNN and the GRU fits
but changes no verdict.

## How the protocol constrains the code

Worth knowing before reusing any of it.

* **The base predictor is frozen.** The predictor-isolation test poisons the deployment segment and
  requires the fitted predictor to be bit-identical, so the pre-deployment boundary is structural
  rather than conventional.
* **No state feature may see the future.** The temporal-leakage suite recomputes an early window
  inside a truncated stream and requires it to equal its value in the full stream.
* **The censoring-aware evaluation needs a non-degenerate sample:** at least 100 valid origins and a
  censoring rate strictly inside (0, 0.6). Streams outside that are excluded from the comparison;
  the TTRV estimand itself remains defined on them.
* **`insects` decides nothing.** It generated the drift-mechanism hypothesis, so it is reported as
  exploratory. That was fixed in the pre-registration, not afterwards.
* **The deployment state is task-specific:** 87 dimensions on classification streams, 79 on
  regression streams, because the prediction and uncertainty views branch on what the predictor
  emits. `scripts/39_state_dimension_check.py` recounts both from the cache and fails on a mismatch.

## Citation

A manuscript describing this work is under review at *Machine Learning with Applications*
(Elsevier). Citation details will be added on acceptance.
