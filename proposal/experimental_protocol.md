# Experimental Protocol — PRISM-V
### Complete specification. Every entry is a plan. `NOT YET RUN` — no result is reported here.

---

## 0. Design principles

1. **Temporal integrity above everything.** All splits are chronological. Normalisation statistics,
   PCA bases, reference windows, thresholds and hyperparameters are fitted on data strictly earlier
   than the evaluation window. `tests/test_temporal_leakage.py` is a blocking test.
2. **Every claim has a falsifier.** Each hypothesis in §8 names the result that would refute it.
3. **Naive baselines are first-class.** The persistence/error-extrapolation family is run before
   anything else (Gate 1).
4. **Negative controls are mandatory**, not optional robustness checks.
5. **No number appears in the manuscript that was not read from a result file.**

---

## 1. Datasets

Selection criteria applied to every candidate: public availability, permissive licence, genuine
temporal structure, sufficient length to contain multiple threshold crossings, computational
feasibility on a single GPU, and prior use in the literature.

### 1.1 Tier A — natural temporal distribution shift (mandatory; at least two used)

| Dataset (Wild-Time suite) | Modality / task | Licence (as stated in the benchmark repository) |
|---|---|---|
| Yearbook | images, classification | MIT |
| FMoW (Wild-Time variant) | satellite imagery, classification | FMoW Challenge Public License |
| MIMIC-IV (mortality / readmission) | clinical tabular, binary prediction | PhysioNet Credentialed Health Data License 1.5.0 — **requires credentialing; access must be confirmed before this dataset is scheduled** |
| Drug-BA | molecular, binding-affinity prediction | MIT |
| arXiv | text, classification | CC0 |
| Huffpost | text, classification | CC0 |

Wild-Time (Yao et al., NeurIPS 2022 Datasets & Benchmarks, arXiv:2211.14238, `V071`) is the primary
Tier-A source: it is purpose-built for *temporal* natural distribution shift, supplies timestamp
metadata, and defines two protocols — **Eval-Fix** (a single ID/OOD split at a chosen timestep, the
default) and **Eval-Stream** (evaluation across successive future timesteps, selected with
`--eval_next_timesteps`). Eval-Stream is the protocol closest to our setting and is the one PRISM-V
adopts and extends with the risk-threshold event.

| Additional Tier-A source | Domain | Why included |
|---|---|---|
| WILDS — FMoW, iWildCam (Koh et al., ICML 2021, `V070`) | satellite imagery; camera traps | large realistic shifts; FMoW carries an explicit temporal axis; widely used, so results are comparable |

**Rule:** at least one genuine natural-shift benchmark must appear in every headline experiment.
Synthetic streams alone are not admissible evidence for H1–H3.

### 1.2 Tier B — long tabular / streaming series with real temporal dependence

Electricity (ELEC2), Airlines, Covertype (as a permuted stream), and a public credit-risk panel.
**Caveat carried into the paper:** the streaming-benchmark literature has documented that ELEC2 and
Airlines have strong temporal autocorrelation which makes naive persistence a deceptively strong
classifier baseline (`V065`, and the stream-benchmarking critiques retained at stage 2). They are used
for *comparability with prior work*, never as the primary evidence.

### 1.3 Tier C — regression/forecasting streams

Electricity Load Diagrams (UCI), a GEFCom load track, and Jena Climate. Used to show the estimand is
not classification-specific: `ℓ` becomes a bounded normalised error and `ε` a service-level threshold.

### 1.4 Tier D — fully controlled synthetic streams

SEA, Hyperplane, AGRAWAL, RandomRBF-style generators plus our own generator (§2). Used **only** for
controlled ablation, forecastability-class construction, and the negative control — never as the
headline evidence.

**Dataset table to be completed in the manuscript** with: name, domain, task, size, time structure,
shift type, reason for inclusion, public source, licence/access. Any dataset whose licence or access
cannot be confirmed is dropped, not substituted with an assumption.

---

## 2. Controlled shift generator (`src/shifts/`)

Supported mechanisms: covariate shift; prior/label shift; conditional (`P(Y|X)`) shift; abrupt;
gradual; incremental; recurring; seasonal; compound; feature dropout; sensor corruption; noise
escalation; unseen operating regime.

Every generated stream is serialised with:

```
shift_type, start_time, end_time, severity, affected_features,
ground_truth_change_point, forecastability_type ∈ {F, PF, U}
```

**Construction of the U (unforecastable) class.** `P(X)` is held fixed; at `τ ~ Uniform(τ_min, τ_max)`
the conditional `P(Y|X)` is replaced (e.g. by permuting the label map on a random subspace); `τ` is
independent of every observable; labels arrive with delay `δ ≥ 1`. No context feature encodes `τ`.
A generator-side unit test asserts `I(S_{1:t}; 1[T_t ≤ h]) ≈ 0` empirically via a permutation test.

---

## 3. Base predictors

XGBoost, LightGBM, MLP, LSTM, and a Transformer encoder — each applied only where it is appropriate for
the modality; a ResNet/ViT backbone for the image streams. Predictors are trained once on the
pre-deployment window and then frozen for the monitoring experiments (adaptation experiments unfreeze
them under the policy being tested). No predictor is forced onto a dataset where it is unsuitable.

---

## 4. Validity-trajectory dataset

For every `(dataset, stream, predictor, seed, window)` the pipeline writes one record:

```
dataset_id, stream_id, model_id, seed, timestamp,
prediction, prediction_probability, target_if_available, loss_if_available,
prediction_features[], representation_features[], uncertainty_features[],
distribution_features[], context_features[], adaptation_features[],
risk_t, risk_t_plus_1, risk_t_plus_5, risk_t_plus_10, risk_t_plus_20, risk_t_plus_50,
risk_threshold, true_validity_horizon, censored,
shift_type, shift_severity, shift_start, shift_end, forecastability_type
```

**Invariant (tested):** all `risk_t_plus_*`, `true_validity_horizon`, `censored`, `shift_*` and
`forecastability_type` fields are **targets or metadata only** and are excluded from the feature
tensor by construction, with an automated schema assertion.

---

## 5. Splits

Origin-level chronological split per stream: train `[0, τ₁)`, calibration `[τ₁, τ₂)`, test `[τ₂, end]`,
with a **gap of `H` windows** between splits so that no training origin's horizon overlaps the test
period. Hyperparameters are chosen on the calibration split only. For cross-stream experiments the
split is at the stream level as well as the time level.

---

## 6. The fourteen experiments

| # | Experiment | Question | Primary metrics | Gate |
|---|---|---|---|---|
| E1 | Risk-path forecastability | can `R̂_{t+h}` beat naive extrapolation | horizon-wise MAE, skill vs persistence | **G1** |
| E2 | Time-to-violation estimation | can `T_t` be estimated | C-index, IPCW-IBS, D-calibration, VHMAE (+censoring rate) | **G2** |
| E3 | **Prognosis conditioned on current validity** | can violation be predicted while `R_t ≤ ε` and `R_{t+1} ≤ ε` | conditioned precision/recall, PVLT | **G3** |
| E4 | Early warning | how much lead time | median PVLT + IQR, precision–lead-time curve | G3 |
| E5 | False-alarm control | is the warning usable | false alarms per 100 valid windows; ROC over operating points | G3 |
| E6 | Drift vs validity dissociation | constructed cases: big shift/small harm, small shift/big harm, high uncertainty/stable, low uncertainty/impending failure | per-case detection vs prognosis outcome | — |
| E7 | Natural distribution shift | does it hold outside synthetic data | all of E1–E5 on Tier A | G1–G3 |
| E8 | Leave-one-shift-type-out | generalisation to unseen shift mechanisms | degradation vs in-distribution training | — |
| E9 | Cross-model transfer | unseen predictor architecture | transfer C-index vs in-domain and vs Kaplan–Meier | **G5** |
| E10 | Cross-domain transfer | unseen application domain | as E9, leave-one-domain-out | **G5** |
| E11 | **Negative control (U streams)** | does it fail as it must | equivalence test vs Kaplan–Meier on C-index and IBS | **G0** |
| E12 | Anticipatory adaptation | does acting help | cumulative exposure, interventions, compute | **G4** |
| E13 | Anticipatory vs reactive **and proactive** retraining | is the forecast a better trigger | cost–exposure Pareto vs all policies incl. proactive | **G4** |
| E14 | Full ablation | which components matter | Δ in C-index / PVLT per removed block | — |

**E11 is reported in the main paper, not the appendix.** A framework that cannot fail on
unforecastable data is not measuring what it claims to measure.

---

## 7. Statistical validation

5 independent seeds per configuration (documented deviation if compute forbids). Report mean, standard
deviation, 95% CI (bootstrap over origins **and** over seeds, blocked by stream to respect temporal
dependence). Paired comparisons across matched streams via Wilcoxon signed-rank; Holm–Bonferroni
correction within each experiment family; effect sizes (Cliff's delta / paired Cohen's d) always
reported alongside p-values. **Never select a seed.** All seeds are reported; the aggregation rule is
fixed in advance.

---

## 8. Hypotheses and falsification criteria

| ID | Hypothesis | Falsified if |
|---|---|---|
| **H1** | Under forecastable evolution, deployment-state history contains information about future risk beyond the recent error series | No naive baseline is beaten at any horizon on F/PF streams (Gate 1 fails) |
| **H2** | Time-to-risk-violation can be estimated more accurately by a learned hazard model over deployment state than by uncertainty, drift magnitude, or a marginal survival curve | PRISM-V does not beat Kaplan–Meier and Cox PH on C-index, or is worse calibrated |
| **H3** | Violations can be flagged with positive lead time while the model is still valid | Median PVLT ≤ 0 on the conditioned subset, or precision below the pre-set floor |
| **H4** | Validity dynamics learned on some architectures/domains transfer to unseen ones | Transfer performance is not above the marginal baseline |
| **H5** | Where H1–H3 hold, acting on the forecast reduces cumulative validity violation at no greater cost than existing reactive **and proactive** policies | PRISM-V's policy is Pareto-dominated by any non-oracle baseline |
| **H6** | Under constructed unforecastable evolution, no method — including ours — achieves skill | PRISM-V beats Kaplan–Meier on U streams ⇒ **leakage presumed**, halt and audit |

---

## 9. Reproducibility record

Every run writes: git commit hash, dataset id and checksum, predictor id, seed, full resolved config,
hyperparameters, hardware description, start/end timestamps, all metrics, captured warnings, captured
exceptions, and the output file path. Tables and figures are generated **only** by
`scripts/14_generate_tables_figures.py` reading `results/`. Manual entry of numbers into the manuscript
is prohibited and is checked by a script that scans the manuscript for numerals without a matching
result-file provenance tag.

---

## 10. Computational requirements (ESTIMATES — not measurements)

These are **planning estimates**, explicitly marked as such, and must be replaced by measured values
once experiments run.

| Component | Estimate |
|---|---|
| Base predictor training (tabular/streaming, all seeds) | ~10–30 CPU-hours — ESTIMATE |
| Base predictor training (image streams, ResNet-scale, all seeds) | ~30–80 GPU-hours (single modern GPU) — ESTIMATE |
| Trajectory generation + feature extraction | ~20–50 CPU-hours; dominated by embedding extraction — ESTIMATE |
| Validity-model training (all encoders × ablations × seeds) | ~20–60 GPU-hours — ESTIMATE |
| Baselines (drift detectors, conformal, naive) | ~10–20 CPU-hours — ESTIMATE |
| Adaptation experiments (repeated retraining) | ~40–120 GPU-hours — the dominant cost — ESTIMATE |
| Storage | ~200–600 GB for trajectories and embeddings — ESTIMATE |
| RAM | 32–64 GB recommended — ESTIMATE |

`ESTIMATE — NOT A MEASUREMENT` is printed next to every figure in this table in the manuscript.
