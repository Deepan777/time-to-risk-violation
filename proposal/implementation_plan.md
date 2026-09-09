# Implementation Plan — PRISM-V
### Component-level specification. Detailed enough that no methodological decision has to be invented.

Python ≥ 3.11. Core stack: `torch`, `scikit-learn`, `xgboost`, `lightgbm`, `river`, `numpy`, `pandas`,
`scipy`, `statsmodels`, `matplotlib`, `optuna`, `pyarrow`, `einops`, `tqdm`, `lifelines`,
`scikit-survival`, `python-docx`. Configuration via YAML + a typed config loader; every run resolves its
config to a single serialised dict written next to the results.

Conventions used in every section: **Purpose · Inputs · Outputs · File · Main API · Algorithm ·
Data structure · Dependencies · Config · Unit tests · Artifacts · Failure cases.**

---

## Phase I — Literature (complete)
Artifacts already produced in `literature/`. `scripts/01_literature_search.py`,
`02_verify_metadata.py`, `03_build_evidence_matrix.py` exist to regenerate the CSVs and to re-run
DOI verification (Crossref for registered DOIs, DataCite for `10.48550/arXiv.*`) before submission.

---

## 1. `src/datasets/`

**Purpose.** Turn every raw source into a canonical, time-ordered stream.
**Inputs.** raw files under `data/raw/<dataset_id>/`; a YAML descriptor in `configs/datasets/`.
**Outputs.** `data/processed/<dataset_id>/stream.parquet` with a strictly increasing `timestamp`,
plus `data/metadata/<dataset_id>.json` (rows, time span, licence, task type, checksum).
**File.** `src/datasets/loaders.py`, `src/datasets/registry.py`, `src/datasets/windowing.py`.
**Main API.** `load_stream(dataset_id: str, cfg: DatasetCfg) -> Stream`;
`Stream.windows(size:int, stride:int) -> Iterator[Window]`.
**Algorithm.** load → validate schema → sort by time → assert monotone time → split into
pre-deployment / deployment segments → emit fixed-size windows.
**Data structure.** `Stream(df: pd.DataFrame, time_col: str, feature_cols: list[str], target_col: str,
task: Literal["binary","multiclass","regression"], meta: dict)`.
**Config.** `window_size`, `stride`, `pre_deployment_frac`, `label_delay_windows δ`, `time_col`.
**Unit tests.** monotone timestamps; no window spans the pre-deployment boundary; window count formula;
checksum stability.
**Artifacts.** parquet + metadata json.
**Failure cases.** unsorted or duplicate timestamps → raise; dataset shorter than
`window_size * (H + 2)` → refuse with an explicit message (the stream cannot contain a horizon);
missing licence field → refuse.

## 2. `src/predictors/`

**Purpose.** Train and serve the base predictor `f_θ`.
**Inputs.** pre-deployment segment; `configs/predictors/<id>.yaml`.
**Outputs.** serialised model, plus a `PredictorHandle` exposing `predict_proba`, `embed`, `params`.
**File.** `src/predictors/{tabular.py, mlp.py, sequence.py, vision.py, base.py}`.
**Main API.** `train_predictor(stream, cfg) -> PredictorHandle`;
`PredictorHandle.embed(X) -> np.ndarray` (penultimate representation; for tree models, leaf-index
one-hot projected by a fixed random projection — documented as such, not called an embedding of the
same kind).
**Algorithm.** fit on pre-deployment only; freeze; expose deterministic inference under a fixed seed.
**Config.** model family, hyperparameters, seed, calibration (Platt/isotonic on a pre-deployment split).
**Unit tests.** determinism under fixed seed; `embed` dimensionality stable; no access to deployment
segment (asserted by passing a poisoned deployment frame that must never be read).
**Failure cases.** model family not applicable to modality → skip with a logged reason, never silently
substitute.

## 3. `src/shifts/`

**Purpose.** Generate controlled streams with ground-truth shift and forecastability metadata.
**Inputs.** a base stream; `configs/shifts/<id>.yaml`.
**Outputs.** `data/streams/<stream_id>/{stream.parquet, shift_manifest.json}`.
**File.** `src/shifts/{generators.py, schedules.py, manifest.py}`.
**Main API.** `generate_stream(base, shift_spec, rng) -> (Stream, ShiftManifest)`.
**Algorithm.** each generator is a callable `(X, y, t, params) -> (X', y')`; a schedule composes
generators over time (abrupt / gradual / incremental / recurring / seasonal / compound).
**Data structure.** `ShiftManifest(shift_type, start_time, end_time, severity, affected_features,
ground_truth_change_point, forecastability_type)`.
**Config.** generator list, severity grid, change-point distribution, `forecastability_type`.
**Unit tests.** covariate-only generators leave `P(Y|X)` invariant (checked by a held-out oracle model);
conditional-shift generators leave the marginal `P(X)` invariant (two-sample test must not reject);
**U-class test**: permutation test shows `I(S_{1:t}; 1[T_t ≤ h]) ≈ 0`; manifest round-trips.
**Failure cases.** severity outside `[0,1]`; change point outside stream bounds; a "U" stream whose
`P(X)` test rejects → refuse to emit.

## 4. `src/monitoring/`

**Purpose.** Compute the deployment state `s_t`.
**Inputs.** windowed stream, `PredictorHandle`, frozen reference window.
**Outputs.** `results/raw/<run>/state.parquet` — one row per window, columns namespaced by block.
**File.** `src/monitoring/{prediction_stats.py, representation_stats.py, uncertainty.py,
discrepancy.py, feedback.py, context.py, adaptation_state.py, assembler.py}`.
**Main API.** `build_state(window, handle, ref: ReferenceStats, hist: History) -> dict[str, float]`.
**Algorithm.** per §4 of `proposed_methodology.md`. `ReferenceStats` (means, covariances, PCA basis,
bin edges, k-NN index) is fitted **once** on the pre-deployment reference window and is immutable.
**Config.** which blocks are enabled; `k` for k-NN; MMD kernel and bandwidth (median heuristic computed
on the reference window only); number of PCA components; trailing-window length for volatility.
**Unit tests.** `build_state` is a pure function of `(window, ref, hist)`; feature names are stable and
sorted; `s^fb` is all-NaN (with mask 0) for windows younger than `δ`; no state column correlates
perfectly with any future-target column (automated leakage screen).
**Failure cases.** empty window; reference statistics not fitted → raise; NaNs propagated rather than
imputed silently (imputation is explicit and logged).

## 5. `src/validity/`

**Purpose.** Build the supervised, censored dataset and train the validity model.
**Inputs.** `state.parquet` + realised risk series + `ε`.
**Outputs.** `data/processed/<run>/validity_dataset.parquet`; trained checkpoints.
**File.** `src/validity/{targets.py, dataset.py, encoders.py, heads.py, model.py, train.py, calibrate.py}`.
**Main API.**
`build_targets(risk: np.ndarray, eps: float, H: int) -> TargetFrame`
`PrismV(cfg).fit(train_ds, cal_ds)` · `.predict_survival(S_1t) -> np.ndarray[H]`
· `.predict_risk_path(S_1t) -> np.ndarray[H]` · `.horizon(S_1t) -> HorizonEstimate`
**Algorithm (targets).** for each origin `t` with `R_t ≤ ε`: scan `h = 1..min(H, remaining)`; first `h`
with `R_{t+h} > ε` gives `Ỹ_t = h, D_t = 1`; otherwise `Ỹ_t = min(H, remaining), D_t = 0`.
**Algorithm (model).** encoder over `s_{t-L+1:t}` → `z_t`; two shared heads with horizon embeddings;
loss `L_surv + α L_risk + β Ω_mono`; early stopping on calibration IPCW-IBS.
**Data structure.** `TargetFrame(origin_t, y_tilde, event, risk_path[H], censored, forecastability_type)`.
**Config.** `H`, `L`, encoder type/width/depth, `α`, `β`, learning rate, batch size, epochs, seed,
`ε` (from the dataset descriptor, not tuned).
**Unit tests.** censoring correctness on hand-built risk series (fixture with known crossings);
`S_φ(h)` monotone non-increasing; likelihood of a synthetic constant-hazard stream recovers the hazard
within tolerance; **no future column present in the feature tensor** (schema assertion);
origins with `R_t > ε` are excluded.
**Failure cases.** fewer than a configured minimum number of uncensored events → refuse to fit and log
"insufficient events at ε = …" rather than producing a degenerate model.

## 6. `src/survival/`

**Purpose.** Classical survival references and survival utilities.
**Main API.** `KaplanMeierBaseline`, `CoxPHBaseline` (via `lifelines` / `scikit-survival`),
`discrete_hazard_to_survival`, `ipcw_weights`, `d_calibration`.
**Unit tests.** IPCW weights sum correctly; KM matches `lifelines` on a reference fixture;
`discrete_hazard_to_survival` inverse-consistent.

## 7. `src/baselines/`

**Purpose.** Every competitor of §6 of `proposed_methodology.md`.
**File.** `src/baselines/{naive_temporal.py, drift_detectors.py, uncertainty_baselines.py,
labelfree_estimators.py, conformal.py, retraining_policies.py}`.
**Main API.** every baseline implements the same protocol:
`class Baseline(Protocol): def warn(self, hist) -> bool; def risk_path(self, hist) -> np.ndarray | None;
def survival(self, hist) -> np.ndarray | None`.
**Notes.** drift detectors come from `river` where available (ADWIN, DDM, EDDM, Page–Hinkley, KSWIN) so
that no detector is re-implemented and mis-tuned; label-free estimators (ATC-style, DoC-style,
confidence-based) are re-implemented from their papers with the reference cited in the docstring.
**Unit tests.** each baseline runs end-to-end on a 200-window fixture; ADWIN fires on a known abrupt
change; persistence reproduces the input series shifted by one.
**Failure cases.** a baseline that cannot produce an output for a given stream returns `None` and is
recorded as "not applicable", never as a loss.

## 8. `src/adaptation/`

**Purpose.** Actions and the anticipatory policy.
**Actions.** `A0` none · `A1` recalibrate (temperature/isotonic on recent matured labels) ·
`A2` retrain prediction head · `A3` partial fine-tune · `A4` replay-based update ·
`A5` full retrain · `A6` abstain/defer.
**Main API.** `Policy.decide(state, survival_curve, costs) -> Action`;
`apply_action(handle, action, buffer) -> PredictorHandle`.
**Algorithm.** solve (7) by enumerating the (small) action set, estimating
`P(T^{post,a}_t > H_min)` with a counterfactual head conditioned on the action embedding, and choosing
the cheapest feasible action; fall back to the penalised objective when infeasible.
**Config.** action costs (in training steps and wall-clock), `H_min`, `ρ`, `κ`, replay buffer size.
**Unit tests.** action set restricted per predictor family (no "fine-tune" for XGBoost);
cost accounting is monotone; policy is deterministic given the same inputs.
**Failure cases.** action inapplicable → excluded from the feasible set with a logged reason.

## 9. `src/metrics/` and `src/evaluation/`

**Main API.** `risk_path_metrics`, `survival_metrics` (C-index time-dependent, IPCW-IBS,
D-calibration), `horizon_metrics` (VHMAE on uncensored + IPCW variant + censoring rate),
`warning_metrics` (PVLT median/IQR, precision, recall, FAR per 100 valid windows),
`conditioned_warning_metrics` (the Gate-3 protocol), `adaptation_metrics`.
**Unit tests.** VHMAE refuses to run without a censoring rate; PVLT is computed only over true
positives; conditioned metrics raise if the conditioning mask is empty; every metric has a
hand-computed fixture.

## 10. `src/utils/`
Seeding, logging, config resolution, run manifests, git-commit capture, hardware capture, result IO.
**Unit test.** a run manifest contains every field required by §9 of `experimental_protocol.md`.

## 11. `scripts/` (execution order)

| Script | Produces |
|---|---|
| `01_literature_search.py` | `literature/search_log.csv`, `candidate_papers.csv` |
| `02_verify_metadata.py` | DOI/DataCite verification, updates `Verified` column |
| `03_build_evidence_matrix.py` | `literature/evidence_matrix.csv`, capability tallies |
| `04_prepare_datasets.py` | `data/processed/*`, `data/metadata/*` |
| `05_train_predictors.py` | frozen predictors + `results/raw/predictors.json` |
| `06_generate_streams.py` | `data/streams/*` + shift manifests |
| `07_generate_validity_dataset.py` | `state.parquet`, `validity_dataset.parquet` |
| `08_train_validity_model.py` | checkpoints, calibration objects |
| `09_run_baselines.py` | baseline result files |
| `10_run_generalization.py` | E8/E9/E10 results |
| `11_run_adaptation.py` | E12/E13 results |
| `12_run_ablations.py` | E14 results |
| `13_statistical_analysis.py` | `results/statistical/*` with CIs, tests, effect sizes |
| `14_generate_tables_figures.py` | `results/tables/*`, `results/figures/*` |

Each script is idempotent, takes `--config`, `--seed`, `--out`, and refuses to overwrite an existing
result directory without `--force`.

## 12. `tests/`

| Test file | Asserts |
|---|---|
| `test_temporal_leakage.py` | no future target/metadata column in features; normalisation fitted on past only; no random split of a temporal stream; test data never used for hyperparameter selection; adaptation never consults future labels; survival targets absent from the feature tensor |
| `test_shift_generation.py` | generator invariants (§3), manifest round-trip, U-class independence |
| `test_survival_targets.py` | censoring construction on fixtures; monotone survival; hazard recovery |
| `test_metrics.py` | every metric against a hand-computed fixture; guards (censoring rate, empty masks) |

CI runs the full test suite plus a 5-minute synthetic end-to-end smoke run.

## 13. Failure-mode register (project level)

| Symptom | Likely cause | Required action |
|---|---|---|
| C-index ≫ 0.9 on the first run | leakage | run Gate 0; inspect feature/target schema |
| Skill on U streams | leakage or a precursor accidentally encoded | regenerate streams; re-run independence test |
| Naive persistence wins everywhere | risk series is smooth/autocorrelated; the state adds nothing | Gate 1 stop rule |
| Almost all origins censored | `ε` too lenient, or `H` too short | sweep `ε`; report censoring rate; do not silently retune |
| Survival curves non-monotone | hazard head instability | check `Ω_mono`, clamp hazards to `[0, 1−1e-6]` |
| Cross-domain transfer at chance | domain-specific dynamics | report the negative result; do not tune on the target domain |
