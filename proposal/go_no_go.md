# Go / No-Go Gates
### Pre-committed stop rules. Written before any experiment is run. `NOT YET RUN`.

These gates exist so that the project cannot quietly degrade into a paper that claims more than the
evidence supports. Each gate names (a) the question, (b) the decisive comparison, (c) the pre-committed
decision rule, and (d) what is deleted from the paper if it fails.

Statistical convention for every gate: 5 seeds, paired comparison across matched streams,
Wilcoxon signed-rank or paired bootstrap, Holm correction within a gate, effect size reported
(Cliff's delta or paired Cohen's d). "Beats" means the corrected paired test rejects at α = 0.05
**and** the effect size exceeds a small-effect threshold pre-set at |δ| ≥ 0.147.

---

## Gate 1 — Is future risk forecastable at all beyond naive extrapolation?
**Question.** Does `R̂_{t+h}` from the deployment state beat extrapolation of the recent error history?

**Comparison.** PRISM-V risk head vs {persistence, moving average, exponential smoothing,
Theil–Sen trend, AR(p)/ARIMA, LSTM-on-risk-series-only} on horizon-wise MAE, on **forecastable (F)
and partially forecastable (PF) streams only**.

**Rule.** PASS iff PRISM-V beats *every* naive baseline at `h ∈ {1, 5, 10}` on at least
two of the three dataset families, with the effect-size criterion met.

**If it fails → `STOP`.** Do not proceed to survival modelling. Return to `NOVELTY_DECISION.md`.
Report the negative result: "deployment-state features do not add information beyond the error series"
is itself a publishable and useful finding, and would be written up as such rather than buried.

---

## Gate 2 — Does validity-horizon estimation beat conventional proxies?
**Question.** Is `S_φ(h)` better than treating drift magnitude, uncertainty, or current risk as a
proxy for remaining validity?

**Comparison.** PRISM-V vs {Kaplan–Meier marginal, Cox PH on `s_t`, ADWIN/DDM/KSWIN warning time,
MSP/entropy/ensemble-variance thresholds, ATC/DoC/CBPE-style current-risk extrapolated by persistence,
ACI coverage-violation rate} on time-dependent concordance, IPCW integrated Brier score, and D-calibration.

**Rule.** PASS iff PRISM-V beats the Kaplan–Meier marginal **and** at least one strong parametric
competitor (Cox PH) on concordance *and* is not worse on calibration.

**If it fails → reformulate.** Drop the survival framing; retain only the multi-horizon risk-path
contribution and re-scope the paper to a smaller claim.

---

## Gate 3 — Can violation be predicted *before* it is observable?
**Question.** Conditioned on `R_t ≤ ε` and `R_{t+1} ≤ ε` — the model is valid now and still valid next
window — can PRISM-V identify origins that will violate within `H`?

**Comparison.** Warning precision/recall and PVLT on this conditioned subset, against all
detection-based baselines (which by construction have near-zero lead time here).

**Rule.** PASS iff median `PVLT > 0` with warning precision above the pre-set operating floor
(precision ≥ 0.6 at recall ≥ 0.5 on validation-chosen operating points), on natural-shift data,
with false-alarm rate reported per 100 valid windows.

**If it fails → remove all anticipatory claims.** The paper becomes a *label-free risk-trajectory
estimation* paper. The words "early warning", "anticipatory", and "before degradation" are struck.

---

## Gate 4 — Does acting on the forecast reduce cumulative validity violation?
**Question.** Does the chance-constrained policy (7) reduce cumulative risk exposure at equal or lower
intervention cost than existing policies?

**Comparison.** vs {never, always, periodic, drift-triggered, uncertainty-triggered, error-triggered,
cost-aware (CARA-style), **proactive baselines** (DDG-DA-style, proactive-adaptation-style), oracle}
on the cost–exposure Pareto plane.

**Rule.** PASS iff PRISM-V's policy is Pareto-dominant or Pareto-non-dominated against **all**
non-oracle baselines including the proactive ones, on at least two dataset families.

**If it fails → remove the adaptation contribution entirely.** The paper keeps the estimand, the
protocol and the theory, and states plainly that acting on the signal did not pay off in these settings.
Beating only *reactive* baselines does **not** pass this gate — the novelty audit showed proactive
adaptation is prior art.

---

## Gate 5 — Do validity dynamics transfer across predictors and domains?
**Question.** Does a validity model trained on trajectories from architectures `{A}` / domains `{D}`
work on an unseen architecture / domain?

**Comparison.** leave-one-architecture-out and leave-one-domain-out against (i) an in-domain-trained
PRISM-V (upper reference) and (ii) the marginal Kaplan–Meier baseline (lower reference).

**Rule.** PASS iff transfer performance is strictly above the marginal baseline. It does **not** need to
match in-domain training.

**If it fails → do not claim universal validity dynamics.** Report the negative result explicitly and
scope every claim to the trained architecture/domain. A clean negative here is a genuine contribution
given that no prior work has tested it.

---

## Gate 0 (runs before all others) — Leakage audit
Not a novelty gate but a hard precondition. `tests/test_temporal_leakage.py` must pass, and on
**unforecastable (U)** streams PRISM-V must be statistically indistinguishable from Kaplan–Meier.

**If PRISM-V beats the marginal baseline on U streams → HALT.** Treat as leakage until proven otherwise.
Suspiciously strong results anywhere trigger this audit before anything is reported.

---

## Decision log

| Gate | Status | Date | Evidence file | Decision |
|---|---|---|---|---|
| Gate 0 | PASS (smoke scale) | 2026-08-26 | PROGRESS.md preliminary check | On 8 class-U streams PRISM-V is indistinguishable from Kaplan-Meier (C=0.549 / 0.478 vs 0.500). No leakage. |
| Gate 1 | **FAIL** | 2026-08-27 | `results/e1_gate1/GATE1_VERDICT.json` | **STOP.** 5 seeds, 2 families, real data. PRISM-V is worse than AR(p) / moving average / exponential smoothing on Tier-A (45% higher MAE at h=1); 0/5 seeds pass on either family. Harness validity confirmed on Tier-A (test/train risk ratio 1.30); Tier-B test underpowered (n=45) and reported as uninformative rather than used to soften the verdict. Negative result to be written up. |
| Gate 2 | **FAIL** | 2026-08-28 | `results/e2_gate2/GATE2_VERDICT.json` | Beats a constant Kaplan-Meier everywhere, but that floor is not honest (D28/D29). Against an age-stratified marginal it beats on insects only (1/3 datasets, needs >=2). |
| Gate 3 | NOT YET RUN | — | — | — |
| Gate 4 | NOT YET RUN | — | — | — |
| Gate 5 | **FAIL** | 2026-08-28 | `results/e9e10_transfer/GATE5_VERDICT.json` | Cross-domain transfer is below chance on all three domains (0.358-0.386). Validity dynamics do not transfer. |
