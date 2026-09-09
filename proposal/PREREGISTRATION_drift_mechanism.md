# Pre-registration — the drift-mechanism boundary hypothesis

**Written:** 2026-08-28
**Status at time of writing:** no new dataset has been downloaded, loaded, or run. The hypothesis
below is committed *before* any confirmatory evidence exists.
**Author's declaration:** this hypothesis was **generated post hoc** by observing that PRISM-V beat
every reference on `insects` and on nothing else. That makes `insects` a hypothesis-*generating*
observation and disqualifies it as confirmatory evidence. It is excluded from the confirmatory test
below and reported separately as the exploratory observation that motivated the study.

---

## 1. The hypothesis

**PR-H1 (boundary hypothesis).** Deployment-state monitoring carries information about
time-to-risk-violation **when the drift has a physical generating mechanism that accumulates in the
observable feature space**, and carries little or none when the drift arises from exogenous social,
editorial or stylistic change.

**Mechanism claimed.** A physical process — a sensor degrading, a component wearing, a temperature
ramp — leaves a monotone, observable trace in the inputs *before* it damages the prediction. The
monitoring panel is a set of gauges, and gauges register accumulating physical change. Social change
has no analogous accumulating substrate: the world simply becomes different, and there is nothing
for a gauge to register in advance.

**PR-H2 (baseline hypothesis).** Deployment age alone is a strong predictor of time-to-risk-violation
across drift regimes, and is not beaten by deployment-state monitoring on socially-driven streams.

---

## 2. Classification rule, fixed in advance

Every dataset is assigned to a class **before** it is run, using only its documentation — never its
results. The assignment is recorded in `configs/drift_mechanism.yaml` and committed before the
experiment executes.

**PHYSICAL** — the dataset's own documentation attributes the drift to a physical or instrumental
process: sensor ageing or fouling, component wear, calibration loss, or a controlled physical
covariate (temperature, load, pressure). The mechanism is stated by the data provider, not inferred
by us.

**SOCIAL** — the drift arises from human behaviour, culture, editorial choice, fashion, or market
sentiment, with no accumulating physical substrate.

**AMBIGUOUS** — anything we cannot assign from documentation alone. Ambiguous datasets are excluded
from the confirmatory test rather than argued into a class.

Pre-assignment of the datasets already in hand:

| dataset | class | justification from documentation |
|---|---|---|
| `insects` | PHYSICAL | drift induced by a controlled temperature ramp changing wing-beat frequency |
| `yearbook` | SOCIAL | photographic convention, hairstyle and styling change across eight decades |
| `huffpost` | SOCIAL | editorial category prevalence and news vocabulary change |
| `elec2` | AMBIGUOUS | electricity market prices: physical demand cycles entangled with market behaviour. **Excluded from the confirmatory test.** |

---

## 3. The confirmatory test

**Confirmatory datasets must be new.** The test is run on datasets not previously loaded in this
project. `insects` is excluded (it generated the hypothesis); `yearbook` and `huffpost` may serve as
SOCIAL comparators because they have already produced their result under a different pre-registered
question and no reanalysis of them is involved.

**Target:** at least **three** new PHYSICAL datasets and, where available, one new SOCIAL dataset.

**Comparison, per dataset:** PRISM-V (first-passage variant, unchanged) against the
**age-stratified marginal** — the honest floor established by decision D29. The constant
Kaplan-Meier curve is reported for continuity but decides nothing.

**"Beats" is the project's existing definition, unchanged:** Holm-corrected paired Wilcoxon
rejecting at alpha = 0.05 **and** Cliff's delta >= 0.147, on all five seeds.

### Decision rule

> **PR-H1 is SUPPORTED** iff PRISM-V beats the age-stratified marginal on **at least two of three**
> new PHYSICAL datasets, **and** fails to beat it on the SOCIAL datasets.
>
> **PR-H1 is REFUTED** if PRISM-V fails on a majority of PHYSICAL datasets, **or** succeeds on a
> SOCIAL dataset. Either outcome falsifies the boundary claim, because the claim is about a boundary
> and not about average performance.
>
> **PR-H1 is UNDECIDED** if fewer than three usable PHYSICAL datasets can be obtained, or if the
> estimand is degenerate on them (fewer than 100 valid origins, or a censoring rate outside
> `(0, 0.6)` — the same D26 usability rule applied everywhere else).

### What would make me abandon the claim

* Any SOCIAL dataset on which PRISM-V beats the age-stratified marginal.
* PHYSICAL datasets succeeding only when they also happen to have the largest sample size, which
  would make sample size the real explanation. **Origin counts are reported for every dataset so a
  reader can check this directly.**
* A pattern explained better by task type (regression vs classification) or by base-predictor family
  than by drift mechanism. Both are recorded per dataset and inspected before the claim is made.

---

## 4. PR-H2, the baseline claim

Reported on **every** dataset, physical and social alike: the concordance of the age-stratified
marginal, against the constant Kaplan-Meier curve and against PRISM-V. No new decision rule is
needed; the claim is descriptive and the evidence is the table itself.

The point being made is methodological: the protocol originally named a *constant* marginal as the
lower reference, PRISM-V clears that on every dataset tested so far, and that comparison is close to
vacuous. Reporting the age-stratified floor alongside it is what turns a flattering number into an
informative one.

---

## 5. What is not being claimed

* Not that monitoring is useless — it is demonstrably informative about *present* risk.
* Not that PRISM-V is a good forecaster. Gate 1 refuted that and the verdict stands.
* Not that the physical/social split is the only boundary, or a sharp one. It is one testable
  boundary, tested once, on a small number of datasets.
* Not that any of this rescues Gate 1, Gate 2, Gate 5 or the full-scale Gate 0. All four failed and
  all four remain recorded as failed.

---

## 6. Sequencing deviation, recorded plainly

`go_no_go.md` stops the project when Gate 1 fails. This study proceeds past that stop at the user's
explicit direction, as a **new hypothesis with its own pre-registration**, not as an appeal of Gate
1's verdict. The deviation is recorded here and in `PROGRESS.md` (D30) rather than presented as the
original plan.
