# Addendum 1 to the drift-mechanism pre-registration

**Written:** 2026-09-02
**Amends:** `proposal/PREREGISTRATION_drift_mechanism.md` (committed 32af9fd) and
`configs/drift_mechanism.yaml` (committed a1af1bd).

**State of knowledge at the time of writing.** Four candidate streams have been built and screened.
**No PRISM-V model has been fitted on any of them, no reference model has been fitted on any of
them, and no concordance, Brier score or test statistic exists for any of them.** Everything below
is decided on the structure of the estimand alone, which is what makes the amendment admissible.

---

## 1. What the screen returned

The pre-registered usability rule is D26, unchanged: a dataset is usable only if it yields **at
least 100 valid origins** and a **censoring rate strictly between 0 and 0.6**, at the operational
margin 0.20 that D26 fixed before Gate 1.

| dataset | class | rows | windows | valid origins | censoring | usable |
|---|---|---|---|---|---|---|
| `gas_drift` | PHYSICAL | 13,910 | 112 | **97** | 0.000 | **no** |
| `air_quality` | PHYSICAL | 7,393 | 118 | **41** | 0.000 | **no** |
| `gas_temp_mod` | PHYSICAL | 384,316 | 768 | 651 | 0.218 | yes |
| `news_popularity` | SOCIAL | 39,644 | 159 | 562 | 0.060 | yes |
| `hydraulic` | PHYSICAL reserve | 2,205 | — | — | — | **refused at load** |

`hydraulic` was refused by the windowing rule before it could be monitored at all, exactly as
`configs/drift_mechanism.yaml` recorded it might be. `gas_drift` misses the origin threshold by
three origins. It is **not** rescued: the threshold is the threshold, and moving it now — with the
count known — is the precise thing pre-registration exists to prevent.

While screening, the same rule was applied to the incumbent SOCIAL comparators. `huffpost` yields
125 origins at **zero** censoring, so it fails the censoring half of D26 — consistent with D27,
which had already excluded it for a related reason. The usable SOCIAL comparators are therefore
`yearbook` and `news_popularity`.

## 2. Why the failures are structural, and what that changes

Both exclusions have the same cause and it is not statistical: the streams are short. 13,910 and
7,393 rows yield roughly 110 windows, an origin needs H + 2 = 22 windows of future, and what is
left is under a hundred origins. The zero censoring has the same root — with so few origins and a
frozen model that breaks early and stays broken, every origin's violation is observed inside the
horizon.

The selection criterion in the original pre-registration screened candidates on **drift mechanism**
and forgot to screen them on **length**. That is a defect in the original document, and the
amendment repairs it prospectively:

> **Added screen (S1).** A confirmatory dataset must carry at least **150,000 rows** before any
> decimation. Applied to the datasets already chosen, it excludes `gas_drift`, `air_quality` and
> `hydraulic` and retains `gas_temp_mod` and `news_popularity` — the same verdicts D26 reached, now
> reachable before download rather than after.

## 3. Replacement PHYSICAL datasets, named before download

The classification rule of section 2 of the pre-registration is **unchanged**. Both replacements
are assigned from provider documentation only, and neither archive has been downloaded at the time
this file is committed.

**`metropt3` — UCI-791, MetroPT-3.** 1,516,948 rows. Pressure, temperature, motor-current and
intake-valve readings from the Air Production Unit of an in-service metro train, February to August
2020. The provider supplies dated failure reports — four air-leak failures with severity and the
maintenance dates that followed. A stated, dated, physical degradation mechanism on a measured
in-service system. **PHYSICAL.**

**`cmapss` — NASA C-MAPSS, subset FD002.** 53,759 cycles across 260 engine units. The provider
attributes the trajectory to modelled component degradation propagating through a turbofan.
**PHYSICAL, with a caveat declared here and not later:** C-MAPSS is **simulator-generated**. That
is why `naval_propulsion` (UCI-316) was rejected outright, but the two differ on the ground that
rejection actually rested on — 316 is a uniform grid sweep with no time order, while C-MAPSS is a
set of ordered degradation trajectories, and the estimand is defined on the latter.

> **Binding consequence of the caveat.** The confirmatory result will be reported **twice**: over
> all three PHYSICAL datasets, and restricted to the two **measured** ones (`gas_temp_mod`,
> `metropt3`). Where the two readings disagree, **the measured-only reading governs the claim.**
> This is fixed now, before either archive is downloaded, so it cannot be chosen afterwards.

**Why these two and not more gas-sensor archives.** UCI's large-instance holdings were enumerated
and every candidate over 150,000 rows was checked against the classification rule; the only other
long chemical-sensor archives (UCI-322, UCI-362, UCI-799) state no drift mechanism and are
therefore AMBIGUOUS, not PHYSICAL. Preferring them would also have made the PHYSICAL class a class
of *MOX gas sensors* rather than of physical mechanisms. As it stands the surviving PHYSICAL set
spans chemical sensing, pneumatics and turbomachinery, which is the diversity the boundary claim
needs. The enumeration is reproducible from `scripts/22_fetch_mechanism_datasets.py`.

## 4. Target rules for the two new datasets, fixed before download

Both archives need a supervised task defined, which is a free choice unless it is ruled. It is
ruled here, by D21 (a base predictor that solves its task drives risk, epsilon and the estimand to
zero), in the same form already applied to `air_quality`:

**`metropt3`.** The archive is unlabelled, so the deployed model is a *soft sensor*. Features are
the seven analog channels (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`,
`Motor_current`). The target is chosen from the eight digital channels by: *among those with a base
rate inside [0.05, 0.95], the one whose maximum absolute correlation with any single analog channel
is lowest.* The full screen is recorded in the loader so the choice can be checked.

**`cmapss`.** Features are the three operational settings and twenty-one sensor channels. The
target is remaining useful life, the benchmark's own task, as a regression. Units are concatenated
in unit order; labels mature when a unit fails, and since a 400-row window spans roughly two units,
the protocol's one-window label delay is approximately correct for this stream. **That
approximation is a limitation of this dataset and will be stated in the manuscript**, not buried in
the loader.

## 5. What is unchanged

The decision rule of section 3 stands verbatim: **PR-H1 is SUPPORTED** iff PRISM-V beats the
age-stratified marginal on at least two of three new PHYSICAL datasets **and** fails on the SOCIAL
datasets; **REFUTED** if it fails a majority of PHYSICAL or succeeds on any SOCIAL; **UNDECIDED** if
fewer than three usable PHYSICAL datasets can be obtained. "Beats" remains Holm-corrected paired
Wilcoxon at alpha = 0.05 **and** Cliff's delta >= 0.147, on all five seeds. `insects` remains
disqualified as confirmatory evidence. The falsifiers in section 3 stand, and the sample-size
falsifier is now more sharply testable, since origin counts differ by an order of magnitude across
the surviving datasets.
