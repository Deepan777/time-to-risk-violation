"""Datasets for the drift-mechanism boundary test (PR-H1).

Every dataset here was assigned a drift class -- PHYSICAL or SOCIAL -- in
`configs/drift_mechanism.yaml`, from the provider's own documentation, and that file was committed
before any of these archives was downloaded. Nothing in this module may change a class; it only
turns an archive into the project's canonical `Stream`.

**Windowing rule, fixed before any run.** A window size chosen per dataset after seeing results
would be a free parameter, so it is fixed by rule instead:

    w = the largest value in (25, 50, 100, 200, 400) for which the stream yields at least 120
        windows. If the stream yields more than 1000 windows at w = 400, it is decimated by taking
        every k-th row, k the smallest integer bringing the count to 1000 or fewer.

The lower bound of 120 windows exists because an origin needs H + 2 = 22 windows of future and the
estimand needs enough origins to fit anything; the upper bound of 1000 is a compute cap on the
quadratic discrepancy statistics, not a statistical choice. Decimation is uniform and
order-preserving: it lowers the sampling rate and leaves the temporal order untouched. A dataset
that cannot reach 120 windows at w = 25 is refused, and `hydraulic` is refused on exactly that
ground -- as `configs/drift_mechanism.yaml` predicted it might be.

**Target choice on `air_quality`.** The device carries five sensors and the certified analyser
reports four reference species, so the target is a real choice. It is made by decision D21, which
requires the task to carry irreducible noise: a base predictor that solves its task drives risk to
zero, and with it epsilon and the whole estimand. The reference species retained is the one whose
*maximum absolute correlation with any single sensor channel* is lowest among those leaving at
least 7,000 usable rows -- that is, the least trivially solvable task. The correlations are a
property of the raw archive, computed before any predictor is fitted, and all four are recorded in
`AIR_QUALITY_TARGET_SCREEN` so the choice can be checked rather than trusted.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .stream import Stream

__all__ = [
    "MECHANISM_DATASETS", "RAW_DIR", "WINDOW_LADDER", "MIN_WINDOWS", "MAX_WINDOWS",
    "AIR_QUALITY_TARGET_SCREEN", "choose_window",
    "load_gas_drift", "load_air_quality", "load_gas_temp_mod", "load_news_popularity",
    "load_hydraulic", "load_metropt3", "load_cmapss", "load_mechanism",
    "METROPT_TARGET_SCREEN", "METROPT_TARGET",
]

RAW_DIR = Path("data/raw/uci")

#: Candidate window sizes, largest-first selection. See the module docstring.
WINDOW_LADDER: tuple[int, ...] = (25, 50, 100, 200, 400)
MIN_WINDOWS = 120
MAX_WINDOWS = 1000

MECHANISM_DATASETS: tuple[str, ...] = (
    "gas_drift", "air_quality", "gas_temp_mod", "news_popularity", "hydraulic",
    "metropt3", "cmapss",
)

#: Recorded screen behind the `air_quality` target choice (D21). Computed on the raw archive
#: before any predictor was fitted; `rows` is what survives dropping the provider's -200 sentinel.
AIR_QUALITY_TARGET_SCREEN: dict[str, dict[str, float]] = {
    "C6H6(GT)": {"rows": 8991, "max_abs_corr": 0.9820},
    "CO(GT)":   {"rows": 7344, "max_abs_corr": 0.9155},
    "NOx(GT)":  {"rows": 7396, "max_abs_corr": 0.7870},
    "NO2(GT)":  {"rows": 7393, "max_abs_corr": 0.7081},   # selected: lowest, and >= 7000 rows
}


def choose_window(n_rows: int) -> tuple[int, int]:
    """Return `(window_size, decimation_k)` under the fixed rule. Refuses a stream that is too short.

    The rule is stated once, in the module docstring, and applied here without exception so that no
    dataset gets a window size tuned to it.
    """
    for w in sorted(WINDOW_LADDER, reverse=True):
        n_win = n_rows // w
        if n_win > MAX_WINDOWS and w == max(WINDOW_LADDER):
            k = int(np.ceil(n_win / MAX_WINDOWS))
            return w, k
        if n_win >= MIN_WINDOWS:
            return w, 1
    raise ValueError(
        f"stream of {n_rows} rows yields at most {n_rows // min(WINDOW_LADDER)} windows at the "
        f"smallest admissible window size ({min(WINDOW_LADDER)}), below the required "
        f"{MIN_WINDOWS}. The estimand cannot be defined on it; the dataset is refused."
    )


def _finalise(df: pd.DataFrame, feature_cols: list[str], target_col: str, task: str,
              meta: dict) -> Stream:
    """Apply the windowing rule, stamp the time index, and hand back a validated Stream."""
    w, k = choose_window(len(df))
    if k > 1:
        df = df.iloc[::k].reset_index(drop=True)
        meta = {**meta, "decimation_k": k,
                "decimation_note": f"every {k}-th row retained under the fixed windowing rule; "
                                   "uniform and order-preserving"}
    df = df.reset_index(drop=True)
    df.insert(0, "t", np.arange(len(df), dtype=np.int64))
    meta = {**meta, "window_size": int(w), "n_rows": int(len(df)),
            "n_windows": int(len(df) // w)}
    return Stream(df=df, time_col="t", feature_cols=feature_cols, target_col=target_col,
                  task=task, meta=meta)


def _zip(name: str) -> zipfile.ZipFile:
    path = RAW_DIR / f"{name}.zip"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent. Run scripts/22_fetch_mechanism_datasets.py, which records the "
            "download and its SHA-256 rather than fetching silently at import time."
        )
    return zipfile.ZipFile(path)


# ---------------------------------------------------------------------------- PHYSICAL


def load_gas_drift() -> Stream:
    """UCI-224, gas sensor array drift: 13,910 measurements, 16 sensors, 36 months.

    The canonical sensor-drift benchmark. The ten batch files are consecutive time periods, so
    concatenating them in batch order is the stream's own chronology -- no reordering is applied
    and none is needed.
    """
    zf = _zip("gas_drift")
    rows, labels, batch_of = [], [], []
    for b in range(1, 11):
        text = zf.read(f"Dataset/batch{b}.dat").decode()
        for line in text.strip().split("\n"):
            parts = line.split()
            labels.append(int(parts[0].split(";")[0]))
            vec = np.zeros(128, dtype=np.float32)
            for tok in parts[1:]:
                j, v = tok.split(":")
                vec[int(j) - 1] = float(v)
            rows.append(vec)
            batch_of.append(b)
    X = np.vstack(rows)
    feature_cols = [f"f{j}" for j in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = np.asarray(labels, dtype=int) - 1        # 1..6 -> 0..5
    return _finalise(
        df, feature_cols, "y", "multiclass",
        meta={
            "dataset_id": "gas_drift", "tier": "P", "family": "physical_sensor",
            "drift_class": "PHYSICAL",
            "licence": "UCI research use only; commercial purposes fully excluded (provider).",
            "source_url": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
            "citation": "Vergara et al., Chemical gas sensor drift compensation using classifier "
                        "ensembles, Sensors and Actuators B, 2012.",
            "domain": "chemical sensing (metal-oxide gas sensor array)",
            "task": "multiclass classification (6 analytes)",
            "temporal_structure": "10 acquisition batches, January 2007 to February 2011, "
                                  "concatenated in batch order",
            "shift_type": "sensor drift (poisoning, ageing, calibration loss) stated by provider",
            "n_batches": 10,
            "batch_boundaries": [int(i) for i in np.cumsum(np.bincount(batch_of)[1:])],
            "reason_for_inclusion":
                "PR-H1 confirmatory PHYSICAL dataset 1 of 3; the provider states the drift "
                "mechanism explicitly.",
        },
    )


def load_air_quality() -> Stream:
    """UCI-360, air quality: one year of a field-deployed metal-oxide multisensor device.

    Features are the five sensor channels plus the device's temperature, relative and absolute
    humidity -- everything the deployed instrument itself observes. The target is a certified
    reference analyser's concentration, so the task is on-field calibration, which is what the
    provider says the drift degrades.

    The provider's missing-value sentinel is -200. Rows missing any retained column are dropped
    rather than imputed: imputing the target would invent the ground truth the loss is measured
    against. `NMHC(GT)` is dropped entirely (8,443 of 9,357 rows missing).
    """
    zf = _zip("air_quality")
    df = pd.read_csv(io.BytesIO(zf.read("AirQualityUCI.csv")), sep=";", decimal=",",
                     encoding="latin-1")
    df = df.dropna(axis=1, how="all").dropna(subset=["Date"])

    feature_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
                    "PT08.S5(O3)", "T", "RH", "AH"]
    target = "NO2(GT)"                                   # D21 screen; see module docstring
    sub = df[feature_cols + [target]].apply(pd.to_numeric, errors="coerce")
    keep = (sub != -200).all(axis=1) & sub.notna().all(axis=1)
    n_dropped = int((~keep).sum())
    out = sub[keep].reset_index(drop=True)
    out = out.rename(columns={target: "y"})
    return _finalise(
        out, feature_cols, "y", "regression",
        meta={
            "dataset_id": "air_quality", "tier": "P", "family": "physical_sensor",
            "drift_class": "PHYSICAL",
            "licence": "UCI research use only; commercial purposes fully excluded (provider).",
            "source_url": "https://archive.ics.uci.edu/dataset/360/air+quality",
            "citation": "De Vito et al., On field calibration of an electronic nose for benzene "
                        "estimation in an urban pollution monitoring scenario, Sensors and "
                        "Actuators B, 2008.",
            "domain": "environmental chemical sensing (on-field electronic nose)",
            "task": f"regression ({target} reference concentration from sensor responses)",
            "temporal_structure": "hourly averages, March 2004 to February 2005",
            "shift_type": "sensor drift and cross-sensitivity, stated by the provider",
            "target_col_source": target,
            "target_screen": AIR_QUALITY_TARGET_SCREEN,
            "target_rule": "D21: among reference species leaving >= 7000 rows, the one with the "
                           "lowest maximum absolute correlation against any single sensor channel",
            "rows_dropped_missing": n_dropped,
            "missing_policy": "rows carrying the provider's -200 sentinel in any retained column "
                              "are dropped; nothing is imputed",
            "reason_for_inclusion":
                "PR-H1 confirmatory PHYSICAL dataset 2 of 3.",
        },
    )


def load_gas_temp_mod() -> Stream:
    """UCI-487, temperature-modulated MOX array: three weeks of continuous CO measurement.

    Features are the 14 sensor resistances, the heater voltage that modulates them, and the chamber
    humidity and temperature -- the quantities a deployed instrument would itself observe. Flow rate
    is excluded: it is a setting of the gas-delivery apparatus that co-determines the delivered CO
    concentration, so it is a channel into the target rather than an observation of the world.

    The 13 daily files are named by acquisition timestamp; sorting the filenames is the acquisition
    order.
    """
    outer = _zip("gas_temp_mod")
    inner = zipfile.ZipFile(io.BytesIO(
        outer.read("gas-sensor-array-temperature-modulation.zip")))
    names = sorted(n for n in inner.namelist() if n.endswith(".csv"))
    keep = (["Humidity (%r.h.)", "Temperature (C)", "Heater voltage (V)"]
            + [f"R{i} (MOhm)" for i in range(1, 15)])
    frames = []
    for n in names:
        d = pd.read_csv(io.BytesIO(inner.read(n)), usecols=keep + ["CO (ppm)"])
        frames.append(d.astype(np.float32))
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.rename(columns={"CO (ppm)": "y"})
    feature_cols = list(keep)
    df = df[feature_cols + ["y"]].dropna().reset_index(drop=True)
    return _finalise(
        df, feature_cols, "y", "regression",
        meta={
            "dataset_id": "gas_temp_mod", "tier": "P", "family": "physical_sensor",
            "drift_class": "PHYSICAL",
            "licence": "UCI research use only; commercial purposes fully excluded (provider).",
            "source_url":
                "https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation",
            "citation": "Burgues and Marco, Multivariate estimation of the limit of detection by "
                        "orthogonal partial least squares in temperature-modulated MOX sensors, "
                        "Analytica Chimica Acta, 2018.",
            "domain": "chemical sensing (temperature-modulated MOX array in a gas chamber)",
            "task": "regression (CO concentration in ppm from sensor resistances)",
            "temporal_structure": "13 daily acquisitions, 30 September to 16 October 2016, "
                                  "concatenated in filename (acquisition) order",
            "shift_type": "controlled physical covariate (heater-voltage modulation) driving the "
                          "observable feature distribution; stated by the provider",
            "excluded_columns": ["Flow rate (mL/min)"],
            "excluded_reason": "a gas-delivery apparatus setting that co-determines the target",
            "reason_for_inclusion":
                "PR-H1 confirmatory PHYSICAL dataset 3 of 3.",
        },
    )


def load_hydraulic() -> Stream:
    """UCI-447, hydraulic condition monitoring. Expected to be refused by the windowing rule.

    Kept as executable code rather than a note so that the refusal is produced by the rule at run
    time and recorded, instead of being asserted in prose.
    """
    zf = _zip("hydraulic")
    cheap = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
    cols, feature_cols = {}, []
    for s in cheap:
        arr = np.loadtxt(io.BytesIO(zf.read(f"{s}.txt")), dtype=np.float32)
        cols[f"{s}_mean"] = arr.mean(axis=1)
        cols[f"{s}_std"] = arr.std(axis=1)
        feature_cols += [f"{s}_mean", f"{s}_std"]
    profile = np.loadtxt(io.BytesIO(zf.read("profile.txt")), dtype=np.float32)
    df = pd.DataFrame(cols)
    df["y"] = (profile[:, 0] < 100).astype(int)          # cooler efficiency below full
    return _finalise(
        df, feature_cols, "y", "binary",
        meta={
            "dataset_id": "hydraulic", "tier": "P", "family": "physical_sensor",
            "drift_class": "PHYSICAL",
            "licence": "UCI research use; citation required.",
            "source_url":
                "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
            "citation": "Helwig, Pignanelli and Schutze, Condition monitoring of a complex "
                        "hydraulic system using multivariate statistics, IEEE I2MTC, 2015.",
            "domain": "hydraulic test rig",
            "task": "binary classification (cooler condition degraded)",
            "temporal_structure": "2205 consecutive 60-second load cycles",
            "shift_type": "component degradation quantitatively varied by the experimenters",
            "reason_for_inclusion": "PR-H1 pre-declared PHYSICAL reserve.",
        },
    )


# ---------------------------------------------------------------------------- SOCIAL


def load_news_popularity() -> Stream:
    """UCI-332, Mashable article popularity over two years. The new SOCIAL comparator.

    The archive is already sorted by `timedelta`, the number of days between publication and
    acquisition, descending -- that is reverse chronological order, so reversing the rows restores
    chronology exactly. `timedelta` has day resolution, so many articles tie; ties are broken by the
    archive's own row order within a day, which is the only ordering information the file carries.
    The tie-break is documented here rather than left implicit, as `Stream` requires.

    The target is the source paper's own threshold: popular iff shares > 1400, the archive's median.
    """
    zf = _zip("news_popularity")
    df = pd.read_csv(io.BytesIO(zf.read("OnlineNewsPopularity/OnlineNewsPopularity.csv")))
    df.columns = [c.strip() for c in df.columns]
    if not bool((np.diff(df["timedelta"].to_numpy()) <= 0).all()):
        raise ValueError(
            "OnlineNewsPopularity is no longer sorted by descending timedelta; the chronology "
            "assumed by this loader does not hold and the ordering rule must be restated."
        )
    df = df.iloc[::-1].reset_index(drop=True)            # reverse -> chronological
    drop = ["url", "timedelta", "shares"]
    feature_cols = [c for c in df.columns if c not in drop]
    out = df[feature_cols].astype(np.float32).copy()
    out["y"] = (df["shares"].to_numpy() > 1400).astype(int)
    return _finalise(
        out, feature_cols, "y", "binary",
        meta={
            "dataset_id": "news_popularity", "tier": "P", "family": "social_editorial",
            "drift_class": "SOCIAL",
            "licence": "UCI research use; derived statistics only, original article content "
                       "rights retained by Mashable.",
            "source_url": "https://archive.ics.uci.edu/dataset/332/online+news+popularity",
            "citation": "Fernandes, Vinagre and Cortez, A proactive intelligent decision support "
                        "system for predicting the popularity of online news, EPIA, 2015.",
            "domain": "online news (Mashable)",
            "task": "binary classification (shares above the archive median of 1400)",
            "temporal_structure": "two years of articles, reversed from the archive's descending "
                                  "timedelta; ties within a day broken by archive row order",
            "shift_type": "editorial output and audience sharing behaviour; no physical substrate",
            "tie_break_rule": "archive row order within a timedelta day",
            "reason_for_inclusion":
                "PR-H1 confirmatory SOCIAL comparator. The boundary claim requires a dataset on "
                "which monitoring is predicted to fail, not only ones where it may succeed.",
        },
    )


# ------------------------------------------------------- PHYSICAL, added by pre-registration
# addendum 1. Both were named and classified before either archive was downloaded; see
# proposal/PREREGISTRATION_drift_mechanism_ADDENDUM_1.md sections 3 and 4.

METROPT_ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
                  "Motor_current"]
METROPT_DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
                   "Caudal_impulses"]

#: The declared D21 screen, run on the raw archive before any predictor was fitted. `admissible`
#: is the pre-registered base-rate window [0.05, 0.95]; the selected target is the admissible
#: channel with the lowest maximum absolute correlation against any single analog channel.
METROPT_TARGET_SCREEN: dict[str, dict[str, object]] = {
    "COMP":            {"base_rate": 0.8370, "max_abs_corr": 0.9714, "admissible": True},
    "DV_eletric":      {"base_rate": 0.1606, "max_abs_corr": 0.9587, "admissible": True},
    "Towers":          {"base_rate": 0.9198, "max_abs_corr": 0.6290, "admissible": True},
    "MPG":             {"base_rate": 0.8327, "max_abs_corr": 0.9543, "admissible": True},
    "LPS":             {"base_rate": 0.0034, "max_abs_corr": 0.3252, "admissible": False},
    "Pressure_switch": {"base_rate": 0.9914, "max_abs_corr": 0.0883, "admissible": False},
    "Oil_level":       {"base_rate": 0.9042, "max_abs_corr": 0.1452, "admissible": True},
    "Caudal_impulses": {"base_rate": 0.9371, "max_abs_corr": 0.0681, "admissible": True},
}
METROPT_TARGET = "Caudal_impulses"          # selected by the screen above


def load_metropt3() -> Stream:
    """UCI-791, MetroPT-3: an in-service metro train's air production unit, February to August 2020.

    The archive is unlabelled, so the deployed model is a *soft sensor*: it predicts one digital
    channel from the seven analog ones. Which digital channel is not a free choice -- it is fixed by
    the D21 screen declared in addendum 1 and recorded in `METROPT_TARGET_SCREEN`, so a reader can
    see every candidate that was considered and why this one won.

    The provider's failure reports (four air leaks between April and July 2020, each followed by a
    dated maintenance action) are carried in the metadata. They are *not* used to build the target,
    which would make the drift its own label; they are there so that the risk trajectory can be
    read against the physical events that generated it.
    """
    zf = _zip("metropt3")
    df = pd.read_csv(io.BytesIO(zf.read("MetroPT3(AirCompressor).csv")))
    if not bool(pd.Series(df["timestamp"]).is_monotonic_increasing):
        raise ValueError("MetroPT-3 is not sorted by timestamp; the loader's chronology fails")
    out = df[METROPT_ANALOG].astype(np.float32).copy()
    out["y"] = (df[METROPT_TARGET].to_numpy() > 0.5).astype(int)
    out = out.dropna().reset_index(drop=True)
    return _finalise(
        out, list(METROPT_ANALOG), "y", "binary",
        meta={
            "dataset_id": "metropt3", "tier": "P", "family": "physical_pneumatic",
            "drift_class": "PHYSICAL", "measured": True,
            "licence": "UCI research use; citation required.",
            "source_url": "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset",
            "citation": "Veloso et al., The MetroPT dataset for predictive maintenance, "
                        "Scientific Data, 2022.",
            "domain": "rail transport (compressor air production unit, in service)",
            "task": f"binary classification ({METROPT_TARGET} from the analog channels)",
            "temporal_structure": "1 February to 31 August 2020, regular sampling, "
                                  "strictly increasing timestamps",
            "shift_type": "component degradation and air leaks on a measured in-service system; "
                          "the provider supplies dated failure reports",
            "target_col_source": METROPT_TARGET,
            "target_screen": METROPT_TARGET_SCREEN,
            "target_rule": "addendum 1: among digital channels with base rate in [0.05, 0.95], "
                           "the lowest maximum absolute correlation with any single analog channel",
            "provider_failure_reports": [
                {"start": "2020-04-18 00:00", "end": "2020-04-18 23:59", "failure": "air leak",
                 "severity": "high stress"},
                {"start": "2020-05-29 23:30", "end": "2020-05-30 06:00", "failure": "air leak",
                 "severity": "high stress", "maintenance": "2020-04-30 12:00"},
                {"start": "2020-06-05 10:00", "end": "2020-06-07 14:30", "failure": "air leak",
                 "severity": "high stress", "maintenance": "2020-06-08 16:00"},
                {"start": "2020-07-15 14:30", "end": "2020-07-15 19:00", "failure": "air leak",
                 "severity": "high stress", "maintenance": "2020-07-16 00:00"},
            ],
            "failure_reports_use": "metadata only; never used to construct the target",
            "reason_for_inclusion":
                "PR-H1 confirmatory PHYSICAL dataset, measured. Replaces air_quality, which failed "
                "the D26 usability screen before any model was fitted.",
        },
    )


#: Column layout of the C-MAPSS text files: unit, cycle, three operational settings, 21 sensors.
CMAPSS_SETTINGS = [f"setting_{i}" for i in (1, 2, 3)]
CMAPSS_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss(subset: str = "FD002") -> Stream:
    """NASA C-MAPSS turbofan degradation, subset FD002. **Simulator-generated.**

    Addendum 1 admits this dataset and fixes the consequence of its being simulated in advance: the
    confirmatory result is reported both over all three PHYSICAL datasets and over the two measured
    ones alone, and where the two readings disagree the measured-only reading governs the claim.

    Units are concatenated in unit order, which is the order the archive stores them in. The target
    is uncapped remaining useful life -- the benchmark's own task, without the piecewise-linear cap
    that the RUL literature usually applies, because applying it would be a modelling choice made
    after the pre-registration rather than the benchmark's own definition. The unit and cycle
    columns are metadata, never features: `cycle` would give RUL away wherever unit lengths are
    similar.

    Labels mature when a unit fails. A 400-row window spans roughly two units, so the protocol's
    one-window label delay is approximately, not exactly, correct here. That approximation is a
    limitation of this dataset and is stated in the manuscript.
    """
    zf = _zip("cmapss")
    inner = zipfile.ZipFile(io.BytesIO(
        zf.read("6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip")))
    arr = np.loadtxt(io.BytesIO(inner.read(f"train_{subset}.txt")), dtype=np.float64)
    cols = ["unit", "cycle"] + CMAPSS_SETTINGS + CMAPSS_SENSORS
    df = pd.DataFrame(arr, columns=cols)
    unit = df["unit"].to_numpy().astype(int)
    if not bool((np.diff(unit) >= 0).all()):
        raise ValueError("C-MAPSS units are not stored in unit order; the concatenation is unsafe")
    last_cycle = df.groupby("unit")["cycle"].transform("max").to_numpy()
    feature_cols = CMAPSS_SETTINGS + CMAPSS_SENSORS
    out = df[feature_cols].astype(np.float32).copy()
    out["y"] = (last_cycle - df["cycle"].to_numpy()).astype(np.float32)
    return _finalise(
        out, feature_cols, "y", "regression",
        meta={
            "dataset_id": "cmapss", "tier": "P", "family": "physical_turbomachinery",
            "drift_class": "PHYSICAL", "measured": False,
            "licence": "NASA open data (US Government work, no copyright).",
            "source_url": "https://phm-datasets.s3.amazonaws.com/NASA/"
                          "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
            "citation": "Saxena, Goebel, Simon and Eklund, Damage propagation modeling for "
                        "aircraft engine run-to-failure simulation, IEEE PHM, 2008.",
            "domain": "turbofan engine prognostics (simulated)",
            "task": "regression (uncapped remaining useful life, in cycles)",
            "temporal_structure": f"{subset}: 260 engine units concatenated in unit order, each "
                                  "run from a healthy state to failure",
            "shift_type": "modelled component degradation propagating through the engine",
            "subset": subset,
            "excluded_columns": ["unit", "cycle"],
            "excluded_reason": "metadata; `cycle` would give the target away",
            "caveat": "SIMULATOR-GENERATED. Reported separately from the measured PHYSICAL "
                      "datasets, and the measured-only reading governs the claim where the two "
                      "disagree (pre-registration addendum 1).",
            "label_delay_caveat": "labels mature at unit failure; a window spans roughly two "
                                  "units, so the one-window label delay is approximate",
            "reason_for_inclusion":
                "PR-H1 confirmatory PHYSICAL dataset. Replaces gas_drift, which failed the D26 "
                "usability screen before any model was fitted.",
        },
    )


_LOADERS = {
    "gas_drift": load_gas_drift,
    "air_quality": load_air_quality,
    "gas_temp_mod": load_gas_temp_mod,
    "news_popularity": load_news_popularity,
    "hydraulic": load_hydraulic,
    "metropt3": load_metropt3,
    "cmapss": load_cmapss,
}


def load_mechanism(name: str) -> Stream:
    if name not in _LOADERS:
        raise ValueError(f"unknown mechanism dataset {name!r}; expected one of {MECHANISM_DATASETS}")
    return _LOADERS[name]()
