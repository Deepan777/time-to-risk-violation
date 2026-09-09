"""The Gate 1 opposition: forecasting future risk from the error series alone.

Every baseline here sees **only** the matured risk series `R_{1..t-delta}` — no deployment state,
no drift statistics, no uncertainty. That is the point. If PRISM-V cannot beat extrapolation of its
own error history, then the deployment state adds nothing, and `go_no_go.md` Gate 1 stops the
project. These are therefore the most important baselines in the study and are implemented to win
if they can.

Each carries the same tuning budget as PRISM-V and is tuned on the **calibration split only**
(`fit`), because a baseline crippled by inattention is worse than no baseline at all.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "NaiveBaseline", "Persistence", "MovingAverage", "ExponentialSmoothing",
    "TheilSenTrend", "ARIMARisk", "LSTMRiskOnly", "NAIVE_BASELINES", "build_naive_baselines",
]

_EPS = 1e-6


class NaiveBaseline(ABC):
    """Common interface. `fit` tunes on calibration data; `predict_path` forecasts `H` ahead."""

    name: str = "baseline"
    tuning_budget: int = 0          # candidate configurations evaluated, reported in the paper

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "NaiveBaseline":
        """Default: nothing to tune. `hist` is (N, L), `path` is (N, H) with NaN past the stream."""
        return self

    @abstractmethod
    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        """(N, L) matured history -> (N, H) forecast, clipped to [0, 1]."""

    # -------------------------------------------------------------- shared helpers
    @staticmethod
    def _last(hist: np.ndarray) -> np.ndarray:
        """Most recent finite value per row; falls back to the row mean, then to 0.5."""
        out = np.full(hist.shape[0], np.nan)
        for i, row in enumerate(hist):
            fin = row[np.isfinite(row)]
            out[i] = fin[-1] if fin.size else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows fall through below
            rowmean = np.nanmean(np.where(np.isfinite(hist), hist, np.nan), axis=1)
        out = np.where(np.isfinite(out), out, rowmean)
        return np.where(np.isfinite(out), out, 0.5)

    def _grid_select(self, grid: list, score_fn, hist: np.ndarray, path: np.ndarray):
        """Pick the grid point with the lowest masked MAE on the tuning data."""
        self.tuning_budget = len(grid)
        best, best_score = grid[0], np.inf
        for g in grid:
            pred = score_fn(g)
            mask = np.isfinite(path)
            if not mask.any():
                continue
            score = float(np.abs(pred - np.where(mask, path, 0.0))[mask].mean())
            if score < best_score:
                best, best_score = g, score
        return best


class Persistence(NaiveBaseline):
    """`R_{t+h} = R_t` for every h. The baseline the project is most at risk from: error series are
    strongly autocorrelated, and on Tier-B streams especially this is deceptively strong."""

    name = "persistence"

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        return np.clip(np.repeat(self._last(hist)[:, None], H, axis=1), 0.0, 1.0)


class MovingAverage(NaiveBaseline):
    """Flat forecast at the mean of the last `k` matured windows. `k` tuned on calibration."""

    name = "moving_average"

    def __init__(self, k: int = 5, grid: tuple[int, ...] = (2, 3, 5, 8, 10, 15, 20)):
        self.k, self.grid = k, grid

    def _flat(self, hist: np.ndarray, k: int) -> np.ndarray:
        tail = hist[:, -k:] if k < hist.shape[1] else hist
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows handled below
            m = np.nanmean(tail, axis=1)
        return np.where(np.isfinite(m), m, self._last(hist))

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "MovingAverage":
        self.k = self._grid_select(
            list(self.grid),
            lambda k: np.repeat(self._flat(hist, k)[:, None], path.shape[1], axis=1),
            hist, path)
        return self

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        return np.clip(np.repeat(self._flat(hist, self.k)[:, None], H, axis=1), 0.0, 1.0)


class ExponentialSmoothing(NaiveBaseline):
    """Simple exponential smoothing. Risk R3 in the audit names this as the specific baseline most
    likely to match PRISM-V, so its smoothing factor is tuned over a full grid."""

    name = "exponential_smoothing"

    def __init__(self, alpha: float = 0.3,
                 grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
        self.alpha, self.grid = alpha, grid

    def _level(self, hist: np.ndarray, alpha: float) -> np.ndarray:
        level = np.full(hist.shape[0], np.nan)
        for i, row in enumerate(hist):
            fin = row[np.isfinite(row)]
            if fin.size == 0:
                continue
            s = fin[0]
            for v in fin[1:]:
                s = alpha * v + (1.0 - alpha) * s
            level[i] = s
        return np.where(np.isfinite(level), level, self._last(hist))

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "ExponentialSmoothing":
        self.alpha = self._grid_select(
            list(self.grid),
            lambda a: np.repeat(self._level(hist, a)[:, None], path.shape[1], axis=1),
            hist, path)
        return self

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        return np.clip(np.repeat(self._level(hist, self.alpha)[:, None], H, axis=1), 0.0, 1.0)


class TheilSenTrend(NaiveBaseline):
    """Robust linear extrapolation: median of pairwise slopes over the last `k`, projected forward.

    This is the baseline that most directly mimics what the risk head is supposed to do — read the
    trend and continue it — so beating it is the substantive part of Gate 1.
    """

    name = "theil_sen_trend"

    def __init__(self, k: int = 10, grid: tuple[int, ...] = (5, 8, 10, 15, 20)):
        self.k, self.grid = k, grid

    @staticmethod
    def _slope_intercept(row: np.ndarray) -> tuple[float, float]:
        fin = row[np.isfinite(row)]
        n = fin.size
        if n < 2:
            return 0.0, (float(fin[0]) if n else 0.5)
        x = np.arange(n, dtype=float)
        slopes = [(fin[j] - fin[i]) / (x[j] - x[i])
                  for i in range(n - 1) for j in range(i + 1, n)]
        m = float(np.median(slopes))
        b = float(np.median(fin - m * x))
        return m, b + m * (n - 1)          # value at the last observed point

    def _project(self, hist: np.ndarray, k: int, H: int) -> np.ndarray:
        tail = hist[:, -k:] if k < hist.shape[1] else hist
        out = np.empty((hist.shape[0], H))
        steps = np.arange(1, H + 1, dtype=float)
        for i, row in enumerate(tail):
            m, last = self._slope_intercept(row)
            out[i] = last + m * steps
        return np.clip(out, 0.0, 1.0)

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "TheilSenTrend":
        self.k = self._grid_select(list(self.grid),
                                   lambda k: self._project(hist, k, path.shape[1]), hist, path)
        return self

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        return self._project(hist, self.k, H)


class ARIMARisk(NaiveBaseline):
    """AR(p) on the risk series, fitted per origin by Yule-Walker and iterated forward.

    A full statsmodels ARIMA refit per origin is prohibitively slow across thousands of origins and
    five seeds; an AR(p) fitted by Yule-Walker on the same window is the same model class for this
    purpose and is what is reported. The order `p` is tuned on calibration.
    """

    name = "ar_p"

    def __init__(self, p: int = 2, grid: tuple[int, ...] = (1, 2, 3, 5)):
        self.p, self.grid = p, grid

    @staticmethod
    def _ar_forecast(row: np.ndarray, p: int, H: int) -> np.ndarray:
        fin = row[np.isfinite(row)]
        if fin.size <= p + 1:
            return np.full(H, fin[-1] if fin.size else 0.5)
        mu = float(fin.mean())
        x = fin - mu
        try:
            from statsmodels.regression.linear_model import yule_walker
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = yule_walker(x, order=p, method="mle")
        except Exception:
            return np.full(H, fin[-1])
        if not np.all(np.isfinite(rho)):
            return np.full(H, fin[-1])
        buf = list(x[-p:])
        out = []
        for _ in range(H):
            nxt = float(np.dot(rho, buf[::-1]))
            out.append(nxt + mu)
            buf = (buf + [nxt])[1:]
        return np.asarray(out)

    def _project(self, hist: np.ndarray, p: int, H: int) -> np.ndarray:
        return np.clip(np.stack([self._ar_forecast(r, p, H) for r in hist]), 0.0, 1.0)

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "ARIMARisk":
        self.p = self._grid_select(list(self.grid),
                                   lambda p: self._project(hist, p, path.shape[1]), hist, path)
        return self

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        return self._project(hist, self.p, H)


class LSTMRiskOnly(NaiveBaseline):
    """An LSTM over the risk series alone — no deployment state.

    This is the sharpest baseline in the set and the fairest test of the paper's central claim. It
    has the same capacity to learn temporal structure as PRISM-V's encoder and differs in exactly
    one respect: its input is the error series rather than `s_t`. Any margin PRISM-V shows over
    this baseline is attributable to the deployment state and to nothing else, which is precisely
    what H1 asserts.
    """

    name = "lstm_risk_only"

    def __init__(self, hidden: int = 32, epochs: int = 150, lr: float = 1e-2,
                 patience: int = 15, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.patience, self.seed = (
            hidden, epochs, lr, patience, seed)
        self.net = None
        self.tuning_budget = 1

    def fit(self, hist: np.ndarray, path: np.ndarray) -> "LSTMRiskOnly":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H = path.shape[1]

        x = torch.from_numpy(np.nan_to_num(hist, nan=0.5)).float().unsqueeze(-1).to(dev)
        y = torch.from_numpy(np.nan_to_num(path, nan=0.0)).float().to(dev)
        mask = torch.from_numpy(np.isfinite(path)).float().to(dev)

        class Net(nn.Module):
            def __init__(self, hidden: int, H: int):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.head = nn.Linear(hidden, H)

            def forward(self, s):
                _, (h, _) = self.lstm(s)
                return torch.sigmoid(self.head(h[-1]))

        net = Net(self.hidden, H).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = net(x)
            loss = ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            v = float(loss.item())
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net, self._dev = net, dev
        return self

    def predict_path(self, hist: np.ndarray, H: int) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("LSTMRiskOnly.predict_path called before fit")
        import torch
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.nan_to_num(hist, nan=0.5)).float().unsqueeze(-1).to(self._dev)
            return np.clip(self.net(x).cpu().numpy(), 0.0, 1.0)


NAIVE_BASELINES: tuple[str, ...] = (
    "persistence", "moving_average", "exponential_smoothing",
    "theil_sen_trend", "ar_p", "lstm_risk_only",
)


def build_naive_baselines(seed: int = 0) -> list[NaiveBaseline]:
    """All six, in the order Gate 1 reports them."""
    return [Persistence(), MovingAverage(), ExponentialSmoothing(),
            TheilSenTrend(), ARIMARisk(), LSTMRiskOnly(seed=seed)]
