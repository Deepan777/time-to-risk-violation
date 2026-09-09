"""`PrismV` — the trainable model, in either variant, behind one interface.

Both variants expose the same three outputs, so every downstream metric, baseline comparison and
gate evaluation treats them identically and E15 is a like-for-like contest:

    .predict_survival(S, eps) -> (N, H)
    .predict_risk_path(S)     -> (N, H)
    .horizon(S, eps)          -> HorizonEstimate

Early stopping is on a **calibration** metric matched to the task being evaluated:

* `ipcw_ibs`  for the survival experiments (E2, Gate 2) - the default;
* `risk_mae`  for the risk-path experiments (E1, Gate 1).

Selecting a checkpoint by one output head and then scoring the other is simply a misconfigured
experiment. It also handicaps PRISM-V in the Gate-1 comparison, because every naive baseline is
already tuned on calibration risk-path MAE; matching the selection metric makes that contest fair
rather than favourable.

Test data touches nothing here: not selection, not thresholding, not the operating point, not the
stopping epoch.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..survival.utils import median_horizon, survival_to_expected_horizon
from .dataset import ValidityDataset
from .encoders import build_encoder
from .heads import FirstPassageModel, TwoHeadModel
from .losses import monotonicity_penalty, path_nll, risk_path_loss, survival_nll

__all__ = ["PrismV", "PrismVConfig", "HorizonEstimate", "VARIANTS"]

VARIANTS: tuple[str, ...] = ("two_head", "first_passage")


@dataclass
class PrismVConfig:
    variant: str = "first_passage"
    encoder: str = "gru"
    hidden: int = 64
    layers: int = 1
    horizon_dim: int = 16
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    n_samples: int = 256          # Monte Carlo paths for first-passage read-out
    alpha: float = 1.0            # two_head only: weight on the risk-path loss
    beta: float = 0.1             # two_head only: weight on Omega_mono
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    early_stop_metric: str = "ipcw_ibs"   # "ipcw_ibs" | "risk_mae"
    #: E16: predict a correction to the online exponential-smoothing level instead of the absolute
    #: risk. A zero correction reproduces that level exactly, so this isolates whether the
    #: deployment state adds anything ON TOP of the best simple extrapolator - the mechanism the
    #: Gate 1 failure identified.
    use_anchor: bool = False

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class HorizonEstimate:
    expected: np.ndarray        # E[T] over the horizon grid
    median: np.ndarray          # first h with S(h) <= 0.5
    survival: np.ndarray        # (N, H)
    lower_bound: np.ndarray     # conservative horizon at the calibrated level


class PrismV:
    """Fit, predict, and nothing else. Calibration lives in `calibrate.py`."""

    def __init__(self, cfg: PrismVConfig, H: int, d_in: int):
        if cfg.variant not in VARIANTS:
            raise ValueError(f"unknown variant {cfg.variant!r}; expected one of {VARIANTS}")
        self.cfg = cfg
        self.H = H
        self.d_in = d_in
        self.device = cfg.resolved_device()

        torch.manual_seed(cfg.seed)
        enc = build_encoder(cfg.encoder, d_in, hidden=cfg.hidden, dropout=cfg.dropout)
        cls = TwoHeadModel if cfg.variant == "two_head" else FirstPassageModel
        self.net = cls(enc, H=H, horizon_dim=cfg.horizon_dim, hidden=cfg.hidden).to(self.device)
        self.history: list[dict] = []
        self.best_epoch: int | None = None

    # ------------------------------------------------------------------ helpers
    def _loader(self, ds: ValidityDataset, shuffle: bool) -> DataLoader:
        """Batches are shuffled *within* a split only. The split itself is chronological; shuffling
        inside it does not move information across time."""
        tensors = TensorDataset(
            torch.from_numpy(ds.S).float(),
            torch.from_numpy(ds.y_tilde).long(),
            torch.from_numpy(ds.event).long(),
            torch.from_numpy(ds.risk_path).float(),
            torch.from_numpy(ds.eps).float(),
            torch.from_numpy(ds.anchor).float(),
        )
        return DataLoader(tensors, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          drop_last=False)

    def _batch_loss(self, S, y_tilde, event, path, eps, anchor) -> torch.Tensor:
        a = anchor if self.cfg.use_anchor else None
        if self.cfg.variant == "two_head":
            risk, hazard = self.net(S, a)
            loss = survival_nll(hazard, y_tilde, event)
            loss = loss + self.cfg.alpha * risk_path_loss(risk, path)
            loss = loss + self.cfg.beta * monotonicity_penalty(risk, hazard, eps)
            return loss
        mu, sigma = self.net(S, true_path=path, anchor=a)
        return path_nll(mu, sigma, path)

    # ------------------------------------------------------------------ fit
    def fit(self, train: ValidityDataset, cal: ValidityDataset, *, verbose: bool = False
            ) -> "PrismV":
        from ..metrics.survival_metrics import ipcw_integrated_brier

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        loader = self._loader(train, shuffle=True)
        best_score, best_state, bad = np.inf, None, 0

        for epoch in range(self.cfg.max_epochs):
            self.net.train()
            total, n = 0.0, 0
            for S, yt, ev, path, eps, anc in loader:
                S, yt, ev = S.to(self.device), yt.to(self.device), ev.to(self.device)
                path, eps, anc = path.to(self.device), eps.to(self.device), anc.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self._batch_loss(S, yt, ev, path, eps, anc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                opt.step()
                total += float(loss.item()) * S.shape[0]
                n += S.shape[0]

            if self.cfg.early_stop_metric == "risk_mae":
                from ..metrics.risk_path_metrics import horizon_mae
                score = float(np.nanmean(horizon_mae(self.predict_risk_path(cal),
                                                     cal.risk_path)))
            elif self.cfg.early_stop_metric == "ipcw_ibs":
                score = ipcw_integrated_brier(self.predict_survival(cal),
                                              cal.y_tilde, cal.event)
            else:
                raise ValueError(
                    f"unknown early_stop_metric {self.cfg.early_stop_metric!r}; "
                    "expected 'ipcw_ibs' or 'risk_mae'")
            self.history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                                 f"cal_{self.cfg.early_stop_metric}": score})
            if verbose:
                print(f"epoch {epoch:3d}  train={total / max(n, 1):.4f}  cal_IBS={score:.4f}")

            if score < best_score - 1e-5:
                best_score, bad = score, 0
                best_state = copy.deepcopy(self.net.state_dict())
                self.best_epoch = epoch
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict_survival(self, ds: ValidityDataset, eps: np.ndarray | float | None = None
                         ) -> np.ndarray:
        """Survival curves `(N, H)`.

        For `first_passage`, `eps` may be overridden at read-out — one trained model answers for
        any operational threshold (D18). For `two_head` the threshold is baked into the hazard head
        at training time and an override is refused rather than silently ignored.
        """
        self.net.eval()
        if eps is None:
            eps_arr = ds.eps
        else:
            if self.cfg.variant == "two_head":
                raise ValueError(
                    "two_head bakes eps into the hazard head at training time and cannot be "
                    "re-thresholded at read-out; retrain, or use the first_passage variant."
                )
            eps_arr = np.full(len(ds), float(eps)) if np.isscalar(eps) else np.asarray(eps)

        out = []
        for i in range(0, len(ds), self.cfg.batch_size):
            S = torch.from_numpy(ds.S[i:i + self.cfg.batch_size]).float().to(self.device)
            e = torch.from_numpy(np.asarray(eps_arr[i:i + self.cfg.batch_size])).float()
            e = e.to(self.device)
            a = (torch.from_numpy(ds.anchor[i:i + self.cfg.batch_size]).float().to(self.device)
                 if self.cfg.use_anchor else None)
            if self.cfg.variant == "two_head":
                s = self.net.survival(S, anchor=a)
            else:
                s = self.net.survival(S, e, n_samples=self.cfg.n_samples, anchor=a)
            out.append(s.cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def predict_risk_path(self, ds: ValidityDataset) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(ds), self.cfg.batch_size):
            S = torch.from_numpy(ds.S[i:i + self.cfg.batch_size]).float().to(self.device)
            a = (torch.from_numpy(ds.anchor[i:i + self.cfg.batch_size]).float().to(self.device)
                 if self.cfg.use_anchor else None)
            r = (self.net.risk_path(S, anchor=a) if self.cfg.variant == "two_head"
                 else self.net.risk_path(S, n_samples=self.cfg.n_samples, anchor=a))
            out.append(r.cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def sample_paths(self, ds: ValidityDataset, n_samples: int | None = None) -> np.ndarray:
        """Raw sampled paths `(N, M, H)` — first-passage only. Used by the dependence diagnostic."""
        if self.cfg.variant != "first_passage":
            raise ValueError("only the first_passage variant produces sampled paths")
        self.net.eval()
        m = n_samples or self.cfg.n_samples
        out = []
        for i in range(0, len(ds), self.cfg.batch_size):
            S = torch.from_numpy(ds.S[i:i + self.cfg.batch_size]).float().to(self.device)
            out.append(self.net.sample_paths(S, m).cpu().numpy())
        return np.concatenate(out, axis=0)

    def horizon(self, ds: ValidityDataset, eps: np.ndarray | float | None = None,
                alpha: float = 0.1) -> HorizonEstimate:
        surv = self.predict_survival(ds, eps)
        expected = np.array([survival_to_expected_horizon(s) for s in surv])
        median = np.array([median_horizon(s) for s in surv])
        # conservative horizon: last h at which survival still exceeds 1 - alpha
        lower = np.array([int(np.sum(s >= 1.0 - alpha)) for s in surv])
        return HorizonEstimate(expected=expected, median=median, survival=surv, lower_bound=lower)

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict:
        return {"cfg": self.cfg.__dict__, "H": self.H, "d_in": self.d_in,
                "net": self.net.state_dict(), "best_epoch": self.best_epoch,
                "history": self.history}

    def load_state_dict(self, sd: dict) -> "PrismV":
        self.net.load_state_dict(sd["net"])
        self.best_epoch = sd.get("best_epoch")
        self.history = sd.get("history", [])
        return self
