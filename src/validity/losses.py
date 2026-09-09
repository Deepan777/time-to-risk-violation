"""Training objectives for the two variants.

For `two_head` these are exactly the terms in `proposed_methodology.md` eq. (9)-(10):
the censored discrete-time survival likelihood in person-period form, a Huber risk-path loss on the
logit scale, and the consistency penalty `Omega_mono`.

For `first_passage` there is a single term: the masked Gaussian negative log-likelihood of the
observed path on the logit scale. No `alpha`, no `beta`, no `Omega_mono` — the survival curve is
derived from this distribution rather than fitted alongside it, so there is no second head to keep
consistent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .heads import logit

__all__ = ["survival_nll", "risk_path_loss", "monotonicity_penalty", "path_nll", "LOG_2PI"]

LOG_2PI = 1.8378770664093453
_EPS = 1e-6


def _period_masks(y_tilde: torch.Tensor, event: torch.Tensor, H: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Person-period expansion.

    `at_risk[b, j]` is 1 while origin `b` is still under observation at horizon `j+1`.
    `failed[b, j]` is 1 exactly at the horizon where a violation was observed.
    """
    idx = torch.arange(1, H + 1, device=y_tilde.device).unsqueeze(0)      # (1, H)
    yt = y_tilde.unsqueeze(1)                                            # (B, 1)
    at_risk = (idx <= yt).float()
    failed = ((idx == yt) & (event.unsqueeze(1) == 1)).float()
    return at_risk, failed


def survival_nll(hazard: torch.Tensor, y_tilde: torch.Tensor, event: torch.Tensor
                 ) -> torch.Tensor:
    """Censored discrete-time survival likelihood, eq. (10).

    A censored origin contributes survival through its whole observed span; an uncensored one
    contributes survival up to the window before the event and the hazard at the event itself.
    """
    H = hazard.shape[1]
    at_risk, failed = _period_masks(y_tilde, event, H)
    h = hazard.clamp(_EPS, 1.0 - _EPS)
    ll = failed * torch.log(h) + (at_risk - failed) * torch.log1p(-h)
    return -(ll.sum(dim=1)).mean()


def risk_path_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0,
                   horizon_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Huber loss on the logit scale, eq. (9), masked where the path runs past the stream.

    The logit scale keeps the loss from being dominated by the many windows whose risk sits near
    the middle of [0, 1], and it matches the scale the first-passage model works on, so the two
    variants are compared on like terms.
    """
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    p = logit(pred)
    t = logit(torch.where(mask, target, torch.full_like(target, 0.5)))
    per = F.huber_loss(p, t, reduction="none", delta=delta) * mask.float()
    if horizon_weights is not None:
        per = per * horizon_weights.unsqueeze(0)
    return per.sum() / mask.float().sum().clamp(min=1.0)


def monotonicity_penalty(risk: torch.Tensor, hazard: torch.Tensor, eps: torch.Tensor,
                         sharpness: float = 20.0) -> torch.Tensor:
    """`Omega_mono`: keep the hazard head consistent with the risk head.

    The two heads describe the same event through different parameterisations, and nothing in the
    architecture forces them to agree — the hazard may claim a violation is imminent while the risk
    head predicts a value comfortably below `eps`. This penalises that disagreement.

    Its necessity is the argument for the first-passage variant: a formulation that *derives* the
    hazard from the path has no such inconsistency to penalise, and needs no `beta`.
    """
    implied = torch.sigmoid(sharpness * (risk - eps.view(-1, 1)))
    return F.mse_loss(hazard, implied.detach())


def path_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Masked Gaussian NLL of the observed path on the logit scale — the whole objective for
    `first_passage`.

    Every observed window contributes, including windows after the first violation. The survival
    likelihood by contrast stops contributing at the event, so on censored data this objective
    extracts strictly more supervision from the same stream.
    """
    mask = torch.isfinite(target)
    if not mask.any():
        return mu.sum() * 0.0
    t = logit(torch.where(mask, target, torch.full_like(target, 0.5)).clamp(_EPS, 1.0 - _EPS))
    var = sigma.clamp(min=1e-3) ** 2
    nll = 0.5 * (LOG_2PI + torch.log(var) + (t - mu) ** 2 / var)
    return (nll * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
