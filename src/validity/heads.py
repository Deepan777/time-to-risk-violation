"""The two model variants compared head to head by experiment E15 (decision D17).

`TwoHeadModel` is the design exactly as specified in `proposed_methodology.md` section 5: a point
risk head and a *separately parameterised* discrete hazard head sharing an encoder, trained with
`L_surv + alpha*L_risk + beta*Omega_mono`.

`FirstPassageModel` is the alternative. The hazard is not an independent quantity — it is
determined by the risk path:

    T_t = min{ h : R_{t+h} > eps }      =>      S(h) = P( max_{j<=h} R_{t+j} <= eps )

So instead of parameterising the hazard, learn the **joint** predictive distribution over the path
and read the survival curve off it as a first-passage probability. The joint matters: for a
dependent path, `P(max_j R_j <= eps) != prod_j P(R_j <= eps)`, so marginal per-horizon forecasts
cannot produce a correct survival curve. That gap is the substantive difference between the two
variants, and E15 measures it.

Three consequences, all of which E15 should show:

* `alpha`, `beta` and `Omega_mono` disappear. Monotonicity is automatic, because a maximum over a
  longer window cannot decrease.
* `eps` moves to read-out. One trained model serves every operational threshold (D18).
* The path model trains on the **whole observed risk path**, whereas the hazard likelihood
  contributes information only up to the event. On heavily censored streams the first-passage model
  therefore sees strictly more supervision from the same data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["TwoHeadModel", "FirstPassageModel", "logit", "sigmoid_safe"]

_EPS = 1e-6


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(_EPS, 1.0 - _EPS)
    return torch.log(p) - torch.log1p(-p)


def sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x).clamp(_EPS, 1.0 - _EPS)


# ------------------------------------------------------------------ variant A: as specified

class TwoHeadModel(nn.Module):
    """Encoder + horizon embedding + two independent heads. The specified design (baseline)."""

    variant = "two_head"

    def __init__(self, encoder: nn.Module, H: int, horizon_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        self.H = H
        self.horizon = nn.Embedding(H, horizon_dim)
        d = encoder.out_dim + horizon_dim
        self.risk_head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.hazard_head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _joint(self, S: torch.Tensor) -> torch.Tensor:
        z = self.encoder(S)                                    # (B, dz)
        e = self.horizon.weight.unsqueeze(0).expand(z.shape[0], -1, -1)   # (B, H, de)
        zz = z.unsqueeze(1).expand(-1, self.H, -1)             # (B, H, dz)
        return torch.cat([zz, e], dim=-1)                      # (B, H, dz+de)

    def forward(self, S: torch.Tensor, anchor: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        j = self._joint(S)
        raw = self.risk_head(j).squeeze(-1)                    # (B, H)
        if anchor is None:
            risk = sigmoid_safe(raw)
        else:
            # residual form (E16): the network predicts a CORRECTION to the online level, so a
            # correction of zero reproduces the anchor exactly and the state can only help or hurt
            # relative to it. This is the direct test of "does deployment state add anything on top
            # of the best simple extrapolator", which is the mechanism Gate 1 identified.
            risk = sigmoid_safe(logit(anchor) + raw)
        hazard = sigmoid_safe(self.hazard_head(j).squeeze(-1))  # (B, H)
        return risk, hazard

    @torch.no_grad()
    def survival(self, S: torch.Tensor, eps: torch.Tensor | None = None,
                 anchor: torch.Tensor | None = None) -> torch.Tensor:
        """S(h) = prod_{j<=h} (1 - lambda(j)). `eps` is ignored: this variant bakes the threshold
        into the hazard head at training time and cannot be re-thresholded afterwards."""
        _, hazard = self.forward(S, anchor)
        return torch.cumprod(1.0 - hazard, dim=1)

    @torch.no_grad()
    def risk_path(self, S: torch.Tensor, anchor: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(S, anchor)[0]


# ------------------------------------------------------------------ variant B: first passage

class FirstPassageModel(nn.Module):
    """Autoregressive joint path model; the survival curve is a derived first-passage probability.

    The path is modelled on the logit scale, where an unbounded Gaussian is a sensible family and
    the implied risk stays in (0, 1) by construction:

        logit(R_{t+h}) | R_{t+1:h-1}, S_{1:t}  ~  Normal(mu_h, sigma_h)

    A GRU cell carries the dependence forward, initialised from the encoder output, so the model
    represents genuine cross-horizon structure rather than H independent marginals.
    """

    variant = "first_passage"

    def __init__(self, encoder: nn.Module, H: int, horizon_dim: int = 16, hidden: int = 64,
                 min_sigma: float = 0.05, max_sigma: float = 3.0):
        super().__init__()
        self.encoder = encoder
        self.H = H
        self.hidden = hidden
        self.min_sigma, self.max_sigma = min_sigma, max_sigma

        self.init_h = nn.Linear(encoder.out_dim, hidden)
        self.horizon = nn.Embedding(H, horizon_dim)
        # step input: previous risk on the logit scale, plus the horizon embedding
        self.cell = nn.GRUCell(1 + horizon_dim, hidden)
        self.out = nn.Linear(hidden, 2)                        # -> (mu, raw_sigma)

    def _sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(raw)

    def forward(self, S: torch.Tensor, true_path: torch.Tensor | None = None,
                anchor: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced parameters of the conditional path distribution.

        Returns `(mu, sigma)`, each `(B, H)`, where step `h` is conditioned on the *observed*
        path up to `h-1` when `true_path` is given.
        """
        B = S.shape[0]
        h_state = torch.tanh(self.init_h(self.encoder(S)))
        prev = torch.zeros(B, 1, device=S.device)

        mus, sigmas = [], []
        for h in range(self.H):
            e = self.horizon.weight[h].unsqueeze(0).expand(B, -1)
            h_state = self.cell(torch.cat([prev, e], dim=-1), h_state)
            params = self.out(h_state)
            mu, sigma = params[:, 0], self._sigma(params[:, 1])
            if anchor is not None:                 # residual form (E16)
                mu = mu + logit(anchor[:, h])
            mus.append(mu)
            sigmas.append(sigma)

            if true_path is not None:
                obs = true_path[:, h]
                # a NaN beyond the end of the stream carries no information; feed the model's own
                # mean there so the recurrence stays defined without inventing an observation
                nxt = torch.where(torch.isnan(obs), mu.detach(), logit(obs.clamp(_EPS, 1 - _EPS)))
                prev = nxt.unsqueeze(-1)
            else:
                prev = mu.detach().unsqueeze(-1)

        return torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)

    @torch.no_grad()
    def sample_paths(self, S: torch.Tensor, n_samples: int = 256,
                     anchor: torch.Tensor | None = None) -> torch.Tensor:
        """Ancestral sampling of `n_samples` full risk paths. Returns `(B, n_samples, H)` in [0,1].

        Sampling — rather than composing marginals — is what makes the first-passage probability
        respect the path's own autocorrelation.
        """
        B = S.shape[0]
        z = self.encoder(S)
        h_state = torch.tanh(self.init_h(z)).repeat_interleave(n_samples, dim=0)
        prev = torch.zeros(B * n_samples, 1, device=S.device)

        out = []
        for h in range(self.H):
            e = self.horizon.weight[h].unsqueeze(0).expand(B * n_samples, -1)
            h_state = self.cell(torch.cat([prev, e], dim=-1), h_state)
            params = self.out(h_state)
            mu, sigma = params[:, 0], self._sigma(params[:, 1])
            if anchor is not None:
                mu = mu + logit(anchor[:, h]).repeat_interleave(n_samples, dim=0)
            draw = mu + sigma * torch.randn_like(mu)
            out.append(torch.sigmoid(draw))
            prev = draw.unsqueeze(-1)

        return torch.stack(out, dim=1).view(B, n_samples, self.H)

    @torch.no_grad()
    def survival(self, S: torch.Tensor, eps: torch.Tensor, n_samples: int = 256,
                 anchor: torch.Tensor | None = None) -> torch.Tensor:
        """S(h) = P( max_{j<=h} R_{t+j} <= eps ), estimated from sampled paths.

        Monotone non-increasing by construction: the running maximum cannot fall, so the surviving
        fraction cannot rise. No monotonicity penalty is needed or used.
        """
        paths = self.sample_paths(S, n_samples, anchor)         # (B, M, H)
        e = eps.view(-1, 1, 1)
        alive = (paths.cummax(dim=2).values <= e).float()       # (B, M, H)
        return alive.mean(dim=1)                                # (B, H)

    @torch.no_grad()
    def risk_path(self, S: torch.Tensor, n_samples: int = 256,
                  anchor: torch.Tensor | None = None) -> torch.Tensor:
        """Posterior mean risk path, for comparability with the two-head variant's point head."""
        return self.sample_paths(S, n_samples, anchor).mean(dim=1)
