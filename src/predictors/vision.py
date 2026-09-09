"""Small CNN for the 32x32 image streams (Yearbook).

The base predictor is the thing being *monitored*, not the contribution, so it is deliberately
modest: three convolutional blocks and a linear head, trained once on the pre-deployment segment
and then frozen. A ResNet would be defensible too, but on 32x32 inputs and a 6 GiB laptop GPU it
buys accuracy the study does not need and costs wall-clock the study cannot spare.

The wrapper exposes exactly the scikit-learn-shaped surface `PredictorHandle` expects
(`predict`, `predict_proba`, `prismv_embed`, `prismv_raw`), so nothing downstream has to know the
predictor is a neural network.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import PredictorHandle

__all__ = ["SmallCNN", "TorchImageClassifier", "train_vision"]


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 2, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 8
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                # 1
        )
        self.embed_dim = width * 4
        self.head = nn.Linear(self.embed_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class TorchImageClassifier:
    """sklearn-shaped wrapper around `SmallCNN`, frozen after `fit`."""

    def __init__(self, input_shape: tuple[int, int, int], n_classes: int = 2,
                 width: int = 32, epochs: int = 12, batch_size: int = 256, lr: float = 2e-3,
                 seed: int = 0, device: str | None = None):
        self.input_shape = tuple(input_shape)          # (H, W, C) as stored
        self.n_classes = n_classes
        self.width, self.epochs, self.batch_size, self.lr, self.seed = (
            width, epochs, batch_size, lr, seed)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net: SmallCNN | None = None
        self.classes_ = np.arange(n_classes)

    # ---------------------------------------------------------------- internals
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        h, w, c = self.input_shape
        arr = np.asarray(X, dtype=np.float32).reshape(-1, h, w, c)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()   # NCHW

    @torch.no_grad()
    def _forward_all(self, X: np.ndarray, what: str) -> np.ndarray:
        assert self.net is not None, "predictor used before fit"
        self.net.eval()
        outs = []
        for i in range(0, len(X), self.batch_size):
            xb = self._to_tensor(X[i:i + self.batch_size]).to(self.device)
            z = self.net.embed(xb)
            outs.append((z if what == "embed" else self.net.head(z)).cpu().numpy())
        return np.concatenate(outs, axis=0)

    # ---------------------------------------------------------------- sklearn surface
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchImageClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        h, w, c = self.input_shape
        self.net = SmallCNN(in_ch=c, n_classes=self.n_classes, width=self.width).to(self.device)

        Xt = self._to_tensor(X)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        n = len(yt)
        g = torch.Generator().manual_seed(self.seed)

        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = Xt[idx].to(self.device), yt[idx].to(self.device)
                opt.zero_grad(set_to_none=True)
                lossf(self.net(xb), yb).backward()
                opt.step()
        self.net.eval()
        for p in self.net.parameters():        # frozen from here on
            p.requires_grad_(False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward_all(X, "logits")
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward_all(X, "logits").argmax(axis=1)

    def prismv_embed(self, X: np.ndarray) -> np.ndarray:
        """Genuine penultimate activations — unlike the tree families, this is a real embedding."""
        return self._forward_all(X, "embed")

    def prismv_raw(self, X: np.ndarray) -> np.ndarray:
        return self._forward_all(X, "logits")


def train_vision(stream, *, seed: int, epochs: int = 12, width: int = 32,
                 hyperparameters: dict | None = None) -> PredictorHandle:
    """Fit the CNN on `stream` (the pre-deployment segment) and return a frozen handle."""
    hp = dict(hyperparameters or {})
    shape = tuple(stream.meta.get("input_shape", ()))
    if len(shape) != 3:
        raise ValueError(
            f"vision predictor needs meta['input_shape'] = (H, W, C); got {shape!r}")

    y = stream.y()
    n_classes = int(np.max(y)) + 1
    model = TorchImageClassifier(input_shape=shape, n_classes=n_classes, width=width,
                                 epochs=hp.get("epochs", epochs), seed=seed,
                                 batch_size=hp.get("batch_size", 256),
                                 lr=hp.get("lr", 2e-3))
    model.fit(stream.X(), y)

    return PredictorHandle(
        model=model,
        predictor_id=f"cnn_seed{seed}",
        family="cnn",
        task=stream.task,
        feature_cols=list(stream.feature_cols),
        seed=seed,
        hyperparameters={"width": width, "epochs": model.epochs, "lr": model.lr,
                         "batch_size": model.batch_size, "input_shape": list(shape),
                         "n_classes": n_classes},
        classes_=model.classes_,
    )
