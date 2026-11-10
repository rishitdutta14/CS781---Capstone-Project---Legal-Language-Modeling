"""Validation-only early stopping for bounded Phase 2 runs."""

import math
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 0
    best: float = -1.0
    stale_epochs: int = 0

    def __post_init__(self):
        if self.patience < 0:
            raise ValueError("patience cannot be negative")

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError("validation score must be finite")
        improved = score > self.best
        if improved:
            self.best = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        stop = self.patience > 0 and self.stale_epochs >= self.patience
        return improved, stop
