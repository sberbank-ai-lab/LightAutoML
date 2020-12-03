"""
Pooling
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from log_calls import record_history


@record_history(enabled=False)
class SequenceAbstractPooler(nn.Module, ABC):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@record_history(enabled=False)
class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        return x[..., 0, :]


@record_history(enabled=False)
class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.data.masked_fill(~x_mask.data, -float("inf"))
        values, _ = torch.max(x, dim=-2)
        return values


@record_history(enabled=False)
class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.data.masked_fill(~x_mask.data, 0)
        values = torch.sum(x, dim=-2)
        return values


@record_history(enabled=False)
class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.data.masked_fill(~x_mask.data, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.data.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active
        return values


@record_history(enabled=False)
class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        return x
