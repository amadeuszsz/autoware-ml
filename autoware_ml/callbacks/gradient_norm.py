# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient norm logging."""

from __future__ import annotations

import lightning as L
import torch
from torch.optim import Optimizer


def _total_norm(module: torch.nn.Module) -> torch.Tensor | None:
    norms = [p.grad.detach().norm(2) for p in module.parameters() if p.grad is not None]
    return torch.stack(norms).norm(2) if norms else None


class GradientNormMonitor(L.Callback):
    """Log the gradient 2-norm before clipping, in total and per top-level module."""

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        """Log the norms of the accumulated gradients."""
        norms = {"train/grad_norm": _total_norm(pl_module)}
        for name, child in pl_module.named_children():
            norms[f"train/grad_norm_{name}"] = _total_norm(child)
        pl_module.log_dict(
            {key: value for key, value in norms.items() if value is not None},
            on_step=True,
            on_epoch=False,
        )
