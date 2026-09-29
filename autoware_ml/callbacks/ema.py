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


"""Exponential moving average of the model weights."""

from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Detached copy of every tensor in the state of a module, on the device it lives on.

    Args:
      module: Module whose parameters and buffers are copied.

    Returns:
      dict[str, torch.Tensor]: Copy of the state, keyed like the state dict of the module.
    """

    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class EMAWeights(Callback):
    """
    Keep an exponential moving average of the weights and validate, test and save with it.

    After every optimizer step the average moves towards the live weights by a fraction of
    their difference. The validation and test loops run on the average, and every checkpoint
    stores it as the model weights, so a later stage that loads the checkpoint trains or
    evaluates the averaged model. The live weights travel in the callback state, so a resumed
    run continues from the weights the optimizer left. Buffers that are not floating point,
    such as batch counters, are copied rather than averaged.

    Attributes:
      decay: Fraction of the average kept at every step. The effective decay grows from a
        lower value towards it over the first steps, so the average is not pinned to the
        initial weights.
      updates: Number of averaging steps taken.
    """

    def __init__(self, decay: float = 0.999) -> None:
        """
        Initialize the callback.

        Args:
          decay: Fraction of the average kept at every step, strictly between 0 and 1.

        Raises:
          ValueError: If the decay does not lie strictly between 0 and 1.
        """

        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"The decay must lie strictly between 0 and 1, got {decay}.")
        self.decay = decay
        self.updates = 0
        self._average: dict[str, torch.Tensor] | None = None
        self._live: dict[str, torch.Tensor] | None = None
        self._resumed_live: dict[str, torch.Tensor] | None = None
        self._module: L.LightningModule | None = None

    @property
    def average(self) -> Mapping[str, torch.Tensor] | None:
        """Averaged weights, None before the first training step of the run."""

        return self._average

    def effective_decay(self) -> float:
        """
        Decay applied at the current step.

        Returns:
          float: The configured decay, lowered during the first steps so the average leaves the
            initial weights quickly.
        """

        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the average from the current weights, or hand a resumed run its live weights."""

        if self._resumed_live is not None:
            pl_module.load_state_dict(self._resumed_live)
            self._resumed_live = None
        if self._average is None:
            self._average = clone_state(pl_module)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Move the average towards the live weights once the optimizer has stepped."""

        stepped = (batch_idx + 1) % trainer.accumulate_grad_batches == 0 or trainer.is_last_batch
        if not stepped:
            return
        if self._average is None:
            raise RuntimeError("The average was not initialised before the first training batch.")
        weight = 1.0 - self.effective_decay()
        with torch.no_grad():
            for name, value in pl_module.state_dict().items():
                average = self._average[name]
                if average.dtype.is_floating_point:
                    average.lerp_(value, weight)
                else:
                    average.copy_(value)
        self.updates += 1

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run the validation on the averaged weights."""

        self._swap_in_average(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Hand the live weights back to the optimizer."""

        self._swap_out_average(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run the test on the averaged weights."""

        self._swap_in_average(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Hand the live weights back."""

        self._swap_out_average(pl_module)

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        """Store the average as the model weights of the checkpoint."""

        if self._average is None:
            return
        checkpoint["state_dict"] = {name: value.clone() for name, value in self._average.items()}

    def state_dict(self) -> dict[str, Any]:
        """
        Callback state a checkpoint carries.

        Returns:
          dict[str, Any]: The averaging progress, the average and the live weights.
        """

        return {
            "updates": self.updates,
            "average": self._average,
            "live": self._live if self._live is not None else self._live_snapshot(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the averaging progress of a resumed run.

        Args:
          state_dict: State a checkpoint of this callback stored.
        """

        self.updates = state_dict["updates"]
        self._average = state_dict["average"]
        self._resumed_live = state_dict["live"]

    def _live_snapshot(self) -> dict[str, torch.Tensor] | None:
        """Live weights as they are when the average is not swapped in, None before fitting."""

        if self._module is None:
            return None
        return clone_state(self._module)

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Remember the module whose weights are averaged."""

        self._module = pl_module

    def _swap_in_average(self, pl_module: L.LightningModule) -> None:
        if self._average is None or self._live is not None:
            return
        self._live = clone_state(pl_module)
        pl_module.load_state_dict(self._average)

    def _swap_out_average(self, pl_module: L.LightningModule) -> None:
        if self._live is None:
            return
        pl_module.load_state_dict(self._live)
        self._live = None
