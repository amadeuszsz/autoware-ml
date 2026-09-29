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


"""Tests of the exponential moving average of the weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autoware_ml.callbacks.ema import EMAWeights

WEIGHT = "0.weight"
COUNTER = "1.num_batches_tracked"


def build_module(weight: float) -> torch.nn.Module:
    """A module with one float parameter and one integer buffer."""
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.BatchNorm1d(1))
    with torch.no_grad():
        module[0].weight.fill_(weight)
    return module


def set_weight(module: torch.nn.Module, weight: float) -> None:
    with torch.no_grad():
        module[0].weight.fill_(weight)


def weight_of(module: torch.nn.Module) -> float:
    return module[0].weight.item()


def build_trainer(accumulate_grad_batches: int = 1, is_last_batch: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        accumulate_grad_batches=accumulate_grad_batches, is_last_batch=is_last_batch
    )


def averaged_once() -> tuple[EMAWeights, torch.nn.Module, SimpleNamespace]:
    """A callback that took one averaging step from weight 1.0 towards live weight 3.0."""
    module = build_module(1.0)
    callback = EMAWeights(decay=0.9)
    trainer = build_trainer()
    callback.setup(trainer, module, "fit")
    callback.on_fit_start(trainer, module)
    set_weight(module, 3.0)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    return callback, module, trainer


@pytest.mark.parametrize("decay", [0.0, 1.0, 1.5])
def test_rejects_a_decay_outside_the_open_unit_interval(decay: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        EMAWeights(decay=decay)


def test_the_average_moves_towards_the_live_weights_by_the_effective_decay() -> None:
    callback, _, _ = averaged_once()

    # The first step applies the warm up decay of 0.1, not the configured 0.9
    assert callback.updates == 1
    assert callback.average[WEIGHT].item() == pytest.approx(1.0 + 0.9 * (3.0 - 1.0))


def test_the_effective_decay_grows_towards_the_configured_one() -> None:
    callback = EMAWeights(decay=0.9)

    decays = []
    for updates in (0, 5, 50, 1000):
        callback.updates = updates
        decays.append(callback.effective_decay())

    assert decays == sorted(decays)
    assert decays[-1] == 0.9


def test_buffers_that_are_not_floating_point_are_copied() -> None:
    module = build_module(1.0)
    callback = EMAWeights(decay=0.9)
    trainer = build_trainer()
    callback.setup(trainer, module, "fit")
    callback.on_fit_start(trainer, module)
    module[1].num_batches_tracked.fill_(5)

    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert callback.average[COUNTER].item() == 5
    assert callback.average[COUNTER].dtype == torch.int64


def test_only_a_completed_accumulation_window_updates_the_average() -> None:
    module = build_module(1.0)
    callback = EMAWeights(decay=0.9)
    trainer = build_trainer(accumulate_grad_batches=2)
    callback.setup(trainer, module, "fit")
    callback.on_fit_start(trainer, module)

    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback.updates == 0
    callback.on_train_batch_end(trainer, module, None, None, 1)
    assert callback.updates == 1
    # The epoch ends mid window, Lightning steps the optimizer there as well
    callback.on_train_batch_end(build_trainer(2, is_last_batch=True), module, None, None, 2)
    assert callback.updates == 2


def test_validation_runs_on_the_average_and_hands_the_live_weights_back() -> None:
    callback, module, trainer = averaged_once()

    callback.on_validation_start(trainer, module)
    assert weight_of(module) == pytest.approx(2.8)
    callback.on_validation_end(trainer, module)
    assert weight_of(module) == pytest.approx(3.0)

    callback.on_test_start(trainer, module)
    assert weight_of(module) == pytest.approx(2.8)
    callback.on_test_end(trainer, module)
    assert weight_of(module) == pytest.approx(3.0)


def test_a_checkpoint_stores_the_average_and_carries_the_live_weights() -> None:
    callback, module, trainer = averaged_once()

    checkpoint = {"state_dict": module.state_dict()}
    callback.on_save_checkpoint(trainer, module, checkpoint)
    state = callback.state_dict()

    assert checkpoint["state_dict"][WEIGHT].item() == pytest.approx(2.8)
    assert state["live"][WEIGHT].item() == pytest.approx(3.0)
    assert state["updates"] == 1


def test_a_checkpoint_written_during_validation_carries_the_parked_live_weights() -> None:
    callback, module, trainer = averaged_once()

    callback.on_validation_start(trainer, module)
    checkpoint = {"state_dict": module.state_dict()}
    callback.on_save_checkpoint(trainer, module, checkpoint)
    state = callback.state_dict()
    callback.on_validation_end(trainer, module)

    assert checkpoint["state_dict"][WEIGHT].item() == pytest.approx(2.8)
    assert state["live"][WEIGHT].item() == pytest.approx(3.0)


def test_a_resumed_run_continues_from_the_live_weights() -> None:
    callback, module, trainer = averaged_once()
    checkpoint = {"state_dict": module.state_dict()}
    callback.on_save_checkpoint(trainer, module, checkpoint)
    state = callback.state_dict()

    # Lightning restores the model from the checkpoint, which holds the average
    resumed_module = build_module(0.0)
    resumed_module.load_state_dict(checkpoint["state_dict"])
    resumed = EMAWeights(decay=0.9)
    resumed.load_state_dict(state)
    resumed.setup(trainer, resumed_module, "fit")
    resumed.on_fit_start(trainer, resumed_module)

    assert weight_of(resumed_module) == pytest.approx(3.0)
    assert resumed.average[WEIGHT].item() == pytest.approx(2.8)
    assert resumed.updates == 1


def test_the_hooks_leave_a_run_without_an_average_alone() -> None:
    module = build_module(1.0)
    callback = EMAWeights()
    trainer = build_trainer()
    checkpoint = {"state_dict": module.state_dict()}

    callback.on_validation_start(trainer, module)
    callback.on_validation_end(trainer, module)
    callback.on_save_checkpoint(trainer, module, checkpoint)

    assert weight_of(module) == pytest.approx(1.0)
    assert checkpoint["state_dict"][WEIGHT].item() == pytest.approx(1.0)
    assert callback.state_dict()["live"] is None
