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

"""Tests for the preprocessing pipeline wrapper and the processed batch resolution."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.preprocessing.base import DataPreprocessing, ModelInputs, ProcessedBatch
from autoware_ml.testing.factories import make_point_cloud, make_sample


class _StageInputs(ModelInputs):
    """Derived inputs of the recorder stage."""

    stage_ran: bool


class _ShadowInputs(ModelInputs):
    """Derived inputs that reuse the name of a batch property."""

    points: tuple[str, ...]


class _ModeRecorder:
    """Pipeline stage recording the mode it was called with."""

    def __init__(self) -> None:
        self.seen_modes: list[bool] = []

    def __call__(self, batch: Batch, *, is_training: bool) -> _StageInputs:
        self.seen_modes.append(is_training)
        return _StageInputs(stage_ran=True)


def _batch() -> Batch:
    return Batch.collate([make_sample(points=make_point_cloud(num_points=3))])


def test_call_forwards_is_training_to_every_layer() -> None:
    first, second = _ModeRecorder(), _ModeRecorder()
    pipeline = DataPreprocessing([first, second])

    pipeline(_batch(), is_training=True)
    pipeline(_batch(), is_training=False)

    assert first.seen_modes == [True, False]
    assert second.seen_modes == [True, False]


def test_call_requires_explicit_is_training() -> None:
    pipeline = DataPreprocessing([_ModeRecorder()])

    with pytest.raises(TypeError):
        pipeline(_batch())


def test_call_rejects_inputs_that_are_no_batch() -> None:
    pipeline = DataPreprocessing([_ModeRecorder()])

    with pytest.raises(TypeError, match="expects a Batch"):
        pipeline({"points": []}, is_training=True)


def test_call_rejects_layers_that_return_no_model_inputs() -> None:
    pipeline = DataPreprocessing([lambda batch, is_training: {"stage_ran": True}])

    with pytest.raises(TypeError, match="must return ModelInputs"):
        pipeline(_batch(), is_training=True)


def test_call_wraps_the_batch_with_one_input_entry_per_layer() -> None:
    batch = _batch()
    pipeline = DataPreprocessing([_ModeRecorder(), _ModeRecorder()])

    processed = pipeline(batch, is_training=True)

    assert isinstance(processed, ProcessedBatch)
    assert processed.batch is batch
    assert len(processed.inputs) == 2
    assert processed.resolve("stage_ran") is True


def test_resolve_prefers_derived_inputs_over_batch_properties() -> None:
    batch = _batch()
    processed = ProcessedBatch(batch=batch, inputs=(_ShadowInputs(points=("derived",)),))

    assert processed.resolve("points") == ("derived",)


def test_resolve_falls_back_to_the_flat_batch_properties() -> None:
    batch = _batch()
    processed = ProcessedBatch(batch=batch)

    assert torch.equal(processed.resolve("points")[0], batch.points[0])
    assert processed.resolve("sample_token") == batch.sample_token


def test_resolve_rejects_unknown_names_listing_the_derived_inputs() -> None:
    processed = ProcessedBatch(batch=_batch(), inputs=(_StageInputs(stage_ran=True),))

    with pytest.raises(AttributeError, match="stage_ran"):
        processed.resolve("voxels")


def test_resolve_rejects_absent_optional_batch_fields() -> None:
    processed = ProcessedBatch(batch=_batch())

    with pytest.raises(ValueError, match="not available on this batch"):
        processed.resolve("gt_boxes")


def test_has_reports_resolvable_names_only() -> None:
    processed = ProcessedBatch(batch=_batch(), inputs=(_StageInputs(stage_ran=True),))

    assert processed.has("stage_ran")
    assert processed.has("points")
    assert not processed.has("gt_boxes")
    assert not processed.has("voxels")
