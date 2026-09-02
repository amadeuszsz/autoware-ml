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

"""Tests for the weighted sampler and the repeat factor frame sampling weights."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samplers import (
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
    compute_frame_sampling_weights,
)
from autoware_ml.datamodule.tests.fakes import make_source
from autoware_ml.testing.factories import make_box3d_data_model, make_record


def _config(**overrides) -> FrameSamplingConfig:
    settings = {
        "repeat_sampling_factor": 1.0,
        "object_bev_range": [-50.0, -50.0, 50.0, 50.0],
        "low_pedestrian_height_threshold": 1.5,
        "low_pedestrian_bev_range": [-50.0, -50.0, 50.0, 50.0],
        "class_names": ["car", "pedestrian"],
        "ignore_label_index": -1,
    }
    settings.update(overrides)
    return FrameSamplingConfig(**settings)


def _car_box(num_lidar_points: int = 10):
    return make_box3d_data_model(
        params=(0.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0, 0.0, 0.0, 0.0),
        label_name="car",
        num_lidar_points=num_lidar_points,
    )


def _records(frame_boxes):
    source = make_source()
    return [
        (make_record(sample_id=f"sample-{index}", boxes_3d=boxes), source)
        for index, boxes in enumerate(frame_boxes)
    ]


class TestDistributedWeightedRandomSampler:
    def test_rejects_a_weight_count_mismatching_the_dataset(self) -> None:
        with pytest.raises(ValueError, match="sampler weights"):
            DistributedWeightedRandomSampler(list(range(4)), [1.0, 1.0])

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DistributedWeightedRandomSampler(list(range(2)), [1.0, -0.1])

    def test_rejects_all_zero_weights(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DistributedWeightedRandomSampler(list(range(2)), [0.0, 0.0])

    def test_never_draws_a_zero_weight_index(self) -> None:
        sampler = DistributedWeightedRandomSampler(list(range(4)), [1.0, 0.0, 1.0, 1.0])

        indices = list(sampler)

        assert len(indices) == 4
        assert 1 not in indices
        assert set(indices) <= {0, 2, 3}

    def test_draws_are_deterministic_per_epoch_and_change_across_epochs(self) -> None:
        sampler = DistributedWeightedRandomSampler(list(range(50)), [1.0] * 50)

        sampler.set_epoch(0)
        first = list(sampler)
        repeat = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)

        assert first == repeat
        assert first != second


class TestComputeFrameSamplingWeights:
    def test_rare_categories_boost_their_frames(self) -> None:
        low_pedestrian = make_box3d_data_model(
            params=(1.0, 1.0, 0.0, 0.6, 0.6, 1.2, 0.0, 0.0, 0.0, 0.0),
            label_name="pedestrian",
            label_index=1,
        )
        records = _records([[_car_box()]] * 5 + [[_car_box(), low_pedestrian]])

        weights = compute_frame_sampling_weights(records, _config())

        assert weights[5] > weights[0]
        assert np.isclose(weights[5], 42**0.25)

    def test_boxes_without_lidar_points_earn_no_boost(self) -> None:
        pedestrian_without_points = make_box3d_data_model(
            params=(1.0, 1.0, 0.0, 0.6, 0.6, 1.2, 0.0, 0.0, 0.0, 0.0),
            label_name="pedestrian",
            label_index=1,
            num_lidar_points=0,
        )
        records = _records(
            [[_car_box(), _car_box()]] * 5 + [[_car_box(), pedestrian_without_points]]
        )

        weights = compute_frame_sampling_weights(
            records, _config(low_pedestrian_height_threshold=0.0)
        )

        assert weights == [1.0] * 6

    def test_filtered_attributes_exclude_boxes_from_category_counting(self) -> None:
        parked_bicycle = make_box3d_data_model(
            params=(1.0, 1.0, 0.0, 1.8, 0.8, 1.2, 0.0, 0.0, 0.0, 0.0),
            label_name="bicycle",
            label_index=1,
            attributes=("vehicle_state.parked",),
        )
        records = _records([[_car_box()], [_car_box(), parked_bicycle]])

        weights = compute_frame_sampling_weights(
            records,
            _config(
                class_names=["car", "bicycle"],
                filter_attributes=[["bicycle", "vehicle_state.parked"]],
            ),
        )

        assert weights == [1.0, 1.0]

    def test_a_class_outside_the_configured_names_is_rejected(self) -> None:
        stranger = make_box3d_data_model(label_name="animal", label_index=7)
        records = _records([[_car_box(), stranger]])

        with pytest.raises(ValueError, match="not one of the configured class names"):
            compute_frame_sampling_weights(records, _config())

    def test_physically_invalid_boxes_contribute_no_category_evidence(self) -> None:
        negative_dimension = make_box3d_data_model(
            params=(1.0, 1.0, 0.0, 0.6, -0.6, 1.2, 0.0, 0.0, 0.0, 0.0),
            label_name="pedestrian",
            label_index=1,
        )
        impossible_speed = make_box3d_data_model(
            params=(2.0, 2.0, 0.0, 0.6, 0.6, 1.7, 0.0, 200.0, 200.0, 0.0),
            label_name="pedestrian",
            label_index=1,
        )
        records = _records([[_car_box()], [negative_dimension, impossible_speed]])

        weights = compute_frame_sampling_weights(records, _config())

        assert weights[1] == 1.0

    def test_a_dataset_without_valid_boxes_is_rejected(self) -> None:
        records = _records([[], []])

        with pytest.raises(ValueError, match="no valid boxes"):
            compute_frame_sampling_weights(records, _config())
