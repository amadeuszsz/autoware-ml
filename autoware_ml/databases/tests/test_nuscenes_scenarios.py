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

"""Tests for the nuScenes scenario split building."""

from __future__ import annotations

import pytest
from nuscenes.utils import splits
from pydantic import ValidationError

from autoware_ml.databases.nuscenes.nuscenes_scenarios import NuscenesScenarios
from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.types.dataset import SplitType


def _scenarios(version: str) -> NuscenesScenarios:
    return NuscenesScenarios(
        scenario_root_path="/data/nuscenes",
        dataset_params=[
            DatasetParams(
                dataset_name=version,
                max_sweeps=2,
                sample_steps=1,
                lidar_pointcloud_num_features=5,
            )
        ],
    )


def test_mini_scenarios_follow_the_official_devkit_splits() -> None:
    scenarios = _scenarios("v1.0-mini")

    train_ids = [entry.scenario_id for entry in scenarios.scenario_data[SplitType.TRAIN]]
    val_ids = [entry.scenario_id for entry in scenarios.scenario_data[SplitType.VAL]]

    assert train_ids == list(splits.mini_train)
    assert val_ids == list(splits.mini_val)


def test_the_test_split_reuses_the_annotated_val_scenes() -> None:
    scenarios = _scenarios("v1.0-mini")

    val_ids = [entry.scenario_id for entry in scenarios.scenario_data[SplitType.VAL]]
    test_ids = [entry.scenario_id for entry in scenarios.scenario_data[SplitType.TEST]]

    assert test_ids == val_ids


def test_scenario_entries_carry_the_dataset_parameters() -> None:
    scenarios = _scenarios("v1.0-mini")

    entry = scenarios.scenario_data[SplitType.TRAIN][0]

    assert entry.dataset_name == "v1.0-mini"
    assert entry.scenario_version == "v1.0-mini"
    assert entry.max_sweeps == 2
    assert entry.sample_steps == 1
    assert entry.lidar_pointcloud_num_features == 5
    assert entry.semantic_masks is False


def test_an_unsupported_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unsupported nuScenes version"):
        _scenarios("v2.0-full")
