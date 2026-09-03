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

"""Tests for the T4 scenario lists."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from autoware_ml.databases.scenarios import DatasetParams
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.types.dataset import SplitType


def _params(**overrides) -> DatasetParams:
    settings = {
        "dataset_name": "db_a",
        "max_sweeps": 2,
        "sample_steps": 5,
        "lidar_pointcloud_num_features": 7,
    }
    settings.update(overrides)
    return DatasetParams(**settings)


def _write_list(tmp_path, entries: dict) -> None:
    (tmp_path / "db_a.yaml").write_text(yaml.safe_dump(entries), encoding="utf-8")


def _scenarios(tmp_path, entries: dict, params: DatasetParams | None = None) -> T4Scenarios:
    # A scenario list names every split, the tests spell out only the ones they fill
    _write_list(tmp_path, {"train": [], "val": [], "test": [], **entries})
    return T4Scenarios(scenario_root_path=tmp_path, dataset_params=[params or _params()])


def test_entries_parse_the_short_and_the_annotated_form(tmp_path) -> None:
    scenarios = _scenarios(
        tmp_path,
        {"train": ["abc/1/Odaiba/J6_x2_Gen2/false"], "val": ["def/0"], "test": []},
    )

    train = scenarios.scenario_data[SplitType.TRAIN][0]
    val = scenarios.scenario_data[SplitType.VAL][0]

    assert (train.scenario_id, train.scenario_version) == ("abc", "1")
    assert (train.location, train.vehicle_type) == ("Odaiba", "J6_x2_Gen2")
    assert (val.scenario_id, val.scenario_version) == ("def", "0")
    assert (val.location, val.vehicle_type) == (None, None)
    assert scenarios.scenario_data[SplitType.TEST] == []


def test_entries_carry_the_dataset_parameters(tmp_path) -> None:
    scenarios = _scenarios(tmp_path, {"train": ["abc/1"]}, _params(semantic_masks=True))

    entry = scenarios.scenario_data[SplitType.TRAIN][0]

    assert entry.dataset_name == "db_a"
    assert entry.max_sweeps == 2
    assert entry.sample_steps == 5
    assert entry.lidar_pointcloud_num_features == 7
    assert entry.semantic_masks is True
    assert scenarios.scenario_data[SplitType.VAL] == []


def test_a_malformed_entry_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError, match="Invalid scenario entry"):
        _scenarios(tmp_path, {"train": ["abc/1/Odaiba"]})


def test_a_missing_scenario_list_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        T4Scenarios(scenario_root_path=tmp_path, dataset_params=[_params()])


def test_a_list_must_name_every_split_with_string_entries(tmp_path) -> None:
    _write_list(tmp_path, {"train": ["abc/1"], "val": []})
    with pytest.raises(ValidationError, match="names no test split"):
        T4Scenarios(scenario_root_path=tmp_path, dataset_params=[_params()])

    _write_list(tmp_path, {"train": ["abc/1"], "val": None, "test": []})
    with pytest.raises(ValidationError, match="must be a list of scenario entries, got NoneType"):
        T4Scenarios(scenario_root_path=tmp_path, dataset_params=[_params()])

    _write_list(tmp_path, {"train": [{"abc": 1}], "val": [], "test": []})
    with pytest.raises(ValidationError, match="non string entry"):
        T4Scenarios(scenario_root_path=tmp_path, dataset_params=[_params()])


def test_a_list_may_carry_metadata_keys_besides_the_splits(tmp_path) -> None:
    scenarios = _scenarios(tmp_path, {"train": ["abc/1"], "version": 3, "amount": [{"total": 9}]})

    assert len(scenarios.scenario_data[SplitType.TRAIN]) == 1


def test_dataset_parameters_are_validated_at_construction() -> None:
    with pytest.raises(ValidationError, match="sample_steps"):
        _params(sample_steps=0)
    with pytest.raises(ValidationError, match="max_sweeps"):
        _params(max_sweeps=-1)
    with pytest.raises(ValidationError, match="lidar_pointcloud_num_features"):
        _params(lidar_pointcloud_num_features=2)


def test_the_string_form_covers_every_parameter(tmp_path) -> None:
    scenarios = _scenarios(tmp_path, {"train": ["abc/1"]}, _params(semantic_masks=True))

    description = str(scenarios)

    assert "semantic_masks=True" in description
    assert "lidar_pointcloud_num_features=7" in description
    assert "scenario_id=abc" in description
