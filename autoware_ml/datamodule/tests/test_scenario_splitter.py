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

"""Tests for the scenario splitter assigning records to splits."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.splitters.scenario_splitter import ScenarioSplitter
from autoware_ml.datamodule.tests.fakes import make_scenarios, make_stored_record, records_dataframe
from autoware_ml.types.dataset import SplitType


def _frame():
    return records_dataframe(
        [
            make_stored_record(scenario_id="a", sample_id="a-0"),
            make_stored_record(scenario_id="a", sample_id="a-1", sample_index=1),
            make_stored_record(scenario_id="b", sample_id="b-0"),
            make_stored_record(scenario_id="c", sample_id="c-0"),
        ]
    )


def test_records_follow_the_split_of_their_scenario(tmp_path) -> None:
    scenarios = {"group": make_scenarios(tmp_path, {"train": ["a"], "val": ["b"], "test": []})}

    frames = ScenarioSplitter().split_by_polars_dataframe(_frame(), scenarios)

    assert sorted(frames[SplitType.TRAIN]["sample_id"].to_list()) == ["a-0", "a-1"]
    assert frames[SplitType.VAL]["sample_id"].to_list() == ["b-0"]
    assert frames[SplitType.TEST].is_empty()


def test_scenario_groups_are_merged_per_split(tmp_path) -> None:
    scenarios = {
        "first": make_scenarios(tmp_path / "first", {"train": ["a"]}),
        "second": make_scenarios(tmp_path / "second", {"train": ["c"], "val": ["b"]}),
    }

    frames = ScenarioSplitter().split_by_polars_dataframe(_frame(), scenarios)

    assert sorted(frames[SplitType.TRAIN]["sample_id"].to_list()) == ["a-0", "a-1", "c-0"]
    assert frames[SplitType.VAL]["sample_id"].to_list() == ["b-0"]


def test_a_scenario_listed_in_two_splits_is_rejected(tmp_path) -> None:
    scenarios = {"group": make_scenarios(tmp_path, {"train": ["a"], "val": ["a"]})}

    with pytest.raises(ValueError, match="listed in both"):
        ScenarioSplitter().split_by_polars_dataframe(_frame(), scenarios)
