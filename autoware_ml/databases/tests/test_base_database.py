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

"""Tests for the database hash, the record table cache and its round trip."""

from __future__ import annotations

import pytest

from autoware_ml.databases.base_database import run_record_workers
from autoware_ml.datamodule.tests.fakes import (
    FakeDatabase,
    make_database,
    make_scenarios,
    make_stored_record,
)


def _records_of(scenario_id: str):
    if scenario_id == "broken":
        raise ValueError(f"scenario {scenario_id} is broken")
    return [make_stored_record(scenario_id=scenario_id, sample_id=f"{scenario_id}-0")]


def _records():
    return [
        make_stored_record(scenario_id="a", sample_id="a-0"),
        make_stored_record(scenario_id="b", sample_id="b-0"),
    ]


def test_equal_definitions_share_the_hash(tmp_path) -> None:
    first = make_database(tmp_path / "first", _records(), splits={"train": ["a", "b"]})
    second = make_database(tmp_path / "second", _records(), splits={"train": ["a", "b"]})

    # The root and cache paths enter the hash, so equal definitions need equal paths.
    assert str(first).replace("first", "x") == str(second).replace("second", "x")
    assert first.database_hash != second.database_hash
    assert make_database(tmp_path / "first", [], splits={"train": ["a", "b"]}) == first


def test_scenarios_and_taxonomy_change_the_hash(tmp_path) -> None:
    base = make_database(tmp_path, _records(), splits={"train": ["a", "b"]})
    other_splits = make_database(tmp_path, _records(), splits={"train": ["a"], "val": ["b"]})
    other_classes = make_database(
        tmp_path, _records(), splits={"train": ["a", "b"]}, class_names=("car", "truck")
    )

    assert base.database_hash != other_splits.database_hash
    assert base.database_hash != other_classes.database_hash


def test_cache_file_is_named_after_the_prefix_and_the_hash(tmp_path) -> None:
    database = make_database(tmp_path, _records())

    assert database.cache_file_path == (
        tmp_path / "cache" / f"database_{database.database_hash}.parquet"
    )


def test_process_writes_the_table_once_and_loading_round_trips(tmp_path) -> None:
    database = make_database(tmp_path, _records())

    database.process_scenario_records()
    database.process_scenario_records()

    assert database.generate_calls == 1
    assert database.cache_file_path.is_file()
    # The table is written next to its final name and renamed, nothing partial stays behind
    assert sorted(path.name for path in database.cache_file_path.parent.iterdir()) == [
        database.cache_file_path.name
    ]
    frame = database.load_polars_scenario_dataframe()
    assert frame.height == 2
    assert [record.sample_id for record in database.load_scenario_records()] == ["a-0", "b-0"]


def test_loading_an_absent_table_fails(tmp_path) -> None:
    database = make_database(tmp_path, _records())

    with pytest.raises(FileNotFoundError, match="does not exist"):
        database.load_polars_scenario_dataframe()


def test_a_database_without_records_is_rejected(tmp_path) -> None:
    database = make_database(tmp_path, [], splits={"train": ["a"]})

    with pytest.raises(ValueError, match="produced no records"):
        database.process_scenario_records()


def test_unique_scenario_data_merges_the_groups(tmp_path) -> None:
    database = FakeDatabase(
        records=_records(),
        scenarios={
            "first": make_scenarios(tmp_path / "first", {"train": ["a", "b"]}),
            "second": make_scenarios(tmp_path / "second", {"val": ["b"], "test": ["c"]}),
        },
        root_path=tmp_path,
        cache_path=tmp_path / "cache",
    )

    unique = database.get_unique_scenario_data()

    assert list(unique) == ["a", "b", "c"]


def test_invalid_definitions_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one scenario group"):
        FakeDatabase(records=[], scenarios={}, root_path=tmp_path, cache_path=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        make_database(tmp_path, [], class_names=())


def test_record_workers_keep_the_parameter_order_across_processes() -> None:
    records = run_record_workers(_records_of, ["c", "a", "b"], num_workers=2)

    assert [record.scenario_id for record in records] == ["c", "a", "b"]


def test_record_workers_raise_the_first_failure() -> None:
    with pytest.raises(ValueError, match="scenario broken is broken"):
        run_record_workers(_records_of, ["a", "broken", "b"], num_workers=2)
