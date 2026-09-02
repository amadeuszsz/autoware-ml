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

"""Tests for the datamodule serving typed batches from fake databases."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.base import DataLoaderConfig, DataModule
from autoware_ml.datamodule.samplers import DistributedWeightedRandomSampler
from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.datamodule.splitters.scenario_splitter import ScenarioSplitter
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.fakes import make_database, make_stored_record
from autoware_ml.testing.factories import make_box3d_data_model

SPLITS = {"train": ["s-train"], "val": ["s-val"], "test": ["s-test"]}


def _records() -> list:
    boxes = [make_box3d_data_model()]
    return [
        make_stored_record(
            scenario_id="s-train", sample_id="train-1", sample_index=1, boxes_3d=boxes
        ),
        make_stored_record(
            scenario_id="s-train", sample_id="train-0", sample_index=0, boxes_3d=boxes
        ),
        make_stored_record(scenario_id="s-val", sample_id="val-0", boxes_3d=boxes),
        make_stored_record(scenario_id="s-test", sample_id="test-0", boxes_3d=boxes),
    ]


def _datamodule(tmp_path, splits=SPLITS, prepare: bool = True, **kwargs) -> DataModule:
    source = {"database": make_database(tmp_path, _records(), splits=splits)}
    datamodule = DataModule(
        dataset=T4Dataset,
        splitter=ScenarioSplitter(),
        train_sources=[source],
        val_sources=[source],
        test_sources=[source],
        **kwargs,
    )
    if prepare:
        datamodule.prepare_data()
    return datamodule


def test_prepare_data_generates_every_database_table_once(tmp_path) -> None:
    datamodule = _datamodule(tmp_path)
    database = datamodule.databases()[0]

    datamodule.prepare_data()

    assert database.cache_file_path.is_file()
    assert database.generate_calls == 1


def test_setup_without_a_generated_table_fails(tmp_path) -> None:
    datamodule = _datamodule(tmp_path, prepare=False)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        datamodule.setup(stage="fit")


def test_setup_none_assigns_every_split_declared_by_the_scenarios(tmp_path) -> None:
    datamodule = _datamodule(tmp_path)

    datamodule.setup(stage=None)

    assert len(datamodule.train_dataset) == 2
    assert len(datamodule.val_dataset) == 1
    assert len(datamodule.test_dataset) == 1
    # The predict split serves the test frames.
    assert len(datamodule.predict_dataset) == 1
    assert datamodule.val_dataset[0].record.sample_id == "val-0"


def test_setup_orders_split_records_by_scenario_and_sample_index(tmp_path) -> None:
    datamodule = _datamodule(tmp_path)

    datamodule.setup(stage="fit")

    dataset = datamodule.train_dataset
    sample_ids = [dataset[index].record.sample_id for index in range(len(dataset))]
    assert sample_ids == ["train-0", "train-1"]


def test_setup_rejects_a_split_the_scenarios_leave_empty(tmp_path) -> None:
    datamodule = _datamodule(tmp_path, splits={"train": ["s-train"]})

    with pytest.raises(ValueError, match="holds no val records"):
        datamodule.setup(stage="validate")


def test_each_split_serves_its_own_sources(tmp_path) -> None:
    train_database = make_database(
        tmp_path, _records()[:2], splits={"train": ["s-train"]}, version="train-db"
    )
    val_database = make_database(
        tmp_path, _records()[2:3], splits={"val": ["s-val"]}, version="val-db"
    )
    datamodule = DataModule(
        dataset=T4Dataset,
        splitter=ScenarioSplitter(),
        train_sources=[{"database": train_database, "repeat": 2}],
        val_sources=[{"database": val_database}],
        test_sources=[{"database": val_database}],
    )
    datamodule.prepare_data()

    datamodule.setup(stage="fit")

    assert len(datamodule.databases()) == 2
    assert len(datamodule.train_dataset) == 4
    assert len(datamodule.val_dataset) == 1
    assert datamodule.val_dataset[0].record.sample_id == "val-0"


def test_setup_rejects_a_factory_that_builds_no_dataset(tmp_path) -> None:
    source = {"database": make_database(tmp_path, _records(), splits=SPLITS)}
    datamodule = DataModule(
        dataset=lambda dataset_transforms: object(),
        splitter=ScenarioSplitter(),
        train_sources=[source],
        val_sources=[source],
        test_sources=[source],
    )
    datamodule.prepare_data()

    with pytest.raises(TypeError, match="must build a Dataset"):
        datamodule.setup(stage="fit")


def test_dataloader_configs_coerce_mappings_and_keep_instances(tmp_path) -> None:
    explicit = DataLoaderConfig(batch_size=4, num_workers=2)
    datamodule = _datamodule(
        tmp_path,
        train_dataloader_cfg=explicit,
        val_dataloader_cfg={"batch_size": 3, "num_workers": 0},
    )

    assert datamodule.train_dataloader_cfg is explicit
    assert isinstance(datamodule.val_dataloader_cfg, DataLoaderConfig)
    assert datamodule.val_dataloader_cfg.batch_size == 3
    assert isinstance(datamodule.test_dataloader_cfg, DataLoaderConfig)

    with pytest.raises(TypeError, match="Expected dataloader config"):
        _datamodule(tmp_path, train_dataloader_cfg=42)


def test_test_dataloader_serves_a_typed_batch(tmp_path) -> None:
    datamodule = _datamodule(tmp_path, test_dataloader_cfg={"batch_size": 1, "num_workers": 0})
    datamodule.setup(stage="test")

    batch = next(iter(datamodule.test_dataloader()))

    assert isinstance(batch, Batch)
    assert batch.batch_size == 1
    assert batch.sample_token == ("test-0",)


def test_train_frame_sampling_installs_the_weighted_sampler(tmp_path) -> None:
    datamodule = _datamodule(
        tmp_path,
        train_dataloader_cfg={"batch_size": 1, "num_workers": 0, "shuffle": True},
        train_frame_sampling={
            "repeat_sampling_factor": 1.0,
            "object_bev_range": [-50.0, -50.0, 50.0, 50.0],
            "low_pedestrian_height_threshold": 1.5,
            "low_pedestrian_bev_range": [-50.0, -50.0, 50.0, 50.0],
            "class_names": ["car"],
            "ignore_label_index": -1,
        },
    )
    datamodule.setup(stage="fit")

    dataloader = datamodule.train_dataloader()

    assert isinstance(dataloader.sampler, DistributedWeightedRandomSampler)
