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

"""Tests for the datamodule serving typed batches from stubbed databases."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.base import DataLoaderConfig, DataModule
from autoware_ml.datamodule.samplers import DistributedWeightedRandomSampler
from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.fakes import make_stored_record, write_record_table
from autoware_ml.testing.factories import make_box3d_data_model


def _records(splits: tuple[str, ...] = ("train", "val", "test")) -> list:
    boxes = [make_box3d_data_model()]
    records = [
        make_stored_record(
            scenario_id="s-train",
            sample_id="train-1",
            sample_index=1,
            split="train",
            boxes_3d=boxes,
        ),
        make_stored_record(
            scenario_id="s-train",
            sample_id="train-0",
            sample_index=0,
            split="train",
            boxes_3d=boxes,
        ),
        make_stored_record(
            scenario_id="s-val", sample_id="val-0", split="val", boxes_3d=boxes
        ),
        make_stored_record(
            scenario_id="s-test", sample_id="test-0", split="test", boxes_3d=boxes
        ),
    ]
    return [record for record in records if record.split in splits]


def _datamodule(tmp_path, splits: tuple[str, ...] = ("train", "val", "test"), **kwargs):
    table = write_record_table(tmp_path, _records(splits))
    return DataModule(
        dataset=T4Dataset,
        sources=[{"records": table}],
        **kwargs,
    )


def test_setup_none_assigns_every_split_declared_by_the_table(tmp_path) -> None:
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


def test_setup_rejects_a_stage_whose_split_the_table_lacks(tmp_path) -> None:
    datamodule = _datamodule(tmp_path, splits=("train",))

    with pytest.raises(ValueError, match="holds no val records"):
        datamodule.setup(stage="validate")


def test_setup_rejects_a_factory_that_builds_no_dataset(tmp_path) -> None:
    datamodule = DataModule(
        dataset=lambda dataset_transforms: object(),
        sources=[{"records": write_record_table(tmp_path, _records())}],
    )

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
            "name_mapping": {"car": "car"},
        },
    )
    datamodule.setup(stage="fit")

    dataloader = datamodule.train_dataloader()

    assert isinstance(dataloader.sampler, DistributedWeightedRandomSampler)
