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

"""Tests for the record driven dataset and the family specific frame metadata."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.base import SourceRecords
from autoware_ml.datamodule.nuscenes.dataset import NuscenesDataset
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.fakes import make_source, make_stored_record, records_dataframe
from autoware_ml.testing.factories import (
    make_box3d_data_model,
    make_camera_frame,
    make_lidar_frame,
    make_record,
)


def _source_records(records, data_root="/data", **source_kwargs) -> SourceRecords:
    return SourceRecords(
        source=make_source(**source_kwargs),
        records=records_dataframe(records),
        data_root=data_root,
    )


def _boxed_record(scenario_id: str, sample_id: str):
    return make_stored_record(
        scenario_id=scenario_id, sample_id=sample_id, boxes_3d=[make_box3d_data_model()]
    )


def test_index_map_walks_sources_in_order_and_repeats_rows() -> None:
    first = _source_records(
        [_boxed_record("s0", "sample-0"), _boxed_record("s0", "sample-1")], repeat=2
    )
    second = _source_records([_boxed_record("s1", "sample-2")], data_root="/other")
    dataset = T4Dataset()

    dataset.assign_source_records([first, second])

    assert len(dataset) == 5
    sample_ids = [dataset.load_record(index)[0].sample_id for index in range(len(dataset))]
    assert sample_ids == ["sample-0", "sample-1", "sample-0", "sample-1", "sample-2"]
    assert dataset.load_record(4)[1].data_root == "/other"


def test_source_without_det3d_supervision_empties_the_boxes() -> None:
    dataset = T4Dataset()
    dataset.assign_source_records([_source_records([_boxed_record("s0", "sample-0")], det3d=False)])

    record, _ = dataset.load_record(0)

    assert record.boxes_3d == []
    assert record.category_mapping is not None


def test_source_without_seg3d_supervision_drops_the_category_mapping() -> None:
    dataset = T4Dataset()
    dataset.assign_source_records([_source_records([_boxed_record("s0", "sample-0")], seg3d=False)])

    record, _ = dataset.load_record(0)

    assert record.category_mapping is None
    assert len(record.boxes_3d) == 1


def test_fully_supervised_source_keeps_boxes_and_category_mapping() -> None:
    dataset = T4Dataset()
    dataset.assign_source_records([_source_records([_boxed_record("s0", "sample-0")])])

    record, _ = dataset.load_record(0)

    assert len(record.boxes_3d) == 1
    assert record.category_mapping.category_names == ["car"]


def test_assigning_no_records_raises() -> None:
    dataset = T4Dataset()

    with pytest.raises(ValueError, match="received no records"):
        dataset.assign_source_records([])


def test_getitem_without_transforms_returns_the_seed_sample() -> None:
    dataset = T4Dataset()
    dataset.assign_source_records([_source_records([_boxed_record("s0", "sample-0")])])

    sample = dataset[0]

    assert isinstance(sample, Sample)
    assert sample.record.sample_id == "sample-0"
    assert sample.data_root == "/data"
    assert sample.points is None
    assert sample.calibration is None


def test_calibration_cameras_expand_every_record_into_one_sample_per_camera() -> None:
    record = make_stored_record(
        camera_frames=[
            make_camera_frame(channel_name="CAM_FRONT"),
            make_camera_frame(channel_name="CAM_LEFT", image_path="db/scene/0/data/cam/1.jpg"),
        ]
    )
    dataset = T4Dataset(calibration_cameras=("CAM_FRONT", "CAM_LEFT"))
    dataset.assign_source_records([_source_records([record])])

    assert len(dataset) == 2
    first = dataset.build_seed_sample(0)
    second = dataset.build_seed_sample(1)
    assert first.calibration.camera_name == "CAM_FRONT"
    assert second.calibration.camera_name == "CAM_LEFT"
    # Identity poses everywhere compose into an identity lidar to camera transformation.
    assert np.allclose(first.calibration.data.lidar_to_camera_transformation, np.eye(4))


def test_calibration_requires_exactly_one_frame_of_the_requested_camera() -> None:
    record = make_stored_record(camera_frames=[make_camera_frame(channel_name="CAM_FRONT")])
    dataset = T4Dataset(calibration_cameras=("CAM_BACK",))
    dataset.assign_source_records([_source_records([record])])

    with pytest.raises(ValueError, match="exactly one camera frame"):
        dataset.build_seed_sample(0)


def test_t4_meta_derives_the_scene_token_from_the_keyframe_path() -> None:
    record = make_record(
        lidar_frames=[make_lidar_frame(pointcloud_path="db/13cabeac/2/data/LIDAR_CONCAT/0.bin")]
    )

    meta = T4Dataset().build_meta(record)

    assert meta.sample_id == "sample-0"
    assert meta.scene_token == "db/13cabeac/2"
    assert meta.timestamp_seconds == 100.0
    assert np.array_equal(meta.ego2global, np.eye(4))
    assert meta.prev_exists is False


def test_t4_meta_rejects_a_lidar_path_too_shallow_for_a_scene_token() -> None:
    record = make_record(lidar_frames=[make_lidar_frame(pointcloud_path="lidar/0.bin")])

    with pytest.raises(ValueError, match="Cannot derive a scene directory"):
        T4Dataset().build_meta(record)


def test_t4_meta_marks_prev_exists_from_the_sample_index() -> None:
    record = make_record().model_copy(update={"sample_index": 3})

    meta = T4Dataset().build_meta(record)

    assert meta.prev_exists is True


def test_nuscenes_meta_uses_the_scenario_id_as_scene_token() -> None:
    record = make_record(scenario_id="scene-0061")

    meta = NuscenesDataset().build_meta(record)

    assert meta.scene_token == "scene-0061"
    assert meta.sample_id == "sample-0"
    assert meta.prev_exists is False
