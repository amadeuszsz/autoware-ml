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

"""Tests for the dataset record serialization round trip."""

from __future__ import annotations

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.camera_frames import CameraFrameDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.testing.factories import (
    make_box3d_data_model,
    make_camera_frame,
    make_lidar_frame,
    make_record,
)


def _full_record() -> DatasetRecord:
    ego_to_global = np.eye(4)
    ego_to_global[:3, 3] = [10.0, -5.0, 0.5]
    return make_record(
        lidar_frames=[
            make_lidar_frame(ego_to_global=ego_to_global),
            make_lidar_frame(
                frame_id="sweep-0",
                keyframe=False,
                timestamp_seconds=99.9,
                pointcloud_path="db/scene/0/data/lidar/sweep.bin",
            ),
        ],
        camera_frames=[
            make_camera_frame(distortion_coefficients=(0.1, -0.2, 0.0, 0.0, 0.05)),
            make_camera_frame(channel_name="CAM_BACK", image_path="db/scene/0/data/cam_back/0.jpg"),
        ],
        boxes_3d=[
            make_box3d_data_model(attributes=("vehicle_state.moving",)),
            make_box3d_data_model(
                params=(3.0, 1.0, 0.5, 0.6, 0.6, 1.7, 0.2, 0.5, -0.5, 0.0),
                label_name="pedestrian",
                instance_id="instance-1",
            ),
        ],
        category_names=("car", "pedestrian"),
        category_indices=(0, 1),
    )


def _assert_record_matches(loaded: DatasetRecord, record: DatasetRecord) -> None:
    assert loaded.scenario_id == record.scenario_id
    assert loaded.sample_id == record.sample_id
    assert loaded.sample_index == record.sample_index
    assert loaded.timestamp_seconds == record.timestamp_seconds
    assert loaded.scenario_name == record.scenario_name

    assert len(loaded.lidar_frames) == 2
    for loaded_frame, frame in zip(loaded.lidar_frames, record.lidar_frames):
        assert loaded_frame.lidar_frame_id == frame.lidar_frame_id
        assert loaded_frame.lidar_keyframe == frame.lidar_keyframe
        assert loaded_frame.lidar_pointcloud_path == frame.lidar_pointcloud_path
        assert np.allclose(
            loaded_frame.lidar_frame_ego_pose_to_global_matrix,
            frame.lidar_frame_ego_pose_to_global_matrix,
        )

    assert len(loaded.camera_frames) == 2
    for loaded_frame, frame in zip(loaded.camera_frames, record.camera_frames):
        assert loaded_frame.camera_sensor_channel_name == frame.camera_sensor_channel_name
        assert loaded_frame.camera_image_path == frame.camera_image_path
        assert loaded_frame.camera_image_width == frame.camera_image_width
        assert loaded_frame.camera_image_height == frame.camera_image_height
        assert np.allclose(loaded_frame.camera_intrinsic_matrix, frame.camera_intrinsic_matrix)
        assert list(loaded_frame.camera_distortion_coefficients) == list(
            frame.camera_distortion_coefficients
        )

    assert len(loaded.boxes_3d) == 2
    for loaded_box, box in zip(loaded.boxes_3d, record.boxes_3d):
        # The stored parameters are cast to float32 on the way out.
        assert np.allclose(loaded_box.box3d_params, box.box3d_params, atol=1e-6)
        assert loaded_box.box3d_label_name == box.box3d_label_name
        assert loaded_box.box3d_instance_id == box.box3d_instance_id
        assert loaded_box.box3d_num_lidar_points == box.box3d_num_lidar_points
        assert loaded_box.box3d_attributes == box.box3d_attributes

    assert list(loaded.category_mapping.category_names) == ["car", "pedestrian"]
    assert list(loaded.category_mapping.category_indices) == [0, 1]


def test_record_round_trips_through_its_dictionary_form() -> None:
    record = _full_record()

    loaded = DatasetRecord.load_from_dictionary(record.to_dictionary())

    _assert_record_matches(loaded, record)


def test_record_round_trips_through_the_polars_cache_schema() -> None:
    record = _full_record()
    dataframe = pl.DataFrame([record.to_dictionary()], schema=DatasetTableSchema.to_polars_schema())

    loaded = DatasetRecord.load_from_dictionary(dataframe.row(0, named=True))

    _assert_record_matches(loaded, record)


def test_absent_annotations_survive_the_round_trip_as_none() -> None:
    record = make_record(category_names=("car",), category_indices=(0,))

    loaded = DatasetRecord.load_from_dictionary(record.to_dictionary())

    assert loaded.boxes_3d is None
    assert loaded.camera_frames is None
    assert loaded.lidar_sources is None


def test_empty_annotations_stay_distinct_from_absent_ones() -> None:
    record = make_record(boxes_3d=[], category_names=("car",), category_indices=(0,))
    dataframe = pl.DataFrame(
        [record.to_dictionary()], schema=DatasetTableSchema.to_polars_schema()
    )

    loaded = DatasetRecord.load_from_dictionary(dataframe.row(0, named=True))

    assert loaded.boxes_3d == []


def test_absent_category_mapping_survives_the_polars_cache() -> None:
    record = make_record()
    dataframe = pl.DataFrame(
        [record.to_dictionary()], schema=DatasetTableSchema.to_polars_schema()
    )

    loaded = DatasetRecord.load_from_dictionary(dataframe.row(0, named=True))

    assert loaded.category_mapping is None
    assert loaded.boxes_3d is None


def test_camera_frame_round_trips_through_its_dictionary_form() -> None:
    intrinsic = np.array([[900.0, 0.0, 960.0], [0.0, 901.0, 540.0], [0.0, 0.0, 1.0]])
    camera_to_ego = np.eye(4)
    camera_to_ego[:3, 3] = [1.5, 0.0, 1.8]
    frame = make_camera_frame(
        channel_name="CAM_FRONT_LEFT",
        intrinsic=intrinsic,
        camera_to_ego=camera_to_ego,
        distortion_coefficients=(0.1, -0.2, 0.001, 0.002),
    )

    loaded = CameraFrameDataModel.load_from_dictionary(frame.to_dictionary())

    assert loaded.camera_frame_id == frame.camera_frame_id
    assert loaded.camera_sensor_channel_name == "CAM_FRONT_LEFT"
    assert loaded.camera_timestamp_seconds == frame.camera_timestamp_seconds
    assert loaded.camera_image_path == frame.camera_image_path
    assert loaded.camera_intrinsic_matrix.dtype == np.float64
    assert np.allclose(loaded.camera_intrinsic_matrix, intrinsic)
    assert np.allclose(loaded.camera_sensor_to_ego_pose_matrix, camera_to_ego)
    assert list(loaded.camera_distortion_coefficients) == [0.1, -0.2, 0.001, 0.002]
    assert loaded.camera_distortion_model == frame.camera_distortion_model


def test_record_table_rejects_a_string_database_selection(tmp_path) -> None:
    """An unfilled mandatory value reaches the constructor as a bare string."""
    import pytest

    from autoware_ml.databases.record_table import RecordTable

    table = tmp_path / "records.parquet"
    table.write_bytes(b"")
    with pytest.raises(TypeError, match="must be a list of database names"):
        RecordTable(path=str(table), data_root=str(tmp_path), databases="???")
