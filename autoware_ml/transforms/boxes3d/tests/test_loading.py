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

"""Tests for the 3D box annotation loading transform."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_box3d_data_model, make_record, make_sample
from autoware_ml.transforms.boxes3d.loading import LoadDet3DAnnotations


def _sample(boxes: Sequence[Box3DDataModel]) -> Sample:
    return make_sample(record=make_record(boxes_3d=boxes))


def test_load_annotations3d_builds_detection_targets() -> None:
    sample = _sample(
        [
            make_box3d_data_model(
                params=(1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1, 0.0),
                label_name="vehicle.car",
                num_lidar_points=12,
            ),
            make_box3d_data_model(
                params=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                label_name="ignore-me",
                num_lidar_points=3,
            ),
        ]
    )

    output = LoadDet3DAnnotations(
        class_names=["car", "pedestrian"], name_mapping={"vehicle.car": "car"}
    )(sample)

    assert output.boxes.params.shape == (1, 9)
    assert output.boxes.labels.tolist() == [0]
    assert output.boxes.names == ("car",)
    assert output.boxes.num_lidar_points.tolist() == [12]
    np.testing.assert_allclose(
        output.boxes.params[0], [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1], atol=1e-6
    )


def test_load_annotations3d_replaces_non_finite_velocity_with_zero() -> None:
    sample = _sample(
        [
            make_box3d_data_model(
                params=(1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, np.nan, np.inf, np.nan),
                label_name="car",
            )
        ]
    )

    output = LoadDet3DAnnotations(class_names=["car"])(sample)

    assert output.boxes.params.shape == (1, 9)
    assert output.boxes.params[0, 7:].tolist() == [0.0, 0.0]


def test_load_annotations3d_drops_physically_invalid_boxes() -> None:
    def box(params: Sequence[float]) -> Box3DDataModel:
        return make_box3d_data_model(params=params, label_name="car", num_lidar_points=12)

    sample = _sample(
        [
            box((1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1, 0.0)),  # sane
            box((1e39, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.0, 0.0, 0.0)),  # f32 overflow
            box((3.0, 1.0, 0.5, 0.5, -0.8, 1.7, 0.0, 0.0, 0.0, 0.0)),  # negative dim
            box((1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 1e6, 0.0, 0.0)),  # absurd velocity
            box((1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 120.0, 120.0, 0.0)),  # speed norm > bound
            box((4.0, 4.0, 0.5, 4.5, 1.9, 1.4, 0.3, 140.0, 30.0, 0.0)),  # fast but physical
            box((5.0, 5.0, 0.5, 2.0, 1.0, 1.5, 0.2, np.nan, np.nan, 0.0)),  # nan vel: kept
        ]
    )

    output = LoadDet3DAnnotations(class_names=["car"])(sample)

    assert output.boxes.params.shape == (3, 9)
    np.testing.assert_allclose(output.boxes.params[0, :3], [1.0, 2.0, 3.0])
    # component > norm / sqrt(2), kept
    np.testing.assert_allclose(output.boxes.params[1, 7:], [140.0, 30.0])
    # nan velocity zeroed, box kept
    np.testing.assert_allclose(output.boxes.params[2, 7:], [0.0, 0.0])


def test_load_annotations3d_ignores_stored_label_name_and_index() -> None:
    # A record generation pipeline may bake in a label name and index under an older
    # taxonomy. The dataset label name together with the configured name_mapping decides
    # the class, so the stored label fields never override it.
    stale = make_box3d_data_model(label_name="pedestrian").create_new_data_model(
        box3d_label_name="car", box3d_label_index=0
    )
    sample = _sample([stale])

    output = LoadDet3DAnnotations(
        class_names=["car", "pedestrian"],
        name_mapping={"car": "car", "pedestrian": "pedestrian"},
    )(sample)

    assert output.boxes.names == ("pedestrian",)
    assert output.boxes.labels.tolist() == [1]


def test_load_annotations3d_ignores_validity_flag() -> None:
    # The valid flag is not a load-time filter: low-point filtering is the point-count
    # filters' job, so an invalid (0-lidar-point) box is loaded and left for the point
    # filter to drop.
    sample = _sample([make_box3d_data_model(label_name="car", num_lidar_points=0, valid=False)])

    output = LoadDet3DAnnotations(class_names=["car"])(sample)

    assert output.boxes.params.shape == (1, 9)
    assert output.boxes.num_lidar_points.tolist() == [0]


def test_load_annotations3d_filters_raw_class_attributes() -> None:
    sample = _sample(
        [
            make_box3d_data_model(
                params=(1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0),
                label_name="motorcycle",
                attributes=["vehicle_state.parked"],
            ),
            make_box3d_data_model(
                params=(2.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0),
                label_name="motorcycle",
                attributes=["two_wheel_vehicle_state.without_rider"],
            ),
        ]
    )

    output = LoadDet3DAnnotations(
        class_names=["bicycle"],
        name_mapping={"motorcycle": "bicycle"},
        filter_attributes=[["motorcycle", "vehicle_state.parked"]],
    )(sample)

    assert output.boxes.params[:, 0].tolist() == [2.0]
    assert output.boxes.names == ("bicycle",)


def test_load_annotations3d_loads_empty_boxes() -> None:
    sample = _sample([])

    output = LoadDet3DAnnotations(class_names=["car"])(sample)

    assert len(output.boxes) == 0
    assert output.boxes.params.shape == (0, 9)


def test_load_annotations3d_requires_record_annotations() -> None:
    sample = make_sample(record=make_record(boxes_3d=None))

    with pytest.raises(ValueError, match="no 3D box annotations"):
        LoadDet3DAnnotations(class_names=["car"])(sample)


def test_load_annotations3d_requires_class_names() -> None:
    with pytest.raises(ValueError, match="at least one detector class name"):
        LoadDet3DAnnotations(class_names=[])
