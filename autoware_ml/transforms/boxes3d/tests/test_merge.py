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

"""Tests for the MergeObjects3D transform (truck plus trailer merging)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from omegaconf import OmegaConf

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_box3d_data_model, make_record, make_sample
from autoware_ml.transforms.boxes3d.loading import LoadDet3DAnnotations
from autoware_ml.transforms.boxes3d.merge import MergeObjects3D

_NAME_MAPPING = {
    "truck": "truck",
    "vehicle.truck": "truck",
    "trailer": "trailer",
    "vehicle.trailer": "trailer",
    "semi_trailer": "trailer",
}


def _box(
    geometry: Sequence[float],
    name: str,
    num_pts: int = 10,
    velocity: Sequence[float] = (0.0, 0.0),
    attrs: Sequence[str] = (),
) -> Box3DDataModel:
    return make_box3d_data_model(
        params=(*geometry, *velocity, 0.0),
        label_name=name,
        num_lidar_points=num_pts,
        attributes=attrs,
    )


def _sample(boxes: Sequence[Box3DDataModel]) -> Sample:
    return make_sample(record=make_record(boxes_3d=boxes))


def _merge(merge_objects=(("truck", ["truck", "trailer"]),), **kwargs) -> MergeObjects3D:
    return MergeObjects3D(merge_objects=list(merge_objects), name_mapping=_NAME_MAPPING, **kwargs)


def test_extend_longer_geometry_matches_reference() -> None:
    # truck dx=4 at x=0, collinear trailer dx=4 at x=5 (1 m gap) merge into one 9 m box.
    output = _merge()(
        _sample(
            [
                _box([0, 0, 0, 4, 2, 2, 0], "truck", num_pts=10, velocity=(1.0, 0.0)),
                _box([5, 0, 0, 4, 2, 2, 0], "trailer", num_pts=5, velocity=(1.0, 0.0)),
            ]
        )
    )

    assert len(output.record.boxes_3d) == 1
    merged = output.record.boxes_3d[0]
    assert merged.box3d_dataset_label_name == "truck"
    assert merged.box3d_label_name == "truck"
    assert merged.box3d_label_index == -1
    np.testing.assert_allclose(
        merged.box3d_params, [2.5, 0.0, 0.0, 9.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6
    )
    assert merged.box3d_num_lidar_points == 15


def test_overlapping_pair_is_merged() -> None:
    output = _merge()(
        _sample(
            [
                _box([0, 0, 0, 4, 2, 2, 0], "truck"),
                _box([1, 0, 0, 4, 2, 2, 0], "vehicle.trailer"),  # overlaps the truck
            ]
        )
    )

    assert len(output.record.boxes_3d) == 1
    assert output.record.boxes_3d[0].box3d_dataset_label_name == "truck"


def test_distant_trailer_is_not_merged() -> None:
    # trailer back face (x=98) is far from truck front face (x=2): no merge.
    output = _merge()(
        _sample(
            [
                _box([0, 0, 0, 4, 2, 2, 0], "truck"),
                _box([100, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )
    )

    assert len(output.record.boxes_3d) == 2
    names = sorted(box.box3d_dataset_label_name for box in output.record.boxes_3d)
    assert names == ["trailer", "truck"]


def test_each_box_merges_at_most_once() -> None:
    # one truck between two trailers: only one merge consumes the truck.
    output = _merge()(
        _sample(
            [
                _box([5, 0, 0, 4, 2, 2, 0], "trailer"),
                _box([0, 0, 0, 4, 2, 2, 0], "truck"),
                _box([-5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )
    )

    boxes = output.record.boxes_3d
    merged = [box for box in boxes if box.box3d_dataset_label_name == "truck"]
    leftover_trailers = [box for box in boxes if box.box3d_dataset_label_name == "trailer"]
    assert len(merged) == 1
    assert len(leftover_trailers) == 1


def test_noop_without_rules() -> None:
    sample = _sample(
        [
            _box([0, 0, 0, 4, 2, 2, 0], "truck"),
            _box([5, 0, 0, 4, 2, 2, 0], "trailer"),
        ]
    )

    output = MergeObjects3D(merge_objects=None, name_mapping=_NAME_MAPPING)(sample)

    assert output is sample


def test_hydra_list_config_rules_do_not_require_truthiness() -> None:
    transform = MergeObjects3D(
        merge_objects=OmegaConf.create([["truck", ["truck", "trailer"]]]),
        name_mapping=_NAME_MAPPING,
    )

    assert transform.merge_objects == [("truck", ["truck", "trailer"])]


def test_invalid_merge_type_raises() -> None:
    with pytest.raises(ValueError, match="merge_type"):
        MergeObjects3D(merge_objects=[("truck", ["truck", "trailer"])], merge_type="bogus")


def test_extend_longer_merges_center_z_and_height_from_box_faces() -> None:
    output = _merge()(
        _sample(
            [
                _box([0, 0, 1, 4, 2, 2, 0], "truck"),
                _box([1, 0, 3, 4, 2, 2, 0], "trailer"),
            ]
        )
    )

    params = output.record.boxes_3d[0].box3d_params
    assert params[2] == pytest.approx(2.0, abs=1e-6)
    assert params[5] == pytest.approx(4.0, abs=1e-6)


def test_union_strategy_covers_both_boxes() -> None:
    output = _merge(merge_type="union")(
        _sample(
            [
                _box([0, 0, 0, 4, 2, 2, 0], "truck"),
                _box([5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        )
    )

    assert len(output.record.boxes_3d) == 1
    # Union footprint spans from the truck back (x=-2) to the trailer front (x=7).
    params = output.record.boxes_3d[0].box3d_params
    assert params[3] == pytest.approx(9.0, abs=1e-6)
    assert params[0] == pytest.approx(2.5, abs=1e-6)


def test_union_strategy_merges_center_z_and_height_from_box_faces() -> None:
    output = _merge(merge_type="union")(
        _sample(
            [
                _box([0, 0, 1, 4, 2, 2, 0], "truck"),
                _box([1, 0, 3, 4, 2, 2, 0], "trailer"),
            ]
        )
    )

    params = output.record.boxes_3d[0].box3d_params
    assert params[2] == pytest.approx(2.0, abs=1e-6)
    assert params[5] == pytest.approx(4.0, abs=1e-6)


def test_merged_record_feeds_annotation_loading() -> None:
    sample = _merge()(
        _sample(
            [
                _box([0, 0, 0, 4, 2, 2, 0], "vehicle.truck", num_pts=10),
                _box([5, 0, 0, 4, 2, 2, 0], "semi_trailer", num_pts=5),
            ]
        )
    )

    output = LoadDet3DAnnotations(class_names=["truck"], name_mapping=_NAME_MAPPING)(sample)

    assert output.boxes.names == ("truck",)
    assert output.boxes.labels.tolist() == [0]
    assert output.boxes.num_lidar_points.tolist() == [15]


def test_missing_record_annotations_raise() -> None:
    sample = make_sample(record=make_record(boxes_3d=None))

    with pytest.raises(ValueError, match="no 3D box annotations"):
        _merge()(sample)
