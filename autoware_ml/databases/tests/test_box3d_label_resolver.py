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

"""Tests for the baking of box labels through the taxonomy and the box pipelines."""

from __future__ import annotations

import pytest

from autoware_ml.databases.box3d_pipelines.box3d_label_resolver import Box3DLabelResolver
from autoware_ml.databases.box3d_pipelines.box3d_merger import Box3DExtendLongerMerger
from autoware_ml.testing.factories import make_box3d_data_model, make_label_taxonomy

NAME_MAPPING = {
    "car": "car",
    "truck": "truck",
    "tractor_unit": "truck",
    "trailer": "trailer",
    "semi_trailer": "trailer",
}


def _online():
    return make_label_taxonomy(
        class_names=("car", "truck"),
        name_mapping=NAME_MAPPING,
        coarsening={"car": "car", "truck": "truck", "trailer": None},
    )


def _offline():
    return make_label_taxonomy(
        class_names=("car", "truck", "trailer"),
        name_mapping=NAME_MAPPING,
    )


def _merger() -> Box3DExtendLongerMerger:
    return Box3DExtendLongerMerger(
        target_labels={"truck": ["truck", "trailer"]}, proximity_distance_threshold=1.0
    )


def _raw_box(raw_name: str, params, instance_id: str):
    return make_box3d_data_model(
        params=params, label_name=raw_name, label_index=-1, instance_id=instance_id
    )


def test_boxes_store_the_fine_name_and_the_class_index_of_the_level() -> None:
    boxes = [
        _raw_box("tractor_unit", (0.0, 0.0, 0.5, 6.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0), "tractor"),
        _raw_box("semi_trailer", (40.0, 0.0, 0.5, 8.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0), "trailer"),
        _raw_box("drainage", (10.0, 5.0, 0.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0), "unknown"),
    ]

    online = Box3DLabelResolver(_online(), [])(boxes)
    offline = Box3DLabelResolver(_offline(), [])(boxes)

    assert [box.box3d_label_name for box in online] == ["truck", "trailer", "drainage"]
    assert [box.box3d_label_index for box in online] == [1, -1, -1]
    assert [box.box3d_label_name for box in offline] == ["truck", "trailer", "drainage"]
    assert [box.box3d_label_index for box in offline] == [1, 2, -1]
    assert [box.box3d_dataset_label_name for box in online] == [
        "tractor_unit",
        "semi_trailer",
        "drainage",
    ]


def test_pipelines_run_on_fine_names_before_the_indices_are_assigned() -> None:
    boxes = [
        _raw_box("tractor_unit", (0.0, 0.0, 0.5, 6.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0), "tractor"),
        _raw_box("semi_trailer", (7.0, 0.0, 0.5, 8.0, 2.5, 3.0, 0.0, 0.0, 0.0, 0.0), "trailer"),
    ]

    merged = Box3DLabelResolver(_online(), [_merger()])(boxes)

    assert len(merged) == 1
    assert merged[0].box3d_label_name == "truck"
    assert merged[0].box3d_label_index == 1
    assert merged[0].box3d_instance_id == "tractor"
    assert merged[0].box3d_params[3] > 6.0


def test_merger_rejects_a_level_that_trains_the_absorbed_label() -> None:
    with pytest.raises(ValueError, match="absorbs \\['trailer'\\]"):
        Box3DLabelResolver(_offline(), [_merger()])


def test_merger_rejects_a_target_outside_the_vocabulary() -> None:
    merger = Box3DExtendLongerMerger(
        target_labels={"bus": ["bus", "trailer"]}, proximity_distance_threshold=1.0
    )

    with pytest.raises(ValueError, match="not fine labels of the vocabulary"):
        Box3DLabelResolver(_online(), [merger])


def test_merger_requires_the_target_among_its_sources() -> None:
    with pytest.raises(ValueError, match="must be one of its source labels"):
        Box3DExtendLongerMerger(
            target_labels={"truck": ["car", "trailer"]}, proximity_distance_threshold=1.0
        )
