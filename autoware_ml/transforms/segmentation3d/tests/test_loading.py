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

"""Tests for the semantic mask loading transform."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.testing.factories import (
    make_label_taxonomy,
    make_lidar_frame,
    make_point_cloud,
    make_record,
    make_sample,
)
from autoware_ml.transforms.segmentation3d.loading import LoadSeg3DAnnotations

IGNORE = 255


def _taxonomy():
    # Raw categories fold onto two classes, the raw vehicle.car spells a fine name of its own
    return make_label_taxonomy(
        class_names=("car", "pedestrian"),
        name_mapping={
            "car": "car",
            "vehicle.car": "car",
            "pedestrian": "pedestrian",
            "stroller": "personal_mobility",
        },
        coarsening={"car": "car", "pedestrian": "pedestrian", "personal_mobility": "pedestrian"},
        ignore_index=IGNORE,
    )


def _seg_sample(tmp_path, raw_labels, dtype="uint8", category_names=(), category_indices=()):
    np.array(raw_labels, dtype=dtype).tofile(tmp_path / "mask.bin")
    frame = make_lidar_frame(semantic_mask_path="mask.bin")
    record = make_record(
        lidar_frames=[frame],
        category_names=category_names,
        category_indices=category_indices,
    )
    points = make_point_cloud(num_points=len(raw_labels), with_time_lag=False)
    return make_sample(record=record, data_root=str(tmp_path), points=points)


def test_categories_resolve_through_the_record_and_the_taxonomy(tmp_path) -> None:
    sample = _seg_sample(
        tmp_path,
        [2, 5, 7, 9],
        category_names=("vehicle.car", "pedestrian", "stroller"),
        category_indices=(2, 5, 7),
    )

    output = LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)

    assert output.segment.labels.dtype == np.int64
    assert np.array_equal(output.segment.labels, np.array([0, 1, 1, IGNORE], dtype=np.int64))


def test_categories_outside_the_taxonomy_take_the_ignore_index(tmp_path) -> None:
    sample = _seg_sample(
        tmp_path,
        [2, 5],
        category_names=("car", "tree"),
        category_indices=(2, 5),
    )

    output = LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)

    assert np.array_equal(output.segment.labels, np.array([0, IGNORE], dtype=np.int64))


def test_an_empty_category_mapping_ignores_every_point(tmp_path) -> None:
    # A source without segmentation supervision serves its records with an empty mapping
    sample = _seg_sample(tmp_path, [0, 3])
    record = sample.record.model_copy(
        update={
            "category_mapping": CategoryMappingDataModel(category_names=[], category_indices=[])
        }
    )
    sample = sample.model_copy(update={"record": record})

    output = LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)

    assert np.array_equal(output.segment.labels, np.array([IGNORE, IGNORE], dtype=np.int64))


def test_missing_mask_path_raises(tmp_path) -> None:
    frame = make_lidar_frame(semantic_mask_path=None)
    record = make_record(lidar_frames=[frame])
    sample = make_sample(
        record=record,
        data_root=str(tmp_path),
        points=make_point_cloud(num_points=4, with_time_lag=False),
    )

    with pytest.raises(ValueError, match="no semantic mask path"):
        LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)


def test_missing_record_category_mapping_raises(tmp_path) -> None:
    np.array([1, 2], dtype="uint8").tofile(tmp_path / "mask.bin")
    record = make_record(lidar_frames=[make_lidar_frame(semantic_mask_path="mask.bin")])
    record = record.model_copy(update={"category_mapping": None})
    sample = make_sample(
        record=record,
        data_root=str(tmp_path),
        points=make_point_cloud(num_points=2, with_time_lag=False),
    )

    with pytest.raises(ValueError, match="no category mapping"):
        LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)


def test_requires_loaded_points() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="points"):
        LoadSeg3DAnnotations(taxonomy=_taxonomy())(sample)
