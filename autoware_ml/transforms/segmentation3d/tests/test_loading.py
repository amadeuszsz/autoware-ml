"""Tests for the segmentation annotation loading transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.testing.factories import (
    make_lidar_frame,
    make_point_cloud,
    make_record,
    make_sample,
)
from autoware_ml.transforms.segmentation3d.loading import LoadSeg3DAnnotations


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


def test_label_mapping_maps_unlisted_labels_to_ignore(tmp_path) -> None:
    sample = _seg_sample(tmp_path, [0, 1, 4, 7])

    output = LoadSeg3DAnnotations(label_mapping={0: 3, 4: 9}, max_label=7, ignore_index=255)(sample)

    assert output.segment.labels.dtype == np.int64
    assert np.array_equal(output.segment.labels, np.array([3, 255, 9, 255], dtype=np.int64))


def test_label_mapping_sizes_lookup_from_mapping_without_max_label(tmp_path) -> None:
    sample = _seg_sample(tmp_path, [0, 1, 4, 7])

    output = LoadSeg3DAnnotations(label_mapping={0: 3, 4: 9}, ignore_index=255)(sample)

    assert np.array_equal(output.segment.labels, np.array([3, 255, 9, 255], dtype=np.int64))


def test_class_mapping_resolves_labels_through_the_record(tmp_path) -> None:
    sample = _seg_sample(
        tmp_path,
        [2, 5, 9],
        category_names=("car", "pedestrian"),
        category_indices=(2, 5),
    )

    output = LoadSeg3DAnnotations(class_mapping={"car": 0, "pedestrian": 1}, ignore_index=255)(
        sample
    )

    assert np.array_equal(output.segment.labels, np.array([0, 1, 255], dtype=np.int64))


def test_class_mapping_ignores_categories_missing_from_the_mapping(tmp_path) -> None:
    sample = _seg_sample(
        tmp_path,
        [2, 5],
        category_names=("car", "tree"),
        category_indices=(2, 5),
    )

    output = LoadSeg3DAnnotations(class_mapping={"car": 0}, ignore_index=255)(sample)

    assert np.array_equal(output.segment.labels, np.array([0, 255], dtype=np.int64))


def test_constructor_rejects_both_or_neither_mapping() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LoadSeg3DAnnotations(label_mapping={0: 1}, class_mapping={"car": 0})
    with pytest.raises(ValueError, match="exactly one"):
        LoadSeg3DAnnotations()


def test_missing_mask_path_raises(tmp_path) -> None:
    frame = make_lidar_frame(semantic_mask_path=None)
    record = make_record(lidar_frames=[frame])
    sample = make_sample(
        record=record,
        data_root=str(tmp_path),
        points=make_point_cloud(num_points=4, with_time_lag=False),
    )

    with pytest.raises(ValueError, match="no semantic mask path"):
        LoadSeg3DAnnotations(label_mapping={0: 1}, ignore_index=255)(sample)


def test_class_mapping_requires_record_category_mapping(tmp_path) -> None:
    sample = _seg_sample(tmp_path, [2, 5])

    with pytest.raises(ValueError, match="category mapping"):
        LoadSeg3DAnnotations(class_mapping={"car": 0}, ignore_index=255)(sample)


def test_requires_loaded_points() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="points"):
        LoadSeg3DAnnotations(label_mapping={0: 1}, ignore_index=255)(sample)
