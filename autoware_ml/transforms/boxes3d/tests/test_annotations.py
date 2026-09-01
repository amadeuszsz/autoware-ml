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

"""Tests for the shared 3D annotation interpretation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.testing.factories import make_box3d_data_model
from autoware_ml.transforms.boxes3d.annotations import (
    box_is_physical,
    normalize_filter_attributes,
    resolve_detection_class,
    sanitize_box_params,
)


def test_sanitize_box_params_drops_vertical_velocity_and_zeroes_non_finite() -> None:
    stored = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, np.nan, np.inf, np.nan], dtype=np.float64)

    params = sanitize_box_params(stored)

    assert params.shape == (9,)
    assert params.dtype == np.float32
    np.testing.assert_allclose(params[:7], stored[:7].astype(np.float32))
    assert params[7:].tolist() == [0.0, 0.0]


def test_sanitize_box_params_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Stored box parameters"):
        sanitize_box_params(np.zeros(9, dtype=np.float64))


def test_box_is_physical_decisions() -> None:
    sane = sanitize_box_params(
        np.array([1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1, 0.0], dtype=np.float64)
    )
    overflow = sanitize_box_params(
        np.array([1e39, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    negative_dim = sanitize_box_params(
        np.array([3.0, 1.0, 0.5, 0.5, -0.8, 1.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    speed_norm_beyond_bound = sanitize_box_params(
        np.array([1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 120.0, 120.0, 0.0], dtype=np.float64)
    )
    fast_but_physical = sanitize_box_params(
        np.array([4.0, 4.0, 0.5, 4.5, 1.9, 1.4, 0.3, 140.0, 30.0, 0.0], dtype=np.float64)
    )

    assert box_is_physical(sane)
    assert not box_is_physical(overflow)
    assert not box_is_physical(negative_dim)
    assert not box_is_physical(speed_norm_beyond_bound)
    assert box_is_physical(fast_but_physical)


def test_resolve_detection_class_uses_the_dataset_label_name() -> None:
    box = make_box3d_data_model(label_name="vehicle.car").create_new_data_model(
        box3d_label_name="stale", box3d_label_index=5
    )

    resolved = resolve_detection_class(
        box, class_names=["car"], name_mapping={"vehicle.car": "car"}
    )

    assert resolved == "car"


def test_resolve_detection_class_rejects_unmapped_and_filtered_boxes() -> None:
    unmapped = make_box3d_data_model(label_name="ignore-me")
    filtered = make_box3d_data_model(label_name="motorcycle", attributes=["vehicle_state.parked"])

    filter_attributes = normalize_filter_attributes([["motorcycle", "vehicle_state.parked"]])

    assert resolve_detection_class(unmapped, class_names=["car"], name_mapping=None) is None
    assert (
        resolve_detection_class(
            filtered,
            class_names=["bicycle"],
            name_mapping={"motorcycle": "bicycle"},
            filter_attributes=filter_attributes,
        )
        is None
    )


def test_normalize_filter_attributes_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="filter_attributes entries"):
        normalize_filter_attributes([["bicycle"]])

    with pytest.raises(TypeError, match="filter_attributes entries"):
        normalize_filter_attributes(["bicycle"])
