# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared 3D annotation interpretation helpers.

The helpers turn stored box annotations into detector training targets. They operate on
Box3DDataModel records and plain arrays so both the annotation loading transform and the
datamodule sampling weights interpret annotations through the exact same rules.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

import numpy as np
from jaxtyping import Float32, Float64

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.datamodule.samples.boxes3d import NUM_BOX_PARAMS
from autoware_ml.types.geometry import Box3DFieldIndex

FilterAttributeSet = frozenset[tuple[str, str]]

# Physical sanity bound for annotation velocity: nothing on a road moves faster than
# 150 m/s (540 km/h). Speeds above this are pipeline garbage and would silently explode
# the velocity regression loss.
MAX_ABSOLUTE_SPEED = 150.0


def sanitize_velocity(velocity: Float64[np.ndarray, " 2"]) -> Float64[np.ndarray, " 2"]:
    """Zero the non-finite components of a ground plane velocity.

    Args:
        velocity: Ground plane velocity as [velocity_x, velocity_y].

    Returns:
        The velocity with every non-finite component replaced by zero.
    """
    sanitized = np.array(velocity, dtype=np.float64)
    sanitized[~np.isfinite(sanitized)] = 0.0
    return sanitized


def sanitize_box_params(
    box3d_params: Float64[np.ndarray, " num_box_fields"],
) -> Float32[np.ndarray, " num_box_params"]:
    """Convert stored box parameters into detection target parameters.

    The vertical velocity is dropped and the ground plane velocity is sanitized before the
    float32 cast, so a stored non-finite velocity component never decides whether the box is
    trainable while a float64 magnitude that overflows float32 still does.

    Args:
        box3d_params: Stored box parameters following Box3DFieldIndex.

    Returns:
        Box parameters [x, y, z, length, width, height, yaw, velocity_x, velocity_y] as float32.
    """
    params = np.asarray(box3d_params, dtype=np.float64)
    if params.shape != (len(Box3DFieldIndex),):
        raise ValueError(
            f"Stored box parameters must have shape ({len(Box3DFieldIndex)},), got {params.shape}."
        )
    velocity = sanitize_velocity(params[Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Z])
    return np.concatenate([params[: Box3DFieldIndex.VELOCITY_X], velocity]).astype(np.float32)


def box_is_physical(params: Float32[np.ndarray, " num_box_params"]) -> bool:
    """Return whether a box annotation can become a valid training target.

    A physically invalid box cannot be trained on: non-finite values, non-positive dimensions
    (box size targets are log encoded), or a ground plane speed beyond the physical bound
    (velocity is never range filtered). Such boxes are dropped like any other non-loadable
    annotation. Geometry outliers with sane values are left to the range filters downstream.

    Args:
        params: Box parameters [x, y, z, length, width, height, yaw, velocity_x, velocity_y]
            with the velocity sanitized via sanitize_box_params, so a non-finite velocity
            component never reaches the drop decision.

    Returns:
        Whether the annotation is trainable.
    """
    values = np.asarray(params, dtype=np.float32)
    if values.shape != (NUM_BOX_PARAMS,):
        raise ValueError(f"Box parameters must have shape ({NUM_BOX_PARAMS},), got {values.shape}.")
    dimensions = values[Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW]
    velocity = values[Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Z]
    return bool(
        np.isfinite(values).all()
        and dimensions.min() > 0.0
        and float(np.linalg.norm(velocity)) <= MAX_ABSOLUTE_SPEED
    )


def normalize_filter_attributes(
    filter_attributes: Iterable[Sequence[str]] | None,
) -> FilterAttributeSet:
    """Normalize configured class-attribute exclusions for repeated lookup.

    Args:
        filter_attributes: Class and attribute name pairs, or None for none.

    Returns:
        The exclusions as a frozenset of (class, attribute) tuples.
    """
    if filter_attributes is None:
        return frozenset()

    normalized: list[tuple[str, str]] = []
    for index, entry in enumerate(filter_attributes):
        if isinstance(entry, str) or not isinstance(entry, Sequence):
            raise TypeError(
                "filter_attributes entries must be two-item sequences, "
                f"got {type(entry).__name__} at index {index}."
            )
        if len(entry) != 2:
            raise ValueError(
                "filter_attributes entries must contain [class_name, attribute], "
                f"got {list(entry)!r} at index {index}."
            )
        class_name, attribute = entry
        normalized.append((str(class_name), str(attribute)))
    return frozenset(normalized)


def resolve_detection_class(
    box: Box3DDataModel,
    *,
    class_names: Sequence[str],
    name_mapping: Mapping[str, str | None] | None,
    filter_attributes: Collection[tuple[str, str]] | None = None,
) -> str | None:
    """Resolve one stored box annotation into a detector class or reject it.

    The stored dataset label name together with the configured name_mapping is the single
    source of truth for the class: the config decides the taxonomy, the record only delivers
    raw dataset categories. The label name and index a record generation pipeline may have
    baked in are ignored, so changing the configured class set never requires regenerating
    the records.

    Low-point boxes are not filtered here, that is the job of the point-count filters
    (min_num_points at train time, the metric suite at eval), which subsume the lidar-point
    validity flag.

    Args:
        box: Stored box annotation.
        class_names: Detector class names.
        name_mapping: Raw dataset category to detector class mapping.
        filter_attributes: Normalized class and attribute exclusions.

    Returns:
        The detector class name, or None when the box is rejected.
    """
    raw_name = box.box3d_dataset_label_name
    mapped_name = _map_name(raw_name, name_mapping)
    if mapped_name is None or mapped_name not in class_names:
        return None
    if _has_filtered_attribute(box.box3d_attributes, raw_name, filter_attributes):
        return None
    return mapped_name


def _map_name(raw_name: str, name_mapping: Mapping[str, str | None] | None) -> str | None:
    """Map a raw dataset category name into the configured detector taxonomy."""
    if name_mapping is None:
        return raw_name
    mapped_name = name_mapping.get(raw_name, raw_name)
    return str(mapped_name) if mapped_name is not None else None


def _has_filtered_attribute(
    attributes: Collection[str],
    raw_name: str,
    filter_attributes: Collection[tuple[str, str]] | None,
) -> bool:
    """Return whether the raw class and attributes match an exclusion rule."""
    if not filter_attributes:
        return False
    return any((raw_name, str(attribute)) in filter_attributes for attribute in attributes)
