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

"""Merge spatially coupled box annotations (for example truck plus trailer).

This transform reproduces the AWML info-generation behaviour where a sub-object (such as a
trailer) that overlaps or sits next to its target object (a truck) is merged into a single,
elongated target box. Unmatched sub-objects are left in place and subsequently dropped by
LoadDet3DAnnotations when their mapped class is not in the detector classes, exactly as AWML's
merge_objects plus class filtering.

It rewrites the stored box annotations of the sample record, before LoadDet3DAnnotations, so the
sub-object class is still distinguishable from the target via name_mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi

import numpy as np
from jaxtyping import Float64
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class MergeObjects3D(BaseTransform):
    """Merge a sub-object box into a nearby target box into a single box.

    For each (target, [primary, secondary]) rule, every primary box is matched against every
    secondary box. A pair that overlaps in BEV or whose front or back face centers are within
    distance_threshold is merged into one box labelled target. Matching is greedy: each box
    participates in at most one merge. Unmatched boxes are left untouched.
    """

    def __init__(
        self,
        *,
        merge_objects: Sequence[tuple[str, Sequence[str]]] | None = None,
        name_mapping: Mapping[str, str | None] | None = None,
        distance_threshold: float = 2.0,
        merge_type: str = "extend_longer",
    ) -> None:
        """Initialize the MergeObjects3D transform.

        Args:
            merge_objects: Rules (target, [primary_class, secondary_class]) using canonical
                class names after name_mapping.
            name_mapping: Optional raw dataset category to canonical class mapping used to
                classify boxes before matching.
            distance_threshold: Maximum front or back face-center distance in meters for
                boxes to be considered adjacent.
            merge_type: Box merge strategy, "extend_longer" or "union".
        """
        if merge_type not in {"extend_longer", "union"}:
            raise ValueError(f"merge_type must be 'extend_longer' or 'union', got {merge_type!r}.")
        merge_rules = [] if merge_objects is None else merge_objects
        self.merge_objects = []
        for index, rule in enumerate(merge_rules):
            if isinstance(rule, str) or not isinstance(rule, Sequence):
                raise TypeError(
                    "merge_objects entries must be two-item sequences, "
                    f"got {type(rule).__name__} at index {index}."
                )
            if len(rule) != 2:
                raise ValueError(
                    "merge_objects entries must contain [target, [primary, secondary]], "
                    f"got {list(rule)!r} at index {index}."
                )
            target, sources = rule
            if isinstance(sources, str) or not isinstance(sources, Sequence):
                raise TypeError(
                    "merge_objects sources must be two-item sequences, "
                    f"got {type(sources).__name__} at index {index}."
                )
            if len(sources) != 2:
                raise ValueError(
                    "merge_objects sources must contain [primary, secondary], "
                    f"got {sources!r} at index {index}."
                )
            primary, secondary = sources
            self.merge_objects.append((str(target), [str(primary), str(secondary)]))
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.distance_threshold = float(distance_threshold)
        self.merge_type = merge_type

    def transform(self, sample: Sample) -> Sample:
        """Merge matched primary and secondary box pairs within the sample record.

        Args:
            sample: Sample whose record carries 3D box annotations.

        Returns:
            Sample whose record has each matched pair replaced by a single merged target box,
            with unmatched boxes left in place. Returned unchanged when there are no merge
            rules, no boxes, or no pairs matched.
        """
        if not self.merge_objects:
            return sample

        boxes_3d = sample.record.boxes_3d
        if boxes_3d is None:
            raise ValueError(
                f"The record of sample {sample.meta.sample_id} carries no 3D box annotations."
            )
        boxes_3d = list(boxes_3d)
        if not boxes_3d:
            return sample

        canonical = [self._canonical_name(box) for box in boxes_3d]
        geometries = [np.asarray(box.box3d_params[:7], dtype=np.float64) for box in boxes_3d]
        merge_function = (
            _merge_boxes_extend_longer if self.merge_type == "extend_longer" else _merge_boxes_union
        )

        consumed: set[int] = set()
        merged_boxes: list[Box3DDataModel] = []
        for target, (primary, secondary) in self.merge_objects:
            primary_indices = [i for i, name in enumerate(canonical) if name == primary]
            secondary_indices = [i for i, name in enumerate(canonical) if name == secondary]
            for i in primary_indices:
                if i in consumed:
                    continue
                for j in secondary_indices:
                    if j in consumed or i == j:
                        continue
                    if _boxes_overlap(geometries[i], geometries[j]) or _boxes_proximity(
                        geometries[i], geometries[j], self.distance_threshold
                    ):
                        merged_boxes.append(
                            self._merge_boxes(
                                boxes_3d[i],
                                boxes_3d[j],
                                merge_function(geometries[i], geometries[j]),
                                target,
                            )
                        )
                        consumed.add(i)
                        consumed.add(j)
                        break

        if not consumed:
            return sample

        survivors = [box for index, box in enumerate(boxes_3d) if index not in consumed]
        record = sample.record.model_copy(update={"boxes_3d": merged_boxes + survivors})
        return sample.model_copy(update={"record": record})

    def _canonical_name(self, box: Box3DDataModel) -> str | None:
        """Resolve the canonical class name of a box via name_mapping.

        Args:
            box: Stored box annotation.

        Returns:
            The canonical class name (the raw dataset name when name_mapping is None), or
            None when the raw name maps to None.
        """
        raw_name = box.box3d_dataset_label_name
        if self.name_mapping is None:
            return raw_name
        mapped = self.name_mapping.get(raw_name, raw_name)
        return str(mapped) if mapped is not None else None

    @staticmethod
    def _merge_boxes(
        primary: Box3DDataModel,
        secondary: Box3DDataModel,
        merged_geometry: list[float],
        target: str,
    ) -> Box3DDataModel:
        """Build one merged box from a matched primary and secondary pair.

        The merged box keeps the identity fields of the primary box, carries the merged
        geometry with the averaged velocity, and is relabelled as a fresh target object so
        downstream resolution keys off the new dataset label name.
        """
        velocity = (
            np.asarray(primary.box3d_params[7:], dtype=np.float64)
            + np.asarray(secondary.box3d_params[7:], dtype=np.float64)
        ) / 2.0
        params = np.concatenate([np.asarray(merged_geometry, dtype=np.float64), velocity])
        return primary.create_new_data_model(
            box3d_params=params,
            box3d_dataset_label_name=target,
            box3d_label_name=target,
            box3d_label_index=-1,
            box3d_num_lidar_points=int(primary.box3d_num_lidar_points)
            + int(secondary.box3d_num_lidar_points),
            box3d_valid=bool(primary.box3d_valid) and bool(secondary.box3d_valid),
            box3d_attributes=set(primary.box3d_attributes) | set(secondary.box3d_attributes),
        )


def _box_corners(box: Float64[np.ndarray, " 7"]) -> Float64[np.ndarray, "4 2"]:
    """Return the four BEV corners of an oriented box [x, y, z, dx, dy, dz, yaw]."""
    x, y, _, dx, dy, _, yaw = box
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    half_dx, half_dy = dx / 2.0, dy / 2.0
    return np.array(
        [
            [x - half_dx * cos_yaw + half_dy * sin_yaw, y - half_dx * sin_yaw - half_dy * cos_yaw],
            [x + half_dx * cos_yaw + half_dy * sin_yaw, y + half_dx * sin_yaw - half_dy * cos_yaw],
            [x + half_dx * cos_yaw - half_dy * sin_yaw, y + half_dx * sin_yaw + half_dy * cos_yaw],
            [x - half_dx * cos_yaw - half_dy * sin_yaw, y - half_dx * sin_yaw + half_dy * cos_yaw],
        ]
    )


def _boxes_overlap(box1: Float64[np.ndarray, " 7"], box2: Float64[np.ndarray, " 7"]) -> bool:
    """Return whether two boxes overlap in the BEV plane."""
    return Polygon(_box_corners(box1)).intersects(Polygon(_box_corners(box2)))


def _boxes_proximity(
    box1: Float64[np.ndarray, " 7"], box2: Float64[np.ndarray, " 7"], distance_threshold: float
) -> bool:
    """Return whether any front or back face centers are within the threshold."""

    def face_centers(
        box: Float64[np.ndarray, " 7"],
    ) -> tuple[Float64[np.ndarray, " 3"], Float64[np.ndarray, " 3"]]:
        """Return the front and back face-center points along the box heading.

        Args:
            box: Oriented box [x, y, z, dx, dy, dz, yaw].

        Returns:
            Tuple (front, back) of face-center coordinates offset from the center by half
            the length dx along the yaw direction.
        """
        x, y, z, dx, _, _, yaw = box
        front = np.array([x + dx / 2.0 * np.cos(yaw), y + dx / 2.0 * np.sin(yaw), z])
        back = np.array([x - dx / 2.0 * np.cos(yaw), y - dx / 2.0 * np.sin(yaw), z])
        return front, back

    front1, back1 = face_centers(box1)
    front2, back2 = face_centers(box2)
    for a in (front1, back1):
        for b in (front2, back2):
            if np.linalg.norm(a - b) <= distance_threshold:
                return True
    return False


def _merge_boxes_extend_longer(
    box1: Float64[np.ndarray, " 7"], box2: Float64[np.ndarray, " 7"]
) -> list[float]:
    """Merge by elongating the larger box up to the far face of the smaller box."""

    def get_box_faces(
        box: Float64[np.ndarray, " 7"],
    ) -> tuple[
        Float64[np.ndarray, " 2"],
        Float64[np.ndarray, " 2"],
        Float64[np.ndarray, " 2"],
        float,
        float,
    ]:
        x, y, _, dx, dy, _, yaw = box
        center = np.array([x, y])
        if dx >= dy:
            face1 = np.array([x + (dx / 2) * np.cos(yaw), y + (dx / 2) * np.sin(yaw)])
            face2 = np.array([x - (dx / 2) * np.cos(yaw), y - (dx / 2) * np.sin(yaw)])
        else:
            face1 = np.array(
                [x + (dy / 2) * np.cos(yaw + pi / 2), y + (dy / 2) * np.sin(yaw + pi / 2)]
            )
            face2 = np.array(
                [x - (dy / 2) * np.cos(yaw + pi / 2), y - (dy / 2) * np.sin(yaw + pi / 2)]
            )
        return center, face1, face2, dx, dy

    c1, c1_f1, c1_f2, dx1, dy1 = get_box_faces(box1)
    c2, c2_f1, c2_f2, dx2, dy2 = get_box_faces(box2)

    if dx1 * dy1 >= dx2 * dy2:
        larger_center, larger_f1, larger_f2, larger_dx, larger_dy, larger_box = (
            c1,
            c1_f1,
            c1_f2,
            dx1,
            dy1,
            box1,
        )
        smaller_center, smaller_f1, smaller_f2 = c2, c2_f1, c2_f2
    else:
        larger_center, larger_f1, larger_f2, larger_dx, larger_dy, larger_box = (
            c2,
            c2_f1,
            c2_f2,
            dx2,
            dy2,
            box2,
        )
        smaller_center, smaller_f1, smaller_f2 = c1, c1_f1, c1_f2

    # Far face of the smaller box relative to the larger box center.
    if np.linalg.norm(smaller_f1 - larger_center) > np.linalg.norm(smaller_f2 - larger_center):
        selected_smaller_face = smaller_f1
    else:
        selected_smaller_face = smaller_f2

    # Near face of the larger box relative to the smaller box center.
    if np.linalg.norm(larger_f1 - smaller_center) < np.linalg.norm(larger_f2 - smaller_center):
        selected_larger_face = larger_f1
    else:
        selected_larger_face = larger_f2

    axis_vector = selected_larger_face - larger_center
    axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)
    to_smaller_box = selected_smaller_face - larger_center
    projection_length = np.dot(to_smaller_box, axis_vector_normalized)
    projection_point = larger_center + projection_length * axis_vector_normalized

    elongation_vector = projection_point - selected_larger_face
    elongation_length = np.linalg.norm(elongation_vector)

    new_dx = larger_dx + elongation_length if larger_dx >= larger_dy else larger_dx
    new_dy = larger_dy + elongation_length if larger_dy > larger_dx else larger_dy
    new_center = larger_center + elongation_vector / 2.0

    new_z, new_dz = _merge_center_z_and_height(box1, box2)
    new_yaw = larger_box[6]

    return [new_center[0], new_center[1], new_z, new_dx, new_dy, new_dz, new_yaw]


def _merge_boxes_union(
    box1: Float64[np.ndarray, " 7"], box2: Float64[np.ndarray, " 7"]
) -> list[float]:
    """Merge via the minimum rotated rectangle covering both BEV footprints."""

    def shapely_box(box: Float64[np.ndarray, " 7"]) -> Polygon:
        x, y, _, dx, dy, _, yaw = box
        rect = Polygon([(-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2)])
        rect = affinity.rotate(rect, yaw, origin=(0, 0), use_radians=True)
        return affinity.translate(rect, x, y)

    merged = unary_union([shapely_box(box1), shapely_box(box2)]).minimum_rotated_rectangle
    coords = list(merged.exterior.coords)[:-1]
    new_x = sum(point[0] for point in coords) / 4.0
    new_y = sum(point[1] for point in coords) / 4.0
    edge1 = float(np.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]))
    edge2 = float(np.hypot(coords[1][0] - coords[2][0], coords[1][1] - coords[2][1]))
    new_dx, new_dy = max(edge1, edge2), min(edge1, edge2)
    new_z, new_dz = _merge_center_z_and_height(box1, box2)
    if edge1 >= edge2:
        new_yaw = float(np.arctan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0]))
    else:
        new_yaw = float(np.arctan2(coords[2][1] - coords[1][1], coords[2][0] - coords[1][0]))
    return [new_x, new_y, new_z, new_dx, new_dy, new_dz, new_yaw]


def _merge_center_z_and_height(
    box1: Float64[np.ndarray, " 7"], box2: Float64[np.ndarray, " 7"]
) -> tuple[float, float]:
    """Return center z and height spanning two center-based boxes."""
    bottom = min(box1[2] - box1[5] / 2.0, box2[2] - box2[5] / 2.0)
    top = max(box1[2] + box1[5] / 2.0, box2[2] + box2[5] / 2.0)
    height = top - bottom
    center_z = bottom + height / 2.0
    return float(center_z), float(height)
