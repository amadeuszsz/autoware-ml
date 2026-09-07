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

"""Data samplers and frame sampling weights shared by Autoware-ML datamodules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from jaxtyping import Float32
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.transforms.boxes3d.annotations import (
    box_is_physical,
    normalize_filter_attributes,
    resolve_box_class,
    sanitize_box_params,
)


class DistributedWeightedRandomSampler(DistributedSampler):
    """Weighted random sampler that partitions one sampled epoch across ranks."""

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        *,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: Dataset sampled by the dataloader.
            weights: Per sample non negative sampling weights.
            replacement: Whether to sample indices with replacement.
            seed: Base seed used with set_epoch for deterministic shuffling.
            drop_last: Whether to drop tail samples when dataset length is not divisible by
                world size.
        """
        if len(weights) != len(dataset):
            raise ValueError(f"Expected {len(dataset)} sampler weights, got {len(weights)}.")
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if torch.any(self.weights < 0):
            raise ValueError("Sampler weights must be non-negative.")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("At least one sampler weight must be positive.")
        self.replacement = replacement

    def __iter__(self):
        """Yield the weighted sample indices for this rank."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


@dataclass(frozen=True)
class FrameSamplingConfig:
    """Configuration for repeat factor frame sampling.

    Attributes:
        repeat_sampling_factor: Target category fraction of the repeat factor formula.
        object_bev_range: BEV range [x_min, y_min, x_max, y_max] a box center must fall in to
            count for its category.
        low_pedestrian_height_threshold: Height below which a pedestrian box counts into the
            low pedestrian bucket.
        low_pedestrian_bev_range: BEV range of the low pedestrian bucket.
        class_names: Trained class names, the categories boxes are counted into.
        ignore_label_index: Label index of a box whose class is not trained.
        filter_attributes: Class and attribute name pairs excluded from class counting.
        low_pedestrian_category_name: Name of the low pedestrian sampling bucket.
    """

    repeat_sampling_factor: float
    object_bev_range: list[float]
    low_pedestrian_height_threshold: float
    low_pedestrian_bev_range: list[float]
    class_names: list[str]
    ignore_label_index: int
    filter_attributes: list[list[str]] | None = None
    low_pedestrian_category_name: str = "low_pedestrian"


def coerce_frame_sampling(
    cfg: FrameSamplingConfig | Mapping[str, Any] | None,
) -> FrameSamplingConfig | None:
    """Normalize frame sampling settings to FrameSamplingConfig.

    Args:
        cfg: Frame sampling settings as a dataclass, a mapping, or None.

    Returns:
        The settings as FrameSamplingConfig, or None when disabled.
    """
    if cfg is None:
        return None
    if isinstance(cfg, FrameSamplingConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return FrameSamplingConfig(**dict(cfg))
    raise TypeError(
        "Expected frame sampling config to be a FrameSamplingConfig, mapping, or None, "
        f"got {type(cfg)!r}."
    )


def compute_frame_sampling_weights(
    records: Sequence[tuple[DatasetRecord, DatasetSource]],
    frame_sampling: FrameSamplingConfig,
) -> list[float]:
    """Compute repeat factor sampling weights for the records of a training split.

    Args:
        records: Records of the split in index order, with their sources.
        frame_sampling: Repeat factor settings.

    Returns:
        One sampling weight per record.
    """
    filter_attributes = normalize_filter_attributes(frame_sampling.filter_attributes)
    sampling_categories = [
        *frame_sampling.class_names,
        frame_sampling.low_pedestrian_category_name,
    ]
    category_frame_counts = {category: 0 for category in sampling_categories}
    category_box_counts = {category: 0 for category in sampling_categories}
    frame_categories = []

    for record, source in records:
        categories = _record_sampling_categories(record, frame_sampling, filter_attributes)
        frame_categories.append(categories)
        for category, count in categories.items():
            if count <= 0:
                continue
            category_frame_counts[category] += 1
            category_box_counts[category] += count

    total_boxes = sum(category_box_counts.values())
    if total_boxes == 0:
        raise ValueError("Cannot compute frame sampling weights for a dataset with no valid boxes.")

    category_factors = {}
    for category in sampling_categories:
        frame_fraction = category_frame_counts[category] / len(records)
        box_fraction = category_box_counts[category] / total_boxes
        if frame_fraction == 0.0 or box_fraction == 0.0:
            category_factors[category] = 1.0
            continue
        category_fraction = math.sqrt(frame_fraction * box_fraction)
        category_factors[category] = max(
            1.0,
            math.sqrt(frame_sampling.repeat_sampling_factor / category_fraction),
        )

    frame_weights = []
    for categories in frame_categories:
        weight = 1.0
        for category, count in categories.items():
            if count > 0:
                weight = max(weight, category_factors[category])
        frame_weights.append(weight)
    return frame_weights


def _record_sampling_categories(
    record: DatasetRecord,
    frame_sampling: FrameSamplingConfig,
    filter_attributes: frozenset[tuple[str, str]],
) -> dict[str, int]:
    """Return sampling category counts for one record.

    Args:
        record: Dataset record of the frame.
        frame_sampling: Repeat factor settings.
        filter_attributes: Normalized class and attribute exclusions.

    Returns:
        Sampling category counts of the frame.
    """
    categories = {
        *frame_sampling.class_names,
        frame_sampling.low_pedestrian_category_name,
    }
    category_counts = {category: 0 for category in categories}

    for box in record.boxes_3d if record.boxes_3d is not None else []:
        class_name = resolve_box_class(
            box,
            class_names=frame_sampling.class_names,
            ignore_label_index=frame_sampling.ignore_label_index,
            filter_attributes=filter_attributes,
        )
        if class_name is None:
            continue
        params = sanitize_box_params(np.asarray(box.box3d_params, dtype=np.float64))
        if not box_is_physical(params):
            continue
        # A box with no lidar points gives no supervision (the point count train filters drop
        # it), so it must not inflate the frame's sampling weight.
        if box.box3d_num_lidar_points <= 0:
            continue
        if not _box_center_in_bev_range(params, frame_sampling.object_bev_range):
            continue

        category = class_name
        if _is_low_pedestrian(class_name, params, frame_sampling):
            category = frame_sampling.low_pedestrian_category_name
        category_counts[category] += 1

    return category_counts


def _is_low_pedestrian(
    class_name: str,
    params: Float32[np.ndarray, " num_box_params"],
    frame_sampling: FrameSamplingConfig,
) -> bool:
    """Return whether a box belongs to the low pedestrian sampling bucket.

    Args:
        class_name: Class name of the box.
        params: Box parameters following Box3DFieldIndex.
        frame_sampling: Repeat factor settings.

    Returns:
        Whether the box belongs to the low pedestrian bucket.
    """
    return (
        class_name == "pedestrian"
        and float(params[5]) < frame_sampling.low_pedestrian_height_threshold
        and _box_center_in_bev_range(params, frame_sampling.low_pedestrian_bev_range)
    )


def _box_center_in_bev_range(
    params: Float32[np.ndarray, " num_box_params"], bev_range: list[float]
) -> bool:
    """Check whether a box center is inside [x_min, y_min, x_max, y_max].

    Args:
        params: Box parameters following Box3DFieldIndex.
        bev_range: BEV range as [x_min, y_min, x_max, y_max].

    Returns:
        Whether the box center is inside the range.
    """
    x, y = float(params[0]), float(params[1])
    return bev_range[0] <= x <= bev_range[2] and bev_range[1] <= y <= bev_range[3]
