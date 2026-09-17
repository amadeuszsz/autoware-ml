"""
Bboxes 3d transforms for loading bboxes (for example, label name filter).
The code is modified based on https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from typing import Sequence

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.transforms.base import BaseTransform


class BBoxesLabelNameFilter(BaseTransform):
    """Filter 3D bounding boxes by label names."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, label_names_to_keep: Sequence[str]) -> None:
        """Initialize the BBoxesLabelNameFilter transform."""
        super().__init__(probability=None)
        self.label_names_to_keep = label_names_to_keep

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Filter 3D bounding boxes by label names."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        bboxes_to_keep_mask = torch.tensor(
            [
                True if label_name in self.label_names_to_keep else False
                for label_name in detection3d_gt_bboxes_3d.bbox_label_names
            ],
            dtype=torch.bool,
        )

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return model_gt_sample


class BBoxesAttributeFilter(BaseTransform):
    """
    Drop the 3D bounding boxes whose class and attributes match an exclusion rule.

    Some annotated objects are no detection target, a parked bicycle or a motorcycle without
    a rider for one. A rule names a class and an attribute, and a box of that class carrying
    that attribute is removed from the sample, so it is neither trained on nor scored.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, filter_attributes: Sequence[Sequence[str]]) -> None:
        """
        Initialize the BBoxesAttributeFilter transform.

        Args:
          filter_attributes: Exclusion rules, each a pair of class name and attribute name.
        """
        super().__init__(probability=None)
        rules = []
        for index, rule in enumerate(filter_attributes):
            if isinstance(rule, str) or len(rule) != 2:
                raise ValueError(
                    f"Exclusion rule {index} must be a [class_name, attribute] pair, got {rule!r}."
                )
            rules.append((str(rule[0]), str(rule[1])))
        self.filter_attributes = frozenset(rules)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes matching an exclusion rule."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d) or not self.filter_attributes:
            return model_gt_sample

        bbox_attributes = detection3d_gt_bboxes_3d.bbox_attributes
        if bbox_attributes is None:
            raise ValueError(
                "The attribute filter needs the attributes of every box, the dataset served none."
            )

        bboxes_to_keep_mask = torch.tensor(
            [
                not any(
                    (label_name, attribute) in self.filter_attributes for attribute in attributes
                )
                for label_name, attributes in zip(
                    detection3d_gt_bboxes_3d.bbox_label_names, bbox_attributes, strict=True
                )
            ],
            dtype=torch.bool,
        )
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return model_gt_sample
