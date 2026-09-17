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
    """Filter 3D bounding boxes by the name of the class they are mapped to.

    A box keeps the label name it was annotated with, which can be finer than the class it
    trains as (an ambulance trains as a car). The class is read from the label index of the
    box, so a box is kept when its class is one of the kept names, whatever its own name.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, label_names_to_keep: Sequence[str], class_names: Sequence[str]) -> None:
        """Initialize the BBoxesLabelNameFilter transform.

        Args:
            label_names_to_keep: Names of the classes whose boxes are kept.
            class_names: Class names in label index order.
        """
        super().__init__(probability=None)
        unknown = sorted(set(label_names_to_keep) - set(class_names))
        if unknown:
            raise ValueError(f"label_names_to_keep names classes that do not exist: {unknown}.")
        self.label_indices_to_keep = torch.tensor(
            [index for index, name in enumerate(class_names) if name in label_names_to_keep],
            dtype=torch.int64,
        )

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Filter 3D bounding boxes by the class of their label index."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        labels = detection3d_gt_bboxes_3d.bbox_labels.to(torch.int64)
        bboxes_to_keep_mask = torch.isin(labels, self.label_indices_to_keep.to(labels.device))

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
