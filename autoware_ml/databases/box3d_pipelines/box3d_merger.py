"""
This script defines classes to merge 3D bounding boxes based on the different implementations.
"""

from typing import Sequence, Tuple, Set
from types import MappingProxyType

import numpy as np
from jaxtyping import Float64
from shapely.geometry import Polygon

from autoware_ml.types.geometry import Box3DFieldIndex
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DMerger(Box3DPipeline):
    """
    Base class for merging 3D bounding boxes. The merger runs on fine label names: a pair of
    boxes carrying the two source labels of a target is replaced by one box carrying the
    target label, and the class index of the level is assigned afterwards.
    """

    def __init__(
        self,
        target_labels: MappingProxyType[str, Sequence[str]],
        proximity_distance_threshold: float,
    ):
        """
        Initialize Box3DMerger.

        Args:
          target_labels: Mapping of the target fine label to its two source fine labels. The
            target is one of the two sources, the other source is absorbed into it.
          proximity_distance_threshold: Proximity distance threshold to check if two boxes are
            close to each other.
        """
        super().__init__()
        self.target_labels = target_labels
        self.proximity_distance_threshold = proximity_distance_threshold

        for target_label, source_labels in self.target_labels.items():
            if len(source_labels) != 2:
                raise ValueError(
                    f"Source labels for target label {target_label} "
                    f"must have exactly 2 labels, but it's {len(source_labels)}"
                )
            if target_label not in source_labels:
                raise ValueError(
                    f"Target label {target_label} must be one of its source labels "
                    f"{list(source_labels)}."
                )

        if self.proximity_distance_threshold <= 0:
            raise ValueError("Proximity distance threshold must be positive")

    def absorbed_labels(self) -> Set[str]:
        """
        Fine labels the merger folds into another label.

        Returns:
          Set[str]: Source labels that differ from their target.
        """

        return {
            source_label
            for target_label, source_labels in self.target_labels.items()
            for source_label in source_labels
            if source_label != target_label
        }

    def validate_taxonomy(self, taxonomy: LabelTaxonomy) -> None:
        """
        Reject a source label the vocabulary does not know, so a misspelled label cannot
        disable the merger, and a level that trains an absorbed label apart from its target,
        because the merge would move boxes between classes.

        Args:
          taxonomy: Taxonomy the boxes are baked with.
        """

        fine_names = set(taxonomy.vocabulary.fine_names)
        source_labels = {label for labels in self.target_labels.values() for label in labels}
        unknown_sources = sorted(source_labels - fine_names)
        if unknown_sources:
            raise ValueError(
                f"Merger source labels {unknown_sources} are not fine labels of the vocabulary."
            )
        for target_label, labels in self.target_labels.items():
            target_class = taxonomy.class_name(target_label)
            for source_label in labels:
                source_class = taxonomy.class_name(source_label)
                if source_class is not None and source_class != target_class:
                    raise ValueError(
                        f"The merger folds {source_label!r} into {target_label!r}, but the "
                        f"taxonomy trains them apart as {source_class!r} and {target_class!r}. "
                        "Use the pipelines that keep them."
                    )

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Process the boxes 3D metadata.
        """
        merged_boxes_3d, merged_indices = self.merge(boxes3d_data_model=boxes3d_data_model)

        new_boxes3d_data_model = []
        for index, box3d_data_model in enumerate(boxes3d_data_model):
            # Remove the merged boxes from the boxes3d_data_model
            if index not in merged_indices:
                new_boxes3d_data_model.append(box3d_data_model)

        # Merge the merged boxes with the boxes3d_data_model
        new_boxes3d_data_model.extend(merged_boxes_3d)
        return new_boxes3d_data_model

    def _check_boxes_overlap(
        self,
        first_box3d: Float64[np.ndarray, " num_box_fields"],
        second_box3d: Float64[np.ndarray, " num_box_fields"],
    ) -> bool:
        """
        Check if two 3D bounding boxes overlap in 2D projection.

        Args:
          first_box3d: Bounding box 1, please check Box3DFieldIndex for the field indices.
          second_box3d: Bounding box 2, please check Box3DFieldIndex for the field indices.

        Returns:
          bool: True if the two boxes overlap, False otherwise.
        """

        # Compute corners for two boxes
        polygons = []
        for box3d in [first_box3d, second_box3d]:
            x, y, length, width, yaw = (
                box3d[Box3DFieldIndex.X],
                box3d[Box3DFieldIndex.Y],
                box3d[Box3DFieldIndex.LENGTH],
                box3d[Box3DFieldIndex.WIDTH],
                box3d[Box3DFieldIndex.YAW],
            )
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            half_length = length / 2
            half_width = width / 2

            corners = np.array(
                [
                    [
                        x - half_length * cos_yaw + half_width * sin_yaw,
                        y - half_length * sin_yaw - half_width * cos_yaw,
                    ],
                    [
                        x + half_length * cos_yaw + half_width * sin_yaw,
                        y + half_length * sin_yaw - half_width * cos_yaw,
                    ],
                    [
                        x + half_length * cos_yaw - half_width * sin_yaw,
                        y + half_length * sin_yaw + half_width * cos_yaw,
                    ],
                    [
                        x - half_length * cos_yaw - half_width * sin_yaw,
                        y - half_length * sin_yaw + half_width * cos_yaw,
                    ],
                ],
            )
            polygons.append(corners)

        polygon_1 = Polygon(polygons[0])
        polygon_2 = Polygon(polygons[1])
        return polygon_1.intersects(polygon_2)

    def _check_boxes_proximity(
        self,
        first_box3d: Float64[np.ndarray, " num_box_fields"],
        second_box3d: Float64[np.ndarray, " num_box_fields"],
    ) -> bool:
        """
        Check if two 3D bounding boxes are close to each other by
          checking distance between their front and back face centers.
        Args:
          first_box3d: Bounding box 1, please check Box3DFieldIndex for the field indices.
          second_box3d: Bounding box 2, please check Box3DFieldIndex for the field indices.

        Returns:
          bool: True if the two boxes are close to each other, False otherwise.
        """

        front_centers = []
        back_centers = []
        for box3d in [first_box3d, second_box3d]:
            x, y, z, length, yaw = (
                box3d[Box3DFieldIndex.X],
                box3d[Box3DFieldIndex.Y],
                box3d[Box3DFieldIndex.Z],
                box3d[Box3DFieldIndex.LENGTH],
                box3d[Box3DFieldIndex.YAW],
            )

            front_center = np.array([x + length / 2 * np.cos(yaw), y + length / 2 * np.sin(yaw), z])
            back_center = np.array([x - length / 2 * np.cos(yaw), y - length / 2 * np.sin(yaw), z])
            front_centers.append(front_center)
            back_centers.append(back_center)

        # Total of 4 combinations to check
        if np.linalg.norm(front_centers[0] - front_centers[1]) <= self.proximity_distance_threshold:
            return True
        if np.linalg.norm(front_centers[0] - back_centers[1]) <= self.proximity_distance_threshold:
            return True
        if np.linalg.norm(back_centers[0] - front_centers[1]) <= self.proximity_distance_threshold:
            return True
        if np.linalg.norm(back_centers[0] - back_centers[1]) <= self.proximity_distance_threshold:
            return True

        return False

    def match_boxes_3d(
        self,
        boxes3d_params: Float64[np.ndarray, "num_boxes num_box_fields"],
        boxes3d_label_names: Sequence[str],
    ) -> MappingProxyType[str, Sequence[Tuple[int, int]]]:
        """
        Match 3D bounding boxes based on the target labels and source labels.

        Args:
          boxes3d_params: 3D bounding boxes, please check Box3DFieldIndex for the field indices.
          boxes3d_label_names: 3D bounding box label names.

        Returns:
          MappingProxyType[str, Sequence[Tuple[int, int]]]: Mapping of target labels to matched pairs of box indices.
        """
        # {target_class: [(source_box_index, source_box_index), ...]}
        matched_pairs = {target_label: [] for target_label in self.target_labels.keys()}
        for target_label, source_labels in self.target_labels.items():
            pairs = []
            first_box3d_indices = [
                box_index
                for box_index, box_label_name in enumerate(boxes3d_label_names)
                if box_label_name == source_labels[0]
            ]
            second_box3d_indices = [
                box_index
                for box_index, box_label_name in enumerate(boxes3d_label_names)
                if box_label_name == source_labels[1]
            ]

            first_box3d_fields = boxes3d_params[first_box3d_indices]
            second_box3d_fields = boxes3d_params[second_box3d_indices]

            for first_box3d_index, first_box3d_field in zip(
                first_box3d_indices, first_box3d_fields
            ):
                for second_box3d_index, second_box3d_field in zip(
                    second_box3d_indices, second_box3d_fields
                ):
                    if self._check_boxes_overlap(
                        first_box3d_field, second_box3d_field
                    ) or self._check_boxes_proximity(first_box3d_field, second_box3d_field):
                        pairs.append((first_box3d_index, second_box3d_index))

            matched_pairs[target_label].extend(pairs)
        return matched_pairs

    def merge(
        self,
        boxes3d_data_model: Sequence[Box3DDataModel],
    ) -> Tuple[Sequence[Box3DDataModel], Set[int]]:
        """
        Merge 3D bounding boxes based on the target labels and source labels by following the steps:
          1) Match boxes based on the target labels and source labels
          2) Merge boxes for each target label
          3) Return the merged boxes metadata

        Args:
          boxes3d_data_model: Sequence of Box3DDataModel of the 3D bounding boxes.

        Returns:
          Sequence[Box3DDataModel]: Merged 3D bounding boxes that saves sequence of merged boxes metadata.
          merged_indices: Set of indices of the merged boxes.
        """
        # 1) Match boxes based on the target labels and source labels
        matched_pairs = self.match_boxes_3d(
            boxes3d_params=np.asarray([box3d.box3d_params for box3d in boxes3d_data_model]),
            boxes3d_label_names=[box3d.box3d_label_name for box3d in boxes3d_data_model],
        )

        # 2) Merge boxes for each target label
        merged_boxes_3d: Sequence[Box3DDataModel] = []
        merged_indices = set()

        # Merge boxes for each target label
        for target_label, pairs in matched_pairs.items():
            for box3d_idx_1, box3d_idx_2 in pairs:
                # Skip if one of the boxes have already been merged
                if box3d_idx_1 in merged_indices or box3d_idx_2 in merged_indices:
                    continue

                merged_box3d_params = self.merge_boxes_3d(
                    first_box3d=boxes3d_data_model[box3d_idx_1].box3d_params,
                    second_box3d=boxes3d_data_model[box3d_idx_2].box3d_params,
                )

                # Always pick the first box's instance ID and dataset label name. The class
                # index of the level is assigned after the pipelines, so the merged box keeps
                # the placeholder index of the first box.
                merged_box3d_instance_id = boxes3d_data_model[box3d_idx_1].box3d_instance_id
                merged_box3d_dataset_label_name = boxes3d_data_model[
                    box3d_idx_1
                ].box3d_dataset_label_name
                merged_box3d_label_name = target_label
                merged_box3d_label_index = boxes3d_data_model[box3d_idx_1].box3d_label_index
                merged_box3d_num_lidar_points = (
                    boxes3d_data_model[box3d_idx_1].box3d_num_lidar_points
                    + boxes3d_data_model[box3d_idx_2].box3d_num_lidar_points
                )
                merged_box3d_num_radar_points = (
                    boxes3d_data_model[box3d_idx_1].box3d_num_radar_points
                    + boxes3d_data_model[box3d_idx_2].box3d_num_radar_points
                )
                merged_box3d_valid = (
                    boxes3d_data_model[box3d_idx_1].box3d_valid
                    or boxes3d_data_model[box3d_idx_2].box3d_valid
                )
                merged_box3d_attributes = boxes3d_data_model[box3d_idx_1].box3d_attributes.union(
                    boxes3d_data_model[box3d_idx_2].box3d_attributes
                )
                merged_box3d_coordinate = boxes3d_data_model[box3d_idx_1].box3d_coordinate

                merged_boxes_3d.append(
                    Box3DDataModel(
                        box3d_params=merged_box3d_params,
                        box3d_instance_id=merged_box3d_instance_id,
                        box3d_dataset_label_name=merged_box3d_dataset_label_name,
                        box3d_label_name=merged_box3d_label_name,
                        box3d_label_index=merged_box3d_label_index,
                        box3d_num_lidar_points=merged_box3d_num_lidar_points,
                        box3d_num_radar_points=merged_box3d_num_radar_points,
                        box3d_valid=merged_box3d_valid,
                        box3d_attributes=merged_box3d_attributes,
                        box3d_coordinate=merged_box3d_coordinate,
                    )
                )
                merged_indices.add(box3d_idx_1)
                merged_indices.add(box3d_idx_2)

        return merged_boxes_3d, merged_indices

    def merge_boxes_3d(
        self,
        first_box3d: Float64[np.ndarray, " num_box_fields"],
        second_box3d: Float64[np.ndarray, " num_box_fields"],
    ) -> Float64[np.ndarray, " num_box_fields"]:
        """
        Merge two 3D bounding boxes. This function is implemented in the subclass.
        Args:
          first_box3d: First 3D bounding box.
          second_box3d: Second 3D bounding box.

        Returns:
          Float64[np.ndarray, " num_box_fields"]: Merged 3D bounding box.
        """

        raise NotImplementedError("Subclass must implement this method")


class Box3DExtendLongerMerger(Box3DMerger):
    """
    Extend the longer box by elongating the larger box to the projection point.
    Please refer to the following presentation for more details:
    https://docs.google.com/presentation/d/17802H6gqApU3mHN2Q5XUcqa_qR5y5a_76QMM2F_9WW8/edit#slide=id.g20a727e0846_3_0
    """

    def __init__(
        self,
        target_labels: MappingProxyType[str, Sequence[str]],
        proximity_distance_threshold: float,
    ):
        """
        Initialize Box3DExtendLongerMerger.

        Args:
          target_labels: Mapping of the target fine label to its two source fine labels.
          proximity_distance_threshold: Proximity distance threshold to check if two boxes are
            close to each other.
        """

        super().__init__(
            target_labels=target_labels,
            proximity_distance_threshold=proximity_distance_threshold,
        )

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        target_labels = ", ".join(
            f"{target_label}: {list(source_labels)}"
            for target_label, source_labels in sorted(self.target_labels.items())
        )
        return (
            f"{self.__class__.__name__}(target_labels=({target_labels}), "
            f"proximity_distance_threshold={self.proximity_distance_threshold})"
        )

    @staticmethod
    def _get_box_faces(
        box: Float64[np.ndarray, " num_box_fields"],
    ) -> Tuple[
        Float64[np.ndarray, " 2"],
        Float64[np.ndarray, " 2"],
        Float64[np.ndarray, " 2"],
        float,
        float,
    ]:
        """
        Get the faces of a 3D bounding box.

        Args:
          box: Bounding box, please check Box3DFieldIndex for the field indices.

        Returns:
          Tuple of the center, face1 center, face2 center, length, and width.
        """

        x, y, length, width, yaw = (
            box[Box3DFieldIndex.X],
            box[Box3DFieldIndex.Y],
            box[Box3DFieldIndex.LENGTH],
            box[Box3DFieldIndex.WIDTH],
            box[Box3DFieldIndex.YAW],
        )
        center = np.array([x, y])
        if length >= width:
            face1_center = np.array(
                [x + (length / 2) * np.cos(yaw), y + (length / 2) * np.sin(yaw)]
            )
            face2_center = np.array(
                [x - (length / 2) * np.cos(yaw), y - (length / 2) * np.sin(yaw)]
            )
        else:
            face1_center = np.array(
                [
                    x + (width / 2) * np.cos(yaw + np.pi / 2),
                    y + (width / 2) * np.sin(yaw + np.pi / 2),
                ]
            )
            face2_center = np.array(
                [
                    x - (width / 2) * np.cos(yaw + np.pi / 2),
                    y - (width / 2) * np.sin(yaw + np.pi / 2),
                ]
            )
        return center, face1_center, face2_center, length, width

    def merge_boxes_3d(
        self,
        first_box3d: Float64[np.ndarray, " num_box_fields"],
        second_box3d: Float64[np.ndarray, " num_box_fields"],
    ) -> Float64[np.ndarray, " num_box_fields"]:
        """
        Gives impression of merging two 3D bounding boxes by elongating the larger box.

        The function identifies the larger and smaller box based on their area in the XY plane.
        The center of the farther end of the smaller box is rotated to meet the length axis of the
        larger box. Then, the larger box is elongated upto that point.

        Args:
          first_box3d: Bounding box 1, please check Box3DFieldIndex for the field indices.
          second_box3d: Bounding box 2, please check Box3DFieldIndex for the field indices.

        Returns:
          Float64[np.ndarray, " num_box_fields"]: Merged 3D bounding box, please check
            Box3DFieldIndex for the field indices.
        """
        # Identify the centers and faces of both boxes
        box1_center, box1_face1, box1_face2, length_1, width_1 = self._get_box_faces(first_box3d)
        box2_center, box2_face1, box2_face2, length_2, width_2 = self._get_box_faces(second_box3d)

        # Determine which box is larger
        if length_1 * width_1 >= length_2 * width_2:
            (
                larger_box_center,
                larger_box_face1,
                larger_box_face2,
                larger_length,
                larger_width,
                larger_box,
            ) = (
                box1_center,
                box1_face1,
                box1_face2,
                length_1,
                width_1,
                first_box3d,
            )
            smaller_box_center, smaller_box_face1, smaller_box_face2 = (
                box2_center,
                box2_face1,
                box2_face2,
            )
        else:
            (
                larger_box_center,
                larger_box_face1,
                larger_box_face2,
                larger_length,
                larger_width,
                larger_box,
            ) = (
                box2_center,
                box2_face1,
                box2_face2,
                length_2,
                width_2,
                second_box3d,
            )
            smaller_box_center, smaller_box_face1, smaller_box_face2 = (
                box1_center,
                box1_face1,
                box1_face2,
            )

        # Choose the farther face of the smaller box
        dist_to_smaller_face1 = np.linalg.norm(smaller_box_face1 - larger_box_center)
        dist_to_smaller_face2 = np.linalg.norm(smaller_box_face2 - larger_box_center)
        if dist_to_smaller_face1 > dist_to_smaller_face2:
            selected_smaller_face = smaller_box_face1
        else:
            selected_smaller_face = smaller_box_face2

        # Choose the nearer face of the larger box
        dist_to_larger_face1 = np.linalg.norm(larger_box_face1 - smaller_box_center)
        dist_to_larger_face2 = np.linalg.norm(larger_box_face2 - smaller_box_center)
        if dist_to_larger_face1 < dist_to_larger_face2:
            selected_larger_face = larger_box_face1
        else:
            selected_larger_face = larger_box_face2

        # Find the projection point on the axis of the larger box
        axis_vector = selected_larger_face - larger_box_center
        axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)
        to_smaller_box = selected_smaller_face - larger_box_center
        projection_length = np.dot(to_smaller_box, axis_vector_normalized)
        projection_point = larger_box_center + projection_length * axis_vector_normalized

        # Elongate the larger box to the projection point
        elongation_vector = projection_point - selected_larger_face
        elongation_length = np.linalg.norm(elongation_vector)

        merged_length = (
            larger_length + elongation_length if larger_length >= larger_width else larger_length
        )
        merged_width = (
            larger_width + elongation_length if larger_width > larger_length else larger_width
        )

        # Adjust the center minimally to balance the elongation
        elongation_shift = elongation_vector / 2
        merged_center = larger_box_center + elongation_shift

        box1_bottom_z = first_box3d[Box3DFieldIndex.Z] - (first_box3d[Box3DFieldIndex.HEIGHT] / 2)
        box2_bottom_z = second_box3d[Box3DFieldIndex.Z] - (second_box3d[Box3DFieldIndex.HEIGHT] / 2)
        merged_bottom_z = min(box1_bottom_z, box2_bottom_z)

        box1_top_z = first_box3d[Box3DFieldIndex.Z] + (first_box3d[Box3DFieldIndex.HEIGHT] / 2)
        box2_top_z = second_box3d[Box3DFieldIndex.Z] + (second_box3d[Box3DFieldIndex.HEIGHT] / 2)
        merged_top_z = max(box1_top_z, box2_top_z)

        merged_dz = merged_top_z - merged_bottom_z
        # New merged box center z is the middle point between merged bottom z and merged top z
        merged_z = merged_bottom_z + (merged_dz / 2)

        # Keep the orientation (yaw) of the larger box
        merged_yaw = larger_box[Box3DFieldIndex.YAW]

        # Merge the velocity by averaging the velocities of the two boxes
        merged_velocity_x = (
            first_box3d[Box3DFieldIndex.VELOCITY_X] + second_box3d[Box3DFieldIndex.VELOCITY_X]
        ) / 2
        merged_velocity_y = (
            first_box3d[Box3DFieldIndex.VELOCITY_Y] + second_box3d[Box3DFieldIndex.VELOCITY_Y]
        ) / 2
        merged_velocity_z = (
            first_box3d[Box3DFieldIndex.VELOCITY_Z] + second_box3d[Box3DFieldIndex.VELOCITY_Z]
        ) / 2

        merged_box3d = np.array(
            [
                merged_center[0],
                merged_center[1],
                merged_z,
                merged_length,
                merged_width,
                merged_dz,
                merged_yaw,
                merged_velocity_x,
                merged_velocity_y,
                merged_velocity_z,
            ]
        )

        return merged_box3d
