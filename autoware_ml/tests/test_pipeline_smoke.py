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

"""End to end smoke tests of the data pipeline, from the record table to a training step.

The tests build a small synthetic T4 corpus on disk, read it through the real dataset, the
real transform pipeline and the real runtime preprocessing, and run one optimizer step of
the PTv3 models on the result.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import MappingProxyType
from functools import partial
import unittest

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import LabelVocabulary, SegmentationTaxonomy
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.t4dataset.detection3d import T4Detection3DTask
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DTask
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import (
    build_bev_neck,
    build_preprocessor,
    build_ptv3_encoder,
    build_seg_head,
    build_seg_model,
    build_trans_model,
    build_transfusion_head,
)
from autoware_ml.models.multi.ptv3_segdet import PTv3SegDetModel
from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.filters import BBoxesBEVDistanceFilter
from autoware_ml.transforms.boxes3d.loading import BBoxesLabelNameFilter
from autoware_ml.transforms.point_cloud.geometry import (
    GlobalBEVRandomFlip,
    GlobalRotScaleTrans,
    PointsRangeFilter,
    RandomRotateTargetAngle,
)
from autoware_ml.transforms.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    SweepWindow,
    LoadPointsFromFile,
)
from autoware_ml.transforms.point_cloud.perturbation import RandomJitter, RandomStrengthJitter
from autoware_ml.transforms.point_cloud.sampling import RandomDropout
from autoware_ml.types.geometry import Box3DFieldIndex

NUM_POINTS = 512
# The record of a sample stores its own frame, then its past sweeps, then its future ones
NUM_PAST_SWEEPS = 1
NUM_FUTURE_SWEEPS = 1
NUM_SWEEPS = 1 + NUM_PAST_SWEEPS + NUM_FUTURE_SWEEPS
POINT_CLOUD_RANGE = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)
SCENE = "db_v1/scene_0/dataset_v1"

VOCABULARY = LabelVocabulary({"car": "car", "vehicle.car": "car", "unpainted": None})
SEGMENTATION_TAXONOMY = SegmentationTaxonomy(
    VOCABULARY, ("car",), {"car": "car"}, -1, {"vehicle": ("car",)}
)


def sweep_time_offset(sweep_index: int) -> float:
    """Seconds subtracted from the sample timestamp to date one stored frame.

    The sample owns position 0, the past sweeps follow it and then the future ones, so the
    loader reads the side of a frame off the sign of its time lag.

    Args:
        sweep_index: Position of the frame in the record.

    Returns:
        float: Offset of the frame, positive for a past sweep and negative for a future one.
    """
    if sweep_index == 0:
        return 0.0
    if sweep_index <= NUM_PAST_SWEEPS:
        return 0.05 * sweep_index
    return -0.05 * (sweep_index - NUM_PAST_SWEEPS)


def write_corpus(root: Path, num_records: int) -> pl.DataFrame:
    """Write a small synthetic corpus and return its record table.

    Args:
        root: Database root the relative paths resolve against.
        num_records: Number of frames to write.

    Returns:
        pl.DataFrame: The record table of the corpus.
    """
    rows = []
    for record_index in range(num_records):
        lidar_frames = []
        for sweep_index in range(NUM_SWEEPS):
            relative = f"{SCENE}/data/LIDAR_TOP/{record_index}_{sweep_index}.pcd.bin"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            points = np.random.default_rng(record_index * NUM_SWEEPS + sweep_index).uniform(
                low=[-6.0, -6.0, -1.0, 0.0, 0.0],
                high=[6.0, 6.0, 1.0, 1.0, 0.0],
                size=(NUM_POINTS, 5),
            )
            points.astype(np.float32).tofile(path)

            mask_relative = f"{SCENE}/data/LIDARSEG/{record_index}_{sweep_index}.bin"
            mask_path = root / mask_relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.zeros(NUM_POINTS, dtype=np.uint8).tofile(mask_path)

            lidar_frames.append(
                {
                    "lidar_frame_id": f"frame-{record_index}-{sweep_index}",
                    "lidar_keyframe": sweep_index == 0,
                    "lidar_sensor_id": "lidar",
                    "lidar_sensor_channel_name": "LIDAR_TOP",
                    "lidar_timestamp_seconds": float(record_index) - sweep_time_offset(sweep_index),
                    "lidar_pointcloud_path": str(path),
                    "lidar_pointcloud_source_path": None,
                    "lidar_pointcloud_num_features": 5,
                    "lidar_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
                    "lidar_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
                    "lidar_sensor_to_lidar_sweep_matrix": np.eye(4).tolist(),
                    "lidar_pointcloud_semantic_mask_path": str(mask_path),
                }
            )

        box = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
        box[Box3DFieldIndex.X] = 1.0
        box[Box3DFieldIndex.Y] = 1.0
        box[Box3DFieldIndex.Z] = 0.0
        box[Box3DFieldIndex.LENGTH] = 4.0
        box[Box3DFieldIndex.WIDTH] = 2.0
        box[Box3DFieldIndex.HEIGHT] = 1.5
        rows.append(
            {
                DatasetTableSchema.SCENARIO_ID.name: "scenario-0",
                DatasetTableSchema.SAMPLE_ID.name: f"sample-{record_index}",
                DatasetTableSchema.SAMPLE_INDEX.name: record_index,
                DatasetTableSchema.TIMESTAMP_SECONDS.name: float(record_index),
                DatasetTableSchema.LOCATION.name: "odaiba",
                DatasetTableSchema.VEHICLE_TYPE.name: "j6gen2",
                DatasetTableSchema.SCENARIO_NAME.name: "scenario",
                DatasetTableSchema.LIDAR_FRAMES.name: lidar_frames,
                DatasetTableSchema.LIDAR_SOURCES.name: [],
                DatasetTableSchema.IMAGE_FRAMES.name: [],
                DatasetTableSchema.CATEGORY_MAPPING.name: {
                    "category_names": ["car"],
                    "category_indices": [0],
                },
                DatasetTableSchema.BOXES_3D.name: [
                    {
                        "box3d_params": box.tolist(),
                        "box3d_instance_id": f"box-{record_index}",
                        "box3d_dataset_label_name": "vehicle.car",
                        "box3d_label_name": "car",
                        "box3d_label_index": 0,
                        "box3d_num_lidar_points": 64,
                        "box3d_num_radar_points": 0,
                        "box3d_valid": True,
                        "box3d_attributes": [],
                        "box3d_coordinate": "gravity_center",
                    }
                ],
            }
        )
    return pl.DataFrame(rows, schema=DatasetTableSchema.to_polars_schema())


def build_transforms() -> TransformsCompose:
    """Build the PTv3 training pipeline the task configs describe."""
    bev_range = (
        POINT_CLOUD_RANGE[0],
        POINT_CLOUD_RANGE[1],
        POINT_CLOUD_RANGE[3],
        POINT_CLOUD_RANGE[4],
    )
    return TransformsCompose(
        pipeline=[
            LoadPointsFromFile(use_dim=(0, 1, 2, 3)),
            LoadMultiSweepPointsFromFile(
                past=SweepWindow(
                    num=NUM_PAST_SWEEPS, time_lag_range=(0.01, 1.0), selection="random"
                ),
                future=SweepWindow(
                    num=NUM_FUTURE_SWEEPS, time_lag_range=(0.01, 1.0), selection="random"
                ),
                use_timestamp_difference=True,
                use_dim=(0, 1, 2, 3),
                bev_remove_radius=0.0,
            ),
            RandomRotateTargetAngle(probability=1.0, yaw_angle_ratios=[0.5, 1.0, 1.5]),
            GlobalRotScaleTrans(
                yaw_rot_range=(-0.2, 0.2),
                scale_ratio_range=(0.95, 1.05),
                translation_std=(0.1, 0.1, 0.05),
            ),
            GlobalBEVRandomFlip(),
            RandomJitter(sigma=0.01, clip=0.05),
            PointsRangeFilter(points_range=POINT_CLOUD_RANGE),
            RandomDropout(dropout_ratio=0.1, probability=1.0),
            RandomStrengthJitter(
                gamma_range=(0.8, 1.25), scale_range=(0.9, 1.1), shift_range=(-0.02, 0.02)
            ),
            BBoxesLabelNameFilter(label_names_to_keep=["car"]),
            BBoxesBEVDistanceFilter(bev_range=bev_range),
        ]
    )


def build_dataset(root: Path, records: pl.DataFrame) -> T4Dataset:
    """Build the dataset the datamodule serves, over the synthetic record table."""
    return T4Dataset(
        database_root_path=str(root),
        max_num_3d_gt_bboxes=8,
        dataset_records_dataframe=records,
        transforms=build_transforms(),
        dataset_tasks=MappingProxyType(
            {
                "Detection3D": T4Detection3DTask,
                "Segmentation3D": partial(T4Segmentation3DTask, taxonomy=SEGMENTATION_TAXONOMY),
            }
        ),
    )


class TestPipelineSmoke(unittest.TestCase):
    """The dataset, the transforms and the preprocessing carry a batch into the models."""

    def setUp(self) -> None:
        """Write a two frame corpus and collate a batch from it."""
        torch.manual_seed(0)
        np.random.seed(0)
        self.root = Path(tempfile.mkdtemp())
        self.records = write_corpus(self.root, num_records=2)
        self.dataset = build_dataset(self.root, self.records)
        self.batch = self.dataset.collate_fn([self.dataset[0], self.dataset[1]])

    def test_dataset_serves_a_collated_batch_of_every_task(self) -> None:
        assert isinstance(self.batch, ModelGTBatch)
        assert self.batch.point_cloud_gt_batch is not None
        assert self.batch.detection3d_gt_batch is not None
        assert self.batch.segmentation3d_gt_batch is not None
        assert self.batch.frame_meta_batch is not None
        self.assertEqual(int(self.batch.infer_batch_size()), 2)
        self.assertEqual(list(self.batch.frame_meta_batch.scene_tokens), [SCENE, SCENE])

    def test_every_point_keeps_its_semantic_label_through_the_pipeline(self) -> None:
        point_batch = self.batch.point_cloud_gt_batch
        segment_batch = self.batch.segmentation3d_gt_batch
        assert point_batch is not None and segment_batch is not None

        self.assertEqual(point_batch.points.shape[0], segment_batch.gt_semantic_masks.shape[0])
        self.assertEqual(point_batch.batch_indices.tolist(), segment_batch.batch_indices.tolist())

    def test_the_sweeps_carry_the_time_lag_and_the_ignore_label(self) -> None:
        point_batch = self.batch.point_cloud_gt_batch
        segment_batch = self.batch.segmentation3d_gt_batch
        assert point_batch is not None and segment_batch is not None

        time_lag = point_batch.points[:, point_batch.timestamp_difference_dim]
        self.assertEqual(point_batch.timestamp_difference_dim, 4)
        # A past sweep was captured before the sample and a future one after it, so the two
        # sides arrive with opposite signs and the current frame keeps the exact zero
        self.assertTrue(bool((time_lag > 0).any()))
        self.assertTrue(bool((time_lag < 0).any()))
        self.assertTrue(bool((time_lag == 0).any()))
        # The current frame is labelled and every sweep return takes the ignore index
        self.assertTrue(bool((segment_batch.gt_semantic_masks[time_lag != 0] == -1).all()))
        self.assertTrue(bool((segment_batch.gt_semantic_masks[time_lag == 0] >= 0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_segmentation_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = build_seg_model(POINT_CLOUD_RANGE).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs_dict = DataPreprocessing([build_preprocessor(POINT_CLOUD_RANGE)])(
            self.batch.to_device(device), is_training=True
        )

        loss = model.training_step(batch_inputs_dict, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_detection_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = build_trans_model(point_cloud_range=POINT_CLOUD_RANGE).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs_dict = DataPreprocessing([build_preprocessor(POINT_CLOUD_RANGE)])(
            self.batch.to_device(device), is_training=True
        )

        loss = model.training_step(batch_inputs_dict, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_segdet_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = PTv3SegDetModel(
            encoder=build_ptv3_encoder(),
            seg3d_head=build_seg_head(),
            bev_neck=build_bev_neck(),
            bbox_head=build_transfusion_head(POINT_CLOUD_RANGE),
            export_output_names=["pred_labels", "pred_probs"],
            optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
            grid_size=1.0,
            point_cloud_range=list(POINT_CLOUD_RANGE),
        ).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs_dict = DataPreprocessing([build_preprocessor(POINT_CLOUD_RANGE)])(
            self.batch.to_device(device), is_training=True
        )

        loss = model.training_step(batch_inputs_dict, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    def test_evaluation_step_builds_the_metric_frames(self) -> None:
        model = build_seg_model(POINT_CLOUD_RANGE)
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        device = next(model.parameters()).device
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs_dict = DataPreprocessing([build_preprocessor(POINT_CLOUD_RANGE)])(
            self.batch.to_device(device), is_training=False
        )

        if device.type != "cuda":
            self.skipTest("the PTv3 stem runs on CUDA only")
        with torch.no_grad():
            outputs = model(**model.bind_forward_inputs(batch_inputs_dict))
            eval_output = model.build_eval_output(batch_inputs_dict, outputs)

        frames = eval_output["seg_frames"]
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(frame["coord"].shape[0], frame["pred"].shape[0])
            self.assertEqual(frame["coord"].shape[0], frame["target"].shape[0])
            self.assertEqual(frame["scene_token"], SCENE)
