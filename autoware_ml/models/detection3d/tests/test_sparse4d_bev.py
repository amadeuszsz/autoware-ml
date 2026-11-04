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

"""Tests for the dense BEV branch of the Sparse4D head."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.heads.sparse4d import (
    BEVDeformableAggregation,
    Sparse4DHead,
    SparseBox3DKeyPointsGenerator,
    SparseDeformableAggregation,
)
from autoware_ml.models.detection3d.ptv3 import PTv3BEVEncoder, PTv3StageBEVPyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_recipe,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel

BEV_CHANNELS = 16


def _pyramid(
    stage_strides: tuple[int, ...] = (1,),
    projected_strides: tuple[int, ...] = (1,),
    num_downsampled: int = 0,
    point_cloud_range: list[float] = POINT_CLOUD_RANGE,
) -> PTv3StageBEVPyramid:
    return PTv3StageBEVPyramid(
        point_cloud_range=point_cloud_range,
        grid_size=1.0,
        stage_strides=stage_strides,
        projected_strides=projected_strides,
        in_channels=32,
        channels=BEV_CHANNELS,
        bev_encoder=PTv3BEVEncoder(BEV_CHANNELS, BEV_CHANNELS, BEV_CHANNELS, dilations=[1]),
        num_downsampled=num_downsampled,
    )


def _stage(
    grid_dims: tuple[int, int, int], num_voxels: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One sample of random voxels inside a stage grid."""
    generator = torch.Generator().manual_seed(seed)
    grid_coord = torch.stack(
        [torch.randint(0, size, (num_voxels,), generator=generator) for size in grid_dims], dim=1
    )
    feat = torch.randn((num_voxels, 32), generator=generator)
    return feat, grid_coord, torch.tensor([num_voxels])


def _bev_head(lidar_sampling: str, **overrides: object) -> Sparse4DHead:
    return build_sparse4d_head(
        make_anchor_recipe(),
        lidar_sampling=lidar_sampling,
        bev=_pyramid(num_downsampled=int(lidar_sampling == "bev")),
        bev_channels=BEV_CHANNELS,
        **overrides,
    )


def _block(**overrides: object) -> SparseDeformableAggregation:
    kwargs: dict[str, object] = {
        "embed_dims": 32,
        "num_groups": 4,
        "num_levels": 1,
        "kps_generator": SparseBox3DKeyPointsGenerator(
            embed_dims=32, fix_scale=((0.0, 0.0, 0.0),), num_learnable_pts=0
        ),
    }
    kwargs.update(overrides)
    return SparseDeformableAggregation(**kwargs)


class TestSparse4DBEV(unittest.TestCase):
    def test_pyramid_projects_the_stages_and_halves_the_encoded_map(self) -> None:
        pyramid = _pyramid(
            stage_strides=(1, 2),
            projected_strides=(1, 2),
            num_downsampled=1,
            point_cloud_range=[0.0, 0.0, 0.0, 8.0, 8.0, 8.0],
        ).eval()
        stages = [_stage((8, 8, 8), 32), _stage((4, 4, 4), 16, seed=1)]

        levels = pyramid(stages)

        self.assertEqual(
            [tuple(level.feat.shape) for level in levels],
            [(1, BEV_CHANNELS, 8, 8), (1, BEV_CHANNELS, 4, 4), (1, BEV_CHANNELS, 2, 2)],
        )
        self.assertEqual([level.cell_size for level in levels], [1.0, 2.0, 4.0])
        self.assertEqual(
            [level.grid_origin for level in levels], [(0.0, 0.0), (0.0, 0.0), (-1.0, -1.0)]
        )

    def test_a_downsampled_level_reads_the_cell_its_convolution_centers_on(self) -> None:
        """
        Input: a halving convolution that copies the center tap, over a map with one hot cell.
        Expected: the coarse level reads the hot value at the metric center of the fine cell, which
          a stride 2 convolution with padding 1 centers its coarse cell on.
        Check: the sampled value at the fine cell center.
        """
        pyramid = _pyramid(num_downsampled=1, point_cloud_range=[0.0, 0.0, 0.0, 8.0, 8.0, 8.0])
        convolution = pyramid.downsample[0][0]
        with torch.no_grad():
            convolution.weight.zero_()
            convolution.weight[range(BEV_CHANNELS), range(BEV_CHANNELS), 1, 1] = 1.0
            fine = torch.zeros((1, BEV_CHANNELS, 8, 8))
            fine[..., 4, 2] = 1.0
            coarse = BEVLevel(convolution(fine), pyramid.grid_origins[1], pyramid.cell_sizes[1])

        # Fine cell (x 2, y 4) of 1 m centers at (2.5, 4.5)
        value = coarse.sample(torch.tensor([[[[2.5, 4.5, 0.0]]]]))

        self.assertTrue(torch.allclose(value, torch.ones_like(value)))

    def test_pyramid_rejects_a_stride_that_is_not_a_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "projected_strides"):
            _pyramid(stage_strides=(1, 2), projected_strides=(4,))

    def test_every_parameter_of_a_bev_head_trains(self) -> None:
        for lidar_sampling in ["sparse_bev", "bev"]:
            with self.subTest(lidar_sampling=lidar_sampling):
                head = _bev_head(lidar_sampling)
                head.cls_threshold_to_reg = 1.0
                stages = [_stage(grid.grid_dims, 48) for grid in head.level_grids]
                gt_boxes = [torch.tensor([[4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]

                outputs = head(stages)
                losses = head.loss(outputs, gt_boxes, [torch.tensor([1])], [torch.tensor([0])], [0])
                losses["loss"].backward()

                missing = [
                    name
                    for name, parameter in head.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                self.assertEqual(missing, [])
                self.assertTrue(
                    all(
                        (block.weights_fc is None) == (lidar_sampling == "bev")
                        for block in head.deformable
                    )
                )

    def test_a_bev_head_exports_the_training_forward(self) -> None:
        head = _bev_head("bev", num_temp_instances=4).eval()
        export_head = head.prepare_for_export()
        stages = [_stage(grid.grid_dims, 48) for grid in head.level_grids]

        with torch.no_grad():
            train_outputs = head(
                stages,
                timestamps_seconds=torch.tensor([100.0], dtype=torch.float64),
                lidar2global=torch.eye(4, dtype=torch.float64)[None],
                scene_tokens=["scene"],
            )
            export_outputs = export_head.forward_temporal_export(
                stages, *export_head.temporal_state_export_inputs(torch.device("cpu"))
            )

        self.assertTrue(
            torch.allclose(
                export_outputs["classification"], train_outputs.classification[:, -1], atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(export_outputs["prediction"], train_outputs.prediction[:, -1], atol=1e-5)
        )

    def test_head_rejects_bev_settings_that_do_not_go_together(self) -> None:
        cases = [
            ("bev", False, "BEV maps"),
            ("sparse", True, "BEV maps"),
            ("dense", True, "lidar_sampling"),
        ]
        for lidar_sampling, with_bev, match in cases:
            with self.subTest(lidar_sampling=lidar_sampling, with_bev=with_bev):
                with self.assertRaisesRegex(ValueError, match):
                    build_sparse4d_head(
                        make_anchor_recipe(),
                        lidar_sampling=lidar_sampling,
                        bev=_pyramid() if with_bev else None,
                        bev_channels=BEV_CHANNELS,
                    )

    def test_a_block_samples_voxels_bev_maps_or_both(self) -> None:
        with self.assertRaisesRegex(ValueError, "sparse voxels, BEV maps or both"):
            _block(sample_sparse=False)

    def test_bev_branch_rejects_a_level_count_it_was_not_built_for(self) -> None:
        branch = BEVDeformableAggregation(
            embed_dims=32, bev_channels=BEV_CHANNELS, num_groups=4, num_levels=2, num_points=1
        )
        with self.assertRaisesRegex(ValueError, "2 levels"):
            branch(torch.zeros(1, 1, 32), torch.zeros(1, 1, 1, 3), [])
