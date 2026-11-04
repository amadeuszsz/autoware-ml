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

"""Tests for the export graph of the motion planner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from omegaconf import OmegaConf

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid, rebuild_point_chain
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_head,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_recipe,
)
from autoware_ml.models.motion.tests.online_map_fixtures import build_head as build_online_map_head
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.motion_head import MapContext
from autoware_ml.models.planning.motion_planning_head import MotionPlanningHead
from autoware_ml.models.planning.tests.planner_fixtures import build_planner
from autoware_ml.models.multi.ptv3_sparsedrive_motion import MOTION_MAP_INPUT_NAMES
from autoware_ml.models.multi.ptv3_sparsedrive import (
    MOTION_PLANNING_EXPORT_INPUT_NAMES,
    MOTION_PLANNING_EXPORT_OUTPUT_NAMES,
    PTv3SparseDriveModel,
)
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import export_to_onnx

DIMS, CLASSES, TEMPORAL = 32, 2, 4
requires_cuda_spconv = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def build_motion_head(
    tmp_path: Path, with_map: bool, det_strides: tuple[int, ...] = (1,)
) -> MotionPlanningHead:
    """A small motion planning head, the planner sampling the coarsest det head stage."""
    rng = np.random.default_rng(0)
    motion_anchors = FixedAnchorRecipe({"anchors": rng.normal(size=(CLASSES, 6, 4, 2))})
    torch.manual_seed(0)
    head = MotionPlanningHead(
        num_classes=CLASSES,
        embed_dims=DIMS,
        anchor_recipe=motion_anchors,
        planner=build_planner(
            tmp_path,
            DIMS,
            with_map=with_map,
            point_cloud_range=POINT_CLOUD_RANGE,
            feature_strides=det_strides[-1:],
        ),
        ego_height=1.5,
        num_heads=4,
        agent_steps=4,
        num_agent_keys=5,
        queue_length=2,
        queue_stride=1,
        num_map_keys=3 if with_map else 0,
    )
    return fill_anchors(head)


def build_model(
    tmp_path: Path, with_map: bool, det_strides: tuple[int, ...] = (1,)
) -> PTv3SparseDriveModel:
    """A small SparseDrive model, the planner sampling the coarsest stage the det head samples.

    The encoder has one stage more than the det head samples, the deepest one fused into them.
    """
    enc_channels = (8, 16, 32)[: len(det_strides) + 1]
    return PTv3SparseDriveModel(
        motion_head=build_motion_head(tmp_path, with_map, det_strides),
        online_map_head=build_online_map_head(tmp_path, num_temp_instances=2) if with_map else None,
        encoder=build_ptv3_encoder(enc_channels=enc_channels),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(
            dec_depths=(1,) * (len(enc_channels) - 1), enc_channels=enc_channels
        ),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=enc_channels, out_channels=DIMS),
        bbox_head=build_sparse4d_head(
            make_anchor_recipe(), num_temp_instances=TEMPORAL, feature_strides=det_strides
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


def frame_inputs(args: tuple, offset: int, frame: int) -> list[torch.Tensor]:
    """Detection outputs of one frame, ``offset`` stage inputs ahead, a stream from frame 1 on."""
    generator = torch.Generator().manual_seed(frame)
    device = args[0].device
    queries = args[offset].shape[1]
    anchor = args[offset + 1]
    prediction = anchor.cpu() + torch.randn(anchor.shape, generator=generator) * 0.1
    prediction[..., 6:8] = torch.nn.functional.normalize(prediction[..., 6:8], dim=-1)
    shift = torch.eye(4)[None].clone()
    shift[0, 0, 3] = -1.0
    tensors = [
        torch.randn(1, queries, DIMS, generator=generator),
        prediction,
        torch.randn(1, queries, CLASSES, generator=generator),
        torch.randn(1, queries, 2, generator=generator),
        torch.randint(0, queries, (1, TEMPORAL), generator=generator),
        torch.full((1,), float(frame > 0)),
        shift,
        torch.ones(1, dtype=torch.long),
    ]
    return [tensor.to(device) for tensor in tensors]


class TestMotionPlanningExport(unittest.TestCase):
    """The export graph of the motion planner."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    @requires_cuda_spconv
    def test_the_planner_graph_plans_a_stream_like_the_head(self) -> None:
        """
        Input: the planner export module run over two frames of one stream, the history and the ego
          velocity of the first frame feeding the second, with and without the map graph outputs.
        Expected: the graph takes the encoder stages of the detection graph first, every output of
          both frames matches the head stepping on the same fused stages in the ego frame.
        """
        for with_map in (False, True):
            with self.subTest(with_map=with_map):
                device = torch.device("cuda")
                model = build_model(self.tmp_path, with_map).to(device).eval()
                specs = model.build_export_specs(build_inputs(device=device))
                spec = specs["ptv3_motion_head"]
                det_spec = specs["ptv3_det3d_head"]
                num_stage_inputs = spec.input_names.index(MOTION_PLANNING_EXPORT_INPUT_NAMES[0])
                self.assertEqual(
                    spec.input_names[:num_stage_inputs], det_spec.input_names[:num_stage_inputs]
                )
                self.assertEqual(
                    spec.input_names[num_stage_inputs:],
                    [
                        *MOTION_PLANNING_EXPORT_INPUT_NAMES,
                        *(MOTION_MAP_INPUT_NAMES if with_map else ()),
                    ],
                )
                self.assertEqual(spec.output_names, list(MOTION_PLANNING_EXPORT_OUTPUT_NAMES))

                stage_args = spec.args[:num_stage_inputs]
                map_args = spec.args[len(spec.args) - 3 :] if with_map else ()
                skip_feat, deepest_feat, cluster, skip_grid_coord = stage_args
                with torch.no_grad():
                    stages = model.det_fusion(
                        rebuild_point_chain([skip_feat, deepest_feat], [cluster], [skip_grid_coord])
                    )
                head = model.motion_head.prepare_for_export()
                state = list(spec.args[num_stage_inputs + 4 : num_stage_inputs + 10])
                for frame in range(2):
                    detections = frame_inputs(spec.args, num_stage_inputs, frame)
                    with torch.no_grad():
                        outputs = spec.module(
                            *stage_args, *detections[:4], *state, *detections[4:], *map_args
                        )
                        map_context = None
                        if with_map:
                            map_context = MapContext(*map_args)
                        reference, next_state, ego_velocity = head.step(
                            detections[0],
                            detections[1],
                            detections[2],
                            spec.module.bbox_head.anchor_encoder,
                            stages[-1][0].mean(dim=0, keepdim=True),
                            state[-1],
                            QueueState(*state[:5]),
                            detections[5] > 0.5,
                            detections[4],
                            detections[6],
                            stages,
                            detections[7],
                            torch.eye(4, device=device)[None],
                            map_context,
                        )
                    named = dict(zip(spec.output_names, outputs, strict=True))
                    torch.testing.assert_close(named["plan_logits"], reference.plan_logits)
                    torch.testing.assert_close(
                        named["plan_trajectories"], reference.plan_trajectories
                    )
                    torch.testing.assert_close(
                        named["agent_trajectories"], reference.agent_trajectories
                    )
                    for name, expected in zip(
                        MOTION_PLANNING_EXPORT_OUTPUT_NAMES[7:], (*next_state, ego_velocity)
                    ):
                        torch.testing.assert_close(named[name], expected)
                    self.assertEqual(named["plan"].shape, (1, 2, 2))
                    state = list(outputs[7:])

    @requires_cuda_spconv
    def test_the_planner_graph_exports_with_one_aggregation_per_sampling(self) -> None:
        """
        Input: the planner graph of a model with two planning layers exported for deployment, the
          det head sampling the fused and the deepest stage, the planner only the deepest.
        Expected: the ONNX file passes the checker and samples the stages with one aggregation node
          per path layer and one for the trajectories of the last layer, the only custom nodes. The
          finer stage feeds no output, so the graph takes no input of it.
        """
        device = torch.device("cuda")
        model = build_model(self.tmp_path, with_map=False, det_strides=(1, 2)).to(device).eval()
        spec = model.build_export_specs(build_inputs(device=device))["ptv3_motion_head"]
        path = self.tmp_path / "ptv3_motion_head.onnx"
        export_to_onnx(
            spec,
            OmegaConf.create({"opset_version": 17, "dynamo": False}),
            path,
        )
        model_proto = onnx.load(str(path))
        onnx.checker.check_model(model_proto)
        custom = [node for node in model_proto.graph.node if node.domain not in ("", "ai.onnx")]
        self.assertEqual([node.op_type for node in custom], ["SparseDeformableAggregation"] * 3)
        self.assertEqual(
            [value.name for value in model_proto.graph.output],
            list(MOTION_PLANNING_EXPORT_OUTPUT_NAMES),
        )
        declared = [value.name for value in model_proto.graph.input]
        self.assertEqual(declared[0], "point_feat_1")
        self.assertFalse(any(name.endswith("_0") for name in declared))
