"""Tests for the PTv3 joint segmentation and Sparse4D detection model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.ptv3 import PTv3DetFeatureFusion, rebuild_coarse_point
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
    build_seg_head,
)
from autoware_ml.models.detection3d.heads.sparse4d import ANCHOR_DIM, TEMPORAL_STATE_INPUT_NAMES
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.multi.ptv3_segdet_sparse4d import PTv3SegDetSparse4DModel
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_model(tmp_path: Path, **head_overrides: object) -> PTv3SegDetSparse4DModel:
    """Build a small model whose head samples the stage the deepest test stage fuses into."""
    return PTv3SegDetSparse4DModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg_head(),
        det_fusion=PTv3DetFeatureFusion(in_channels=16, skip_channels=8, out_channels=32),
        bbox_head=build_sparse4d_head(make_anchor_file(tmp_path), **head_overrides),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


def _frame_inputs(device: torch.device, timestamp: float) -> ModelBatchInputs:
    """One frame of the test scene with its frame metadata."""
    batch = build_inputs(device=device)
    detections = batch.multi_task_gt_batch.detection3d_gt_batch
    gt_batch = batch.multi_task_gt_batch._replace(
        detection3d_gt_batch=detections._replace(
            gt_bboxes_num_points=torch.full_like(detections.gt_bboxes_num_points, 5)
        ),
        frame_meta_batch=FrameMetaBatch(
            ego2globals=torch.eye(4, device=device)[None],
            scene_tokens=["db/scene/0"],
            timestamps_seconds=torch.tensor([timestamp], dtype=torch.float64, device=device),
        ),
    )
    return batch.replace(multi_task_gt_batch=gt_batch)


@requires_cuda_spconv
def test_temporal_model_trains_on_a_stream_and_reaches_every_parameter(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4, dn_query_budget=8).to(device)

    for frame_index in range(2):
        model.zero_grad()
        batch = _frame_inputs(device, 100.0 + 0.1 * frame_index)
        outputs = model(**model.forward_inputs(batch))
        metrics = model.compute_metrics(batch, outputs)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        assert outputs.detection3d().sparse4d_head().denoising is not None
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing == []

    # The second frame of the scene reads the instances the first one cached
    assert model.bbox_head.instance_bank._slot_valid.tolist() == [True]


@requires_cuda_spconv
def test_model_eval_output_and_predictions(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path).to(device).eval()
    batch = _frame_inputs(device, 100.0)

    with torch.no_grad():
        outputs = model(**model.forward_inputs(batch))
        eval_out = model.build_eval_output(batch, outputs)
        predictions = model.predict_outputs(batch, outputs)

    assert eval_out["scene_token"] == ["db/scene/0"]
    assert predictions.detection3d_predictions[0].bboxes_3d.shape[1] == 9
    assert predictions.segmentation3d_predictions is not None


@requires_cuda_spconv
def test_temporal_model_eval_output_carries_the_tracks_of_one_stream(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    eval_outputs = []
    with torch.no_grad():
        for frame_index in range(2):
            batch = _frame_inputs(device, 100.0 + 0.1 * frame_index)
            outputs = model(**model.forward_inputs(batch))
            eval_outputs.append(model.build_eval_output(batch, outputs))

    first, second = eval_outputs
    assert first["gt_observed"][0].all()
    assert first["stream_ids"][0] == second["stream_ids"][0]
    assert second["predictions"][0]["instance_ids"] is not None
    assert len(second["gt_instance_ids"][0]) == len(second["gt_boxes"][0])


@requires_cuda_spconv
def test_forward_inputs_need_frame_metadata(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    with pytest.raises(ValueError, match="frame metadata"):
        model.forward_inputs(build_inputs(device=torch.device("cuda")))


@requires_cuda_spconv
def test_split_export_decodes_like_the_model_and_carries_the_bank_state(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    specs = model.build_export_specs(_frame_inputs(device, 100.0))
    spec = specs["ptv3_det3d_head"]
    stage_args, reset_state = spec.args[:4], spec.args[4:]
    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args)))
        # The det branch of the model on the same encoder stages
        feat, grid_coord, offset = model.det_fusion(rebuild_coarse_point(*stage_args))
        reference = model.bbox_head.forward_temporal_export(feat, grid_coord, offset, *reset_state)

    assert list(specs) == ["ptv3_encoder", "ptv3_seg3d_head", "ptv3_det3d_head"]
    assert spec.input_param_names[4:] == list(TEMPORAL_STATE_INPUT_NAMES)
    for name in spec.output_names:
        assert torch.allclose(first[name], reference[name], atol=1e-5), name
    assert first["temp_anchor_out"].shape == (1, 4, ANCHOR_DIM)
    assert spec.output_names[-1] == "temp_query_index_out"
    assert first["temp_query_index_out"].shape == (1, 4)

    # The next frame reads the state the first one returned
    next_state = (
        first["temp_instance_feature_out"],
        first["temp_anchor_out"],
        first["temp_confidence_out"],
        torch.ones(1, device=device),
        torch.full((1,), 0.1, device=device),
        torch.eye(4, device=device)[None],
    )
    with torch.no_grad():
        second = dict(zip(spec.output_names, spec.module(*stage_args, *next_state)))
    assert second["classification"].shape == first["classification"].shape
    assert not torch.equal(second["classification"], first["classification"])
