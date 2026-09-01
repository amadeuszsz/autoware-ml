"""Tests for FRNet-specific training behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from autoware_ml.datamodule.samples.batch import (
    Batch,
    FrameMetaBatch,
    PointCloudBatch,
    SegmentationBatch,
)
from autoware_ml.models.segmentation3d.frnet import FRNet
from autoware_ml.preprocessing.base import DataPreprocessing, ProcessedBatch
from autoware_ml.preprocessing.segmentation3d.frustum_range import (
    FrustumInputs,
    FrustumRangePreprocessor,
)
from autoware_ml.types.geometry import PointFeatureName

_FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


class _IdentityEncoder(torch.nn.Module):
    """Return passthrough encoder tensors for FRNet tests."""

    def forward(
        self,
        points: torch.Tensor,
        inverse_map: torch.Tensor,
        voxel_coors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Echo inputs as a one-level pyramid."""
        del inverse_map
        return voxel_coors, points, [points]


class _IdentityBackbone(torch.nn.Module):
    """Forward the encoder pyramid through unchanged for FRNet tests."""

    def __init__(self) -> None:
        super().__init__()
        self.last_sample_count: int | None = None

    def forward(
        self,
        point_feats_pyramid: list[torch.Tensor],
        voxel_feats: torch.Tensor,
        voxel_coors: torch.Tensor,
        point_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Record the sample count and echo the encoder pyramid as backbone output."""
        del voxel_coors, point_coors, inverse_map
        self.last_sample_count = sample_count
        return [voxel_feats], list(point_feats_pyramid)


class _DecodeHead(torch.nn.Module):
    """Provide the classifier interface expected by :class:`FRNet`."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the decode-head test double.

        Args:
            num_classes: Number of semantic classes.
        """
        super().__init__()
        self.classifier = torch.nn.Linear(4, num_classes)
        self.ignore_index = num_classes - 1

    def forward(
        self,
        point_coors: torch.Tensor,
        point_feats_encoder: list[torch.Tensor],
        voxel_feats_backbone: list[torch.Tensor],
        point_feats_backbone: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return a deterministic point logits tensor for tests."""
        del point_coors, point_feats_encoder, voxel_feats_backbone
        return self.classifier(point_feats_backbone[0])

    def loss(self, point_logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute a simple cross-entropy test loss."""
        return {"loss_ce": torch.nn.functional.cross_entropy(point_logits, target)}

    def predict(self, point_logits: torch.Tensor) -> torch.Tensor:
        """Return point-wise argmax predictions."""
        return point_logits.argmax(dim=1)


def _make_frnet(num_classes: int = 3) -> FRNet:
    """Return a minimal FRNet using identity encoder/backbone."""
    return FRNet(
        voxel_encoder=_IdentityEncoder(),
        backbone=_IdentityBackbone(),
        decode_head=_DecodeHead(num_classes=num_classes),
        optimizer=torch.optim.AdamW,
    )


def _make_typed_batch(points: torch.Tensor, labels: torch.Tensor) -> Batch:
    """Return a one-sample typed batch with points and segmentation labels."""
    return Batch(
        meta=FrameMetaBatch(
            sample_ids=("sample-0",),
            scene_tokens=None,
            timestamps=(0.0,),
            ego2globals=None,
            prev_exists=None,
        ),
        point_cloud=PointCloudBatch(
            features=(points,),
            feature_names=_FEATURE_NAMES,
            num_current_points=(points.shape[0],),
        ),
        segmentation=SegmentationBatch(labels=(labels,)),
    )


def _make_processed(num_points: int = 5, num_classes: int = 3) -> ProcessedBatch:
    """Return a minimal processed batch with hand-built frustum inputs."""
    points = torch.rand(num_points, 4)
    coors = torch.stack(
        [
            torch.zeros(num_points, dtype=torch.long),
            torch.arange(num_points, dtype=torch.long) % 2,
            torch.arange(num_points, dtype=torch.long) % 2,
        ],
        dim=1,
    )
    voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
    pts_semantic_mask = torch.randint(0, num_classes - 1, (num_points,))
    frustum_inputs = FrustumInputs(
        points=points,
        coors=coors,
        voxel_coors=voxel_coors,
        inverse_map=inverse_map,
        sample_count=1,
        pts_semantic_mask=pts_semantic_mask,
        semantic_seg=torch.zeros(1, 2, 2, dtype=torch.long),
    )
    return ProcessedBatch(
        batch=_make_typed_batch(points, pts_semantic_mask),
        inputs=(frustum_inputs,),
    )


def test_frnet_shared_step_returns_scalar_loss_with_grad() -> None:
    """_shared_step should return a differentiable scalar loss tensor."""
    model = _make_frnet()
    model.log_dict = MagicMock()
    processed = _make_processed()

    metrics, _ = model._shared_step(processed, "train")

    assert "loss" in metrics
    assert metrics["loss"].shape == torch.Size([])
    assert metrics["loss"].requires_grad
    assert model.backbone.last_sample_count == 1


def test_frnet_get_log_batch_size_counts_batch_samples() -> None:
    model = _make_frnet(num_classes=4)
    processed = _make_processed(num_points=8, num_classes=4)

    assert model.get_log_batch_size(processed) == processed.batch.batch_size == 1


def test_frnet_forward_uses_explicit_sample_count() -> None:
    model = _make_frnet(num_classes=4)
    points = torch.rand(4, 4)
    coors = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

    outputs = model(points, coors, voxel_coors, inverse_map, sample_count=2)

    point_logits, *voxel_feats = outputs
    assert point_logits.shape == (4, 4)
    assert len(voxel_feats) == 1  # _IdentityBackbone returns a single-level pyramid
    assert model.backbone.last_sample_count == 2


def test_frnet_with_preprocessing_runs_shared_step_end_to_end() -> None:
    """A real preprocessor attached via set_data_preprocessing should feed FRNet."""
    model = _make_frnet(num_classes=3)
    model.log_dict = MagicMock()
    preprocessor = FrustumRangePreprocessor(
        height=2,
        width=4,
        fov_up=10.0,
        fov_down=-10.0,
        ignore_index=2,
        num_classes=3,
    )
    model.set_data_preprocessing(DataPreprocessing([preprocessor]))

    points = torch.tensor(
        [[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2], [1.0, 1.0, 0.0, 0.3]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0], dtype=torch.long)

    processed = model.on_after_batch_transfer(_make_typed_batch(points, labels), dataloader_idx=0)
    metrics, _ = model._shared_step(processed, "train")

    assert "loss" in metrics
    assert metrics["loss"].requires_grad
    assert model.backbone.last_sample_count == 1
