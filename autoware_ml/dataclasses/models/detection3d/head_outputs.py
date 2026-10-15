"""
Modules to save raw outputs from a detection3d head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jaxtyping import Bool, Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class TransFusionSeparateHeadOutputs(BaseModel):
    """
    Dataclass to save the outputs (1D) from a separate head in a Transfusion-based 3D detection model.

    Attributes:
      heatmap: Heatmap to save probability for each class in a BEV heatmap.
      center: Center_x and center_y translation from each cell in a BEV heatmap.
      height: Height value from each cell in a BEV heatmap.
      dim: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rot: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vel: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    heatmap: Float[torch.Tensor, "batch_size num_classes num_proposals"]
    center: Float[torch.Tensor, "batch_size 2 num_proposals"]
    height: Float[torch.Tensor, "batch_size 1 num_proposals"]
    dim: Float[torch.Tensor, "batch_size 3 num_proposals"]
    rot: Float[torch.Tensor, "batch_size 2 num_proposals"]
    vel: Float[torch.Tensor, "batch_size 2 num_proposals"] | None

    @model_validator(mode="after")
    def _check_batch_and_proposals(self) -> TransFusionSeparateHeadOutputs:
        """Check that every branch covers the same batch and the same proposals as heatmap."""
        batch_size, _, num_proposals = self.heatmap.shape
        for name, tensor in (
            ("center", self.center),
            ("height", self.height),
            ("dim", self.dim),
            ("rot", self.rot),
            ("vel", self.vel),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2] != num_proposals:
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, "
                    f"num_proposals={num_proposals}), got {tuple(tensor.shape)}."
                )
        return self


class TransFusionHeadOutputs(BaseModel):
    """
    Dataclass to save Transfusion-based outputs from a 3D detection model.

    Attributes:
      dense_heatmap: Heatmap to save probability for each class in a BEV heatmap.
      query_heatmap_score: Heatmap scores gathered at each query position.
      query_labels: Class index predicted for each query.
      separate_head_outputs: Per-proposal outputs from the separate head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    dense_heatmap: Float[torch.Tensor, "batch_size num_classes height width"]
    query_heatmap_score: Float[torch.Tensor, "batch_size num_classes num_queries"]
    query_labels: Int64[torch.Tensor, "batch_size num_queries"]

    separate_head_outputs: TransFusionSeparateHeadOutputs

    @model_validator(mode="after")
    def _check_batch_classes_and_queries(self) -> TransFusionHeadOutputs:
        """
        Check that the dense, query and per-proposal outputs share a batch and class count.

        ``num_queries`` and ``num_proposals`` are deliberately not compared: with auxiliary
        decoder layers the separate head concatenates the proposals of every layer.
        """
        batch_size, num_classes = self.dense_heatmap.shape[:2]
        num_queries = self.query_labels.shape[1]
        if self.query_labels.shape[0] != batch_size:
            raise ValueError(
                f"query_labels must have batch_size={batch_size}, got {tuple(self.query_labels.shape)}."
            )
        if self.query_heatmap_score.shape != (batch_size, num_classes, num_queries):
            raise ValueError(
                f"query_heatmap_score must have shape ({batch_size}, {num_classes}, "
                f"{num_queries}), got {tuple(self.query_heatmap_score.shape)}."
            )
        if self.separate_head_outputs.heatmap.shape[:2] != (batch_size, num_classes):
            raise ValueError(
                f"separate_head_outputs.heatmap must have batch_size={batch_size} and "
                f"num_classes={num_classes}, got {tuple(self.separate_head_outputs.heatmap.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> TransFusionHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        separate = self.separate_head_outputs
        return TransFusionHeadOutputs(
            dense_heatmap=self.dense_heatmap[mask],
            query_heatmap_score=self.query_heatmap_score[mask],
            query_labels=self.query_labels[mask],
            separate_head_outputs=TransFusionSeparateHeadOutputs(
                heatmap=separate.heatmap[mask],
                center=separate.center[mask],
                height=separate.height[mask],
                dim=separate.dim[mask],
                rot=separate.rot[mask],
                vel=None if separate.vel is None else separate.vel[mask],
            ),
        )

    def export_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """
        Pick the output tensors in the order an export names them.

        Args:
            names: Output names in the exported order, taken from both the dense and the
                per-proposal outputs.

        Returns:
            The tensors of the names, in the same order.

        Raises:
            ValueError: If a name is not an output of the head or the head left it unset.
        """
        separate = self.separate_head_outputs
        tensors = {
            "dense_heatmap": self.dense_heatmap,
            "query_heatmap_score": self.query_heatmap_score,
            "query_labels": self.query_labels,
        } | {name: getattr(separate, name) for name in type(separate).model_fields}
        return _pick_export_tensors(tensors, names)


class CenterHeadOutputs(BaseModel):
    """
    Dataclass to save CenterHead-based outputs from a 3D detection model.

    Attributes:
      heatmap: Heatmap to save probability for each class in a BEV heatmap.
      reg: Center_x and center_y offset from each cell in a BEV heatmap.
      height: Height value from each cell in a BEV heatmap.
      dim: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rot: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vel: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    heatmap: Float[torch.Tensor, "batch_size num_classes height width"]
    reg: Float[torch.Tensor, "batch_size 2 height width"]
    height: Float[torch.Tensor, "batch_size 1 height width"]
    dim: Float[torch.Tensor, "batch_size 3 height width"]
    rot: Float[torch.Tensor, "batch_size 2 height width"]
    vel: Float[torch.Tensor, "batch_size 2 height width"] | None

    @model_validator(mode="after")
    def _check_batch_and_grid(self) -> CenterHeadOutputs:
        """Check that every branch covers the same batch on the same BEV grid as heatmap."""
        batch_size, _, height, width = self.heatmap.shape
        for name, tensor in (
            ("reg", self.reg),
            ("height", self.height),
            ("dim", self.dim),
            ("rot", self.rot),
            ("vel", self.vel),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2:] != (height, width):
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, height={height}, "
                    f"width={width}), got {tuple(tensor.shape)}."
                )
        return self

    def export_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """
        Pick the output tensors in the order an export names them.

        Args:
            names: Output names in the exported order.

        Returns:
            The tensors of the names, in the same order.

        Raises:
            ValueError: If a name is not an output of the head or the head left it unset.
        """
        tensors = {name: getattr(self, name) for name in type(self).model_fields}
        return _pick_export_tensors(tensors, names)


class Sparse4DDenoisingOutputs(BaseModel):
    """
    Dataclass to save the denoising queries of a Sparse4D-based head and their targets.

    Targets are fixed at construction, padding queries carry zero weights.

    Attributes:
      classification: Class logits of every denoising query for every decoder layer.
      prediction: Anchor states of every denoising query for every decoder layer.
      cls_targets: One-hot class targets of every denoising query.
      box_targets: Anchor state targets of every denoising query.
      box_weights: Regression weight of every anchor state channel of every denoising query.
      cls_weights: Classification weight of every denoising query.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    classification: Float[torch.Tensor, "batch_size num_layers num_queries num_classes"]
    prediction: Float[torch.Tensor, "batch_size num_layers num_queries anchor_dim"]
    cls_targets: Float[torch.Tensor, "batch_size num_queries num_classes"]
    box_targets: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    box_weights: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    cls_weights: Float[torch.Tensor, "batch_size num_queries"]

    @model_validator(mode="after")
    def _check_queries(self) -> Sparse4DDenoisingOutputs:
        """Check that the outputs and the targets describe the same denoising queries."""
        batch_size, _, num_queries, num_classes = self.classification.shape
        anchor_dim = self.prediction.shape[-1]
        expected = {
            "prediction": (batch_size, self.classification.shape[1], num_queries, anchor_dim),
            "cls_targets": (batch_size, num_queries, num_classes),
            "box_targets": (batch_size, num_queries, anchor_dim),
            "box_weights": (batch_size, num_queries, anchor_dim),
            "cls_weights": (batch_size, num_queries),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(getattr(self, name).shape)}."
                )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> Sparse4DDenoisingOutputs:
        """
        Keep the denoising queries of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The denoising queries of the selected samples.
        """
        return Sparse4DDenoisingOutputs(
            **{name: getattr(self, name)[mask] for name in type(self).model_fields}
        )


class Sparse4DHeadOutputs(BaseModel):
    """
    Dataclass to save Sparse4D-based outputs from a 3D detection model.

    Anchor states are ``[cx, cy, cz, ln l, ln w, ln h, sin yaw, cos yaw, vx, vy]``.

    Attributes:
      classification: Class logits of every query for every decoder layer.
      prediction: Anchor states of every query for every decoder layer.
      quality: Predicted centerness and yawness logits of every query for every decoder layer.
      instance_ids: Track id of every query of the last layer, -1 without a track. None for a
        head without temporal memory.
      denoising: Denoising queries of the training step, None when the head adds none.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    classification: Float[torch.Tensor, "batch_size num_layers num_queries num_classes"]
    prediction: Float[torch.Tensor, "batch_size num_layers num_queries anchor_dim"]
    quality: Float[torch.Tensor, "batch_size num_layers num_queries 2"]
    instance_ids: Int64[torch.Tensor, "batch_size num_queries"] | None
    denoising: Sparse4DDenoisingOutputs | None

    @model_validator(mode="after")
    def _check_layers_and_queries(self) -> Sparse4DHeadOutputs:
        """Check that every output covers the same batch, decoder layers and queries."""
        leading = tuple(self.classification.shape[:3])
        for name in ("prediction", "quality"):
            if tuple(getattr(self, name).shape[:3]) != leading:
                raise ValueError(
                    f"{name} must lead with (batch_size, num_layers, num_queries)={leading}, "
                    f"got {tuple(getattr(self, name).shape)}."
                )
        if self.instance_ids is not None and tuple(self.instance_ids.shape) != (
            self.classification.shape[0],
            self.classification.shape[2],
        ):
            raise ValueError(
                "instance_ids must have shape (batch_size, num_queries), got "
                f"{tuple(self.instance_ids.shape)}."
            )
        if self.denoising is not None and (
            self.denoising.classification.shape[:2] != self.classification.shape[:2]
        ):
            raise ValueError(
                "The denoising queries must cover the batch and the decoder layers of the "
                f"regular queries, got {tuple(self.denoising.classification.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> Sparse4DHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        return Sparse4DHeadOutputs(
            classification=self.classification[mask],
            prediction=self.prediction[mask],
            quality=self.quality[mask],
            instance_ids=None if self.instance_ids is None else self.instance_ids[mask],
            denoising=None if self.denoising is None else self.denoising.select_samples(mask),
        )

    def export_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """
        Pick the final decoder layer tensors in the order an export names them.

        Args:
            names: Output names in the exported order.

        Returns:
            The tensors of the names, in the same order.

        Raises:
            ValueError: If a name is not an output of the head.
        """
        tensors = {
            "classification": self.classification[:, -1],
            "prediction": self.prediction[:, -1],
            "quality": self.quality[:, -1],
        }
        return _pick_export_tensors(tensors, names)


class Detection3DHeadOutputs(BaseModel):
    """
    Dataclass to save outputs from 3D detection models.

    Exactly one of the heads must be set.

    Attributes:
      center_head_outputs: Outputs from a CenterHead-based 3D detection model.
      transfusion_head_outputs: Outputs from a TransFusion-based 3D detection model.
      sparse4d_head_outputs: Outputs from a Sparse4D-based 3D detection model.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    center_head_outputs: CenterHeadOutputs | None
    transfusion_head_outputs: TransFusionHeadOutputs | None
    sparse4d_head_outputs: Sparse4DHeadOutputs | None = None

    def center_head(self) -> CenterHeadOutputs:
        """
        Read the outputs of a CenterHead-based model.

        Raises:
            ValueError: If another head family produced the outputs.
        """
        if self.center_head_outputs is None:
            raise ValueError(f"The detection outputs come from a {self._head_family()} head.")
        return self.center_head_outputs

    def transfusion_head(self) -> TransFusionHeadOutputs:
        """
        Read the outputs of a TransFusion-based model.

        Raises:
            ValueError: If another head family produced the outputs.
        """
        if self.transfusion_head_outputs is None:
            raise ValueError(f"The detection outputs come from a {self._head_family()} head.")
        return self.transfusion_head_outputs

    def sparse4d_head(self) -> Sparse4DHeadOutputs:
        """
        Read the outputs of a Sparse4D-based model.

        Raises:
            ValueError: If another head family produced the outputs.
        """
        if self.sparse4d_head_outputs is None:
            raise ValueError(f"The detection outputs come from a {self._head_family()} head.")
        return self.sparse4d_head_outputs

    def _head_family(self) -> str:
        """Name the head family that produced the outputs."""
        return {
            "center_head_outputs": "CenterHead",
            "transfusion_head_outputs": "TransFusion",
            "sparse4d_head_outputs": "Sparse4D",
        }[next(name for name in self._set_heads())]

    def _set_heads(self) -> list[str]:
        """List the head output fields that are set."""
        return [name for name in type(self).model_fields if getattr(self, name) is not None]

    @model_validator(mode="after")
    def _check_exactly_one_head(self) -> Detection3DHeadOutputs:
        """Check that exactly one head family produced the outputs."""
        set_heads = self._set_heads()
        if len(set_heads) != 1:
            raise ValueError(
                "Exactly one of center_head_outputs, transfusion_head_outputs and "
                f"sparse4d_head_outputs must be set, got {set_heads}."
            )
        return self


def _pick_export_tensors(
    tensors: Mapping[str, torch.Tensor | None], names: Sequence[str]
) -> tuple[torch.Tensor, ...]:
    """
    Pick the named head outputs in the exported order.

    Args:
        tensors: Every output of a head keyed by its tensor name, unset ones as None.
        names: Output names in the exported order.

    Returns:
        The tensors of the names, in the same order.

    Raises:
        ValueError: If a name is not an output of the head or the head left it unset.
    """
    unknown = [name for name in names if name not in tensors or tensors[name] is None]
    if unknown:
        raise ValueError(
            f"The head has no output named {unknown}, it provides "
            f"{[name for name, tensor in tensors.items() if tensor is not None]}."
        )
    return tuple(tensors[name] for name in names)
