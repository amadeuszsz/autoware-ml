"""Abstract base class and shared export modules for PTv3-based task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.models.base import BaseModel
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointTransformerV3Encoder,
    SerializedPooling,
    SerializedPoolingMeta,
    _pooling_depth,
    build_serialized_pooling_meta,
    collect_encoder_stage_points,
)
from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import (
    Point,
    bit_length_tensor,
    invert_permutation,
    serialize_point_cloud_batch,
)

_BLOCK_STAGE_META_FIELDS = ("serialized_order", "serialized_inverse", "grid_coord")

SERIALIZED_POOLING_FIELDS = tuple(field.name for field in fields(SerializedPoolingMeta))
# The encoder-only encoder graph never consumes `cluster` (it only drives the
# heads' unpooling), so the split encoder export excludes it - the tracer
# would prune it anyway, breaking the declared interface.
ENCODER_EXPORT_POOLING_FIELDS = tuple(
    name for name in SERIALIZED_POOLING_FIELDS if name != "cluster"
)
SERIALIZED_POOLING_INPUT_SIZED_FIELDS = frozenset({"indices", "cluster"})
SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS = frozenset({"indptr"})
SERIALIZED_POOLING_ORDER_FIELDS = frozenset({"serialized_order", "serialized_inverse"})


def validate_serialization_geometry(
    encoder: nn.Module, grid_size: float, point_cloud_range: Sequence[float]
) -> None:
    """Raise if the configured geometry cannot cover the encoder's pooling hierarchy."""
    pooling_depth = sum(
        m.pooling_depth for m in encoder.modules() if isinstance(m, SerializedPooling)
    )
    extent = max(point_cloud_range[i + 3] - point_cloud_range[i] for i in range(3))
    if int(bit_length_tensor(extent / grid_size).item()) < pooling_depth:
        raise ValueError(
            f"point_cloud_range {tuple(point_cloud_range)} with grid_size {grid_size} cannot "
            f"cover the encoder's cumulative pooling depth {pooling_depth}."
        )


def split_block_parameters(
    module: nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable parameters into non-block and attention-block groups.

    Args:
        module: Module hierarchy whose parameters are grouped structurally.

    Returns:
        ``(default_params, block_params)`` where block parameters belong to
        :class:`Block` submodules and default parameters are all the rest.
    """
    block_parameter_ids = {
        id(parameter)
        for child in module.modules()
        if isinstance(child, Block)
        for parameter in child.parameters()
        if parameter.requires_grad
    }
    default_params: list[torch.nn.Parameter] = []
    block_params: list[torch.nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in block_parameter_ids:
            block_params.append(parameter)
        else:
            default_params.append(parameter)
    return default_params, block_params


class PTv3BaseModel(BaseModel):
    """Abstract base class for all PTv3 task models.

    Provides shared encoder management, export geometry computation, and
    export helpers. Detection and segmentation subclasses inherit from this
    class (potentially with additional base classes via MRO).
    """

    EXPORT_ORDER = ("z", "z-trans")
    EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        grid_size: float | None,
        point_cloud_range: Sequence[float] | None,
        freeze_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 base model.

        Args:
            encoder: PTv3 encoder module.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth for export.
            point_cloud_range: Six-element sequence ``[x_min, y_min, z_min,
                x_max, y_max, z_max]`` used to derive sparse shape for export.
            freeze_encoder: When ``True``, the encoder is permanently kept
                in eval mode with its parameters frozen.
            **kwargs: Keyword arguments forwarded to :class:`BaseModel` (and
                further up the MRO chain).
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.grid_size = grid_size
        self.point_cloud_range = (
            tuple(float(v) for v in point_cloud_range) if point_cloud_range is not None else None
        )
        if self.grid_size is not None and self.point_cloud_range is not None:
            validate_serialization_geometry(encoder, self.grid_size, self.point_cloud_range)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> PTv3BaseModel:
        """Keep the frozen encoder in eval mode during training.

        Args:
            mode: When ``True``, set the model to training mode; otherwise to
                evaluation mode.

        Returns:
            This model instance.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Record encoder-freeze provenance in saved checkpoints.

        Args:
            checkpoint: Mutable checkpoint dictionary to annotate.
        """
        checkpoint["autoware_ml_checkpoint_recipe"] = {
            "type": "ptv3",
            "freeze_encoder": self.freeze_encoder,
        }

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        """Infer the effective sample batch size for logging.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "offset" in batch_inputs_dict:
            return int(batch_inputs_dict["offset"].numel())
        return super().get_log_batch_size(batch_inputs_dict)

    def _compute_export_geometry(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse shape and serialization depth for export.

        Args:
            batch_inputs_dict: Preprocessed batch containing at least
                ``coord`` (used for device inference).

        Returns:
            ``(sparse_shape, serialization_depth)`` as long tensors on the
            same device as ``batch_inputs_dict["coord"]``.
        """
        device = batch_inputs_dict["coord"].device
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32, device=device)
        axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size
        serialization_depth = bit_length_tensor(torch.max(axis_extents))
        sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
        return sparse_shape, serialization_depth

    def _prepare_encoder_export(self) -> PointTransformerV3Encoder:
        """Return an export-ready copy of the encoder.

        Returns:
            Copy of the encoder prepared for ONNX export with the configured
            export order.
        """
        return self.encoder.prepare_for_export(self.EXPORT_ORDER)


def build_serialized_pooling_metadata(
    grid_coord: torch.Tensor,
    serialized_code: torch.Tensor,
    serialized_order: torch.Tensor,
    strides: Sequence[int],
) -> list[SerializedPoolingMeta]:
    """Build serialized-pooling metadata for every encoder pooling stage."""
    metadata = []
    for stride in strides:
        meta, serialized_code = build_serialized_pooling_meta(
            grid_coord, serialized_code, serialized_order, stride
        )
        metadata.append(meta)
        grid_coord = meta.grid_coord
        serialized_order = meta.serialized_order
    return metadata


def normalize_pooling_plan(
    stage_count: int, field_names: Sequence[str] | Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    """Expand a field specification into one explicit field tuple per stage.

    Accepts either a flat sequence of field names, applied to every stage, or
    an already per-stage sequence of field tuples.

    Args:
        stage_count: Number of pooling stages the plan must cover.
        field_names: Flat or per-stage field specification.

    Returns:
        One field tuple per pooling stage.

    Raises:
        ValueError: Raised when a per-stage plan has the wrong length.
    """
    if len(field_names) > 0 and isinstance(field_names[0], str):
        return tuple(tuple(field_names) for _ in range(stage_count))  # type: ignore[arg-type]
    plan = tuple(tuple(stage_fields) for stage_fields in field_names)
    if len(plan) != stage_count:
        raise ValueError(f"pooling plan must cover {stage_count} stages, got {len(plan)}.")
    return plan


def flatten_serialized_pooling_inputs(
    metadata: Sequence[SerializedPoolingMeta],
    field_names: Sequence[str] | Sequence[Sequence[str]] = SERIALIZED_POOLING_FIELDS,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Flatten per-stage metadata into ONNX args and input names.

    Args:
        metadata: Per-pooling-stage metadata objects.
        field_names: Metadata fields exported, in order, either shared by every
            stage or given per stage. The encoder-only graph excludes
            ``cluster`` (it only drives head-side unpooling), and stages whose
            encoder stage has no attention also exclude the order fields.
    """
    plan = normalize_pooling_plan(len(metadata), field_names)
    inputs: list[torch.Tensor] = []
    names: list[str] = []
    for stage_index, (meta, stage_fields) in enumerate(zip(metadata, plan)):
        for field in stage_fields:
            inputs.append(getattr(meta, field))
            names.append(f"serialized_pooling_{stage_index}_{field}")
    return tuple(inputs), names


def _serialized_pooling_dynamic_axis(input_name: str) -> dict[int, str]:
    _, _, stage_index, field = input_name.split("_", 3)
    stage_prefix = f"serialized_pooling_{stage_index}"
    if field in SERIALIZED_POOLING_INPUT_SIZED_FIELDS:
        return {0: f"{stage_prefix}_in_voxels"}
    if field in SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS:
        return {0: f"{stage_prefix}_out_voxels_plus_one"}
    if field in SERIALIZED_POOLING_ORDER_FIELDS:
        return {1: f"{stage_prefix}_out_voxels"}
    return {0: f"{stage_prefix}_out_voxels"}


def stage_voxel_axis_name(stage_index: int) -> str:
    """Return the dynamic-axis name for the voxel count of one encoder stage."""
    if stage_index == 0:
        return "num_voxels"
    return f"serialized_pooling_{stage_index - 1}_out_voxels"


def stage_feature_names(stage_count: int) -> list[str]:
    """Return the per-stage encoder feature tensor names, finest to deepest."""
    return [f"point_feat_{stage_index}" for stage_index in range(stage_count)]


def pooling_cluster_names(stage_count: int) -> list[str]:
    """Return the per-pooling cluster tensor names consumed by the decoder."""
    return [f"pooling_cluster_{stage_index}" for stage_index in range(stage_count - 1)]


def build_stage_feature_dynamic_axes(stage_count: int) -> dict[str, dict[int, str]]:
    """Build dynamic axes for per-stage encoder feature tensors."""
    return {
        name: {0: stage_voxel_axis_name(stage_index)}
        for stage_index, name in enumerate(stage_feature_names(stage_count))
    }


def build_pooling_cluster_dynamic_axes(stage_count: int) -> dict[str, dict[int, str]]:
    """Build dynamic axes for per-pooling cluster tensors."""
    return {
        name: {0: f"serialized_pooling_{stage_index}_in_voxels"}
        for stage_index, name in enumerate(pooling_cluster_names(stage_count))
    }


def build_point_feature_dynamic_axes(tensor_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for tensors indexed by the decoded point/voxel count."""
    return {tensor_name: {0: "num_voxels"} for tensor_name in tensor_names}


def build_ptv3_input_dynamic_axes(input_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for generated PTv3 encoder export inputs."""
    dynamic_axes: dict[str, dict[int, str]] = {}
    for input_name in input_names:
        if input_name in {"grid_coord", "feat"}:
            dynamic_axes[input_name] = {0: "num_voxels"}
        elif input_name == "serialized_code":
            dynamic_axes[input_name] = {1: "num_voxels"}
        elif input_name.startswith("serialized_pooling_"):
            dynamic_axes[input_name] = _serialized_pooling_dynamic_axis(input_name)
    return dynamic_axes


def build_ptv3_encoder_dynamic_axes(
    input_names: Sequence[str], stage_count: int
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split PTv3 encoder export graph."""
    dynamic_axes = build_ptv3_input_dynamic_axes(input_names)
    dynamic_axes.update(build_stage_feature_dynamic_axes(stage_count))
    return dynamic_axes


def make_serialized_pooling_from_flat_inputs(
    serialized_pooling_inputs: tuple[torch.Tensor, ...],
    field_names: Sequence[str] | Sequence[Sequence[str]] = SERIALIZED_POOLING_FIELDS,
    stage_count: int | None = None,
) -> list[SerializedPoolingMeta]:
    """Reconstruct per-stage metadata objects from flattened ONNX graph inputs.

    Fields absent from a stage's plan are filled with empty placeholders; only
    fields the target graph never consumes for that stage may be omitted.

    Args:
        serialized_pooling_inputs: Flattened per-stage tensors, in plan order.
        field_names: Flat or per-stage field specification matching the tensors.
        stage_count: Number of pooling stages. Required only for a flat
            specification, where it is otherwise inferred from the tensor count.

    Returns:
        One metadata object per pooling stage.

    Raises:
        ValueError: Raised when the tensors do not match the plan.
    """
    if stage_count is None:
        if len(field_names) > 0 and isinstance(field_names[0], str):
            if len(serialized_pooling_inputs) % len(field_names) != 0:
                raise ValueError(
                    "serialized-pooling inputs are not divisible by metadata field count."
                )
            stage_count = len(serialized_pooling_inputs) // len(field_names)
        else:
            stage_count = len(field_names)
    plan = normalize_pooling_plan(stage_count, field_names)
    if sum(len(stage_fields) for stage_fields in plan) != len(serialized_pooling_inputs):
        raise ValueError("serialized-pooling inputs do not match the declared pooling plan.")

    metadata: list[SerializedPoolingMeta] = []
    cursor = 0
    for stage_fields in plan:
        values = dict(
            zip(stage_fields, serialized_pooling_inputs[cursor : cursor + len(stage_fields)])
        )
        cursor += len(stage_fields)
        placeholder = values[stage_fields[0]].new_zeros(0)
        for field in SERIALIZED_POOLING_FIELDS:
            values.setdefault(field, placeholder)
        metadata.append(SerializedPoolingMeta(**values))
    return metadata


@dataclass(frozen=True)
class EncoderExportContract:
    """Declare which serialization tensors an encoder configuration exports.

    PTv3 derives its whole pooling hierarchy from one serialization at input
    resolution, so pooling metadata is always exported. The *order* tensors are
    different: only stages that attend ever read them. A configuration that
    gates attention off for some stages - LitePT's early stages, for instance -
    therefore exports fewer tensors, and the base ``serialized_code`` drops out
    entirely when the finest stage does not attend.

    This is not merely an optimization. Inputs the traced graph never consumes
    are pruned by the exporter, so the declared interface has to match what the
    configuration actually reads or export produces a mismatched signature.
    """

    needs_base_code: bool
    pooling_plan: tuple[tuple[str, ...], ...]

    @classmethod
    def from_encoder(
        cls, encoder: PointTransformerV3Encoder, include_cluster: bool
    ) -> "EncoderExportContract":
        """Derive the contract from an encoder's stage gating.

        Args:
            encoder: Encoder whose configuration defines the contract.
            include_cluster: Whether ``cluster`` is part of the graph inputs.
                The encoder-only graph excludes it because only head-side
                unpooling consumes it.
        """
        base = SERIALIZED_POOLING_FIELDS if include_cluster else ENCODER_EXPORT_POOLING_FIELDS
        without_order = tuple(
            field for field in base if field not in SERIALIZED_POOLING_ORDER_FIELDS
        )
        plan = tuple(
            tuple(base) if encoder.pooling_stage_needs_order(pooling_index) else without_order
            for pooling_index in range(len(encoder.stride))
        )
        return cls(needs_base_code=encoder.needs_base_serialization, pooling_plan=plan)

    @property
    def pooling_input_names(self) -> list[str]:
        """Return the per-stage pooling input names, in graph order."""
        return [
            f"serialized_pooling_{stage_index}_{field}"
            for stage_index, stage_fields in enumerate(self.pooling_plan)
            for field in stage_fields
        ]

    @property
    def input_names(self) -> list[str]:
        """Return every encoder graph input name, in graph order."""
        names = ["grid_coord", "feat"]
        if self.needs_base_code:
            names.append("serialized_code")
        return names + self.pooling_input_names

    def pooling_args(self, metadata: Sequence[SerializedPoolingMeta]) -> tuple[torch.Tensor, ...]:
        """Flatten metadata into the pooling tensors this contract declares."""
        return flatten_serialized_pooling_inputs(metadata, self.pooling_plan)[0]

    def input_args(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        metadata: Sequence[SerializedPoolingMeta],
    ) -> tuple[torch.Tensor, ...]:
        """Assemble the sample inputs matching :attr:`input_names`."""
        head: tuple[torch.Tensor, ...] = (grid_coord, feat)
        if self.needs_base_code:
            head = head + (serialized_code,)
        return head + self.pooling_args(metadata)

    def split(
        self, tensors: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        """Split traced inputs after ``grid_coord``/``feat`` into code and pooling.

        Args:
            tensors: Everything the export module received after ``feat``.

        Returns:
            The base ``serialized_code`` when the contract declares one, and the
            flattened pooling tensors.
        """
        if self.needs_base_code:
            return tensors[0], tuple(tensors[1:])
        return None, tuple(tensors)

    def metadata(self, pooling_tensors: Sequence[torch.Tensor]) -> list[SerializedPoolingMeta]:
        """Rebuild per-stage metadata objects from the flattened pooling tensors."""
        return make_serialized_pooling_from_flat_inputs(
            tuple(pooling_tensors), self.pooling_plan, stage_count=len(self.pooling_plan)
        )


class PTv3EncoderExportBase(nn.Module):
    """Share the encoder half of every PTv3 export graph.

    Subclasses add their own task head and declare their outputs; the encoder
    inputs, the baked geometry buffers, and the contract-driven unpacking are
    identical across segmentation, detection, and the joint model.
    """

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        contract: EncoderExportContract,
    ) -> None:
        """Initialize the shared encoder export half.

        Args:
            encoder: Export-prepared PTv3 encoder copy.
            sparse_shape: Static sparse shape baked at export time.
            serialized_depth: Serialization depth baked at export time.
            contract: Declared encoder input contract.
        """
        super().__init__()
        self.encoder = encoder
        self.contract = contract
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def run_encoder(
        self, grid_coord: torch.Tensor, feat: torch.Tensor, *tensors: torch.Tensor
    ) -> Point:
        """Run the encoder over contract-ordered inputs.

        Args:
            grid_coord: Discretized grid coordinates.
            feat: Point features whose first three channels are xyz.
            tensors: The base ``serialized_code`` when the contract declares
                one, followed by the flattened pooling metadata tensors.

        Returns:
            Deepest encoder point with the full pooling chain attached.
        """
        serialized_code, pooling_tensors = self.contract.split(tensors)
        return _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *pooling_tensors,
            contract=self.contract,
        )


def _run_ptv3_encoder_export(
    encoder: PointTransformerV3Encoder,
    grid_coord: torch.Tensor,
    feat: torch.Tensor,
    serialized_depth: torch.Tensor,
    serialized_code: torch.Tensor | None,
    sparse_shape: torch.Tensor,
    *serialized_pooling_inputs: torch.Tensor,
    pooling_field_names: "Sequence[str] | Sequence[Sequence[str]]" = SERIALIZED_POOLING_FIELDS,
    contract: "EncoderExportContract | None" = None,
) -> Point:
    """Run the shared tensor-only PTv3 encoder export path.

    Args:
        encoder: Export-prepared encoder.
        grid_coord: Discretized grid coordinates.
        feat: Point features whose first three channels are xyz.
        serialized_depth: Baked serialization depth.
        serialized_code: Base serialization codes, or ``None`` when the finest
            stage carries no attention and therefore reads no base order.
        sparse_shape: Baked sparse shape.
        serialized_pooling_inputs: Flattened per-stage pooling metadata.
        pooling_field_names: Field plan for the flattened metadata. Ignored when
            ``contract`` is given.
        contract: Declared input contract, which supersedes
            ``pooling_field_names`` when present.

    Returns:
        Deepest encoder point with the full pooling chain attached.
    """
    point_count = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
    if serialized_code is None:
        # Nothing in the graph reads the base serialization, so the input is
        # absent and these placeholders are never consumed.
        empty = grid_coord.new_zeros((1, 0))
        serialized_code = empty
        serialized_order = empty
        serialized_inverse = empty
    else:
        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = invert_permutation(serialized_order)
    if contract is not None:
        metadata = contract.metadata(serialized_pooling_inputs)
    else:
        metadata = make_serialized_pooling_from_flat_inputs(
            serialized_pooling_inputs, pooling_field_names
        )
    return encoder.export_forward(
        {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": point_count,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "serialized_pooling": metadata,
            "sparse_shape": sparse_shape,
        }
    )


@dataclass(frozen=True)
class PTv3ExportContext:
    """Shared front half of every split PTv3 export.

    Built once per export: the serialized batch, per-stage pooling metadata,
    the export-ready encoder module, and its per-stage features. Artifact
    spec builders pair this context with their own input-name rule.
    """

    sparse_shape: torch.Tensor
    serialization_depth: torch.Tensor
    grid_coord: torch.Tensor
    feat: torch.Tensor
    serialized_code: torch.Tensor
    strides: tuple[int, ...]
    pooling_metadata: tuple[SerializedPoolingMeta, ...]
    encoder_module: nn.Module
    stage_feats: tuple[torch.Tensor, ...]
    contract: EncoderExportContract

    @property
    def stage_count(self) -> int:
        return len(self.stage_feats)

    @property
    def encoder_input_args(self) -> tuple[torch.Tensor, ...]:
        return self.contract.input_args(
            self.grid_coord, self.feat, self.serialized_code, self.pooling_metadata
        )

    @property
    def encoder_input_names(self) -> list[str]:
        return self.contract.input_names


def build_ptv3_export_context(
    model: "PTv3BaseModel", batch: Mapping[str, torch.Tensor]
) -> PTv3ExportContext:
    """Serialize the batch, precompute pooling metadata, and run the encoder once."""
    sparse_shape, serialization_depth = model._compute_export_geometry(batch)
    point, input_args = serialize_point_cloud_batch(batch, model.EXPORT_ORDER, serialization_depth)
    pooling_metadata = build_serialized_pooling_metadata(
        point["grid_coord"],
        point["serialized_code"],
        point["serialized_order"],
        model.encoder.stride,
    )
    contract = EncoderExportContract.from_encoder(model.encoder, include_cluster=False)
    encoder_module = _PTv3EncoderExportModule(
        encoder=model._prepare_encoder_export(),
        sparse_shape=sparse_shape,
        serialized_depth=serialization_depth,
        contract=contract,
    ).eval()
    encoder_args = contract.input_args(
        input_args[0], input_args[1], input_args[3], pooling_metadata
    )
    with torch.no_grad():
        stage_feats = encoder_module(*encoder_args)
    return PTv3ExportContext(
        sparse_shape=sparse_shape,
        serialization_depth=serialization_depth,
        grid_coord=input_args[0],
        feat=input_args[1],
        serialized_code=input_args[3],
        strides=tuple(model.encoder.stride),
        pooling_metadata=tuple(pooling_metadata),
        encoder_module=encoder_module,
        stage_feats=tuple(stage_feats),
        contract=contract,
    )


@dataclass(frozen=True)
class MonolithicExportInputs:
    """Encoder-side inputs shared by every single-graph PTv3 export."""

    sparse_shape: torch.Tensor
    serialization_depth: torch.Tensor
    contract: EncoderExportContract
    args: tuple[torch.Tensor, ...]
    input_names: list[str]


def build_monolithic_export_inputs(
    model: "PTv3BaseModel", batch: Mapping[str, torch.Tensor]
) -> MonolithicExportInputs:
    """Serialize a batch and derive the encoder inputs for a single-graph export.

    Single-graph exports keep the whole model in one engine, so unlike the split
    encoder graph they do consume ``cluster`` for head-side unpooling.

    Args:
        model: Task model being exported.
        batch: Preprocessed batch with ``coord``, ``feat``, ``grid_coord``, and
            ``offset``.

    Returns:
        Baked geometry, the declared contract, and the matching sample inputs.
    """
    sparse_shape, serialization_depth = model._compute_export_geometry(batch)
    point, input_args = serialize_point_cloud_batch(batch, model.EXPORT_ORDER, serialization_depth)
    pooling_metadata = build_serialized_pooling_metadata(
        point["grid_coord"],
        point["serialized_code"],
        point["serialized_order"],
        model.encoder.stride,
    )
    contract = EncoderExportContract.from_encoder(model.encoder, include_cluster=True)
    return MonolithicExportInputs(
        sparse_shape=sparse_shape,
        serialization_depth=serialization_depth,
        contract=contract,
        args=contract.input_args(input_args[0], input_args[1], input_args[3], pooling_metadata),
        input_names=contract.input_names,
    )


def build_encoder_export_spec(context: PTv3ExportContext) -> "ExportSpec":
    """Build the shared per-stage-feature encoder export spec."""
    input_names = context.encoder_input_names
    return ExportSpec(
        module=context.encoder_module,
        args=context.encoder_input_args,
        input_param_names=input_names,
        output_names=stage_feature_names(context.stage_count),
        dynamic_axes=build_ptv3_encoder_dynamic_axes(input_names, context.stage_count),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


def build_seg_head_export_spec(
    context: PTv3ExportContext, seg3d_head: nn.Module, output_names: Sequence[str]
) -> "ExportSpec":
    """Build the segmentation-head export spec for any decoder configuration.

    Args:
        context: Shared export context.
        seg3d_head: Export-prepared decoder head copy.
        output_names: Ordered head output names.
    """
    module = _PTv3SegHeadExportModule(
        seg3d_head, context.stage_count, context.sparse_shape, context.strides
    ).eval()
    dec_attn = getattr(seg3d_head, "dec_attn", None)
    input_names = seg_head_export_input_names(context.stage_count, seg3d_head.dec_depths, dec_attn)
    dynamic_axes = build_seg_head_input_dynamic_axes(
        context.stage_count, seg3d_head.dec_depths, dec_attn
    )
    dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
    return ExportSpec(
        module=module,
        args=build_seg_head_export_args(
            context.stage_feats,
            context.pooling_metadata,
            context.serialized_code,
            context.grid_coord,
            seg3d_head.dec_depths,
            dec_attn,
        ),
        input_param_names=input_names,
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


class _PTv3EncoderExportModule(PTv3EncoderExportBase):
    """Export-only PTv3 encoder producing per-stage point features."""

    def forward(
        self, grid_coord: torch.Tensor, feat: torch.Tensor, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run the encoder and return per-stage features, finest to deepest."""
        point = self.run_encoder(grid_coord, feat, *tensors)
        return tuple(stage.feat for stage in collect_encoder_stage_points(point))


def link_stage_points(
    stage_feats: Sequence[torch.Tensor],
    clusters: Sequence[torch.Tensor],
    block_stage_metadata: Mapping[int, tuple[torch.Tensor, ...]] | None = None,
) -> Point:
    """Rebuild the encoder pooling chain from per-stage tensors.

    Args:
        stage_feats: Per-stage features ordered finest to deepest.
        clusters: Per-pooling cluster tensors mapping each finer-stage voxel
            to its pooled voxel.
        block_stage_metadata: For every stage whose decoder has attention
            blocks, ``(serialized_order, serialized_inverse, grid_coord,
            sparse_shape)`` used to rebuild the serialization and sparse-conv
            views the blocks read. Single-sample export is assumed for the
            derived batch offsets.

    Returns:
        Deepest point whose ``pooling_parent``/``pooling_inverse`` chain links
        every finer stage.
    """
    if len(clusters) != len(stage_feats) - 1:
        raise ValueError(
            f"Expected {len(stage_feats) - 1} cluster tensors for {len(stage_feats)} stages, "
            f"got {len(clusters)}."
        )
    points = [Point(feat=feat) for feat in stage_feats]
    for stage_index in range(1, len(points)):
        points[stage_index]["pooling_parent"] = points[stage_index - 1]
        points[stage_index]["pooling_inverse"] = clusters[stage_index - 1]
    for stage_index, metadata in (block_stage_metadata or {}).items():
        serialized_order, serialized_inverse, grid_coord, sparse_shape = metadata
        point = points[stage_index]
        # Convolution-only decoder stages still need the sparse view rebuilt,
        # but carry no serialization order to attach.
        if serialized_order is not None:
            point["serialized_order"] = serialized_order
            point["serialized_inverse"] = serialized_inverse
        point["grid_coord"] = grid_coord
        point["offset"] = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
        point["batch"] = torch.zeros_like(grid_coord[:, 0]).long()
        point["sparse_shape"] = sparse_shape
        point.sparsify()
    return points[-1]


def _block_stage_indices(dec_depths: Sequence[int]) -> list[int]:
    """Return the decoder stages that contain blocks of any kind."""
    return [stage for stage, depth in enumerate(dec_depths) if depth > 0]


def _block_stage_fields(
    stage: int, dec_depths: Sequence[int], dec_attn: Sequence[bool] | None
) -> tuple[str, ...]:
    """Return the serialization tensors one decoder stage's blocks consume.

    A stage with no blocks reads nothing. A stage whose blocks are
    convolution-only still needs ``grid_coord`` to rebuild its sparse tensor,
    but no serialization order. Only attending stages need all three.

    Args:
        stage: Decoder stage index.
        dec_depths: Decoder block counts per stage.
        dec_attn: Per-stage attention flags, or ``None`` when every stage
            attends (the PTv3 default).

    Returns:
        Field names for the stage, in graph order.
    """
    if dec_depths[stage] == 0:
        return ()
    if dec_attn is None or dec_attn[stage]:
        return _BLOCK_STAGE_META_FIELDS
    return ("grid_coord",)


def seg_head_export_input_names(
    stage_count: int, dec_depths: Sequence[int], dec_attn: Sequence[bool] | None = None
) -> list[str]:
    """Return the split seg-head export input names for a decoder configuration.

    The rule is the deployment contract: per-stage features and pooling
    clusters always; for every stage with decoder blocks, that stage's
    serialization metadata (the same tensors, under the same names, that the
    encoder graph consumes - stage 0 uses the base ``serialized_code`` and
    ``grid_coord`` inputs instead).

    Args:
        stage_count: Number of encoder stages.
        dec_depths: Decoder block counts per stage (``stage_count - 1`` entries).
        dec_attn: Per-stage decoder attention flags. Convolution-only stages
            need no serialization order, so they contribute fewer inputs.
    """
    if len(dec_depths) != stage_count - 1:
        raise ValueError(
            f"dec_depths must have {stage_count - 1} entries for {stage_count} stages, "
            f"got {len(dec_depths)}."
        )
    names = [*stage_feature_names(stage_count), *pooling_cluster_names(stage_count)]
    for stage in _block_stage_indices(dec_depths):
        fields = _block_stage_fields(stage, dec_depths, dec_attn)
        if stage == 0:
            if "serialized_order" in fields:
                names.append("serialized_code")
            names.append("grid_coord")
        else:
            prefix = f"serialized_pooling_{stage - 1}_"
            names += [prefix + field for field in fields]
    return names


def build_seg_head_export_args(
    stage_feats: Sequence[torch.Tensor],
    pooling_metadata: Sequence[SerializedPoolingMeta],
    serialized_code: torch.Tensor,
    grid_coord: torch.Tensor,
    dec_depths: Sequence[int],
    dec_attn: Sequence[bool] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Assemble the split seg-head export args matching the input-name rule."""
    args = [*stage_feats, *(meta.cluster for meta in pooling_metadata)]
    for stage in _block_stage_indices(dec_depths):
        fields = _block_stage_fields(stage, dec_depths, dec_attn)
        if stage == 0:
            if "serialized_order" in fields:
                args.append(serialized_code)
            args.append(grid_coord)
        else:
            meta = pooling_metadata[stage - 1]
            args += [getattr(meta, field) for field in fields]
    return tuple(args)


def build_seg_head_input_dynamic_axes(
    stage_count: int, dec_depths: Sequence[int], dec_attn: Sequence[bool] | None = None
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split seg-head export inputs."""
    dynamic_axes = build_stage_feature_dynamic_axes(stage_count)
    dynamic_axes.update(build_pooling_cluster_dynamic_axes(stage_count))
    for stage in _block_stage_indices(dec_depths):
        fields = _block_stage_fields(stage, dec_depths, dec_attn)
        if stage == 0:
            if "serialized_order" in fields:
                dynamic_axes["serialized_code"] = {1: "num_voxels"}
            dynamic_axes["grid_coord"] = {0: "num_voxels"}
        else:
            prefix = f"serialized_pooling_{stage - 1}_"
            for field in fields:
                dynamic_axes[prefix + field] = _serialized_pooling_dynamic_axis(prefix + field)
    return dynamic_axes


class _PTv3SegHeadExportModule(nn.Module):
    """Export-only segmentation head decoding per-stage encoder features."""

    def __init__(
        self,
        seg3d_head: nn.Module,
        stage_count: int,
        sparse_shape: torch.Tensor,
        strides: Sequence[int],
    ) -> None:
        """Initialize the segmentation head export module.

        Args:
            seg3d_head: Export-prepared decoder head copy.
            stage_count: Number of encoder stages feeding the decoder.
            sparse_shape: Static base sparse shape baked at export time;
                block stages use it right-shifted by their cumulative pooling
                depth.
            strides: Encoder pooling strides (one per pooling stage).
        """
        super().__init__()
        self.seg3d_head = seg3d_head
        self.stage_count = int(stage_count)
        self.dec_depths = list(seg3d_head.dec_depths)
        self.dec_attn = list(getattr(seg3d_head, "dec_attn", [True] * len(self.dec_depths)))
        cumulative_depth = 0
        stage_depths = [0]
        for stride in strides:
            cumulative_depth += _pooling_depth(int(stride))
            stage_depths.append(cumulative_depth)
        for stage in _block_stage_indices(self.dec_depths):
            self.register_buffer(
                f"_sparse_shape_{stage}",
                sparse_shape.to(dtype=torch.long) >> stage_depths[stage],
                persistent=False,
            )

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode per-stage features and return labels and class probabilities.

        Args:
            tensors: ``stage_count`` per-stage feature tensors, then
                ``stage_count - 1`` pooling cluster tensors, then the
                per-block-stage serialization tensors in the order defined by
                :func:`seg_head_export_input_names`.
        """
        stage_feats = tensors[: self.stage_count]
        clusters = tensors[self.stage_count : 2 * self.stage_count - 1]
        extras = list(tensors[2 * self.stage_count - 1 :])

        block_stage_metadata: dict[int, tuple[torch.Tensor, ...]] = {}
        for stage in _block_stage_indices(self.dec_depths):
            fields = _block_stage_fields(stage, self.dec_depths, self.dec_attn)
            attends = "serialized_order" in fields
            serialized_order = None
            serialized_inverse = None
            if stage == 0:
                if attends:
                    serialized_code = extras.pop(0)
                    serialized_order = torch.stack(
                        [argsort(code) for code in serialized_code], dim=0
                    )
                    serialized_inverse = invert_permutation(serialized_order)
                grid_coord = extras.pop(0)
            else:
                if attends:
                    serialized_order = extras.pop(0)
                    serialized_inverse = extras.pop(0)
                grid_coord = extras.pop(0)
            block_stage_metadata[stage] = (
                serialized_order,
                serialized_inverse,
                grid_coord,
                getattr(self, f"_sparse_shape_{stage}"),
            )

        logits = self.seg3d_head(link_stage_points(stage_feats, clusters, block_stage_metadata))
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1), probs
