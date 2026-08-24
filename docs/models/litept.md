---
icon: lucide/feather
---

# LitePT

LitePT is a LiDAR encoder that keeps PointTransformerV3's serialization,
pooling, and export contract and changes only **which operations run at which
resolution**. Early stages are pure submanifold convolution, late stages are
pure attention with rotary position embedding, and the decoder is unpooling
only. The two expensive subsystems never share a block.

It is implemented as `LitePTEncoder`, a subclass of
`PointTransformerV3Encoder`, so every PTv3 task model, head, and export path
accepts it unchanged. Nothing about LitePT is a fork of PTv3 - the gating and
the rotary embedding are options on the shared components, and PTv3's defaults
leave both off.

## Summary

| Property     | Value                                                             |
| ------------ | ----------------------------------------------------------------- |
| Task         | 3D semantic segmentation, 3D object detection                     |
| Modality     | LiDAR                                                             |
| Input        | Point cloud                                                       |
| Output       | Point-wise semantic labels or 3D boxes/scores/classes             |
| Architecture | PTv3 hierarchy with per-stage conv/attention gating and 3D RoPE   |
| Datasets     | T4Dataset                                                          |

## What differs from PTv3

| Aspect             | PTv3                                    | LitePT                                              |
| ------------------ | --------------------------------------- | --------------------------------------------------- |
| Block composition  | Every block: conv + attention + MLP     | Stages 0-2 conv only, stages 3-4 attention only     |
| Positional encoding| Submanifold conv per block              | Conv early; axis-split 3D RoPE on `grid_coord` late |
| Decoder            | `dec_depths` blocks per stage           | `dec_depths` all zero - unpooling only              |
| Encoder blocks     | 14 conv + 14 attention                  | 6 conv + 8 attention                                |
| Serialization      | Every stage attends, so every stage sorts | Only the two coarsest stages need an order        |

Everything else - the code-prefix pooling, the `SegmentCSR` reductions, the
sparse-convolution plugins, the head interfaces, the task models - is shared.

## Rotary position embedding

`Point3DRoPE` splits the head dimension into three equal per-axis chunks and
rotates each with a shared frequency ladder. Positions come from integer voxel
coordinates rather than metric ones, because per-stage `grid_coord` is already
part of the export contract while mean-pooled metric coordinates are not - and
because integer positions are exactly reproducible in the engine.

**The head dimension must be divisible by six, not three.** Each chunk pairs
dimension `i` with `i + chunk // 2`, so an odd chunk width mispairs the halves
and silently stops being a rotation. `rotary_span` returns the largest multiple
of six that fits, and the remaining one to five dimensions are passed through
unrotated. That keeps tensor-core-friendly head dimensions usable: at
`head_dim = 32`, thirty dimensions rotate and two form a NoPE tail. Partial
rotation is standard practice in rotary implementations, and the graph cost is
unchanged either way because the rotation is emitted as one gather plus two
elementwise operations over the full head dimension, with `cos = 1`/`sin = 0`
across the tail.

The default `enc_rope_base` of 100.0 mirrors what LitePT's reference CUDA
operator calls `rope_freq`. The two have **not** been matched numerically, so a
checkpoint trained against that operator needs verification before reuse.

## Available Configurations

| Config Name                                     | Task           | Dataset   | Range | Purpose                |
| ------------------------------------------------ | -------------- | --------- | ----- | ---------------------- |
| `segmentation3d/litept/voxel012_122m_t4dataset_j6gen2` | segmentation3d | T4Dataset | 122 m | T4Dataset segmentation |

The config inherits its dataset, transforms, optimizer, and deploy settings
from the PTv3 config of the same name and swaps only
`/tasks/segmentation3d/litept/encoder` and `/tasks/segmentation3d/litept/seg_head`.

```bash
autoware-ml train --config-name segmentation3d/litept/voxel012_122m_t4dataset_j6gen2
```

## Export contract

LitePT exports through the same machinery as PTv3, described in
[PointTransformerV3](ptv3.md#onnx-preprocessing-contract). The contract is
**narrower**, and `EncoderExportContract` derives the difference from the stage
gating rather than hard-coding it:

- **`serialized_code` is absent.** Its only in-graph consumer is stage-0
  attention. With `enc_attn[0]` false, nothing reads the base order, so the
  input and its input-resolution `autoware::Argsort` calls both disappear.
- **Order tensors only where a stage attends.** Pooling stage `i` feeds encoder
  stage `i + 1`, so it supplies `serialized_order`/`serialized_inverse` only
  when that stage carries attention. With the default gating, poolings 0 and 1
  supply `indices`, `indptr`, `head_indices`, and `grid_coord` alone.
- **The head loses its serialization inputs.** With `dec_depths` all zero, the
  segmentation head graph reduces to per-stage features plus pooling clusters.

This is not only an optimization. Inputs the traced graph never consumes are
pruned by the exporter, so a contract that over-declares produces a mismatched
signature. Deployment consumers must mirror the rule from the artifact's
`enc_attn` and `dec_depths`, exactly as they already mirror `dec_depths` for
the PTv3 head.

With the default five-stage configuration the encoder graph takes 22 inputs
against PTv3's 27.

!!! note "Patch size at the coarsest stages"

    Attention runs only at 1/8 and 1/16 resolution, where the voxel count can
    approach or fall below `enc_patch_size`. That is handled by the cyclic
    window fill described in
    [PointTransformerV3](ptv3.md#attention-window-fill-at-export), which keeps
    the window size static and needs no configuration. It is still worth
    measuring the stage-3 and stage-4 voxel counts on the sparsest frames you
    deploy against: a stage that sits below one window becomes global attention
    over that stage, which is correct but changes what the layer is doing.

## Implementation

LitePT adds no module of its own beyond the encoder subclass and the rotary
embedding; the rest is shared PTv3 code.

| Path                                                 | Description                                     |
| ---------------------------------------------------- | ----------------------------------------------- |
| `autoware_ml/models/segmentation3d/encoders/ptv3.py` | `LitePTEncoder`, `Point3DRoPE`, block gating    |
| `autoware_ml/models/segmentation3d/ptv3_base.py`     | `EncoderExportContract`, `PTv3EncoderExportBase` |
| `autoware_ml/models/segmentation3d/heads/ptv3.py`    | Decoder gating (`dec_conv`, `dec_attn`)         |
| `autoware_ml/configs/tasks/segmentation3d/litept/`   | Task configurations                             |
| `autoware_ml/tests/models/test_litept.py`            | Rotary, gating, and export-contract tests       |

## Acknowledgment

LitePT follows the published topology of the LitePT point-transformer variant,
reimplemented on top of the Autoware-ML PTv3 components. The reference
implementation's kernel-5 submanifold stem is not adopted here: it exports
poorly and showed no accuracy advantage over the linear stem.
