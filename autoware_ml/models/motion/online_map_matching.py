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

"""One to one matching of predicted polylines to map elements, invariant to the point order.

An open element matches in either direction, a closed ring from any start point in either
direction (Liao et al., MapTR, 2023). Coordinates are compared normalized to the mapped region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from autoware_ml.dataclasses.batch.online_map import OnlineMapGTBatch
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost


def normalize_polylines(points: torch.Tensor, roi: Sequence[float]) -> torch.Tensor:
    """Map points of the region ``(x_min, y_min, x_max, y_max)`` to the unit square.

    Args:
        points: Points ``(..., 2)``.
        roi: The mapped region.

    Returns:
        The normalized points.
    """
    x_min, y_min, x_max, y_max = roi
    origin = points.new_tensor([x_min, y_min])
    extent = points.new_tensor([x_max - x_min, y_max - y_min])
    return (points - origin) / extent


def point_orders(points: torch.Tensor, closed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every point order an element matches in.

    Args:
        points: Points of every element ``(M, N, 2)``, a closed ring repeating its first point
            last.
        closed: Whether every element is a closed ring ``(M,)``.

    Returns:
        Orders ``(M, 2 * (N - 1), N, 2)`` and whether each order applies ``(M, 2 * (N - 1))``:
        both directions of an open element, every start of a ring in both directions.
    """
    num_elements, num_points = points.shape[:2]
    ring = points[:, :-1]
    starts = torch.arange(num_points - 1, device=points.device)
    index = (starts[:, None] + starts[None, :]) % (num_points - 1)
    forward = ring[:, index]
    backward = ring.flip(1)[:, index]
    rolled = torch.cat([forward, backward], dim=1)
    rolled = torch.cat([rolled, rolled[:, :, :1]], dim=2)
    open_orders = torch.cat([points[:, None], points.flip(1)[:, None], rolled[:, 2:]], dim=1)
    orders = torch.where(closed[:, None, None, None], rolled, open_orders)
    valid = closed[:, None] | (torch.arange(2 * (num_points - 1), device=points.device) < 2)
    return orders, valid.expand(num_elements, -1)


def polyline_cost(
    pred: torch.Tensor, orders: torch.Tensor, valid: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth L1 distance of every prediction to every element in its closest order.

    Args:
        pred: Normalized predicted points ``(Q, N, 2)``.
        orders: Normalized point orders of the elements ``(M, P, N, 2)``.
        valid: Whether each order applies ``(M, P)``.
        beta: Smooth L1 transition point.

    Returns:
        Mean per point distance ``(Q, M)`` and the closest order ``(Q, M)``.
    """
    distance = (
        F.smooth_l1_loss(
            pred[:, None, None].expand(-1, *orders.shape),
            orders[None].expand(pred.shape[0], *orders.shape),
            reduction="none",
            beta=beta,
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )
    distance = distance.masked_fill(~valid[None], torch.finfo(distance.dtype).max)
    return distance.min(dim=-1)


@dataclass(frozen=True)
class PolylineTargets:
    """Targets of one decoder layer.

    Attributes:
        cls_targets: One hot class of every query, zeros for background ``(B, Q, C)``.
        point_targets: Normalized element points in the matched order ``(B, Q, N, 2)``.
        matched: Whether a query is matched ``(B, Q)``.
        num_pos: Number of matched queries.
    """

    cls_targets: torch.Tensor
    point_targets: torch.Tensor
    matched: torch.Tensor
    num_pos: int


def assign_polylines(
    cls_logits: torch.Tensor,
    pred_points: torch.Tensor,
    ground_truth: OnlineMapGTBatch,
    roi: Sequence[float],
    cls_cost: ClassificationCost,
    line_cost_weight: float,
    beta: float,
) -> PolylineTargets:
    """Hungarian match the queries of one layer to the elements of every sample.

    Args:
        cls_logits: Class logits ``(B, Q, C)``.
        pred_points: Predicted points ``(B, Q, N, 2)`` in meters.
        ground_truth: Elements of the batch in meters.
        roi: Mapped region the coordinates are normalized to.
        cls_cost: Focal classification cost.
        line_cost_weight: Weight of the line distance in the cost.
        beta: Smooth L1 transition point of the line distance.

    Returns:
        The targets of the layer.
    """
    batch_size, num_queries, num_classes = cls_logits.shape
    device = cls_logits.device
    pred = normalize_polylines(pred_points.detach().float(), roi)
    gt_points = normalize_polylines(ground_truth.points.float(), roi)
    valid_gt = ground_truth.labels >= 0
    class_cost = cls_cost(cls_logits.detach().float(), ground_truth.labels, valid_gt)
    cls_targets = cls_logits.new_zeros((batch_size, num_queries, num_classes))
    point_targets = pred.new_zeros(pred.shape)
    matched = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=device)
    num_pos = 0
    for sample in range(batch_size):
        elements = valid_gt[sample].nonzero().flatten()
        if elements.numel() == 0:
            continue
        orders, valid = point_orders(
            gt_points[sample, elements], ground_truth.closed[sample, elements]
        )
        line_cost, best_order = polyline_cost(pred[sample], orders, valid, beta)
        cost = class_cost[sample][:, elements] + line_cost_weight * line_cost
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        queries = torch.from_numpy(rows).to(device)
        matches = torch.from_numpy(columns).to(device)
        labels = ground_truth.labels[sample, elements[matches]]
        cls_targets[sample, queries, labels] = 1.0
        point_targets[sample, queries] = orders[matches, best_order[queries, matches]]
        matched[sample, queries] = True
        num_pos += int(queries.numel())
    return PolylineTargets(
        cls_targets=cls_targets,
        point_targets=point_targets,
        matched=matched,
        num_pos=num_pos,
    )
