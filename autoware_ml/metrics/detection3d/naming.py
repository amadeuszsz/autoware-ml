"""Metric-key token helpers for detection components.

Keeps key formatting in one place so components stay focused on selecting values.
"""

from __future__ import annotations

from autoware_ml.metrics.base import _distance_token


def metric_token(value: str) -> str:
    """Lowercase, underscore-separated token safe for a metric key."""
    return value.lower().replace(" ", "_").replace("/", "_")


def label_metric_name(label: int, class_names: tuple[str, ...] | None) -> str:
    """Token for a class label, ``class_{label}`` only when no names exist.

    With ``class_names`` configured, an out-of-range label is a folding or
    configuration bug and raises instead of silently minting a phantom class key.
    """
    if class_names is None:
        return f"class_{label}"
    if not 0 <= label < len(class_names):
        raise ValueError(f"label {label} is outside the configured {len(class_names)} class names.")
    return metric_token(class_names[label])


def number_token(value: float) -> str:
    """Key-safe token for a plain number, e.g. ``99p5`` for 99.5."""
    return f"{float(value):g}".replace("-", "minus").replace(".", "p")


def threshold_token(threshold: float) -> str:
    """Collision-free token for a distance threshold, e.g. ``0p5m`` for 0.5.

    Delegates to the shared distance-token rule so range suffixes and threshold
    tokens can never drift apart.
    """
    return _distance_token(threshold)
