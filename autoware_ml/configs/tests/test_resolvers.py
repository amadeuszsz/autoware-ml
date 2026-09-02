"""Tests for bundled OmegaConf config resolvers."""

from __future__ import annotations

from types import SimpleNamespace

from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import (
    merge_lists,
    list_length,
    register_config_resolvers,
)


def test_list_length_counts_the_elements() -> None:
    assert list_length(["car", "truck"]) == 2
    assert list_length(OmegaConf.create([1, 2, 3])) == 3
    assert list_length([]) == 0


def test_list_length_resolver_derives_a_count_from_a_list() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "class_names": ["car", "truck", "bus"],
            "num_classes": "${list_length:${class_names}}",
        }
    )
    assert OmegaConf.to_container(cfg, resolve=True)["num_classes"] == 3


def test_merge_lists_concatenates_in_order() -> None:
    assert OmegaConf.to_container(merge_lists([1, 2], [3], [4, 5])) == [1, 2, 3, 4, 5]
    assert OmegaConf.to_container(merge_lists([])) == []
    assert OmegaConf.to_container(merge_lists()) == []


def test_merge_lists_resolver_appends_across_namespaces() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "det": {"metrics": [{"name": "map", "classes": "${classes}"}]},
            "seg": {"metrics": [{"name": "iou"}]},
            "classes": ["car", "truck"],
            "metrics": "${merge_lists:${det.metrics},${seg.metrics}}",
        }
    )
    merged = OmegaConf.to_container(cfg, resolve=True)["metrics"]
    assert [m["name"] for m in merged] == ["map", "iou"]
    assert merged[0]["classes"] == ["car", "truck"]


def test_merge_lists_resolver_preserves_hydra_recursive_instantiation() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "det": {
                "metrics": [
                    {
                        "_target_": "types.SimpleNamespace",
                        "name": "map",
                        "classes": "${classes}",
                    }
                ]
            },
            "seg": {"metrics": [{"_target_": "types.SimpleNamespace", "name": "iou"}]},
            "classes": ["car", "truck"],
            "model": {
                "_target_": "types.SimpleNamespace",
                "metrics": "${merge_lists:${det.metrics},${seg.metrics}}",
            },
        }
    )

    model = instantiate(cfg.model)

    assert [type(metric) for metric in model.metrics] == [SimpleNamespace, SimpleNamespace]
    assert [metric.name for metric in model.metrics] == ["map", "iou"]
    assert model.metrics[0].classes == ["car", "truck"]
