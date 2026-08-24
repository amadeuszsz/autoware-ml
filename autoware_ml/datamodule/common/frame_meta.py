"""Per-frame evaluation metadata derived from the dataset infos.

The region and collision evaluation filters need two things the raw model inputs
do not carry: the frame's ego pose in the map frame (attached directly as
``ego2global``) and a scene identifier the lanelet map provider can resolve to
the scene's ``map/lanelet2_map.osm``. The scene identifier is derived here; the
datasets attach both at metadata time and the eval-output builders pass them
through per frame.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def scene_dir_fragment(lidar_path: str) -> str:
    """Scene directory fragment ``<db>/<uuid>/<version>`` from a lidar path.

    This is the ``scene_token`` convention the metrics use: unlike the opaque
    annotation token, it lets :class:`T4LaneletMapResolver` locate the scene's
    lanelet map directly under ``data_root``.
    """
    parts = PurePosixPath(lidar_path).parts
    if len(parts) < 4:
        raise ValueError(
            f"Cannot derive a scene directory from lidar path {lidar_path!r}; expected "
            "'<db>/<scene_uuid>/<version>/...'."
        )
    return "/".join(parts[:3])
