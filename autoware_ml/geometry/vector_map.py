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

"""Map elements of a lanelet2 map as classed polylines, for online mapping supervision.

A class gathers either line strings by their ``type`` tag, such as painted lane markings or road
borders, or region polygons by their lanelet2 token, such as crosswalks. Connected line strings
of a class merge into one polyline, overlapping polygons of a class into one area. A scene's map
is read once and cropped around every frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

from autoware_ml.metrics.geometry.lanelet import (
    KNOWN_REGION_TOKENS,
    load_osm_map,
    local_xy,
    region_polygons,
)


@dataclass(frozen=True)
class MapClassSource:
    """Map elements that form one class, line strings or region polygons.

    Attributes:
        line_types: ``type`` tags of the line strings of the class.
        regions: Region tokens of the polygons of the class, lanelet subtypes or area way types.
    """

    line_types: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check that the class draws on exactly one kind of element.

        Raises:
            ValueError: If both or neither kind is set, or a region token is unknown.
        """
        object.__setattr__(self, "line_types", tuple(self.line_types))
        object.__setattr__(self, "regions", tuple(self.regions))
        if bool(self.line_types) == bool(self.regions):
            raise ValueError(
                "A map class takes either line_types or regions, got "
                f"line_types={self.line_types} and regions={self.regions}."
            )
        unknown = set(self.regions) - KNOWN_REGION_TOKENS
        if unknown:
            raise ValueError(f"Unknown region tokens {sorted(unknown)}.")


@dataclass(frozen=True)
class FrameMapElements:
    """Map elements around one frame, in the frame.

    Attributes:
        points: Vertices of every element, one element after another ``(P, 2)``.
        lengths: Vertex count of every element ``(M,)``.
        labels: Class index of every element ``(M,)``.
        areas: Whether an element outlines an area, its first and last vertex equal ``(M,)``.
    """

    points: npt.NDArray[np.float64]
    lengths: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    areas: npt.NDArray[np.bool_]

    def geometries(self) -> npt.NDArray[np.object_]:
        """Rebuild every element as a shapely geometry, a polygon for an area outline.

        Returns:
            One geometry per element.
        """
        offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        rings = [self.points[begin:end] for begin, end in zip(offsets[:-1], offsets[1:])]
        return np.array(
            [
                shapely.Polygon(ring) if area else LineString(ring)
                for ring, area in zip(rings, self.areas, strict=True)
            ],
            dtype=object,
        )


class VectorMap:
    """Classed map elements of one scene in the map frame."""

    def __init__(self, geometries: Sequence, labels: Sequence[int]) -> None:
        """Initialize the map.

        Args:
            geometries: Line strings and polygons in the map frame.
            labels: Class index of every geometry.

        Raises:
            ValueError: If geometries and labels disagree in count.
        """
        if len(geometries) != len(labels):
            raise ValueError(f"Got {len(geometries)} geometries and {len(labels)} labels.")
        self.geometries = np.asarray(geometries, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tree = STRtree(self.geometries)

    @staticmethod
    def from_osm(osm_path: str, classes: Mapping[str, MapClassSource]) -> VectorMap:
        """Read the elements of every class from a lanelet2 OSM, cached by path and classes.

        Args:
            osm_path: Path to the ``lanelet2_map.osm`` file.
            classes: Source of every class, in class index order.

        Returns:
            The map of the scene.
        """
        return _vector_map(osm_path, tuple(classes.values()))

    def crop(self, frame_to_map: npt.NDArray[np.float64], half_size: float) -> FrameMapElements:
        """Move the elements near a frame into its ground plane and cut them to a square.

        Args:
            frame_to_map: Pose of the frame in the map frame ``(4, 4)``.
            half_size: Half the side of the square around the frame origin.

        Returns:
            The elements inside the square. A line the square cuts in several pieces yields one
            element per piece.
        """
        # The elements carry no height, so only the position and heading of the frame place them
        heading = np.arctan2(frame_to_map[1, 0], frame_to_map[0, 0])
        cos, sin = np.cos(heading), np.sin(heading)
        to_map = np.array([[cos, -sin], [sin, cos]])
        rotation = to_map.T
        translation = -rotation @ frame_to_map[:2, 3]
        corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64) * half_size
        corners = corners @ to_map.T + frame_to_map[:2, 3]
        near = self.tree.query(shapely.box(*corners.min(axis=0), *corners.max(axis=0)))
        moved = shapely.transform(self.geometries[near], lambda xy: xy @ rotation.T + translation)
        cut = shapely.clip_by_rect(moved, -half_size, -half_size, half_size, half_size)
        parts, index = shapely.get_parts(cut, return_index=True)
        keep = ~shapely.is_empty(parts) & np.isin(
            shapely.get_type_id(parts),
            (shapely.GeometryType.LINESTRING, shapely.GeometryType.POLYGON),
        )
        parts, labels = parts[keep], self.labels[near][index[keep]]
        areas = shapely.get_type_id(parts) == shapely.GeometryType.POLYGON
        outlines = np.where(areas, shapely.get_exterior_ring(parts), parts)
        coords, counts = shapely.get_coordinates(outlines), shapely.get_num_coordinates(outlines)
        return FrameMapElements(
            points=coords,
            lengths=counts.astype(np.int64),
            labels=labels,
            areas=areas,
        )


@lru_cache(maxsize=64)
def _vector_map(osm_path: str, sources: tuple[MapClassSource, ...]) -> VectorMap:
    """Read and merge the elements of every class, see :meth:`VectorMap.from_osm`."""
    lanelet_map = load_osm_map(osm_path)
    regions = (
        region_polygons(osm_path, lanelet_map) if any(source.regions for source in sources) else {}
    )
    geometries, labels = [], []
    for label, source in enumerate(sources):
        if source.line_types:
            lines = [
                LineString([local_xy(point) for point in line_string])
                for line_string in lanelet_map.lineStringLayer
                if "type" in line_string.attributes
                and line_string.attributes["type"] in source.line_types
                and len(line_string) >= 2
            ]
            merged = shapely.get_parts(linemerge(lines)) if lines else []
        else:
            polygons = [polygon for token in source.regions for polygon in regions.get(token, [])]
            merged = shapely.get_parts(unary_union(polygons)) if polygons else []
        geometries.extend(merged)
        labels.extend([label] * len(merged))
    return VectorMap(geometries, labels)
