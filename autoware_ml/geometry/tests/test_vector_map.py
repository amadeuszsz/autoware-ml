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

"""Tests for reading classed map elements from a lanelet2 map and cutting them per frame."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import shapely

from autoware_ml.geometry.tests.vector_map_fixtures import CLASSES, write_map
from autoware_ml.geometry.vector_map import MapClassSource, VectorMap


def frame_pose(x: float, yaw: float) -> np.ndarray:
    """Transform from a frame at map position (x, 0) with heading yaw to the map."""
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[0, 3] = x
    return pose


class TestVectorMap(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_connected_lines_merge_and_untyped_classes_are_skipped(self) -> None:
        """
        Input: two lane markings meeting end to end, a road border, a virtual line and a crosswalk.
        Expected: one 20 m divider, one 20 m boundary and one 16 square meter crossing area.
        """
        vector_map = VectorMap.from_osm(str(write_map(self.tmp_path / "map.osm")), CLASSES)

        self.assertEqual(vector_map.labels.tolist(), [0, 1, 2])
        np.testing.assert_allclose(
            shapely.length(vector_map.geometries[:2]), [20.0, 20.0], rtol=1e-6
        )
        self.assertAlmostEqual(shapely.area(vector_map.geometries[2]), 16.0, delta=1e-5)

    def test_crop_moves_the_elements_into_the_frame_and_cuts_them(self) -> None:
        """
        Input: a frame at map (10, 0) facing map +y, cut to a 12 m square.
        Expected: the divider runs along frame x = 0 and the border along x = -5, both cut to 12 m.
          The crossing lies whole in the square as a closed outline.
        """
        vector_map = VectorMap.from_osm(str(write_map(self.tmp_path / "map.osm")), CLASSES)

        elements = vector_map.crop(frame_pose(10.0, np.pi / 2), half_size=6.0)

        geometries = elements.geometries()
        order = np.argsort(elements.labels)
        divider, boundary, crossing = geometries[order]
        self.assertEqual(elements.labels[order].tolist(), [0, 1, 2])
        self.assertEqual(elements.areas[order].tolist(), [False, False, True])
        np.testing.assert_allclose(shapely.get_coordinates(divider)[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(shapely.get_coordinates(boundary)[:, 0], -5.0, atol=1e-9)
        self.assertAlmostEqual(shapely.length(divider), 12.0, delta=1e-5)
        self.assertAlmostEqual(shapely.length(boundary), 12.0, delta=1e-5)
        np.testing.assert_allclose(
            shapely.bounds(crossing), [-4.0, -4.0, 4.0, -2.0], rtol=1e-6, atol=1e-9
        )

    def test_the_height_and_tilt_of_the_frame_do_not_move_the_elements(self) -> None:
        """
        Input: the frame of the crop test at 900 m height, pitched 4 deg and rolled 2 deg.
        Expected: the divider still runs along frame x = 0 and the border along x = -5.
        """
        vector_map = VectorMap.from_osm(str(write_map(self.tmp_path / "map.osm")), CLASSES)
        pose = frame_pose(10.0, np.pi / 2)
        pitch, roll = np.radians(4.0), np.radians(2.0)
        tilt_y = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        tilt_x = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )
        pose[:3, :3] = pose[:3, :3] @ tilt_y @ tilt_x
        pose[2, 3] = 900.0

        elements = vector_map.crop(pose, half_size=6.0)

        geometries = elements.geometries()
        order = np.argsort(elements.labels)
        divider, boundary = geometries[order][:2]
        np.testing.assert_allclose(shapely.get_coordinates(divider)[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(shapely.get_coordinates(boundary)[:, 0], -5.0, atol=1e-9)

    def test_a_frame_away_from_the_elements_gets_none(self) -> None:
        """
        Input: a frame 100 m away from every element.
        Expected: no element.
        """
        vector_map = VectorMap.from_osm(str(write_map(self.tmp_path / "map.osm")), CLASSES)

        elements = vector_map.crop(frame_pose(120.0, 0.0), half_size=6.0)

        self.assertEqual(len(elements.labels), 0)
        self.assertEqual(elements.points.shape, (0, 2))

    def test_a_class_needs_one_known_kind_of_element(self) -> None:
        """
        Input: a class with both kinds, with none, or with an unknown region token.
        Expected: a ValueError.
        """
        cases = [
            ({"line_types": ("road_border",), "regions": ("crosswalk",)}, "either line_types"),
            ({}, "either line_types"),
            ({"regions": ("parking",)}, "Unknown region tokens"),
        ]
        for source, message in cases:
            with self.subTest(message=message, source=source):
                with self.assertRaisesRegex(ValueError, message):
                    MapClassSource(**source)
