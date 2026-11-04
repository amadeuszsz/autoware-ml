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

"""A small lanelet2 map with every kind of element the vector map reads or skips."""

from __future__ import annotations

from pathlib import Path

from autoware_ml.geometry.vector_map import MapClassSource

CLASSES = {
    "divider": MapClassSource(line_types=("line_thin", "line_thick")),
    "boundary": MapClassSource(line_types=("road_border",)),
    "ped_crossing": MapClassSource(regions=("crosswalk",)),
}

_NODES = {
    1: (0.0, 0.0),
    2: (10.0, 0.0),
    3: (20.0, 0.0),
    4: (0.0, -5.0),
    5: (20.0, -5.0),
    6: (0.0, 5.0),
    7: (20.0, 5.0),
    8: (12.0, -4.0),
    9: (12.0, 4.0),
    10: (14.0, -4.0),
    11: (14.0, 4.0),
}

# Two lane markings meeting at node 2, a road border, a virtual line and a crosswalk lanelet
_ELEMENTS = """  <way id="100"><nd ref="1"/><nd ref="2"/>
    <tag k="type" v="line_thin"/><tag k="subtype" v="dashed"/></way>
  <way id="101"><nd ref="2"/><nd ref="3"/>
    <tag k="type" v="line_thin"/><tag k="subtype" v="solid"/></way>
  <way id="102"><nd ref="4"/><nd ref="5"/><tag k="type" v="road_border"/></way>
  <way id="103"><nd ref="6"/><nd ref="7"/><tag k="type" v="virtual"/></way>
  <way id="104"><nd ref="8"/><nd ref="9"/><tag k="type" v="virtual"/></way>
  <way id="105"><nd ref="10"/><nd ref="11"/><tag k="type" v="virtual"/></way>
  <relation id="200">
    <member type="way" ref="104" role="left"/>
    <member type="way" ref="105" role="right"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="crosswalk"/>
  </relation>"""


def write_map(path: Path) -> Path:
    """Write the map to ``path`` and return it.

    In the map frame the markings run along y = 0 from x = 0 to 20, the road border along
    y = -5, and the crosswalk covers x 12 to 14, y -4 to 4.
    """
    nodes = "\n".join(
        f'  <node id="{node}"><tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/></node>'
        for node, (x, y) in _NODES.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<?xml version="1.0"?>\n<osm>\n{nodes}\n{_ELEMENTS}\n</osm>\n')
    return path
