from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import os

from .convert_osm_network import (
    RoadGraph,
    convert_osm_network,
    graph_divide_arcs,
    graph_remove_disconnected_arcs,
    graph_remove_elements,
    graph_tag_sp_elements,
    read_osm_network_file,
    travel_mode_field,
    write_arcs_file,
    write_nodes_file,
)
from .map import find_nearest_node_linear
from .read_sim_files import read_hospitals_file, read_map_file, read_stations_file


@dataclass
class RoadsRunConfig:
    osm_filename: str
    hospitals_filename: str
    stations_filename: str
    map_filename: str
    nodes_output_filename: str
    arcs_output_filename: str
    border_filename: Optional[str] = None

    levels: set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5, 6})
    class_speeds: List[Dict[int, float]] = field(default_factory=list)
    class_off_road_access: Dict[int, bool] = field(default_factory=dict)
    sp_tag_class: List[bool] = field(default_factory=lambda: [True, True, True, True, True, False, False, False])
    max_arc_travel_time: float = 10.0 / 60.0 / 60.0 / 24.0

    def __post_init__(self) -> None:
        if not self.class_speeds:
            v2 = [30.0, 20.0, 15.0, 10.0, 8.0, 8.0]
            mode2 = {i + 1: v for i, v in enumerate(v2)}
            mode1 = {i + 1: v * 1.3 for i, v in enumerate(v2)}
            self.class_speeds = [mode1, mode2]
        if not self.class_off_road_access:
            self.class_off_road_access = {
                1: False,
                2: False,
                3: True,
                4: True,
                5: True,
                6: True,
                7: True,
                8: True,
            }


def _point_in_ring(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        cond = (y2 > y) != (y1 > y)
        if cond:
            x_on_edge = (x1 - x2) * (y - y2) / (y1 - y2) + x2
            if x < x_on_edge:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _load_shapefile_rings(border_filename: str) -> List[List[Tuple[float, float]]]:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Border clipping requires the 'pyshp' package (module name: shapefile). "
            "Install it with: pip install pyshp"
        ) from exc

    reader = shapefile.Reader(border_filename)
    shapes = reader.shapes()
    if len(shapes) != 1:
        raise ValueError(f"Expected exactly one polygon in border shapefile, found {len(shapes)}")

    shape = shapes[0]
    parts = list(shape.parts) + [len(shape.points)]
    rings: List[List[Tuple[float, float]]] = []
    seen = set()
    for i in range(len(parts) - 1):
        pts = shape.points[parts[i] : parts[i + 1]]
        ring = [(float(x), float(y)) for x, y in pts]
        key = tuple(ring)
        if key in seen:
            continue
        seen.add(key)
        rings.append(ring)
    return rings


def _point_in_rings(lon: float, lat: float, rings: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    count = 0
    for ring in rings:
        if _point_in_ring(lon, lat, ring):
            count += 1
    return (count % 2) == 1


def _nearest_od_nodes(nodes, hospitals_filename: str, stations_filename: str, map_filename: str) -> List[int]:
    hospitals = read_hospitals_file(hospitals_filename)
    stations = read_stations_file(stations_filename)
    mp = read_map_file(map_filename)

    od_nodes: List[int] = []
    for collection in (hospitals[1:], stations[1:]):
        for item in collection:
            idx, _dist = find_nearest_node_linear(mp, nodes, item.location)
            od_nodes.append(int(idx))
    return sorted(set(od_nodes))


def run_roads_pipeline(cfg: RoadsRunConfig, *, do_print: bool = True) -> RoadGraph:
    if do_print:
        print("reading OSM network")
    g = read_osm_network_file(cfg.osm_filename, levels=set(cfg.levels))

    if do_print:
        print("converting network classes/speeds")
    g = convert_osm_network(
        g.nodes,
        g.arcs,
        levels=set(cfg.levels),
        class_speeds=cfg.class_speeds,
        class_off_road_access=cfg.class_off_road_access,
    )

    if do_print:
        print("dividing long arcs")
    g = graph_divide_arcs(g.nodes, g.arcs, max_arc_travel_time=float(cfg.max_arc_travel_time))

    if do_print:
        print("finding hospital/station OD nodes")
    od_nodes = _nearest_od_nodes(g.nodes, cfg.hospitals_filename, cfg.stations_filename, cfg.map_filename)

    node_chosen = [False] * (len(g.nodes) + 1)
    for arc in g.arcs:
        cls = int(arc.fields["osm_class"])
        if 1 <= cls <= len(cfg.sp_tag_class) and cfg.sp_tag_class[cls - 1]:
            node_chosen[int(arc.from_node_index)] = True
            node_chosen[int(arc.to_node_index)] = True

    if cfg.border_filename:
        if do_print:
            print("clipping candidate nodes to border polygon")
        rings = _load_shapefile_rings(cfg.border_filename)
        for i in range(1, len(g.nodes) + 1):
            if not node_chosen[i]:
                continue
            loc = g.nodes[i - 1].location
            node_chosen[i] = _point_in_rings(float(loc.x), float(loc.y), rings)

    for i in od_nodes:
        node_chosen[i] = True
    chosen_nodes = [i for i in range(1, len(node_chosen)) if node_chosen[i]]

    if do_print:
        print("tagging shortest-path elements")
    graph_tag_sp_elements(
        g.nodes,
        g.arcs,
        chosen_nodes=chosen_nodes,
        origin_nodes=od_nodes,
        dest_nodes=od_nodes,
        sp_field="in_a_sp",
    )

    if do_print:
        print("filtering nodes/arcs by SP tags")
    g = graph_remove_elements(
        g.nodes,
        g.arcs,
        node_filter=lambda n: bool(n.fields.get("in_a_sp", False)),
        arc_filter=lambda a: (
            (
                1 <= int(a.fields["osm_class"]) <= len(cfg.sp_tag_class)
                and cfg.sp_tag_class[int(a.fields["osm_class"]) - 1]
            )
            or bool(a.fields.get("in_a_sp", False))
        ),
    )
    g = graph_remove_disconnected_arcs(g.nodes, g.arcs)

    # Keep only travel-time fields before writing.
    mode_count = 0
    if g.arcs:
        while travel_mode_field(mode_count + 1) in g.arcs[0].fields:
            mode_count += 1
    keep_fields = {travel_mode_field(i) for i in range(1, mode_count + 1)}

    for node in g.nodes:
        node.fields = {}
    for arc in g.arcs:
        arc.fields = {k: v for k, v in arc.fields.items() if k in keep_fields}

    os.makedirs(os.path.dirname(cfg.nodes_output_filename), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.arcs_output_filename), exist_ok=True)
    write_nodes_file(cfg.nodes_output_filename, g.nodes)
    write_arcs_file(cfg.arcs_output_filename, g.arcs, arc_form="directed")

    if do_print:
        print(f"wrote nodes: {cfg.nodes_output_filename} ({len(g.nodes)} rows)")
        print(f"wrote arcs:  {cfg.arcs_output_filename} ({len(g.arcs)} rows)")
    return g


def _profile_config(profile: str, jemss_root: str) -> RoadsRunConfig:
    if profile == "fdny_nyc":
        folder = os.path.join(jemss_root, "data/fdny/nyc/models/1")
        return RoadsRunConfig(
            osm_filename=os.path.join(jemss_root, "data/fdny/nyc/data/roads/nyc_main_roads.osm"),
            hospitals_filename=os.path.join(folder, "hospitals/hospitals_all.csv"),
            stations_filename=os.path.join(folder, "stations/stations_updated.csv"),
            map_filename=os.path.join(folder, "maps/map_all.csv"),
            nodes_output_filename=os.path.join(folder, "travel/roads/nodes.csv"),
            arcs_output_filename=os.path.join(folder, "travel/roads/arcs.csv"),
            # Mirrors the reference run.jl (which currently points at this file).
            border_filename=os.path.join(jemss_root, "data/fdny/bronx/data/border/Bronx.shp"),
        )
    if profile == "fdny_bronx":
        folder = os.path.join(jemss_root, "data/fdny/bronx/models/1")
        return RoadsRunConfig(
            osm_filename=os.path.join(jemss_root, "data/fdny/bronx/data/roads/bronx_main_roads.osm"),
            hospitals_filename=os.path.join(folder, "hospitals/hospitals_1.csv"),
            stations_filename=os.path.join(folder, "stations/stations_1.csv"),
            map_filename=os.path.join(folder, "maps/map_1.csv"),
            nodes_output_filename=os.path.join(folder, "travel/roads/nodes.csv"),
            arcs_output_filename=os.path.join(folder, "travel/roads/arcs.csv"),
            border_filename=os.path.join(jemss_root, "data/fdny/bronx/data/border/Bronx.shp"),
        )
    raise ValueError(f"Unknown profile: {profile}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Python equivalent of JEMSS travel/roads/run.jl.")
    parser.add_argument(
        "--profile",
        choices=["fdny_nyc", "fdny_bronx"],
        default="fdny_nyc",
        help="Preset matching the reference run.jl parameters",
    )
    parser.add_argument(
        "--jemss-root",
        default="/Users/longderek/Documents/FDNY/JEMSS/gksYr",
        help="Path to the reference JEMSS repository root",
    )
    parser.add_argument(
        "--no-border-clip",
        action="store_true",
        help="Skip shapefile clipping even if profile provides a border file",
    )
    parser.add_argument(
        "--max-arc-travel-time",
        type=float,
        default=10.0 / 60.0 / 60.0 / 24.0,
        help="Maximum per-arc travel time (days) before splitting arcs",
    )
    args = parser.parse_args(argv)

    cfg = _profile_config(args.profile, os.path.abspath(args.jemss_root))
    cfg.max_arc_travel_time = float(args.max_arc_travel_time)
    if args.no_border_clip:
        cfg.border_filename = None

    run_roads_pipeline(cfg, do_print=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
