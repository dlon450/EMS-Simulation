from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import argparse
import copy
import heapq
import math
import os
import xml.etree.ElementTree as ET

try:  # package import
    from .file_io import Table, write_tables_to_file
    from .geo import Arc, Location, Node
except ImportError:  # direct script import
    from file_io import Table, write_tables_to_file
    from geo import Arc, Location, Node


# OSM highway class mapping compatible with the OpenStreetMapX class numbering
# used in the reference tooling.
ROAD_CLASS_BY_HIGHWAY: Dict[str, int] = {
    "motorway": 1,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 3,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 5,
    "tertiary_link": 5,
    "unclassified": 6,
    "residential": 6,
    "living_street": 8,
    "pedestrian": 8,
    "service": 7,
}


def travel_mode_field(mode_index: int) -> str:
    return f"mode_{mode_index}"


@dataclass
class OSMBounds:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def contains_lon_lat(self, lon: float, lat: float) -> bool:
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat


@dataclass
class RoadGraph:
    nodes: List[Node] = field(default_factory=list)
    arcs: List[Arc] = field(default_factory=list)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_osm_bounds(elem: ET.Element) -> OSMBounds:
    return OSMBounds(
        min_lat=float(elem.attrib["minlat"]),
        min_lon=float(elem.attrib["minlon"]),
        max_lat=float(elem.attrib["maxlat"]),
        max_lon=float(elem.attrib["maxlon"]),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _oneway_direction(oneway: str) -> int:
    s = (oneway or "").strip().lower()
    if s in {"yes", "true", "1"}:
        return 1
    if s == "-1":
        return -1
    return 0


def _normalise_bounds(
    parsed_bounds: Optional[OSMBounds],
    bounds_lla: Optional[Tuple[float, float, float, float]],
) -> OSMBounds:
    if bounds_lla is not None:
        min_lat, min_lon, max_lat, max_lon = bounds_lla
        return OSMBounds(min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon)
    if parsed_bounds is None:
        raise ValueError("OSM file has no <bounds> and no bounds_lla was provided")
    return parsed_bounds


def read_osm_network_file(
    osm_filename: str,
    *,
    levels: Optional[Set[int]] = None,
    bounds_lla: Optional[Tuple[float, float, float, float]] = None,
) -> RoadGraph:
    if levels is None:
        levels = set(ROAD_CLASS_BY_HIGHWAY.values())
    if not os.path.isfile(osm_filename):
        raise FileNotFoundError(osm_filename)

    node_lon_lat: Dict[int, Tuple[float, float]] = {}
    ways: List[Tuple[List[int], int, str]] = []
    parsed_bounds: Optional[OSMBounds] = None

    for event, elem in ET.iterparse(osm_filename, events=("start", "end")):
        tag = _strip_ns(elem.tag)
        if event == "start" and tag == "bounds":
            parsed_bounds = _parse_osm_bounds(elem)
            continue

        if event == "end" and tag == "node":
            node_id = int(elem.attrib["id"])
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            node_lon_lat[node_id] = (lon, lat)
            elem.clear()
            continue

        if event == "end" and tag == "way":
            nd_refs: List[int] = []
            highway: Optional[str] = None
            oneway = ""
            for child in list(elem):
                ctag = _strip_ns(child.tag)
                if ctag == "nd":
                    ref = child.attrib.get("ref")
                    if ref:
                        nd_refs.append(int(ref))
                elif ctag == "tag":
                    k = child.attrib.get("k", "")
                    v = child.attrib.get("v", "")
                    if k == "highway":
                        highway = v
                    elif k == "oneway":
                        oneway = v

            if highway is not None and highway in ROAD_CLASS_BY_HIGHWAY:
                osm_class = ROAD_CLASS_BY_HIGHWAY[highway]
                if osm_class in levels and len(nd_refs) >= 2:
                    ways.append((nd_refs, osm_class, oneway))
            elem.clear()

    bounds = _normalise_bounds(parsed_bounds, bounds_lla)

    used_node_ids: Set[int] = set()
    for nd_refs, _, _ in ways:
        for node_id in nd_refs:
            if node_id not in node_lon_lat:
                continue
            lon, lat = node_lon_lat[node_id]
            if bounds.contains_lon_lat(lon, lat):
                used_node_ids.add(node_id)

    # Deterministic 1..n indexing.
    sorted_node_ids = sorted(used_node_ids)
    node_index: Dict[int, int] = {}
    nodes: List[Node] = []
    for i, node_id in enumerate(sorted_node_ids, start=1):
        lon, lat = node_lon_lat[node_id]
        n = Node(index=i, location=Location(x=lon, y=lat))
        n.fields["osm_key"] = node_id
        nodes.append(n)
        node_index[node_id] = i

    arcs: List[Arc] = []
    arc_i = 0
    for nd_refs, osm_class, oneway in ways:
        direction = _oneway_direction(oneway)
        for a, b in zip(nd_refs[:-1], nd_refs[1:]):
            if a not in node_index or b not in node_index:
                continue
            ia = node_index[a]
            ib = node_index[b]
            lon_a, lat_a = node_lon_lat[a]
            lon_b, lat_b = node_lon_lat[b]
            dist_km = _haversine_km(lat_a, lon_a, lat_b, lon_b)

            def add_arc(fr: int, to: int) -> None:
                nonlocal arc_i
                arc_i += 1
                arc = Arc(index=arc_i, from_node_index=fr, to_node_index=to, distance=dist_km)
                arc.fields["osm_class"] = osm_class
                arcs.append(arc)

            if direction == 1:
                add_arc(ia, ib)
            elif direction == -1:
                add_arc(ib, ia)
            else:
                add_arc(ia, ib)
                add_arc(ib, ia)

    return RoadGraph(nodes=nodes, arcs=arcs)


def _node_by_index(nodes: Sequence[Node]) -> Dict[int, Node]:
    return {int(n.index): n for n in nodes if n.index is not None}


def _renumber_graph(nodes: List[Node], arcs: List[Arc]) -> RoadGraph:
    old_to_new: Dict[int, int] = {}
    for i, node in enumerate(nodes, start=1):
        old = int(node.index)
        old_to_new[old] = i
        node.index = i

    new_arcs: List[Arc] = []
    for arc in arcs:
        fr = old_to_new.get(int(arc.from_node_index))
        to = old_to_new.get(int(arc.to_node_index))
        if fr is None or to is None:
            continue
        arc.from_node_index = fr
        arc.to_node_index = to
        new_arcs.append(arc)

    for i, arc in enumerate(new_arcs, start=1):
        arc.index = i
    return RoadGraph(nodes=nodes, arcs=new_arcs)


def graph_remove_elements(
    nodes: List[Node],
    arcs: List[Arc],
    *,
    node_filter=lambda _n: True,
    arc_filter=lambda _a: True,
) -> RoadGraph:
    kept_nodes = [n for n in nodes if node_filter(n)]
    kept_arcs = [a for a in arcs if arc_filter(a)]
    return _renumber_graph(kept_nodes, kept_arcs)


def graph_remove_disconnected_arcs(nodes: List[Node], arcs: List[Arc]) -> RoadGraph:
    num_nodes = len(nodes)
    return graph_remove_elements(
        nodes,
        arcs,
        arc_filter=lambda a: 1 <= int(a.from_node_index) <= num_nodes and 1 <= int(a.to_node_index) <= num_nodes,
    )


def _build_node_adjacency(num_nodes: int, arcs: Sequence[Arc]) -> Tuple[List[List[int]], List[List[int]]]:
    adj = [[] for _ in range(num_nodes + 1)]
    rev = [[] for _ in range(num_nodes + 1)]
    for arc in arcs:
        u = int(arc.from_node_index)
        v = int(arc.to_node_index)
        adj[u].append(v)
        rev[v].append(u)
    return adj, rev


def _largest_strong_component_nodes(num_nodes: int, arcs: Sequence[Arc]) -> Set[int]:
    adj, rev = _build_node_adjacency(num_nodes, arcs)

    visited = [False] * (num_nodes + 1)
    order: List[int] = []
    for start in range(1, num_nodes + 1):
        if visited[start]:
            continue
        stack: List[Tuple[int, int]] = [(start, 0)]
        visited[start] = True
        while stack:
            u, i = stack[-1]
            if i < len(adj[u]):
                v = adj[u][i]
                stack[-1] = (u, i + 1)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
                stack.pop()

    visited = [False] * (num_nodes + 1)
    largest: Set[int] = set()
    for start in reversed(order):
        if visited[start]:
            continue
        comp: Set[int] = set()
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in rev[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) > len(largest):
            largest = comp
    return largest


def graph_keep_largest_component(nodes: List[Node], arcs: List[Arc]) -> RoadGraph:
    largest = _largest_strong_component_nodes(len(nodes), arcs)
    return graph_remove_elements(
        nodes,
        arcs,
        node_filter=lambda n: int(n.index) in largest,
        arc_filter=lambda a: int(a.from_node_index) in largest and int(a.to_node_index) in largest,
    )


def merge_duplicate_osm_arcs(target: Arc, source: Arc, *, class_field: str = "osm_class") -> None:
    target.fields[class_field] = min(int(target.fields[class_field]), int(source.fields[class_field]))


def graph_merge_duplicate_arcs(
    nodes: List[Node],
    arcs: List[Arc],
    *,
    merge_fn=merge_duplicate_osm_arcs,
) -> RoadGraph:
    grouped: Dict[Tuple[int, int], List[Arc]] = {}
    for arc in arcs:
        key = (int(arc.from_node_index), int(arc.to_node_index))
        grouped.setdefault(key, []).append(arc)

    merged: List[Arc] = []
    for group in grouped.values():
        base = group[0]
        for extra in group[1:]:
            merge_fn(base, extra)
        merged.append(base)
    return _renumber_graph(list(nodes), merged)


def get_num_travel_modes(arcs: Sequence[Arc]) -> int:
    if not arcs:
        return 0
    i = 1
    while travel_mode_field(i) in arcs[0].fields:
        i += 1
    return i - 1


def convert_osm_network(
    nodes: List[Node],
    arcs: List[Arc],
    *,
    levels: Optional[Set[int]] = None,
    class_speeds: Sequence[Dict[int, float]],
    class_off_road_access: Dict[int, bool],
    merge_arcs_fn=merge_duplicate_osm_arcs,
) -> RoadGraph:
    if levels is None:
        levels = set(ROAD_CLASS_BY_HIGHWAY.values())
    if not class_speeds:
        raise ValueError("class_speeds must contain at least one travel mode")

    for level in levels:
        for mode_speeds in class_speeds:
            if level not in mode_speeds:
                raise ValueError(f"class_speeds missing class {level}")
        if level not in class_off_road_access:
            raise ValueError(f"class_off_road_access missing class {level}")

    # Keep desired classes and normalise graph topology.
    g = graph_remove_elements(nodes, arcs, arc_filter=lambda a: int(a.fields.get("osm_class", -1)) in levels)
    g = graph_remove_disconnected_arcs(g.nodes, g.arcs)
    g = graph_keep_largest_component(g.nodes, g.arcs)
    g = graph_merge_duplicate_arcs(g.nodes, g.arcs, merge_fn=merge_arcs_fn)

    for node in g.nodes:
        node.off_road_access = False
    for arc in g.arcs:
        cls = int(arc.fields["osm_class"])
        if class_off_road_access.get(cls, False):
            g.nodes[int(arc.from_node_index) - 1].off_road_access = True
            g.nodes[int(arc.to_node_index) - 1].off_road_access = True

    for mode_i, mode_speeds in enumerate(class_speeds, start=1):
        f = travel_mode_field(mode_i)
        for arc in g.arcs:
            cls = int(arc.fields["osm_class"])
            speed = float(mode_speeds[cls])
            arc.fields[f] = float(arc.distance) / (speed * 24.0)

    return g


def _segment_nodes(start: int, end: int, mids: Sequence[int], reverse: bool) -> List[int]:
    if not reverse:
        return [start, *mids, end]
    return [end, *list(reversed(mids)), start]


def graph_divide_arcs(nodes: List[Node], arcs: List[Arc], *, max_arc_travel_time: float = math.inf) -> RoadGraph:
    if max_arc_travel_time == math.inf:
        return RoadGraph(nodes=list(nodes), arcs=list(arcs))
    if max_arc_travel_time <= 0:
        raise ValueError("max_arc_travel_time must be > 0")
    if not arcs:
        return RoadGraph(nodes=list(nodes), arcs=[])

    mode_fields = [travel_mode_field(i) for i in range(1, get_num_travel_modes(arcs) + 1)]
    grouped: Dict[Tuple[int, int], List[Arc]] = {}
    for arc in arcs:
        u = int(arc.from_node_index)
        v = int(arc.to_node_index)
        key = (u, v) if u <= v else (v, u)
        grouped.setdefault(key, []).append(arc)

    out_nodes: List[Node] = [copy.deepcopy(n) for n in nodes]
    out_arcs: List[Arc] = []
    next_node = len(out_nodes) + 1

    for _, group in grouped.items():
        max_tt = 0.0
        for arc in group:
            for f in mode_fields:
                max_tt = max(max_tt, float(arc.fields[f]))
        num_divides = max(0, int(math.ceil(max_tt / max_arc_travel_time) - 1))

        if num_divides == 0:
            out_arcs.extend(copy.deepcopy(a) for a in group)
            continue

        # Use the first arc orientation as baseline for interpolation.
        base = group[0]
        u = int(base.from_node_index)
        v = int(base.to_node_index)
        nu = out_nodes[u - 1]
        nv = out_nodes[v - 1]

        mids: List[int] = []
        for i in range(1, num_divides + 1):
            frac = i / float(num_divides + 1)
            mid = Node(index=next_node)
            mid.location = Location(
                x=float(nu.location.x) + frac * (float(nv.location.x) - float(nu.location.x)),
                y=float(nu.location.y) + frac * (float(nv.location.y) - float(nu.location.y)),
            )
            mid.off_road_access = bool(nu.off_road_access and nv.off_road_access)
            out_nodes.append(mid)
            mids.append(next_node)
            next_node += 1

        for arc in group:
            reverse = not (int(arc.from_node_index) == u and int(arc.to_node_index) == v)
            path_nodes = _segment_nodes(u, v, mids, reverse)

            seg_count = float(num_divides + 1)
            seg_distance = float(arc.distance) / seg_count
            seg_fields = dict(arc.fields)
            for f in mode_fields:
                seg_fields[f] = float(arc.fields[f]) / seg_count

            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                seg = Arc(from_node_index=a, to_node_index=b, distance=seg_distance, fields=dict(seg_fields))
                out_arcs.append(seg)

    return _renumber_graph(out_nodes, out_arcs)


def _build_weighted_adjacency(
    num_nodes: int,
    arcs: Sequence[Arc],
    *,
    mode_field: str,
    reverse: bool = False,
) -> List[List[Tuple[int, float, int]]]:
    adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(num_nodes + 1)]
    for arc in arcs:
        u = int(arc.from_node_index)
        v = int(arc.to_node_index)
        w = float(arc.fields[mode_field])
        arc_i = int(arc.index)
        if reverse:
            adj[v].append((u, w, arc_i))
        else:
            adj[u].append((v, w, arc_i))
    return adj


def _dijkstra(
    num_nodes: int,
    source: int,
    adj: Sequence[Sequence[Tuple[int, float, int]]],
) -> Tuple[List[float], List[int], List[int]]:
    inf = float("inf")
    dist = [inf] * (num_nodes + 1)
    parent_node = [0] * (num_nodes + 1)
    parent_arc = [0] * (num_nodes + 1)
    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w, arc_i in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent_node[v] = u
                parent_arc[v] = arc_i
                heapq.heappush(heap, (nd, v))
    return dist, parent_node, parent_arc


def graph_tag_sp_elements(
    nodes: List[Node],
    arcs: List[Arc],
    *,
    chosen_nodes: Sequence[int],
    origin_nodes: Sequence[int],
    dest_nodes: Sequence[int],
    sp_field: str = "in_a_sp",
) -> None:
    if not arcs:
        for node in nodes:
            node.fields[sp_field] = False
        return

    mode_fields = [travel_mode_field(i) for i in range(1, get_num_travel_modes(arcs) + 1)]
    num_nodes = len(nodes)
    num_arcs = len(arcs)

    node_tagged = [False] * (num_nodes + 1)
    arc_tagged = [False] * (num_arcs + 1)

    def tag_tree(source: int, targets: Sequence[int], adj: Sequence[Sequence[Tuple[int, float, int]]]) -> None:
        _, parent_node, parent_arc = _dijkstra(num_nodes, source, adj)
        local_tag = [False] * (num_nodes + 1)
        local_tag[source] = True
        for t in targets:
            j = int(t)
            while j > 0 and not local_tag[j]:
                local_tag[j] = True
                k = j
                j = parent_node[j]
                if j > 0:
                    a = parent_arc[k]
                    if a > 0:
                        arc_tagged[a] = True
        for i in range(1, num_nodes + 1):
            if local_tag[i]:
                node_tagged[i] = True

    targets_from_origin = list(dict.fromkeys([*chosen_nodes, *dest_nodes]))
    targets_from_dest = list(dict.fromkeys([*chosen_nodes, *origin_nodes]))

    for mode_field in mode_fields:
        adj = _build_weighted_adjacency(num_nodes, arcs, mode_field=mode_field, reverse=False)
        rev = _build_weighted_adjacency(num_nodes, arcs, mode_field=mode_field, reverse=True)

        for o in origin_nodes:
            tag_tree(int(o), targets_from_origin, adj)
        for d in dest_nodes:
            tag_tree(int(d), targets_from_dest, rev)

    for node in nodes:
        node.fields[sp_field] = node_tagged[int(node.index)]
    for arc in arcs:
        arc.fields[sp_field] = arc_tagged[int(arc.index)]


def write_nodes_file(filename: str, nodes: Sequence[Node]) -> None:
    table = Table(
        name="nodes",
        header=["index", "x", "y", "offRoadAccess"],
        data=[
            [
                float(n.index),
                float(n.location.x),
                float(n.location.y),
                float(1 if n.off_road_access else 0),
            ]
            for n in nodes
        ],
    )
    write_tables_to_file(filename, [table])


def write_arcs_file(filename: str, arcs: Sequence[Arc], *, arc_form: str = "directed") -> None:
    if arc_form not in {"directed", "undirected"}:
        raise ValueError("arc_form must be 'directed' or 'undirected'")
    num_modes = get_num_travel_modes(arcs)
    mode_headers = [travel_mode_field(i) for i in range(1, num_modes + 1)]

    misc = Table(name="miscData", header=["arcForm", "numModes"], data=[[arc_form, int(num_modes)]])
    arcs_table = Table(
        name="arcs",
        header=["index", "fromNode", "toNode", "distance", *mode_headers],
        data=[
            [
                float(a.index),
                float(a.from_node_index),
                float(a.to_node_index),
                float(a.distance),
                *[float(a.fields[h]) for h in mode_headers],
            ]
            for a in arcs
        ],
    )
    write_tables_to_file(filename, [misc, arcs_table])


def convert_osm_network_file(
    osm_filename: str,
    *,
    levels: Optional[Set[int]],
    class_speeds: Sequence[Dict[int, float]],
    class_off_road_access: Dict[int, bool],
    bounds_lla: Optional[Tuple[float, float, float, float]] = None,
    max_arc_travel_time: float = math.inf,
) -> RoadGraph:
    g = read_osm_network_file(osm_filename, levels=levels, bounds_lla=bounds_lla)
    g = convert_osm_network(
        g.nodes,
        g.arcs,
        levels=levels,
        class_speeds=class_speeds,
        class_off_road_access=class_off_road_access,
    )
    g = graph_divide_arcs(g.nodes, g.arcs, max_arc_travel_time=max_arc_travel_time)
    return g


def _parse_levels(s: str) -> Set[int]:
    out: Set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    if not out:
        raise ValueError("No levels parsed")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert an OSM road network into JEMSS-compatible nodes/arcs CSVs.")
    parser.add_argument("--osm", required=True, help="Input .osm file")
    parser.add_argument("--nodes-out", required=True, help="Output nodes.csv")
    parser.add_argument("--arcs-out", required=True, help="Output arcs.csv")
    parser.add_argument("--levels", default="1,2,3,4,5,6", help="Comma-separated OSM class levels to keep")
    parser.add_argument(
        "--max-arc-travel-time",
        type=float,
        default=10.0 / 60.0 / 60.0 / 24.0,
        help="Maximum arc travel time in days before splitting arcs",
    )
    args = parser.parse_args(argv)

    levels = _parse_levels(args.levels)
    v2 = [30.0, 20.0, 15.0, 10.0, 8.0, 8.0]
    mode2 = {i + 1: v for i, v in enumerate(v2)}
    mode1 = {i + 1: v * 1.3 for i, v in enumerate(v2)}
    class_speeds = [mode1, mode2]
    class_off_road_access = {1: False, 2: False, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True}

    g = convert_osm_network_file(
        args.osm,
        levels=levels,
        class_speeds=class_speeds,
        class_off_road_access=class_off_road_access,
        max_arc_travel_time=float(args.max_arc_travel_time),
    )
    write_nodes_file(args.nodes_out, g.nodes)
    write_arcs_file(args.arcs_out, g.arcs)
    print(f"wrote {len(g.nodes)} nodes -> {args.nodes_out}")
    print(f"wrote {len(g.arcs)} arcs  -> {args.arcs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
