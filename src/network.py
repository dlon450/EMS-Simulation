from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

from .defs import Priority
from .geo import Arc, Node


@dataclass
class Graph:
    is_reduced: bool = False
    nodes: List[Node] = field(default_factory=list)
    arcs: List[Arc] = field(default_factory=list)
    arc_dists: List[float] = field(default_factory=list)
    # adjacency lists in lightgraphs style
    fadj_list: List[List[int]] = field(default_factory=list)
    badj_list: List[List[int]] = field(default_factory=list)
    # mapping (from_node, to_node) -> arc_index
    node_pair_arc_index: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # outgoing arcs per node (index 0 unused for 1-based graphs)
    out_arcs: List[List[int]] = field(default_factory=list)

    def add_node(self, node: Node) -> int:
        """Append a node and return its index.

        The port uses **1-based indexing** (index 0 is a dummy element) to
        mirror the Julia implementation and avoid off-by-one errors when
        comparing against Julia outputs.
        """

        if not self.nodes:
            # Ensure a dummy element at index 0.
            self.nodes.append(Node(index=0))
        node.index = len(self.nodes)
        self.nodes.append(node)
        # Ensure adjacency structures are long enough.
        while len(self.fadj_list) <= node.index:
            self.fadj_list.append([])
            self.badj_list.append([])
        while len(self.out_arcs) <= node.index:
            self.out_arcs.append([])
        return int(node.index)

    def add_arc(self, arc: Arc) -> int:
        """Append an arc and update adjacency lists.

        Like :meth:`add_node`, this method assumes 1-based indexing with a
        dummy element at index 0.
        """

        if not self.arcs:
            # Ensure a dummy element at index 0.
            self.arcs.append(Arc(index=0))
            self.arc_dists.append(0.0)
        arc.index = len(self.arcs)
        self.arcs.append(arc)
        dist = arc.distance
        self.arc_dists.append(float(dist) if dist is not None else math.nan)

        if arc.from_node_index is None or arc.to_node_index is None:
            return int(arc.index)

        u = int(arc.from_node_index)
        v = int(arc.to_node_index)

        # Ensure adjacency structures are long enough.
        while len(self.fadj_list) <= max(u, v):
            self.fadj_list.append([])
            self.badj_list.append([])
        while len(self.out_arcs) <= u:
            self.out_arcs.append([])

        self.fadj_list[u].append(v)
        self.badj_list[v].append(u)
        self.out_arcs[u].append(int(arc.index))
        self.node_pair_arc_index[(u, v)] = int(arc.index)
        return int(arc.index)

    def build_adjacency(self) -> None:
        """(Re)build adjacency lists from the current node/arc arrays.

        When reading from Julia-style tables, nodes and arcs are populated as
        **1-based** arrays (index 0 is a dummy element). This helper builds
        ``fadj_list``, ``badj_list``, ``out_arcs`` and ``node_pair_arc_index``
        to enable shortest-path calculations without depending on Julia's
        precomputed reduced network.
        """

        if not self.nodes or self.nodes[0].index != 0:
            raise ValueError("Graph.nodes must be 1-based with a dummy element at index 0")
        if not self.arcs or self.arcs[0].index != 0:
            raise ValueError("Graph.arcs must be 1-based with a dummy element at index 0")

        n = len(self.nodes) - 1
        m = len(self.arcs) - 1

        self.fadj_list = [[] for _ in range(n + 1)]
        self.badj_list = [[] for _ in range(n + 1)]
        self.out_arcs = [[] for _ in range(n + 1)]
        self.node_pair_arc_index = {}

        # Keep a 1-based arc distance array.
        self.arc_dists = [0.0] * (m + 1)
        for i in range(1, m + 1):
            a = self.arcs[i]
            dist = a.distance
            self.arc_dists[i] = float(dist) if dist is not None else math.nan

            if a.from_node_index is None or a.to_node_index is None:
                continue
            u = int(a.from_node_index)
            v = int(a.to_node_index)
            if not (1 <= u <= n and 1 <= v <= n):
                raise ValueError(f"Arc {i} references invalid nodes: from={u}, to={v}, n={n}")

            self.fadj_list[u].append(v)
            self.badj_list[v].append(u)
            self.out_arcs[u].append(i)
            self.node_pair_arc_index[(u, v)] = i


@dataclass
class NetTravel:
    is_reduced: bool = False
    mode_index: Optional[int] = None
    arc_times: List[float] = field(default_factory=list)

    # reduced-graph helpers
    arc_dists: List[float] = field(default_factory=list)
    fadj_list: List[List[int]] = field(default_factory=list)
    sp_times: Any = None
    sp_dists: Any = None
    sp_fadj_index: Any = None
    sp_node_pair_arc_index: Any = None
    sp_fadj_arc_list: List[List[int]] = field(default_factory=list)

    # full-graph helpers
    fnode_to_rnode_time: List[Dict[int, float]] = field(default_factory=list)
    fnode_from_rnode_time: List[Dict[int, float]] = field(default_factory=list)
    rarc_fnodes_times: List[List[float]] = field(default_factory=list)

    # common node helpers
    common_fnode_to_fnode_time: Any = None
    fnode_to_common_fnode_time: Any = None
    common_fnode_to_fnode_dist: Any = None
    fnode_to_common_fnode_dist: Any = None
    common_fnode_to_fnode_rnodes: Any = None
    fnode_to_common_fnode_rnodes: Any = None
    fnode_nearest_hospital_index: List[Optional[int]] = field(default_factory=list)


@dataclass
class Network:
    f_graph: Graph = field(default_factory=lambda: Graph(is_reduced=False))
    r_graph: Graph = field(default_factory=lambda: Graph(is_reduced=True))
    f_net_travels: List[NetTravel] = field(default_factory=list)
    r_net_travels: List[NetTravel] = field(default_factory=list)

    # mapping between full and reduced graphs
    rnode_fnode: List[int] = field(default_factory=list)
    fnode_rnode: List[Optional[int]] = field(default_factory=list)
    rarc_fnodes: List[List[int]] = field(default_factory=list)
    fnode_rarcs: List[List[int]] = field(default_factory=list)
    rarc_fnode_index: List[Dict[int, int]] = field(default_factory=list)
    fnode_to_rnodes: List[List[int]] = field(default_factory=list)
    fnode_from_rnodes: List[List[int]] = field(default_factory=list)
    fnode_to_rnode_next_fnode: List[Dict[int, int]] = field(default_factory=list)
    rarc_farcs: List[List[int]] = field(default_factory=list)

    fnode_to_rnode_dist: List[Dict[int, float]] = field(default_factory=list)
    fnode_from_rnode_dist: List[Dict[int, float]] = field(default_factory=list)

    common_fnodes: List[int] = field(default_factory=list)
    is_fnode_common: List[bool] = field(default_factory=list)
    fnode_common_fnode_index: List[Optional[int]] = field(default_factory=list)


@dataclass
class TravelMode:
    index: Optional[int] = None
    off_road_speed: Optional[float] = None
    f_net_travel: NetTravel = field(default_factory=NetTravel)
    r_net_travel: NetTravel = field(default_factory=lambda: NetTravel(is_reduced=True))


@dataclass
class Travel:
    num_modes: int = 0
    num_sets: int = 0
    modes: List[TravelMode] = field(default_factory=list)
    # mode_lookup[set_index][priority_index] -> travel mode index
    mode_lookup: List[List[int]] = field(default_factory=list)
    sets_start_times: List[float] = field(default_factory=list)
    sets_time_order: List[int] = field(default_factory=list)
    recent_sets_start_times_index: int = 0

    def current_set_index(self, t: float) -> Optional[int]:
        """Return the active travel-set index at time t."""
        if not self.sets_start_times:
            return None
        # assumes sets_start_times sorted
        idx = 0
        for i, start in enumerate(self.sets_start_times):
            if t >= start:
                idx = i
            else:
                break
        self.recent_sets_start_times_index = idx
        if idx < len(self.sets_time_order):
            return self.sets_time_order[idx]
        return None

    def mode_index_for_priority(self, priority: Priority, t: float) -> int:
        """Return the travel-mode index to use for a given priority at time ``t``.

        This mirrors Julia's ``getTravelMode!(travel, priority, time)`` selection:
        pick the active travel set at ``t`` and then use the per-priority lookup.
        """

        set_idx = self.current_set_index(t)
        if set_idx is None:
            raise ValueError("Travel.sets_start_times is empty; cannot select a travel set")
        if not (1 <= set_idx < len(self.mode_lookup)):
            raise ValueError(f"Invalid travel set index {set_idx}")

        pr_idx = int(priority.value)
        if pr_idx <= 0:
            raise ValueError(f"Invalid priority index {pr_idx} for {priority}")
        try:
            mode_idx = int(self.mode_lookup[set_idx][pr_idx])
        except Exception as e:  # IndexError or type errors
            raise ValueError(f"travel.mode_lookup is not populated for set={set_idx}, priority={priority}") from e

        if mode_idx <= 0 or mode_idx >= len(self.modes):
            raise ValueError(
                f"Invalid travel mode index {mode_idx} for set={set_idx}, priority={priority}; "
                f"expected 1..{len(self.modes)-1}"
            )
        return mode_idx

    def mode_for_priority(self, priority: Priority, t: float) -> TravelMode:
        """Return the :class:`~jemss.network.TravelMode` for a given priority and time."""

        return self.modes[self.mode_index_for_priority(priority, t)]