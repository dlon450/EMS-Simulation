from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from geo import Node, Arc


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

    def add_node(self, node: Node) -> int:
        """Append a node and return its index."""
        node.index = len(self.nodes)
        self.nodes.append(node)
        if len(self.fadj_list) <= node.index:
            self.fadj_list.append([])
            self.badj_list.append([])
        return node.index

    def add_arc(self, arc: Arc) -> int:
        """Append an arc and update adjacency lists."""
        arc.index = len(self.arcs)
        self.arcs.append(arc)
        self.arc_dists.append(arc.distance or 0.0)
        if arc.from_node_index is not None and arc.to_node_index is not None:
            while len(self.fadj_list) <= arc.from_node_index:
                self.fadj_list.append([])
                self.badj_list.append([])
            while len(self.badj_list) <= arc.to_node_index:
                self.fadj_list.append([])
                self.badj_list.append([])
            self.fadj_list[arc.from_node_index].append(arc.to_node_index)
            self.badj_list[arc.to_node_index].append(arc.from_node_index)
            self.node_pair_arc_index[(arc.from_node_index, arc.to_node_index)] = arc.index
        return arc.index


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