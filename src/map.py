from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import bisect
import math
import random

from .defs import Priority, RouteStatus
from .geo import Location, Node
from .misc import DistrRng
from .network import TravelMode

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import Simulation


@dataclass
class Map:
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_scale: float = 1.0
    y_scale: float = 1.0

    @property
    def x_range(self) -> Optional[float]:
        if self.x_min is None or self.x_max is None:
            return None
        return self.x_max - self.x_min

    @property
    def y_range(self) -> Optional[float]:
        if self.y_min is None or self.y_max is None:
            return None
        return self.y_max - self.y_min

    def distance(self, a: Location, b: Location) -> float:
        """Off-road distance between two locations using this map's scaling."""
        return a.distance_to(b, x_scale=self.x_scale, y_scale=self.y_scale)


def square_dist(map_: Map, loc1: Location, loc2: Location) -> float:
    """Squared Euclidean distance between two locations in map distance units."""
    if loc1.x is None or loc1.y is None or loc2.x is None or loc2.y is None:
        raise ValueError("Cannot compute distance with unset coordinates")
    dx = (loc1.x - loc2.x) * map_.x_scale
    dy = (loc1.y - loc2.y) * map_.y_scale
    return dx * dx + dy * dy


def norm_dist(map_: Map, loc1: Location, loc2: Location) -> float:
    """Euclidean distance between two locations in map distance units."""
    return math.sqrt(square_dist(map_, loc1, loc2))


def off_road_travel_time(travel_mode: TravelMode, *args: Any) -> float:
    """Compute off-road travel time.

    This mirrors the two reference methods:

    * ``offRoadTravelTime(travelMode, map, loc1, loc2)``
    * ``offRoadTravelTime(travelMode, dist)``
    """
    if travel_mode.off_road_speed is None or travel_mode.off_road_speed <= 0:
        raise ValueError("travel_mode.off_road_speed must be a positive number")

    if len(args) == 1:
        dist = float(args[0])
        return dist / travel_mode.off_road_speed
    if len(args) == 3:
        map_, loc1, loc2 = args
        if not isinstance(map_, Map):
            raise TypeError("First argument must be a Map")
        return norm_dist(map_, loc1, loc2) / travel_mode.off_road_speed
    raise TypeError("off_road_travel_time expects (dist) or (map, loc1, loc2)")


def linear_interp_location(
    start_loc: Location,
    end_loc: Location,
    start_time: float,
    end_time: float,
    current_time: float,
) -> Location:
    """Linear interpolation between two locations in time.

    Mirrors reference ``linearInterpLocation``.
    """
    if not (start_time <= current_time <= end_time and start_time < end_time):
        raise ValueError("Require start_time <= current_time <= end_time and start_time < end_time")
    if start_loc.x is None or start_loc.y is None or end_loc.x is None or end_loc.y is None:
        raise ValueError("Cannot interpolate with unset coordinates")
    p = (current_time - start_time) / (end_time - start_time)
    return Location(
        x=(1 - p) * start_loc.x + p * end_loc.x,
        y=(1 - p) * start_loc.y + p * end_loc.y,
    )


def rand_location(map_: Map, *, trim: float = 0.0, rng: Optional[random.Random] = None) -> Location:
    """Uniform random location within map bounds.

    ``trim`` is the fractional trimming of the border, matching reference's
    ``randLocation(map; trim=...)``.
    """
    if not (0.0 <= trim <= 1.0):
        raise ValueError("trim must be between 0 and 1")
    if map_.x_min is None or map_.x_max is None or map_.y_min is None or map_.y_max is None:
        raise ValueError("Map bounds must be set")

    r = rng if rng is not None else random
    # two independent uniforms scaled into [trim/2, 1-trim/2]
    u1 = r.random() * (1 - trim) + trim / 2
    u2 = r.random() * (1 - trim) + trim / 2
    xr = map_.x_range
    yr = map_.y_range
    if xr is None or yr is None:
        raise ValueError("Map range must be set")
    return Location(
        x=map_.x_min + xr * u1,
        y=map_.y_min + yr * u2,
    )


def trimmed_map(map_: Map, trim: float = 0.0) -> Map:
    """Return a new map with borders trimmed by a fraction."""
    if not (0.0 <= trim <= 1.0):
        raise ValueError("trim must be between 0 and 1")
    if map_.x_min is None or map_.x_max is None or map_.y_min is None or map_.y_max is None:
        raise ValueError("Map bounds must be set")
    xr = map_.x_range
    yr = map_.y_range
    if xr is None or yr is None:
        raise ValueError("Map range must be set")

    dx = xr * trim / 2
    dy = yr * trim / 2
    return Map(
        x_min=map_.x_min + dx,
        x_max=map_.x_max - dx,
        y_min=map_.y_min + dy,
        y_max=map_.y_max - dy,
        x_scale=map_.x_scale,
        y_scale=map_.y_scale,
    )


@dataclass
class GridSearchRect:
    x_dist: List[float] = field(default_factory=lambda: [0.0, 0.0])
    y_dist: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ix_search: List[int] = field(default_factory=lambda: [0, 0])
    iy_search: List[int] = field(default_factory=lambda: [0, 0])
    ix_searched: List[int] = field(default_factory=lambda: [0, 0])
    iy_searched: List[int] = field(default_factory=lambda: [0, 0])


@dataclass
class GridRect:
    node_indices: List[int] = field(default_factory=list)


@dataclass
class Grid:
    nx: int = 0
    ny: int = 0
    x_range: float = 0.0
    y_range: float = 0.0
    x_range_dist: float = 0.0
    y_range_dist: float = 0.0
    rects: List[List[GridRect]] = field(default_factory=list)
    search_rect: GridSearchRect = field(default_factory=GridSearchRect)

    @classmethod
    def from_map(cls, map_: Map, nx: int, ny: int) -> "Grid":
        grid = cls(nx=nx, ny=ny)
        # Use 1-based indexing for rects (index 0 is unused) to mirror reference.
        grid.rects = [[GridRect() for _ in range(ny + 1)] for _ in range(nx + 1)]
        grid.x_range = (map_.x_range or 0.0) / nx if nx > 0 else 0.0
        grid.y_range = (map_.y_range or 0.0) / ny if ny > 0 else 0.0
        grid.x_range_dist = grid.x_range * map_.x_scale
        grid.y_range_dist = grid.y_range * map_.y_scale
        return grid


def location_to_grid_index(map_: Map, grid: Grid, location: Location) -> Tuple[int, int]:
    """Return (ix, iy) grid indices for a given location.

    Mirrors reference ``locationToGridIndex(map, grid, location)``.

    Notes
    -----
    * Indices are 1-based.
    * When ``location.x == map.x_min`` (or y), reference forces the index to 1.
    """

    if map_.x_min is None or map_.y_min is None or map_.x_range is None or map_.y_range is None:
        raise ValueError("Map bounds must be initialised")
    if location.x is None or location.y is None:
        raise ValueError("Location coordinates must be set")
    if grid.nx <= 0 or grid.ny <= 0:
        raise ValueError("Grid must have positive nx and ny")

    # reference uses ceil on a 1..nx scale.
    ix = int(math.ceil((location.x - map_.x_min) / map_.x_range * grid.nx))
    iy = int(math.ceil((location.y - map_.y_min) / map_.y_range * grid.ny))

    if location.x == map_.x_min:
        ix = 1
    if location.y == map_.y_min:
        iy = 1

    if not (1 <= ix <= grid.nx and 1 <= iy <= grid.ny):
        raise ValueError(
            f"Location outside grid: ix={ix}, iy={iy}, nx={grid.nx}, ny={grid.ny}, x={location.x}, y={location.y}"
        )
    return ix, iy


def grid_place_nodes(
    map_: Map,
    grid: Grid,
    nodes: Sequence[Node],
    *,
    off_road_access_required: bool = True,
) -> None:
    """Place node indices into the grid rectangles.

    Mirrors reference ``gridPlaceNodes!(map, grid, nodes; offRoadAccessRequired=true)``.

    ``nodes`` is expected to be 1-based (index 0 unused) but this function also
    tolerates 0-based input by skipping nodes where ``node.index`` is falsy.
    """

    # Clear any existing placement.
    for i in range(1, grid.nx + 1):
        for j in range(1, grid.ny + 1):
            grid.rects[i][j].node_indices.clear()

    for node in nodes:
        if not node.index:
            continue
        if off_road_access_required and not node.off_road_access:
            continue
        ix, iy = location_to_grid_index(map_, grid, node.location)
        grid.rects[ix][iy].node_indices.append(int(node.index))


def find_nearest_node(
    map_: Map,
    grid: Grid,
    nodes: Sequence[Node],
    location: Location,
) -> Tuple[int, float]:
    """Find the nearest node to ``location`` using the grid.

    Mirrors reference ``findNearestNode(map, grid, nodes, location)``.

    Returns
    -------
    (node_index, distance)
        ``node_index`` is 1-based.
    """

    ix, iy = location_to_grid_index(map_, grid, location)

    gsr = grid.search_rect

    # Initialise search rectangle (1-based semantics).
    gsr.ix_searched[0] = ix
    gsr.ix_searched[1] = ix
    gsr.iy_searched[0] = iy
    gsr.iy_searched[1] = iy
    gsr.ix_search[0] = ix
    gsr.ix_search[1] = ix
    gsr.iy_search[0] = iy
    gsr.iy_search[1] = iy

    if map_.x_min is None or map_.y_min is None:
        raise ValueError("Map bounds must be set")
    if location.x is None or location.y is None:
        raise ValueError("Location coordinates must be set")

    # Distances from location to the current search rectangle borders.
    gsr.x_dist[0] = (location.x - (map_.x_min + grid.x_range * (ix - 1))) * map_.x_scale
    gsr.x_dist[1] = grid.x_range_dist - gsr.x_dist[0]
    gsr.y_dist[0] = (location.y - (map_.y_min + grid.y_range * (iy - 1))) * map_.y_scale
    gsr.y_dist[1] = grid.y_range_dist - gsr.y_dist[0]

    nearest_node_index: int = -1
    best_dist: float = math.inf

    found = False
    skip_search = False
    while not found:
        if not skip_search:
            for i in range(gsr.ix_search[0], gsr.ix_search[1] + 1):
                for j in range(gsr.iy_search[0], gsr.iy_search[1] + 1):
                    nearest_in_cell: int = -1
                    best_sqr = math.inf
                    for node_index in grid.rects[i][j].node_indices:
                        sqr = square_dist(map_, nodes[node_index].location, location)
                        if sqr < best_sqr:
                            best_sqr = sqr
                            nearest_in_cell = node_index
                    dist = math.sqrt(best_sqr)
                    if dist < best_dist:
                        best_dist = dist
                        nearest_node_index = nearest_in_cell

        nearest_border_dist = min(gsr.x_dist[0], gsr.x_dist[1], gsr.y_dist[0], gsr.y_dist[1])
        if best_dist <= nearest_border_dist:
            found = True
        else:
            skip_search = False
            # Extend the nearest border (tie-break order matches reference).
            if gsr.x_dist[0] == nearest_border_dist:
                if gsr.ix_searched[0] > 1:
                    gsr.ix_searched[0] -= 1
                    gsr.x_dist[0] += grid.x_range_dist
                    gsr.ix_search[0] = gsr.ix_searched[0]
                    gsr.ix_search[1] = gsr.ix_searched[0]
                    gsr.iy_search[0] = gsr.iy_searched[0]
                    gsr.iy_search[1] = gsr.iy_searched[1]
                else:
                    gsr.x_dist[0] = math.inf
                    skip_search = True
            elif gsr.x_dist[1] == nearest_border_dist:
                if gsr.ix_searched[1] < grid.nx:
                    gsr.ix_searched[1] += 1
                    gsr.x_dist[1] += grid.x_range_dist
                    gsr.ix_search[0] = gsr.ix_searched[1]
                    gsr.ix_search[1] = gsr.ix_searched[1]
                    gsr.iy_search[0] = gsr.iy_searched[0]
                    gsr.iy_search[1] = gsr.iy_searched[1]
                else:
                    gsr.x_dist[1] = math.inf
                    skip_search = True
            elif gsr.y_dist[0] == nearest_border_dist:
                if gsr.iy_searched[0] > 1:
                    gsr.iy_searched[0] -= 1
                    gsr.y_dist[0] += grid.y_range_dist
                    gsr.ix_search[0] = gsr.ix_searched[0]
                    gsr.ix_search[1] = gsr.ix_searched[1]
                    gsr.iy_search[0] = gsr.iy_searched[0]
                    gsr.iy_search[1] = gsr.iy_searched[0]
                else:
                    gsr.y_dist[0] = math.inf
                    skip_search = True
            elif gsr.y_dist[1] == nearest_border_dist:
                if gsr.iy_searched[1] < grid.ny:
                    gsr.iy_searched[1] += 1
                    gsr.y_dist[1] += grid.y_range_dist
                    gsr.ix_search[0] = gsr.ix_searched[0]
                    gsr.ix_search[1] = gsr.ix_searched[1]
                    gsr.iy_search[0] = gsr.iy_searched[1]
                    gsr.iy_search[1] = gsr.iy_searched[1]
                else:
                    gsr.y_dist[1] = math.inf
                    skip_search = True

    if nearest_node_index == -1 or not math.isfinite(best_dist):
        raise RuntimeError("Failed to locate nearest node (grid has no nodes?)")
    return nearest_node_index, best_dist


def find_nearest_node_linear(
    map_: Map,
    nodes: Sequence[Node],
    location: Location,
    *,
    off_road_access_required: bool = True,
) -> Tuple[int, float]:
    """Brute-force nearest-node search.

    This mirrors reference's slower ``findNearestNode(map, nodes, location)`` and is
    useful for validating the grid implementation.
    """

    best_sqr = math.inf
    chosen = -1
    for node in nodes:
        if not node.index:
            continue
        if off_road_access_required and not node.off_road_access:
            continue
        sqr = square_dist(map_, node.location, location)
        if sqr < best_sqr:
            best_sqr = sqr
            chosen = int(node.index)
    if chosen == -1:
        raise RuntimeError("No nodes available for nearest-node search")
    return chosen, math.sqrt(best_sqr)


@dataclass
class Raster:
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[List[float]] = field(default_factory=list)

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def dx(self) -> float:
        if self.nx <= 1:
            return 0.0
        return (max(self.x) - min(self.x)) / (self.nx - 1)

    @property
    def dy(self) -> float:
        if self.ny <= 1:
            return 0.0
        return (max(self.y) - min(self.y)) / (self.ny - 1)


@dataclass
class RasterSampler:
    raster: Raster
    cell_distr_rng: DistrRng
    cell_loc_rng: Any  # random state for within-cell sampling


@dataclass
class Route:
    priority: Optional[Priority] = None
    travel_mode_index: Optional[int] = None

    start_loc: Location = field(default_factory=Location)
    start_time: Optional[float] = None
    end_loc: Location = field(default_factory=Location)
    end_time: Optional[float] = None

    start_fnode: Optional[int] = None
    start_fnode_time: Optional[float] = None
    start_fnode_dist: Optional[float] = None
    start_rnode: Optional[int] = None
    start_rnode_time: Optional[float] = None
    end_rnode: Optional[int] = None
    end_rnode_time: Optional[float] = None
    end_fnode: Optional[int] = None
    end_fnode_time: Optional[float] = None

    first_rarc: Optional[int] = None

    status: RouteStatus = RouteStatus.NULL
    recent_update_time: Optional[float] = None

    recent_rarc: Optional[int] = None
    recent_rarc_start_time: Optional[float] = None
    recent_rarc_end_time: Optional[float] = None

    recent_rarc_recent_fnode: Optional[int] = None
    recent_fnode: Optional[int] = None
    recent_fnode_time: Optional[float] = None

    recent_rarc_next_fnode: Optional[int] = None
    next_fnode: Optional[int] = None
    next_fnode_time: Optional[float] = None
    next_fnode_dist: Optional[float] = None

    # ------------------------------------------------------------------
    # Step 6: full-graph path representation
    # ------------------------------------------------------------------
    # These fields are **not** part of the original reference `Route` struct.
    # They let the Python port compute route progression without relying on
    # the reduced graph / precomputed rNetTravels.

    # On-road path from start_fnode -> end_fnode as full-graph arc indices.
    path_arcs: List[int] = field(default_factory=list)
    # Nodes along the on-road path (len = len(path_arcs)+1 when path exists).
    path_nodes: List[int] = field(default_factory=list)
    # Absolute times when each node in `path_nodes` is reached.
    path_node_times: List[float] = field(default_factory=list)
    # Cumulative on-road distance from start_fnode to each node in `path_nodes`.
    path_node_dists: List[float] = field(default_factory=list)

    path_total_time: float = 0.0
    path_total_dist: float = 0.0

    # 0-based index of `recent_fnode` within `path_nodes` when ON_PATH.
    path_segment_index: Optional[int] = None

    def progress_fraction(self, t: float) -> float:
        """Return fraction of route completed at time t (0..1)."""
        if self.start_time is None or self.end_time is None or self.end_time <= self.start_time:
            return 0.0
        return max(0.0, min(1.0, (t - self.start_time) / (self.end_time - self.start_time)))

    # ------------------------------------------------------------------
    # Step 6: route mechanics (full graph)
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_loc(loc: Location) -> Location:
        return Location(x=loc.x, y=loc.y)

    def init_at_location(self, *, start_loc: Location, start_fnode: int, start_fnode_dist: float) -> None:
        """Initialise an "empty" route located at ``start_loc``.

        This mirrors the intent of reference's ``initRoute!``:

        * the route is set to an inert state at a location (typically a station)
        * ``start_time`` / ``start_fnode_time`` are set to +∞
        * ``end_time`` is left unset so the route is treated as *already ended*
          by :meth:`next_node`, causing it to return the nearest node and the
          off-road time to reach it.
        """

        if start_loc.x is None or start_loc.y is None:
            raise ValueError("start_loc coordinates must be set")
        if start_fnode <= 0:
            raise ValueError("start_fnode must be a positive 1-based node index")
        if start_fnode_dist < 0:
            raise ValueError("start_fnode_dist must be >= 0")

        self.priority = None
        self.travel_mode_index = None

        self.start_loc = self._copy_loc(start_loc)
        self.start_time = math.inf
        self.end_loc = self._copy_loc(start_loc)
        # Leave unset (None) to make next_node(...) treat the route as finished.
        self.end_time = None

        self.start_fnode = int(start_fnode)
        self.start_fnode_dist = float(start_fnode_dist)
        self.start_fnode_time = math.inf

        self.end_fnode = int(start_fnode)
        self.end_fnode_time = math.inf

        # Reduced-graph fields are not used in the Python port yet.
        self.start_rnode = None
        self.start_rnode_time = None
        self.end_rnode = None
        self.end_rnode_time = None
        self.first_rarc = None

        self.path_arcs = []
        self.path_nodes = [int(start_fnode)]
        self.path_node_times = [math.inf]
        self.path_node_dists = [0.0]
        self.path_total_time = 0.0
        self.path_total_dist = 0.0
        self.path_segment_index = None

        self.recent_update_time = None
        self.next_fnode_dist = None

        self._set_state_before_start_fnode()

    # ---- state setters -------------------------------------------------

    def _set_state_before_start_fnode(self) -> None:
        """Set the temporally-varying fields to represent state before start_fnode."""

        self.status = RouteStatus.BEFORE_START_NODE

        self.recent_rarc = None
        self.recent_rarc_start_time = None
        self.recent_rarc_end_time = None

        self.recent_rarc_recent_fnode = None
        self.recent_fnode = None
        self.recent_fnode_time = None

        self.recent_rarc_next_fnode = None
        self.next_fnode = self.start_fnode
        self.next_fnode_time = self.start_fnode_time
        self.path_segment_index = None

    def _set_state_after_end_fnode(self) -> None:
        """Set temporally-varying fields to represent state after leaving end_fnode."""

        self.status = RouteStatus.AFTER_END_NODE

        self.recent_rarc = None
        self.recent_rarc_start_time = None
        self.recent_rarc_end_time = None

        self.recent_rarc_recent_fnode = None
        self.recent_fnode = self.end_fnode
        self.recent_fnode_time = self.end_fnode_time

        self.recent_rarc_next_fnode = None
        self.next_fnode = None
        self.next_fnode_time = None
        self.path_segment_index = None

    def _set_state_on_path(self, seg_index: int) -> None:
        """Set route state for being on-road between path_nodes[seg_index] -> [seg_index+1]."""

        self.status = RouteStatus.ON_PATH

        self.recent_rarc = None
        self.recent_rarc_start_time = None
        self.recent_rarc_end_time = None

        self.recent_rarc_recent_fnode = None
        self.recent_fnode = self.path_nodes[seg_index]
        self.recent_fnode_time = self.path_node_times[seg_index]

        self.recent_rarc_next_fnode = None
        self.next_fnode = self.path_nodes[seg_index + 1]
        self.next_fnode_time = self.path_node_times[seg_index + 1]
        self.path_segment_index = seg_index

    # ---- progression / queries ----------------------------------------

    def update_to_time(self, sim: "Simulation", t: float) -> None:
        """Update cached `recent_*`/`next_*` fields for time ``t``.

        This is a simplified analogue of reference's ``updateRouteToTime!`` that
        uses the stored full-graph `path_*` arrays rather than reduced-network
        rArc metadata.
        """

        if self.status == RouteStatus.NULL:
            raise ValueError("Route has not been initialised")

        if self.recent_update_time is not None and t < self.recent_update_time:
            raise ValueError("Route cannot be updated backwards in time")
        if self.recent_update_time == t:
            return

        self.recent_update_time = t
        self.next_fnode_dist = None  # invalidate cached value

        if self.start_fnode_time is None or self.end_fnode_time is None:
            self._set_state_before_start_fnode()
            return

        # reference uses `time <= startFNodeTime` to keep BEFORE_START_NODE on the boundary.
        if t <= self.start_fnode_time:
            self._set_state_before_start_fnode()
            return
        if t >= self.end_fnode_time:
            self._set_state_after_end_fnode()
            return

        # On-road segment.
        if len(self.path_node_times) < 2 or len(self.path_nodes) < 2:
            # No on-road path (start==end), but we shouldn't get here due to the
            # boundary checks. Treat as AFTER_END_NODE defensively.
            self._set_state_after_end_fnode()
            return

        # Find seg_index so that path_node_times[seg] <= t < path_node_times[seg+1].
        seg = bisect.bisect_right(self.path_node_times, t) - 1
        seg = max(0, min(seg, len(self.path_node_times) - 2))
        self._set_state_on_path(seg)

    def current_location(self, sim: "Simulation", t: float) -> Location:
        """Return the route's current location at time ``t``.

        Mirrors reference's ``getRouteCurrentLocation!``.
        """

        self.update_to_time(sim, t)
        f_nodes = sim.net.f_graph.nodes

        if self.start_time is None:
            raise ValueError("Route.start_time is not set")

        if t <= self.start_time:
            return self.start_loc
        if self.end_time is not None and t >= self.end_time:
            return self.end_loc

        if self.start_fnode is None or self.start_fnode_time is None:
            return self.start_loc
        if t <= self.start_fnode_time:
            return linear_interp_location(
                self.start_loc,
                f_nodes[self.start_fnode].location,
                self.start_time,
                self.start_fnode_time,
                t,
            )

        if self.end_fnode is None or self.end_fnode_time is None:
            return self.end_loc
        if t >= self.end_fnode_time:
            if self.end_time is None or self.end_time <= self.end_fnode_time:
                # Unplanned/degenerate route: treat end_fnode as end.
                return f_nodes[self.end_fnode].location
            return linear_interp_location(
                f_nodes[self.end_fnode].location,
                self.end_loc,
                self.end_fnode_time,
                self.end_time,
                t,
            )

        # On-road.
        if (
            self.recent_fnode is None
            or self.next_fnode is None
            or self.recent_fnode_time is None
            or self.next_fnode_time is None
        ):
            raise RuntimeError("Route is on-road but recent/next fields are unset")

        return linear_interp_location(
            f_nodes[self.recent_fnode].location,
            f_nodes[self.next_fnode].location,
            self.recent_fnode_time,
            self.next_fnode_time,
            t,
        )

    def next_node(self, sim: "Simulation", travel_mode_index: int, t: float) -> Tuple[int, float]:
        """Return (next_fnode, time_remaining_to_reach_it) at time ``t``.

        Mirrors reference's ``getRouteNextNode!``.
        """

        self.update_to_time(sim, t)

        # "Already past" the route: for unplanned routes, end_time is None.
        if self.end_time is None or self.end_time <= t:
            if self.end_fnode is None:
                raise ValueError("Route.end_fnode is not set")
            if self.end_loc.x is None or self.end_loc.y is None:
                raise ValueError("Route.end_loc coordinates are not set")

            travel_mode = sim.travel.modes[travel_mode_index]
            node_loc = sim.net.f_graph.nodes[self.end_fnode].location
            tt = off_road_travel_time(travel_mode, sim.map, self.end_loc, node_loc)
            return int(self.end_fnode), float(tt)

        if self.next_fnode is not None:
            if self.next_fnode_time is None:
                raise RuntimeError("Route.next_fnode_time is unset")
            return int(self.next_fnode), max(0.0, float(self.next_fnode_time - t))

        # Between end_fnode and end_loc: "return" to recent_fnode (end_fnode).
        if self.recent_fnode is None or self.recent_fnode_time is None:
            raise RuntimeError("Route.recent_fnode/recent_fnode_time is unset")
        travel_time = max(0.0, float(t - self.recent_fnode_time))

        # Compatibility: if travel mode changes while off-road after end node,
        # scale travel time by off-road speeds (mirrors reference's check).
        if self.travel_mode_index is not None and self.travel_mode_index != travel_mode_index:
            old_mode = sim.travel.modes[self.travel_mode_index]
            new_mode = sim.travel.modes[travel_mode_index]
            if old_mode.off_road_speed and new_mode.off_road_speed:
                travel_time *= float(old_mode.off_road_speed) / float(new_mode.off_road_speed)

        return int(self.recent_fnode), travel_time

    def next_node_distance(self, sim: "Simulation", t: float) -> float:
        """Distance associated with :meth:`next_node` at time ``t``.

        Mirrors reference's ``getRouteNextNodeDist!`` semantics:

        * BEFORE_START_NODE: remaining distance to reach start_fnode
        * ON_PATH: remaining distance to reach next_fnode along the current arc
        * AFTER_END_NODE: distance required to return to end_fnode from current off-road position
        """

        self.update_to_time(sim, t)
        if self.next_fnode_dist is not None:
            return float(self.next_fnode_dist)

        f_graph = sim.net.f_graph

        dist: float
        if self.status == RouteStatus.AFTER_END_NODE:
            if self.end_fnode is None or self.end_fnode_time is None:
                raise ValueError("Route.end_fnode/end_fnode_time is not set")
            dist = norm_dist(sim.map, self.end_loc, f_graph.nodes[self.end_fnode].location)
            if self.end_time is not None and t < self.end_time and self.end_time > self.end_fnode_time:
                dist *= (t - self.end_fnode_time) / (self.end_time - self.end_fnode_time)

        elif self.status == RouteStatus.BEFORE_START_NODE:
            if self.start_fnode_dist is None:
                raise ValueError("Route.start_fnode_dist is not set")
            if self.start_time is None or self.start_fnode_time is None:
                raise ValueError("Route.start_time/start_fnode_time is not set")
            dist = float(self.start_fnode_dist)
            if t > self.start_time and self.start_fnode_time > self.start_time:
                dist *= (self.start_fnode_time - t) / (self.start_fnode_time - self.start_time)

        else:
            # ON_PATH
            if (
                self.recent_fnode is None
                or self.next_fnode is None
                or self.recent_fnode_time is None
                or self.next_fnode_time is None
            ):
                raise RuntimeError("Route ON_PATH but recent/next fields are unset")
            arc_idx = f_graph.node_pair_arc_index.get((int(self.recent_fnode), int(self.next_fnode)))
            if not arc_idx:
                raise KeyError(f"No arc for node pair ({self.recent_fnode},{self.next_fnode})")
            arc = f_graph.arcs[arc_idx]
            arc_dist = float(arc.distance) if arc.distance is not None else math.nan
            if self.next_fnode_time > self.recent_fnode_time:
                dist = arc_dist * (self.next_fnode_time - t) / (self.next_fnode_time - self.recent_fnode_time)
            else:
                dist = arc_dist

        self.next_fnode_dist = dist
        return float(dist)

    def distance_travelled(self, sim: "Simulation", t: float) -> float:
        """Distance travelled along the route from route.start_time up to time ``t``.

        Mirrors reference's ``calcRouteDistance!`` using the stored full-graph path.
        """

        if self.status == RouteStatus.NULL:
            raise ValueError("Route has not been initialised")

        self.update_to_time(sim, t)
        next_dist = self.next_node_distance(sim, t)

        start_off = float(self.start_fnode_dist or 0.0)

        if self.status == RouteStatus.BEFORE_START_NODE:
            if self.start_fnode_dist is None:
                return 0.0
            return max(0.0, float(self.start_fnode_dist) - next_dist)

        if self.status == RouteStatus.AFTER_END_NODE:
            return start_off + float(self.path_total_dist) + next_dist

        # ON_PATH
        seg = self.path_segment_index
        if seg is None:
            # compute from times defensively
            seg = bisect.bisect_right(self.path_node_times, t) - 1
            seg = max(0, min(seg, len(self.path_node_times) - 2))

        dist_to_recent = float(self.path_node_dists[seg]) if seg < len(self.path_node_dists) else 0.0

        # distance travelled on current arc
        f_graph = sim.net.f_graph
        if self.recent_fnode is None or self.next_fnode is None:
            raise RuntimeError("Route ON_PATH but recent/next nodes are unset")
        arc_idx = f_graph.node_pair_arc_index.get((int(self.recent_fnode), int(self.next_fnode)))
        if not arc_idx:
            raise KeyError(f"No arc for node pair ({self.recent_fnode},{self.next_fnode})")
        arc = f_graph.arcs[arc_idx]
        arc_dist = float(arc.distance) if arc.distance is not None else math.nan
        dist_on_arc = arc_dist - next_dist

        return start_off + dist_to_recent + dist_on_arc

    # ---- planning ------------------------------------------------------

    def plan_to_fnode(
        self,
        sim: "Simulation",
        *,
        priority: Priority,
        start_time: float,
        end_loc: Location,
        end_fnode: int,
    ) -> None:
        """Re-plan this route to travel from the current position to ``end_loc``.

        This is the Python analogue of reference's ``changeRoute!`` but using the
        full-graph runtime shortest-path backend.
        """

        # Determine travel mode.
        travel_mode_index = sim.travel.mode_index_for_priority(priority, start_time)
        travel_mode = sim.travel.modes[travel_mode_index]

        # Route start location is the current location of the existing route.
        start_loc = self.current_location(sim, start_time)

        # First node on the new route (may be the next node on the existing route).
        start_fnode, time_to_start_fnode = self.next_node(sim, travel_mode_index, start_time)
        start_fnode_time = start_time + float(time_to_start_fnode)
        start_fnode_dist = self.next_node_distance(sim, start_time)

        # On-road shortest path.
        path_arcs = sim.shortest_path_arcs(start_fnode, end_fnode, travel_mode_index)
        arc_times = sim.arc_travel_times[travel_mode_index] if sim.arc_travel_times is not None else None
        if arc_times is None:
            raise ValueError("Simulation.arc_travel_times is not available")

        path_time = 0.0
        path_dist = 0.0

        path_nodes: List[int] = [int(start_fnode)]
        path_node_times: List[float] = [float(start_fnode_time)]
        path_node_dists: List[float] = [0.0]

        f_graph = sim.net.f_graph
        for a in path_arcs:
            tt = float(arc_times[a])
            path_time += tt
            arc = f_graph.arcs[a]
            ad = float(arc.distance) if arc.distance is not None else math.nan
            path_dist += ad
            # append node + cumulatives
            if arc.to_node_index is None:
                raise ValueError(f"Arc {a} is missing to_node_index")
            path_nodes.append(int(arc.to_node_index))
            path_node_times.append(float(start_fnode_time + path_time))
            path_node_dists.append(float(path_dist))

        end_fnode_time = float(start_fnode_time + path_time)

        # Off-road from end_fnode to end_loc.
        end_node_loc = f_graph.nodes[end_fnode].location
        end_off_time = off_road_travel_time(travel_mode, sim.map, end_node_loc, end_loc)
        end_time = end_fnode_time + float(end_off_time)

        # Mutate route.
        self.priority = priority
        self.travel_mode_index = int(travel_mode_index)

        self.start_loc = self._copy_loc(start_loc)
        self.start_time = float(start_time)
        self.end_loc = self._copy_loc(end_loc)
        self.end_time = float(end_time)

        self.start_fnode = int(start_fnode)
        self.start_fnode_time = float(start_fnode_time)
        self.start_fnode_dist = float(start_fnode_dist)
        self.end_fnode = int(end_fnode)
        self.end_fnode_time = float(end_fnode_time)

        # Reduced-graph fields remain unset in this port.
        self.start_rnode = None
        self.start_rnode_time = None
        self.end_rnode = None
        self.end_rnode_time = None
        self.first_rarc = None

        self.path_arcs = list(path_arcs)
        self.path_nodes = path_nodes
        self.path_node_times = path_node_times
        self.path_node_dists = path_node_dists
        self.path_total_time = float(path_time)
        self.path_total_dist = float(path_dist)
        self.path_segment_index = None

        # Reset dynamic state to match the new route start.
        self.recent_update_time = None
        self.next_fnode_dist = None
        self._set_state_before_start_fnode()

    # ---- helper: travel time preview ----------------------------------

    def travel_time_to_location(
        self,
        sim: "Simulation",
        *,
        travel_mode_index: int,
        t: float,
        end_loc: Location,
        end_fnode: int,
        end_fnode_dist: Optional[float] = None,
    ) -> float:
        """Compute travel time from the route's current position to a destination.

        This mirrors reference's ``shortestRouteTravelTime!`` in the common case
        where we start from an existing route.
        """

        start_fnode, time_to_start_fnode = self.next_node(sim, travel_mode_index, t)

        on_road = sim.shortest_path_time(start_fnode, end_fnode, travel_mode_index)
        travel_mode = sim.travel.modes[travel_mode_index]

        if end_fnode_dist is None:
            end_node_loc = sim.net.f_graph.nodes[end_fnode].location
            end_off = off_road_travel_time(travel_mode, sim.map, end_loc, end_node_loc)
        else:
            end_off = off_road_travel_time(travel_mode, float(end_fnode_dist))

        return float(time_to_start_fnode) + float(on_road) + float(end_off)
