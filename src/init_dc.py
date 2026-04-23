from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .defs import Priority, PRIORITIES
from .geo import Point
from .map import Raster, off_road_travel_time, find_nearest_node
from .network import TravelMode

from .read_sim_files import (
    read_demand_file
)

from .simulator import Simulation
from .demand import Demand, DemandCoverage, DemandMode, PointsCoverageMode

from .travel import get_travel_mode





def update_demand_to_time(demand: Demand, start_time: float) -> None:
    sets_start_times = demand.sets_start_times
    if not sets_start_times:
        return

    i = demand.recent_sets_start_times_index
    n = len(sets_start_times)

    if not (0 <= i < n):
        raise AssertionError("recent_sets_start_times_index out of bounds")

    if sets_start_times[i] > start_time:
        raise AssertionError("sets_start_times[i] <= start_time violated (went back in time?)")

    if i == n - 1 or start_time < sets_start_times[i + 1]:
        return

    while i < n - 1 and sets_start_times[i + 1] <= start_time:
        i += 1

    demand.recent_sets_start_times_index = i


def get_demand_mode(demand: Demand, priority: Priority, start_time: float) -> DemandMode:
    if start_time is None:
        raise AssertionError("start_time cannot be None")
    if priority is None:
        raise AssertionError("priority cannot be None")

    update_demand_to_time(demand, start_time)
    idx_in_time_order = demand.recent_sets_start_times_index
    if idx_in_time_order < 0 or idx_in_time_order >= len(demand.sets_time_order):
        raise AssertionError("recent_sets_start_times_index invalid for sets_time_order")

    demand_set_index = demand.sets_time_order[idx_in_time_order] - 1

    pr_idx = int(priority.value)  # adapt if needed
    if 1 <= pr_idx <= len(PRIORITIES):
        pr_col = pr_idx - 1
    else:
        pr_col = pr_idx

    demand_mode_index = demand.mode_lookup[demand_set_index][pr_col]
    return demand.modes[demand_mode_index - 1]


def create_demand_points_from_rasters(
    demand: Demand, *, num_cell_rows: int = 1, num_cell_cols: int = 1
) -> Tuple[List[Point], List[List[float]]]:
    if demand.num_rasters < 1 and not demand.rasters:
        raise AssertionError("demand must have at least 1 raster")
    if num_cell_rows < 1 or num_cell_cols < 1:
        raise AssertionError("num_cell_rows/num_cell_cols must be >= 1")

    rasters = demand.rasters
    num_rasters = len(rasters)
    r0 = rasters[0]
    nx, ny = r0.nx, r0.ny


    for r in rasters[1:]:
        if r.nx != nx or r.ny != ny:
            raise AssertionError("All rasters must have same nx, ny")
        for j in range(nx):
            if abs(r.x[j] - r0.x[j]) > 1e-12:
                raise AssertionError("Raster x grids do not match")
        for j in range(ny):
            if abs(r.y[j] - r0.y[j]) > 1e-12:
                raise AssertionError("Raster y grids do not match")

    num_cell_points = num_cell_rows * num_cell_cols
    points: List[Point] = []
    max_num_points = (nx * num_cell_cols) * (ny * num_cell_rows)

    rasters_point_demands: List[List[float]] = [
        [0.0] * max_num_points for _ in range(num_rasters)
    ]

    for i in range(nx):
        for j in range(ny):
            all_zero = True
            for r in rasters:
                if r.z[i, j] != 0:
                    all_zero = False
                    break
            if all_zero:
                continue

            for cell_row in range(1, num_cell_rows + 1):
                for cell_col in range(1, num_cell_cols + 1):
                    p = Point()
                    p.index = len(points)

                    p.location.x = r0.x[i] + r0.dx * (((cell_col - 0.5) / num_cell_cols) - 0.5)
                    p.location.y = r0.y[j] + r0.dy * (((cell_row - 0.5) / num_cell_rows) - 0.5)

                    for raster_idx, r in enumerate(rasters):
                        rasters_point_demands[raster_idx][p.index] = r.z[i, j] / num_cell_points

                    points.append(p)

    num_points = len(points)
    rasters_point_demands = [arr[:num_points] for arr in rasters_point_demands]
    return points, rasters_point_demands


def set_points_nearest_nodes(sim: Simulation, points: List[Point]) -> None:
    nodes = getattr(getattr(sim.net, "f_graph", None), "nodes", None) if sim.net is not None else None
    for p in points:
        if p.location.x is None or p.location.y is None:
            raise AssertionError("Point location must be set")
        node_idx, node_dist = find_nearest_node(sim.map, sim.grid, nodes, p.location)
        p.nearest_node_index = node_idx
        p.nearest_node_dist = node_dist


def create_points_coverage_mode(sim: Simulation, travel_mode: TravelMode, cover_time: float) -> PointsCoverageMode:
    if cover_time < 0:
        raise AssertionError("cover_time must be >= 0")
    if not sim.demand_coverage.nodes_points:
        raise AssertionError("sim.demand_coverage.nodes_points is empty; call initDemandCoverage first?")

    pcm = PointsCoverageMode()
    pcm.cover_time = cover_time
    pcm.travel_mode = travel_mode
    pcm.points = sim.demand_coverage.points

    points = pcm.points
    num_points = len(points)
    stations = sim.stations
    num_stations = len(stations)

    nodes = getattr(getattr(sim.net, "f_graph", None), "nodes", None)
    num_nodes = len(nodes) - 1 if nodes is not None else int(getattr(sim.net, "fgraph_nodes_count", 0))
    if num_nodes <= 0:
        raise AssertionError("Cannot determine num_nodes from sim.net")

    nodes_points = sim.demand_coverage.nodes_points

    stations_cover_points: List[List[bool]] = [[False] * num_points for _ in range(num_stations)]

    for i, station in enumerate(stations):
        if i == 0:
            continue
        node1 = station.nearest_node_index
        dist1 = station.nearest_node_dist
        time1 = off_road_travel_time(travel_mode, dist1)

        for node2 in range(1, num_nodes + 1):
            path_tt = sim.shortest_path_time(node1, node2, travel_mode.index)  # <-- renamed
            if time1 + path_tt <= cover_time:
                for k in nodes_points[node2]:
                    pt = points[k]
                    if pt.nearest_node_index != node2:
                        raise AssertionError("nodes_points inconsistent with point.nearest_node_index")
                    time2 = off_road_travel_time(travel_mode, pt.nearest_node_dist)
                    stations_cover_points[i][k] = (time1 + path_tt + time2 <= cover_time)

    station_sets_points: Dict[Tuple[bool, ...], List[int]] = {}
    for k in range(num_points):
        station_set_key = tuple(stations_cover_points[i][k] for i in range(num_stations))
        station_sets_points.setdefault(station_set_key, []).append(k)

    point_sets: List[List[int]] = list(station_sets_points.values())
    station_sets: List[List[int]] = [
        [i for i, covered in enumerate(key) if covered] for key in station_sets_points.keys()
    ]

    stations_cover_point_sets: List[List[int]] = [[] for _ in range(num_stations)]
    for s in range(num_stations):
        stations_cover_point_sets[s] = [ps_i for ps_i, st_set in enumerate(station_sets) if s in st_set]

    pcm.point_sets = point_sets
    pcm.station_sets = station_sets
    pcm.stations_cover_point_sets = stations_cover_point_sets
    return pcm


def get_points_coverage_mode(demand_coverage: DemandCoverage, travel_mode: TravelMode, cover_time: float) -> PointsCoverageMode:
    tmi = int(travel_mode.index) - 1
    if tmi not in demand_coverage.points_coverage_mode_lookup:
        raise AssertionError("travel_mode.index not in points_coverage_mode_lookup")
    if cover_time not in demand_coverage.points_coverage_mode_lookup[tmi]:
        raise AssertionError("cover_time not in points_coverage_mode_lookup[travel_mode.index]")

    pcm_index = demand_coverage.points_coverage_mode_lookup[tmi][cover_time]
    return demand_coverage.points_coverage_modes[pcm_index]


def get_points_coverage_mode_mut(sim: Simulation, demand_priority: Priority, current_time: float) -> PointsCoverageMode:
    if demand_priority is None:
        raise AssertionError("demand_priority cannot be None")
    if current_time is None:
        raise AssertionError("current_time cannot be None")

    travel_priority = sim.response_travel_priorities[demand_priority]

    travel_mode = get_travel_mode(sim.travel, travel_priority, current_time)

    cover_time = sim.demand_coverage.cover_times[demand_priority]
    pcm = get_points_coverage_mode(sim.demand_coverage, travel_mode, cover_time)
    if pcm is None:
        raise AssertionError("points coverage mode is None")
    return pcm


def get_point_sets_demands_mut(
    sim: Simulation,
    demand_priority: Priority,
    current_time: float,
    *,
    points_coverage_mode: Optional[PointsCoverageMode] = None,
) -> List[float]:
    if points_coverage_mode is None:
        points_coverage_mode = get_points_coverage_mode_mut(sim, demand_priority, current_time)

    demand_mode = get_demand_mode(sim.demand, demand_priority, current_time)

    if points_coverage_mode.index is None:
        raise AssertionError("points_coverage_mode.index must be set")
    if demand_mode.raster_index is None:
        raise AssertionError("demand_mode.raster_index must be set")

    return sim.demand_coverage.point_sets_demands[points_coverage_mode.index][demand_mode.raster_index]


def get_points_demands_mut(
    sim: Simulation,
    demand_priority: Priority,
    current_time: float,
) -> List[float]:
    """Return current demand per demand point (atom) for one priority."""

    if not sim.demand_coverage.initialised:
        raise AssertionError("sim.demand_coverage must be initialised before updating atom demands")

    demand_mode = get_demand_mode(sim.demand, demand_priority, current_time)
    if demand_mode.raster_index is None:
        raise AssertionError("demand_mode.raster_index must be set")

    raster_index = int(demand_mode.raster_index)
    if not (0 <= raster_index < len(sim.demand_coverage.rasters_point_demands)):
        raise AssertionError("demand_mode.raster_index out of bounds for rasters_point_demands")

    base_point_demands = sim.demand_coverage.rasters_point_demands[raster_index]
    mult = float(demand_mode.raster_multiplier)
    return [v * mult for v in base_point_demands]


def update_atom_demands_mut(
    sim: Simulation,
    current_time: float,
    *,
    demand_priorities: Optional[List[Priority]] = None,
    set_point_values: bool = True,
) -> Dict[Priority, List[float]]:
    """Update current demand for each atom.

    Here an "atom" is one entry of ``sim.demand_coverage.points``. The returned
    dict stores a demand vector for each priority at ``current_time``. When
    ``set_point_values`` is true, each point's ``value`` is replaced with a
    ``{Priority: float}`` dict holding its current demand values.
    """

    if current_time is None:
        raise AssertionError("current_time cannot be None")
    if not sim.demand_coverage.initialised:
        raise AssertionError("sim.demand_coverage must be initialised before updating atom demands")

    if demand_priorities is None:
        demand_priorities = list(PRIORITIES)

    atoms_demands: Dict[Priority, List[float]] = {}
    for pr in demand_priorities:
        atoms_demands[pr] = get_points_demands_mut(sim, pr, current_time)

    if set_point_values:
        points = sim.demand_coverage.points
        for pt_idx, point in enumerate(points):
            point.value = {pr: atoms_demands[pr][pt_idx] for pr in demand_priorities}

    return atoms_demands


def init_demand(sim: Simulation, demand: Optional[Demand] = None, *, demand_filename: str = "") -> None:
    if demand is None:
        if demand_filename:
            demand = read_demand_file(demand_filename)
        else:
            if sim.demand.num_sets == 0:
                if "demand" in sim.input_files:
                    demand = read_demand_file(sim.input_files["demand"].path)
                else:
                    raise RuntimeError("no demand data given")
            else:
                demand = sim.demand

    demand.initialised = True
    sim.demand = demand

    old = sim.demand_coverage
    sim.demand_coverage = DemandCoverage(
        cover_times=dict(old.cover_times),
        raster_cell_num_rows=old.raster_cell_num_rows,
        raster_cell_num_cols=old.raster_cell_num_cols,
        initialised=False,
    )


def init_points_coverage_modes(sim: Simulation) -> None:
    dc = sim.demand_coverage
    travel = sim.travel

    dc.points_coverage_mode_lookup = {i: {} for i in range(travel.num_modes)}

    for set_i in range(travel.num_sets):
        for demand_priority in PRIORITIES:
            travel_priority = sim.response_travel_priorities[demand_priority]

            tp_col = int(travel_priority.value)

            travel_mode_index = travel.mode_lookup[set_i + 1][tp_col]
            travel_mode = travel.modes[travel_mode_index - 1]
            cover_time = dc.cover_times[demand_priority]

            tmi = int(travel_mode.index) - 1
            if cover_time not in dc.points_coverage_mode_lookup[tmi]:
                pcm = create_points_coverage_mode(sim, travel_mode, cover_time)
                pcm.index = len(dc.points_coverage_modes)
                dc.points_coverage_modes.append(pcm)
                dc.points_coverage_mode_lookup[tmi][cover_time] = pcm.index



def init_demand_coverage(
    sim: Simulation,
    *,
    cover_times: Optional[Dict[Priority, float]] = None,
    raster_cell_num_rows: int = 1,
    raster_cell_num_cols: int = 1,
) -> None:
    if sim.used:
        raise AssertionError("sim.used must be False before init_demand_coverage")
    if sim.travel.recent_sets_start_times_index != 0:
        raise AssertionError("travel.recent_sets_start_times_index must start at 0")

    if not sim.demand.initialised:
        init_demand(sim)

    if sim.demand.recent_sets_start_times_index != 0:
        raise AssertionError("demand.recent_sets_start_times_index must start at 0")

    demand = sim.demand
    travel = sim.travel

    old_dc = sim.demand_coverage
    if cover_times is not None:
        sim.demand_coverage = DemandCoverage(
            cover_times=cover_times,
            raster_cell_num_rows=raster_cell_num_rows,
            raster_cell_num_cols=raster_cell_num_cols,
            initialised=False,
        )
    else:
        if not old_dc.cover_times:
            raise AssertionError("cover_times not provided and sim.demand_coverage.cover_times is empty")
        sim.demand_coverage = DemandCoverage(
            cover_times=dict(old_dc.cover_times),
            raster_cell_num_rows=old_dc.raster_cell_num_rows,
            raster_cell_num_cols=old_dc.raster_cell_num_cols,
            initialised=False,
        )

    dc = sim.demand_coverage

    for p in PRIORITIES:
        if p not in dc.cover_times:
            raise AssertionError("coverTimes not set for all priorities")
    if dc.raster_cell_num_rows < 1 or dc.raster_cell_num_cols < 1:
        raise AssertionError("raster_cell_num_rows/cols must be >= 1")

    dc.points, dc.rasters_point_demands = create_demand_points_from_rasters(
        demand, num_cell_rows=dc.raster_cell_num_rows, num_cell_cols=dc.raster_cell_num_cols
    )
    set_points_nearest_nodes(sim, dc.points)

    nodes = getattr(getattr(sim.net, "f_graph", None), "nodes", None)
    num_nodes = len(nodes) - 1 if nodes is not None else int(getattr(sim.net, "fgraph_nodes_count", 0))
    if num_nodes <= 0:
        raise AssertionError("Cannot determine num_nodes from sim.net")

    dc.nodes_points = [[] for _ in range(num_nodes + 1)]
    for pt in dc.points:
        dc.nodes_points[pt.nearest_node_index].append(pt.index)

    init_points_coverage_modes(sim)

    num_pcms = len(dc.points_coverage_modes)
    num_rasters = demand.num_rasters if demand.num_rasters > 0 else len(demand.rasters)

    dc.point_sets_demands = [[[] for _ in range(num_rasters)] for _ in range(num_pcms)]

    times = sorted(set((travel.sets_start_times or []) + (demand.sets_start_times or [])))

    for t in times:
        for demand_priority in PRIORITIES:
            pcm = get_points_coverage_mode_mut(sim, demand_priority, t)
            dm = get_demand_mode(demand, demand_priority, t)
            raster_index = dm.raster_index
            if raster_index is None:
                raise AssertionError("DemandMode.raster_index must be set")

            if pcm.index is None:
                raise AssertionError("PointsCoverageMode.index must be set (after init_points_coverage_modes)")
            if not dc.point_sets_demands[pcm.index][raster_index]:
                point_demands = dc.rasters_point_demands[raster_index]
                point_sets_demand = [0.0] * len(pcm.point_sets)
                for ps_idx, ps in enumerate(pcm.point_sets):
                    s = 0.0
                    for pt_idx in ps:
                        s += point_demands[pt_idx]
                    point_sets_demand[ps_idx] = s
                dc.point_sets_demands[pcm.index][raster_index] = point_sets_demand


    travel.recent_sets_start_times_index = 0
    demand.recent_sets_start_times_index = 0

    dc.initialised = True


def calc_point_sets_cover_counts(pcm: PointsCoverageMode, stations_num_ambs: List[int]) -> List[int]:
    if any(n < 0 for n in stations_num_ambs):
        raise AssertionError("stations_num_ambs must be >= 0")

    num_point_sets = len(pcm.point_sets)
    cover_count = [0] * num_point_sets
    for ps_i in range(num_point_sets):
        for station_idx in pcm.station_sets[ps_i]:
            cover_count[ps_i] += stations_num_ambs[station_idx]
    return cover_count


def calc_point_sets_cover_counts_mut(
    sim: Simulation,
    current_time: float,
    stations_num_ambs: List[int],
    *,
    demand_priorities: Optional[List[Priority]] = None,
) -> Dict[Priority, List[int]]:
    if demand_priorities is None:
        demand_priorities = list(PRIORITIES)

    if len(stations_num_ambs) != sim.num_stations:
        raise AssertionError("stations_num_ambs length must equal sim.num_stations")
    if any(n < 0 for n in stations_num_ambs):
        raise AssertionError("stations_num_ambs must be >= 0")

    demands_point_sets_cover_counts: Dict[Priority, List[int]] = {}
    pcm_used_by: Dict[int, Priority] = {}

    for pr in demand_priorities:
        pcm = get_points_coverage_mode_mut(sim, pr, current_time)
        if pcm.index is None:
            raise AssertionError("PointsCoverageMode.index must be set")
        if pcm.index in pcm_used_by:
            other_pr = pcm_used_by[pcm.index]
            demands_point_sets_cover_counts[pr] = demands_point_sets_cover_counts[other_pr]
        else:
            demands_point_sets_cover_counts[pr] = calc_point_sets_cover_counts(pcm, stations_num_ambs)
            pcm_used_by[pcm.index] = pr

    return demands_point_sets_cover_counts


def calc_demand_cover_counts_mut(
    sim: Simulation,
    current_time: float,
    stations_num_ambs: List[int],
    *,
    demand_priorities: Optional[List[Priority]] = None,
) -> Dict[Priority, List[float]]:
    if demand_priorities is None:
        demand_priorities = list(PRIORITIES)

    if len(stations_num_ambs) != sim.num_stations:
        raise AssertionError("stations_num_ambs length must equal sim.num_stations")
    if any(n < 0 for n in stations_num_ambs):
        raise AssertionError("stations_num_ambs must be >= 0")

    num_ambs = sum(stations_num_ambs)

    demands_ps_cover_counts = calc_point_sets_cover_counts_mut(
        sim, current_time, stations_num_ambs, demand_priorities=demand_priorities
    )

    demand_cover_counts: Dict[Priority, List[float]] = {}

    cover_counts_calculated: List[Dict[Tuple[int, ...], List[float]]] = [
        {} for _ in range(sim.demand.num_rasters if sim.demand.num_rasters > 0 else len(sim.demand.rasters))
    ]

    for pr in demand_priorities:
        pcm = get_points_coverage_mode_mut(sim, pr, current_time)
        dm = get_demand_mode(sim.demand, pr, current_time)

        if pcm.index is None:
            raise AssertionError("PointsCoverageMode.index must be set")
        if dm.raster_index is None:
            raise AssertionError("DemandMode.raster_index must be set")

        raster_index = dm.raster_index
        ps_cover_count = demands_ps_cover_counts[pr]
        ps_demands = sim.demand_coverage.point_sets_demands[pcm.index][raster_index]

        key = tuple(ps_cover_count)
        if key in cover_counts_calculated[raster_index]:
            cover_counts = cover_counts_calculated[raster_index][key]
        else:
            cover_counts = [0.0] * num_ambs
            for ps_i in range(len(pcm.point_sets)):
                c = ps_cover_count[ps_i]
                if c > 0:
                    cover_counts[c - 1] += ps_demands[ps_i]
            cover_counts_calculated[raster_index][key] = cover_counts

        mult = float(dm.raster_multiplier)
        demand_cover_counts[pr] = [v * mult for v in cover_counts]

    return demand_cover_counts
