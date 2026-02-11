"""jemss.init_sim

Initialise a :class:`~jemss.simulator.Simulation` from a Julia-style ``simConfig`` XML.

As of *Step 7*, this also:

* initialises station occupancy statistics
* creates the initial future-event list:
  - first call arrival event
  - (optional) ambulance wake-up events
  - (optional) tour-window active/inactive events (FDNY fork behaviour)
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .config import SimConfig, load_sim_config
from .decision import add_call_to_queue_sort_priority_then_time, find_nearest_dispatchable_amb_als_bls
from .defs import AmbStatus, EventForm
from .map import Grid, Route, find_nearest_node, grid_place_nodes
from .pathfinding import ShortestPathCache
from .read_sim_files import (
    read_ambs_file,
    read_arcs_file,
    read_calls_file,
    read_demand_coverage_file,
    read_hospitals_file,
    read_map_file,
    read_mobilisation_delay_file,
    read_nodes_file,
    read_priorities_file,
    read_redispatch_file,
    read_stations_file,
    read_stats_control_file,
    read_travel_file,
)
from .simulator import Simulation


def init_sim(
    config_filename: str,
    *,
    allow_write_output: bool = False,
    compute_checksums: bool = True,
    do_print: bool = False,
    seed: Optional[int] = None,
) -> Simulation:
    """Initialise a :class:`~jemss.simulator.Simulation` from a simConfig XML."""

    cfg = load_sim_config(
        config_filename,
        allow_write_output=allow_write_output,
        compute_checksums=compute_checksums,
    )
    return init_sim_from_config(cfg, do_print=do_print, seed=seed)


def init_sim_from_config(cfg: SimConfig, *, do_print: bool = False, seed: Optional[int] = None) -> Simulation:
    sim = Simulation()

    # Optional simulation-level seed. This does *not* change behaviour when
    # all distributions have explicit seeds, but it makes runs deterministic
    # when some files request random seeds (seed < 0), matching Julia's
    # GLOBAL_RNG-driven behaviour.
    if seed is not None:
        sim.rng_seed = int(seed)
        sim.rng = random.Random(int(seed))

    # config layer
    sim.input_path = cfg.input_path
    sim.output_path = cfg.output_path
    sim.write_output = cfg.write_output
    sim.input_files = dict(cfg.input_files)
    sim.output_files = dict(cfg.output_files)

    if cfg.events_file_filter is not None:
        sim.events_file.event_filter.update(cfg.events_file_filter)

    # default decision functions for now
    sim.add_call_to_queue = add_call_to_queue_sort_priority_then_time
    sim.find_amb_to_dispatch = find_nearest_dispatchable_amb_als_bls

    def p(msg: str) -> None:
        if do_print:
            print(msg)

    # required input files
    required = [
        "ambulances",
        "hospitals",
        "stations",
        "nodes",
        "arcs",
        "map",
        "priorities",
        "travel",
    ]
    missing = [k for k in required if k not in cfg.input_files]
    if missing:
        raise KeyError(f"Missing required simFiles entries: {missing}")

    # calls vs callGenConfig (generator not ported yet)
    has_calls = "calls" in cfg.input_files
    has_call_gen = "callGenConfig" in cfg.input_files
    if (1 if has_calls else 0) + (1 if has_call_gen else 0) != 1:
        raise ValueError("Need exactly one of these input files: calls, callGenConfig")

    p("reading sim data")
    sim.ambulances = read_ambs_file(cfg.input_files["ambulances"].path)
    sim.hospitals = read_hospitals_file(cfg.input_files["hospitals"].path)
    sim.stations = read_stations_file(cfg.input_files["stations"].path)

    # Backward compatibility: older ambulance files may not include cross-street
    # coordinates. In that case, default the cross-street to the assigned station.
    def _coord_missing(v: object) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v))

    for a in sim.ambulances[1:]:
        if a.station_index is None or not (1 <= int(a.station_index) < len(sim.stations)):
            raise ValueError(f"Ambulance {a.index} has invalid station_index {a.station_index}")
        if _coord_missing(a.cross_street_location.x) or _coord_missing(a.cross_street_location.y):
            st = sim.stations[int(a.station_index)]
            a.cross_street_location.x = st.location.x
            a.cross_street_location.y = st.location.y

    if has_calls:
        sim.calls, sim.start_time = read_calls_file(cfg.input_files["calls"].path)
    else:
        raise NotImplementedError("callGenConfig is not implemented yet")

    sim.time = sim.start_time

    # counts (match Julia cached counts; convenient later)
    sim.num_ambs = len(sim.ambulances) - 1
    sim.num_calls = len(sim.calls) - 1
    sim.num_hospitals = len(sim.hospitals) - 1
    sim.num_stations = len(sim.stations) - 1

    p("reading network tables")
    sim.net.f_graph.nodes = read_nodes_file(cfg.input_files["nodes"].path)
    sim.net.f_graph.arcs, sim.arc_travel_times = read_arcs_file(cfg.input_files["arcs"].path)

    # Step 5: adjacency + shortest-path backend
    p("building adjacency + shortest-path cache")
    sim.net.f_graph.build_adjacency()
    sim.sp_cache = ShortestPathCache(sim.net.f_graph)

    p("reading misc tables")
    sim.map = read_map_file(cfg.input_files["map"].path)
    sim.target_response_durations, sim.response_travel_priorities = read_priorities_file(
        cfg.input_files["priorities"].path
    )
    sim.travel = read_travel_file(cfg.input_files["travel"].path)

    # Step 4: grid placement + nearest-node linkage
    p("placing nodes in grid")

    num_nodes = len(sim.net.f_graph.nodes) - 1  # nodes are 1-based; index 0 unused
    xr = sim.map.x_range
    yr = sim.map.y_range
    if xr is None or yr is None:
        raise ValueError("Map range must be set")
    x_dist = xr * sim.map.x_scale
    y_dist = yr * sim.map.y_scale
    if x_dist <= 0 or y_dist <= 0:
        raise ValueError("Map scaled extents must be positive")
    nx = int(math.ceil(math.sqrt(num_nodes * x_dist / y_dist)))
    ny = int(math.ceil(math.sqrt(num_nodes * y_dist / x_dist)))
    nx = max(1, nx)
    ny = max(1, ny)

    sim.grid = Grid.from_map(sim.map, nx, ny)
    grid_place_nodes(sim.map, sim.grid, sim.net.f_graph.nodes)

    p("linking entities to nearest nodes")

    # Calls
    for call in sim.calls[1:]:
        if call.nearest_node_index is None:
            call.nearest_node_index, call.nearest_node_dist = find_nearest_node(
                sim.map, sim.grid, sim.net.f_graph.nodes, call.location
            )

    # Hospitals / stations
    for h in sim.hospitals[1:]:
        h.nearest_node_index, h.nearest_node_dist = find_nearest_node(
            sim.map, sim.grid, sim.net.f_graph.nodes, h.location
        )
    for s in sim.stations[1:]:
        s.nearest_node_index, s.nearest_node_dist = find_nearest_node(
            sim.map, sim.grid, sim.net.f_graph.nodes, s.location
        )

    # Ambulance cross-street nodes
    for a in sim.ambulances[1:]:
        a.cross_street_node_index, a.cross_street_node_dist = find_nearest_node(
            sim.map, sim.grid, sim.net.f_graph.nodes, a.cross_street_location
        )

    # ------------------------------------------------------------------
    # Step 6/7: initialise station stats + ambulance state/routes + initial
    # events.
    # ------------------------------------------------------------------

    if sim.start_time is None:
        raise ValueError("sim.start_time must be set")

    p("initialising station stats")
    for st in sim.stations[1:]:
        st.init_stats(start_time=float(sim.start_time))

    p("initialising ambulance routes + sleep status")
    for a in sim.ambulances[1:]:
        if a.station_index is None or a.station_index <= 0:
            raise ValueError(f"Ambulance {a.index} has invalid station_index {a.station_index}")
        station = sim.stations[int(a.station_index)]
        if station.nearest_node_index is None or station.nearest_node_dist is None:
            raise ValueError(f"Station {station.index} is missing nearest-node linkage")

        a.route = Route()
        a.route.init_at_location(
            start_loc=station.location,
            start_fnode=int(station.nearest_node_index),
            start_fnode_dist=float(station.nearest_node_dist),
        )

        # Initialise at station; sleep until activated by events.
        a.set_status(sim, AmbStatus.SLEEPING, float(sim.start_time))

        a.current_loc = station.location
        a.dest_loc = station.location
        a.moved_loc = False

        # Mirror Julia's initAmbulance!: if the ambulance has any tour defined,
        # add an initial wake-up event at sim.start_time.
        has_tour = any(
            t is not None
            for t in (
                a.tour1_start,
                a.tour2_start,
                a.tour3_start,
            )
        )
        if has_tour:
            sim.add_event(
                form=EventForm.AMB_WAKES_UP,
                time=float(sim.start_time),
                ambulance=a,
                station=station,
            )

    # Calls: add first call-arrival event (later call arrivals are chained).
    if sim.num_calls > 0:
        first = sim.calls[1]
        if first.arrival_time is None:
            raise ValueError("first call missing arrival_time")
        sim.add_event(form=EventForm.CALL_ARRIVES, time=float(first.arrival_time), call=first, add_event_to_amb=False)

    # Tour-window events (FDNY fork behaviour): schedule daily on/off events
    # through the day after the last call arrival.
    max_arrival: float = 0.0
    for c in sim.calls[1:]:
        if c.arrival_time is not None:
            max_arrival = max(max_arrival, float(c.arrival_time))
    last_day = int(math.ceil(max_arrival)) + 1

    for a in sim.ambulances[1:]:
        tours = [
            (a.tour1_start, a.tour1_end),
            (a.tour2_start, a.tour2_end),
            (a.tour3_start, a.tour3_end),
        ]
        for ts, te in tours:
            if ts is None or te is None:
                continue
            ts = float(ts)
            te = float(te)
            for day in range(0, last_day + 1):
                t_on = ts + float(day)
                t_off = (te if te >= ts else te + 1.0) + float(day)
                if t_on >= float(sim.start_time):
                    sim.add_event(form=EventForm.AMB_BECOMES_ACTIVE, time=t_on, ambulance=a, add_event_to_amb=False)
                if t_off >= float(sim.start_time):
                    sim.add_event(form=EventForm.AMB_BECOMES_INACTIVE, time=t_off, ambulance=a, add_event_to_amb=False)

    # optional inputs
    if "demandCoverage" in cfg.input_files:
        sim.demand_coverage = read_demand_coverage_file(cfg.input_files["demandCoverage"].path)

    if "mobilisationDelay" in cfg.input_files:
        sim.mobilisation_delay = read_mobilisation_delay_file(
            cfg.input_files["mobilisationDelay"].path,
            seed_rng=sim.rng,
        )

    if "redispatch" in cfg.input_files:
        sim.redispatch = read_redispatch_file(cfg.input_files["redispatch"].path)

    if "statsControl" in cfg.input_files:
        sim.stats_control = read_stats_control_file(cfg.input_files["statsControl"].path)

    # sanity checks across files
    if sim.arc_travel_times is not None and sim.travel.num_modes:
        num_modes_arcs = len(sim.arc_travel_times) - 1
        if num_modes_arcs != sim.travel.num_modes:
            raise ValueError(
                f"Mismatch between arcs.numModes ({num_modes_arcs}) and travel.num_modes ({sim.travel.num_modes})"
            )

    sim.initialised = True
    return sim
