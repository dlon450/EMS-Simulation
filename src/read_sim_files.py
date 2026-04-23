from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import ast
import math
import numpy as np
from osgeo import gdal
import os

from .defs import AmbClass, Priority, PRIORITIES
from .demand import DemandCoverage, Demand, DemandMode
from .entities import Ambulance, Call, Hospital, Station
from .file_io import (
    Table,
    parse_attributes_column,
    read_tables_from_file,
    table_rows_field_dicts,
    join_path_if_not_abs,
    interpolate_string
)
from .geo import Arc, Location, Node
from .map import Map, Raster
from .misc import DistrRng, MobilisationDelay, Redispatch
from .network import Travel, TravelMode


def _is_missing(x: Any) -> bool:
    return x is None or x == "" or (isinstance(x, float) and math.isnan(x))


def _as_int(x: Any) -> Optional[int]:
    if _is_missing(x):
        return None
    return int(x)


def _as_float(x: Any) -> Optional[float]:
    if _is_missing(x):
        return None
    return float(x)


def _as_float_req(x: Any, *, field: str = "value") -> float:
    v = _as_float(x)
    if v is None:
        raise ValueError(f"Missing required numeric field: {field}")
    return v


_PRIORITY_TOKEN_MAP: Dict[str, Priority] = {
    "HIGH": Priority.HIGH,
    "HIGHPriority".upper(): Priority.HIGH,
    "HIGHPRIORITY": Priority.HIGH,
    "MED": Priority.MED,
    "MEDIUM": Priority.MED,
    "MEDPRIORITY": Priority.MED,
    "LOW": Priority.LOW,
    "LOWPRIORITY": Priority.LOW,
}


def parse_priority_token(x: Any) -> Priority:
    """Parse a priority token as found in reference input tables.

    Supports ints (1..), :class:`Priority` values, and strings such as
    ``"highPriority"``, ``"Priority.HIGH"``, or ``"HIGH"``.
    """

    if isinstance(x, Priority):
        return x
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return Priority(int(x))
    if not isinstance(x, str):
        return Priority(int(x))

    s = x.strip()
    if s == "":
        raise ValueError("Empty priority token")

    # Strip common prefixes.
    if s.startswith("Priority."):
        s = s.split(".", 1)[1]
    if s.startswith(":"):
        s = s[1:]

    su = s.upper()

    # reference enum string forms: highPriority, medPriority, lowPriority
    if su in _PRIORITY_TOKEN_MAP:
        return _PRIORITY_TOKEN_MAP[su]

    # Also accept pythonic names.
    if su in Priority.__members__:
        return Priority[su]

    # Accept "Priority(1)" style.
    if su.startswith("PRIORITY(") and su.endswith(")"):
        inner = su[len("PRIORITY(") : -1]
        return Priority(int(inner))

    raise ValueError(f"Unrecognised priority token: {x!r}")


_AMB_CLASS_TOKEN_MAP: Dict[str, AmbClass] = {
    "ALS": AmbClass.ALS,
    "BLS": AmbClass.BLS,
    # reference enum names are lowercase: als/bls
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    # common reference string(AmbClass) values
    "ALS".replace("ALS", "ALS").upper(): AmbClass.ALS,
    "BLS".replace("BLS", "BLS").upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    "ALS".upper(): AmbClass.ALS,
    "BLS".upper(): AmbClass.BLS,
    # reference exact enum strings: "als" / "bls"
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    "ALS".lower().upper(): AmbClass.ALS,
    "BLS".lower().upper(): AmbClass.BLS,
    # allow old-style names
    "AMBCLASSALS": AmbClass.ALS,
    "AMBCLASSBLS": AmbClass.BLS,
}


def parse_amb_class_token(x: Any) -> AmbClass:
    if isinstance(x, AmbClass):
        return x
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return AmbClass(int(x))
    if not isinstance(x, str):
        return AmbClass(int(x))

    s = x.strip()
    if s == "":
        raise ValueError("Empty ambulance class token")

    if s.startswith("AmbClass."):
        s = s.split(".", 1)[1]

    su = s.upper()
    if su in _AMB_CLASS_TOKEN_MAP:
        return _AMB_CLASS_TOKEN_MAP[su]

    if su in AmbClass.__members__:
        return AmbClass[su]

    raise ValueError(f"Unrecognised ambulance class token: {x!r}")


def read_ambs_file(filename: str) -> List[Ambulance]:
    tables = read_tables_from_file(filename)
    table = tables["ambulances"]
    n = table.num_rows
    if n < 1:
        raise ValueError("ambulances table must have at least one row")

    cols = table.columns
    attrs = parse_attributes_column(table)

    ambs: List[Ambulance] = [Ambulance(index=0)]

    for row_i in range(n):
        i = row_i + 1
        a = Ambulance()
        a.index = int(cols["index"][row_i])
        a.station_index = int(cols["stationIndex"][row_i])
        a.amb_class = parse_amb_class_token(cols["class"][row_i])
        a.unit_name = str(cols.get("unitName", [""] * n)[row_i])

        # Some legacy model files do not include cross-street columns.
        # Leave coordinates unset here; init_sim will fall back to station coords.
        cross_x = _as_float(cols.get("crossStreetx", [None] * n)[row_i])
        cross_y = _as_float(cols.get("crossStreety", [None] * n)[row_i])
        a.cross_street_location = Location(
            x=cross_x,
            y=cross_y,
        )

        # tours may be blank
        a.tour1_start = _as_float(cols.get("tour1Start", [""] * n)[row_i])
        a.tour1_end = _as_float(cols.get("tour1End", [""] * n)[row_i])
        a.tour2_start = _as_float(cols.get("tour2Start", [""] * n)[row_i])
        a.tour2_end = _as_float(cols.get("tour2End", [""] * n)[row_i])
        a.tour3_start = _as_float(cols.get("tour3Start", [""] * n)[row_i])
        a.tour3_end = _as_float(cols.get("tour3End", [""] * n)[row_i])

        a.attributes = attrs[row_i]

        if a.index != i:
            raise ValueError(f"Ambulance index mismatch at row {i}: got {a.index}")
        ambs.append(a)

    return ambs


def read_hospitals_file(filename: str) -> List[Hospital]:
    tables = read_tables_from_file(filename)
    table = tables["hospitals"]
    n = table.num_rows
    if n < 1:
        raise ValueError("hospitals table must have at least one row")

    cols = table.columns
    attrs = parse_attributes_column(table)

    hospitals: List[Hospital] = [Hospital(index=0)]
    for row_i in range(n):
        i = row_i + 1
        h = Hospital()
        h.index = int(cols["index"][row_i])
        h.location = Location(x=_as_float_req(cols["x"][row_i], field="x"), y=_as_float_req(cols["y"][row_i], field="y"))
        h.attributes = attrs[row_i]
        if h.index != i:
            raise ValueError(f"Hospital index mismatch at row {i}: got {h.index}")
        hospitals.append(h)

    return hospitals


def read_stations_file(filename: str) -> List[Station]:
    tables = read_tables_from_file(filename)
    table = tables["stations"]
    n = table.num_rows
    if n < 1:
        raise ValueError("stations table must have at least one row")

    cols = table.columns
    attrs = parse_attributes_column(table)

    stations: List[Station] = [Station(index=0)]
    for row_i in range(n):
        i = row_i + 1
        s = Station()
        s.index = int(cols["index"][row_i])
        s.location = Location(x=_as_float_req(cols["x"][row_i], field="x"), y=_as_float_req(cols["y"][row_i], field="y"))
        s.capacity = int(cols.get("capacity", [0] * n)[row_i])
        s.attributes = attrs[row_i]
        if s.index != i:
            raise ValueError(f"Station index mismatch at row {i}: got {s.index}")
        if s.capacity < 0:
            raise ValueError(f"Station capacity must be >= 0 (station {s.index})")
        stations.append(s)

    return stations


def _parse_optional_hospital_index(x: Any) -> Optional[int]:
    # Mirrors reference parseHosp.
    if _is_missing(x):
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    # reference uses nullIndex = -1; convert to Pythonic None.
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        if int(x) == -1:
            return None
    return int(x)


def read_calls_file(filename: str) -> Tuple[List[Call], float]:
    tables = read_tables_from_file(filename)

    start_time = float(tables["miscData"].columns["startTime"][0])

    table = tables["calls"]
    n = table.num_rows
    if n < 1:
        raise ValueError("calls table must have at least one row")

    c = table.columns
    attrs = parse_attributes_column(table)

    # transport/handover compat
    if "transport" in c and "handoverDuration" in c:
        transport_col = c["transport"]
        handover_col = c["handoverDuration"]
    elif "transfer" in c and "transferDuration" in c:
        transport_col = c["transfer"]
        handover_col = c["transferDuration"]
    else:
        raise KeyError("Calls file must contain either (transport, handoverDuration) or (transfer, transferDuration)")

    calls: List[Call] = [Call(index=0)]

    for row_i in range(n):
        i = row_i + 1
        call = Call()
        call.index = int(c["index"][row_i])
        call.priority = parse_priority_token(c["priority"][row_i])
        call.location = Location(x=_as_float_req(c["x"][row_i], field="x"), y=_as_float_req(c["y"][row_i], field="y"))
        call.arrival_time = _as_float_req(c["arrivalTime"][row_i], field="arrivalTime")
        call.dispatch_delay = float(c["dispatchDelay"][row_i])
        call.on_scene_duration = float(c["onSceneDuration"][row_i])
        call.transport = bool(transport_col[row_i])
        call.handover_duration = float(handover_col[row_i]) if call.transport else 0.0
        call.hospital_index = _parse_optional_hospital_index(c.get("hospitalIndex", [""] * n)[row_i])

        rec = c.get("recommendedAmbulanceClass", [""] * n)[row_i]
        if isinstance(rec, str) and rec.strip().upper() == "ALS":
            call.recommended_amb_class = AmbClass.ALS
        elif isinstance(rec, str) and rec.strip().upper() == "BLS":
            call.recommended_amb_class = AmbClass.BLS
        else:
            # default to BLS for compatibility with reference's else branch
            call.recommended_amb_class = AmbClass.BLS

        call.attributes = attrs[row_i]

        if call.index != i:
            raise ValueError(f"Call index mismatch at row {i}: got {call.index}")
        if call.arrival_time < 0:
            raise ValueError(f"Call arrivalTime must be >= 0 (call {call.index})")
        if call.dispatch_delay < 0:
            raise ValueError(f"Call dispatchDelay must be >= 0 (call {call.index})")
        if call.on_scene_duration < 0:
            raise ValueError(f"Call onSceneDuration must be >= 0 (call {call.index})")
        if call.handover_duration < 0:
            raise ValueError(f"Call handoverDuration must be >= 0 (call {call.index})")

        calls.append(call)

    # check ordered by arrival time
    for i in range(1, len(calls) - 1):
        if calls[i].arrival_time is None or calls[i + 1].arrival_time is None:
            continue
        if calls[i].arrival_time > calls[i + 1].arrival_time:
            raise ValueError(
                f"Calls are not ordered by arrivalTime: calls[{i}].arrival_time={calls[i].arrival_time}, "
                f"calls[{i+1}].arrival_time={calls[i+1].arrival_time}"
            )

    if calls[1].arrival_time is not None and start_time > calls[1].arrival_time:
        raise ValueError("startTime must be <= first call arrivalTime")

    # Either all strictly increasing arrival times OR all dispatch delays > 0.
    strictly_increasing = all(
        calls[i].arrival_time < calls[i + 1].arrival_time
        for i in range(1, len(calls) - 1)
        if calls[i].arrival_time is not None and calls[i + 1].arrival_time is not None
    )
    if not strictly_increasing:
        if not all(call.dispatch_delay > 0 for call in calls[1:]):
            raise ValueError(
                "Calls have equal arrival times and some dispatchDelay == 0; "
                "this can cause priority-ordering bugs (mirrors reference assertion)."
            )

    return calls, start_time


def read_nodes_file(filename: str) -> List[Node]:
    tables = read_tables_from_file(filename)
    table = tables["nodes"]
    n = table.num_rows
    if n < 2:
        raise ValueError("nodes table must have at least two rows")

    c = table.columns
    offroad = c.get("offRoadAccess")
    field_names = [h for h in table.header if h not in {"index", "x", "y", "offRoadAccess"}]
    rows_fields = table_rows_field_dicts(table, field_names)

    nodes: List[Node] = [Node(index=0)]
    for row_i in range(n):
        i = row_i + 1
        node = Node()
        node.index = int(c["index"][row_i])
        node.location = Location(x=_as_float_req(c["x"][row_i], field="x"), y=_as_float_req(c["y"][row_i], field="y"))
        node.off_road_access = bool(offroad[row_i]) if offroad is not None else True
        node.fields = rows_fields[row_i]

        if node.index != i:
            raise ValueError(f"Node index mismatch at row {i}: got {node.index}")
        nodes.append(node)

    return nodes


def read_arcs_file(filename: str, *, keep_all_fields: bool = False) -> Tuple[List[Arc], List[List[float]]]:
    tables = read_tables_from_file(filename)

    misc = tables["miscData"]
    arc_form = str(misc.columns["arcForm"][0])
    if arc_form not in {"directed", "undirected"}:
        raise ValueError(f"arcForm must be 'directed' or 'undirected', got {arc_form!r}")

    num_modes = int(misc.columns["numModes"][0])
    if num_modes < 1:
        raise ValueError("numModes must be >= 1")

    table = tables["arcs"]
    num_arcs_in = table.num_rows
    if num_arcs_in < 1:
        raise ValueError("arcs table must have at least one row")

    # Check travel-time columns exist.
    for j in range(1, num_modes + 1):
        col = f"mode_{j}"
        if col not in table.header:
            raise KeyError(f"Missing travel mode column: {col}")

    total_arcs = num_arcs_in if arc_form == "directed" else (num_arcs_in * 2)

    # travel_times[mode_index][arc_index] is 1-based; index 0 is dummy.
    travel_times: List[List[float]] = [[0.0] * (total_arcs + 1)]
    for _ in range(num_modes):
        travel_times.append([0.0] * (total_arcs + 1))

    c = table.columns
    has_dist = "distance" in c
    mode_cols = [c[f"mode_{j}"] for j in range(1, num_modes + 1)]

    exclude = [] if keep_all_fields else [
        "index",
        "fromNode",
        "toNode",
        "distance",
        *[f"mode_{j}" for j in range(1, num_modes + 1)],
    ]
    field_names = [h for h in table.header if h not in set(exclude)]
    rows_fields = table_rows_field_dicts(table, field_names)

    arcs: List[Arc] = [Arc(index=0)]

    for row_i in range(num_arcs_in):
        i = row_i + 1
        arc = Arc()
        arc.index = int(c["index"][row_i])
        arc.from_node_index = int(c["fromNode"][row_i])
        arc.to_node_index = int(c["toNode"][row_i])
        arc.distance = float(c["distance"][row_i]) if has_dist else math.nan
        arc.fields = rows_fields[row_i]

        if arc.index != i:
            raise ValueError(f"Arc index mismatch at row {i}: got {arc.index}")
        if has_dist and (arc.distance is None or not (arc.distance >= 0)):
            # Note: NaN fails the >= check, mirroring reference's assertion.
            raise ValueError(f"Arc distance must be >= 0 (arc {arc.index})")

        arcs.append(arc)

        for mode_i in range(1, num_modes + 1):
            tt = float(mode_cols[mode_i - 1][row_i])
            travel_times[mode_i][i] = tt

    # If arcs are undirected, make them directed by duplicating reversed arcs.
    if arc_form == "undirected":
        for i in range(1, num_arcs_in + 1):
            j = i + num_arcs_in
            a = arcs[i]
            a2 = Arc(
                index=j,
                from_node_index=a.to_node_index,
                to_node_index=a.from_node_index,
                distance=a.distance,
                fields=dict(a.fields),
            )
            arcs.append(a2)
            for mode_i in range(1, num_modes + 1):
                travel_times[mode_i][j] = travel_times[mode_i][i]

    # travel times should be positive
    for mode_i in range(1, num_modes + 1):
        if not all(travel_times[mode_i][k] > 0 for k in range(1, total_arcs + 1)):
            raise ValueError("All travel times must be > 0")

    return arcs, travel_times


def read_map_file(filename: str) -> Map:
    tables = read_tables_from_file(filename)
    table = tables["map"]
    c = table.columns
    m = Map()
    m.x_min = float(c["xMin"][0])
    m.x_max = float(c["xMax"][0])
    m.y_min = float(c["yMin"][0])
    m.y_max = float(c["yMax"][0])
    m.x_scale = float(c["xScale"][0])
    m.y_scale = float(c["yScale"][0])

    if not (m.x_min < m.x_max and m.y_min < m.y_max):
        raise ValueError("Map bounds must satisfy xMin < xMax and yMin < yMax")
    if not (m.x_scale > 0 and m.y_scale > 0):
        raise ValueError("Map scales must be positive")

    return m


def read_priorities_file(filename: str) -> Tuple[List[float], Dict[Priority, Priority]]:
    tables = read_tables_from_file(filename)
    table = tables["priorities"]
    n = table.num_rows
    if n < 1:
        raise ValueError("priorities table must have at least one row")

    cols = table.columns

    target_col = "targetResponseDuration" if "targetResponseDuration" in cols else "targetResponseTime"

    # durations[priority_int] -> float; index 0 dummy.
    durations: List[float] = [0.0] * (n + 1)

    response_travel_priorities: Dict[Priority, Priority] = {p: p for p in PRIORITIES}

    for row_i in range(n):
        i = row_i + 1
        if int(cols["priority"][row_i]) != i:
            raise ValueError(f"Priority index mismatch at row {i}")

        name_tok = cols.get("name", [""] * n)[row_i]
        p = parse_priority_token(name_tok) if not _is_missing(name_tok) else Priority(i)
        if p != Priority(i):
            raise ValueError(f"Priority name mismatch at row {i}: {name_tok!r}")

        d = float(cols[target_col][row_i])
        durations[i] = d

        if "responseTravelPriority" in cols:
            response_travel_priorities[p] = parse_priority_token(cols["responseTravelPriority"][row_i])

    if not all(d > 0 for d in durations[1:]):
        raise ValueError("All target response durations must be > 0")

    return durations, response_travel_priorities


def read_travel_file(filename: str) -> Travel:
    tables = read_tables_from_file(filename)

    travel = Travel()

    # travel modes
    table = tables["travelModes"]
    n = table.num_rows
    if n < 1:
        raise ValueError("travelModes table must have at least one row")

    cols = table.columns

    travel.num_modes = n
    travel.modes = []
    #[TravelMode(index=0, off_road_speed=None)]
    for row_i in range(n):
        i = row_i + 1
        tm = TravelMode()
        tm.index = int(cols["travelModeIndex"][row_i])
        tm.off_road_speed = float(cols["offRoadSpeed"][row_i])
        if tm.index != i:
            raise ValueError(f"TravelMode index mismatch at row {i}: got {tm.index}")
        if not (tm.off_road_speed and tm.off_road_speed > 0):
            raise ValueError(f"TravelMode offRoadSpeed must be > 0 (mode {tm.index})")
        travel.modes.append(tm)

    # travel sets
    table = tables["travelSets"]
    n_rows = table.num_rows
    if n_rows < 1:
        raise ValueError("travelSets table must have at least one row")

    cols = table.columns
    set_indices = [int(x) for x in cols["travelSetIndex"]]
    mode_indices = [int(x) for x in cols["travelModeIndex"]]

    num_sets = max(set_indices)
    travel.num_sets = num_sets

    # allocate lookup as 1-based: [0..num_sets] x [0..num_priorities]
    num_priorities = len(PRIORITIES)
    travel.mode_lookup = [[0] * (num_priorities + 1) for _ in range(num_sets + 1)]

    # checks analogous to reference
    if sorted(set(set_indices)) != list(range(1, num_sets + 1)):
        raise ValueError("travelSets must use all travelSetIndex values from 1..numSets")
    if sorted(set(mode_indices)) != list(range(1, travel.num_modes + 1)):
        raise ValueError("travelSets must use all travelModeIndex values from 1..numModes")

    for row_i in range(n_rows):
        set_idx = int(cols["travelSetIndex"][row_i])
        pr = parse_priority_token(cols["priority"][row_i])
        mode_idx = int(cols["travelModeIndex"][row_i])

        if travel.mode_lookup[set_idx][pr.value] != 0:
            raise ValueError(f"Duplicate travel mode for set {set_idx}, priority {pr}")
        if not (1 <= mode_idx <= travel.num_modes):
            raise ValueError(f"Invalid travelModeIndex {mode_idx}")
        travel.mode_lookup[set_idx][pr.value] = mode_idx

    if any(travel.mode_lookup[si][pi] == 0 for si in range(1, num_sets + 1) for pi in range(1, num_priorities + 1)):
        raise ValueError("travel.mode_lookup not fully populated")

    # travel sets timing
    table = tables["travelSetsTiming"]
    cols = table.columns
    travel.sets_start_times = [float(x) for x in cols["startTime"]]
    travel.sets_time_order = [int(x) for x in cols["travelSetIndex"]]

    if not travel.sets_start_times or travel.sets_start_times[0] <= -1e9:
        # reference checks against nullTime; here we just ensure a real number is present.
        raise ValueError("travelSetsTiming startTime must be set")
    if any(travel.sets_start_times[i] > travel.sets_start_times[i + 1] for i in range(len(travel.sets_start_times) - 1)):
        raise ValueError("travelSetsTiming startTime values must be non-decreasing")
    if sorted(set(travel.sets_time_order)) != list(range(1, num_sets + 1)):
        raise ValueError("Each travel set must be used at least once in travelSetsTiming")

    travel.recent_sets_start_times_index = 0

    return travel


def read_demand_coverage_file(filename: str) -> DemandCoverage:
    tables = read_tables_from_file(filename)

    dc = DemandCoverage()

    table = tables["coverTimes"]
    n = table.num_rows
    if n != len(PRIORITIES):
        raise ValueError(f"coverTimes must have {len(PRIORITIES)} rows")

    cols = table.columns
    for row_i in range(n):
        pr = parse_priority_token(cols["demandPriority"][row_i])
        cover_time = float(cols["coverTime"][row_i])
        if cover_time < 0:
            raise ValueError("coverTime must be >= 0")
        dc.cover_times[pr] = cover_time

    if not all(p in dc.cover_times for p in PRIORITIES):
        raise ValueError("coverTimes not set for all priorities")

    table = tables["demandRasterCellNumPoints"]
    if table.num_rows != 1:
        raise ValueError("demandRasterCellNumPoints must have exactly one row")

    cols = table.columns
    dc.raster_cell_num_rows = int(cols["rows"][0])
    dc.raster_cell_num_cols = int(cols["cols"][0])
    if dc.raster_cell_num_rows < 1 or dc.raster_cell_num_cols < 1:
        raise ValueError("demandRasterCellNumPoints rows/cols must be >= 1")

    return dc


def read_mobilisation_delay_file(filename: str, *, seed_rng=None) -> MobilisationDelay:
    tables = read_tables_from_file(filename)
    table = tables["mobilisationDelay"]
    cols = table.columns

    use = bool(cols["use"][0])
    distr = cols.get("distribution", [None])[0]
    seed = _as_int(cols.get("seed", [None])[0])
    expected = float(cols.get("expectedDuration", [0.0])[0])

    drng = DistrRng(distribution=distr, seed=seed, seed_rng=seed_rng)
    return MobilisationDelay(use=use, distr_rng=drng, expected_duration=expected)


def _detect_priority_cycle(conditions: Dict[Priority, Dict[Priority, bool]]) -> bool:
    # Floyd-Warshall on a tiny graph.
    ps = list(PRIORITIES)
    idx = {p: i for i, p in enumerate(ps)}
    n = len(ps)
    reach = [[False] * n for _ in range(n)]
    for p1 in ps:
        for p2 in ps:
            if conditions.get(p1, {}).get(p2, False):
                reach[idx[p1]][idx[p2]] = True

    for k in range(n):
        for i in range(n):
            if not reach[i][k]:
                continue
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    return any(reach[i][i] for i in range(n))


def read_redispatch_file(filename: str) -> Redispatch:
    tables = read_tables_from_file(filename)

    redispatch = Redispatch.default()

    table = tables["miscData"]
    redispatch.allow = bool(table.columns["allowRedispatch"][0])

    # reset all conditions to False, then fill from file
    redispatch.conditions = {p1: {p2: False for p2 in PRIORITIES} for p1 in PRIORITIES}

    table = tables["redispatchConditions"]
    cols = table.columns
    used: set[tuple[Priority, Priority]] = set()

    for row_i in range(table.num_rows):
        p1 = parse_priority_token(cols["fromCallPriority"][row_i])
        p2 = parse_priority_token(cols["toCallPriority"][row_i])
        allow = bool(cols["allowRedispatch"][row_i])

        if (p1, p2) in used:
            raise ValueError(f"Duplicate redispatch condition for priority pair ({p1}, {p2})")
        used.add((p1, p2))

        if p1 == p2 and allow:
            raise ValueError(f"Redispatch must not allow same-priority redispatch ({p1})")

        redispatch.conditions[p1][p2] = allow

    if _detect_priority_cycle(redispatch.conditions):
        raise ValueError("Redispatch conditions contain a cycle that may cause unlimited redispatching")

    return redispatch


def _parse_vector_literal(text: str) -> List[float]:
    """Parse a simple reference/Python vector literal like `[1.0, 2.0]`."""

    s = text.strip()
    if s == "":
        raise ValueError("Empty vector literal")

    # Accept reference booleans too.
    s = s.replace("true", "True").replace("false", "False")

    try:
        val = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Could not parse vector literal: {text!r}") from e

    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]

    raise ValueError(f"Expected a list/tuple literal, got: {type(val).__name__}")


def read_stats_control_file(filename: str) -> Dict[str, Any]:
    """Read stats-control parameters.

    Returns a dict compatible with how the reference code stores these
    settings (we'll later map this into a structured Python type).
    """

    tables = read_tables_from_file(filename)

    table = tables["params"]
    cols = table.columns

    def param(name: str) -> Any:
        return cols[name][0]

    warm_up_duration = float(param("warmUpDuration"))
    if warm_up_duration < 0:
        raise ValueError("warmUpDuration must be >= 0")

    period_durations_str = str(param("periodDurations"))
    do_cycle = bool(param("doCyclePeriodDurations"))

    period_durations = _parse_vector_literal(period_durations_str)
    if not all(x > 0 for x in period_durations):
        raise ValueError("All period durations must be > 0")

    record_hist = False
    bin_width = None
    if "responseDurationHist_params" in tables:
        t2 = tables["responseDurationHist_params"]
        c2 = t2.columns
        record_hist = bool(c2["doRecord"][0])
        bw = c2["binWidth"][0]
        bin_width = float(_parse_vector_literal(str(bw))[0] if isinstance(bw, str) and bw.strip().startswith("[") else bw)

    return {
        "warmUpDuration": warm_up_duration,
        "periodDurations": period_durations,
        "doCyclePeriodDurations": do_cycle,
        "recordResponseDurationHist": record_hist,
        "responseDurationHistBinWidth": bin_width,
    }


def read_demand_file(path: str) -> Demand:
    tables = read_tables_from_file(path)

    demand = Demand()

    # -------------------------
    # demand rasters
    # -------------------------
    table = tables["demandRasters"]
    n = len(table.data)  # number of rows
    demand.num_rasters = n
    if n < 1:
        raise AssertionError("demandRasters must have at least 1 row")

    cols = table.columns
    demand.rasters = [None] * n  # type: ignore[list-item]
    demand.raster_filenames = [""] * n

    # warn on duplicates (Julia warns but does not fail)
    raster_fns = cols.get("rasterFilename", [])
    if len(set(raster_fns)) != len(raster_fns):
        # replace with your logger if desired
        print("Warning: duplicate raster filenames in demand file (unnecessary).")

    demand_file_dir = os.path.dirname(os.path.realpath(path))

    for i in range(n):
        raster_index_cell = cols["rasterIndex"][i]
        if raster_index_cell != i + 1:
            raise AssertionError(f"demandRasters.rasterIndex must be 1..n; got {raster_index_cell} at row {i}")

        raster_filename = str(cols["rasterFilename"][i])
        raster_filename = join_path_if_not_abs(demand_file_dir, interpolate_string(raster_filename))

        if not os.path.isfile(raster_filename):
            raise FileNotFoundError(raster_filename)

        demand.raster_filenames[i] = raster_filename
        demand.rasters[i] = read_raster_file(raster_filename)

    # -------------------------
    # demand modes
    # -------------------------
    table = tables["demandModes"]
    n = len(table.data)
    demand.num_modes = n
    if n < 1:
        raise AssertionError("demandModes must have at least 1 row")

    cols = table.columns
    demand.modes = [DemandMode() for _ in range(n)]

    for i in range(n):
        mode_index_cell = cols["modeIndex"][i]
        if mode_index_cell != i + 1:
            raise AssertionError(f"demandModes.modeIndex must be 1..n; got {mode_index_cell} at row {i}")

        dm = DemandMode()
        dm.index = i + 1  # keep 1-based indices if your data files are 1-based

        raster_index = int(cols["rasterIndex"][i])
        dm.raster_index = raster_index - 1
        if not (1 <= raster_index <= demand.num_rasters):
            raise AssertionError(f"rasterIndex out of range: {raster_index}")

        dm.raster = demand.rasters[raster_index - 1]

        # Julia: columns["priority"][i] |> Meta.parse |> eval
        # In Python, do NOT eval. Prefer parsing a string token into a Priority.
        dm.priority = parse_priority_token(str(cols["priority"][i]))

        dm.arrival_rate = float(cols["arrivalRate"][i])
        if dm.arrival_rate < 0:
            raise AssertionError("arrivalRate must be >= 0")

        # rasterMultiplier = arrivalRate / sum(raster.z)
        total = float(dm.raster.z.sum()) if hasattr(dm.raster.z, "sum") else float(sum(sum(row) for row in dm.raster.z))
        if total <= 0:
            # Julia would divide by 0 if sum(z)==0; better to guard explicitly.
            raise ValueError(f"Raster sum(z) must be > 0 to compute raster_multiplier (mode {dm.index})")
        dm.raster_multiplier = dm.arrival_rate / total

        demand.modes[i] = dm

    # -------------------------
    # demand sets
    # -------------------------
    table = tables["demandSets"]
    n = len(table.data)
    if n < 1:
        raise AssertionError("demandSets must have at least 1 row")

    cols = table.columns
    # Julia: numSets = maximum(columns["setIndex"])
    num_sets = int(max(cols["setIndex"]))
    demand.num_sets = num_sets

    # check all sets 1..num_sets appear
    used_sets = sorted(set(int(x) for x in cols["setIndex"]))
    if used_sets != list(range(1, num_sets + 1)):
        raise AssertionError("All sets from 1..num_sets must be used")

    num_priorities = len(PRIORITIES)

    # mode_lookup[set_idx][priority_idx] -> mode index (store 1-based mode indices like Julia)
    # Initialize with -1 (nullIndex equivalent)
    NULL_INDEX = -1
    demand.mode_lookup = [[NULL_INDEX for _ in range(num_priorities)] for _ in range(num_sets)]

    for i in range(n):
        set_index = int(cols["setIndex"][i])
        if set_index != i + 1:
            raise AssertionError(f"demandSets.setIndex must be 1..n in order; got {set_index} at row {i}")

        # Julia: modeIndices parsed+eval (probably a vector literal like [1,2,3])
        # In Python: assume parser returns either a list already, or a string like "[1,2]".
        mode_indices_cell = cols["modeIndices"][i]
        if isinstance(mode_indices_cell, str):
            mode_indices = ast.literal_eval(mode_indices_cell)
            #mode_indices = parse_int_list_literal(mode_indices_cell)  # implement safely (no eval)
        else:
            mode_indices = list(mode_indices_cell)

        for mode_index in mode_indices:
            mode_index = int(mode_index)
            if not (1 <= mode_index <= demand.num_modes):
                raise AssertionError(f"modeIndex out of range: {mode_index}")

            pr = demand.modes[mode_index - 1].priority
            # map Priority to column index 0..num_priorities-1
            try:
                pr_col = PRIORITIES.index(pr)
            except ValueError as e:
                raise AssertionError(f"Unknown priority {pr} in PRIORITIES") from e

            if demand.mode_lookup[set_index - 1][pr_col] != NULL_INDEX:
                raise AssertionError("demand.mode_lookup entry already filled")
            demand.mode_lookup[set_index - 1][pr_col] = mode_index
        print('demand set index', set_index, demand.mode_lookup[set_index - 1])


    # check fully filled
    if any(v == NULL_INDEX for row in demand.mode_lookup for v in row):
        raise AssertionError("demand.mode_lookup not fully filled (missing some priorities in some sets)")

    # -------------------------
    # demand sets timing
    # -------------------------
    table = tables["demandSetsTiming"]
    cols = table.columns

    demand.sets_start_times = [float(x) for x in cols["startTime"]]
    if not demand.sets_start_times:
        raise AssertionError("demandSetsTiming.startTime is empty")

    # Julia: first startTime > nullTime; here we just require > 0 unless you have a nullTime constant.
    if demand.sets_start_times[0] < 0:
        raise AssertionError("First startTime must be > nullTime (expected positive)")

    # Julia: issorted(..., lt=<=) i.e., nondecreasing
    if any(demand.sets_start_times[i] > demand.sets_start_times[i + 1] for i in range(len(demand.sets_start_times) - 1)):
        raise AssertionError("startTime must be nondecreasing")

    demand.sets_time_order = [int(x) for x in cols["setIndex"]]
    used_time_sets = sorted(set(demand.sets_time_order))
    if used_time_sets != list(range(1, num_sets + 1)):
        raise AssertionError("Each demand set should be used at least once in demandSetsTiming")


    demand.recent_sets_start_times_index = 0  
    return demand

def read_raster_file(raster_filename: str, *, agg_x: int = 100, agg_y: int = 100,
    agg_method: str = "mean", trim: bool = True,) -> Raster:
    """
    Read raster (band 1) and optionally aggregate/downsample to reduce grid size.

    agg_x, agg_y:
      - aggregation factors along x and y of the returned z (shape (nx, ny))
      - Example: agg_x=10, agg_y=10 reduces 1000x1000 -> 100x100

    agg_method:
      - 'sum' preserves total demand mass (recommended for demand rasters)
      - 'mean' preserves average intensity per cell

    trim:
      - if True, drop remainder cells that don't fit an exact block
      - if False, pad with zeros before aggregating
    """
    if not os.path.isfile(raster_filename):
        raise FileNotFoundError(raster_filename)

    ds = gdal.Open(raster_filename, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open raster: {raster_filename}")

    try:
        band = ds.GetRasterBand(1)  # first band only
        z_raw = band.ReadAsArray()  # GDAL returns shape (rows, cols) = (ny, nx)
        geo_transform = ds.GetGeoTransform()  # (x1, dxdi, dxdj, y1, dydi, dydj)
    finally:
        ds = None  # close

    # Unpack geotransform
    x1, dxdi, dxdj, y1, dydi, dydj = geo_transform

    # Data checks (same as Julia)
    if not (dxdj == 0 and dydi == 0):
        raise AssertionError("Raster is sloping/rotated: expected dxdj==0 and dydi==0")
    if dxdi <= 0:
        raise AssertionError("Expected dxdi > 0")
    if dydj == 0:
        raise AssertionError("Expected dydj != 0")

    dx = float(dxdi)       # cell width in x direction
    dy0 = float(dydj)      # cell height in y direction (may be negative)

    # Convert z to match Julia's convention z[i,j] with i over x (nx) and j over y (ny)
    # GDAL gives z_raw[j,i] ~ (y,x) so transpose to (x,y)
    z = np.asarray(z_raw).T  # shape becomes (nx, ny)

    nx, ny = z.shape

    # x centers are always increasing since dx>0
    x_min = x1 + 0.5 * dx

    # For y: if dy0 < 0, flip sign and reverse y-axis dimension (dims=2 in Julia => axis=1 in (nx,ny))


    if dy0 > 0:
        y_min = y1 + 0.5 * dy0
        dy = dy0
    else:
        # Julia: (yMin, dy, z) = (y1 + (ny - 0.5)*dy, -dy, reverse(z, dims=2))
        y_min = y1 + (ny - 0.5) * dy0
        dy = -dy0
        z = z[:, ::-1]  # reverse along y dimension


    # --- Aggregate if requested ---
    if agg_x != 1 or agg_y != 1:
        # IMPORTANT: if we are aggregating with SUM, total mass preserved automatically.
        z = _block_reduce_2d(z, fx=agg_x, fy=agg_y, method=agg_method, trim=trim)

        # Update grid spacing: each new cell spans agg_x/agg_y original cells
        dx = dx * agg_x
        dy = dy * agg_y

        nx, ny = z.shape


    x = x_min + dx * np.arange(nx)
    y = y_min + dy * np.arange(ny)

    return Raster(x=x, y=y, z=z)


def _block_reduce_2d(z: np.ndarray, fx: int, fy: int, method: str = "sum", trim: bool = True) -> np.ndarray:
    """
    Reduce z of shape (nx, ny) into (nx', ny') by aggregating blocks of (fx, fy).
    """
    if fx < 1 or fy < 1:
        raise ValueError("fx and fy must be >= 1")
    nx, ny = z.shape

    if trim:
        nx2 = (nx // fx) * fx
        ny2 = (ny // fy) * fy
        z = z[:nx2, :ny2]
    else:
        # pad with zeros to next multiple
        nx2 = int(np.ceil(nx / fx) * fx)
        ny2 = int(np.ceil(ny / fy) * fy)
        pad_x = nx2 - nx
        pad_y = ny2 - ny
        if pad_x or pad_y:
            z = np.pad(z, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=0.0)

    nx2, ny2 = z.shape
    z4 = z.reshape(nx2 // fx, fx, ny2 // fy, fy)

    if method == "sum":
        return z4.sum(axis=(1, 3))
    if method == "mean":
        return z4.mean(axis=(1, 3))
    raise ValueError("method must be 'sum' or 'mean'")