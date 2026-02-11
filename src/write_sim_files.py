from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

import io
import csv
import json
import os

from .defs import AmbClass, AmbStatus, AmbStatusSet, EventForm
from .file_io import DELIMITER, NEWLINE, Table

if TYPE_CHECKING:  # pragma: no cover
    from .events import Event
    from .simulator import Simulation
    from .entities import Ambulance, Call, Hospital


# ---------------------------------------------------------------------------
# Julia-compatible sentinels
# ---------------------------------------------------------------------------

NULL_INDEX: int = -1
NULL_TIME: float = -1.0
NULL_X: float = -1.0
NULL_Y: float = -1.0


def _json(obj: object) -> str:
    """Compact JSON, stable for diffs."""

    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------------------
# Name/key mappings (Python enums -> Julia strings/ints)
# ---------------------------------------------------------------------------


_EVENTFORM_JULIA: Dict[EventForm, tuple[int, str]] = {
    EventForm.NULL: (0, "nullEvent"),
    EventForm.AMB_GOES_TO_SLEEP: (1, "ambGoesToSleep"),
    EventForm.AMB_WAKES_UP: (2, "ambWakesUp"),
    EventForm.CALL_ARRIVES: (3, "callArrives"),
    EventForm.CONSIDER_DISPATCH: (4, "considerDispatch"),
    EventForm.AMB_DISPATCHED: (5, "ambDispatched"),
    EventForm.AMB_MOBILISED: (6, "ambMobilised"),
    EventForm.AMB_REACHES_CALL: (7, "ambReachesCall"),
    EventForm.AMB_GOES_TO_HOSPITAL: (8, "ambGoesToHospital"),
    EventForm.AMB_REACHES_HOSPITAL: (9, "ambReachesHospital"),
    EventForm.AMB_BECOMES_FREE: (10, "ambBecomesFree"),
    EventForm.AMB_RETURNS_TO_STATION: (11, "ambReturnsToStation"),
    EventForm.AMB_REACHES_STATION: (12, "ambReachesStation"),
    EventForm.CONSIDER_MOVE_UP: (13, "considerMoveUp"),
    # Julia has ambMoveUpToStation=14; intentionally skipped in this port.
    EventForm.AMB_RETURNS_TO_CROSS_STREET: (15, "ambReturnsToCrossStreet"),
    EventForm.AMB_REACHES_CROSS_STREET: (16, "ambReachesCrossStreet"),
    EventForm.AMB_BECOMES_INACTIVE: (17, "ambBecomesInactive"),
    EventForm.AMB_BECOMES_ACTIVE: (18, "ambBecomesActive"),
}


def julia_event_key(form: EventForm) -> int:
    return _EVENTFORM_JULIA.get(form, (int(form.value) - 1, form.name))[0]


def julia_event_name(form: EventForm) -> str:
    return _EVENTFORM_JULIA.get(form, (int(form.value) - 1, form.name))[1]


_AMBSTATUS_JULIA_NAME: Dict[AmbStatus, str] = {
    AmbStatus.NULL: "ambNullStatus",
    AmbStatus.SLEEPING: "ambSleeping",
    AmbStatus.IDLE_AT_STATION: "ambIdleAtStation",
    AmbStatus.IDLE_AT_CROSS_STREET: "ambIdleAtCrossStreet",
    AmbStatus.MOBILISING: "ambMobilising",
    AmbStatus.GOING_TO_CALL: "ambGoingToCall",
    AmbStatus.AT_CALL: "ambAtCall",
    AmbStatus.GOING_TO_HOSPITAL: "ambGoingToHospital",
    AmbStatus.AT_HOSPITAL: "ambAtHospital",
    AmbStatus.FREE_AFTER_CALL: "ambFreeAfterCall",
    AmbStatus.RETURNING_TO_STATION: "ambReturningToStation",
    AmbStatus.MOVING_UP_TO_STATION: "ambMovingUpToStation",
    AmbStatus.INACTIVE: "ambInactive",
    AmbStatus.RETURNING_TO_CROSS_STREET: "ambReturningToCrossStreet",
}

_AMBSTATUSSET_JULIA_NAME: Dict[AmbStatusSet, str] = {
    AmbStatusSet.WORKING: "ambWorking",
    AmbStatusSet.BUSY: "ambBusy",
    AmbStatusSet.FREE: "ambFree",
    AmbStatusSet.TRAVELLING: "ambTravelling",
    AmbStatusSet.GOING_TO_STATION: "ambGoingToStation",
}


def julia_amb_status_name(status: Optional[AmbStatus]) -> str:
    if status is None:
        return _AMBSTATUS_JULIA_NAME[AmbStatus.NULL]
    return _AMBSTATUS_JULIA_NAME.get(status, status.name)


def julia_amb_status_set_name(sset: AmbStatusSet) -> str:
    return _AMBSTATUSSET_JULIA_NAME.get(sset, sset.name)


# Canonical status order used by the Julia writers.
_AMB_STATUS_ORDER: List[AmbStatus] = [
    AmbStatus.SLEEPING,
    AmbStatus.IDLE_AT_STATION,
    AmbStatus.IDLE_AT_CROSS_STREET,
    AmbStatus.MOBILISING,
    AmbStatus.GOING_TO_CALL,
    AmbStatus.AT_CALL,
    AmbStatus.GOING_TO_HOSPITAL,
    AmbStatus.AT_HOSPITAL,
    AmbStatus.FREE_AFTER_CALL,
    AmbStatus.RETURNING_TO_STATION,
    AmbStatus.MOVING_UP_TO_STATION,
    AmbStatus.INACTIVE,
    AmbStatus.RETURNING_TO_CROSS_STREET,
]

_AMB_STATUS_SET_ORDER: List[AmbStatusSet] = [
    AmbStatusSet.WORKING,
    AmbStatusSet.BUSY,
    AmbStatusSet.FREE,
    AmbStatusSet.TRAVELLING,
    AmbStatusSet.GOING_TO_STATION,
]

# Sorted travel-status order used in Julia's stats writer.
_AMB_TRAVEL_STATUS_ORDER: List[AmbStatus] = [
    AmbStatus.GOING_TO_CALL,
    AmbStatus.GOING_TO_HOSPITAL,
    AmbStatus.RETURNING_TO_STATION,
    AmbStatus.MOVING_UP_TO_STATION,
    AmbStatus.RETURNING_TO_CROSS_STREET,
]


# ---------------------------------------------------------------------------
# Low-level stream writers (Julia-style delimiter + newlines)
# ---------------------------------------------------------------------------


def write_dlm_line(f: IO[str], *args: object, delim: str = DELIMITER, newline: str = NEWLINE) -> None:
    """Write a delimiter-separated line with a *trailing* delimiter."""

    for a in args:
        f.write(str(a) + delim)
    f.write(newline)


def write_table_to_stream(
    f: IO[str],
    table: Table,
    *,
    delim: str = DELIMITER,
    newline: str = NEWLINE,
    write_num_rows: bool = False,
    write_num_cols: bool = False,
) -> None:
    # Header line (with trailing delimiter)
    write_dlm_line(
        f,
        table.name,
        str(table.num_rows) if write_num_rows else "",
        str(table.num_cols) if write_num_cols else "",
        delim=delim,
        newline=newline,
    )
    # Column headers (with trailing delimiter)
    write_dlm_line(f, *table.header, delim=delim, newline=newline)
    # Data rows: use csv.writer so strings with commas (e.g. JSON) are quoted.
    writer = csv.writer(
        f,
        delimiter=delim,
        lineterminator=newline,
        quoting=csv.QUOTE_MINIMAL,
    )
    for row in table.data:
        writer.writerow(["" if v is None else v for v in row])
    # Blank separator row
    f.write(delim + newline)


def write_tables_to_stream(
    f: IO[str],
    tables: Iterable[Table] | Mapping[str, Table] | Table,
    *,
    delim: str = DELIMITER,
    newline: str = NEWLINE,
) -> None:
    if isinstance(tables, Table):
        table_list: List[Table] = [tables]
    elif isinstance(tables, Mapping):
        table_list = list(tables.values())
    else:
        table_list = list(tables)

    for t in table_list:
        write_table_to_stream(f, t, delim=delim, newline=newline)


# ---------------------------------------------------------------------------
# Events output
# ---------------------------------------------------------------------------


def open_output_files(sim: "Simulation") -> None:
    """Open output files needed during simulation.

    Currently only the events file is streamed during execution.
    """

    if not sim.write_output:
        return

    if "events" not in sim.output_files:
        return

    # Create output path if needed.
    if sim.output_path:
        os.makedirs(sim.output_path, exist_ok=True)

    if sim.resim.use:
        # Resimulation reuses an existing events file; not implemented here.
        return

    path = sim.output_files["events"].path
    f = open(path, "w", newline="")
    sim.output_files["events"].stream = f
    sim.events_file.io = f

    # Save checksum of input files (Julia sorts by key)
    input_names = sorted(sim.input_files.keys())
    checksum_strings = [f"'{sim.input_files[n].checksum}'" for n in input_names]
    input_tbl = Table(name="inputFiles", header=["name", "checksum"], data=list(map(list, zip(input_names, checksum_strings))))

    # Save event dictionary (key/name/filter)
    forms = list(EventForm)
    keys = [julia_event_key(e) for e in forms]
    names = [julia_event_name(e) for e in forms]
    filters = [1 if sim.events_file.event_filter.get(e, True) else 0 for e in forms]
    event_tbl = Table(name="eventDict", header=["key", "name", "filter"], data=[[k, n, flt] for k, n, flt in zip(keys, names, filters)])

    write_tables_to_stream(f, [input_tbl, event_tbl])

    # Write the streaming events table header.
    write_dlm_line(f, "events")
    write_dlm_line(f, "index", "parentIndex", "time", "eventKey", "ambIndex", "callIndex", "stationIndex")


def write_event_to_file(sim: "Simulation", event: "Event") -> None:
    if not sim.write_output or sim.resim.use:
        return
    if not sim.events_file.event_filter.get(event.form, True):
        return
    f = sim.events_file.io
    if isinstance(f, io.StringIO):
        # Not opened.
        return

    parent = int(event.parent_index) if event.parent_index is not None else NULL_INDEX
    amb = int(event.amb_index) if event.amb_index is not None else NULL_INDEX
    call = int(event.call_index) if event.call_index is not None else NULL_INDEX
    st = int(event.station_index) if event.station_index is not None else NULL_INDEX
    t_str = f"{float(event.time or 0.0):0.5f}"
    write_dlm_line(f, int(event.index or 0), parent, t_str, julia_event_key(event.form), amb, call, st)


def close_output_files(sim: "Simulation") -> None:
    if not sim.write_output:
        return
    if sim.resim.use:
        return

    f = sim.events_file.io
    if not isinstance(f, io.StringIO):
        write_dlm_line(f, "end")
        f.close()
        sim.events_file.io = io.StringIO()


# ---------------------------------------------------------------------------
# Misc output files (written after simulation)
# ---------------------------------------------------------------------------


def _int_or_null(x: Optional[int]) -> int:
    return int(x) if x is not None else NULL_INDEX


def _float_or_null_time(x: Optional[float]) -> float:
    return float(x) if x is not None else float(NULL_TIME)


def _float_or_null_x(x: Optional[float]) -> float:
    return float(x) if x is not None else float(NULL_X)


def _float_or_null_y(y: Optional[float]) -> float:
    return float(y) if y is not None else float(NULL_Y)


def write_ambulances_file(filename: str, ambulances: Sequence["Ambulance"], *, write_output_fields: bool = False) -> None:
    header: List[str] = ["index", "stationIndex", "class", "attributes"]

    count_headers = [
        "numCallsTreated",
        "numCallsTransported",
        "numDispatches",
        "numDispatchesFromStation",
        "numDispatchesWhileMobilising",
        "numDispatchesOnRoad",
        "numDispatchesOnFree",
        "numRedispatches",
        # move-up counts (kept for Julia-format parity)
        "numMoveUps",
        "numMoveUpsFromStation",
        "numMoveUpsOnRoad",
        "numMoveUpsOnFree",
        "numMoveUpsReturnToPrevStation",
    ]

    statuses = [julia_amb_status_name(s) for s in _AMB_STATUS_ORDER] + [julia_amb_status_set_name(s) for s in _AMB_STATUS_SET_ORDER]
    status_duration_headers = [f"statusDurations_{s}" for s in statuses]

    travel_statuses = [julia_amb_status_name(s) for s in _AMB_TRAVEL_STATUS_ORDER] + [julia_amb_status_set_name(s) for s in _AMB_STATUS_SET_ORDER]
    status_distance_headers = [f"statusDistances_{s}" for s in travel_statuses]

    if write_output_fields:
        header = header + count_headers + status_duration_headers + status_distance_headers

    rows: List[List[object]] = []
    for a in ambulances[1:]:
        cls = 0
        if a.amb_class is not None:
            cls = 1 if a.amb_class == AmbClass.ALS else 2

        base = [
            _int_or_null(a.index),
            _int_or_null(a.station_index),
            cls,
            _json(a.attributes),
        ]

        if not write_output_fields:
            rows.append(base)
            continue

        counts = [
            int(getattr(a, "num_calls_treated", 0)),
            int(getattr(a, "num_calls_transported", 0)),
            int(getattr(a, "num_dispatches", 0)),
            int(getattr(a, "num_dispatches_from_station", 0)),
            int(getattr(a, "num_dispatches_while_mobilising", 0)),
            int(getattr(a, "num_dispatches_on_road", 0)),
            int(getattr(a, "num_dispatches_on_free", 0)),
            int(getattr(a, "num_redispatches", 0)),
            0,
            0,
            0,
            0,
            0,
        ]

        # durations/distances keyed by enum objects
        dur_keys: List[object] = list(_AMB_STATUS_ORDER) + list(_AMB_STATUS_SET_ORDER)
        dist_keys: List[object] = list(_AMB_TRAVEL_STATUS_ORDER) + list(_AMB_STATUS_SET_ORDER)
        durations = [float(a.status_durations.get(k, 0.0)) for k in dur_keys]
        distances = [float(a.status_distances.get(k, 0.0)) for k in dist_keys]

        rows.append(base + counts + durations + distances)

    from .file_io import write_tables_to_file

    write_tables_to_file(filename, Table(name="ambulances", header=header, data=rows))


def write_calls_file(
    filename: str,
    *,
    start_time: float,
    calls: Sequence["Call"],
    write_output_fields: bool = False,
) -> None:
    misc = Table(name="miscData", header=["startTime"], data=[[float(start_time)]])

    header: List[str] = [
        "index",
        "priority",
        "x",
        "y",
        "arrivalTime",
        "dispatchDelay",
        "onSceneDuration",
        "transport",
        "hospitalIndex",
        "handoverDuration",
        "attributes",
    ]

    if write_output_fields:
        header += [
            "dispatchTime",
            "ambArrivalTime",
            "hospitalArrivalTime",
            "numBumps",
            "wasQueued",
            "ambDispatchLoc.x",
            "ambDispatchLoc.y",
            "ambStatusBeforeDispatch",
            "chosenHospitalIndex",
            "queuedDuration",
            "bumpedDuration",
            "waitingForAmbDuration",
            "responseDuration",
            "ambGoingToCallDuration",
            "transportDuration",
            "serviceDuration",
        ]

    rows: List[List[object]] = []
    for c in calls[1:]:
        pr = int(c.priority.value) if c.priority is not None else 0
        base = [
            _int_or_null(c.index),
            pr,
            _float_or_null_x(c.location.x),
            _float_or_null_y(c.location.y),
            _float_or_null_time(c.arrival_time),
            float(c.dispatch_delay or 0.0),
            float(c.on_scene_duration or 0.0),
            1 if bool(c.transport) else 0,
            _int_or_null(c.hospital_index),
            float(c.handover_duration or 0.0),
            _json(c.attributes),
        ]

        if not write_output_fields:
            rows.append(base)
            continue

        extra = [
            _float_or_null_time(c.dispatch_time),
            _float_or_null_time(c.amb_arrival_time),
            _float_or_null_time(c.hospital_arrival_time),
            int(c.num_bumps or 0),
            1 if bool(c.was_queued) else 0,
            _float_or_null_x(c.amb_dispatch_loc.x),
            _float_or_null_y(c.amb_dispatch_loc.y),
            julia_amb_status_name(c.amb_status_before_dispatch),
            _int_or_null(c.chosen_hospital_index),
            float(c.queued_duration or 0.0),
            float(c.bumped_duration or 0.0),
            float(c.waiting_for_amb_duration or 0.0),
            float(c.response_duration or 0.0),
            float(c.amb_going_to_call_duration or 0.0),
            float(c.transport_duration or 0.0),
            float(c.service_duration or 0.0),
        ]
        rows.append(base + extra)

    from .file_io import write_tables_to_file

    write_tables_to_file(filename, [misc, Table(name="calls", header=header, data=rows)])


def write_hospitals_file(
    filename: str, hospitals: Sequence["Hospital"], *, write_output_fields: bool = False
) -> None:
    header: List[str] = ["index", "x", "y", "attributes"]
    if write_output_fields:
        header += ["numCalls"]

    rows: List[List[object]] = []
    for h in hospitals[1:]:
        row: List[object] = [
            _int_or_null(h.index),
            _float_or_null_x(h.location.x),
            _float_or_null_y(h.location.y),
            _json(h.attributes),
        ]
        if write_output_fields:
            row.append(int(getattr(h, "num_calls", 0)))
        rows.append(row)

    from .file_io import write_tables_to_file

    write_tables_to_file(filename, Table(name="hospitals", header=header, data=rows))


def write_misc_output_files(sim: "Simulation") -> None:
    if not sim.write_output:
        return

    out = sim.output_files
    if "ambulances" in out:
        write_ambulances_file(out["ambulances"].path, sim.ambulances, write_output_fields=True)
    if "calls" in out:
        if sim.start_time is None:
            raise ValueError("sim.start_time must be set to write calls output")
        write_calls_file(out["calls"].path, start_time=float(sim.start_time), calls=sim.calls, write_output_fields=True)
    if "hospitals" in out:
        write_hospitals_file(out["hospitals"].path, sim.hospitals, write_output_fields=True)


def write_output_files(sim: "Simulation") -> None:
    """Write all configured outputs that are produced *after* simulation."""

    write_misc_output_files(sim)
