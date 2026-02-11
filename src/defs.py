from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Set, Tuple


class Priority(Enum):
    """Call or demand priority.

    Julia models priorities as small 1-based integers.  We keep :class:`Enum`
    values with the same property (``.value`` is 1..N) so they can index
    1-based tables read from the original input files.
    """

    HIGH = auto()
    MED = auto()
    LOW = auto()


PRIORITIES: Tuple[Priority, ...] = (Priority.HIGH, Priority.MED, Priority.LOW)


class AmbClass(Enum):
    """Ambulance clinical class."""

    ALS = auto()
    BLS = auto()


class EventForm(Enum):
    """Forms of events in the event list / trace."""

    NULL = auto()
    AMB_GOES_TO_SLEEP = auto()
    AMB_WAKES_UP = auto()
    CALL_ARRIVES = auto()
    CONSIDER_DISPATCH = auto()
    AMB_DISPATCHED = auto()
    AMB_MOBILISED = auto()
    AMB_REACHES_CALL = auto()
    AMB_GOES_TO_HOSPITAL = auto()
    AMB_REACHES_HOSPITAL = auto()
    AMB_BECOMES_FREE = auto()
    AMB_RETURNS_TO_STATION = auto()
    AMB_REACHES_STATION = auto()

    # Move-up is present in the Julia codebase but intentionally skipped in this port for now.
    CONSIDER_MOVE_UP = auto()

    # Cross-street / tour-window events (FDNY fork behaviour).
    AMB_RETURNS_TO_CROSS_STREET = auto()
    AMB_REACHES_CROSS_STREET = auto()
    AMB_BECOMES_INACTIVE = auto()
    AMB_BECOMES_ACTIVE = auto()


class AmbStatus(Enum):
    """Operational state of an ambulance."""

    NULL = auto()
    SLEEPING = auto()
    IDLE_AT_STATION = auto()
    IDLE_AT_CROSS_STREET = auto()
    MOBILISING = auto()
    GOING_TO_CALL = auto()
    AT_CALL = auto()
    GOING_TO_HOSPITAL = auto()
    AT_HOSPITAL = auto()
    FREE_AFTER_CALL = auto()
    RETURNING_TO_STATION = auto()
    MOVING_UP_TO_STATION = auto()
    INACTIVE = auto()
    RETURNING_TO_CROSS_STREET = auto()


class AmbStatusSet(Enum):
    """Logical groupings of ambulance statuses."""

    WORKING = auto()
    BUSY = auto()
    FREE = auto()
    TRAVELLING = auto()
    GOING_TO_STATION = auto()


def is_busy(status: AmbStatus) -> bool:
    return status in {
        AmbStatus.MOBILISING,
        AmbStatus.GOING_TO_CALL,
        AmbStatus.AT_CALL,
        AmbStatus.GOING_TO_HOSPITAL,
        AmbStatus.AT_HOSPITAL,
    }


def is_free(status: AmbStatus) -> bool:
    # Mirrors Julia's `isFree` which intentionally includes "going to station"
    # states: dispatch can interrupt those routes.
    return status in {
        AmbStatus.IDLE_AT_STATION,
        AmbStatus.IDLE_AT_CROSS_STREET,
        AmbStatus.FREE_AFTER_CALL,
        AmbStatus.RETURNING_TO_STATION,
        AmbStatus.MOVING_UP_TO_STATION,
        AmbStatus.RETURNING_TO_CROSS_STREET,
    }


def is_working(status: AmbStatus) -> bool:
    # Julia: `!in(s, (ambNullStatus, ambSleeping))`
    return status not in {AmbStatus.NULL, AmbStatus.SLEEPING}


def is_going_to_station(status: AmbStatus) -> bool:
    # Julia's definition includes RETURNING_TO_CROSS_STREET in the "going-to-station"
    # family so that an in-progress return can be cancelled by dispatch.
    return status in {
        AmbStatus.RETURNING_TO_STATION,
        AmbStatus.MOVING_UP_TO_STATION,
        AmbStatus.RETURNING_TO_CROSS_STREET,
    }


def is_travelling(status: AmbStatus) -> bool:
    return status in {AmbStatus.GOING_TO_CALL, AmbStatus.GOING_TO_HOSPITAL} or is_going_to_station(
        status
    )


AMB_STATUS_SETS: Dict[AmbStatusSet, Set[AmbStatus]] = {
    AmbStatusSet.WORKING: {s for s in AmbStatus if is_working(s)},
    AmbStatusSet.BUSY: {s for s in AmbStatus if is_busy(s)},
    AmbStatusSet.FREE: {s for s in AmbStatus if is_free(s)},
    AmbStatusSet.TRAVELLING: {s for s in AmbStatus if is_travelling(s)},
    AmbStatusSet.GOING_TO_STATION: {s for s in AmbStatus if is_going_to_station(s)},
}

# Reverse mapping: status -> sets containing it.
AMB_STATUS_TO_SETS: Dict[AmbStatus, Set[AmbStatusSet]] = {s: set() for s in AmbStatus}
for _set, members in AMB_STATUS_SETS.items():
    for _status in members:
        AMB_STATUS_TO_SETS[_status].add(_set)


class CallStatus(Enum):
    """Lifecycle state of a call."""

    NULL = auto()
    SCREENING = auto()
    QUEUED = auto()
    WAITING_FOR_AMB = auto()
    ON_SCENE_TREATMENT = auto()
    GOING_TO_HOSPITAL = auto()
    AT_HOSPITAL = auto()
    PROCESSED = auto()


class RouteStatus(Enum):
    """High-level position of a route progress."""

    NULL = auto()
    BEFORE_START_NODE = auto()
    ON_PATH = auto()
    AFTER_END_NODE = auto()
