from enum import Enum, auto
from typing import Dict, Set, Tuple


class Priority(Enum):
    """Call or demand priority."""

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
    CONSIDER_MOVE_UP = auto()
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


AMB_STATUS_SETS: Dict[AmbStatusSet, Set[AmbStatus]] = {
    AmbStatusSet.WORKING: {
        AmbStatus.GOING_TO_CALL,
        AmbStatus.AT_CALL,
        AmbStatus.GOING_TO_HOSPITAL,
        AmbStatus.AT_HOSPITAL,
        AmbStatus.FREE_AFTER_CALL,
    },
    AmbStatusSet.BUSY: {
        AmbStatus.GOING_TO_CALL,
        AmbStatus.AT_CALL,
        AmbStatus.GOING_TO_HOSPITAL,
        AmbStatus.AT_HOSPITAL,
    },
    AmbStatusSet.FREE: {
        AmbStatus.IDLE_AT_STATION,
        AmbStatus.IDLE_AT_CROSS_STREET,
        AmbStatus.FREE_AFTER_CALL,
    },
    AmbStatusSet.TRAVELLING: {
        AmbStatus.GOING_TO_CALL,
        AmbStatus.GOING_TO_HOSPITAL,
        AmbStatus.RETURNING_TO_STATION,
        AmbStatus.RETURNING_TO_CROSS_STREET,
    },
    AmbStatusSet.GOING_TO_STATION: {
        AmbStatus.RETURNING_TO_STATION,
        AmbStatus.MOVING_UP_TO_STATION,
    },
}


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
