from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from defs import Event, EventForm, Priority, PRIORITIES
import io as io_


@dataclass
class DistrRng:
    pass


@dataclass
class XMLElement:
    pass


@dataclass
class Histogram:
    pass


@dataclass
class File:
    name: str = ""
    path: str = ""
    stream: io_.IOBase = field(default_factory=io_.StringIO)
    checksum: int = 0


@dataclass
class EventsFile:
    io: io_.IOBase = field(default_factory=io_.StringIO)
    event_filter: Dict[EventForm, bool] = field(
        default_factory=lambda: {e: True for e in EventForm}
    )


@dataclass
class Resimulation:
    use: bool = False
    time_tolerance: float = 0.0
    events: List[Event] = field(default_factory=list)
    events_children: List[List[Event]] = field(default_factory=list)
    prev_event_index: Optional[int] = None
    event_filter: Dict[EventForm, bool] = field(default_factory=dict)
    do_dispatch: bool = False
    do_move_up: bool = False


@dataclass
class MobilisationDelay:
    use: bool = False
    distr_rng: Optional[DistrRng] = None
    expected_duration: float = 0.0


@dataclass
class Redispatch:
    allow: bool = False
    # conditions[from_priority][to_priority] -> bool
    conditions: Dict[Priority, Dict[Priority, bool]] = field(
        default_factory=lambda: {p1: {p2: False for p2 in PRIORITIES} for p1 in PRIORITIES}
    )

    @classmethod
    def default(cls) -> "Redispatch":
        """Default redispatch policy: only to higher priority."""
        r = cls(allow=True)
        for p1 in PRIORITIES:
            for p2 in PRIORITIES:
                if p2 == Priority.HIGH and p1 != Priority.HIGH:
                    r.conditions[p1][p2] = True
        return r

    def can_redispatch(self, from_priority: Priority, to_priority: Priority) -> bool:
        if not self.allow:
            return False
        return self.conditions.get(from_priority, {}).get(to_priority, False)


Histogram = Any
Distribution = Any
Sampleable = Any