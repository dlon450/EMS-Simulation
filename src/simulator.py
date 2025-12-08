from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
from network import Network, Travel
from map import Map, Grid
from defs import EventForm, Priority
from entities import Ambulance, Call, Hospital, Station
from misc import File, XMLElement, Resimulation, Redispatch, MobilisationDelay, EventsFile
from demand import Demand, DemandCoverage
from stats import SimStats


@dataclass
class Event:
    index: Optional[int] = None
    parent_index: Optional[int] = None
    form: EventForm = EventForm.NULL
    time: Optional[float] = None
    amb_index: Optional[int] = None
    call_index: Optional[int] = None
    station_index: Optional[int] = None


@dataclass
class Simulation:
    start_time: Optional[float] = None
    time: Optional[float] = None
    end_time: Optional[float] = None

    net: Network = field(default_factory=Network)
    travel: Travel = field(default_factory=Travel)
    map: Map = field(default_factory=Map)
    grid: Grid = field(default_factory=Grid)

    ambulances: List[Ambulance] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    hospitals: List[Hospital] = field(default_factory=list)
    stations: List[Station] = field(default_factory=list)

    event_list: List[Event] = field(default_factory=list)
    event_index: int = 0
    queued_call_list: List[Call] = field(default_factory=list)

    resim: Resimulation = field(default_factory=Resimulation)
    mobilisation_delay: MobilisationDelay = field(default_factory=MobilisationDelay)

    add_call_to_queue: Callable[[Call], None] = lambda c: None
    find_amb_to_dispatch: Callable[[Call], Optional[int]] = lambda c: None
    redispatch: Redispatch = field(default_factory=Redispatch.default)

    demand: Demand = field(default_factory=Demand)
    demand_coverage: DemandCoverage = field(default_factory=DemandCoverage)
    response_travel_priorities: Dict[Priority, Priority] = field(default_factory=dict)
    target_response_durations: List[float] = field(default_factory=list)

    current_calls: Set[Call] = field(default_factory=set)
    previous_calls: Set[Call] = field(default_factory=set)
    stats: SimStats = field(default_factory=SimStats)

    reps: List["Simulation"] = field(default_factory=list)
    num_reps: int = 0
    is_runnable: bool = True

    input_path: str = ""
    output_path: str = ""
    input_files: Dict[str, File] = field(default_factory=dict)
    output_files: Dict[str, File] = field(default_factory=dict)
    events_file: EventsFile = field(default_factory=EventsFile)

    write_output: bool = False
    initialised: bool = False
    used: bool = False
    complete: bool = False
    animating: bool = False

    backup: Optional["Simulation"] = None
    config_root_elt: Optional[XMLElement] = None

    def reset(self) -> None:
        """Reset dynamic state so the simulation can be re-run."""
        self.time = self.start_time
        self.event_list.clear()
        self.event_index = 0
        self.queued_call_list.clear()
        self.current_calls.clear()
        self.previous_calls.clear()
        self.complete = False
        self.used = False

    def schedule_event(self, event: Event) -> None:
        """Add an event to the future event list, sorted by time."""
        if event.time is None:
            raise ValueError("Event.time must be set")
        self.event_list.append(event)
        self.event_list.sort(key=lambda e: e.time or 0.0)

    def next_event(self) -> Optional[Event]:
        """Pop the next event from the list."""
        if not self.event_list:
            self.complete = True
            return None
        event = self.event_list.pop(0)
        self.time = event.time
        self.event_index += 1
        return event
