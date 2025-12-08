from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from geo import Location
from defs import AmbClass, AmbStatus, CallStatus, Event, Priority, Route, PRIORITIES


@dataclass
class Hospital:
    index: Optional[int] = None
    location: Location = field(default_factory=Location)
    attributes: Dict[str, Any] = field(default_factory=dict)
    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None
    num_calls: int = 0


@dataclass
class Station:
    index: Optional[int] = None
    location: Location = field(default_factory=Location)
    capacity: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None
    # num_idle_ambs_total_duration[k] = total time with k idle ambs
    num_idle_ambs_total_duration: Dict[int, float] = field(default_factory=dict)
    current_num_idle_ambs: int = 0
    current_num_idle_ambs_set_time: Optional[float] = None
    

@dataclass
class Call:
    index: Optional[int] = None
    priority: Optional[Priority] = None
    transport: bool = True
    hospital_index: Optional[int] = None
    location: Location = field(default_factory=Location)
    recommended_amb_class: Optional[AmbClass] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    arrival_time: Optional[float] = None
    dispatch_delay: float = 0.0
    on_scene_duration: float = 0.0
    handover_duration: float = 0.0

    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None

    status: CallStatus = CallStatus.NULL
    amb_index: Optional[int] = None

    current_loc: Location = field(default_factory=Location)
    moved_loc: bool = False

    dispatch_time: Optional[float] = None
    amb_arrival_time: Optional[float] = None
    hospital_arrival_time: Optional[float] = None
    num_bumps: int = 0
    was_queued: bool = False
    amb_dispatch_loc: Location = field(default_factory=Location)
    amb_status_before_dispatch: Optional[AmbStatus] = None
    chosen_hospital_index: Optional[int] = None

    queued_duration: float = 0.0
    bumped_duration: float = 0.0
    waiting_for_amb_duration: float = 0.0
    response_duration: float = 0.0
    amb_going_to_call_duration: float = 0.0
    transport_duration: float = 0.0
    service_duration: float = 0.0

    status_set_time: Optional[float] = None

    def set_status(self, new_status: CallStatus, t: float) -> None:
        """Update status and accumulate durations for coarse stats."""
        if self.status_set_time is not None:
            dt = max(0.0, t - self.status_set_time)
            if self.status == CallStatus.QUEUED:
                self.queued_duration += dt
            elif self.status == CallStatus.WAITING_FOR_AMB:
                self.waiting_for_amb_duration += dt
            elif self.status == CallStatus.ON_SCENE_TREATMENT:
                self.on_scene_duration += dt
            elif self.status == CallStatus.GOING_TO_HOSPITAL:
                self.transport_duration += dt
        self.status = new_status
        self.status_set_time = t

    def total_time_in_system(self) -> Optional[float]:
        if self.arrival_time is None or self.hospital_arrival_time is None:
            return None
        return self.hospital_arrival_time - self.arrival_time


@dataclass
class Ambulance:
    index: Optional[int] = None
    status: AmbStatus = AmbStatus.NULL
    station_index: Optional[int] = None
    call_index: Optional[int] = None
    route: Route = field(default_factory=Route)
    event: Event = field(default_factory=Event)
    amb_class: Optional[AmbClass] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    unit_name: str = ""
    cross_street_location: Location = field(default_factory=Location)
    cross_street_node_index: Optional[int] = None
    cross_street_node_dist: Optional[float] = None
    tour1_start: Optional[float] = None
    tour1_end: Optional[float] = None
    tour2_start: Optional[float] = None
    tour2_end: Optional[float] = None
    tour3_start: Optional[float] = None
    tour3_end: Optional[float] = None
    end_current_tour: bool = False
    start_next_tour: bool = False

    dispatch_time: Optional[float] = None
    mobilisation_time: Optional[float] = None

    current_loc: Location = field(default_factory=Location)
    moved_loc: bool = False
    dest_loc: Location = field(default_factory=Location)

    num_calls_treated: int = 0
    num_calls_transported: int = 0
    num_dispatches: int = 0
    num_dispatches_from_station: int = 0
    num_dispatches_while_mobilising: int = 0
    num_dispatches_on_road: int = 0
    num_dispatches_on_free: int = 0
    num_redispatches: int = 0

    status_durations: Dict[AmbStatus, float] = field(default_factory=dict)
    status_distances: Dict[AmbStatus, float] = field(default_factory=dict)
    status_transition_counts: Dict[Tuple[AmbStatus, AmbStatus], int] = field(default_factory=dict)

    status_set_time: Optional[float] = None
    prev_status: AmbStatus = AmbStatus.NULL

    dispatch_start_loc_counts: Dict[Tuple[float, float], int] = field(default_factory=dict)

    def is_available(self) -> bool:
        """True if ambulance is free to take a new call (simple heuristic)."""
        return self.status in {
            AmbStatus.IDLE_AT_STATION,
            AmbStatus.IDLE_AT_CROSS_STREET,
            AmbStatus.FREE_AFTER_CALL,
        }

    def set_status(self, new_status: AmbStatus, t: float) -> None:
        """Update status and accumulate per‑status durations."""
        if self.status_set_time is not None:
            dt = max(0.0, t - self.status_set_time)
            self.status_durations[self.status] = self.status_durations.get(self.status, 0.0) + dt
        self.prev_status = self.status
        self.status = new_status
        self.status_set_time = t
        if self.prev_status is not AmbStatus.NULL:
            key = (self.prev_status, self.status)
            self.status_transition_counts[key] = self.status_transition_counts.get(key, 0) + 1

    def record_distance(self, distance: float) -> None:
        """Accumulate distance travelled in current status."""
        self.status_distances[self.status] = self.status_distances.get(self.status, 0.0) + max(0.0, distance)
