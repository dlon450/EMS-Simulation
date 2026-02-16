from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING

from .defs import (
    AMB_STATUS_TO_SETS,
    AmbClass,
    AmbStatus,
    AmbStatusSet,
    CallStatus,
    Priority,
    is_going_to_station,
    is_travelling,
)
from .events import Event
from .geo import Location
from .map import Route

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import Simulation


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
    capacity: int = 0  # maximum number of ambulances the station can hold
    attributes: Dict[str, Any] = field(default_factory=dict)

    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None

    # num_idle_ambs_total_duration[k] = total time with k idle ambulances at station
    num_idle_ambs_total_duration: Dict[int, float] = field(default_factory=dict)
    current_num_idle_ambs: int = 0
    current_num_idle_ambs_set_time: Optional[float] = None

    def init_stats(self, *, start_time: float) -> None:
        # OffsetVector(Float[], 0) in reference, later resized to capacity+1.
        self.num_idle_ambs_total_duration = {k: 0.0 for k in range(0, max(0, self.capacity) + 1)}
        self.current_num_idle_ambs = 0
        self.current_num_idle_ambs_set_time = float(start_time)

    def update_stats(self, *, num_idle_ambs_change: int = 0, time: Optional[float] = None) -> None:
        """Update station idle-ambulance occupancy statistics.

        Mirrors reference's ``updateStationStats!``.
        """

        if num_idle_ambs_change == 0:
            return
        if time is None:
            raise ValueError("time is required when changing num_idle_ambs")
        if self.current_num_idle_ambs_set_time is None:
            self.current_num_idle_ambs_set_time = float(time)

        if time < self.current_num_idle_ambs_set_time:
            raise ValueError("station stats time moved backwards")

        prev_k = int(self.current_num_idle_ambs)
        dt = float(time - self.current_num_idle_ambs_set_time)
        self.num_idle_ambs_total_duration[prev_k] = self.num_idle_ambs_total_duration.get(prev_k, 0.0) + dt

        self.current_num_idle_ambs = int(self.current_num_idle_ambs + num_idle_ambs_change)
        self.current_num_idle_ambs_set_time = float(time)


@dataclass
class Call:
    index: Optional[int] = None
    priority: Optional[Priority] = None
    transport: bool = True  # true if requires transport to hospital
    hospital_index: Optional[int] = None  # pre-specified hospital, if any
    location: Location = field(default_factory=Location)
    recommended_amb_class: Optional[AmbClass] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # time/duration params
    arrival_time: Optional[float] = None
    dispatch_delay: float = 0.0
    on_scene_duration: float = 0.0
    handover_duration: float = 0.0

    # nearest node
    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None

    # state
    status: CallStatus = CallStatus.NULL
    amb_index: Optional[int] = None

    # for animation
    current_loc: Location = field(default_factory=Location)
    moved_loc: bool = False

    # statistics
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

    def set_status(self, status: CallStatus, time: float) -> None:
        """Set the call status and update derived statistics.

        Port of reference ``setCallStatus!``.
        """

        prev_status = self.status

        if self.status_set_time is not None:
            if time < self.status_set_time:
                raise ValueError("call status time moved backwards")

            status_duration = float(time - self.status_set_time)

            # stats for previous status
            if prev_status == CallStatus.QUEUED:
                self.queued_duration += status_duration
            elif prev_status == CallStatus.WAITING_FOR_AMB:
                self.waiting_for_amb_duration += status_duration
                # overwrite if multiple dispatch attempts
                self.amb_going_to_call_duration = status_duration
            elif prev_status == CallStatus.GOING_TO_HOSPITAL:
                if self.transport_duration != 0.0:
                    raise ValueError("transport_duration already set")
                self.transport_duration = status_duration

        # stats for new status
        if status == CallStatus.QUEUED:
            self.was_queued = True
        elif status == CallStatus.WAITING_FOR_AMB:
            if prev_status == CallStatus.SCREENING and self.arrival_time is not None:
                # screening -> waiting should only incur dispatch_delay
                # (allow small floating error)
                expected = float(self.arrival_time + self.dispatch_delay)
                if abs(time - expected) > 1e-9:
                    raise ValueError("dispatch time does not match arrival_time + dispatch_delay")
            self.dispatch_time = float(time)
        elif status == CallStatus.ON_SCENE_TREATMENT:
            if self.amb_arrival_time is not None:
                raise ValueError("amb_arrival_time already set")
            self.amb_arrival_time = float(time)
            if self.arrival_time is not None:
                self.response_duration = float(time - self.arrival_time)
            # bumped duration is total waiting minus final travel time
            self.bumped_duration = float(self.waiting_for_amb_duration - self.amb_going_to_call_duration)
        elif status == CallStatus.AT_HOSPITAL:
            self.hospital_arrival_time = float(time)
        elif status == CallStatus.PROCESSED:
            if self.service_duration != 0.0:
                raise ValueError("service_duration already set")
            if self.arrival_time is not None:
                self.service_duration = float(time - self.arrival_time)

        self.status = status
        self.status_set_time = float(time)


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

    # for animation
    current_loc: Location = field(default_factory=Location)
    moved_loc: bool = False
    dest_loc: Location = field(default_factory=Location)

    # statistics counters
    num_calls_treated: int = 0
    num_calls_transported: int = 0
    num_dispatches: int = 0
    num_dispatches_from_station: int = 0
    num_dispatches_while_mobilising: int = 0
    num_dispatches_on_road: int = 0
    num_dispatches_on_free: int = 0
    num_redispatches: int = 0

    # durations/distances by AmbStatus and AmbStatusSet
    status_durations: Dict[object, float] = field(default_factory=dict)
    status_distances: Dict[object, float] = field(default_factory=dict)
    status_transition_counts: Dict[Tuple[AmbStatus, AmbStatus], int] = field(default_factory=dict)

    status_set_time: Optional[float] = None
    prev_status: AmbStatus = AmbStatus.NULL

    # move-up statistics omitted for now

    def _add_status_duration(self, status: AmbStatus, duration: float) -> None:
        self.status_durations[status] = self.status_durations.get(status, 0.0) + float(duration)
        for sset in AMB_STATUS_TO_SETS.get(status, set()):
            self.status_durations[sset] = self.status_durations.get(sset, 0.0) + float(duration)

    def _add_status_distance(self, status: AmbStatus, dist: float) -> None:
        # only valid for travelling statuses
        self.status_distances[status] = self.status_distances.get(status, 0.0) + float(dist)
        for sset in AMB_STATUS_TO_SETS.get(status, set()):
            self.status_distances[sset] = self.status_distances.get(sset, 0.0) + float(dist)

    def set_status(self, sim: "Simulation", status: AmbStatus, time: float) -> None:
        """Set the ambulance status and update derived statistics.

        Port of reference ``setAmbStatus!`` (minus move-up bookkeeping).
        """

        if self.status_set_time is not None and time < self.status_set_time:
            raise ValueError("ambulance status time moved backwards")

        prev_status = self.status

        # stats - previous status duration (+ distance if travelling)
        if self.status_set_time is not None:
            status_duration = float(time - self.status_set_time)
            self._add_status_duration(prev_status, status_duration)

            if is_travelling(prev_status):
                dist = float(self.route.distance_travelled(sim, time))
                self._add_status_distance(prev_status, dist)

        self.status_set_time = float(time)
        self.prev_status = prev_status
        self.status = status

        # Finalisation pass: we only want to close out durations/distances.
        if sim.complete:
            if prev_status != status:
                raise ValueError("finalising with a status change is unexpected")
            return

        if prev_status is not AmbStatus.NULL:
            key = (prev_status, status)
            self.status_transition_counts[key] = self.status_transition_counts.get(key, 0) + 1

        # stats
        if status == AmbStatus.MOBILISING:
            self.num_dispatches += 1
            if prev_status in (AmbStatus.IDLE_AT_STATION, AmbStatus.IDLE_AT_CROSS_STREET):
                self.num_dispatches_from_station += 1
            elif prev_status == AmbStatus.MOBILISING:
                self.num_dispatches_while_mobilising += 1
                self.num_redispatches += 1
        elif status == AmbStatus.GOING_TO_CALL:
            if prev_status != AmbStatus.MOBILISING:
                self.num_dispatches += 1

            if prev_status in (AmbStatus.IDLE_AT_STATION, AmbStatus.IDLE_AT_CROSS_STREET):
                self.num_dispatches_from_station += 1
            elif prev_status == AmbStatus.MOBILISING:
                # already counted above for MOBILISING
                pass
            elif is_going_to_station(prev_status):
                self.num_dispatches_on_road += 1
            elif prev_status == AmbStatus.GOING_TO_CALL:
                self.num_dispatches_on_road += 1
                self.num_redispatches += 1
            elif prev_status == AmbStatus.FREE_AFTER_CALL:
                self.num_dispatches_on_free += 1
        elif status == AmbStatus.AT_CALL:
            self.num_calls_treated += 1
        elif status == AmbStatus.AT_HOSPITAL:
            self.num_calls_transported += 1
