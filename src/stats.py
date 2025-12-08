from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
from defs import AmbStatus, Priority
from misc import Histogram
import math


@dataclass
class MeanAndHalfWidth:
    mean: float = 0.0
    half_width: float = 0.0


@dataclass
class AmbulanceStats:
    amb_index: Optional[int] = None
    num_calls_treated: int = 0
    num_calls_transported: int = 0
    num_dispatches: int = 0
    num_dispatches_from_station: int = 0
    num_dispatches_while_mobilising: int = 0
    num_dispatches_on_road: int = 0
    num_dispatches_on_free: int = 0
    num_redispatches: int = 0
    num_move_ups: int = 0
    num_move_ups_from_station: int = 0
    num_move_ups_on_road: int = 0
    num_move_ups_on_free: int = 0
    num_move_ups_return_to_prev_station: int = 0
    status_durations: Dict[AmbStatus, float] = field(default_factory=dict)
    status_distances: Dict[AmbStatus, float] = field(default_factory=dict)
    status_transition_counts: Dict[Tuple[AmbStatus, AmbStatus], int] = field(default_factory=dict)


@dataclass
class CallStats:
    call_index: Optional[int] = None
    num_calls: int = 0
    num_queued: int = 0
    num_bumped: int = 0
    num_bumps: int = 0
    num_transports: int = 0
    num_responses_in_time: int = 0
    total_dispatch_delay: float = 0.0
    total_on_scene_duration: float = 0.0
    total_handover_duration: float = 0.0
    total_queued_duration: float = 0.0
    total_bumped_duration: float = 0.0
    total_waiting_for_amb_duration: float = 0.0
    total_response_duration: float = 0.0
    total_amb_going_to_call_duration: float = 0.0
    total_transport_duration: float = 0.0
    total_service_duration: float = 0.0
    response_duration_hist: Optional[Histogram] = None


@dataclass
class HospitalStats:
    hospital_index: Optional[int] = None
    num_calls: int = 0


@dataclass
class StationStats:
    station_index: Optional[int] = None
    num_idle_ambs_total_duration: Dict[int, float] = field(default_factory=dict)


@dataclass
class SimPeriodStats:
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    ambulances: List[AmbulanceStats] = field(default_factory=list)
    hospitals: List[HospitalStats] = field(default_factory=list)
    stations: List[StationStats] = field(default_factory=list)
    ambulance: AmbulanceStats = field(default_factory=AmbulanceStats)
    call: CallStats = field(default_factory=CallStats)
    call_priorities: Dict[Priority, CallStats] = field(default_factory=dict)
    hospital: HospitalStats = field(default_factory=HospitalStats)
    station: StationStats = field(default_factory=StationStats)

    def compute_duration(self) -> None:
        if self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time


@dataclass
class SimStats:
    captures: List[SimPeriodStats] = field(default_factory=list)
    periods: List[SimPeriodStats] = field(default_factory=list)
    do_capture: bool = True
    warm_up_duration: float = 0.0
    period_durations_iter: Iterable[float] = field(default_factory=tuple)
    next_capture_time: float = math.inf
    sim_start_time: Optional[float] = None
    sim_end_time: Optional[float] = None
    warm_up_end_time: Optional[float] = None
    last_call_arrival_time: Optional[float] = None
    record_dispatch_start_loc_counts: bool = False
    record_move_up_start_loc_counts: bool = False
    record_response_duration_hist: bool = False
    response_duration_hist_bin_width: float = 0.0
    response_duration_hist_num_bins: int = 0

    def start(self, t0: float) -> None:
        self.sim_start_time = t0
        self.warm_up_end_time = t0 + self.warm_up_duration
        it = iter(self.period_durations_iter)
        self.next_capture_time = t0 + (next(it, math.inf))
        # we keep the iterator as-is; user can manage it externally

    def maybe_capture(self, t: float) -> None:
        """Hook point to trigger statistics capture at time t."""
        if not self.do_capture or self.sim_start_time is None:
            return
        if t >= self.next_capture_time:
            # Actual capture logic is left to calling code
            pass
