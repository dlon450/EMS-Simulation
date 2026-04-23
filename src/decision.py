from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .defs import Priority, PRIORITIES, AmbStatus, is_free
from .entities import Ambulance, Call

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import Simulation


DEFAULT_COVERAGE_DISPATCH_RHO = 0.5


@dataclass(frozen=True)
class _CoverageDispatchCandidate:
    amb_index: int
    response_duration: float
    coverage_loss: float


def _priority_rank(p: Priority) -> int:
    """Smaller rank = higher priority (HIGH=1, MED=2, LOW=3)."""
    return int(p.value)


def add_call_to_queue_sort_priority_then_time(queue: List[Call], call: Call) -> None:
    """Insert *call* into *queue*.

    Mirrors reference's ``addCallToQueueSortPriorityThenTime!``:
    calls nearer to the end of the list are higher priority and arrived sooner.
    """

    if call.priority is None or call.arrival_time is None:
        raise ValueError("call.priority and call.arrival_time must be set to queue a call")

    i = len(queue)
    while i > 0:
        prev = queue[i - 1]
        if prev.priority is None or prev.arrival_time is None:
            break

        # Move earlier while the new call has *lower* priority (larger rank),
        # or equal priority but arrived later.
        if _priority_rank(call.priority) > _priority_rank(prev.priority):
            i -= 1
            continue
        if _priority_rank(call.priority) == _priority_rank(prev.priority) and call.arrival_time > prev.arrival_time:
            i -= 1
            continue
        break

    queue.insert(i, call)


def get_next_call(queue: List[Call]) -> Optional[Call]:
    return queue.pop() if queue else None


def is_amb_redispatchable(sim: "Simulation", amb: Ambulance, from_call: Call, to_call: Call) -> bool:
    if not sim.redispatch.allow:
        return False
    if from_call.priority is None or to_call.priority is None:
        return False
    return sim.redispatch.can_redispatch(from_call.priority, to_call.priority)


def is_amb_dispatchable(sim: "Simulation", amb: Ambulance, call: Call) -> bool:
    """Whether *amb* can be (re)assigned to *call* now.

    Port of reference ``isAmbDispatchable``.
    """

    if amb.end_current_tour:
        return False

    if is_free(amb.status):
        return True

    if amb.status in (AmbStatus.MOBILISING, AmbStatus.GOING_TO_CALL):
        if amb.call_index is None:
            return False
        from_call = sim.calls[amb.call_index]
        return is_amb_redispatchable(sim, amb, from_call, call)

    return False


def _estimate_mobilisation_start_time(sim: "Simulation", amb: Ambulance) -> float:
    """Dispatcher's estimate of when the ambulance will start moving."""

    t0 = float(sim.time or 0.0)
    if not sim.mobilisation_delay.use:
        return t0

    expected = float(sim.mobilisation_delay.expected_duration or 0.0)

    if amb.status in (AmbStatus.IDLE_AT_STATION, AmbStatus.IDLE_AT_CROSS_STREET):
        return t0 + expected
    if amb.status == AmbStatus.MOBILISING and amb.dispatch_time is not None:
        # Remaining expected duration.
        return t0 + max(0.0, expected - (t0 - float(amb.dispatch_time)))

    return t0


def _estimated_response_duration(sim: "Simulation", amb: Ambulance, call: Call) -> float:
    if call.nearest_node_index is None or call.nearest_node_dist is None:
        raise ValueError("call.nearest_node_index/dist must be set")

    mobilisation_time = _estimate_mobilisation_start_time(sim, amb)
    delay = mobilisation_time - float(sim.time or 0.0)

    # Determine travel mode based on the *response* travel priority mapping.
    if call.priority is None:
        raise ValueError("call.priority must be set")
    resp_priority = sim.response_travel_priorities.get(call.priority, call.priority)
    travel_mode_index = sim.travel.mode_index_for_priority(resp_priority, mobilisation_time)

    travel = amb.route.travel_time_to_location(
        sim,
        travel_mode_index=travel_mode_index,
        t=float(sim.time or 0.0),
        end_loc=call.location,
        end_fnode=int(call.nearest_node_index),
        end_fnode_dist=float(call.nearest_node_dist),
    )

    return float(delay + travel)


def find_nearest_dispatchable_amb(sim: "Simulation", call: Call, *, require_recommended_class: bool = False) -> Optional[int]:
    """Return the index of the best ambulance to respond to *call*.

    Port of reference ``findNearestDispatchableAmb!`` (and the ALS/BLS variant).
    """

    best_idx: Optional[int] = None
    best_dur = float("inf")

    for amb in sim.ambulances[1:]:
        if amb.index is None:
            continue
        if not is_amb_dispatchable(sim, amb, call):
            continue

        if require_recommended_class and call.recommended_amb_class is not None:
            if amb.amb_class != call.recommended_amb_class:
                continue

        dur = _estimated_response_duration(sim, amb, call)
        if dur < best_dur:
            best_dur = dur
            best_idx = int(amb.index)

    return best_idx


def _amb_contributes_to_demand_coverage(amb: Ambulance, *, num_stations: int) -> bool:
    """Whether an ambulance should be counted in station-based demand coverage."""

    if amb.station_index is None:
        return False
    station_idx = int(amb.station_index)
    if station_idx <= 0 or station_idx >= num_stations:
        return False
    return is_free(amb.status) and not amb.end_current_tour


def _station_demand_coverage_counts(sim: "Simulation") -> List[int]:
    """Count currently free coverage ambulances by assigned station index."""

    counts = [0] * len(sim.stations)
    num_stations = len(sim.stations)
    for amb in sim.ambulances[1:]:
        if _amb_contributes_to_demand_coverage(amb, num_stations=num_stations):
            counts[int(amb.station_index)] += 1
    return counts


def _demand_coverage_shortage(sim: "Simulation", stations_num_ambs: List[int]) -> float:
    """Total current demand not covered by any available station ambulance."""

    if not sim.demand_coverage.initialised:
        raise ValueError("demand coverage must be initialised before coverage dispatch")

    # Lazy import avoids a module cycle: init_dc imports simulator, which imports decision.
    from .init_dc import get_demand_mode, get_points_coverage_mode_mut

    current_time = float(sim.time or 0.0)
    shortage = 0.0

    for priority in PRIORITIES:
        pcm = get_points_coverage_mode_mut(sim, priority, current_time)
        demand_mode = get_demand_mode(sim.demand, priority, current_time)

        if pcm.index is None:
            raise AssertionError("PointsCoverageMode.index must be set")
        if demand_mode.raster_index is None:
            raise AssertionError("DemandMode.raster_index must be set")

        point_set_demands = sim.demand_coverage.point_sets_demands[pcm.index][demand_mode.raster_index]
        multiplier = float(demand_mode.raster_multiplier)

        for point_set_idx, station_set in enumerate(pcm.station_sets):
            is_covered = any(
                0 <= station_idx < len(stations_num_ambs) and stations_num_ambs[station_idx] > 0
                for station_idx in station_set
            )
            if not is_covered:
                shortage += float(point_set_demands[point_set_idx]) * multiplier

    return shortage


def _scaled_minimize_value(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def find_coverage_dispatchable_amb(
    sim: "Simulation",
    call: Call,
    *,
    require_recommended_class: bool = False,
    rho: float = DEFAULT_COVERAGE_DISPATCH_RHO,
) -> Optional[int]:
    """Return the ambulance minimizing rho * response + (1-rho) * coverage loss.

    Both objectives are min-max scaled over the dispatchable candidates for the
    current call. ``rho=1`` is nearest-response only; ``rho=0`` is coverage-loss
    only.
    """

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")

    candidates: List[_CoverageDispatchCandidate] = []
    station_counts = _station_demand_coverage_counts(sim)
    current_shortage = _demand_coverage_shortage(sim, station_counts)
    shortage_cache: Dict[Optional[int], float] = {}
    num_stations = len(sim.stations)

    for amb in sim.ambulances[1:]:
        if amb.index is None:
            continue
        if not is_amb_dispatchable(sim, amb, call):
            continue

        if require_recommended_class and call.recommended_amb_class is not None:
            if amb.amb_class != call.recommended_amb_class:
                continue

        station_idx: Optional[int] = None
        if _amb_contributes_to_demand_coverage(amb, num_stations=num_stations):
            station_idx = int(amb.station_index)

        if station_idx not in shortage_cache:
            candidate_station_counts = station_counts.copy()
            if station_idx is not None:
                candidate_station_counts[station_idx] = max(0, candidate_station_counts[station_idx] - 1)
            shortage_cache[station_idx] = _demand_coverage_shortage(sim, candidate_station_counts)

        shortage_after_dispatch = shortage_cache[station_idx]
        coverage_loss = max(0.0, shortage_after_dispatch - current_shortage)
        candidates.append(
            _CoverageDispatchCandidate(
                amb_index=int(amb.index),
                response_duration=_estimated_response_duration(sim, amb, call),
                coverage_loss=coverage_loss,
            )
        )

    if not candidates:
        return None

    min_response = min(c.response_duration for c in candidates)
    max_response = max(c.response_duration for c in candidates)
    min_coverage_loss = min(c.coverage_loss for c in candidates)
    max_coverage_loss = max(c.coverage_loss for c in candidates)

    best_idx: Optional[int] = None
    best_score: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

    for candidate in candidates:
        scaled_response = _scaled_minimize_value(candidate.response_duration, min_response, max_response)
        scaled_coverage_loss = _scaled_minimize_value(
            candidate.coverage_loss,
            min_coverage_loss,
            max_coverage_loss,
        )
        objective = rho * scaled_response + (1.0 - rho) * scaled_coverage_loss
        score = (objective, candidate.response_duration, candidate.coverage_loss)
        if score < best_score:
            best_score = score
            best_idx = candidate.amb_index

    return best_idx


def find_nearest_dispatchable_amb_als_bls(sim: "Simulation", call: Call) -> Optional[int]:
    return find_nearest_dispatchable_amb(sim, call, require_recommended_class=True)


def find_coverage_dispatchable_amb_als_bls(
    sim: "Simulation",
    call: Call,
    *,
    rho: float = DEFAULT_COVERAGE_DISPATCH_RHO,
) -> Optional[int]:
    return find_coverage_dispatchable_amb(sim, call, require_recommended_class=True, rho=rho)
