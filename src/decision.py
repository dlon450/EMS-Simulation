from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .defs import Priority, PRIORITIES, AmbStatus, is_free
from .entities import Ambulance, Call

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import Simulation


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


def find_nearest_dispatchable_amb_als_bls(sim: "Simulation", call: Call) -> Optional[int]:
    return find_nearest_dispatchable_amb(sim, call, require_recommended_class=True)
