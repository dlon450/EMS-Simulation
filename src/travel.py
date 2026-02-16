from typing import Optional

from .defs import Priority
from .network import TravelMode
from .simulator import Simulation   # adjust if Simulation is defined elsewhere


# -------------------------------------------------
# Get travel mode for given time and priority
# current_time is sim.time
# start_time is time that travel starts (>= current_time)
# Mutates: travel.recent_sets_start_times_index
# -------------------------------------------------

def get_travel_mode(
    travel,
    priority: Priority,
    current_time: float,
    *,
    start_time: Optional[float] = None,
) -> TravelMode:

    if start_time is None:
        start_time = current_time

    if current_time is None:
        raise AssertionError("current_time cannot be None")
    if priority is None:
        raise AssertionError("priority cannot be None")
    if current_time > start_time:
        raise AssertionError("current_time must be <= start_time")

    travel.recent_sets_start_times_index = get_travel_sets_start_times_index(
        travel, current_time
    )

    i = get_travel_sets_start_times_index(travel, start_time)

    if i < 0 or i >= len(travel.sets_time_order):
        raise AssertionError("Invalid sets_time_order index")

    travel_set_index = travel.sets_time_order[i]

    pr_idx = int(priority.value)
    pr_col = pr_idx - 1 if pr_idx >= 1 else pr_idx

    travel_mode_index = travel.mode_lookup[travel_set_index][pr_col]

    return travel.modes[travel_mode_index]


# -------------------------------------------------
# Get travel sets_start_times index for given time
# -------------------------------------------------

def get_travel_sets_start_times_index(travel, start_time: float) -> int:
    sets_start_times = travel.sets_start_times
    i = travel.recent_sets_start_times_index
    n = len(sets_start_times)

    if n == 0:
        raise AssertionError("travel.sets_start_times is empty")

    if not (0 <= i < n):
        raise AssertionError("recent_sets_start_times_index out of bounds")

    if sets_start_times[i] > start_time:
        raise AssertionError("Have gone back in time")

    while i < n - 1 and sets_start_times[i + 1] <= start_time:
        i += 1

    return i


# -------------------------------------------------
# Find nearest hospital to call location
# -------------------------------------------------

def nearest_hospital_to_call(sim: Simulation, call, priority: Priority) -> int:
    """
    Returns hospital index for the nearest hospital under travel priority.
    Assumes:
      - call.nearest_node_index exists
      - travel_mode.f_net_travel.f_node_nearest_hospital_index exists
    """
    travel_mode = get_travel_mode(sim.travel, priority, sim.time)

    hospital_index = (
        travel_mode.f_net_travel.f_node_nearest_hospital_index[
            call.nearest_node_index
        ]
    )

    return hospital_index
