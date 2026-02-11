from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import random

from .defs import (
    AmbStatus,
    CallStatus,
    EventForm,
    Priority,
    is_free,
    is_going_to_station,
)
from .decision import add_call_to_queue_sort_priority_then_time, find_nearest_dispatchable_amb_als_bls, get_next_call
from .demand import Demand, DemandCoverage
from .entities import Ambulance, Call, Hospital, Station
from .events import Event
from .map import Grid, Map
from .misc import EventsFile, File, MobilisationDelay, Redispatch, Resimulation, XMLElement
from .network import Network, Travel
from .pathfinding import ShortestPathCache
from .stats import SimStats
from .trace import TraceRecorder


@dataclass
class Simulation:
    """Core simulation state + event engine."""

    start_time: Optional[float] = None
    time: Optional[float] = None
    end_time: Optional[float] = None

    # Optional simulation-level RNG for end-to-end determinism (Step 9).
    # Used to generate seeds for distribution RNGs when input files request
    # random seeds (seed < 0).
    rng_seed: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random, repr=False)

    # cached counts (Julia stores these for convenience/performance)
    num_ambs: int = 0
    num_calls: int = 0
    num_hospitals: int = 0
    num_stations: int = 0

    # arc travel times read from the arcs file; indexed as [mode_index][arc_index]
    # with a dummy element at index 0 so indices match the 1-based Julia input.
    arc_travel_times: Optional[List[List[float]]] = None

    # Runtime shortest-path engine (Step 5). This is populated by init_sim
    # once the graph adjacency has been built.
    sp_cache: Optional[ShortestPathCache] = None

    net: Network = field(default_factory=Network)
    travel: Travel = field(default_factory=Travel)
    map: Map = field(default_factory=Map)
    grid: Grid = field(default_factory=Grid)

    ambulances: List[Ambulance] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    hospitals: List[Hospital] = field(default_factory=list)
    stations: List[Station] = field(default_factory=list)

    # Future event list (sorted by *non-increasing* time; next event is at the end).
    event_list: List[Event] = field(default_factory=list)
    event_index: int = 0
    queued_call_list: List[Call] = field(default_factory=list)

    # Optional trace recorder (Step 9). When enabled, executed events are
    # recorded and a digest can be computed for regression checks.
    trace: Optional[TraceRecorder] = None

    resim: Resimulation = field(default_factory=Resimulation)
    mobilisation_delay: MobilisationDelay = field(default_factory=MobilisationDelay)

    # Decision functions (can be swapped later based on simConfig.decision)
    add_call_to_queue: Callable[[List[Call], Call], None] = add_call_to_queue_sort_priority_then_time
    find_amb_to_dispatch: Callable[["Simulation", Call], Optional[int]] = find_nearest_dispatchable_amb_als_bls
    redispatch: Redispatch = field(default_factory=Redispatch.default)

    demand: Demand = field(default_factory=Demand)
    demand_coverage: DemandCoverage = field(default_factory=DemandCoverage)
    response_travel_priorities: Dict[Priority, Priority] = field(default_factory=dict)
    target_response_durations: List[float] = field(default_factory=list)

    # Parameters from statsControl (if provided).
    stats_control: Dict[str, Any] = field(default_factory=dict)

    current_calls: Set[int] = field(default_factory=set)  # call indices
    previous_calls: Set[int] = field(default_factory=set)  # call indices
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

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset dynamic state so the simulation can be re-run.

        Note: entity-level state reset (ambulance/call fields) is not fully
        implemented yet; for now this only resets the event engine.
        """

        self.time = self.start_time
        self.end_time = None
        self.event_list.clear()
        self.event_index = 0
        self.queued_call_list.clear()
        self.current_calls.clear()
        self.previous_calls.clear()
        self.complete = False
        self.used = False

        # Reset trace recorder (keep it enabled if it was enabled).
        if self.trace is not None:
            self.trace = TraceRecorder(store_events=self.trace.store_events, compute_digest=self.trace.compute_digest)

    # ------------------------------------------------------------------
    # Step 9: tracing helpers
    # ------------------------------------------------------------------

    def enable_trace(self, *, store_events: bool = False, compute_digest: bool = True) -> None:
        """Enable event tracing.

        When enabled, executed events are recorded (optionally stored in
        memory) and a digest can be computed for regression checks.
        """

        self.trace = TraceRecorder(store_events=store_events, compute_digest=compute_digest)

    def disable_trace(self) -> None:
        self.trace = None

    # ------------------------------------------------------------------
    # Step 5: shortest-path helpers
    # ------------------------------------------------------------------

    def ensure_sp_cache(self) -> None:
        """Ensure the shortest-path cache is initialised."""

        if self.sp_cache is None:
            self.net.f_graph.build_adjacency()
            self.sp_cache = ShortestPathCache(self.net.f_graph)

    def shortest_path_time(self, start_node: int, end_node: int, mode_index: int) -> float:
        """Shortest travel time between two full-graph nodes for a travel mode."""

        if start_node == end_node:
            return 0.0
        self.ensure_sp_cache()
        assert self.arc_travel_times is not None
        return self.sp_cache.shortest_time(mode_index, start_node, end_node, self.arc_travel_times[mode_index])

    def shortest_path_arcs(self, start_node: int, end_node: int, mode_index: int) -> List[int]:
        """Return the arc indices along the shortest path."""

        if start_node == end_node:
            return []
        self.ensure_sp_cache()
        assert self.arc_travel_times is not None
        return self.sp_cache.shortest_path_arcs(mode_index, start_node, end_node, self.arc_travel_times[mode_index])

    # ------------------------------------------------------------------
    # Event list management (Julia-compatible ordering / tie-breaking)
    # ------------------------------------------------------------------

    def add_event(
        self,
        *,
        parent: Optional[Event] = None,
        form: EventForm,
        time: float,
        ambulance: Optional[Ambulance] = None,
        call: Optional[Call] = None,
        station: Optional[Station] = None,
        add_event_to_amb: bool = True,
    ) -> Event:
        """Insert a new event into the future-event list.

        Ordering/tie-breaking mirrors Julia's ``addEvent!``:

        * ``event_list`` is kept sorted by **non-increasing** time.
        * The next event to execute is popped from the *end*.
        * For equal times, newly inserted events are executed **before** older
          ones (because they are inserted later in the list, closer to the end).
        """

        ev = Event()
        ev.parent_index = parent.index if parent is not None else None
        ev.form = form
        ev.time = float(time)
        ev.amb_index = int(ambulance.index) if ambulance is not None and ambulance.index is not None else None
        ev.call_index = int(call.index) if call is not None and call.index is not None else None
        ev.station_index = int(station.index) if station is not None and station.index is not None else None

        # Binary search insertion point in descending-by-time list:
        # insert after all events with time >= ev.time.
        lo = 0
        hi = len(self.event_list)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_t = float(self.event_list[mid].time or 0.0)
            if mid_t >= ev.time:
                lo = mid + 1
            else:
                hi = mid
        self.event_list.insert(lo, ev)

        if add_event_to_amb and ambulance is not None:
            ambulance.event = ev

        return ev

    def delete_event(self, event: Event) -> None:
        """Remove *event* from the future-event list (by identity)."""

        for i, e in enumerate(self.event_list):
            if e is event:
                self.event_list.pop(i)
                return
        # If it is already gone, silently ignore (Julia would error; this is practical for Python).

    def next_event(self) -> Optional[Event]:
        """Pop and return the next event (earliest time)."""

        if not self.event_list:
            self.complete = True
            return None

        ev = self.event_list.pop()
        self.time = ev.time
        self.event_index += 1
        ev.index = self.event_index
        self.used = True
        return ev

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    def simulate(
        self,
        *,
        time: float = float("inf"),
        duration: float = float("inf"),
        num_events: int = -1,
    ) -> bool:
        """Run until completion or until a stop condition is met."""

        if self.time is None and self.start_time is not None:
            self.time = self.start_time

        if duration != float("inf") and self.time is not None:
            time = min(time, float(self.time) + float(duration))

        event_count = 0
        max_events = num_events if num_events >= 0 else float("inf")

        while not self.complete and self.event_list and float(self.event_list[-1].time or 0.0) <= time and event_count < max_events:
            self.simulate_next_event()
            event_count += 1

        if not self.event_list and not self.complete:
            # no more events
            self.complete = True
            self.end_time = self.time

        return self.complete

    def simulate_next_event(self) -> None:
        """Execute the next event (Julia ``simulateNextEvent!``)."""

        ev = self.next_event()
        if ev is None:
            return

        # Step 10: write the executed event to the events output file.
        if self.write_output and not self.resim.use:
            from .write_sim_files import write_event_to_file

            write_event_to_file(self, ev)

        # Step 9: trace executed event (respect events_file filter).
        if self.trace is not None and self.events_file.event_filter.get(ev.form, True):
            self.trace.record(ev)

        # Dispatch to handler.
        self.simulate_event(ev)

        # If that was the last event, finalise.
        if not self.event_list:
            self.end_time = self.time
            self.complete = True

            # Close out ambulance status durations (no transition counted).
            for amb in self.ambulances[1:]:
                amb.set_status(self, amb.status, float(self.time or 0.0))

            # Close out station occupancy totals.
            end_t = float(self.time or 0.0)
            for st in self.stations[1:]:
                if st.current_num_idle_ambs_set_time is None:
                    continue
                if end_t < st.current_num_idle_ambs_set_time:
                    continue
                k = int(st.current_num_idle_ambs)
                dt = end_t - float(st.current_num_idle_ambs_set_time)
                st.num_idle_ambs_total_duration[k] = st.num_idle_ambs_total_duration.get(k, 0.0) + float(dt)
                st.current_num_idle_ambs_set_time = end_t

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def simulate_event(self, event: Event) -> None:
        form = event.form

        if form == EventForm.CALL_ARRIVES:
            self._ev_call_arrives(event)
        elif form == EventForm.CONSIDER_DISPATCH:
            self._ev_consider_dispatch(event)
        elif form == EventForm.AMB_DISPATCHED:
            self._ev_amb_dispatched(event)
        elif form == EventForm.AMB_MOBILISED:
            self._ev_amb_mobilised(event)
        elif form == EventForm.AMB_REACHES_CALL:
            self._ev_amb_reaches_call(event)
        elif form == EventForm.AMB_GOES_TO_HOSPITAL:
            self._ev_amb_goes_to_hospital(event)
        elif form == EventForm.AMB_REACHES_HOSPITAL:
            self._ev_amb_reaches_hospital(event)
        elif form == EventForm.AMB_BECOMES_FREE:
            self._ev_amb_becomes_free(event)
        elif form == EventForm.AMB_RETURNS_TO_CROSS_STREET:
            self._ev_amb_returns_to_cross_street(event)
        elif form == EventForm.AMB_REACHES_CROSS_STREET:
            self._ev_amb_reaches_cross_street(event)
        elif form == EventForm.AMB_RETURNS_TO_STATION:
            self._ev_amb_returns_to_station(event)
        elif form == EventForm.AMB_REACHES_STATION:
            self._ev_amb_reaches_station(event)
        elif form == EventForm.AMB_BECOMES_INACTIVE:
            self._ev_amb_becomes_inactive(event)
        elif form == EventForm.AMB_BECOMES_ACTIVE:
            self._ev_amb_becomes_active(event)
        elif form == EventForm.AMB_WAKES_UP:
            self._ev_amb_wakes_up(event)
        elif form == EventForm.AMB_GOES_TO_SLEEP:
            self._ev_amb_goes_to_sleep(event)
        else:
            # Move-up and other forms not yet implemented.
            pass

    # ---- individual handlers -----------------------------------------

    def _ev_call_arrives(self, event: Event) -> None:
        if event.call_index is None:
            raise ValueError("CALL_ARRIVES requires call_index")
        call = self.calls[event.call_index]

        call.set_status(CallStatus.SCREENING, float(self.time or 0.0))
        if call.index is not None:
            self.current_calls.add(int(call.index))

        # schedule next call arrival
        if call.index is not None and call.index < self.num_calls:
            nxt = self.calls[int(call.index) + 1]
            if nxt.arrival_time is None:
                raise ValueError("next call missing arrival_time")
            self.add_event(parent=event, form=EventForm.CALL_ARRIVES, time=float(nxt.arrival_time), call=nxt, add_event_to_amb=False)

        # schedule consider-dispatch for this call
        if call.arrival_time is None:
            raise ValueError("call missing arrival_time")
        self.add_event(
            parent=event,
            form=EventForm.CONSIDER_DISPATCH,
            time=float(call.arrival_time + call.dispatch_delay),
            call=call,
            add_event_to_amb=False,
        )

    def _ev_consider_dispatch(self, event: Event) -> None:
        if event.call_index is None:
            raise ValueError("CONSIDER_DISPATCH requires call_index")
        call = self.calls[event.call_index]

        amb_idx = self.find_amb_to_dispatch(self, call)

        if amb_idx is None:
            call.set_status(CallStatus.QUEUED, float(self.time or 0.0))
            self.add_call_to_queue(self.queued_call_list, call)
            return

        amb = self.ambulances[int(amb_idx)]

        # If ambulance already has a scheduled event that should be cancelled, do so.
        if is_going_to_station(amb.status):
            self.delete_event(amb.event)
        elif amb.status in (AmbStatus.MOBILISING, AmbStatus.GOING_TO_CALL):
            self.delete_event(amb.event)
            if amb.call_index is None:
                raise ValueError("redispatchable ambulance missing call_index")
            bumped = self.calls[int(amb.call_index)]
            bumped.amb_index = None
            bumped.num_bumps += 1
            self.add_event(parent=event, form=EventForm.CONSIDER_DISPATCH, time=float(self.time or 0.0), call=bumped, add_event_to_amb=False)

        # dispatch now
        self.add_event(parent=event, form=EventForm.AMB_DISPATCHED, time=float(self.time or 0.0), ambulance=amb, call=call)

    def _sample_mobilisation_delay(self) -> float:
        """Sample the *actual* mobilisation delay.

        Julia schedules the mobilisation completion event using a random draw
        from ``sim.mobilisationDelay.distrRng``. Dispatch *selection* uses the
        expected duration, but the realised mobilisation time uses sampling.
        """

        md = self.mobilisation_delay
        if md.distr_rng is not None:
            return float(md.distr_rng.sample())
        return float(md.expected_duration or 0.0)

    def _ev_amb_dispatched(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_DISPATCHED requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]
        if amb.station_index is None:
            raise ValueError("ambulance missing station_index")
        station = self.stations[int(amb.station_index)]

        amb.call_index = call.index
        amb.dispatch_time = float(self.time or 0.0)

        mobilise = bool(self.mobilisation_delay.use) and amb.status in (
            AmbStatus.IDLE_AT_STATION,
            AmbStatus.IDLE_AT_CROSS_STREET,
            AmbStatus.MOBILISING,
        )

        prev_status = amb.status

        if mobilise:
            amb.set_status(self, AmbStatus.MOBILISING, float(self.time or 0.0))
        else:
            amb.set_status(self, AmbStatus.GOING_TO_CALL, float(self.time or 0.0))
            if call.priority is None:
                raise ValueError("call.priority missing")
            resp_priority = self.response_travel_priorities.get(call.priority, call.priority)
            if call.nearest_node_index is None:
                raise ValueError("call.nearest_node_index missing")
            amb.route.plan_to_fnode(
                self,
                priority=resp_priority,
                start_time=float(self.time or 0.0),
                end_loc=call.location,
                end_fnode=int(call.nearest_node_index),
            )
            self.add_event(
                parent=event,
                form=EventForm.AMB_REACHES_CALL,
                time=float(amb.route.end_time or 0.0),
                ambulance=amb,
                call=call,
            )

        call.set_status(CallStatus.WAITING_FOR_AMB, float(self.time or 0.0))
        call.amb_index = amb.index
        call.amb_dispatch_loc = amb.route.current_location(self, float(self.time or 0.0))
        call.amb_status_before_dispatch = prev_status

        # station stats: decrement when dispatching from station idle
        if prev_status == AmbStatus.IDLE_AT_STATION:
            station.update_stats(num_idle_ambs_change=-1, time=float(self.time or 0.0))

        if mobilise:
            if prev_status == AmbStatus.MOBILISING and amb.mobilisation_time is not None:
                mobilisation_time = float(amb.mobilisation_time)
            else:
                mobilisation_time = float(self.time or 0.0) + self._sample_mobilisation_delay()
                amb.mobilisation_time = mobilisation_time

            self.add_event(
                parent=event,
                form=EventForm.AMB_MOBILISED,
                time=mobilisation_time,
                ambulance=amb,
                call=call,
            )

    def _ev_amb_mobilised(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_MOBILISED requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]

        amb.set_status(self, AmbStatus.GOING_TO_CALL, float(self.time or 0.0))
        amb.mobilisation_time = None

        if call.priority is None:
            raise ValueError("call.priority missing")
        resp_priority = self.response_travel_priorities.get(call.priority, call.priority)

        if call.nearest_node_index is None:
            raise ValueError("call.nearest_node_index missing")
        amb.route.plan_to_fnode(
            self,
            priority=resp_priority,
            start_time=float(self.time or 0.0),
            end_loc=call.location,
            end_fnode=int(call.nearest_node_index),
        )

        self.add_event(
            parent=event,
            form=EventForm.AMB_REACHES_CALL,
            time=float(amb.route.end_time or 0.0),
            ambulance=amb,
            call=call,
        )

    def _ev_amb_reaches_call(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_REACHES_CALL requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]

        amb.set_status(self, AmbStatus.AT_CALL, float(self.time or 0.0))
        call.set_status(CallStatus.ON_SCENE_TREATMENT, float(self.time or 0.0))

        next_time = float(self.time or 0.0) + float(call.on_scene_duration)

        if call.transport:
            self.add_event(parent=event, form=EventForm.AMB_GOES_TO_HOSPITAL, time=next_time, ambulance=amb, call=call)
        else:
            self.add_event(parent=event, form=EventForm.AMB_BECOMES_FREE, time=next_time, ambulance=amb, call=call)

    def nearest_hospital_to_call(self, call: Call, *, priority: Priority = Priority.LOW) -> int:
        """Choose the nearest hospital to *call* by travel time."""

        if call.nearest_node_index is None or call.nearest_node_dist is None:
            raise ValueError("call missing nearest-node linkage")
        travel_mode_index = self.travel.mode_index_for_priority(priority, float(self.time or 0.0))

        best_idx = 0
        best_t = float("inf")

        for hosp in self.hospitals[1:]:
            if hosp.nearest_node_index is None or hosp.nearest_node_dist is None:
                continue
            t = 0.0
            # off-road from call to its nearest node
            mode = self.travel.modes[travel_mode_index]
            t += float(call.nearest_node_dist) / float(mode.off_road_speed or 1.0)
            # on-road
            t += self.shortest_path_time(int(call.nearest_node_index), int(hosp.nearest_node_index), travel_mode_index)
            # off-road from nearest node to hospital
            t += float(hosp.nearest_node_dist) / float(mode.off_road_speed or 1.0)

            if t < best_t:
                best_t = t
                best_idx = int(hosp.index or 0)

        if best_idx <= 0:
            raise ValueError("no hospitals available")
        return best_idx

    def _ev_amb_goes_to_hospital(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_GOES_TO_HOSPITAL requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]

        pr = Priority.LOW

        if call.hospital_index is None:
            call.chosen_hospital_index = self.nearest_hospital_to_call(call, priority=pr)
        else:
            call.chosen_hospital_index = int(call.hospital_index)

        hosp = self.hospitals[int(call.chosen_hospital_index)]

        amb.set_status(self, AmbStatus.GOING_TO_HOSPITAL, float(self.time or 0.0))
        if hosp.nearest_node_index is None:
            raise ValueError("hospital.nearest_node_index missing")
        amb.route.plan_to_fnode(
            self,
            priority=pr,
            start_time=float(self.time or 0.0),
            end_loc=hosp.location,
            end_fnode=int(hosp.nearest_node_index),
        )

        call.set_status(CallStatus.GOING_TO_HOSPITAL, float(self.time or 0.0))

        self.add_event(
            parent=event,
            form=EventForm.AMB_REACHES_HOSPITAL,
            time=float(amb.route.end_time or 0.0),
            ambulance=amb,
            call=call,
        )

    def _ev_amb_reaches_hospital(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_REACHES_HOSPITAL requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]

        if call.chosen_hospital_index is None:
            raise ValueError("call.chosen_hospital_index missing")
        hosp = self.hospitals[int(call.chosen_hospital_index)]

        amb.set_status(self, AmbStatus.AT_HOSPITAL, float(self.time or 0.0))
        call.set_status(CallStatus.AT_HOSPITAL, float(self.time or 0.0))
        hosp.num_calls += 1

        self.add_event(
            parent=event,
            form=EventForm.AMB_BECOMES_FREE,
            time=float(self.time or 0.0) + float(call.handover_duration),
            ambulance=amb,
            call=call,
        )

    def _ev_amb_becomes_free(self, event: Event) -> None:
        if event.amb_index is None or event.call_index is None:
            raise ValueError("AMB_BECOMES_FREE requires amb_index and call_index")
        amb = self.ambulances[int(event.amb_index)]
        call = self.calls[int(event.call_index)]

        if amb.station_index is None:
            raise ValueError("ambulance missing station_index")
        station = self.stations[int(amb.station_index)]

        call.set_status(CallStatus.PROCESSED, float(self.time or 0.0))
        if call.index is not None:
            self.current_calls.discard(int(call.index))
            self.previous_calls.add(int(call.index))

        amb.set_status(self, AmbStatus.FREE_AFTER_CALL, float(self.time or 0.0))
        amb.call_index = None

        if amb.end_current_tour:
            self.add_event(parent=event, form=EventForm.AMB_RETURNS_TO_STATION, time=float(self.time or 0.0), ambulance=amb, station=station)
            return

        nxt = get_next_call(self.queued_call_list)
        if nxt is not None:
            self.add_event(parent=event, form=EventForm.AMB_DISPATCHED, time=float(self.time or 0.0), ambulance=amb, call=nxt)
        else:
            self.add_event(
                parent=event,
                form=EventForm.AMB_RETURNS_TO_CROSS_STREET,
                time=float(self.time or 0.0),
                ambulance=amb,
                station=station,
            )

    def _ev_amb_returns_to_cross_street(self, event: Event) -> None:
        if event.amb_index is None:
            raise ValueError("AMB_RETURNS_TO_CROSS_STREET requires amb_index")
        amb = self.ambulances[int(event.amb_index)]

        amb.set_status(self, AmbStatus.RETURNING_TO_CROSS_STREET, float(self.time or 0.0))

        if amb.cross_street_node_index is None:
            raise ValueError("ambulance missing cross_street_node_index")
        amb.route.plan_to_fnode(
            self,
            priority=Priority.LOW,
            start_time=float(self.time or 0.0),
            end_loc=amb.cross_street_location,
            end_fnode=int(amb.cross_street_node_index),
        )
        self.add_event(parent=event, form=EventForm.AMB_REACHES_CROSS_STREET, time=float(amb.route.end_time or 0.0), ambulance=amb)

    def _ev_amb_reaches_cross_street(self, event: Event) -> None:
        if event.amb_index is None:
            raise ValueError("AMB_REACHES_CROSS_STREET requires amb_index")
        amb = self.ambulances[int(event.amb_index)]
        amb.set_status(self, AmbStatus.IDLE_AT_CROSS_STREET, float(self.time or 0.0))
        amb.event = Event()

    def _ev_amb_returns_to_station(self, event: Event) -> None:
        if event.amb_index is None or event.station_index is None:
            raise ValueError("AMB_RETURNS_TO_STATION requires amb_index and station_index")
        amb = self.ambulances[int(event.amb_index)]
        station = self.stations[int(event.station_index)]

        amb.set_status(self, AmbStatus.RETURNING_TO_STATION, float(self.time or 0.0))

        if station.nearest_node_index is None:
            raise ValueError("station missing nearest_node_index")
        amb.route.plan_to_fnode(
            self,
            priority=Priority.LOW,
            start_time=float(self.time or 0.0),
            end_loc=station.location,
            end_fnode=int(station.nearest_node_index),
        )

        self.add_event(parent=event, form=EventForm.AMB_REACHES_STATION, time=float(amb.route.end_time or 0.0), ambulance=amb, station=station)

    def _ev_amb_reaches_station(self, event: Event) -> None:
        if event.amb_index is None or event.station_index is None:
            raise ValueError("AMB_REACHES_STATION requires amb_index and station_index")
        amb = self.ambulances[int(event.amb_index)]
        station = self.stations[int(event.station_index)]

        if amb.end_current_tour:
            amb.end_current_tour = False
            if amb.start_next_tour:
                amb.start_next_tour = False
                amb.set_status(self, AmbStatus.IDLE_AT_STATION, float(self.time or 0.0))
                self.add_event(parent=event, form=EventForm.AMB_RETURNS_TO_CROSS_STREET, time=float(self.time or 0.0), ambulance=amb, station=station)
            else:
                amb.set_status(self, AmbStatus.SLEEPING, float(self.time or 0.0))
                amb.event = Event()
        else:
            amb.set_status(self, AmbStatus.IDLE_AT_STATION, float(self.time or 0.0))
            amb.event = Event()
            station.update_stats(num_idle_ambs_change=1, time=float(self.time or 0.0))

    def _ev_amb_becomes_inactive(self, event: Event) -> None:
        if event.amb_index is None:
            raise ValueError("AMB_BECOMES_INACTIVE requires amb_index")
        amb = self.ambulances[int(event.amb_index)]

        amb.end_current_tour = True

        if amb.status == AmbStatus.IDLE_AT_STATION:
            amb.end_current_tour = False
            amb.set_status(self, AmbStatus.SLEEPING, float(self.time or 0.0))
            amb.event = Event()
            return

        if amb.status in (AmbStatus.IDLE_AT_CROSS_STREET, AmbStatus.RETURNING_TO_CROSS_STREET):
            # Cancel any pending "return-to-cross-street" events.
            if amb.event.form in (EventForm.AMB_RETURNS_TO_CROSS_STREET, EventForm.AMB_REACHES_CROSS_STREET):
                self.delete_event(amb.event)

            if amb.station_index is None:
                raise ValueError("ambulance missing station_index")
            station = self.stations[int(amb.station_index)]
            self.add_event(parent=event, form=EventForm.AMB_RETURNS_TO_STATION, time=float(self.time or 0.0), ambulance=amb, station=station)

    def _ev_amb_becomes_active(self, event: Event) -> None:
        if event.amb_index is None:
            raise ValueError("AMB_BECOMES_ACTIVE requires amb_index")
        amb = self.ambulances[int(event.amb_index)]

        amb.start_next_tour = True

        if amb.status == AmbStatus.SLEEPING:
            amb.start_next_tour = False
            if amb.station_index is None:
                raise ValueError("ambulance missing station_index")
            station = self.stations[int(amb.station_index)]
            amb.set_status(self, AmbStatus.IDLE_AT_STATION, float(self.time or 0.0))
            self.add_event(parent=event, form=EventForm.AMB_RETURNS_TO_CROSS_STREET, time=float(self.time or 0.0), ambulance=amb, station=station)

    def _ev_amb_wakes_up(self, event: Event) -> None:
        if event.amb_index is None or event.station_index is None:
            raise ValueError("AMB_WAKES_UP requires amb_index and station_index")
        amb = self.ambulances[int(event.amb_index)]
        station = self.stations[int(event.station_index)]
        amb.set_status(self, AmbStatus.IDLE_AT_STATION, float(self.time or 0.0))
        amb.event = Event()
        station.update_stats(num_idle_ambs_change=1, time=float(self.time or 0.0))

    def _ev_amb_goes_to_sleep(self, event: Event) -> None:
        if event.amb_index is None or event.station_index is None:
            raise ValueError("AMB_GOES_TO_SLEEP requires amb_index and station_index")
        amb = self.ambulances[int(event.amb_index)]
        station = self.stations[int(event.station_index)]
        amb.set_status(self, AmbStatus.SLEEPING, float(self.time or 0.0))
        amb.event = Event()
        station.update_stats(num_idle_ambs_change=-1, time=float(self.time or 0.0))
