"""jemss.golden

Step 9: small utilities for *repeatable* simulator runs.

This module is designed to help you:

* run a scenario from a simConfig XML
* compute a compact JSON-serialisable summary of outcomes
* optionally compute an event-trace digest for regression testing

The intent is that you can run the Julia simulator and the Python simulator on
the same scenario and compare a small set of key outputs (counts, mean response
times, event trace digest, etc.).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import json
import math

from .defs import PRIORITIES, Priority
from .init_sim import init_sim
from .simulator import Simulation


def summarize_sim(sim: Simulation, *, include_trace: bool = True) -> Dict[str, Any]:
    """Build a compact summary dict for *sim*.

    The summary is intentionally simple and stable so it can be saved to JSON.
    """

    calls = sim.calls[1:]

    num_processed = sum(1 for c in calls if c.status.name == "PROCESSED")
    num_queued = sum(1 for c in calls if getattr(c, "was_queued", False))
    num_bumped = sum(1 for c in calls if getattr(c, "num_bumps", 0) > 0)
    num_transports = sum(1 for c in calls if getattr(c, "transport", False))

    # Response durations (only where they were actually computed).
    resp = [float(c.response_duration) for c in calls if getattr(c, "amb_arrival_time", None) is not None]
    mean_resp = float(sum(resp) / len(resp)) if resp else math.nan
    max_resp = float(max(resp)) if resp else math.nan

    # Per-priority breakdown.
    by_pr: Dict[str, Any] = {}
    for p in PRIORITIES:
        pcalls = [c for c in calls if c.priority == p]
        presp = [float(c.response_duration) for c in pcalls if getattr(c, "amb_arrival_time", None) is not None]
        target = None
        if sim.target_response_durations and 0 <= p.value < len(sim.target_response_durations):
            target = float(sim.target_response_durations[p.value])
        in_time = None
        if target is not None and not math.isnan(target):
            in_time = sum(1 for d in presp if d <= target)
        by_pr[p.name] = {
            "count": len(pcalls),
            "processed": sum(1 for c in pcalls if c.status.name == "PROCESSED"),
            "mean_response": float(sum(presp) / len(presp)) if presp else math.nan,
            "max_response": float(max(presp)) if presp else math.nan,
            "target_response": target,
            "in_time": in_time,
            "pct_in_time": (float(in_time) / len(presp) if presp and in_time is not None else math.nan),
        }

    out: Dict[str, Any] = {
        "rng_seed": sim.rng_seed,
        "start_time": sim.start_time,
        "end_time": sim.end_time,
        "events_executed": sim.event_index,
        "num_calls": sim.num_calls,
        "num_processed": num_processed,
        "num_queued": num_queued,
        "num_bumped": num_bumped,
        "num_transports": num_transports,
        "mean_response": mean_resp,
        "max_response": max_resp,
        "by_priority": by_pr,
    }

    if include_trace and sim.trace is not None:
        out["trace_digest"] = sim.trace.digest_hex()

    return out


def run_golden(
    config_filename: str,
    *,
    seed: Optional[int] = None,
    trace: bool = True,
    store_events: bool = False,
    allow_write_output: bool = False,
    compute_checksums: bool = True,
    do_print: bool = False,
    stop_time: float = float("inf"),
    duration: float = float("inf"),
    num_events: int = -1,
) -> Dict[str, Any]:
    """Initialise + run a simulation and return a JSON-friendly summary."""

    sim = init_sim(
        config_filename,
        allow_write_output=allow_write_output,
        compute_checksums=compute_checksums,
        do_print=do_print,
        seed=seed,
    )

    if trace:
        sim.enable_trace(store_events=store_events, compute_digest=True)

    sim.simulate(time=float(stop_time), duration=float(duration), num_events=int(num_events))
    summary = summarize_sim(sim, include_trace=trace)

    if trace and store_events and sim.trace is not None:
        # Include the raw event rows (can be large).
        summary["events"] = list(sim.trace.events)

    return summary


def save_summary(summary: Dict[str, Any], filename: str) -> None:
    """Save *summary* to *filename* as JSON with stable formatting."""

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def load_summary(filename: str) -> Dict[str, Any]:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
