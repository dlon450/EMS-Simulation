#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

# Make `src` imports robust no matter where this script is launched from.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.init_sim import init_sim
from src.write_sim_files import close_output_files, open_output_files, write_output_files


FuncKey = Tuple[str, int, str]


@dataclass
class GroupSummary:
    name: str
    primitive_calls: int
    total_calls: int
    total_time: float
    cumulative_time: float
    matches: int


def _basename(path: str) -> str:
    return os.path.basename(path)


def _is_file(path: str, name: str) -> bool:
    return _basename(path) == name


def _format_func(key: FuncKey) -> str:
    filename, line, func = key
    return f"{_basename(filename)}:{line}:{func}"


def _resolve_from_repo(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def _group_entries(
    stats: pstats.Stats,
    predicate: Callable[[str, str], bool],
) -> List[Tuple[FuncKey, int, int, float, float]]:
    out: List[Tuple[FuncKey, int, int, float, float]] = []
    for key, row in stats.stats.items():
        filename, _line, func = key
        if predicate(filename, func):
            primitive_calls, total_calls, total_time, cumulative_time, _callers = row
            out.append((key, primitive_calls, total_calls, total_time, cumulative_time))
    return out


def _summarise_group(name: str, entries: Iterable[Tuple[FuncKey, int, int, float, float]]) -> GroupSummary:
    prim = 0
    total = 0
    tt = 0.0
    ct = 0.0
    n = 0
    for _key, primitive_calls, total_calls, total_time, cumulative_time in entries:
        prim += primitive_calls
        total += total_calls
        tt += total_time
        ct += cumulative_time
        n += 1
    return GroupSummary(
        name=name,
        primitive_calls=prim,
        total_calls=total,
        total_time=tt,
        cumulative_time=ct,
        matches=n,
    )


def _print_group_details(
    stats: pstats.Stats,
    name: str,
    predicate: Callable[[str, str], bool],
    per_group_top: int,
) -> None:
    entries = _group_entries(stats, predicate)
    summary = _summarise_group(name, entries)
    print(
        f"- {summary.name}: cum={summary.cumulative_time:.3f}s "
        f"tot={summary.total_time:.3f}s "
        f"calls={summary.total_calls} prim_calls={summary.primitive_calls} "
        f"functions={summary.matches}"
    )
    if not entries:
        return

    entries.sort(key=lambda x: x[4], reverse=True)
    for key, primitive_calls, total_calls, total_time, cumulative_time in entries[:per_group_top]:
        print(
            f"    {_format_func(key)} | cum={cumulative_time:.3f}s "
            f"tot={total_time:.3f}s calls={total_calls}/{primitive_calls}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded simulation under cProfile and report hotspot groups."
    )
    parser.add_argument(
        "--config",
        default="config/bronx_sim_config.xml",
        help="Path to sim config XML (relative paths are resolved from repo root)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        help="Simulation duration to run (days)",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=-1,
        help="Optional event cap; -1 means unlimited",
    )
    parser.add_argument(
        "--allow-write-output",
        action="store_true",
        default=False,
        help="Enable output writes during profiled run (includes file writing costs)",
    )
    parser.add_argument(
        "--profile-out",
        default="profiling/profile_sim.prof",
        help="Path for raw .prof output (relative paths are resolved from repo root)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Top N functions to show for global rankings",
    )
    parser.add_argument(
        "--group-top",
        type=int,
        default=8,
        help="Top N functions to show per hotspot group",
    )
    args = parser.parse_args()
    config_path = _resolve_from_repo(args.config)
    profile_out = _resolve_from_repo(args.profile_out)

    sim = init_sim(
        config_path,
        allow_write_output=bool(args.allow_write_output),
        compute_checksums=False,
        do_print=False,
    )

    if sim.write_output:
        open_output_files(sim)

    profiler = cProfile.Profile()
    profiler.enable()

    sim.simulate(duration=float(args.duration), num_events=int(args.num_events), do_print=False)

    if sim.write_output:
        close_output_files(sim)
        write_output_files(sim)

    profiler.disable()

    out_dir = os.path.dirname(profile_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    profiler.dump_stats(profile_out)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    print("== Profile Run ==")
    print(f"config: {config_path}")
    print(f"duration(days): {args.duration}")
    print(f"num_events_cap: {args.num_events}")
    print(f"allow_write_output: {args.allow_write_output}")
    print(f"events_executed: {sim.event_index}")
    print(f"profile_out: {profile_out}")
    print(f"total_profiled_time: {stats.total_tt:.3f}s")
    print("")

    print(f"== Top {args.top} by cumulative time ==")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top)
    print("")

    print(f"== Top {args.top} by internal time ==")
    stats.sort_stats(pstats.SortKey.TIME).print_stats(args.top)
    print("")

    groups: List[Tuple[str, Callable[[str, str], bool]]] = [
        (
            "dispatch selection",
            lambda f, fn: _is_file(f, "decision.py")
            and fn
            in {
                "find_nearest_dispatchable_amb_als_bls",
                "find_nearest_dispatchable_amb",
                "_estimated_response_duration",
                "_estimate_mobilisation_start_time",
                "is_amb_dispatchable",
                "is_amb_redispatchable",
            },
        ),
        (
            "route travel-time estimation",
            lambda f, fn: _is_file(f, "map.py")
            and fn
            in {
                "travel_time_to_location",
                "next_node",
                "next_node_distance",
                "update_to_time",
                "current_location",
                "off_road_travel_time",
                "norm_dist",
                "square_dist",
            },
        ),
        (
            "shortest-path lookup",
            lambda f, fn: (_is_file(f, "pathfinding.py") and fn.startswith("shortest_"))
            or (_is_file(f, "pathfinding.py") and fn in {"get_tree", "_dijkstra_all"})
            or (_is_file(f, "simulator.py") and fn in {"shortest_path_time", "shortest_path_arcs"}),
        ),
        (
            "event insertion / next-event pop",
            lambda f, fn: (_is_file(f, "simulator.py") and fn in {"add_event", "next_event"})
            or ("list.insert" in fn)
            or ("list.pop" in fn),
        ),
        (
            "nearest-hospital lookup",
            lambda f, fn: _is_file(f, "simulator.py") and fn == "nearest_hospital_to_call",
        ),
        (
            "stats/output writing",
            lambda f, fn: _is_file(f, "write_sim_files.py")
            or (_is_file(f, "entities.py") and fn in {"set_status", "update_stats"}),
        ),
    ]

    print("== Hotspot Groups ==")
    for name, predicate in groups:
        _print_group_details(stats, name, predicate, args.group_top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
