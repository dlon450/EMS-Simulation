"""jemss.config

Step 2: configuration file parsing.

The Julia project uses an XML ``simConfig`` file to describe where to
find the simulation input tables, where to write outputs, and a handful
of run-time options. The heavy lifting of reading the simulation data
and constructing the network happens later (``initSim`` in Julia).

In this Python port we start by implementing the configuration layer in a way that's:

* faithful to the XML structure used by Julia (e.g., <simFiles> with child tags for each file)
* safe (no arbitrary expression evaluation)
* explicit about path interpolation and base directories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

import os

from .checksum import file_checksum
from .defs import EventForm
from .misc import File
from .xml_utils import (
    children_node_names,
    contains_elt,
    elt_interp_text_required,
    elt_text_required,
    elt_value_required,
    find_elt,
    parse_value,
    xml_file_root,
)


def join_path_if_not_abs(abs_path: str, path: str) -> str:
    """If *path* is absolute return it; otherwise join it to *abs_path*."""

    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(abs_path, p))


def _event_form_from_token(token: str) -> EventForm:
    """Best-effort mapping from a config token to an :class:`EventForm`."""

    t = token.strip()
    if t.startswith("EventForm."):
        t = t.split(".", 1)[1]
    # allow common Julia-ish camelCase names by converting to upper snake
    # e.g. ambGoesToSleep -> AMB_GOES_TO_SLEEP
    if t and any(c.islower() for c in t) and "_" not in t:
        out = []
        for ch in t:
            if ch.isupper() and out:
                out.append("_")
            out.append(ch.upper())
        t = "".join(out)
    t = t.upper()
    return EventForm[t]


@dataclass(frozen=True)
class SimConfig:
    """Parsed contents of a ``simConfig`` XML file."""

    config_filename: str
    config_dir: str

    input_path: str
    output_path: str
    write_output: bool

    input_files: Dict[str, File]
    output_files: Dict[str, File]

    # If present, maps event forms to whether they should be written.
    events_file_filter: Optional[Dict[EventForm, bool]] = None


def load_sim_config(
    config_filename: str,
    *,
    allow_write_output: bool = False,
    compute_checksums: bool = True,
) -> SimConfig:
    """Load and parse a Julia-style ``simConfig`` XML file.

    This function only parses the configuration and resolves paths; it does
    *not* read the underlying simulation data tables.
    """

    config_filename = os.path.abspath(os.path.expanduser(config_filename))
    if not os.path.isfile(config_filename):
        raise FileNotFoundError(config_filename)

    root = xml_file_root(config_filename)
    if root.tag != "simConfig":
        raise ValueError(f"xml root has incorrect name: {root.tag}")

    config_dir = os.path.dirname(config_filename)
    vars_ = {"configFileDir": config_dir}

    # Input/output base paths (relative to config file directory unless absolute).
    input_path_raw = elt_interp_text_required(root, "inputPath", vars_)
    input_path = join_path_if_not_abs(config_dir, input_path_raw)

    output_path_raw = elt_interp_text_required(root, "outputPath", vars_)
    output_path = join_path_if_not_abs(config_dir, output_path_raw)
    os.makedirs(output_path, exist_ok=True)

    write_output_val = bool(parse_value(elt_text_required(root, "writeOutput")))
    write_output = allow_write_output and write_output_val

    # simFiles
    sim_files_elt = find_elt(root, "simFiles")
    if sim_files_elt is None:
        raise KeyError("Element not found: simFiles")

    input_files: Dict[str, File] = {}
    for name in children_node_names(sim_files_elt):
        raw = elt_interp_text_required(sim_files_elt, name, vars_)
        path = join_path_if_not_abs(input_path, raw)
        f = File(name=os.path.basename(path), path=path)
        if compute_checksums and name != "rNetTravels":
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            f.checksum = file_checksum(path)
        input_files[name] = f

    # outputFiles
    out_files_elt = find_elt(root, "outputFiles")
    if out_files_elt is None:
        raise KeyError("Element not found: outputFiles")

    output_files: Dict[str, File] = {}
    for name in children_node_names(out_files_elt):
        raw = elt_interp_text_required(out_files_elt, name, vars_)
        path = join_path_if_not_abs(output_path, raw)
        output_files[name] = File(name=os.path.basename(path), path=path)

    # Optional events file filter: only write the listed event forms.
    events_file_filter: Optional[Dict[EventForm, bool]] = None
    if contains_elt(root, "eventsFileFilter"):
        val = elt_value_required(root, "eventsFileFilter")
        # val can be a list (from literal_eval) or list of tokens.
        if isinstance(val, (list, tuple)):
            tokens = [str(x) for x in val]
        else:
            tokens = [str(val)]

        include: Set[EventForm] = set()
        for tok in tokens:
            include.add(_event_form_from_token(tok))

        events_file_filter = {e: (e in include) for e in EventForm}

    return SimConfig(
        config_filename=config_filename,
        config_dir=config_dir,
        input_path=input_path,
        output_path=output_path,
        write_output=write_output,
        input_files=input_files,
        output_files=output_files,
        events_file_filter=events_file_filter,
    )
