from __future__ import annotations

from typing import Optional

from .init_sim import init_sim
from .write_sim_files import close_output_files, open_output_files, write_output_files


def run_config(
    config_filename: str,
    *,
    allow_write_output: bool = True,
    compute_checksums: bool = True,
    do_print: bool = False,
    printing_interval: float = 0.1,
    seed: Optional[int] = None,
):
    """Run a simulation from a reference-style simConfig XML.

    Returns the populated :class:`~jemss.simulator.Simulation`.
    """

    sim = init_sim(
        config_filename,
        allow_write_output=allow_write_output,
        compute_checksums=compute_checksums,
        do_print=do_print,
        seed=seed,
    )
    
    if sim.write_output:
        open_output_files(sim)

    sim.simulate(do_print=do_print, printing_interval=printing_interval)

    if sim.write_output:
        close_output_files(sim)
        write_output_files(sim)
        # Mirror reference: disable writing so the same sim can be re-run.
        sim.write_output = False

    return sim
