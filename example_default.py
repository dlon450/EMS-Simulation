import os

from src.decision import find_nearest_dispatchable_amb_als_bls
from src.init_sim import init_sim
from src.write_sim_files import close_output_files, open_output_files, write_output_files


CONFIG_FILENAME = "config/bronx_sim_config.xml"
OUTPUT_SUBDIR = "bronx_default"


def _redirect_output(sim, output_subdir: str) -> None:
    output_path = os.path.abspath(os.path.join("output", output_subdir))
    os.makedirs(output_path, exist_ok=True)
    sim.output_path = output_path
    for file in sim.output_files.values():
        file.path = os.path.join(output_path, file.name)


def main() -> None:
    sim = init_sim(
        CONFIG_FILENAME,
        allow_write_output=True,
        compute_checksums=False,
        do_print=True,
    )

    _redirect_output(sim, OUTPUT_SUBDIR)
    sim.find_amb_to_dispatch = find_nearest_dispatchable_amb_als_bls

    if sim.write_output:
        open_output_files(sim)

    sim.simulate(
        do_print=True,
        printing_interval=0.01,
    )

    if sim.write_output:
        close_output_files(sim)
        write_output_files(sim)
        sim.write_output = False

    print(
        "policy: nearest_dispatchable_amb_als_bls",
        "output_dir:", sim.output_path,
        "events:", sim.event_index,
        "processed:", sum(1 for c in sim.calls[1:] if c.status.name == "PROCESSED"),
    )


if __name__ == "__main__":
    main()
