import os

from src.decision import find_coverage_dispatchable_amb_als_bls
from src.init_dc import init_demand_coverage
from src.init_sim import init_sim
from src.read_sim_files import read_demand_file
from src.write_sim_files import close_output_files, open_output_files, write_output_files


CONFIG_FILENAME = "config/bronx_sim_config.xml"
OUTPUT_SUBDIR = "bronx_new_policy"
ALPHA = 0.9
RHO = 0.1
LOOKBACK_HOURS = 6.0
UPDATE_INTERVAL_HOURS = 6.0


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

    if "demand" in sim.input_files:
        demand_filename = sim.input_files["demand"].path
    else:
        demand_filename = os.path.join(sim.input_path, "demand.csv")

    sim.demand = read_demand_file(demand_filename)
    init_demand_coverage(sim)

    sim.enable_blended_demand_estimate(
        alpha=ALPHA,
        lookback_hours=LOOKBACK_HOURS,
        update_interval_hours=UPDATE_INTERVAL_HOURS,
    )
    sim.update_blended_demand_estimate(time=sim.start_time)

    sim.find_amb_to_dispatch = lambda s, c: find_coverage_dispatchable_amb_als_bls(
        s,
        c,
        rho=RHO,
    )

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
        "alpha:", ALPHA,
        "rho:", RHO,
        "output_dir:", sim.output_path,
        "events:", sim.event_index,
        "processed:", sum(1 for c in sim.calls[1:] if c.status.name == "PROCESSED"),
    )


if __name__ == "__main__":
    main()
