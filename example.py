from src.run_config import run_config

sim = run_config(
    "config/bronx_sim_config.xml", 
    allow_write_output=True, 
    compute_checksums=False, 
    do_print=True,
    printing_interval=0.01
)

print(
    "events:", sim.event_index, 
    "processed:", sum(1 for c in sim.calls[1:] if c.status.name == "PROCESSED")
)