# EMS-Simulation
Python EMS simulator and network tooling, aligned with the JEMSS model workflow.

## Quick Start (Simulation)
Run the bundled Bronx baseline config:

```bash
cd /Users/longderek/Documents/FDNY/EMS-Simulation
python3 example.py
```

If you want output files, set `<writeOutput>true</writeOutput>` in the config file used by `example.py`.

## OSM Network Generation
Use the following files for network-generation:

#### 1. `src/convert_osm_network.py`
Converts an `.osm` file into JEMSS-style `nodes.csv` and `arcs.csv`:

```bash
python3 src/convert_osm_network.py \
  --osm <path_to_roads.osm_file> \
  --nodes-out /tmp/nyc_nodes.csv \
  --arcs-out /tmp/nyc_arcs.csv
```

#### 2. `src/run_osm_roads.py`
Runs the full preprocessing pipeline (conversion, arc splitting, OD-node shortest-path tagging, filtering, and CSV write):

```bash
python3 -m src.run_osm_roads
```

## Dependencies
Minimum:
- Python 3.9+

Optional but recommended:
- `tqdm` (used by current init/import path)
- `pyshp` (`shapefile` module, required for border clipping in `run_osm_roads.py`)

Install:

```bash
pip install tqdm pyshp
```
