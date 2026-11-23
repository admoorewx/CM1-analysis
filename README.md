# CM1-analysis

An analysis and plotting pipeline for output from the
[CM1 cloud model](https://www2.mmm.ucar.edu/people/bryan/cm1/), developed for
idealized derecho simulations.

The goal is one tool that takes any CM1 output file and produces a complete,
consistent set of analysis figures:

| Category | Contents |
|---|---|
| `planar` | Plan views at a level or the surface, wind and reflectivity swaths, run comparisons |
| `timeseries` | Domain-reduced quantities through a run, one line per run in an experiment |
| `statistics` | Distributions, effect sizes and correlations across experiments |
| `cross_section` | Vertical sections through the storm |
| `pressure` | Vertical momentum forcing: dynamic pressure-gradient force and effective buoyancy |
| `wind` | Helmholtz partition of the horizontal wind (rotational, divergent, harmonic) |

> **Status:** early development. The shared core (file reading, configuration,
> plot registry and figure style) is in place and tested. Plots are being
> ported into the categories above one at a time; `cm1plot list` shows what
> is available.

## Installation

Requires Python 3.12 or later. Use the conda environment, which pins
versions the tests pass with:

```bash
git clone git@github.com:admoorewx/CM1-analysis.git
cd CM1-analysis
conda env create -f environment.yml
conda activate cm1analysis
```

## Usage

```bash
# Grid, fields, and which plots this file supports
cm1plot inventory /path/to/cm1out_000120.nc

# All registered plots and the fields they need
cm1plot list
```

CM1's output namelist decides which fields a file holds. Every plot declares
the fields it reads, so the pipeline runs whatever a file supports and reports
the rest as skipped, with the missing fields named, instead of stopping.

### Configuration

Paths, experiment groups and plot options come from a YAML file. Start from
[`config/example.yaml`](config/example.yaml); put real configurations in
`config/local/`, which git ignores. Paths may use environment variables such
as `$SCRATCH`, and unknown keys are reported as errors.

### HPC jobs

[`jobs/`](jobs/) holds example SLURM scripts with placeholders. Copy them to
`jobs/local/` (ignored) before filling in accounts, emails and paths.

## Layout

```
src/cm1analysis/
  io.py         CM1File reader: grid, staggering, field lookup, run file lists
  registry.py   Plot registry and per-file field checks
  config.py     YAML configuration
  style.py      Shared colormaps, font sizes and figure saving
  cli.py        cm1plot command
  plots/        One module per category (being ported)
tests/          pytest suite using small synthetic CM1 files
```

## Development

```bash
pytest              # tests; no model output needed
ruff check .        # lint
ruff format .       # format
```

CI runs both on every push and pull request.

## License

[MIT](LICENSE)
