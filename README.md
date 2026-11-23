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

> **Status:** all six categories are ported from the original analysis
> scripts and tested; `cm1plot list` shows every plot, the fields it needs
> and its options.

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

# All registered plots, the fields they need, and their options
cm1plot list --category planar

# Every planar plot for one run (a directory of cm1out_*.nc, or specific files)
cm1plot run planar /path/to/control -o images/

# A few plots at chosen times, quickly
cm1plot run planar /path/to/control -o images/ --plots cref,overview --times 7200,14400 --dpi 100

# An experiment group from the config: each run's plots plus comparisons with the reference run
cm1plot run planar -c config/local/derecho.yaml -e lapse_rate

# Storm-following west-east cross sections for one run
cm1plot run cross_section /path/to/control -o images/

# Domain-reduced timeseries (figures plus CSV of the values); with -e, one line per run
cm1plot run timeseries -c config/local/derecho.yaml -e lapse_rate

# Swath and cold-pool distributions and effect sizes against the reference run
cm1plot run statistics -c config/local/derecho.yaml -e lapse_rate --scope experiment

# Pressure (DVPGF / effective buoyancy) and wind (Helmholtz) decompositions
cm1plot run pressure /path/to/control -c config/local/derecho.yaml --times 7200
cm1plot run wind /path/to/control -o images/
```

Cross sections follow the system: by default each time's section runs
through the bow apex (the forward-most point of the surface cold pool), with
most of the window behind the gust front, and is averaged over 40 km across
the line. Storm-relative winds use the system speed c_x, a line fit to the
leading edge's position over an hour. `y_km`, `window_km`,
`trailing_fraction`, `average_km` and `storm_u` in the config override
these; see `cm1analysis/sections.py`. The momentum-transport plot
(`cmt_section`) follows Mahoney et al. (2009, MWR 137, 3316).

The `pressure` and `wind` decompositions solve elliptic problems per output
time. Each plot computes its decomposition once and caches it as netCDF
under `<output>/<run>/<category>/derived/`; later runs reuse a cached file
when the stored compute options match (set `recompute: true` to force a new
solve). Because there is one derived file per time, HPC jobs can split a run
with `--times`. The pressure solve is expensive at full resolution: restrict
it with `roi_km` and turn off the diagnostic tests (`sensitivity: []`,
`crop_test: false`, `frame_test: false`) for production runs. The wind
partition can also be run on its own with `python -m cm1analysis.wind`.

Plots run at one of three scopes: once per output **time**, once per **run**
(plots that accumulate through time, such as swaths), or once per
**experiment** (comparisons between runs). `--scope time,run,experiment`
selects which to make. Figures are written to
`<output>/<run or experiment>/<category>/<plot>/<plot>_<run>_<time>s.png`.
A plot that fails is reported at the end and the rest still run; the exit
status is nonzero if anything failed.

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
  io.py           CM1File reader: grid, staggering, field lookup, run file lists
  diagnostics.py  Derived fields: low-level wind, surface pressure, buoyancy, column reductions
  registry.py     Plot registry: required fields, scope, options
  pipeline.py     Runs plots over runs, files and times; contexts passed to plots
  sections.py     Storm-following cross-section placement, storm motion, section cache
  stats.py        Effect sizes and sample statistics (Cohen's d, Cliff's delta, K-S D, ...)
  pressure/       Vertical momentum forcing: Poisson solves for DVPGF and effective buoyancy
  wind/           Lynch (1989) partition of the horizontal wind on the C-grid
  config.py       YAML configuration
  style.py        Shared colormaps, font sizes and figure saving
  cli.py          cm1plot command
  plots/          One module per category
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
