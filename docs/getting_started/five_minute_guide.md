# Five Minute Guide

## What is Octavius?

Octavius is a galaxy formation simulation analysis toolkit designed to build analysis catalogues of haloes and galaxies from simulation snapshots. As simulations continue to grow in size and complexity, the analysis of their outputs for science use has become its own big data computational challenge. Octavius provides a parallel, high-performance post-processing pipeline that produces a lightweight (<1% the size of the original snapshot) catalogue of many galaxy and halo properties, thus bridging the gap between simulations and analysis. The package is designed to be fast, lean and lightweight to slot cleanly into analysis workflows.

## How do I get it?

Octavius is available on pip:

```bash
pip install octavius
```

By design, the package has minimal and well-supported dependencies; it is written entirely in Python.

:::{note}
The Octavius Analyser package for post-catalog analysis may require additional dependencies specific to a particular task.  This will be noted in the documentation for that Analyser package.
:::

## Which snapshots and halo catalogues are supported?

Octavius currently supports a variety of snapshot formats and corresponding halo catalogues. The code is designed to be trivially extensible: new simulation types and halo catalogues can be supported in as few as fifty lines of code. Currently, the [SWIFT ecosystem](https://swift.strw.leidenuniv.nl/) is well-supported, as are [SIMBA](https://arxiv.org/abs/1901.10203) snapshots. Provisional support currently exists for [IllustrisTNG](https://www.tng-project.org/).

[HBT-HERONS](https://hbt-herons.strw.leidenuniv.nl/), [AHF](https://arxiv.org/abs/0904.3662), and [SUBFIND](https://www.tng-project.org/data/docs/specifications/#sec2b) catalogues are supported with full hierarchy mapping and inclusive properties. If a snapshot has no halo assignments at all, `FOF` runs a built-in friends-of-friends halo finder instead; `SNAP_OR_FOF` does so only when the snapshot has no halo IDs, and with `write_halo_ids` the FOF results are saved to the snapshot for future runs.

The full current list of supported formats is as follows:

| Snapshot  | Halo Catalogue |
| ------------- | ------------- |
| SWIFT-KIARA  | HBT-HERONS, AHF, SNAPSHOT, FOF, SNAP_OR_FOF  |
| SWIFT-EAGLE  | HBT-HERONS, AHF, SNAPSHOT, FOF, SNAP_OR_FOF |
| SWIFT-COLIBRE  | HBT-HERONS, AHF, SNAPSHOT, FOF, SNAP_OR_FOF  |
| SIMBA  | AHF, SNAPSHOT, FOF, SNAP_OR_FOF  |
| SIMBA-C  | AHF, SNAPSHOT, FOF, SNAP_OR_FOF  |
| TNG  | SUBFIND, FOF  |

:::{note}
If your simulation or halo finder is not listed, we provide a Developer's Guide that will (in most cases) quickly and easily allow you to add it in.  We gratefully accept pull requests.
:::

## How does it work?

The catalogue generation pipeline is designed around considering haloes as a unit of work. To this end, it starts by determining which particles in the snapshot are in haloes; then, using a [simple binning algorithm](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling), it divides haloes amongst ranks such that the computational load is balanced. A [simple topological sorting algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm) is run to determine the most efficient way to run the user-requested pipeline stages. Assuming all stages are enabled, the analysis proceeds as follows:

- Run a 6D friends-of-friends algorithm to locate galaxies within haloes
- Construct membership mapping between haloes and galaxies, and their constituent particles
- Compute an assortment of aggregate properties for all groups
- Compute photometric properties for galaxies, including line-of-sight extinction to each star
- Synthesise the outputs from the pipeline into a lightweight HDF5 catalogue

:::{note}
Data is dynamically loaded and released between stages to minimise the memory footprint. Stable sorting algorithms are used in conjunction with a global sort order to ensure the outputs of a catalogue are invariant and deterministic.
:::

## How fast is it?

The pipeline is primarily written in [numba](https://numba.readthedocs.io/en/stable/user/overview.html), a [just-in-time](https://en.wikipedia.org/wiki/Just-in-time_compilation) (JIT) compiler for Python code which achieves machine-level speed. The computations are therefore competitive with low-level languages. 

Indicative benchmark: On a 192-core node using 48 MPI ranks with 4 threads each, catalog generation for the flagship SIMBA $z=0$ snapshot with $10^{9}$ gas and $10^{9}$ dark matter particles took under 3 minutes, including running an initial 3DFoF to identify halos.  For comparison, the predecessor Caesar package took 3-4 hours.

:::{tip}
Our tests suggest that it is optimal to use a fairly small number of threads (e.g. 4) and distribute the workload over more cores using MPI ranks.
:::

## How much memory does it need?

In addition to the raw speed advantage, numba avoids the intermediate array allocations which the same code in idiomatic numpy would produce. The memory profile is therefore lean relative to the size of the snapshots being analysed.  On the aforementioned 241GB flagship SIMBA snapshot, the peak memory usage was about 150GB. 

## How do I know it works?

Octavius contains unit tests for its functions to ensure the obtained results are accurate. Furthermore, it contains a full validation suite with reference catalogues to catch any bugs or regressions in the code, and verify outputs are both valid and physically sensible. The codebase contains an authoritative astropy-backed internal source for its units and constants to ensure dimensional self-consistency. The validation suite includes detailed runtime and memory diagnostic profiling which is run on large snapshots to ensure the code is performant and scaleable.

Commits and pull requests must pass a continuous-integration (CI) workflow, which runs automated tests and codebase linting to prevent any bugs from reaching production code.
