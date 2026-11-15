# Quickstart

Once installed, running the analysis requires a configuration YAML file that contains all of the parameters and settings necessary to run the pipeline. A baseline configuration file can be automatically generated in your current working directory from the terminal:

```bash
% octavius init
```

Photometry uses the [Python-FSPS](https://github.com/dfm/python-fsps) package, which is used to build a lookup table for emission from a Single Stellar Population (SSP).  The Octavius package includes a standard SSP table using a Chabrier IMF, but also provides a method to generate your own SSP table using any FSPS options you like:

```python
from pathlib import Path  # pathlib is in the standard library
from octavius import generate_photometry_table, generate_photometry_table_from_sp

photometry_table_filepath = Path("/path/to/table.hdf5")

generate_photometry_table(photometry_table_filepath)  # default method with basic options 

import fsps
sp = fsps.StellarPopulation(...)  # full control over FSPS options
generate_photometry_table_from_sp(photometry_table_filepath, sp)
```

Snapshots can be analysed from either the command line or within a Python script: both methods are straightforward. For large snapshots and more complex workflows, it is recommended to run from the command line for greater MPI flexibility:

```bash
% octavius analyse --help  # for a list of command-line arguments
% mpirun -n [N_TASKS] octavius analyse -c /path/to/config.yaml
```

:::{note}
The mpirun command must use your native Python MPI, NOT your system MPI used e.g. for running C code.  For instance, if you use the anaconda package, it would be in ${HOME}/anaconda3/bin/mpirun.
:::

To run the catalogue generation within a Python script, import the analyse_snapshot function; it is natively MPI-aware:

```python
from pathlib import Path
from octavius import analyse_snapshot, OctaviusConfig

config_filepath = Path("/path/to/config.yaml")
config = OctaviusConfig.from_yaml(config_filepath)  # can optionally add overrides

# you can also type the parameters manually
config = OctaviusConfig(...)

catalogue_path = analyse_snapshot(config)
```

Once the catalogue is generated, a convenient object-oriented API is provided for loading computed properties.  Here we load the stellar masses of all galaxies into a numpy array, converting the values to grams:

```python
from pathlib import Path
from octavius import load_catalogue

catalogue_filepath = Path("/path/to/catalogue.hdf5")

catalogue = load_catalogue(catalogue_filepath)  # returns catalogue object

print(catalogue)  # shows basic cataloguealogue information
catalogue.galaxies.describe()  # lists datasets available in galaxies

# array of galaxy stellar masses in grams
star_mass_grams = catalogue.galaxies.get_dataset("mass_star", to_units="g")
```

Octavius provides an easy way traverse the halo hierarchy via stored indices.  Here we get the total stellar mass within the halos of those galaxies:

```python
# get the stellar masses of their field haloes
field_halo_idx = catalogue.galaxies.get_membership(name="field_halo_index")  # indices into halo data
halo_star_mass_grams = catalogue.haloes.get_dataset("mass_star", to_units="g", mask=field_halo_idx)
```

Catalogue objects provide unit and physical/comoving conversions, masks, and metadata information for group properties, as well as simulation metadata and cosmological information.
They further store the list of particle indices for every galaxy and (sub)halo:

```python
# get simulation info
boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)    

# get the indices into the original snapshot of stars in galaxy 0
star_indices = catalogue.galaxies.get_particle_indices(ptype="star", group_index=0)

```

The Example Scripts section show more detailed use cases.


