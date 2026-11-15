# Quickstart

<b>YAML configuration file:</b>  Generating a catalogue requires a configuration YAML file that contains all of the parameters and settings necessary to run the pipeline. A baseline configuration file can be automatically generated in your current working directory from the terminal:

```bash
% octavius init
```

After changing the paths appropriately, the YAML file should be fully functional.  The YAML file entries are described in more detail in the User's Manual.

<b>Photometry:</b> To compute magnitudes, Octavius requires an Single Stellar Population (SSP) spectrum file lookup table generated with [Python-FSPS](https://github.com/dfm/python-fsps).  Octavius ships with a lookup table constructed for a standard Chabrier IMF including emission lines, set as the YAML file default.

Octavius provides a method to generate your own table with any desired FSPS options.  First install [Python-FSPS](https://github.com/dfm/python-fsps), and then do as follows:

```python
from pathlib import Path  # pathlib is in the standard library
from octavius import generate_photometry_table_from_sp
import fsps

photometry_table_filepath = Path("/path/to/table.hdf5")
sp = fsps.StellarPopulation(...)  # full control over FSPS options; see FSPS docs
generate_photometry_table_from_sp(photometry_table_filepath, sp)
```

<b>Catalogue Generation:</b> Snapshots can be analysed from either the command line or within a Python script: both methods are straightforward. 

For large snapshots and more complex workflows, it is recommended to run from the command line for greater MPI flexibility (don't forget to set the number of threads per MPI task in the YAML file):

```bash
% octavius analyse --help  # for a list of command-line arguments
% mpirun -np N_TASKS octavius analyse -c /path/to/config.yaml
```

:::{note}
The mpirun command must use your native Python MPI, NOT your system MPI used for running C/C++/FORTRAN codes.  For instance, if you use the anaconda package, it is likely to be in `${HOME}/anaconda3/bin/mpirun`.
:::

To run the catalogue generation within a Python script, import the `analyse_snapshot` function:

```python
from pathlib import Path
from octavius import analyse_snapshot, OctaviusConfig

config_filepath = Path("/path/to/config.yaml")
config = OctaviusConfig.from_yaml(config_filepath)  # can optionally add overrides

# you can also type the parameters manually
config = OctaviusConfig(...)

catalogue_path = analyse_snapshot(config)
```

Note that to set the number of MPI ranks, you must start your python shell using MPI, as in `mpirun -np N_TASKS python my_script.py`.

<b>Working with catalogues:</b> A convenient object-oriented API is provided for loading computed properties; the full list of properties is detailed in the Features section.  Catalogue objects provide unit and physical/comoving conversions, masks, and metadata information for group properties.

Here we get some basic info and then grab the stellar masses of all galaxies in grams:

```python
from pathlib import Path
from octavius import load_catalogue

catalogue_filepath = Path("/path/to/catalogue.hdf5")

catalogue = load_catalogue(catalogue_filepath)  # returns catalogue object

print(catalogue)  # shows basic catalogue information
catalogue.galaxies.describe()  # lists datasets available in galaxies

# get some simulation info, namely the box size in physical Mpc
boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)    

# array of galaxy stellar masses in grams
stellar_mass_grams = catalogue.galaxies.get_dataset("mass_star", to_units="g")
```

Octavius provides an easy way to traverse the halo hierarchy via stored index masks.  
Here we get the total stellar mass within the haloes of all galaxies:

```python
# get the stellar masses of their field haloes
field_halo_idx = catalogue.galaxies.get_membership(name="field_halo_index")  # indices into halo data
halo_star_mass_grams = catalogue.haloes.get_dataset("mass_star", to_units="g", mask=field_halo_idx)
```

The catalogues also store the list of particle indices for every object, and it is possible to retrieve a specific property for all particles in an object:

```python
# get the indices into the original snapshot of stars in galaxy 2
star_indices = catalogue.galaxies.get_particle_indices(ptype="star", group_index=2)

# get the iron abundance of every gas particle in haloes 0, 5, and 17.
iron_abundances = analyser.get_particle_data(name="fraction_Fe", ptype="gas", group_type='haloes', group_index=[0, 5, 17])
```

The Example Scripts section shows more detailed use cases.


