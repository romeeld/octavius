# Mock X-ray Observations

Octavius can make mock X-ray observations of haloes with [moxha](https://github.com/Fred-Jennings/moxha), written by Fred Jennings and included in Octavius (`octavius.xray.moxha`) with his permission. Octavius reads the gas around a halo from the snapshot, through the same readers as the pipeline (so this works for every supported simulation type), and hands it to moxha as an in-memory yt dataset. moxha then generates photons from the gas with pyXSIM and observes them with the instrument models of SOXS, producing event files, images, spectra and profiles.

## Installation

The X-ray modules need packages that Octavius does not otherwise use, so they are optional extras:

```bash
pip install "octavius[xray]"           # mock observations: yt, pyXSIM, SOXS and others
pip install "octavius[xray-fitting]"   # also moxha's profile and spectral fits and tessellated maps
```

The fitting extra adds threeML, astromodels, emcee, lmfit and others. Using a module whose packages are missing gives an error saying which extra to install.

## Data files

SOXS downloads the instrument files it needs (responses, backgrounds, PSFs) the first time they are used, into its data directory: your cache directory by default (e.g. `~/.cache/soxs`), or the directory given by the `SOXS_DATA_DIR` environment variable or SOXS's configuration file. yt's X-ray emissivity fields need an APEC emissivity table, which yt does not download itself, so Octavius fetches it from yt's data server into the same directory on first use. On a cluster whose compute nodes have no internet access, run a small observation on a login node first, or set `SOXS_DATA_DIR` to a directory where the files have already been downloaded.

## Observing a halo

The catalogue must contain `r500c` and `m500c`, which it does when 500 is among the `virial_factors` in the configuration (the default).

```python
import unyt
import octavius as oc

catalogue = oc.load_catalogue("/path/to/catalogue.hdf5")
analyser = oc.build_analyser(catalogue=catalogue)

ebins = [[0.5, 2.0], [0.5, 7.0]]  # keV bands for the maps and total luminosities

observation = analyser.xray_observation(
    0,                       # halo index
    radius_factor=5,         # read the gas within 5 r500c
    save_dir="xray_mocks",
    run_ID="my_run",
    emin=0.05,
    emax=10,
    emin_for_EW_values=0.5,  # band whose emission weights the profiles
    emax_for_EW_values=2.0,
    energies_for_Lx_tot=ebins,
    redshift=0.05,           # redshift at which the halo is observed (a z = 0 snapshot needs one above 0)
)

observation.add_instrument(Name="chandra_acisi_cy0", ID="chandra_1000ks", exp_time=(1000, "ks"), reblock=1)
observation.add_cut(field=("gas", "temperature"), gtr_than=2e5 * unyt.K)
observation.add_cut(field=("gas", "DelayTime"), equals=0)  # exclude wind particles

observation.MakePhotons(sphere_R500s=5, area=(2.5, "m**2"), model="CIE APEC", overwrite=True)
observation.ObservePhotons(
    image_energies=[{"name": f"{emin}_{emax}_keV", "emin": emin, "emax": emax} for emin, emax in ebins],
    overwrite=True,
)
```

`xray_observation()` returns a moxha `Observation`, so everything moxha offers is available from it. To work with the gas directly instead, `analyser.get_xray_gas(halo_index)` returns it (with the halo's centre, r500c, m500c and the other particle types in the same region) in physical units.

By default the gas within `radius_factor` × r500c of the halo centre is read, searching the whole snapshot. `region="halo"` reads only the gas belonging to the halo, which is much quicker for large snapshots but misses gas outside the halo's boundary. `MakePhotons(sphere_R500s=...)` should not exceed `radius_factor`.

## Fitting

moxha's fitting modules are not imported with the rest, as they need the `xray-fitting` extra:

```python
from octavius.xray.moxha import profilefits, spectralfits, tessmaps
```
