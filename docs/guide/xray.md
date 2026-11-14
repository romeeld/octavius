# Mock X-ray Observations

Every halo in an Octavius catalogue already has the X-ray luminosities and temperatures of its hot gas (see the halo outputs in {doc}`../features/aggregate_properties`), computed from the same APEC emissivities with no extra packages. For individual haloes, Octavius can also make mock X-ray observations with [moxha](https://github.com/Fred-Jennings/moxha), written by Fred Jennings and included in Octavius (`octavius.xray.moxha`) with his permission. Octavius reads the gas around a halo from the snapshot, through the same readers as the pipeline (so this works for every supported simulation type), and hands it to moxha as an in-memory yt dataset. moxha then generates photons from the gas with pyXSIM and observes them with the instrument models of SOXS, producing event files, images, spectra and profiles.

If you use the mock X-ray observations, please cite the moxha paper: Jennings, F. & Davé, R. (2023), MNRAS, 526, 1367.

## Installation

The X-ray modules need packages that Octavius does not otherwise use, so they are optional extras:

```bash
pip install "octavius[xray]"           # mock observations: yt, pyXSIM, SOXS and others
pip install "octavius[xray-fitting]"   # also spectral fits and tessellated maps
```

The fitting extra adds emcee and vorbin (no XSPEC or HEASoft installation is needed). Using a module whose packages are missing gives an error saying which extra to install.

## Data files

SOXS downloads the instrument files it needs (responses, backgrounds, PSFs) the first time they are used, into its data directory: your cache directory by default (e.g. `~/.cache/soxs`), or the directory given by the `SOXS_DATA_DIR` environment variable or SOXS's configuration file. yt's X-ray emissivity fields need an APEC emissivity table, which yt does not download itself, so Octavius fetches it from yt's data server into the same directory on first use. On a cluster whose compute nodes have no internet access, run a small observation on a login node first, or set `SOXS_DATA_DIR` to a directory where the files have already been downloaded.

## Observing a halo

A complete script, from the snapshot to fitted radial profiles, is in the [examples](../examples/xray_observations.md).

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

By default (`region="sphere"`) the gas within `radius_factor` × r500c of the halo centre is read, searching the whole snapshot. `region="halo"` reads only the gas belonging to the halo, which is much quicker for large snapshots but misses gas outside the halo's boundary. `MakePhotons(sphere_R500s=...)` should not exceed `radius_factor`.

## Fitting

moxha's fitting modules are not imported with the rest, as they need the `xray-fitting` extra:

```python
from octavius.xray.moxha import spectralfits, tessmaps
```

moxha's profile fitting and scaling-relation analysis (`profilefits`) is not included in Octavius; it remains available in [moxha's repository](https://github.com/Fred-Jennings/moxha).

### Spectral fits

The spectra of the deprojected annuli (`spectralfits.FitRadialSpectra`) and of the tessellated map's tiles (`tessmaps`) are fitted with an absorbed APEC model, TBabs × APEC, built from SOXS: the same AtomDB tables and absorption model that pyXSIM and SOXS use to generate and observe the photons, so the fitted model is the one the mock spectra were made from. The temperature, abundance and normalisation (XSPEC convention) are free, with the redshift and absorbing column fixed. The fit maximises the Cash likelihood (C-statistic, which also suits the non-integer counts of deprojected spectra), and the posterior is sampled with emcee, with flat priors in log kT, abundance and log normalisation. Errors are equal-tailed 68% intervals about the best fit, and posterior samples are saved with each annulus's results.

`FitRadialSpectra.fit_spectra()` groups consecutive channels to at least `min_counts_per_bin` counts before fitting (default 25, a common observational choice; `None` fits every channel). Coarser groups smooth out the line structure that constrains the abundance; where the spectra still cannot constrain it, fix it with `abund` (e.g. `abund=0.3`, as is common in observational analyses); each annulus's results record whether it was fixed (`"abund_fixed"`). Deprojected spectra are not Poisson counts (the background and outer shells have been subtracted), so the C-statistic is a sound fitting criterion for them but not a goodness-of-fit measure.

The fit can also be used directly on any PHA spectrum with its ARF and RMF:

```python
from octavius.xray.spectral_fitting import XraySpectrum, fit_apec_spectrum, plot_spectral_fit

spectrum = XraySpectrum.from_pha("spectrum.pha")
result = fit_apec_spectrum(spectrum, redshift=0.05, emin=0.5, emax=7.0)
print(result.best_fit, result.negative_error, result.positive_error)
plot_spectral_fit(spectrum, result, emin=0.5, emax=7.0, output_path="fit.png")
```

The model is folded through the instrument response once per instrument and redshift (about 10 s), after which each fit, including its posterior, takes a second or two.

Each annulus's results also include its X-ray luminosities: the unabsorbed luminosity of the fitted model in each energy band set in the `xray` section of the configuration, with errors from the posterior samples:

```yaml
xray:
  lx_bands:   [[0.5, 2.0], [0.1, 2.4]]  # [emin, emax] in keV
  lx_frame:   REST                      # or OBSERVED
```

The bands are plain integrals of the model spectrum, with no instrument response. With `REST`, a band is the luminosity emitted between `emin` and `emax` in the source's frame (so K-corrected); with `OBSERVED`, it is the luminosity implied by the flux observed between `emin` and `emax`. The luminosities are stored under `"Lx"` in the fitted data, keyed by band (e.g. `"Lx_0.5_2keV"`), in erg/s. For deprojected annuli they are the luminosities of the spherical shells.

Each annulus's electron density is stored under `"ne"` (cm$^{-3}$), from its APEC normalisation as $n_e = \sqrt{10^{14}\, 4\pi\, \mathrm{norm}\, [D_A(1+z)]^2 / (0.82\, V)}$, where $n_H = 0.82\, n_e$ and $V = \frac{4\pi}{3}(r_{\rm out}^2 - r_{\rm in}^2)^{3/2}$ is the volume of the spherical shell within the annulus (the volume a deprojected annulus's emission comes from), with errors from the posterior samples of the normalisation.

