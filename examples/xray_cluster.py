"""

Mock X-ray observation of a cluster, deprojected into annuli and fitted, giving radial profiles of temperature,
abundance, electron density and X-ray luminosity (tested on a SIMBA-C cluster with M500 = 2.6e14 Msun at z = 0.1,
observed for 50 ks with Chandra ACIS-I: five fitted annuli in about 10 minutes on one node).

Steps:
  1. Octavius reads the cluster's gas from the snapshot and hands it to moxha as a yt dataset.
  2. moxha (pyXSIM) generates photons from the hot gas and (SOXS) observes them with Chandra ACIS-I, with the
     instrumental and Galactic foreground backgrounds, plus a matching blank-sky observation.
  3. moxha deprojects the observation into annuli, each with enough signal to noise.
  4. Each annulus's spectrum is fitted with an absorbed APEC model (posteriors from emcee).

Needs: pip install "octavius[xray-fitting]" (yt, pyXSIM, SOXS, emcee, ...). On a cluster without internet on the
compute nodes, run once on a login node first so SOXS can download its instrument files.

Usage: python xray_cluster.py

"""

import glob
import os

import numpy as np
import octavius as oc
import unyt
from octavius.xray.moxha.postprocess import PostProcess
from octavius.xray.moxha.spectralfits import FitRadialSpectra

######################################################
####################  Edit these  ####################
######################################################

CATALOGUE = "/path/to/catalogue.hdf5"  # an Octavius catalogue with r500c and m500c (500 in virial_factors)
OUT = "./xray_mocks"  # output directory (photons, events, images, spectra, fits)
HALO = 0  # catalogue index of the cluster
REDSHIFT = 0.1  # redshift at which the cluster is observed
EXPOSURE = 50  # ks
INSTRUMENT = "chandra_acisi_cy0"  # any SOXS instrument
AREA = (0.08, "m**2")  # collecting area for photon generation; must exceed the instrument's peak effective area
SPHERE_R500S = 2  # gas within this many R500 is read (region="sphere") and makes photons

S2N_THRESHOLD = 5  # annuli: S/N per channel that a fraction S2N_GOOD_FRACTION of the channels in S2N_RANGE must reach
S2N_RANGE = (1, 2)  # keV
S2N_GOOD_FRACTION = 0.7
MAX_ANN_WIDTH = 60  # arcsec, maximum width of the outermost annulus

FIT_EMIN, FIT_EMAX = 0.5, 7.0  # keV, energy range of the spectral fits
FIXED_ABUNDANCE = None  # e.g. 0.3 to fix the abundance (solar) instead of fitting it
MIN_COUNTS_PER_BIN = 25  # channels grouped to at least this many counts for the fits

LX_BANDS = [[0.5, 2.0], [0.1, 2.4]]  # keV, bands for the X-ray luminosities
LX_FRAME = "REST"  # REST (K-corrected) or OBSERVED

######################################################

OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)

# 1. the cluster's gas, read from the snapshot through Octavius; region="halo" reads only the halo's own particles,
#    which is quick even for a very large snapshot (region="sphere" searches the whole snapshot for gas out to
#    SPHERE_R500S * R500)
catalogue = oc.load_catalogue(CATALOGUE)
analyser = oc.build_analyser(catalogue=catalogue)
analyser.update_config(lx_bands=LX_BANDS, lx_frame=LX_FRAME)

instrument = {"Name": INSTRUMENT, "ID": f"{INSTRUMENT}_{EXPOSURE}ks", "exp_time": (EXPOSURE, "ks"), "reblock": 1}
obs = analyser.xray_observation(
    HALO,
    radius_factor=SPHERE_R500S,
    region="halo",
    save_dir=OUT,
    run_ID="cluster",
    emin=0.05,  # keV, energy range of the photons
    emax=10,
    emin_for_EW_values=0.5,  # band whose emission weights moxha's profiles
    emax_for_EW_values=2.0,
    energies_for_Lx_tot=LX_BANDS,  # moxha's "true" luminosities from the photons, for comparison
    redshift=REDSHIFT,
)
obs.add_instrument(**instrument)

# only the hot, diffuse gas emits: no dense (ISM) or star-forming gas, no cool gas, no wind particles
obs.add_cut(field=("gas", "density"), less_than=(0.01 * unyt.physical_constants.proton_mass) * unyt.cm**-3)
obs.add_cut(field=("gas", "temperature"), gtr_than=2e5 * unyt.K)
obs.add_cut(field=("gas", "DelayTime"), equals=0)
obs.add_cut(field=("gas", "StarFormationRate"), equals=0)

# 2. photons, the observation (with instrumental and foreground backgrounds) and a blank sky
metals = [f"{element}_metallicity" for element in ["He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe"]]
obs.MakePhotons(
    metals=metals,  # needed: moxha has no default
    sphere_R500s=SPHERE_R500S,
    area=AREA,
    model="CIE APEC",
    photon_sample_exp=1.5 * EXPOSURE,  # ks, must exceed the exposure
    overwrite=True,
    make_profiles=False,
    make_phaseplots=False,
)
obs.ObservePhotons(
    image_energies=[{"name": "0.5_2.0_keV", "emin": 0.5, "emax": 2.0}],
    overwrite=True,
    instr_bkgnd=True,
    foreground=True,
    ptsrc_bkgnd=False,
)
obs.BlankSkyBackgrounds(1)

# 3. deprojected annuli
pp = PostProcess(obs)
pp.clear_instruments()
pp.add_instrument(**instrument)
pp.generate_annuli(
    S2N_threshold=S2N_THRESHOLD,
    S2N_range=S2N_RANGE,
    S2N_good_fraction=S2N_GOOD_FRACTION,
    max_ann_width=MAX_ANN_WIDTH,
    overwrite=True,
)

# 4. spectral fits of the annuli
sf = FitRadialSpectra(obs)
sf.fit_spectra(
    fit_emin=FIT_EMIN,
    fit_emax=FIT_EMAX,
    overwrite=True,
    abund=FIXED_ABUNDANCE,
    min_counts_per_bin=MIN_COUNTS_PER_BIN,
)

# the results: one entry per fitted annulus, each quantity with value, negative_error, positive_error and unit
fitted = glob.glob(f"{OUT}/cluster/{INSTRUMENT}/ANNULI/*_h{HALO:03d}_*/DATA/*_fitted_data.npy")
if not fitted:
    raise SystemExit("No annuli were fitted (see the logs in OUT/cluster/LOGS).")

lx_names = [f"Lx_{emin:g}_{emax:g}keV" for emin, emax in LX_BANDS]
print(f"{'r_in':>6} {'r_out':>6}   {'kT':>15}   {'Z':>15}   {'n_e':>21}   " + "   ".join(f"{n:>16}" for n in lx_names))
print(
    f"{'kpc':>6} {'kpc':>6}   {'keV':>15}   {'solar':>15}   {'cm^-3':>21}   "
    + "   ".join(f"{'erg/s':>16}" for _ in lx_names)
)


def show(quantity: dict, fmt: str) -> str:
    return f"{quantity['value']:{fmt}} {quantity['negative_error']:+.1g}/{quantity['positive_error']:+.1g}"


for annulus in np.load(fitted[0], allow_pickle=True):
    r_in, r_out = (float(r.to("kpc").value) for r in annulus["radii"])
    print(
        f"{r_in:6.0f} {r_out:6.0f}   {show(annulus['kT'], '.2f'):>15}   {show(annulus['abund'], '.2f'):>15}   "
        f"{show(annulus['ne'], '.2e'):>21}   " + "   ".join(f"{show(annulus['Lx'][n], '.2e'):>16}" for n in lx_names)
    )
