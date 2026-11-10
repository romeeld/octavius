"""

Tests the collection of gas around haloes for mock X-ray observations. The observations themselves need the optional
X-ray dependencies (yt, pyXSIM and SOXS: pip install octavius[xray]), so the test of the moxha dataset is skipped
without them.

"""

import importlib
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import h5py
import pytest

from octavius import load_catalogue, build_analyser
from octavius.xray.data_files import xray_data_dir
from octavius.xray.extras import (
    SPECTRAL_FITTING_PACKAGES,
    TESSELLATION_PACKAGES,
    require_packages,
)
from octavius.xray.spectral_fitting import XraySpectrum, _channel_groups, _cstat, apec_table, fit_apec_spectrum
from octavius.xray.xray_gas import _flag_within_radius


def test_flag_within_radius() -> None:
    """
    Checks the sphere search uses the nearest periodic image.

    AI-generated: claude-opus-5-5
    """
    boxsize = 100.0
    centre = np.array([1.0, 50.0, 50.0])
    positions = np.array(
        [
            [99.0, 50.0, 50.0],  # 2 away across the boundary
            [5.0, 50.0, 50.0],  # 4 away
            [50.0, 50.0, 50.0],  # 49 away
            [1.0, 53.0, 53.0],  # sqrt(18) ~ 4.24 away
        ]
    )

    within = _flag_within_radius(positions=positions, centre=centre, radius=4.1, boxsize=boxsize)
    np.testing.assert_array_equal(within, [True, True, False, False])


def test_get_xray_gas(mock_catalogue: h5py.File) -> None:
    """
    Checks the gas read around a halo against reading the whole snapshot, and its conversion to physical units.

    AI-generated: claude-opus-5-5
    """
    cat = load_catalogue(Path(mock_catalogue.filename))
    analyser = build_analyser(catalogue=cat)
    reader = analyser._reader
    scale_factor = reader.simulation_attributes.scale_factor
    boxsize = reader.simulation_attributes.boxsize

    gas = analyser.get_xray_gas(halo_index=0, radius_factor=5.0)

    # count the gas within the sphere by brute force over the whole snapshot (comoving code units)
    all_positions = reader.read_full_dataset(ptype="gas", dataset="pos")
    offsets = all_positions - gas.centre / scale_factor
    offsets -= boxsize * np.round(offsets / boxsize)
    distances = np.sqrt(np.sum(offsets**2, axis=1))
    assert gas.n_particles == np.sum(distances <= gas.region_radius / scale_factor)

    # positions are physical and unwrapped around the centre, so all lie within the sphere
    if gas.n_particles > 0:
        physical_distances = np.sqrt(np.sum((gas.columns["pos"] - gas.centre) ** 2, axis=1))
        assert np.max(physical_distances) <= gas.region_radius * (1 + 1e-10)

    # the other matter is read from the same region
    for matter in gas.other_matter.values():
        assert len(matter["pos"]) == len(matter["mass"])

    # the halo's own gas should add up to its catalogue gas mass
    halo_gas = analyser.get_xray_gas(halo_index=0, region="halo")
    catalogue_gas_mass = cat.haloes.get_dataset("mass_gas")[0]
    np.testing.assert_allclose(halo_gas.columns["mass"].sum(), catalogue_gas_mass, rtol=1e-6)

    cat.close()


def test_xray_dataset(mock_catalogue: h5py.File) -> None:
    """
    Checks the yt dataset handed to moxha holds the gas and the total mass that octavius read.

    AI-generated: claude-opus-5-5
    """
    pytest.importorskip("octavius.xray.moxha.particle_dataset", exc_type=ImportError)
    from octavius.xray.xray_observations import dataset_from_xray_gas

    cat = load_catalogue(Path(mock_catalogue.filename))
    analyser = build_analyser(catalogue=cat)
    gas = analyser.get_xray_gas(halo_index=0, region="halo")

    if gas.n_particles > 0:
        ds = dataset_from_xray_gas(gas=gas)
        everything = ds.all_data()

        np.testing.assert_allclose(everything["gas", "mass"].to("Msun").sum().v, gas.columns["mass"].sum(), rtol=1e-6)

        total_mass = gas.columns["mass"].sum() + sum(matter["mass"].sum() for matter in gas.other_matter.values())
        np.testing.assert_allclose(everything["all", "Masses"].to("Msun").sum().v, total_mass, rtol=1e-6)

        # moxha cuts on DelayTime, so it must exist even when the snapshot has no wind data (then nothing is a wind)
        delay_times = everything["gas", "DelayTime"].to("Gyr").v
        if "wind_delay_time" in gas.columns:
            np.testing.assert_allclose(np.sort(delay_times), np.sort(gas.columns["wind_delay_time"]))
        else:
            np.testing.assert_array_equal(delay_times, 0.0)

    cat.close()


def test_require_packages() -> None:
    """
    A missing optional package gives an ImportError naming the package and the extra that installs it.
    """
    require_packages(packages=["numpy"], extra="xray", purpose="Anything")  # installed, so no error

    with pytest.raises(ImportError, match=r"not_a_real_package.*octavius\[xray-fitting\]"):
        require_packages(packages=["numpy", "not_a_real_package"], extra="xray-fitting", purpose="Testing")


def test_fitting_modules_need_their_packages() -> None:
    """
    Importing one of moxha's fitting modules without its packages says to install octavius[xray-fitting].
    """
    pytest.importorskip("octavius.xray.moxha", exc_type=ImportError)
    modules = {
        "spectralfits": SPECTRAL_FITTING_PACKAGES,
        "tessmaps": TESSELLATION_PACKAGES,
    }
    for module, packages in modules.items():
        if all(find_spec(package) is not None for package in packages):
            continue  # installed, so it imports (or fails for other reasons tested elsewhere)
        with pytest.raises(ImportError, match=r"octavius\[xray-fitting\]"):
            importlib.import_module(f"octavius.xray.moxha.{module}")


def test_cstat_and_channel_groups() -> None:
    """
    The C-statistic is zero when the model equals the data, and positive otherwise; grouping collects consecutive
    channels to a minimum number of counts, merging a short last group into the one before.
    """
    data = np.array([0.0, 3.0, 2.5, 10.0])
    assert _cstat(data, data + 1e-300) == pytest.approx(0.0, abs=1e-12)
    assert _cstat(data, data + 1.0) > 0.0
    assert _cstat(data, np.zeros(4)) == np.inf

    counts = np.array([1.0, 1.0, 3.0, 5.0, 0.0, 1.0])
    np.testing.assert_array_equal(_channel_groups(counts, min_counts=4), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(_channel_groups(counts, min_counts=None), np.arange(6))


def _have_soxs_files(file_names: list[str]) -> bool:
    """
    Whether SOXS's data directory already holds these files (so the test does not download ~400 MB of APEC tables).
    """
    pytest.importorskip("soxs")
    pytest.importorskip("emcee")
    directory = xray_data_dir()
    return all((directory / name).exists() for name in file_names)


APEC_TEST_FILES = ["apec_v3.1.3_coco.fits", "apec_v3.1.3_line.fits", "tbabs_table.h5"]
CHANDRA_TEST_FILES = ["acisi_aimpt_cy0.arf", "acisi_aimpt_cy0.rmf"]


def simulate_chandra_spectrum(path: Path, kT: float, abund: float, norm: float, noisy: bool) -> None:
    """
    Writes a 50 ks Chandra ACIS-I spectrum of TBabs x APEC at z = 0.05, simulated by SOXS.
    """
    import soxs

    generator = soxs.ApecGenerator(0.05, 12.0, 23900, broadening=True)
    spectrum = generator.get_spectrum(kT, abund, 0.05, norm)
    spectrum.apply_foreground_absorption(0.018, model="tbabs")
    soxs.simulate_spectrum(spectrum, "chandra_acisi_cy0", (50.0, "ks"), str(path), overwrite=True, noisy=noisy, prng=3)
    generator.line_handle.close()  # SOXS leaves its APEC tables open
    generator.coco_handle.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")  # SOXS leaves files open
def test_apec_model_matches_soxs(tmp_path: Path) -> None:
    """
    The folded model used for fitting gives the counts that SOXS simulates for the same spectrum (without noise).
    """
    if not _have_soxs_files(APEC_TEST_FILES + CHANDRA_TEST_FILES):
        pytest.skip("SOXS's APEC tables and Chandra responses are not downloaded")

    simulate_chandra_spectrum(tmp_path / "exact.pha", kT=2.0, abund=0.3, norm=1e-3, noisy=False)
    spectrum = XraySpectrum.from_pha(tmp_path / "exact.pha")
    model = apec_table(spectrum.rmf_path, spectrum.arf_path, 0.05, 0.018).rate(2.0, 0.3, 1e-3) * spectrum.exposure

    bright = spectrum.counts > 1e-3 * spectrum.counts.max()
    np.testing.assert_allclose(model[bright], spectrum.counts[bright], rtol=0.01)
    np.testing.assert_allclose(model.sum(), spectrum.counts.sum(), rtol=0.002)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")  # SOXS leaves files open
def test_fit_recovers_parameters(tmp_path: Path) -> None:
    """
    Fitting a simulated spectrum recovers the temperature, abundance and normalisation it was made with.
    """
    if not _have_soxs_files(APEC_TEST_FILES + CHANDRA_TEST_FILES):
        pytest.skip("SOXS's APEC tables and Chandra responses are not downloaded")

    truth = {"kT": 2.0, "abund": 0.3, "norm": 1e-3}
    simulate_chandra_spectrum(tmp_path / "noisy.pha", noisy=True, **truth)
    spectrum = XraySpectrum.from_pha(tmp_path / "noisy.pha")
    result = fit_apec_spectrum(spectrum, redshift=0.05, emin=0.5, emax=7.0, n_steps=1000, n_burn=300, seed=1)

    for name, value in truth.items():
        lower = result.best_fit[name] + 3 * result.negative_error[name]
        upper = result.best_fit[name] + 3 * result.positive_error[name]
        assert lower <= value <= upper, (name, result.best_fit[name], value)
    assert result.parameter_summary("kT")["unit"] == "keV"


def test_hot_pixel_cleaning() -> None:
    """
    The hot-pixel boxes cover exactly their image pixels, so excluding them removes only the events in those pixels;
    the masking correction is n / (n - n_hot), and 1 for an annulus with no pixels left.
    """
    pytest.importorskip("octavius.xray.moxha", exc_type=ImportError)
    from types import SimpleNamespace

    from soxs.events.utils import _region_filter

    from octavius.xray.moxha.postprocess import _hot_pixel_region, _masking_correction

    # the event pixel grid starts at 0.5, so image pixel i (reblock 1) holds 0.5 + i <= x < 1.5 + i
    x = np.array([10.6, 11.4, 11.6, 20.9, 50.0])
    y = np.array([10.6, 11.4, 10.9, 20.9, 50.0])
    events = SimpleNamespace(data={"X": x, "Y": y, "ENERGY": np.ones(len(x))})
    kept = _region_filter(events, _hot_pixel_region([(10, 10), (20, 20)], reblock=1), exclude=True)
    np.testing.assert_array_equal(kept, [False, False, True, False, True])

    assert _masking_correction(10, 2) == pytest.approx(1.25)
    assert _masking_correction(0, 0) == 1.0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")  # SOXS leaves files open
def test_band_luminosities() -> None:
    """
    The luminosity in each band is 4 pi D_L^2 times the model's energy flux, integrated by SOXS, between emin and emax
    divided by 1 + z (rest frame) or between emin and emax (observed frame); with no posterior spread, no errors.
    """
    if not _have_soxs_files(APEC_TEST_FILES):
        pytest.skip("SOXS's APEC tables are not downloaded")
    import soxs

    from octavius.xray.spectral_fitting import SpectralFitResult, band_luminosities

    z, distance_cm = 0.1, 1.5e27
    best = {"kT": 4.0, "abund": 0.3, "norm": 1e-3}
    result = SpectralFitResult(
        best_fit=best,
        negative_error={},
        positive_error={},
        samples=np.tile([best["kT"], best["abund"], best["norm"]], (10, 1)),
        cstat=0.0,
        n_bins=10,
        fixed={"redshift": z, "nH": 0.018},
    )
    generator = soxs.ApecGenerator(0.01, 20.0, 100000, binscale="log", broadening=True)
    spectrum = generator.get_spectrum(best["kT"], best["abund"], z, best["norm"])
    generator.line_handle.close()
    generator.coco_handle.close()

    bands = [[0.5, 2.0], [2.0, 10.0]]
    for frame, scale in (("REST", 1 / (1 + z)), ("OBSERVED", 1.0)):
        luminosities = band_luminosities(result, bands=bands, frame=frame, luminosity_distance_cm=distance_cm)
        for emin, emax in bands:
            expected = 4 * np.pi * distance_cm**2 * spectrum.get_flux_in_band(emin * scale, emax * scale)[1].value
            luminosity = luminosities[f"Lx_{emin:g}_{emax:g}keV"]
            assert luminosity["value"] == pytest.approx(expected, rel=1e-3)
            assert luminosity["negative_error"] == pytest.approx(0.0, abs=1e-6 * expected)
            assert luminosity["positive_error"] == pytest.approx(0.0, abs=1e-6 * expected)


def test_annulus_electron_density() -> None:
    """
    An annulus's electron density follows from its APEC norm: norm = 1e-14 / (4 pi [D_A (1 + z)]^2) n_e n_H V with
    n_H = 0.82 n_e and V the volume of the shell within the annulus's cylinder, (4 pi / 3) (r_out^2 - r_in^2)^(3/2).
    """
    pytest.importorskip("octavius.xray.moxha", exc_type=ImportError)
    pytest.importorskip("emcee")
    from types import SimpleNamespace

    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    from octavius.xray.moxha.spectralfits import FitRadialSpectra

    observation = SimpleNamespace(
        redshift=0.1, hubble=0.68, _run_ID="test", _top_save_path=Path("."), emin=0.05, emax=10,
        active_halos=[], active_instruments=[], _instrument_defaults=[],
    )  # fmt: skip
    fits = FitRadialSpectra(observation)
    r_in, r_out, norm = 100 * u.kpc, 200 * u.kpc, 1e-3

    distance = FlatLambdaCDM(H0=68, Om0=0.3, Ob0=0.048, Tcmb0=2.7).angular_diameter_distance(0.1) * 1.1
    volume = (4 / 3) * np.pi * (r_out**2 - r_in**2) ** 1.5
    expected = np.sqrt(norm * u.cm**-5 * 1e14 * 4 * np.pi * distance**2 / (0.82 * volume)).to(u.cm**-3).value

    assert fits._electron_density(norm, r_in, r_out) == pytest.approx(expected, rel=1e-10)
    assert 1e-4 < expected < 1e-2  # a sensible intracluster density


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")  # SOXS leaves files open
def test_fit_with_fixed_abundance(tmp_path: Path) -> None:
    """
    With the abundance fixed, the fit keeps it at that value (with no error) and still recovers the temperature.
    """
    if not _have_soxs_files(APEC_TEST_FILES + CHANDRA_TEST_FILES):
        pytest.skip("SOXS's APEC tables and Chandra responses are not downloaded")

    simulate_chandra_spectrum(tmp_path / "noisy.pha", kT=2.0, abund=0.3, norm=1e-3, noisy=True)
    spectrum = XraySpectrum.from_pha(tmp_path / "noisy.pha")
    result = fit_apec_spectrum(
        spectrum, redshift=0.05, emin=0.5, emax=7.0, abund=0.3, min_counts_per_bin=25, n_steps=1000, n_burn=300, seed=1
    )

    assert result.fixed["abund"] == 0.3
    summary = result.parameter_summary("abund")
    assert (summary["value"], summary["negative_error"], summary["positive_error"]) == (0.3, 0.0, 0.0)
    kT = result.parameter_summary("kT")
    assert kT["value"] + 3 * kT["negative_error"] <= 2.0 <= kT["value"] + 3 * kT["positive_error"]
