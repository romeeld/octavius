"""

Fits absorbed APEC models to X-ray spectra (OGIP PHA files with their ARF and RMF), replacing moxha's use of threeML
and XSPEC models with SOXS alone.

The model is TBabs x APEC from SOXS: the same AtomDB tables and absorption that pyXSIM and SOXS use to generate and
observe moxha's photons, so the fit model is the one the mock spectra were made from. For a given response, redshift
and absorbing column the model counts in each channel are linear in the normalisation and abundance,

    counts(kT, Z, norm) = norm * [C_HHe(kT) + Z * C_metals(kT)],

so C_HHe and C_metals are folded through the response once, at SOXS's tabulated temperatures, and interpolated
linearly in kT between them exactly as SOXS does. Each model evaluation is then cheap enough for MCMC.

The fit maximises the Cash likelihood (C-statistic, valid for non-integer counts such as deprojected spectra), with
the normalisation profiled analytically, and samples the posterior with emcee (flat priors in log10 kT, the abundance
and log10 norm). Errors are equal-tailed intervals of the posterior around the best fit.

The X-ray packages are optional dependencies (pip install octavius[xray-fitting]), so they are imported inside the
functions.

"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse
from scipy.optimize import minimize

from .data_files import instrument_file

DEFAULT_NH = 0.018  # Galactic absorbing column in 1e22 cm^-2, as moxha used
DEFAULT_KT_RANGE = (0.08, 32.0)  # keV
DEFAULT_ABUND_RANGE = (0.01, 10.0)  # solar (Anders & Grevesse), as moxha used
GENERATOR_BIN_WIDTH = 0.002  # keV; 2 eV resolves absorption edges to <0.5% per channel against a 0.5 eV model
APEC_NORM_FACTOR = 1.0e14  # SOXS's factor turning its tables into XSPEC-normalised spectra
PARAMETER_NAMES = ("kT", "abund", "norm")
PARAMETER_UNITS = ("keV", "", "cm**-5")
LUMINOSITY_BINS = 20000  # log-spaced bins of the unabsorbed model spectrum integrated over each luminosity band
KEV_TO_ERG = 1.602176634e-9


@dataclass
class XraySpectrum:
    """
    A spectrum in detector channels with what is needed to fold a model through its instrument response.

    AI-generated: claude-opus-5-5
    """

    counts: np.ndarray  # counts per channel (need not be integers, e.g. for deprojected spectra)
    exposure: float  # s
    channel_emin: np.ndarray  # keV, nominal energy range of each channel (EBOUNDS)
    channel_emax: np.ndarray
    rmf_path: str
    arf_path: str

    @classmethod
    def from_pha(cls, pha_path: Path | str) -> "XraySpectrum":
        """
        Reads a PHA spectrum and locates the RMF and ARF named in its header (downloaded by SOXS if needed).

        AI-generated: claude-opus-5-5
        """
        from astropy.io import fits
        from soxs.response import RedistributionMatrixFile

        pha_path = Path(pha_path)
        with fits.open(pha_path) as hdul:
            header = hdul["SPECTRUM"].header
            counts = np.asarray(hdul["SPECTRUM"].data["COUNTS"], dtype=np.float64)
            exposure = float(header["EXPOSURE"])
            rmf_name, arf_name = header["RESPFILE"], header["ANCRFILE"]

        rmf_path = _locate_response(rmf_name, pha_path.parent)
        arf_path = _locate_response(arf_name, pha_path.parent)
        rmf = RedistributionMatrixFile(rmf_path)
        channel_emin = np.array(rmf.ebounds_data["E_MIN"], dtype=np.float64)  # copied, as the file is closed below
        channel_emax = np.array(rmf.ebounds_data["E_MAX"], dtype=np.float64)
        n_channels = rmf.n_ch
        rmf.handle.close()  # SOXS leaves the RMF open, which runs out of file handles over many spectra
        if len(counts) != n_channels:
            raise ValueError(f"{pha_path} has {len(counts)} channels but its RMF has {n_channels}.")

        return cls(
            counts=counts,
            exposure=exposure,
            channel_emin=channel_emin,
            channel_emax=channel_emax,
            rmf_path=rmf_path,
            arf_path=arf_path,
        )


@dataclass
class ApecTable:
    """
    TBabs x APEC folded through an instrument response: counts per second per channel, for unit normalisation, of the
    H, He and trace elements (cosmic) and of the metals at solar abundance, at SOXS's tabulated temperatures.

    AI-generated: claude-opus-5-5
    """

    kT_values: np.ndarray  # keV, (n_temperatures,)
    cosmic_rate: np.ndarray  # (n_temperatures, n_channels)
    metal_rate: np.ndarray  # (n_temperatures, n_channels)

    def rate(self, kT: float, abund: float, norm: float) -> np.ndarray:
        """
        Model count rate per channel, interpolated linearly in kT between tabulated temperatures (as SOXS does).

        AI-generated: claude-opus-5-5
        """
        index = int(np.clip(np.searchsorted(self.kT_values, kT) - 1, 0, len(self.kT_values) - 2))
        weight = (kT - self.kT_values[index]) / (self.kT_values[index + 1] - self.kT_values[index])
        cosmic = self.cosmic_rate[index] * (1.0 - weight) + self.cosmic_rate[index + 1] * weight
        metal = self.metal_rate[index] * (1.0 - weight) + self.metal_rate[index + 1] * weight
        return norm * (cosmic + abund * metal)


@dataclass
class SpectralFitResult:
    """
    Best-fit (maximum likelihood) parameters of an absorbed APEC fit, with equal-tailed posterior intervals.

    AI-generated: claude-opus-5-5
    """

    best_fit: dict[str, float]
    negative_error: dict[str, float]  # <= 0: lower end of the interval minus the best fit
    positive_error: dict[str, float]  # >= 0: upper end of the interval minus the best fit
    samples: np.ndarray  # posterior samples, (n_samples, 3) in the order of PARAMETER_NAMES
    cstat: float  # C-statistic at the best fit
    n_bins: int  # number of (grouped) channels fitted
    fixed: dict[str, float] = field(default_factory=dict)  # redshift, nH and any fixed abundance

    def parameter_summary(self, name: str) -> dict:
        """
        The parameter in the form moxha's profile fitting reads (as threeML's results table gave it): value,
        negative_error, positive_error, error and an astropy unit.

        AI-generated: claude-opus-5-5
        """
        import astropy.units as u

        negative, positive = self.negative_error[name], self.positive_error[name]
        return {
            "value": self.best_fit[name],
            "negative_error": negative,
            "positive_error": positive,
            "error": 0.5 * (positive - negative),
            "unit": u.Unit(PARAMETER_UNITS[PARAMETER_NAMES.index(name)]),
        }


def _locate_response(file_name: str, directory: Path) -> str:
    """
    Returns the path to a response file named in a spectrum's header: next to the spectrum if it is there, otherwise
    from SOXS's data directory (downloading it if needed).

    AI-generated: claude-opus-5-5
    """
    local = directory / file_name
    return str(local) if local.exists() else instrument_file(file_name)


def _response_matrix(rmf_path: str) -> tuple[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Returns the RMF's energy bin edges and its redistribution matrix (energy bins x channels) with each row
    normalised to one, as SOXS applies it when simulating spectra (the ARF is applied separately).

    AI-generated: claude-opus-5-5
    """
    from soxs.response import RedistributionMatrixFile

    rmf = RedistributionMatrixFile(rmf_path)
    rows, columns, values = [], [], []
    for energy_index in range(rmf.n_e):
        first_channels = np.atleast_1d(np.nan_to_num(rmf.data["F_CHAN"][energy_index])).astype(np.int64) - rmf.cmin
        n_channels = np.atleast_1d(np.nan_to_num(rmf.data["N_CHAN"][energy_index])).astype(np.int64)
        weights = np.atleast_1d(np.nan_to_num(np.float64(rmf.data["MATRIX"][energy_index])))
        if weights.sum() <= 0.0:
            continue
        weights = weights / weights.sum()
        offset = 0
        for first, n in zip(first_channels, n_channels, strict=True):
            rows.append(np.full(n, energy_index))
            columns.append(np.arange(first, first + n))
            values.append(weights[offset : offset + n])
            offset += n

    matrix = scipy.sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(rmf.n_e, rmf.n_ch)
    )
    energy_bins = np.array(rmf.ebins, dtype=np.float64)  # copied, as the file is closed below
    rmf.handle.close()  # SOXS leaves the RMF open
    return energy_bins, matrix


@lru_cache(maxsize=16)
def apec_table(
    rmf_path: str,
    arf_path: str,
    redshift: float,
    nH: float,
    kT_range: tuple[float, float] = DEFAULT_KT_RANGE,
) -> ApecTable:
    """
    Folds TBabs x APEC through an ARF and RMF at SOXS's tabulated temperatures spanning kT_range, following SOXS's
    simulate_spectrum: the photon spectrum is regridded onto the ARF's energy bins, multiplied by the effective area,
    regridded onto the RMF's energy bins and redistributed into channels. Cached, as every spectrum taken with the
    same instrument at the same redshift shares it.

    AI-generated: claude-opus-5-5
    """
    import soxs
    from soxs.response import AuxiliaryResponseFile
    from soxs.spectra import get_tbabs_absorb
    from soxs.utils import regrid_spectrum

    rmf_energy_bins, response = _response_matrix(rmf_path)
    arf = AuxiliaryResponseFile(arf_path)
    arf_energy_bins = np.asarray(arf.ebins, dtype=np.float64)
    effective_area = np.asarray(arf.eff_area, dtype=np.float64)  # cm^2

    emin = min(rmf_energy_bins[0], arf_energy_bins[0])
    emax = max(rmf_energy_bins[-1], arf_energy_bins[-1])
    n_bins = int(np.ceil((emax - emin) / GENERATOR_BIN_WIDTH))
    generator = soxs.ApecGenerator(emin, emin + n_bins * GENERATOR_BIN_WIDTH, n_bins, broadening=True)
    generator_bins = np.asarray(generator.ebins, dtype=np.float64)
    absorption = get_tbabs_absorb(0.5 * (generator_bins[1:] + generator_bins[:-1]), nH)

    all_kT = np.asarray(generator.Tvals, dtype=np.float64)
    first = max(int(np.searchsorted(all_kT, kT_range[0])) - 1, 0)
    last = min(int(np.searchsorted(all_kT, kT_range[1])) + 1, len(all_kT))
    indices = list(range(first, last))
    # photons per bin (per s per cm^2) at unit norm; _get_table is what SOXS's get_spectrum interpolates between
    cosmic, metals, _ = generator._get_table(indices, redshift, 0.0)
    generator.line_handle.close()  # SOXS leaves its APEC tables open
    generator.coco_handle.close()

    def fold(photons: np.ndarray) -> np.ndarray:
        on_arf_bins = regrid_spectrum(arf_energy_bins, generator_bins, APEC_NORM_FACTOR * photons * absorption)
        rate = on_arf_bins * effective_area
        if len(arf_energy_bins) != len(rmf_energy_bins) or not np.allclose(arf_energy_bins, rmf_energy_bins):
            rate = regrid_spectrum(rmf_energy_bins, arf_energy_bins, rate)
        return response.T @ rate

    return ApecTable(
        kT_values=all_kT[indices],
        cosmic_rate=np.array([fold(row) for row in cosmic]),
        metal_rate=np.array([fold(row) for row in metals]),
    )


@lru_cache(maxsize=16)
def band_flux_table(
    redshift: float,
    bands: tuple[tuple[float, float], ...],
    frame: str,
    kT_range: tuple[float, float] = DEFAULT_KT_RANGE,
) -> ApecTable:
    """
    The unabsorbed APEC energy flux reaching the observer (erg/s/cm^2) in each band, for unit normalisation, of the H,
    He and trace elements and of the metals at solar abundance, at SOXS's tabulated temperatures; its "rate" is then
    the flux in each band. Bands are [emin, emax] in keV: in the source's frame (frame "REST", so the flux observed
    between emin / (1 + z) and emax / (1 + z)) or the observer's ("OBSERVED").

    AI-generated: claude-opus-5-5
    """
    import soxs
    from soxs.utils import regrid_spectrum

    scale = 1.0 / (1.0 + redshift) if frame.upper() == "REST" else 1.0
    observed_bands = [(emin * scale, emax * scale) for emin, emax in bands]
    lowest, highest = min(b[0] for b in observed_bands), max(b[1] for b in observed_bands)
    generator = soxs.ApecGenerator(0.99 * lowest, 1.01 * highest, LUMINOSITY_BINS, binscale="log", broadening=True)
    energy_bins = np.asarray(generator.ebins, dtype=np.float64)
    energies = 0.5 * (energy_bins[1:] + energy_bins[:-1])

    all_kT = np.asarray(generator.Tvals, dtype=np.float64)
    first = max(int(np.searchsorted(all_kT, kT_range[0])) - 1, 0)
    last = min(int(np.searchsorted(all_kT, kT_range[1])) + 1, len(all_kT))
    indices = list(range(first, last))
    cosmic, metals, _ = generator._get_table(indices, redshift, 0.0)  # photons per bin per unit norm, / 1e14
    generator.line_handle.close()  # SOXS leaves its APEC tables open
    generator.coco_handle.close()

    def band_fluxes(photons: np.ndarray) -> np.ndarray:
        energy = APEC_NORM_FACTOR * photons * energies * KEV_TO_ERG  # erg/s/cm^2 per bin
        return np.array(
            [regrid_spectrum(np.array([emin, emax]), energy_bins, energy)[0] for emin, emax in observed_bands]
        )

    return ApecTable(
        kT_values=all_kT[indices],
        cosmic_rate=np.array([band_fluxes(row) for row in cosmic]),
        metal_rate=np.array([band_fluxes(row) for row in metals]),
    )


def band_luminosity_name(emin: float, emax: float) -> str:
    """
    The name of a band's luminosity, e.g. "Lx_0.5_2keV".

    AI-generated: claude-opus-5-5
    """
    return f"Lx_{emin:g}_{emax:g}keV"


def band_luminosities(
    result: SpectralFitResult,
    bands: list[list[float]],
    frame: str,
    luminosity_distance_cm: float,
    kT_range: tuple[float, float] = DEFAULT_KT_RANGE,
    credible_interval: float = 0.68,
    max_samples: int = 5000,
) -> dict[str, dict]:
    """
    The X-ray luminosity (erg/s) of a fitted spectrum's unabsorbed APEC model in each band ([emin, emax] in keV, in
    the source's frame if frame is "REST", else the observer's): 4 pi D_L^2 times the energy flux in the band, at the
    best fit and with equal-tailed errors from (up to max_samples of) the posterior samples. Returned in the form of
    SpectralFitResult.parameter_summary, keyed by band_luminosity_name.

    AI-generated: claude-opus-5-5
    """
    import astropy.units as u

    table = band_flux_table(
        result.fixed["redshift"], tuple((float(lo), float(hi)) for lo, hi in bands), frame.upper(), tuple(kT_range)
    )
    to_luminosity = 4.0 * np.pi * luminosity_distance_cm**2
    best = result.best_fit
    best_luminosity = to_luminosity * table.rate(best["kT"], best["abund"], best["norm"])
    samples = result.samples[:: max(1, len(result.samples) // max_samples)]
    sample_luminosity = to_luminosity * np.array([table.rate(kT, abund, norm) for kT, abund, norm in samples])

    tail = 100.0 * (1.0 - credible_interval) / 2.0
    luminosities = {}
    for column, (emin, emax) in enumerate(bands):
        lower, upper = np.percentile(sample_luminosity[:, column], [tail, 100.0 - tail])
        negative, positive = min(lower - best_luminosity[column], 0.0), max(upper - best_luminosity[column], 0.0)
        luminosities[band_luminosity_name(emin, emax)] = {
            "value": best_luminosity[column],
            "negative_error": negative,
            "positive_error": positive,
            "error": 0.5 * (positive - negative),
            "unit": u.erg / u.s,
        }
    return luminosities


def _channel_groups(counts: np.ndarray, min_counts: float | None) -> np.ndarray:
    """
    Returns a group number for each channel, grouping consecutive channels until each group has at least min_counts
    (the last group is merged into the one before if it falls short); every channel is its own group if min_counts is
    None.

    AI-generated: claude-opus-5-5
    """
    if min_counts is None:
        return np.arange(len(counts))

    groups = np.zeros(len(counts), dtype=np.int64)
    group, total = 0, 0.0
    for channel, count in enumerate(counts):
        groups[channel] = group
        total += count
        if total >= min_counts:
            group, total = group + 1, 0.0
    if total < min_counts and group > 0:  # an unfinished last group joins the previous one
        groups[groups == group] = group - 1
    return groups


def _cstat(data: np.ndarray, model: np.ndarray) -> float:
    """
    The Cash C-statistic, 2 sum[m - d + d ln(d / m)], for (possibly non-integer) counts d and model counts m.

    AI-generated: claude-opus-5-5
    """
    if np.any(model <= 0.0):
        return np.inf
    positive = data > 0.0
    return 2.0 * float(np.sum(model - data) + np.sum(data[positive] * np.log(data[positive] / model[positive])))


def fit_apec_spectrum(
    spectrum: XraySpectrum,
    redshift: float,
    emin: float,
    emax: float,
    nH: float = DEFAULT_NH,
    abund: float | None = None,
    min_counts_per_bin: float | None = None,
    kT_range: tuple[float, float] = DEFAULT_KT_RANGE,
    abund_range: tuple[float, float] = DEFAULT_ABUND_RANGE,
    credible_interval: float = 0.68,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    seed: int | None = None,
) -> SpectralFitResult:
    """
    Fits TBabs x APEC (redshift and nH fixed) to a spectrum between emin and emax (keV): maximum likelihood with the
    C-statistic, then the posterior sampled with emcee.

    Parameters
    ----------
    spectrum : XraySpectrum
        The spectrum, e.g. from XraySpectrum.from_pha().
    redshift : float
        Redshift of the source (fixed).
    emin, emax : float
        Energy range fitted, in keV; channels whose nominal energy range lies inside it are used.
    nH : float, optional
        Absorbing column in 1e22 cm^-2 (fixed; default 0.018).
    abund : float, optional
        If given, the abundance is fixed at this value; otherwise it is fitted (default).
    min_counts_per_bin : float, optional
        Group consecutive channels to at least this many counts before fitting (default: no grouping).
    kT_range, abund_range : tuple of float, optional
        Ranges of the flat priors on kT (keV, flat in log10 kT) and abundance (solar).
    credible_interval : float, optional
        Probability within the equal-tailed interval used for the errors (default 0.68).
    n_walkers, n_steps, n_burn : int, optional
        emcee walkers, steps per walker, and initial steps discarded.
    seed : int, optional
        Seed for the sampler's random numbers.

    Returns
    -------
    SpectralFitResult

    AI-generated: claude-opus-5-5
    """
    import emcee

    table = apec_table(spectrum.rmf_path, spectrum.arf_path, float(redshift), float(nH), tuple(kT_range))
    in_range = (spectrum.channel_emin >= emin) & (spectrum.channel_emax <= emax)
    if not np.any(in_range):
        raise ValueError(f"No channels of the spectrum lie within {emin}-{emax} keV.")
    groups = _channel_groups(spectrum.counts[in_range], min_counts_per_bin)
    n_groups = int(groups.max()) + 1
    data = np.bincount(groups, weights=spectrum.counts[in_range], minlength=n_groups)
    if data.sum() <= 0.0:
        raise ValueError("The spectrum has no counts in the energy range fitted.")

    fit_abund = abund is None
    n_free = 3 if fit_abund else 2
    if n_groups <= n_free:
        raise ValueError(
            f"Only {n_groups} (grouped) channels in {emin}-{emax} keV for {n_free} free parameters: too few to fit."
        )
    log_kT_bounds = np.log10(kT_range)

    def unit_model(kT: float, abundance: float) -> np.ndarray:
        rate = table.rate(kT, abundance, 1.0)[in_range] * spectrum.exposure
        return np.bincount(groups, weights=rate, minlength=n_groups)

    def profiled(parameters: np.ndarray) -> tuple[float, float]:
        """C-statistic and best normalisation for log10 kT (and abundance), the norm solved for analytically."""
        log_kT = parameters[0]
        abundance = parameters[1] if fit_abund else abund
        if not (log_kT_bounds[0] <= log_kT <= log_kT_bounds[1]) or not (abund_range[0] <= abundance <= abund_range[1]):
            return np.inf, np.nan
        model = unit_model(10.0**log_kT, abundance)
        norm = data.sum() / model.sum()  # where the C-statistic is stationary in the norm
        return _cstat(data, norm * model), norm

    # start the optimiser from the best point of a coarse grid, which guards against local minima
    kT_grid = np.linspace(log_kT_bounds[0], log_kT_bounds[1], 40)
    abund_grid = np.array([0.1, 0.3, 1.0]) if fit_abund else np.array([abund])
    starts = [np.array([k, a]) if fit_abund else np.array([k]) for k in kT_grid for a in abund_grid]
    start = min(starts, key=lambda point: profiled(point)[0])
    optimum = minimize(lambda point: profiled(point)[0], start, method="Nelder-Mead", options={"xatol": 1e-6})
    best_cstat, best_norm = profiled(optimum.x)
    best_abund = float(optimum.x[1]) if fit_abund else float(abund)
    best = np.array([optimum.x[0], best_abund, np.log10(best_norm)])  # log10 kT, abund, log10 norm

    def log_probability(parameters: np.ndarray) -> float:
        log_kT, abundance, log_norm = parameters
        if not fit_abund:
            abundance = abund
        if not (log_kT_bounds[0] <= log_kT <= log_kT_bounds[1]) or not (abund_range[0] <= abundance <= abund_range[1]):
            return -np.inf
        model = 10.0**log_norm * unit_model(10.0**log_kT, abundance)
        return -0.5 * _cstat(data, model)

    rng = np.random.default_rng(seed)
    n_dimensions = 3 if fit_abund else 2
    centre = best if fit_abund else best[[0, 2]]
    spread = np.array([1e-3, 1e-3 * max(best_abund, 0.01), 1e-3]) if fit_abund else np.array([1e-3, 1e-3])

    if fit_abund:
        sampled_log_probability = log_probability
    else:

        def sampled_log_probability(parameters: np.ndarray) -> float:
            return log_probability(np.array([parameters[0], abund, parameters[1]]))

    initial = centre + spread * rng.standard_normal((n_walkers, n_dimensions))
    sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, sampled_log_probability)
    if seed is not None:  # emcee keeps its own numpy RandomState
        sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(initial, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_burn, flat=True)
    if not fit_abund:
        chain = np.column_stack([chain[:, 0], np.full(len(chain), abund), chain[:, 1]])
    samples = np.column_stack([10.0 ** chain[:, 0], chain[:, 1], 10.0 ** chain[:, 2]])  # kT, abund, norm

    best_fit = {"kT": 10.0 ** best[0], "abund": best_abund, "norm": 10.0 ** best[2]}
    tail = 100.0 * (1.0 - credible_interval) / 2.0
    negative_error, positive_error = {}, {}
    for column, name in enumerate(PARAMETER_NAMES):
        lower, upper = np.percentile(samples[:, column], [tail, 100.0 - tail])
        negative_error[name] = min(lower - best_fit[name], 0.0)
        positive_error[name] = max(upper - best_fit[name], 0.0)

    fixed = {"redshift": float(redshift), "nH": float(nH)}
    if not fit_abund:
        fixed["abund"] = float(abund)
    return SpectralFitResult(
        best_fit=best_fit,
        negative_error=negative_error,
        positive_error=positive_error,
        samples=samples,
        cstat=best_cstat,
        n_bins=n_groups,
        fixed=fixed,
    )


def plot_spectral_fit(
    spectrum: XraySpectrum,
    result: SpectralFitResult,
    emin: float,
    emax: float,
    output_path: Path | str,
    kT_range: tuple[float, float] = DEFAULT_KT_RANGE,
) -> None:
    """
    Plots the spectrum (counts per second per keV) with the best-fit model and the residuals, and saves it.

    AI-generated: claude-opus-5-5
    """
    import matplotlib.pyplot as plt

    table = apec_table(
        spectrum.rmf_path, spectrum.arf_path, result.fixed["redshift"], result.fixed["nH"], tuple(kT_range)
    )
    model = table.rate(result.best_fit["kT"], result.best_fit["abund"], result.best_fit["norm"]) * spectrum.exposure
    in_range = (spectrum.channel_emin >= emin) & (spectrum.channel_emax <= emax)
    groups = _channel_groups(spectrum.counts[in_range], None)
    centres = 0.5 * (spectrum.channel_emin + spectrum.channel_emax)[in_range]
    widths = (spectrum.channel_emax - spectrum.channel_emin)[in_range]
    scale = 1.0 / (spectrum.exposure * widths)
    data, model = spectrum.counts[in_range][groups], model[in_range][groups]

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]})
    top.errorbar(centres, data * scale, yerr=np.sqrt(np.maximum(data, 1.0)) * scale, fmt=".", ms=2, lw=0.5)
    top.plot(centres, model * scale, color="C3", lw=1)
    top.set_yscale("log")
    top.set_ylabel("counts / s / keV")
    best = result.best_fit
    top.set_title(f"kT = {best['kT']:.3g} keV, Z = {best['abund']:.3g}, norm = {best['norm']:.3g}", fontsize=9)
    bottom.plot(centres, (data - model) / np.sqrt(np.maximum(model, 1e-10)), ".", ms=2)
    bottom.axhline(0.0, color="k", lw=0.5)
    bottom.set_xlabel("energy (keV)")
    bottom.set_ylabel("(data - model) / sigma")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
