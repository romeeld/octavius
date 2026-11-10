"""

X-ray emission of each group's hot gas: luminosities in configured energy bands (rest frame) and X-ray-like
temperatures, computed per gas particle from an APEC emissivity table and summed over the group.

The table is yt's apec_emissivity_v3.h5 (APEC 3.0.9; shipped in octavius/data), giving the emissivity of gas with
primordial composition and of the metals at solar abundance, per energy bin, over 1e5 - 7e8 K. Integrated over a
band and interpolated in log T exactly as yt does (and so as pyXSIM's mock photons are made), a particle's luminosity
is n_e n_H V [eps_primordial(T) + (Z / Z_sun) eps_metals(T)], for fully ionised gas.

The X-ray emitting gas is hotter than xray_temperature_min, less dense than xray_density_max (in proton masses per
cm^3), not star-forming, and not a wind particle (where the snapshot records winds), as in the mock observations.
Only the part above xray_tsl_temperature_min enters the spectroscopic-like temperature, whose T^(-3/4) weighting
would otherwise be dominated by the coolest gas (Mazzotta et al. 2004 define it for gas above about 0.5 keV).

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConstants

# default libraries
from functools import lru_cache
from importlib.resources import files

# other packages
import h5py
import numpy as np

# internal imports
from ..utils import guarded_divide
from .aggregate_helpers import sum_per_group

APEC_TABLE = "apec_emissivity_v3.h5"


@lru_cache(maxsize=8)
def apec_band_emissivities(bands: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns log10 T (K) and log10 of the primordial and metal emissivities (erg cm^3 / s; shape n_bands x n_T)
    integrated over each band ([emin, emax] in keV), with partial edge bins as yt's XrayEmissivityIntegrator.

    AI-generated: claude-opus-5-5
    """
    with h5py.File(files("octavius").joinpath("data", APEC_TABLE), "r") as f:
        log_T = f["log_T"][...]
        energy_bins = f["E"][...]
        primordial = f["emissivity_primordial"][...]
        metals = f["emissivity_metals"][...]

    bin_widths = np.diff(energy_bins)
    log_primordial, log_metals = [], []
    for emin, emax in bands:
        if emin < energy_bins[0] * (1 - 1e-3) or emax > energy_bins[-1] * (1 + 1e-3):
            raise ValueError(
                f"The X-ray band {emin}-{emax} keV is outside the emissivity table's {energy_bins[0]:.2f}-"
                f"{energy_bins[-1]:.0f} keV."
            )
        first, last = np.digitize([emin, emax], energy_bins)
        first = np.clip(first - 1, 0, energy_bins.size - 1)
        last = np.clip(last, 0, energy_bins.size - 1)
        widths = bin_widths[first:last].copy()
        widths[0] -= emin - energy_bins[first]  # the part of the edge bins within the band
        widths[-1] -= energy_bins[last] - emax
        for emissivity, logs in ((primordial, log_primordial), (metals, log_metals)):
            band_emissivity = (emissivity[:, first:last] * widths).sum(axis=-1)
            logs.append(np.log10(np.maximum(band_emissivity, 1e-300)))  # zero emission stays (effectively) zero

    return log_T, np.array(log_primordial), np.array(log_metals)


def compute_xray_properties(
    masses: np.ndarray,
    densities: np.ndarray,
    temperatures: np.ndarray,
    metallicities: np.ndarray,
    helium_fractions: np.ndarray,
    sfrs: np.ndarray,
    wind_delay_times: np.ndarray | None,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    bands: list[list[float]],
    band_names: list[str],
    temperature_min: float,
    density_max: float,
    solar_metallicity: float,
    tsl_temperature_min: float,
    constants: OctaviusConstants,
) -> dict[str, np.ndarray]:
    """
    Computes each group's hot-gas X-ray properties, returning a dict of:

    - Lx_{band}keV: rest-frame luminosity in each band (erg/s)
    - temperature_xray_weighted: temperature weighted by the luminosity in the first band (K)
    - temperature_spectroscopic_like: sum(m rho T^(1/4)) / sum(m rho T^(-3/4)) over the hot gas above
      tsl_temperature_min (K; Mazzotta et al. 2004)

    Masses are in Msun, densities in g/cm^3 (physical), temperatures in K, and metallicities and helium fractions are
    mass fractions; wind_delay_times may be None for snapshots without winds (then no gas is a wind).

    AI-generated: claude-opus-5-5
    """
    proton_mass = constants.PROTON_MASS_G
    is_hot = (temperatures > temperature_min) & (densities / proton_mass < density_max) & (sfrs <= 0.0)
    if wind_delay_times is not None:
        is_hot &= wind_delay_times <= 0.0

    # fully ionised gas: n_H = X rho / m_p and n_e = n_H + 2 n_He = (X + Y / 2) rho / m_p
    hydrogen_fraction = 1.0 - metallicities - helium_fractions
    n_H = hydrogen_fraction * densities / proton_mass
    n_e = (hydrogen_fraction + 0.5 * helium_fractions) * densities / proton_mass
    volume = masses * constants.M_SUN_G / densities  # cm^3
    emission_measure = np.where(is_hot, n_e * n_H * volume, 0.0)  # cm^-3

    log_T, log_primordial, log_metals = apec_band_emissivities(tuple((float(lo), float(hi)) for lo, hi in bands))
    log_temperatures = np.log10(np.maximum(temperatures, 1.0))
    solar_units = metallicities / solar_metallicity

    results: dict[str, np.ndarray] = {}
    first_band_luminosities = None
    for band, name in enumerate(band_names):
        emissivity = 10.0 ** np.interp(log_temperatures, log_T, log_primordial[band]) + solar_units * (
            10.0 ** np.interp(log_temperatures, log_T, log_metals[band])
        )  # erg cm^3 / s, clamped at the table's edges as yt does
        luminosities = emission_measure * emissivity
        results[f"Lx_{name}keV"] = sum_per_group(
            values=luminosities, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
        )
        if first_band_luminosities is None:
            first_band_luminosities = luminosities

    weighted_temperature = sum_per_group(
        values=first_band_luminosities * temperatures, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    results["temperature_xray_weighted"] = guarded_divide(
        numerator=weighted_temperature, denominator=results[f"Lx_{band_names[0]}keV"]
    )

    hot_weight = np.where(is_hot & (temperatures > tsl_temperature_min), masses * densities, 0.0)
    numerator = sum_per_group(
        values=hot_weight * temperatures**0.25, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    denominator = sum_per_group(
        values=hot_weight * temperatures**-0.75, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    results["temperature_spectroscopic_like"] = guarded_divide(numerator=numerator, denominator=denominator)

    return results
