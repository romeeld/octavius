"""

Builds the in-memory yt datasets of gas particles (e.g. supplied by Octavius) that moxha observes.

Part of moxha (https://github.com/Fred-Jennings/moxha), by Fred Jennings, included in Octavius with permission.

"""

"""
Builds an in-memory yt dataset of SPH gas particles from plain arrays, so that moxha can observe gas
supplied by another code (e.g. Octavius, which reads SWIFT, GIZMO and TNG snapshots) rather than a
snapshot file that yt loads itself.

Fields are named as yt's GIZMO frontend names them (under PartType0, aliased to "gas"), so the
fields, cuts and pyXSIM source models used for Simba snapshots work unchanged.
"""

import numpy as np
import unyt
import yt
from yt.utilities.cosmology import Cosmology

GAS_PTYPE = "PartType0"


def dataset_from_gas_particles(
    positions,
    velocities,
    masses,
    densities,
    temperatures,
    smoothing_lengths,
    metallicities,
    redshift,
    hubble_constant,
    omega_matter,
    omega_lambda,
    element_fractions=None,
    extra_fields=None,
    other_particles=None,
    bounding_box=None,
):
    """
    Creates a yt dataset of SPH gas particles from arrays in physical units.

    Parameters
    ----------
    positions: ndarray (N, 3)
        Physical positions in kpc. They should not straddle a periodic boundary: unwrap them around
        the region of interest first.
    velocities: ndarray (N, 3)
        Peculiar velocities in km/s.
    masses: ndarray (N,)
        Masses in Msun.
    densities: ndarray (N,)
        Physical mass densities in g/cm**3.
    temperatures: ndarray (N,)
        Temperatures in K.
    smoothing_lengths: ndarray (N,)
        Physical smoothing lengths in kpc, in the snapshot's own convention (as yt's frontends use).
    metallicities: ndarray (N,)
        Total metal mass fractions.
    redshift, hubble_constant, omega_matter, omega_lambda: float
        The snapshot's redshift and cosmology (hubble_constant is little h).
    element_fractions: dict, optional
        Element mass fractions keyed by symbol (e.g. {"O": array}); each becomes ("gas", "O_metallicity").
    extra_fields: dict, optional
        Further particle fields as {name: (array, unit string)}, e.g. {"StarFormationRate": (sfr, "Msun/yr")};
        each becomes ("PartType0", name) and ("gas", name).
    other_particles: dict, optional
        Positions (physical kpc) and masses (Msun) of other particle types as {ptype: (positions, masses)},
        with GIZMO ptype names (PartType1 for dark matter, PartType4 for stars, PartType5 for black holes).
        They are needed for the true total mass profiles that moxha makes from ("all", "Masses").
    bounding_box: ndarray (3, 2), optional
        Domain edges in physical kpc; by default the particle extent padded by the largest smoothing length.

    Returns
    -------
    ds: yt StreamParticlesDataset
        A non-periodic dataset whose length unit is the physical kpc.

    AI-generated: claude-opus-5-5
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    smoothing_lengths = np.asarray(smoothing_lengths, dtype=np.float64)

    data = {}
    for axis_number, axis in enumerate("xyz"):
        data[(GAS_PTYPE, f"particle_position_{axis}")] = (
            positions[:, axis_number],
            "kpc",
        )
        data[(GAS_PTYPE, f"particle_velocity_{axis}")] = (
            velocities[:, axis_number],
            "km/s",
        )

    data[(GAS_PTYPE, "particle_mass")] = (np.asarray(masses, dtype=np.float64), "Msun")
    data[(GAS_PTYPE, "density")] = (np.asarray(densities, dtype=np.float64), "g/cm**3")
    data[(GAS_PTYPE, "temperature")] = (np.asarray(temperatures, dtype=np.float64), "K")
    data[(GAS_PTYPE, "smoothing_length")] = (smoothing_lengths, "kpc")
    data[(GAS_PTYPE, "metallicity")] = (np.asarray(metallicities, dtype=np.float64), "")

    element_fractions = element_fractions or {}
    for symbol, fractions in element_fractions.items():
        data[(GAS_PTYPE, f"{symbol}_metallicity")] = (
            np.asarray(fractions, dtype=np.float64),
            "",
        )

    extra_fields = extra_fields or {}
    for name, (values, unit) in extra_fields.items():
        data[(GAS_PTYPE, name)] = (np.asarray(values, dtype=np.float64), unit)

    # every particle type gets a GIZMO-style "Masses" field, so that ("all", "Masses") is the total mass
    data[(GAS_PTYPE, "Masses")] = data[(GAS_PTYPE, "particle_mass")]
    other_particles = other_particles or {}
    for ptype, (other_positions, other_masses) in other_particles.items():
        other_positions = np.asarray(other_positions, dtype=np.float64)
        for axis_number, axis in enumerate("xyz"):
            data[(ptype, f"particle_position_{axis}")] = (
                other_positions[:, axis_number],
                "kpc",
            )
        data[(ptype, "particle_mass")] = (
            np.asarray(other_masses, dtype=np.float64),
            "Msun",
        )
        data[(ptype, "Masses")] = data[(ptype, "particle_mass")]

    if bounding_box is None:
        padding = smoothing_lengths.max() if len(smoothing_lengths) > 0 else 1.0
        bounding_box = np.column_stack([positions.min(axis=0) - padding, positions.max(axis=0) + padding])

    ds = yt.load_particles(
        data,
        length_unit="kpc",
        mass_unit="Msun",
        velocity_unit="km/s",
        bbox=np.asarray(bounding_box, dtype=np.float64),
        periodicity=(False, False, False),
        default_species_fields="ionized",
    )

    # cosmology, which moxha reads for redshift, h and X-ray emissivity fields
    ds.cosmological_simulation = 1
    ds.current_redshift = float(redshift)
    ds.hubble_constant = float(hubble_constant)
    ds.omega_matter = float(omega_matter)
    ds.omega_lambda = float(omega_lambda)
    # yt builds ds.cosmology from these when loading a snapshot, but not when they are set afterwards, and pyXSIM
    # takes the distance to the source from ds.cosmology
    ds.cosmology = Cosmology(
        hubble_constant=ds.hubble_constant,
        omega_matter=ds.omega_matter,
        omega_lambda=ds.omega_lambda,
    )

    # in-memory particle datasets define a grid-style ("gas", "cell_volume"), which particle filters of gas (e.g.
    # moxha's filtered_gas) inherit; pyXSIM would then take particle volumes from it, which fails. yt's GIZMO frontend
    # has no such field, so remove it and pyXSIM uses mass / density, as it does for GIZMO snapshots
    derived_fields = ds.derived_field_list  # accessing it builds the field lists
    ds.field_info.pop(("gas", "cell_volume"), None)
    if ("gas", "cell_volume") in derived_fields:
        derived_fields.remove(("gas", "cell_volume"))

    # yt only aliases the standard SPH fields to "gas"; alias the rest so moxha can address them as it does for GIZMO
    standard = {"density", "temperature", "smoothing_length", "metallicity"}
    aliased = [f"{symbol}_metallicity" for symbol in element_fractions] + list(extra_fields)
    for name in aliased:
        if name in standard:
            continue
        units = str(ds.field_info[(GAS_PTYPE, name)].units)
        ds.add_field(
            ("gas", name),
            function=_alias(name),
            sampling_type="particle",
            units=units if units else "dimensionless",
            force_override=True,
        )

    return ds


def _alias(name):
    """
    Returns a yt field function that reads (PartType0, name).
    """

    def _field(field, data):
        return data[GAS_PTYPE, name]

    return _field


def halo_dictionary(index, centre, r500, m500):
    """
    Builds the halo dictionary for Observation.set_active_halos() from physical kpc and Msun values.

    AI-generated: claude-opus-5-5
    """
    return {
        "index": int(index),
        "center": unyt.unyt_array(np.asarray(centre, dtype=np.float64), "kpc"),
        "R500": unyt.unyt_quantity(float(r500), "kpc"),
        "M500": unyt.unyt_quantity(float(m500), "Msun"),
    }
