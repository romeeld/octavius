"""

Tests the collection of gas around haloes for mock X-ray observations. The observations themselves need the optional
X-ray dependencies (moxha, yt, pyXSIM and SOXS), so the test of the moxha dataset is skipped without them.

"""

from pathlib import Path
import numpy as np
import h5py
import pytest

from octavius import load_catalogue, build_analyser
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
    pytest.importorskip("moxha.particle_dataset", exc_type=ImportError)
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

    cat.close()
