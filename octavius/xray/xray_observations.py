"""

Hands the gas collected by xray_gas.py to moxha (octavius/xray/moxha), which makes the mock X-ray observations with
yt, pyXSIM and SOXS. These packages are optional dependencies (pip install octavius[xray]), so moxha is only imported
here, when an observation is requested.

"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .xray_gas import XrayGas

# other packages
import numpy as np

# internal imports
from ..log import get_logger

logger = get_logger()


# moxha field names (as yt's GIZMO frontend names them) of the optional gas quantities, with their units
MOXHA_EXTRA_FIELDS = {
    "sfr": ("StarFormationRate", "Msun/yr"),
    "wind_delay_time": ("DelayTime", "Gyr"),
    "internal_energy": ("specific_thermal_energy", "cm**2/s**2"),
}

# quantities given to moxha as zeros when the snapshot lacks them, so cuts on them always work: without wind data, no
# particle is treated as a wind
ZERO_IF_MISSING = ["wind_delay_time"]

# moxha (GIZMO) names of the other particle types
MOXHA_PTYPES = {"dm": "PartType1", "star": "PartType4", "bh": "PartType5"}


def _import_moxha() -> tuple[Any, Any, Any]:
    """
    Imports the moxha functions used here; moxha raises an informative error if the X-ray dependencies are missing.

    AI-generated: claude-opus-5-5
    """
    from .moxha.observations import Observation
    from .moxha.particle_dataset import dataset_from_gas_particles, halo_dictionary

    return Observation, dataset_from_gas_particles, halo_dictionary


def dataset_from_xray_gas(gas: XrayGas) -> Any:
    """
    Builds the in-memory yt dataset of the gas that moxha observes.

    AI-generated: claude-opus-5-5
    """
    _, dataset_from_gas_particles, _ = _import_moxha()
    columns = gas.columns

    columns = dict(columns)  # leave the XrayGas as read
    for name in ZERO_IF_MISSING:
        if name not in columns:
            logger.info(f"The snapshot has no {name}, so moxha is given zeros for {MOXHA_EXTRA_FIELDS[name][0]}.")
            columns[name] = np.zeros(gas.n_particles)

    extra_fields = {
        field_name: (columns[name], unit) for name, (field_name, unit) in MOXHA_EXTRA_FIELDS.items() if name in columns
    }

    # a cube around the halo, so the domain covers the whole sphere even where it holds no particles
    all_positions = [columns["pos"]] + [matter["pos"] for matter in gas.other_matter.values()]
    half_width = max(gas.region_radius, max(np.max(np.abs(pos - gas.centre), initial=0.0) for pos in all_positions))
    half_width *= 1.01
    half_width += np.max(columns["smoothing_length"])
    bounding_box = np.column_stack([gas.centre - half_width, gas.centre + half_width])

    return dataset_from_gas_particles(
        positions=columns["pos"],
        velocities=columns["vel"],
        masses=columns["mass"],
        densities=columns["rho"],
        temperatures=columns["temperature"],
        smoothing_lengths=columns["smoothing_length"],
        metallicities=columns["metallicity"],
        redshift=gas.redshift,
        hubble_constant=gas.h,
        omega_matter=gas.omega_matter,
        omega_lambda=gas.omega_lambda,
        element_fractions={symbol: columns[f"fraction_{symbol}"] for symbol in gas.elements},
        extra_fields=extra_fields,
        other_particles={
            MOXHA_PTYPES[ptype]: (matter["pos"], matter["mass"])
            for ptype, matter in gas.other_matter.items()
            if len(matter["mass"]) > 0
        },
        bounding_box=bounding_box,
    )


def build_xray_observation(gas: XrayGas, make_grad_fields: bool, observation_kwargs: dict[str, Any]) -> Any:
    """
    Creates a moxha Observation of the gas's halo with the dataset loaded and the halo set as its active halo.

    AI-generated: claude-opus-5-5
    """
    Observation, _, halo_dictionary = _import_moxha()

    observation = Observation(**observation_kwargs)
    observation.load_ds(ds=dataset_from_xray_gas(gas=gas), make_grad_fields=make_grad_fields)
    observation.set_active_halos(halo_dictionary(index=gas.halo_index, centre=gas.centre, r500=gas.r500, m500=gas.m500))

    return observation
