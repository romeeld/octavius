"""

Collects the gas around a halo, in physical units, for making mock X-ray observations of it. The gas is read through
the snapshot readers, so this works for every supported simulation type; the mock observations themselves are made
by moxha (with yt, pyXSIM and SOXS), which is only imported when they are requested.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SnapshotReader

# default libraries
from dataclasses import dataclass

# other packages
import numpy as np
from numba import njit, prange

# internal imports
from ..data_management import CODE_UNITS
from ..data_management.conventions import element_fraction_name
from ..data_management.parallel_reading import split_slab
from ..utils.helpers import unwrap_positions
from ..log import get_logger

logger = get_logger()

# gas quantities needed for X-ray emission, and quantities read when the snapshot has them (used to exclude gas)
XRAY_GAS_COLUMNS = ["pos", "vel", "mass", "rho", "temperature", "smoothing_length", "metallicity"]
OPTIONAL_XRAY_GAS_COLUMNS = ["sfr", "internal_energy", "wind_delay_time"]

PARTICLES_PER_POSITION_READ = 4_000_000  # ~100 MB of float64 positions per read when searching the snapshot


@dataclass(frozen=True, slots=True)
class XrayGas:
    """
        Gas particles around a halo in physical units: positions and smoothing lengths in kpc (unwrapped around the halo
        centre), velocities in km/s, masses in Msun, densities in g/cm**3, temperatures in K, internal energies in
        cm**2/s**2, star formation rates in Msun/yr, wind delay times in Gyr, and metallicities and element fractions
        (fraction_<symbol>) as mass fractions. The positions and masses of the other particle types in the same region are
    kept in other_matter, for total mass profiles.

        AI-generated: claude-opus-5-5
    """

    columns: dict[str, np.ndarray]
    other_matter: dict[str, dict[str, np.ndarray]]  # positions (pos) and masses (mass) of the other ptypes, by ptype
    halo_index: int
    centre: np.ndarray  # physical kpc
    r500: float  # physical kpc
    m500: float  # Msun
    region_radius: float  # physical kpc (0 for region="halo")
    redshift: float
    h: float
    omega_matter: float
    omega_lambda: float

    @property
    def n_particles(self) -> int:
        return len(self.columns["mass"])

    @property
    def elements(self) -> list[str]:
        """
        Symbols of the elements whose mass fractions were read.
        """
        return [name.removeprefix("fraction_") for name in self.columns if name.startswith("fraction_")]


@njit(cache=True, parallel=True)
def _flag_within_radius(positions: np.ndarray, centre: np.ndarray, radius: float, boxsize: float) -> np.ndarray:
    """
    Flags the positions within radius of centre, using the nearest periodic image. Returns a boolean array.

    AI-generated: claude-opus-5-5
    """
    n_particles = len(positions)
    half_box = 0.5 * boxsize
    radius_sq = radius * radius
    within = np.zeros(n_particles, dtype=np.bool_)

    for i in prange(n_particles):
        distance_sq = 0.0
        for axis in range(3):
            delta = positions[i, axis] - centre[axis]
            if delta > half_box:
                delta -= boxsize
            elif delta < -half_box:
                delta += boxsize
            distance_sq += delta * delta

        within[i] = distance_sq <= radius_sq

    return within


def find_particles_within_radius(reader: SnapshotReader, ptype: str, centre: np.ndarray, radius: float) -> np.ndarray:
    """
    Searches the whole snapshot, a slab at a time, for the particles of one type within radius of centre (both in
    comoving kpc). Returns the sorted snapshot indices of the particles found.

    AI-generated: claude-opus-5-5
    """
    n_particles = reader.particle_counts[ptype]
    boxsize = reader.simulation_attributes.boxsize
    n_slabs = max(1, int(np.ceil(n_particles / PARTICLES_PER_POSITION_READ)))

    found = []
    for slab in split_slab(slice(0, n_particles), n_slabs):
        positions = reader.read_full_dataset(ptype=ptype, dataset="pos", slab=slab)
        within = _flag_within_radius(positions=positions, centre=centre, radius=radius, boxsize=boxsize)
        found.append(slab.start + np.flatnonzero(within))

    return np.concatenate(found).astype(np.int64)


def readable_xray_columns(reader: SnapshotReader) -> list[str]:
    """
    The gas quantities to read: the required ones, plus the optional ones and element fractions in the snapshot.

    AI-generated: claude-opus-5-5
    """
    optional = OPTIONAL_XRAY_GAS_COLUMNS + [element_fraction_name(symbol) for symbol in reader.available_elements]

    # check the reader's maps first, since has_dataset warns about quantities that the reader does not define at all
    defined = [name for name in optional if name in reader.dataset_map or name in reader.derived_columns]

    return XRAY_GAS_COLUMNS + [name for name in defined if reader.has_dataset(ptype="gas", dataset=name)]


def read_physical_columns(
    reader: SnapshotReader, ptype: str, columns: list[str], snapshot_indices: np.ndarray, centre: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Reads the columns of one ptype for the given (sorted) snapshot indices and converts them to physical units, with
    positions unwrapped around centre (comoving kpc). Returns the columns keyed by Octavius name.

    AI-generated: claude-opus-5-5
    """
    scale_factor = reader.simulation_attributes.scale_factor

    data = reader.read_requested_columns(ptype=ptype, datasets=columns, sorted_snapshot_indices=snapshot_indices)
    reader.subset_indices = None  # do not leave the reader restricted to these particles

    unwrap_positions(positions=data["pos"], centre=centre, boxsize=reader.simulation_attributes.boxsize)

    for name in columns:
        data[name] *= scale_factor ** CODE_UNITS[name].a_exponent  # comoving to physical

    return data
