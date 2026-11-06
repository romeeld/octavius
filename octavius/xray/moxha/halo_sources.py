"""

Builds the halo dictionaries passed to Observation.set_active_halos() from an Octavius catalogue.

Part of moxha (https://github.com/Fred-Jennings/moxha), by Fred Jennings, included in Octavius with permission.

"""

from pathlib import Path

import numpy as np
import unyt

from ...utils.loader import load_catalogue


def halos_from_octavius(catalogue_path, halo_indices, centre="minpot_pos"):
    """
    Read the halo dictionaries for Observation.set_active_halos() from an Octavius catalogue.

    The catalogue must contain r500c and m500c, i.e. 500 must be in the virial_factors
    of the Octavius config (it is by default).

    Parameters
    ----------
    catalogue_path: str or pathlib.Path
        Path to the Octavius catalogue.
    halo_indices: int or list of int
        Indices into the catalogue's halo list.
    centre: str, optional
        Name of the catalogue position used as the halo centre (default: minpot_pos).

    Returns
    -------
    halos: list of dict
        One {"index", "center", "R500", "M500"} dictionary per halo.

    AI-generated: claude-opus-5-5
    """
    halo_indices = np.atleast_1d(halo_indices).astype(int)

    with load_catalogue(Path(catalogue_path)) as catalogue:
        haloes = catalogue.haloes
        if haloes is None:
            raise ValueError(f"{catalogue_path} contains no haloes.")
        # keys() is needed: GroupCollection itself does not support "in"
        available = haloes.keys()
        missing = [name for name in (centre, "r500c", "m500c") if name not in available]
        if missing:
            raise KeyError(
                f"{catalogue_path} is missing {missing} (r500c and m500c need 500 in the Octavius virial_factors)."
            )

        # converted to physical kpc: comoving units ("kpccm") only exist within a yt dataset's unit registry,
        # and yt converts physical lengths back to the snapshot's comoving frame itself
        centres = haloes.get_dataset(centre, to_units="kpc", to_physical=True)[halo_indices]
        r500 = haloes.get_dataset("r500c", to_units="kpc", to_physical=True)[halo_indices]
        m500 = haloes.get_dataset("m500c", to_units="Msun")[halo_indices]

    return [
        {
            "index": int(halo_index),
            "center": unyt.unyt_array(centres[i], "kpc"),
            "R500": unyt.unyt_quantity(r500[i], "kpc"),
            "M500": unyt.unyt_quantity(m500[i], "Msun"),
        }
        for i, halo_index in enumerate(halo_indices)
    ]
