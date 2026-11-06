"""
Builders for the halo dictionaries passed to Observation.set_active_halos(), reading
halo centres, R500 and M500 from a halo catalogue.

The catalogue packages (octavius, caesar) are imported inside each function, so moxha
does not depend on either of them.
"""

from pathlib import Path

import numpy as np
import unyt


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
    import octavius as oc

    halo_indices = np.atleast_1d(halo_indices).astype(int)

    with oc.load_catalogue(Path(catalogue_path)) as catalogue:
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


def halos_from_caesar(caesar_path, halo_indices):
    """
    Read the halo dictionaries for Observation.set_active_halos() from a Caesar file.

    AI-generated: claude-opus-5-5
    """
    import caesar

    caesar_object = caesar.load(str(caesar_path))
    halos = []
    for halo_index in np.atleast_1d(halo_indices).astype(int):
        halo = caesar_object.halos[halo_index]
        halos.append(
            {
                "index": int(halo_index),
                "center": halo.minpotpos,
                "R500": halo.virial_quantities["r500c"],
                "M500": halo.virial_quantities["m500c"],
            }
        )
    return halos
