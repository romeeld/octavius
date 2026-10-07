"""

Built-in halo source for snapshots without HaloIDs: runs Octavius's periodic 3D friends-of-friends halo finder on
the snapshot itself, so no external catalogue is required.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SnapshotReader
    from mpi4py.MPI import Comm

# default libraries
from collections.abc import Iterator, Mapping

# other packages
import numpy as np

# internal imports
from .halo_data_structures import HaloSource, HaloAssignments, SubhaloInformation, distribute_ids
from ..galaxy_finding.fof_halo_algorithm import find_fof_haloes
from ..log import get_logger

logger = get_logger()


class FOFHaloSource(HaloSource):
    """
    Identifies field haloes with a 3D friends-of-friends on the DM particles (rank 0 only); baryons are attached to
    the halo of their nearest DM particle. FOF groups carry no subhalo information.
    """

    def __init__(self, reader: SnapshotReader, b: float, min_members: int) -> None:

        super().__init__(reader=reader)
        self.b = b
        self.min_members = min_members

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Reads all particle positions and runs the halo finder, returning HaloAssignments.
        """
        if "dm" not in ptypes:
            raise ValueError("FOF halo finding requires DM particles, but none are available in the snapshot.")

        boxsize = self.reader.simulation_attributes.boxsize
        n_dm = self.reader.particle_counts["dm"]
        logger.info(f"FOF: finding haloes in {n_dm:,} DM particles (b = {self.b}, min. {self.min_members} members).")

        halo_ids = find_fof_haloes(
            dm_pos=self.reader.read_full_dataset(ptype="dm", dataset="pos"),
            boxsize=boxsize,
            b=self.b,
            baryon_pos=_LazyPositions(reader=self.reader, ptypes=[pt for pt in ptypes if pt != "dm"]),
            min_members=self.min_members,
        )

        n_haloes = int(halo_ids["dm"].max()) + 1 if n_dm > 0 else 0
        logger.info(f"FOF: {n_haloes:,} field haloes | no subhalo information")

        for ptype, ids in halo_ids.items():
            n_assigned = np.sum(ids != -1)
            logger.info(f"  {ptype}: {n_assigned:,} / {len(ids):,} particles assigned to haloes.")

        return HaloAssignments(
            field_ids=halo_ids,
            n_field_haloes=n_haloes,
            sub_ids=None,
            original_field_ids=None,
        )

    def read_subhalo_info(self) -> SubhaloInformation | None:
        """
        No subhaloes on FOFHaloSource.
        """
        return None

    def distribute_field_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Wrapper around distribute_ids() for field halo IDs.
        """
        return distribute_ids(
            slabs=slabs,
            particle_counts=self.reader.particle_counts,
            ptypes=sorted(self.reader.available_ptypes),
            comm=comm,
            global_ids=global_ids,
        )

    def distribute_sub_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        No subhaloes on FOFHaloSource.
        """
        return None


class _LazyPositions(Mapping):
    """
    Reads each ptype's positions only when the halo finder reaches it, so only one baryonic ptype is in memory at once.
    """

    def __init__(self, reader: SnapshotReader, ptypes: list[str]) -> None:
        self.reader = reader
        self.ptypes = ptypes

    def __getitem__(self, ptype: str) -> np.ndarray:
        return self.reader.read_full_dataset(ptype=ptype, dataset="pos")

    def __iter__(self) -> Iterator[str]:
        return iter(self.ptypes)

    def __len__(self) -> int:
        return len(self.ptypes)
