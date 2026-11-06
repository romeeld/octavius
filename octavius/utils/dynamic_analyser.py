"""

Functionality for dynamic, on-the-fly analysis of raw snapshots by calling the pipeline stages
on a subset of groups in a provided catalogue. Allows users to recompute properties with different
config params, alleviating some of the expensive cost of running the full pipeline.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import OctaviusCatalogue, GroupCollection
    from ..data_management import SnapshotReader
    from ..xray import XrayGas
    from moxha.observations import Observation
    import h5py

# default libraries
from dataclasses import dataclass, replace
from pathlib import Path

# other packages
import numpy as np
import h5py
import numba
import astropy.units as u
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.sparse import csr_array

# internal imports
from ..data_management import (
    SimulationData,
    ParticleStore,
    GroupStore,
    OctaviusConfig,
    OctaviusConstants,
    build_reader,
    CODE_UNITS,
)
from ..data_management.pipeline_management import load_internals, Internals, PipelineStage

from .helpers import guarded_divide
from ..log import get_logger, configure_logger

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"

logger = get_logger()


def build_analyser(
    catalogue: OctaviusCatalogue,
    config: OctaviusConfig | None = None,
    snapshot_path: Path | None = None,
) -> OctaviusAnalyser:
    """
    Constructs an ``Analyser`` object for dynamic, on-the-fly analysis of groups in an Octavius catalogue
    via the existing runtime stages.

    Parameters
    ----------
    catalogue: OctaviusCatalogue
        OctaviusCatalogue object corresponding to the snapshot.
    config: OctaviusConfig or None
        OctaviusConfig object. If not given, the config stored in the catalogue is used.
    snapshot_path: pathlib.Path or None
        Path object pointing to the original, raw snapshot (overrides config field).

    Returns
    -------
    analyser: OctaviusAnalyser
        An OctaviusAnalyser object.

    Notes
    -----
    The config can be modified from the one used to create the snapshot. Pipeline-specific parameters will be disregarded.
    """
    config_from_catalogue = config is None
    if config_from_catalogue:
        overrides = {"snapshot_path": snapshot_path} if snapshot_path is not None else {}
        config = OctaviusConfig.from_catalogue(catalogue_path=catalogue._cat, **overrides)

        # the catalogue may have been made on a machine with more cores than this one
        if config.cores_per_rank > numba.config.NUMBA_NUM_THREADS:
            config = replace(config, cores_per_rank=numba.config.NUMBA_NUM_THREADS)

    snapshot_path = Path(snapshot_path or config.snapshot_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot not found at {snapshot_path}; please pass its location with build_analyser(snapshot_path=...)."
        )

    # initialise data structures
    constants = OctaviusConstants(mu=config.MU, frad=config.FRAD)
    reader = build_reader(snapshot_path=snapshot_path, constants=constants, config=config)
    internals = load_internals(internals_filepath=INTERNALS_FILEPATH, config=config)

    analyser = OctaviusAnalyser(
        reader=reader,
        catalogue=catalogue,
        config=config,
        constants=constants,
        internals=internals,
    )

    if config_from_catalogue:  # logged here as the analyser sets up the logger
        logger.info(
            f"No config given, so using the config stored in {catalogue._cat.name} (snapshot: {snapshot_path})."
        )

    return analyser


@dataclass(frozen=True, slots=True)
class StageResult:
    """
    Contains the outputs of a pipeline stage when run in dynamic, on-the-fly mode.

    Attributes
    ----------
    group_type: str
        The type of group the result belongs to.
    group_indices: np.ndarray
        Indices into the catalogue for the groups.
    columns: dict[str, np.ndarray]
        The output columns.
    """

    group_type: str
    group_indices: np.ndarray
    columns: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Controls behaviour of StageResult["column_name"].
        """
        return self.columns[key]

    def __len__(self) -> int:
        """
        Returns the number of groups.
        """
        return len(self.group_indices)

    def keys(self) -> list[str]:
        """
        Returns a list of available output columns.
        """
        return list(self.columns.keys())


class OctaviusAnalyser:
    """
    Object-oriented interface for dynamic, on-the-fly analysis of groups in a snapshot from the corresponding
    Octavius catalogue.
    """

    def __init__(
        self,
        reader: SnapshotReader,
        catalogue: OctaviusCatalogue,
        config: OctaviusConfig,
        constants: OctaviusConstants,
        internals: Internals,
    ) -> None:

        self._reader = reader
        self._catalogue = catalogue
        self._config = config
        self._constants = constants
        self._internals = internals

        numba.set_num_threads(config.cores_per_rank)
        configure_logger(snapshot_path=config.snapshot_path.stem, rank=0, output_level=config.terminal_output_level)

        self._collections: dict[str, GroupCollection] = {}
        if catalogue.galaxies is not None:
            self._collections["galaxies"] = catalogue.galaxies
        if catalogue.haloes is not None:
            self._collections["haloes"] = catalogue.haloes

        if not self._collections:  # high-z snapshots
            raise ValueError("Catalogue contains no group data")

    def __repr__(self) -> str:
        """
        Controls behaviour of print(OctaviusAnalyser)
        """
        return f"<OctaviusAnalyser> | {self._reader.snapshot_path.stem}"

    def update_config(self, **kwargs) -> None:
        """
        Updates the config object with the specified parameters.

        Parameters
        ----------
        **kwargs
            The keyword arguments are replaced in the config.
        """
        self._config = replace(self._config, **kwargs)

    def compute_photometry(
        self,
        group_indices: list[int] | np.ndarray,
        keep_spectra: bool = False,
        orientation: str | np.ndarray | None = None,
    ) -> StageResult:
        """
        Runs the photometry pipeline for the galaxies specified by group_indices, with optional rotation.

        Parameters
        ----------
        group_indices: list[int]
            The indices into the Octavius catalogue of the galaxies to run on.
        keep_spectra: bool
            Whether or not to keep the galaxy spectra (uses more memory). Default: ``False``
        orientation: str | np.ndarray | None
            Orientation to rotate the galaxies into. ``face-on`` and ``side-on`` are available, or a bespoke (3, 3) rotation
            matrix can be passed. If left blank, the default positions will be used.

        Returns
        -------
        result: StageResult
            A StageResult dataclass from which output columns, aligned to group_indices, can be accessed.
        """
        from ..photometry import run_photometry  # avoid circular import

        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

        config = replace(  # HACK: insert _keep_spectra (see photometry)
            self._config, _keep_spectra=keep_spectra
        )

        group_type = "galaxies"  # only runs on galaxies
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")

        group_indices = np.sort(np.array(group_indices))
        photometry = self._internals.stages["photometry"]
        self._verify_dependencies(stage=photometry, group_type=group_type)

        # get columns needed by photometry
        ptype_columns = self._resolve_stage_columns(stage=photometry, group_type=group_type)
        gal_reqs = {"star": ptype_columns["star"], "gas": ptype_columns["gas"]}
        gal_data = self._extract_groups(group_type=group_type, group_indices=group_indices, ptype_columns=gal_reqs)

        particles = _build_particle_stores(subset_data=gal_data, internals=self._internals)
        galaxies = _build_group_store(
            group_type="galaxies", group_indices=group_indices, subset_data=gal_data, internals=self._internals
        )

        # dependencies
        _preload_stage_dependencies(
            collection=self._collections["galaxies"],
            group_store=galaxies,
            group_indices=group_indices,
            stage=photometry,
            internals=self._internals,
        )

        groups = {"galaxies": galaxies}

        if orientation is not None:
            align_orientations(galaxies=galaxies, orientation=orientation)  # modifies in place
            if isinstance(orientation, str):
                los_axis = "Z" if orientation == "face-on" else "X"
                config = replace(config, viewing_axis=los_axis)
                logger.debug(
                    f"Overriding viewing_axis parameter to '{los_axis}' for requested orientation {orientation}."
                )

        sim_data = SimulationData(
            simulation=self._reader.simulation_attributes,
            constants=self._constants,
            particles=particles,
            groups=groups,
        )

        pre_columns = set(galaxies.columns.keys())
        run_photometry(simulation_data=sim_data, config=config)
        results = _extract_results(
            group_store=galaxies, group_type=group_type, group_indices=group_indices, pre_columns=pre_columns
        )

        with h5py.File(config.photometry_table_path, "r") as f:
            wavelengths = f["ssp"]["wavelengths"][:]

        results.columns["wavelengths"] = wavelengths

        return results

    def compute_ptype_specific_properties(self, group_indices: list[int] | np.ndarray, group_type: str) -> StageResult:
        """
        Runs the particle-type specific properties routine for the groups (of one group type) specified by group_indices.

        Parameters
        ----------
        group_indices: list[int]
            The indices into the Octavius catalogue of the groups to run on. You should only pass indices corresponding
            to one type of group (e.g. only galaxies or only haloes).
        group_type: str
            The type of group the indices are specified for: ``galaxies`` or ``haloes``.

        Returns
        -------
        result: StageResult
            A StageResult dataclass from which output columns, aligned to group_indices, can be accessed.
        """
        from ..aggregate_properties import run_ptype_specific_properties  # avoid circular import

        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

        if group_type not in self._collections:
            raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")

        stage = self._internals.stages["properties_ptype_specific"]
        self._verify_dependencies(stage=stage, group_type=group_type)

        ptype_columns = self._resolve_stage_columns(stage=stage, group_type=group_type)
        subset_data = self._extract_groups(
            group_type=group_type, group_indices=group_indices, ptype_columns=ptype_columns
        )
        particles = _build_particle_stores(subset_data=subset_data, internals=self._internals)
        group_store = _build_group_store(
            group_type=group_type, group_indices=group_indices, subset_data=subset_data, internals=self._internals
        )
        _preload_stage_dependencies(
            collection=self._collections[group_type],
            group_store=group_store,
            group_indices=group_indices,
            stage=stage,
            internals=self._internals,
        )

        sim_data = SimulationData(
            simulation=self._reader.simulation_attributes,
            constants=self._constants,
            particles=particles,
            groups={group_type: group_store},
        )

        pre_columns = set(group_store.columns.keys())
        run_ptype_specific_properties(sim_data, self._config)
        results = _extract_results(
            group_store=group_store, group_type=group_type, group_indices=group_indices, pre_columns=pre_columns
        )
        return results

    def compute_local_densities(
        self, group_indices: list[int] | np.ndarray, radii: list[float], group_type: str
    ) -> StageResult:
        """
        Computes the local number and mass density for the groups (of one group type) specified by group_indices.

        Parameters
        ----------
        group_indices: list[int]
            The indices into the Octavius catalogue of the groups to run on. You should only pass indices corresponding
            to one type of group (e.g. only galaxies or only haloes).
        group_type: str
            The type of group the indices are specified for: ``galaxies`` or ``haloes``.

        Returns
        -------
        result: StageResult
            A StageResult dataclass from which output columns, aligned to group_indices, can be accessed.
        """
        if group_type not in self._collections:
            raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

        stage = self._internals.stages["properties_local_environment"]
        self._verify_dependencies(stage=stage, group_type=group_type)
        data = self._collections[group_type]._data["properties"]
        columns = {}

        if group_type == "haloes":
            positions = data["core/minpot_pos"][:] if "core/minpot_pos" in data else data["core/com_pos_total"][:]
            masses = data["core/mass_total"][:]
        else:
            positions = data["core/com_pos_baryon"][:]
            masses = data["core/mass_baryon"][:]

        n_groups = len(masses)

        r_max = np.max(radii)
        tree = KDTree(data=positions, boxsize=self._catalogue.sim_info("boxsize", to_units="kpc"))
        sdm = tree.sparse_distance_matrix(
            other=tree, max_distance=r_max, output_type="coo_array"
        )  # if your scipy is pre-1.18, use "coo_matrix"
        vol_prefactor = 4.0 / 3.0 * np.pi

        for radius in radii:
            in_range = (
                sdm.data <= radius
            )  # NOTE: the matrix does include self-distance 0 as we passed other=tree ^, I verified this manually with test data
            valid_array = np.ones(in_range.sum())
            adj = csr_array(
                arg1=(valid_array, (sdm.row[in_range], sdm.col[in_range])), shape=(n_groups, n_groups)
            )  # unhelpful argument name

            volume = vol_prefactor * radius**3
            local_mass_density = (adj @ masses / volume)[
                group_indices
            ]  # adjacency matrix algebra gets the quantities in a vectorised way
            local_number_density = (adj @ np.ones(shape=n_groups) / volume)[group_indices]

            columns[f"local_mass_density_{radius:.0f}kpc"] = local_mass_density
            columns[f"local_number_density_{radius:.0f}kpc"] = local_number_density

        result = StageResult(
            group_type=group_type,
            group_indices=group_indices,
            columns=columns,
        )

        return result

    def compute_core_properties(self, group_indices: list[int] | np.ndarray, group_type: str) -> StageResult:
        """
        Runs the core properties routine for the groups (of one group type) specified by group_indices.

        Parameters
        ----------
        group_indices: list[int]
            The indices into the Octavius catalogue of the groups to run on. You should only pass indices corresponding
            to one type of group (e.g. only galaxies or only haloes).
        group_type: str
            The type of group the indices are specified for: ``galaxies`` or ``haloes``.

        Returns
        -------
        result: StageResult
            A StageResult dataclass from which output columns, aligned to group_indices, can be accessed.
        """
        from ..aggregate_properties import run_core_properties  # avoid circular import

        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

        if group_type not in self._collections:
            raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")

        stage = self._internals.stages["properties_core"]
        self._verify_dependencies(stage=stage, group_type=group_type)

        ptype_columns = self._resolve_stage_columns(stage=stage, group_type=group_type)
        subset_data = self._extract_groups(
            group_type=group_type, group_indices=group_indices, ptype_columns=ptype_columns
        )
        particles = _build_particle_stores(subset_data=subset_data, internals=self._internals)
        group_store = _build_group_store(
            group_type=group_type, group_indices=group_indices, subset_data=subset_data, internals=self._internals
        )
        _preload_stage_dependencies(
            collection=self._collections[group_type],
            group_store=group_store,
            group_indices=group_indices,
            stage=stage,
            internals=self._internals,
        )

        sim_data = SimulationData(
            simulation=self._reader.simulation_attributes,
            constants=self._constants,
            particles=particles,
            groups={group_type: group_store},
        )

        pre_columns = set(group_store.columns.keys())
        run_core_properties(sim_data, self._config)
        results = _extract_results(
            group_store=group_store, group_type=group_type, group_indices=group_indices, pre_columns=pre_columns
        )
        return results

    def get_particle_data(
        self,
        name: str,
        ptype: str,
        group_indices: int | list[int] | np.ndarray,
        group_type: str = "galaxies",
        to_units: str | None = None,
        to_physical: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Reads a particle quantity from the raw snapshot for the members of one or more groups. Values are returned in
        Octavius code units (h factors removed) unless converted with to_units.

        Parameters
        ----------
        name: str
            The Octavius name of the particle quantity (e.g. ``mass``, ``sfr``, ``metallicity``, ``temperature``).
        ptype: str
            The particle type to read: ``star``, ``gas``, ``dm`` or ``bh``.
        group_indices: int or list[int] or np.ndarray
            The index (or indices) into the Octavius catalogue of the group(s) to read particles for.
        group_type: str, optional
            The type of group the indices are specified for: ``galaxies`` (default) or ``haloes``.
        to_units: str, optional
            Target units to convert the quantity to.
        to_physical: bool, optional
            Apply conversion to physical (default: code units, which may be comoving).

        Returns
        -------
        values: np.ndarray
            If group_indices is a single integer, the particle values of that group.
        values, offsets: tuple[np.ndarray, np.ndarray]
            If group_indices is a list or array, the particle values of all the groups concatenated in the order given,
            and the offsets such that ``values[offsets[i] : offsets[i + 1]]`` belongs to ``group_indices[i]``.

        AI-generated: claude-opus-5-5
        """
        if group_type not in self._collections:
            raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")

        membership_group = self._collections[group_type]._data["membership"]
        if f"{ptype}_indices" not in membership_group:
            raise KeyError(f"No {ptype} membership for {group_type} in the catalogue.")

        if not self._reader.has_dataset(ptype=ptype, dataset=name):
            raise KeyError(self._reader.missing_dataset_message(ptype=ptype, dataset=name))

        single_group = np.ndim(group_indices) == 0
        indices = np.atleast_1d(np.asarray(group_indices, dtype=np.int64))

        n_groups = len(self._collections[group_type])
        if np.any(indices < 0) or np.any(indices >= n_groups):
            raise IndexError(f"Group indices must lie in [0, {n_groups}) for {group_type}.")

        subset = self._extract_ptype(
            ptype=ptype, membership_group=membership_group, group_indices=indices, columns=[name]
        )
        values = subset.columns[name]

        if to_physical:
            values *= self._reader.simulation_attributes.scale_factor ** CODE_UNITS[name].a_exponent

        if to_units is not None:
            values *= CODE_UNITS[name].unit.to(u.Unit(to_units))

        if single_group:
            return values

        return values, subset.offsets

    def get_xray_gas(self, halo_index: int, radius_factor: float = 5.0, region: str = "sphere") -> XrayGas:
        """
        Reads the gas around a halo, in physical units, for making mock X-ray observations of it.

        Parameters
        ----------
        halo_index: int
            The index into the Octavius catalogue of the halo.
        radius_factor: float, optional
            For ``region="sphere"``, the radius of the sphere in units of the halo's r500c (default: 5).
        region: str, optional
            ``sphere`` (default) reads all gas within radius_factor * r500c of the halo's minimum potential, searching
            the whole snapshot; ``halo`` reads only the gas belonging to the halo, which is much quicker for large
            snapshots but misses gas outside the halo's boundary.

        Returns
        -------
        gas: XrayGas
            The gas quantities (keyed by Octavius name) with the halo's centre, r500c, m500c and the cosmology, and
            the positions and masses of the other particle types in the same region (for total mass profiles).

        AI-generated: claude-opus-5-5
        """
        from ..xray import (  # avoid circular import
            XrayGas,
            find_particles_within_radius,
            read_physical_columns,
            readable_xray_columns,
        )

        haloes = self._collections.get("haloes")
        if haloes is None:
            raise KeyError("'haloes' are not present in the catalogue.")
        if halo_index < 0 or halo_index >= len(haloes):
            raise IndexError(f"Halo index must lie in [0, {len(haloes)}).")

        available = haloes.keys()
        missing = [name for name in ("minpot_pos", "r500c", "m500c") if name not in available]
        if missing:
            raise KeyError(f"The catalogue is missing {missing}; r500c and m500c need 500 in virial_factors.")

        simulation = self._reader.simulation_attributes
        scale_factor = simulation.scale_factor

        # catalogue values in code units (comoving kpc)
        centre = haloes.get_dataset("minpot_pos")[halo_index].astype(np.float64)
        r500 = float(haloes.get_dataset("r500c")[halo_index])
        m500 = float(haloes.get_dataset("m500c")[halo_index])

        if region not in ("sphere", "halo"):
            raise ValueError(f"region must be 'sphere' or 'halo', not '{region}'.")
        region_radius = radius_factor * r500 if region == "sphere" else 0.0

        def _region_indices(ptype: str) -> np.ndarray:
            if region == "sphere":
                return find_particles_within_radius(
                    reader=self._reader, ptype=ptype, centre=centre, radius=region_radius
                )
            return np.unique(haloes.get_particle_indices(ptype=ptype, group_index=halo_index))

        gas_indices = _region_indices(ptype="gas")
        logger.info(f"Reading {len(gas_indices):,} gas particles for X-ray observations of halo {halo_index}.")
        columns = read_physical_columns(
            reader=self._reader,
            ptype="gas",
            columns=readable_xray_columns(reader=self._reader),
            snapshot_indices=gas_indices,
            centre=centre,
        )

        # the other matter is only needed for total mass profiles, so positions and masses suffice
        other_matter = {}
        processed = [ptype for ptype, wanted in self._config.process_ptypes.items() if wanted]
        for ptype in self._reader.available_ptypes:
            if ptype == "gas" or ptype not in processed:
                continue
            other_matter[ptype] = read_physical_columns(
                reader=self._reader,
                ptype=ptype,
                columns=["pos", "mass"],
                snapshot_indices=_region_indices(ptype=ptype),
                centre=centre,
            )

        return XrayGas(
            columns=columns,
            other_matter=other_matter,
            halo_index=int(halo_index),
            centre=centre * scale_factor,
            r500=r500 * scale_factor,
            m500=m500,
            region_radius=region_radius * scale_factor,
            redshift=simulation.redshift,
            h=simulation.h,
            omega_matter=simulation.omega_matter,
            omega_lambda=simulation.omega_lambda,
        )

    def xray_observation(
        self,
        halo_index: int,
        radius_factor: float = 5.0,
        region: str = "sphere",
        make_grad_fields: bool = False,
        **observation_kwargs,
    ) -> Observation:
        """
        Prepares a moxha Observation of a halo, ready for its MakePhotons() and ObservePhotons() methods. Needs the
        optional X-ray dependencies (moxha, yt, pyXSIM and SOXS).

        Parameters
        ----------
        halo_index: int
            The index into the Octavius catalogue of the halo.
        radius_factor: float, optional
            Radius of the gas sphere read, in units of r500c (default: 5); MakePhotons(sphere_R500s=...) should not
            exceed it.
        region: str, optional
            ``sphere`` (default) or ``halo``; see get_xray_gas().
        make_grad_fields: bool, optional
            Passed to moxha's Observation.load_ds() to add gradient fields to the dataset (default: ``False``).
        **observation_kwargs
            Passed to moxha's Observation (save_dir, run_ID, emin, emax, emin_for_EW_values, emax_for_EW_values,
            energies_for_Lx_tot, redshift, ...). box_path and snap_num default to the snapshot's path and number.

        Returns
        -------
        observation: moxha.observations.Observation
            The Observation, with its dataset loaded and the halo set as its active halo.

        AI-generated: claude-opus-5-5
        """
        from ..xray.xray_observations import build_xray_observation  # avoid circular import

        snapshot_path = self._reader.snapshot_path
        snapshot_digits = "".join(character for character in snapshot_path.stem.split("_")[-1] if character.isdigit())
        observation_kwargs.setdefault("box_path", str(snapshot_path))
        observation_kwargs.setdefault("snap_num", int(snapshot_digits) if snapshot_digits else 0)

        gas = self.get_xray_gas(halo_index=halo_index, radius_factor=radius_factor, region=region)

        return build_xray_observation(gas=gas, make_grad_fields=make_grad_fields, observation_kwargs=observation_kwargs)

    def _extract_groups(
        self,
        group_type: str,
        group_indices: np.ndarray,
        ptype_columns: dict[str, list[str]],
    ) -> dict[str, SubsetParticleData]:
        """
        Extracts groups from the catalogue and then their particle-level requisite data. Returns:

        - result: a dictionary of SubsetParticleData classes, keyed by ptype.
        """
        membership_group = self._collections[group_type]._data["membership"]  # NOTE: hardcoded, update if changing

        result: dict[str, SubsetParticleData] = {}

        for ptype, columns in ptype_columns.items():
            result[ptype] = self._extract_ptype(
                ptype=ptype, membership_group=membership_group, group_indices=group_indices, columns=columns
            )

        return result

    def _extract_ptype(
        self,
        ptype: str,
        membership_group: h5py.Group,
        group_indices: np.ndarray,
        columns: list[str],
    ) -> SubsetParticleData:
        """
        Extracts data for requested columns for the specified ptype. Returns:

        - result: a SubsetParticleData for the requested ptype.
        """
        indices_dataset = membership_group[f"{ptype}_indices"]  # slab-read indices (large)
        offsets = membership_group[f"{ptype}_offsets"][:]  # full read offsets

        starts = offsets[group_indices]
        ends = offsets[group_indices + 1]
        lengths = ends - starts  # to make code cleaner
        total_particles = np.sum(lengths)

        # allocate outputs
        particle_indices = np.empty(total_particles, dtype=np.int64)
        new_offsets = np.empty(len(group_indices) + 1, dtype=np.int64)  # not snapshot aligned

        # construct new offsets from old offsets masked by group indices
        new_offsets[0] = 0
        np.cumsum(lengths, out=new_offsets[1:])

        # loop over groups, collect their particles
        position = 0  # positional index
        for i in range(len(group_indices)):
            if lengths[i] > 0:  # need this because a group can miss a ptype
                particle_indices[position : position + lengths[i]] = indices_dataset[starts[i] : ends[i]]

            position += lengths[i]

        # reads need strictly increasing indices; inclusive groups (e.g. a halo and its subhalo) can share particles
        unique_indices, inverse = np.unique(particle_indices, return_inverse=True)

        column_data = self._reader.read_requested_columns(
            ptype=ptype,
            datasets=columns,
            sorted_snapshot_indices=unique_indices,
        )

        # then map back to group order (necessary for sorted_idx in pipeline stages)
        for col in column_data:
            column_data[col] = column_data[col][inverse]

        data = SubsetParticleData(columns=column_data, offsets=new_offsets)

        return data

    def _resolve_stage_columns(
        self,
        stage: PipelineStage,
        group_type: str,
    ) -> dict[str, list[str]]:
        """
        Stages require different columns for each ptype and not all columns exist for each ptype; this
        helper resolves this. Returns:

        - result: dict keyed by ptype containing the list of columns needed for that ptype.
        """
        group_config = self._internals.group_types[group_type]

        # discern which ptypes to access from what exists, what the stage needs, and what the user wants
        all_group_ptypes = group_config["ptypes"]  # from internals
        requested = {pt for pt, on in self._config.process_ptypes.items() if on}  # from config
        available = set(self._reader.available_ptypes)  # from raw snap
        valid_ptypes = [pt for pt in all_group_ptypes if pt in requested and pt in available]

        result: dict[str, list[str]] = {}
        # columns required by all ptypes (e.g. mass)
        shared_required = stage.needs_particle_columns.get("all", frozenset())
        shared_optional = stage.optional_particle_columns.get("all", frozenset())

        for ptype in valid_ptypes:
            # union the shared reqs with the ptype-specific needs
            columns = shared_required | stage.needs_particle_columns.get(ptype, frozenset())

            # union optional columns
            optional = shared_optional | stage.optional_particle_columns.get(ptype, frozenset())
            columns |= {col for col in optional if self._reader.has_dataset(ptype, col)}

            # ensure columns are loaded in the same order
            if columns:
                result[ptype] = sorted(columns)

        return result

    def _verify_dependencies(
        self,
        stage: PipelineStage,
        group_type: str,
    ) -> None:
        """
        Verifies the analysis can be done from what exists in the catalogue (stage dependencies).
        """
        data = self._collections[group_type]._data

        for dependent in stage.requires:
            dependent_label = self._internals.stages[dependent].label

            if f"properties/{dependent_label}" not in data:
                raise KeyError(f"Dependent stage {dependent_label} is not present in the catalogue.")


@dataclass(frozen=True, slots=True)
class SubsetParticleData:
    """
    Container for a subset of snapshot particle data, to be passed to ParticleStores.
    """

    columns: dict[str, np.ndarray]
    offsets: np.ndarray


def _extract_results(
    group_store: GroupStore,
    group_type: str,
    group_indices: np.ndarray,
    pre_columns: set[str],
) -> StageResult:
    """
    Diffs the GroupStore from before & after the stage (and strips _-prefixed columns) to return the
    StageResult of outputs you would expect from the pipeline.
    """
    new_columns = set(group_store.columns.keys()) - pre_columns
    result_columns = {
        col: group_store[col]
        for col in new_columns
        if not col.startswith("_")  # convention in pipeline is to prefix non-HDF5 columns with _
    }

    return StageResult(
        group_type=group_type,
        group_indices=group_indices,
        columns=result_columns,
    )


def _build_particle_stores(
    subset_data: dict[str, SubsetParticleData], internals: Internals
) -> dict[str, ParticleStore]:
    """
    Constructs subset ParticleStores from the snapshot data stored in subset_data. Returns the
    particles dict, same as the pipeline.
    """
    particles: dict[str, ParticleStore] = {}

    for ptype, subset in subset_data.items():
        n_particles = subset.offsets[-1]
        is_baryonic = ptype in internals.baryonic_ptypes
        store = ParticleStore(ptype=ptype, n_particles=n_particles, is_baryonic=is_baryonic)

        for col_name, col_data in subset.columns.items():
            store[col_name] = col_data

        particles[ptype] = store

    return particles


def _build_group_store(
    group_type: str,
    group_indices: np.ndarray,
    subset_data: dict[str, SubsetParticleData],
    internals: Internals,
) -> GroupStore:
    """
    Constructs a GroupStore; creates sorted_idx with np.arange().
    """
    group_config = internals.group_types[group_type]

    store = GroupStore(
        group_ids=group_indices,  # this works because IDs aren't really IDs, rather, indices
        group_key=group_config["key"],
        kind=group_config["kind"],
    )

    for ptype, subset in subset_data.items():
        sorted_idx = np.arange(subset.offsets[-1], dtype=np.int64)
        store.csr_membership[ptype] = (subset.offsets, sorted_idx)

    return store


def _resolve_linked_groups(
    collection: GroupCollection,
    group_idx: np.ndarray,
    linking_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves groups for stages which require another group to be present (e.g. photometry needs halo gas). Returns:

    - unique_map_idx: unique indices of mapped groups which need to be loaded (e.g. field haloes)
    - multi_map_link: array where multi_map_link[g] links to mapped group (for representing many-to-one)
    """
    link_values = collection._data["membership"][linking_column][group_idx]

    unique_map_idx = np.unique(link_values)  # only load parent/child groups once
    unique_map_idx = unique_map_idx[unique_map_idx != -1]  # filter sentinel values

    # map from catalogue level to the subset we're running on
    subset_map = np.full(shape=(np.max(unique_map_idx) + 1), fill_value=-1, dtype=np.int64)
    subset_map[unique_map_idx] = np.arange(len(unique_map_idx), dtype=np.int64)
    multi_map_link = subset_map[link_values]  # multi-membership mapping (since we did np.unique)

    return unique_map_idx, multi_map_link


def _collect_all_dependencies(internals: Internals, stage: PipelineStage) -> list[str]:
    """
    The simple first half of pipeline_management/resolve_dependencies, to get all dependencies beyond
    the immediate.
    """
    visited: set[str] = set()
    stack = list(stage.requires)

    while stack:
        dep_name = stack.pop()
        if dep_name in visited:
            continue
        visited.add(dep_name)
        stack.extend(internals.stages[dep_name].requires)

    return list(visited)


def _preload_stage_dependencies(
    collection: GroupCollection,
    group_store: GroupStore,
    group_indices: np.ndarray,
    stage: PipelineStage,
    internals: Internals,
) -> None:
    """
    Load columns output by a stage's dependencies from the catalogue (file checks will run before this
    to raise an error if they don't exist)
    """
    all_deps = _collect_all_dependencies(internals=internals, stage=stage)

    for dep_name in all_deps:
        dep_label = internals.stages[dep_name].label
        dep_group = collection._data[f"properties/{dep_label}"]

        for col_name in dep_group:
            all_data = dep_group[col_name][:]  # load all for speed
            group_store[col_name] = all_data[group_indices]


def align_orientations(
    galaxies: GroupStore,
    orientation: str | np.ndarray,
) -> None:
    """
    In-place modifies the GroupStores to rotate particles into the requested reference frame (or rotation
    matrix). This will rotate galaxies relative to the z component of their angular momentum.
    """
    n_galaxies = galaxies.n_groups
    rotation_matrices = np.empty(shape=(n_galaxies, 3, 3), dtype=np.float64)

    if isinstance(orientation, np.ndarray):  # early return if pre-passed matrix
        rotation_matrices[:] = orientation
        galaxies["_rotation_matrices"] = rotation_matrices
        return

    L_vectors = galaxies["L_baryon"]
    L_mags = np.linalg.norm(L_vectors, axis=1)
    L_unit_vectors = guarded_divide(L_vectors, L_mags[:, np.newaxis])  # new axis for shape consistency

    if orientation == "face-on":
        rotation_vector = [0, 0, 1]
    elif orientation == "side-on":
        rotation_vector = [1, 0, 0]
    else:
        raise ValueError(f"{orientation} is not currently supported; check typo?")

    for g in range(n_galaxies):
        if np.any(np.isnan(L_unit_vectors[g])):  # prevent NaN galaxies from falling through
            logger.debug(f"Galaxy {g} has NaN in its L unit vector.")
            rotation_matrices[g] = np.eye(3)  # give them identity matrix
        else:
            rotation_matrices[g] = Rotation.align_vectors([rotation_vector], [L_unit_vectors[g]])[0].as_matrix()

    galaxies["_rotation_matrices"] = rotation_matrices
