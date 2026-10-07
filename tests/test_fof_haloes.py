"""

Tests the periodic 3D friends-of-friends halo finder against a brute-force reference (scipy k-d tree pairs ->
connected components), including haloes straddling the periodic boundary and slab boundaries.

"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from octavius.data_management import OctaviusConstants
from octavius.data_management.conventions import OctaviusConfig
from octavius.data_management.snapshot_readers import build_reader
from octavius.external_halo_sources.halo_data_structures import SnapshotHaloSource, build_halo_source
from octavius.galaxy_finding.fof_halo_algorithm import find_fof_haloes
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot, generate_swift_snapshot
from tests.validation.output_validation import validate_group_counts, validate_halo_membership

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"

SEED = 8812391
BOXSIZE = 100.0
LINKING_LENGTH = 1.5


def make_clustered_positions(rng: np.random.Generator, n_background: int, n_clusters: int) -> np.ndarray:
    """
    Uniform background plus Gaussian clusters, some centred on the box edges/corners to test periodicity.
    """
    centres = rng.uniform(0, BOXSIZE, size=(n_clusters, 3))
    centres[0] = [0.0, 50.0, 50.0]
    centres[1] = [BOXSIZE, BOXSIZE, 0.0]
    clusters = [c + rng.normal(0, 1.5, size=(rng.integers(20, 400), 3)) for c in centres]
    background = rng.uniform(0, BOXSIZE, size=(n_background, 3))

    return np.mod(np.concatenate([background, *clusters]), BOXSIZE)


def reference_groups(pos: np.ndarray, min_members: int) -> np.ndarray:
    """
    Brute-force FOF; returns component labels with groups below min_members set to -1 (labels arbitrary).
    """
    pairs = cKDTree(pos, boxsize=BOXSIZE).query_pairs(r=LINKING_LENGTH, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pos), len(pos)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    labels[sizes[labels] < min_members] = -1

    return labels


def assert_same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> None:
    """
    Asserts two labellings define the same groups (up to relabelling), with -1 meaning ungrouped in both.
    """
    assert np.array_equal(labels_a == -1, labels_b == -1)
    grouped = labels_a != -1
    pairs = np.unique(np.stack([labels_a[grouped], labels_b[grouped]], axis=1), axis=0)
    assert len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))


@pytest.mark.parametrize("n_slabs", [1, 3, 16, 1000, None])  # 1000 is clipped to the thinnest allowed slabs
@pytest.mark.parametrize("min_members", [1, 20])
def test_fof_matches_brute_force(n_slabs: int | None, min_members: int) -> None:
    rng = np.random.default_rng(SEED)
    pos = make_clustered_positions(rng, n_background=20000, n_clusters=40)

    halo_ids = find_fof_haloes(
        dm_pos=pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, min_members=min_members, n_slabs=n_slabs
    )["dm"]

    assert_same_partition(halo_ids, reference_groups(pos, min_members))


def test_halo_ids_ordered_by_size() -> None:
    rng = np.random.default_rng(SEED)
    pos = make_clustered_positions(rng, n_background=5000, n_clusters=20)

    halo_ids = find_fof_haloes(dm_pos=pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, min_members=20)["dm"]
    sizes = np.bincount(halo_ids[halo_ids >= 0])

    assert sizes.min() >= 20
    assert np.all(np.diff(sizes) <= 0)


def test_baryons_attach_to_nearest_dm() -> None:
    rng = np.random.default_rng(SEED)
    dm_pos = make_clustered_positions(rng, n_background=5000, n_clusters=20)
    gas_pos = make_clustered_positions(rng, n_background=3000, n_clusters=20) - 0.3  # some fall below 0: wrapped

    halo_ids = find_fof_haloes(
        dm_pos=dm_pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, baryon_pos={"gas": gas_pos}, min_members=20
    )

    dist, nearest = cKDTree(dm_pos, boxsize=BOXSIZE).query(
        np.mod(gas_pos, BOXSIZE), distance_upper_bound=LINKING_LENGTH
    )
    expected = np.full(len(gas_pos), -1, dtype=np.int64)
    found = np.isfinite(dist)
    expected[found] = halo_ids["dm"][nearest[found]]

    assert np.array_equal(halo_ids["gas"], expected)


def test_rejects_oversized_linking_length() -> None:
    with pytest.raises(ValueError):
        find_fof_haloes(dm_pos=np.zeros((10, 3)), boxsize=BOXSIZE, linking_length=BOXSIZE / 2)


@pytest.mark.parametrize("sim_type", ["SIMBA", "SWIFT-KIARA"])
def test_fof_source_recovers_snapshot_haloes(sim_type: str, tmp_path: Path) -> None:
    """
    The synthetic snapshots place tight haloes amongst uniform interlopers, so FOF should recover the snapshot's own
    haloes exactly (up to relabelling) for every ptype. An interloper may legitimately land within a linking length of
    a halo and be attached, so those are only required to be rare.
    """
    snapshot_path = tmp_path / "snapshot.hdf5"
    if sim_type == "SIMBA":
        generate_simba_snapshot(path=snapshot_path)
    else:
        generate_swift_snapshot(path=snapshot_path, model="KIARA")

    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type=sim_type,
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="FOF",
        photometry_table_path=None,
        min_dm_per_halo=10,
    )
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    fof = build_halo_source(config=config, reader=reader).read_halo_ids(ptypes=reader.available_ptypes)
    snapshot = SnapshotHaloSource(reader=reader).read_halo_ids(ptypes=reader.available_ptypes)

    assert fof.n_field_haloes == snapshot.n_field_haloes
    for ptype in reader.available_ptypes:
        bound = snapshot.field_ids[ptype] >= 0
        assert_same_partition(fof.field_ids[ptype][bound], snapshot.field_ids[ptype][bound])
        assert np.sum(fof.field_ids[ptype][~bound] >= 0) <= 0.05 * np.sum(~bound)


@pytest.mark.parametrize("without_mpi", [False, True])
def test_pipeline_runs_with_fof_haloes(without_mpi: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if without_mpi:  # exercises the serial path taken when mpi4py is not installed
        monkeypatch.setattr("octavius.run_octavius.get_mpi_communicator", lambda: None)

    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)

    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type="SIMBA",
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="FOF",
        photometry_table_path=None,
        stages={
            "find_galaxies": True,
            "properties_core": True,
            "properties_ptype_specific": True,
            "properties_local_environment": True,
            "photometry": False,
        },
        min_dm_per_halo=10,
        min_stars_per_galaxy=2,
        b=1.5,
        velocity_factor=5,
        compress_catalogue=False,
    )

    with h5py.File(analyse_snapshot(config=config), "r") as catalogue:
        assert len(catalogue["halo_data"]["properties/core/n_dm"]) == 3
        validate_halo_membership(f=catalogue)
        validate_group_counts(f=catalogue, group_data="halo_data")
