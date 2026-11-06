"""

Finds the data files that mock X-ray observations need, downloading them on first use.

SOXS downloads its own instrument files (responses, backgrounds, PSFs, absorption tables) into its data directory when
they are first needed: by default the user's cache directory, or the SOXS_DATA_DIR environment variable or SOXS
configuration file if set. yt's X-ray emissivity fields need an APEC emissivity table, which yt does not download, so
it is fetched here from yt's data server into the same directory.

The X-ray packages are optional dependencies (pip install octavius[xray]), so they are imported inside the functions.

"""

import os
from pathlib import Path

# the APEC emissivity table read by yt.add_xray_emissivity_field (version 3, as yt expects)
APEC_TABLE = "apec_emissivity_v3.h5"
APEC_TABLE_URL = f"https://yt-project.org/data/{APEC_TABLE}"
APEC_TABLE_HASH = "sha256:3fe97a20e059735ba00ac2c64c9faaf1962920522fb4e7b46616cc3b6146e492"


def xray_data_dir() -> Path:
    """
    Returns the directory where SOXS keeps its downloaded data files, which also holds the APEC emissivity table.

    AI-generated: claude-opus-5-5
    """
    from soxs.utils import soxs_cfg

    # SOXS's downloader uses the SOXS_DATA_DIR environment variable when it is set, else the configured directory
    return Path(os.environ.get("SOXS_DATA_DIR") or soxs_cfg.get("soxs", "soxs_data_dir"))


def instrument_file(file_name: str) -> str:
    """
    Returns the path to a SOXS instrument file (e.g. an ARF or RMF), which SOXS downloads if it does not have it yet.

    AI-generated: claude-opus-5-5
    """
    from soxs.utils import get_data_file

    return get_data_file(file_name)


def apec_table_dir() -> Path:
    """
    Returns the directory holding the APEC emissivity table, to pass as data_dir to yt.add_xray_emissivity_field;
    the table (0.8 MB) is downloaded from yt's data server on first use, and checked against its known hash.

    AI-generated: claude-opus-5-5
    """
    import pooch

    directory = xray_data_dir()
    pooch.retrieve(url=APEC_TABLE_URL, known_hash=APEC_TABLE_HASH, fname=APEC_TABLE, path=directory, progressbar=False)
    return directory
