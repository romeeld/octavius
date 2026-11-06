"""

Checks that the optional packages for the X-ray modules are installed, with an error saying how to install them if not:
octavius[xray] for mock observations (yt, pyXSIM, SOXS), and octavius[xray-fitting] for the profile and spectral fits
and tessellated maps (threeML, emcee, lmfit and others).

"""

from importlib.util import find_spec

XRAY_PACKAGES = ["yt", "pyxsim", "soxs", "unyt", "pooch", "regions", "tqdm"]
XRAY_FITTING_PACKAGES = [
    "threeML",
    "astromodels",
    "emcee",
    "corner",
    "lmfit",
    "bces",
    "nmmn",
    "pandas",
    "vorbin",
    "IPython",
]


def require_packages(packages: list[str], extra: str, purpose: str) -> None:
    """
    Raises an ImportError naming the missing packages, and the Octavius extra that installs them, if any of packages
    cannot be imported.

    AI-generated: claude-opus-5-5
    """
    missing = [package for package in packages if find_spec(package) is None]
    if missing:
        raise ImportError(
            f"{purpose} needs {', '.join(missing)}, which are not installed: install them with "
            f"'pip install octavius[{extra}]'."
        )
