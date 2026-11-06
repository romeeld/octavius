"""

moxha: mock X-ray observations of haloes in hydrodynamical simulations, by Fred Jennings
(https://github.com/Fred-Jennings/moxha), included in Octavius with permission.

The mock observations need the optional extra octavius[xray] (yt, pyXSIM, SOXS). The fitting modules (profilefits,
spectralfits and tessmaps) need octavius[xray-fitting] as well, so they are not imported here: import them explicitly,
e.g. ``from octavius.xray.moxha import profilefits``.

"""

from ..extras import XRAY_PACKAGES, require_packages

require_packages(packages=XRAY_PACKAGES, extra="xray", purpose="Mock X-ray observations")

from . import halo_sources as halo_sources  # noqa: E402
from . import observations as observations  # noqa: E402
from . import particle_dataset as particle_dataset  # noqa: E402
from . import postprocess as postprocess  # noqa: E402
from . import tools as tools  # noqa: E402
