"""eddytools

Python package to compute the Okubo-Weiss parameter, detect, track, sample
and average ocean eddies and their properties.
The package is based on Chelton et al. (2011) and Oliver et al. (2015),
modified by Rafael Abel, Tobias Schulzki and Klaus Getzlaff
(https://git.geomar.de/Eddy_tracking/WGC_Eddies), originating
from work by Chirstopher Bull.
"""

from . import okuboweiss
from . import interp
from . import detection
from . import tracking
from . import sample
from . import average
from . import dummy_ds
