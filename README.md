# eddytools

Python package to detect, track, sample, cluster, and average eddies and their properties based on ocean model output from the NEMO model on ORCA grids and the MITgcm on cartesian grids. The detection can be based on the Okubo-Weiss parameter or Sea Surface Height.  

Based on work by Tobias Schulzki, Klaus Getzlaff and Rafael Abel. The original algorithm was implemented by [Eric Oliver](https://github.com/ecjoliver/eddyTracking) and further developed by [Christopher Bull](https://github.com/chrisb13/eddyTracking). It is based on the algorithm described in [Chelton et al. (2011)](https://doi.org/10.1016/j.pocean.2011.01.002). The original eddy detection and tracking routines are described in [Oliver et al. (2015)](https://doi.org/10.1002/2015JC010993). The contributions by Tobias Schulzki, Klaus Getzlaff and Rafael Abel are documented [here](https://git.geomar.de/Eddy_tracking/WGC_Eddies).   

The package works based on output loaded with  `load_xorca_dataset` from the [xorca](https://github.com/willirath/xorca) module (for NEMO model output) or [`xmitgcm`](https://xmitgcm.readthedocs.io/en/latest/) (for MITgcm model output) and additionally requires [xgcm](https://github.com/xgcm/xgcm).   


## Install a minimal environment

~~~bash
# clone the git repository
git clone git@git.geomar.de:jrieck-archive/eddytools.git
cd eddytools
# create and activate the environment
conda env create -f environment.yml
conda activate py3_eddy
# install modules not installed through environment.yml
pip install git+https://github.com/willirath/xorca.git@master
# install eddytools
pip install -e .
~~~


## Install in existing environment

1. Make sure you have `python` 3, `numpy`, `scipy`, `pandas`, `xarray`,
   `xesmf`, `xgcm`, `xorca`, `operator` and `pip` installed.

2. Install from the repository using
  ```shell
  pip install git+https://git.geomar.de/jrieck-archive/eddytools.git@v0.1
  ```

## Usage

See the example notebooks for the different cases:  

| use case |
|:-|
| [ORCA Okubo-Weiss](examples/run_eddytools_example_ORCA_OW.ipynb) |
| [ORCA SSH](examples/run_eddytools_example_ORCA_SSH.ipynb) |
| [MITgcm Okubo-Weiss](examples/run_eddytools_example_MITgcm_OW.ipynb) |
| [MITgcm SSH](examples/run_eddytools_example_MITgcm_SSH.ipynb) |
