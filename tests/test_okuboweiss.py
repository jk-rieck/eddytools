import eddytools as et
import numpy as np

# Define a 2D xgcm-compatible dummy dataset
ds_xgcm, grid_xgcm = et.dummy_ds.dummy()

u_var = 'u'
v_var = 'v'


def test_calc():
    ow = et.okuboweiss.calc(ds_xgcm, grid_xgcm, u_var, v_var)
    ow_values = np.around(ow['OW'].values, 4)
    vort_values = np.around(ow['vort'].values, 4)
    assert (ow_values == ow['expected_ow'].values).all(),\
        'The calculated Okubo-Weiss parameter does not have the expected'\
        + ' values'
    assert (vort_values == ow['expected_vort'].values).all(),\
        'The calculated vorticity does not have the expected values'
