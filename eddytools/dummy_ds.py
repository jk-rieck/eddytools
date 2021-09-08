'''dummy_ds

Create a dummy dataset that is xgcm compatible and can be used for testing.
Additionally gives expected values for some variables that can be
calulated from this dataset. E.g. the Okubo-Weiss parameter.

'''
import numpy as np
import xarray as xr
import xgcm
import pandas as pd


def dummy():
    '''Define a 2D xgcm-compatible dummy dataset and return it along with
    the according xgcm grid instance and some derived quantities like
    the Okubo-Weiss parameter, vorticty etc.
    '''
    time = pd.to_datetime(['2009-01-03T12:00:00.0', '2009-01-08T12:00:00.0'])
    z_c = np.arange(1., 3.)
    z_l = np.arange(0.5, 2.)
    y_c = np.arange(1., 4.)
    y_r = np.arange(1.5, 4.)
    x_c = np.arange(1., 4.)
    x_r = np.arange(1.5, 4.)
    llat_c = np.linspace(-40.2, -40.0, len(y_c))
    llat_r = np.linspace(-40.15, -39.95, len(y_r))
    llon_c = np.linspace(170.0, 170.2, len(x_c))
    llon_r = np.linspace(170.05, 170.25, len(x_r))
    llon_cc, llat_cc = np.meshgrid(llon_c, llat_c * (-1.))
    llon_cr, llat_cr = np.meshgrid(llon_r, llat_c * (-1.))
    llon_rc, llat_rc = np.meshgrid(llon_c, llat_r * (-1.))
    llon_rr, llat_rr = np.meshgrid(llon_r, llat_r * (-1.))
    dummy_u2d = np.array([[2., 1., 2.], [1., 2., 2.], [0., 1., 3.]])
    dummy_u3d = np.array([dummy_u2d, dummy_u2d])
    dummy_u4d = np.array([dummy_u3d, dummy_u3d])
    dummy_v2d = np.array([[3., 2., 1.], [3., 2., 0.], [2., 1., 1.]])
    dummy_v3d = np.array([dummy_v2d, dummy_v2d])
    dummy_v4d = np.array([dummy_v3d, dummy_v3d])
    dummy_2d = np.meshgrid(np.arange(1., 4.), np.arange(1., 4.))[1]
    dummy_3d = np.ones(np.shape(dummy_u3d))
    bathymetry = dummy_2d * 1000.

    ds_xgcm = xr.Dataset({'u': (['t', 'z_c', 'y_c', 'x_r'], dummy_u4d),
                          'v': (['t', 'z_c', 'y_r', 'x_c'], dummy_v4d),
                          'bathymetry': (['y_c', 'x_c'], bathymetry)},
                         coords={'t': (['t'], time),
                                 'z_c': (['z_c'], z_c,
                                         {'axis': 'Z'}),
                                 'z_l': (['z_l'], z_l,
                                         {'axis': 'Z',
                                          'c_grid_axis_shift': -0.5}),
                                 'y_c': (['y_c'], y_c,
                                         {'axis': 'Y'}),
                                 'y_r': (['y_r'], y_r,
                                         {'axis': 'Y',
                                          'c_grid_axis_shift': 0.5}),
                                 'x_c': (['x_c'], x_c,
                                         {'axis': 'X'}),
                                 'x_r': (['x_r'], x_r,
                                         {'axis': 'X',
                                          'c_grid_axis_shift': 0.5}),
                                 'llat_cc': (['y_c', 'x_c'], llat_cc),
                                 'llat_cr': (['y_c', 'x_r'], llat_cr),
                                 'llat_rc': (['y_r', 'x_c'], llat_rc),
                                 'llat_rr': (['y_r', 'x_r'], llat_rr),
                                 'llon_cc': (['y_c', 'x_c'], llon_cc),
                                 'llon_cr': (['y_c', 'x_r'], llon_cr),
                                 'llon_rc': (['y_r', 'x_c'], llon_rc),
                                 'llon_rr': (['y_r', 'x_r'], llon_rr),
                                 'e1t': (['y_c', 'x_c'], dummy_2d),
                                 'e2t': (['y_c', 'x_c'], dummy_2d),
                                 'e3t': (['z_c', 'y_c', 'x_c'], dummy_3d),
                                 'e1u': (['y_c', 'x_r'], dummy_2d),
                                 'e2u': (['y_c', 'x_r'], dummy_2d),
                                 'e3u': (['z_c', 'y_c', 'x_r'], dummy_3d),
                                 'e1v': (['y_r', 'x_c'], dummy_2d),
                                 'e2v': (['y_r', 'x_c'], dummy_2d),
                                 'e3v': (['z_c', 'y_r', 'x_c'], dummy_3d),
                                 'e1f': (['y_r', 'x_r'], dummy_2d),
                                 'e2f': (['y_r', 'x_r'], dummy_2d),
                                 'e3f': (['z_c', 'y_r', 'x_r'], dummy_3d),
                                 'at': (['y_c', 'x_c'], dummy_2d * dummy_2d),
                                 'au': (['y_c', 'x_r'], dummy_2d * dummy_2d),
                                 'av': (['y_r', 'x_c'], dummy_2d * dummy_2d),
                                 'af': (['y_r', 'x_r'], dummy_2d * dummy_2d)})

    metrics = {('X',): ['e1t', 'e1u', 'e1v', 'e1f'],  # X distances
               ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'],  # Y distances
               ('Z',): ['e3t', 'e3u', 'e3v', 'e3f'],  # Z distances
               ('X', 'Y'): ['at', 'au', 'av', 'af']}  # Areas

    grid_xgcm = xgcm.Grid(ds_xgcm, periodic=False, metrics=metrics)

    exp_vort_2d = np.array([[+0.0000, -2.0000, +0.0000],
                            [+0.0000, -1.0000, -1.0000],
                            [-1.0000, +0.0000, +0.0000]])
    exp_vort_3d = np.array([exp_vort_2d, exp_vort_2d])
    expected_vort = np.array([exp_vort_3d, exp_vort_3d])
    exp_ow_2d = np.array([[+4.0000, -3.7500, +0.2500],
                          [+5.0000, +9.5625, +0.0000],
                          [+2.2500, +2.2500, +1.0000]])
    exp_ow_3d = np.array([exp_ow_2d, exp_ow_2d])
    expected_ow = np.array([exp_ow_3d, exp_ow_3d])

    ds_xgcm = ds_xgcm.update({'expected_vort': (['t', 'z_c', 'y_r', 'x_r'],
                                                expected_vort),
                              'expected_ow': (['t', 'z_c', 'y_r', 'x_r'],
                                              expected_ow)})
    return ds_xgcm, grid_xgcm
