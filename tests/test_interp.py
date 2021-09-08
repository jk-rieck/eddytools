import numpy as np
import pytest
import xesmf as xe
import eddytools as et

# Define a 2D xgcm-compatible dummy dataset
ds_xgcm, grid_xgcm = et.dummy_ds.dummy()

# Create interpolation parameters
int_param = {'start_time': '2009-01-01',
             'end_time': '2009-01-10',
             'lon1': np.floor(ds_xgcm['llon_cc'][0, 0].values * 10.) / 10.,
             'lon2': np.ceil(ds_xgcm['llon_rr'][0, -1].values * 10.) / 10.,
             'lat1': np.floor(ds_xgcm['llat_cc'].min().values * 10.) / 10.,
             'lat2': np.ceil(ds_xgcm['llat_rr'].max().values * 10.) / 10.,
             'res': 1./10.,
             'vars_to_interpolate': ['OW', 'vort'],
             'mask_to_interpolate': ['bathymetry']}

var_int_u1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.5, 2.0, 0.0],
                       [0.0, 1.5, 1.5, 0.0]])
var_int_u = np.array([[var_int_u1, var_int_u1], [var_int_u1, var_int_u1]])
var_int_v1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 2.5, 1.0, 0.0],
                       [0.0, 2.5, 1.5, 0.0]])
var_int_v = np.array([[var_int_v1, var_int_v1], [var_int_v1, var_int_v1]])


@pytest.mark.parametrize('var',
                         ['u', 'v'])
def test_horizontal(var):
    int_para = int_param.copy()
    int_para['vars_to_interpolate'] = [var]
    data_int = et.interp.horizontal(ds_xgcm, int_para)
    data_int_values = np.around(data_int[var].values, 3)
    if var == 'u':
        assert (data_int_values == var_int_u).all(),\
            'interpolation failed'
    if var == 'v':
        assert (data_int_values == var_int_v).all(),\
            'interpolation failed'


@pytest.mark.parametrize('lon1',
                         [170.])
@pytest.mark.parametrize('lon2',
                         [170.25, 170.25 - 360.])
@pytest.mark.parametrize('lat1',
                         [-40.2])
@pytest.mark.parametrize('lat2',
                         [-40., -40.4])
def test_create_rect_grid(lon1, lon2, lat1, lat2):
    int_para = int_param.copy()
    int_para['lon1'] = lon1
    int_para['lon2'] = lon2
    int_para['lat1'] = lat1
    int_para['lat2'] = lat2
    rect_grid, lon, lat = et.interp.create_rect_grid(int_para)
    assert 'lat' in rect_grid,\
        'rectangular grid `rect_grid` is missing `lat`'
    assert 'lon' in rect_grid,\
        'rectangular grid `rect_grid` is missing `lon`'
    assert lat[0] < lat[-1],\
        'latitude vector is decreasing, not increasing'
    assert lon[0] < lon[-1],\
        'longitude vector is decreasing, not increasing'
    assert np.mean(lon[1::] - lon[0:-1]) == (lon[1] - lon[0]),\
        'longitude is not regular'
    assert np.mean(lat[1::] - lat[0:-1]) == (lat[1] - lat[0]),\
        'latitude is not regular'


@pytest.mark.parametrize('data',
                         [ds_xgcm, ds_xgcm.isel(z_c=0), ds_xgcm.isel(z_l=0),
                          ds_xgcm.isel(z_c=0, z_l=0)])
def test_create_empty_ds(data):
    _, lon, lat = et.interp.create_rect_grid(int_param)
    data_int = et.interp.create_empty_ds(data, int_param, lon, lat)
    if 'z_c' in data.dims or 'z_l' in data.dims:
        assert 'z' in data_int.dims,\
            'no depth dimension in data_int'
    assert 'time' in data_int.dims,\
        'no time dimension in data_int'
    assert 'lon' in data_int.dims,\
        'no latitude dimension in data_int'
    assert 'lat' in data_int.dims,\
        'no longitude dimension in data_int'
    assert len(data_int.data_vars) == 0,\
        'data_int is not empty'


@pytest.mark.parametrize('var',
                         ['u', 'v', 'bathymetry'])
def test_rename_dims(var):
    var_to_int = ds_xgcm[var]
    var_to_int = et.interp.rename_dims(var_to_int)
    assert 'lon' in var_to_int.coords,\
        'longitude has not been renamed'
    assert 'lat' in var_to_int.coords,\
        'latitude has not been renamed'
    if var == 'u' or var == 'v':
        assert 'z' in var_to_int.coords,\
            'depth has not been renamed'


@pytest.mark.parametrize('var',
                         ['u', 'v', 'bathymetry'])
def test_monotonic_lon(var):
    var_to_int = ds_xgcm[var]
    var_to_int = et.interp.rename_dims(var_to_int)
    var_to_int = et.interp.monotonic_lon(var_to_int)
    assert var_to_int['lon'][0, 0] < var_to_int['lon'][0, -1],\
        'longitude is still not increasing'


@pytest.mark.parametrize('var',
                         ['u', 'v', 'bathymetry'])
def test_update_data(var):
    rect_grid, lon, lat = et.interp.create_rect_grid(int_param)
    data_int = et.interp.create_empty_ds(ds_xgcm, int_param, lon, lat)
    var_to_int = ds_xgcm[var]
    var_to_int = et.interp.rename_dims(var_to_int)
    var_to_int = et.interp.monotonic_lon(var_to_int)
    regridder = xe.Regridder(var_to_int, rect_grid, 'bilinear',
                             reuse_weights=True)
    var_int = regridder(var_to_int)
    data_int = et.interp.update_data(data_int, var_int, var)
    assert var in data_int,\
        'variable has not been added to data_int'
