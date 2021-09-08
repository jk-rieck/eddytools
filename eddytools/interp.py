""" interp

Collection of functions to interpolate xgcm compatible data onto a regular
(rectangular) grid to later perform eddy detection (detection.py).

Note that the interpolation takes a long time, especially for high
resolution, long time series datasets. Until the performance of the methods
in this module is improved, it might be advisable to use different approaches
for the interpolation. For NEMO data, e.g. there is
https://git.geomar.de/jan-klaus-rieck/interpolate-orca.

"""

import numpy as np
import xarray as xr
import xesmf as xe
import xgcm


def horizontal(data, metrics, int_param):
    """ Horizontal interpolation of variables using `xesmf`
    (https://xesmf.readthedocs.io/en/latest/).

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset containing the variables to interpolate. The dataset
        needs to be compatible with xgcm.
    metrics : dict
        Dictionary with the metrics of `data` used by xgcm.
    int_param : dict
        Dictionary specifying all the parameters needed for the interpolation.
        The parameters are
        int_param = {
            'model': 'MITgcm' # MITgcm or ORCA
            'grid': 'cartesian' # cartesian or latlon
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': '360_day', # calendar, must be either 360_day or
                                   # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, (-180, 180)
                          # degrees or m
            'lat1': -55, # minimum latitude of detection region, (-90, 90)
                         # degrees or m
            'lat2': -30, # maximum latitude of detection region, (-90, 90)
                         # degrees or m
            'res': 1./10., # resolution of the regular grid to interpolate to
                           # only used when 'grid' == 'latlon'
            'vars_to_interpolate': ['var1', 'var2', ...],
            'mask_to_interpolate': ['mask1', 'mask2', ...]
            }

    Returns
    -------
    data_int : xarray.DataSet
        Data interpolated onto a regular grid.
    """
    # Initialize all booleans as False
    latlon = False
    cart = False
    m = False
    o = False
    # Check if `model` is either 'MITgcm' or 'ORCA' and which grid it uses
    if (int_param['model'] == 'MITgcm'):
        if int_param['grid'] == 'latlon':
            print('Grid ' + int_param['grid'] + ' not implemented for model '
                  + int_param['model'])
            latlon = True
            m = True
            return
        elif int_param['grid'] == 'cartesian':
            print('Interpolating from model grid: ' + int_param['model'])
            m = True
            cart = True
    elif (int_param['model'] == 'ORCA'):
        if int_param['grid'] == 'cartesian':
            print('Grid ' + int_param['grid'] + ' not possible for model '
                  + int_param['model'])
        elif int_param['grid'] == 'latlon':
            print('Interpolating from model grid: ' + int_param['model'])
            o = True
            latlon = True
    else:
        print('Interpolation from model grid: ' + int_param['model']
              + ' not implemented!')
        return
    # Define the names of the variables in the corresponding model/grid
    # combination. Then add 2 degrees (ORCA) or 200km (MITgcm) in longitude and
    # 1 degree (ORCA) or 100km (MITgcm) in latitude in every direction of the
    # region to make computations of the surroundings of eddies at the regions # boundary possible.
    if cart:
        if m:
            x_r = 'XG'
            x_c = 'XC'
            y_r = 'YG'
            y_c = 'YC'
            t = 'time'
            e1f = 'dxV'
            e2f = 'dyU'
            fmask = 'maskZ'
        add_lon = 200e3
        add_lat = 100e3
    elif latlon:
        if o:
            llon_cc = 'llon_cc'
            llon_rc = 'llon_rc'
            llon_cr = 'llon_cr'
            llon_rr = 'llon_rr'
            llat_rr = 'llat_rr'
            x_r = 'x_r'
            x_c = 'x_c'
            y_r = 'y_r'
            y_c = 'y_c'
            t = 't'
            fmask = 'fmask'
            e1f = 'e1f'
            e2f = 'e2f'
        add_lon = 2
        add_lat = 1
    lon1 = int_param['lon1'] - add_lon
    lon2 = int_param['lon2'] + add_lon
    lat1 = int_param['lat1'] - add_lat
    lat2 = int_param['lat2'] + add_lat
    # If the grid is cartesian, everything is easy, regridding is not
    # necessary and we will not have problems with non-monotonic longitudes
    if cart:
        regrid = False
        # select the desired longitude and latitude ranges
        lon = data[x_r].sel({x_r: slice(lon1, lon2)})
        lat = data[y_r].sel({y_r: slice(lat1, lat2)})
        # Create an empty dataset in which to store the interpolated variables
        data_int = create_empty_ds(data, int_param, lon, lat, t)
        # Construct a xgcm.Grid instance for the interpolation
        grid = xgcm.Grid(data, periodic=["X", "Y"], metrics=metrics)
        lon_coord = data.reset_coords()[x_r][:]
        lat_coord = data.reset_coords()[y_r][:]
    # If the grid is latlon, we need to do some more checks on the longitudes
    # and potentially regrid the data onto a regular grid
    elif latlon:
        # If longitudes at one index i are all the same, the grid is assumed
        # to be regular and we do not need to regrid
        if (np.diff(data[llon_cc][:, 0]) == 0).all():
            print('No regridding necessary,'
                + ' just interpolating to vorticity grid point.')
            regrid = False
            # We still need to make sure that the longitude is increasing from
            # index 0 to the last index
            # If lon2 is smaller 0, we add 360 to all negative longitudes, so
            # that longitudes are then in the range (0, 360)
            if (lon2 < 0):
                for lonn in [llon_cc, llon_rc, llon_cr, llon_rr]:
                    tmp_lon = data[lonn].copy().where(data[lonn] > 0,
                                                      other=data[lonn] + 360)
                    data = data.assign_coords({lonn: tmp_lon})
                # If lon1 is also negative, we need to add 360 to lon1
                if (lon1 < 0):
                    lon = data[llon_rr][0, :].swap_dims({
                        x_r: llon_rr}
                        ).sel(
                        llon_rr=slice(lon1 + 360, lon2 + 360))
                else:
                    lon = data[llon_rr][0, :].swap_dims({
                            x_r: llon_rr}
                            ).sel(
                            llon_rr=slice(lon1, lon2 + 360))
            # Else, we simply extract the desired longitude range
            else:
                lon = data[llon_rr][0, :].swap_dims({
                        x_r: llon_rr}
                        ).sel(
                        llon_rr=slice(lon1, lon2))
            # make sure we extract the correct range of latitudes, even when
            # lat2 is more south then lat1
            if (lat2 < lat1):
                lat = data[llat_rr][:, 0].swap_dims(
                        {y_r: llat_rr}
                        ).sel(
                        llat_rr=slice(lat2, lat1))
            else:
                lat = data[llat_rr][:, 0].swap_dims({
                        y_r: llat_rr}
                        ).sel(
                        llat_rr=slice(lat1, lat2))
            # Create an empty dataset in which to store the interpolated
            # variables
            data_int = create_empty_ds(data, int_param, lon, lat, t)
            # Construct a xgcm.Grid instance for the interpolation
            grid = xgcm.Grid(data, metrics=metrics)
            lon_coord = data.reset_coords()[llon_rr][0, :]
            lat_coord = data.reset_coords()[llat_rr][:, 0]
        else:
            print('Regridding to regular grid is necessary.')
            regrid = True
            # Create a xarray DataSet of the regular grid
            rect_grid, lon, lat = create_rect_grid(int_param)
            # Now create a Dataset with the coordinates of the regular grid,
            # but no interpolated data yet
            data_int = create_empty_ds(data, int_param, lon, lat)
            data = data.chunk({x_c: -1, x_r: -1, y_c: -1, y_r: -1})

    for var in int_param['vars_to_interpolate']:
        # Loop through all the variables specified for interpolation
        # and extract only the time slice wanted
        var_to_int = data[var].sel({t: slice(int_param['start_time'],
                                             int_param['end_time'])})
        # Define how longitude and latitude coordinates are called in the
        # original dataset to rename them later
        if o:
            var_to_int = rename_dims(var_to_int)
        # Make sure the longitude is monotonically increasing for the
        # interpolation in case we have a latlon grid
        if latlon:
            var_to_int = monotonic_lon(var_to_int)
        if regrid:
            # Interpolate on regular grid using the xesmf Regridder
            # For grids that have similar resolutions, method could be changed
            # to 'nearest_s2d', which is quicker and might be sufficient....
            regridder = xe.Regridder(var_to_int, rect_grid, 'bilinear',
                                     reuse_weights=True)
            print('Regridding ' + str(var))
            var_int = regridder(var_to_int)
        else:
            print('Interpolating ' + str(var))
            # The grids are different for MITgcm and ORCA, such that we need to
            # either move the data to the left or right
            if m:
                to = 'left'
            elif o:
                to = 'right'
            # Try to interpolate in X direction, if this fails, data is already
            # at the right location
            try:
                var_to_int = grid.interp(var_to_int, axis='X',
                                         to=to, metric_weighted=["X", "Y"])
            except:
                pass
            # Repeat for the Y direction
            try:
                var_to_int = grid.interp(var_to_int, axis='Y',
                                         to=to, metric_weighted=["X", "Y"])
            except:
                pass
            if m:
                var_to_int = rename_dims(var_to_int)
                var_int = var_to_int.sel(lon=slice(lon[0], lon[-1]),
                                    lat=slice(lat[0], lat[-1]))
            elif o:
                var_to_int = var_to_int.assign_coords({'lon': lon_coord,
                                                       'lat': lat_coord})
                var_int = var_to_int.swap_dims(
                              {x_r: 'lon', y_r: 'lat'}
                              ).sel(lon=slice(lon[0], lon[-1]),
                                    lat=slice(lat[0], lat[-1]))
        # Update `data_int` with the regridded/interpolated variable
        data_int = update_data(data_int, var_int, var)

    # Always interpolate fmask, e1f and e2f
    if fmask not in int_param['mask_to_interpolate']:
        int_param['mask_to_interpolate'].extend([fmask])
    if e1f not in int_param['mask_to_interpolate']:
        int_param['mask_to_interpolate'].extend([e1f])
    if e2f not in int_param['mask_to_interpolate']:
        int_param['mask_to_interpolate'].extend([e2f])
    for mask in int_param['mask_to_interpolate']:
        # Loop through all the masks specified for interpolation
        mask_to_int = data[mask]
        # Define how longitude and latitude coordinates are called in the
        # original dataset to rename them later
        if o:
            mask_to_int = rename_dims(mask_to_int)
        # Make sure the longitude is monotonically increasing for the
        # interpolation
        if latlon:
            mask_to_int = monotonic_lon(mask_to_int)
        if regrid:
            # Interpolate on regular grid using the xesmf Regridder
            # For grids that have similar resolutions, method could be changed
            # to 'nearest_s2d', which is quicker and might be sufficient....
            regridder = xe.Regridder(mask_to_int, rect_grid, 'bilinear',
                                     reuse_weights=True)
            print('Regridding ' + str(mask))
            mask_int = regridder(mask_to_int)
        else:
            print('Interpolating ' + str(mask))
            if m:
                to = 'left'
            elif o:
                to = 'right'
            try:
                mask_to_int = grid.interp(mask_to_int, axis='X',
                                          to=to, metric_weighted=["X", "Y"])
            except:
                pass
            try:
                mask_to_int = grid.interp(mask_to_int, axis='Y',
                                          to=to, metric_weighted=["X", "Y"])
            except:
                pass
            if m:
                mask_to_int = rename_dims(mask_to_int)
                mask_int = mask_to_int.sel(lon=slice(lon[0], lon[-1]),
                                           lat=slice(lat[0], lat[-1]))
            elif o:
                mask_to_int = mask_to_int.assign_coords({'lon': lon_coord,
                                                         'lat': lat_coord})
                mask_int = mask_to_int.swap_dims(
                                           {x_r: 'lon', y_r: 'lat'}
                                           ).sel(lon=slice(lon[0], lon[-1]),
                                                 lat=slice(lat[0], lat[-1]))

        # Update `data_int` with the regridded/interpolated mask
        data_int = update_data(data_int, mask_int, mask)
    if latlon:
        tmp_lon = data_int['lon'].copy().where(data_int['lon'] <= 180.,
                                               other=data_int['lon'] - 360)
        data_int = data_int.assign_coords({'lon': tmp_lon})
        tmp_lon = data_int['lon'].copy().where(data_int['lon'] >= -180.,
                                               other=data_int['lon'] + 360)
        data_int = data_int.assign_coords({'lon': tmp_lon})
    return data_int


def create_rect_grid(int_param):
    '''Create a rectangular grid based on the information (min/max longitude
    and latitude, and resolution) in `int_param`.

    Parameters
    ----------
    int_param : dict
        Dictionary specifying all the parameters needed for the interpolation.
        The parameters are
        int_param = {
            'model': 'MITgcm' # MITgcm or ORCA
            'grid': 'cartesian' # cartesian or latlon
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': '360_day', # calendar, must be either 360_day or
                                   # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, (-180, 180)
                          # degrees or m
            'lat1': -55, # minimum latitude of detection region, (-90, 90)
                         # degrees or m
            'lat2': -30, # maximum latitude of detection region, (-90, 90)
                         # degrees or m
            'res': 1./10., # resolution of the regular grid to interpolate to
                           # only used when 'grid' == 'latlon'
            'vars_to_interpolate': ['var1', 'var2', ...],
            'mask_to_interpolate': ['mask1', 'mask2', ...]
            }

    Returns
    -------
    rect_grid : xarray.DataSet
        xarray dataset containing the rectangular grid created.
    '''
    # Add 2 degrees in longitude and 1 degree in latitude in every direction
    # of the region to make computations of the surroundings of eddies at the
    # regions boundary possible.
    lon1 = int_param['lon1'] - 2
    lon2 = int_param['lon2'] + 2
    lat1 = int_param['lat1'] - 1
    lat2 = int_param['lat2'] + 1
    # First create the longitude vector of the regular grid
    if (lon2 < lon1):
        # Make sure `lon` is increasing
        lon = np.arange(lon1 - 360, lon2, int_param['res'])
    else:
        lon = np.arange(lon1, lon2, int_param['res'])

    # Create the latitude vector of the regular grid
    if (lat2 < lat1):
        lat = np.arange(lat2, lat1, int_param['res'])
    else:
        lat = np.arange(lat1, lat2, int_param['res'])
    XI, YI = np.meshgrid(lon, lat)

    # Create a xarray DataSet of the regular grid
    rect_grid = xr.Dataset({'lat': (['y', 'x'], YI),
                            'lon': (['y', 'x'], XI), })
    return rect_grid, lon, lat


def create_empty_ds(data, int_param, lon, lat, t):
    '''Create an empty dataset with the coordinates required for the
    interpolated data. Will return a dataset with only one depth dimension
    (`z_c` is favoured if there are `z_c` and `z_l` present). If you want to
    interpolate variables on different depth dimensions, you need to do that
    seperately.

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset containing the variables to interpolate. The dataset
        needs to be compatible with xgcm.
    int_param : dict
        Dictionary specifying all the parameters needed for the interpolation.
        The parameters are
        int_param = {
            'model': 'MITgcm' # MITgcm or ORCA
            'grid': 'cartesian' # cartesian or latlon
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': '360_day', # calendar, must be either 360_day or
                                   # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, (-180, 180)
                          # degrees or m
            'lat1': -55, # minimum latitude of detection region, (-90, 90)
                         # degrees or m
            'lat2': -30, # maximum latitude of detection region, (-90, 90)
                         # degrees or m
            'res': 1./10., # resolution of the regular grid to interpolate to
                           # only used when 'grid' == 'latlon'
            'vars_to_interpolate': ['var1', 'var2', ...],
            'mask_to_interpolate': ['mask1', 'mask2', ...]
            }


    Returns
    -------
    data_int : xarray.DataSet
        xarray dataset containing the coordinates required for the
        interpolated data.
    '''
    if int_param['model'] == 'ORCA':
        if 'z_c' in data.dims:
            # Detect whether the data has a depth dimension and construct the
            # dataset accordingly
            z_dim = data['z_c']
            data_int = xr.Dataset({'time': ('time', data[t].sel({t:
                                                slice(int_param['start_time'],
                                                int_param['end_time'])}).data),
                                   'z': ('z', z_dim.data),
                                   'lat': ('lat', lat.data),
                                   'lon': ('lon', lon.data), })
            data_int = data_int.set_coords(['time', 'z', 'lat', 'lon'])
        elif 'z_l' in data.dims:
            z_dim = data['z_l']
            data_int = xr.Dataset({'time': ('time', data[t].sel({t:
                                                slice(int_param['start_time'],
                                                int_param['end_time'])}).data),
                                   'z': ('z', z_dim.data),
                                   'lat': ('lat', lat.data),
                                   'lon': ('lon', lon.data), })
            data_int = data_int.set_coords(['time', 'z', 'lat', 'lon'])
        else:
            data_int = xr.Dataset({'time': ('time', data[t].sel({t:
                                                slice(int_param['start_time'],
                                                int_param['end_time'])}).data),
                                   'lat': ('lat', lat.data),
                                   'lon': ('lon', lon.data), })
            data_int = data_int.set_coords(['time', 'lat', 'lon'])
    # The MITgcm output always has a depth dimension, thus no checks are
    # necessary
    elif int_param['model'] == 'MITgcm':
        data_int = xr.Dataset({'time': ('time', data[t].sel({t:
                                                slice(int_param['start_time'],
                                                int_param['end_time'])}).data),
                               'z': ('z', data['Z'].data),
                               'lat': ('lat', lat.data),
                               'lon': ('lon', lon.data), })
    return data_int


def rename_dims(var_to_int):
    '''Rename dimensions of `var_to_int` so they are compatible with the
    grid that we want to interpolate to.

    Parameters
    ----------
    var_to_int : xarray.DataArray
        Data array with the variable that is later to be interpolated.

    Returns
    -------
    var_to_int_out : xarray.DataArray
        Same as input `var_to_int` but with renamed dimensions.
    '''
    # Define how the variables are called that should be renamed
    if 'llon_cc' in var_to_int.coords:
        lon_rename = 'llon_cc'
        lat_rename = 'llat_cc'
    elif 'llon_cr' in var_to_int.coords:
        lon_rename = 'llon_cr'
        lat_rename = 'llat_cr'
    elif 'llon_rc' in var_to_int.coords:
        lon_rename = 'llon_rc'
        lat_rename = 'llat_rc'
    elif 'llon_rr' in var_to_int.coords:
        lon_rename = 'llon_rr'
        lat_rename = 'llat_rr'
    elif 'XC' in var_to_int.coords and 'YC' in var_to_int.coords:
        lon_rename = 'XC'
        lat_rename = 'YC'
    elif 'XC' in var_to_int.coords and 'YG' in var_to_int.coords:
        lon_rename = 'XC'
        lat_rename = 'YG'
    elif 'XG' in var_to_int.coords and 'YC' in var_to_int.coords:
        lon_rename = 'XG'
        lat_rename = 'YC'
    elif 'XG' in var_to_int.coords and 'YG' in var_to_int.coords:
        lon_rename = 'XG'
        lat_rename = 'YG'
    else:
        raise ValueError('No valid coordinates have been found. Data must be'
                         + 'compatible with xgcm!')
    # Define how the depth dimension is called in the original dataset
    # to rename it later
    if 'z_c' in var_to_int.coords:
        z_rename = 'z_c'
    elif 'z_l' in var_to_int.coords:
        z_rename = 'z_l'
    elif 'Z' in var_to_int.coords:
        z_rename = 'Z'
    elif 'Zl' in var_to_int.coords:
        z_rename = 'Zl'
    # Rename dimensions to `lon`, `lat` and `z` to be compatible with
    # `data_int`
    if ('z_c' in var_to_int.dims or 'z_l' in var_to_int.dims
        or 'Z' in var_to_int.dims or 'Zl' in var_to_int.dims):
        var_to_int_out = var_to_int.rename({lon_rename: 'lon',
                                            lat_rename: 'lat',
                                            z_rename: 'z'})
    else:
        var_to_int_out = var_to_int.rename({lon_rename: 'lon',
                                            lat_rename: 'lat'})
    if 't' in var_to_int.dims:
        var_to_int_out = var_to_int_out.rename({'t': 'time'})
    return var_to_int_out


def monotonic_lon(var_to_int):
    '''Make sure longitude is monotonically increasing in `var_to_int`.

    Parameters
    ----------
    var_to_int : xarray.DataArray
        Data array with the variable that is later to be interpolated.

    Returns
    -------
    var_to_int_out : xarray.DataArray
        Same as input `var_to_int` but with montonic longitude.
    '''
    var_to_int_out = var_to_int
    # Make sure the longitude is monotonically increasing for the interpolation
    if var_to_int['lon'][0, 0] > var_to_int['lon'][0, -1]:
        lon_mod = var_to_int['lon']\
            .where(var_to_int['lon']
                   > np.around(var_to_int['lon'][0, 0].values),
                   other=var_to_int['lon'] + 360)
        var_to_int_out = var_to_int_out.assign_coords({'lon': lon_mod})
    return var_to_int_out


def update_data(data_int, var_int, var):
    '''Add the interpolated variable `var_int` to the dataset `data_int`.

    Parameters
    ----------
    data_int : xarray.DataSet
        Dataset of the interpolated variables.
    var_int : xarray.DataArray
        Data array of the interpolated variable.

    Returns
    -------
    data_int_out : xarray.DataSet
        Same as input `data_int` but with `var_int` added.
    '''
    if 'time' in var_int.dims:
        if 'z' in var_int.dims:
            data_int_out = data_int.update({var: (['time', 'z', 'lat', 'lon'],
                                                  var_int)})
        else:
            data_int_out = data_int.update({var: (['time', 'lat', 'lon'],
                                                  var_int)})
    else:
        if 'z' in var_int.dims:
            data_int_out = data_int.update({var: (['z', 'lat', 'lon'],
                                                  var_int)})
        else:
            data_int_out = data_int.update({var: (['lat', 'lon'],
                                                  var_int)})
    return data_int_out
