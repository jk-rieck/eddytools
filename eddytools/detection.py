'''detection

Collection of functions needed for the detection of mesoscale eddies
based on the Okubo-Weiss parameter. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.

'''

import numpy as np
import pandas as pd
from scipy import ndimage


def maskandcut(data, var, det_param):
    ''' Mask regions in the dataset where the ocean is shallower than a
    depth threshold and only select specified horizontal and temporal
    extent.

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset that contains `var` to be masked and cut out.
        The dataset must contain dimensions `time`, `lat` and `lon`, as well
        as the variable `bathymetry`.
    var : str
        Name of the variable to be masked and cut out.
    det_param : dict
        Dictionary of the parameters needed for the detection (is to be
        forwarded from `detect.py`).
        The parameters are:
        det_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'min_dep': 1000, # minimum ocean depth where to look for eddies
            'res': 1./10., # resolution of the fields
            'OW_thr': -0.0001, # Okubo-Weiss threshold for eddy detection,
                               # either scalar or 2D field of the same size
                               # as data
            'OW_thr_name': 'OW_std', # name of Okubo-Weiss parameter threshold
            'OW_thr_factor': -0.3, # Okubo-Weiss parameter factor
            'Npix_min': 15, # min. num. grid cells to be considered as eddy
            'Npix_max': 1000 # max. num. grid cells to be considered as eddy
            }

    Returns
    -------
    data_masked : xarray.DataArray
        xarray data array that contains the masked, and cut out variable
        `var`.
    '''
    # Define name of bathymetry depending on model
    if det_param['model'] == 'MITgcm':
        bathy = 'Depth'
    elif det_param['model'] == 'ORCA':
        bathy = 'bathymetry'
    # Mask the areas where the ocean is shallower than min_dep and cut out
    # longitude, latitude (and time if time is a dimension of the variable)
    if 'time' in data[var].dims:
        data_masked = data[var].where(
            data[bathy] >= det_param['min_dep']).sel(
                lat=slice(det_param['lat1'], det_param['lat2']),
                lon=slice(det_param['lon1'], det_param['lon2']),
                time=slice(det_param['start_time'], det_param['end_time']))
    else:
        data_masked = data[var].where(
            data[bathy] >= det_param['min_dep']).sel(
                lat=slice(det_param['lat1'], det_param['lat2']),
                lon=slice(det_param['lon1'], det_param['lon2']))
    return data_masked


def monotonic_lon(var, det_param):
    '''Make sure longitude is monotonically increasing in `var`.

    Parameters
    ----------
    var : xarray.DataArray
        Data array with the variable that is later to be interpolated.
    det_param : dict
        Dictionary of the parameters needed for the detection (is to be
        forwarded from `detect.py`).
        The parameters are:
        det_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'min_dep': 1000, # minimum ocean depth where to look for eddies
            'res': 1./10., # resolution of the fields
            'OW_thr': -0.0001, # Okubo-Weiss threshold for eddy detection,
                               # either scalar or 2D field of the same size
                               # as data
            'OW_thr_name': 'OW_std', # name of Okubo-Weiss parameter threshold
            'OW_thr_factor': -0.3, # Okubo-Weiss parameter factor
            'Npix_min': 15, # min. num. grid cells to be considered as eddy
            'Npix_max': 1000 # max. num. grid cells to be considered as eddy
            }

    Returns
    -------
    var : xarray.DataArray
        Same as input `var` but with montonic longitude.
    det_param: dict
        Same as input `det_param` but with modified values for lon1 and lon2
    '''
    # Make sure the longitude is monotonically increasing for the interpolation
    if var['lon'][0] > var['lon'][-1]:
        lon_mod = var['lon']\
            .where(var['lon']
                   >= np.around(var['lon'][0].values),
                   other=var['lon'] + 360)
        var = var.assign_coords({'lon': lon_mod})
        if det_param['lon1'] < lon_mod[0]:
            det_param['lon1'] = det_param['lon1'] + 360
        if det_param['lon2'] < lon_mod['lon'][0]:
            det_param['lon2'] = det_param['lon2'] + 360
    return var, det_param


def restore_lon(data):
    '''Make sure longitude is between -180 and 180 in `data`.

    Parameters
    ----------
    data : xarray.DataArray
        Data array with the variable that is later to be interpolated.

    Returns
    -------
    data : xarray.DataArray
        Same as input `data` but with modified longitude.
    '''
    lon_mod = data['lon'].where(data['lon'].values < 180.,
                                other=data['lon'].values - 360)
    data = data.assign_coords({'lon': lon_mod})
    return data


def get_indeces(data):
    '''Retrieve indeces of regions detected with ndimage.label

    Parameters
    ----------
    data : ndimage.label
        Return of ndimage.label()

    Returns
    -------
        Indeces of regions
    '''
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)


def detect_core(data, det_param, OW, vort, t, OW_thr, e1f, e2f):
    ''' Core function for the detection of eddies, used by detect().

    Parameters
    ----------
    det_param : dict
        Dictionary of the parameters needed for the detection.
        The parameters are:
        det_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'min_dep': 1000, # minimum ocean depth where to look for eddies
            'res': 1./10., # resolution of the fields
            'OW_thr': -0.0001, # Okubo-Weiss threshold for eddy detection,
                               # either scalar or 2D field of the same size
                               # as data
            'OW_thr_name': 'OW_std', # name of Okubo-Weiss parameter threshold
            'OW_thr_factor': -0.3, # Okubo-Weiss parameter factor
            'Npix_min': 15, # min. num. grid cells to be considered as eddy
            'Npix_max': 1000 # max. num. grid cells to be considered as eddy
            }
    OW : xarray.DataArray
        Xarray DataArray of the Okubo-Weiss parameter.
    vort : xarray.DataArray
        Xarray DataArray of the vorticty.
    t : int
        Time step at which to look for eddies.
    OW_thr : xarray.DataArray or float
        Threshold for the Okubo-Weiss parameter. Can be either a Xarray
        DataArray of two dimensions or a scalar float (to use the same
        threshold everywhere.)
    e1f : xarray.DataArray
        Xarray DataArray with the grid cell size in x direction, used to
        calculate the area of the eddies.
    e2f : xarray.DataArray
        Xarray DataArray with the grid cell size in y direction, used to
        calculate the area of the eddies.

    Returns
    -------
    eddies : dict
        Dictionary containing information on all detected eddies at time step
        t.
        The dict has the form:
        eddies = {e: {'time': array, # time stamp
                      'type': str, # 'cyclonic' or 'anticyclonic'
                      'lon': array, # central longitude
                      'lat': array, # central latitude
                      'scale': array, # diameter of the eddy
                      'area': array, # area of the eddy
                      'vort_extr': array, # vorticity max/min
                      'amp': array, # vorticity amplitude
                      'eddy_i': array, # i-indeces of the eddy
                      'eddy_j': array # j-indeces of the eddy
                      }}}
        where `e` is the eddy number.
    '''
    # Construct dictionary for each time step
    # (Note: This nested-dictionaries approach seems to be rather slow
    # and needs further improvement!)
    eddi = {}
    # Find all regions with OW parameter exceeding threshold (=Eddy)
    regions, nregions = ndimage.label((OW.isel(time=t).values
                                       < OW_thr).astype(int))
    region_index = get_indeces(regions)
    #
    len_OW_lat = len(OW['lat'])
    len_OW_lon = len(OW['lon'])
    # length of 1 degree of latitude [km]
    e = 0
    for iregion in list(range(nregions - 1)):
        index = region_index[iregion + 1]
        # Loop through all regions detected as eddy at each time step
        eddi[e] = {}
        # Calculate number of pixels comprising detected region, reject if
        # not within [Npix_min, Npix_max]
        region_Npix = len(index[0])
        eddy_area_within_limits = ((region_Npix < det_param['Npix_max'])
                                   * (region_Npix > det_param['Npix_min']))
        if eddy_area_within_limits:
            # If the region is not too small and not too big, extract
            # eddy information
            eddi[e]['time'] = OW.isel(time=t)['time'].values
            # get eddy type from Vorticity and store extrema
            # Note this is written for the Southern Hemisphere
            # Need to generalize it!
            if vort.isel(time=t).values[index].mean() < 0:
                eddi[e]['type'] = 'cyclonic'
                eddi[e]['vort_extr'] = np.array(
                    [vort.isel(time=t).values[index].max()])
            elif vort.isel(time=t).values[index].mean() > 0:
                eddi[e]['type'] = 'anticyclonic'
                eddi[e]['vort_extr'] = np.array(
                    [vort.isel(time=t).values[index].min()])
            # calc vorticity amplitude
            eddi[e]['amp'] = np.array(
                [vort.isel(time=t).values[index].max()
                 - vort.isel(time=t).values[index].min()])
            # find centre of mass of eddy
            iimin = index[1].min()
            iimax = index[1].max() + 1
            ijmin = index[0].min()
            ijmax = index[0].max() + 1
            index_eddy = (index[0] - ijmin, index[1] - iimin)
            tmp = OW.isel(time=t, lat=slice(ijmin, ijmax),
                          lon=slice(iimin, iimax)).values.copy()
            eddy_object_with_mass = np.zeros_like(tmp)
            eddy_object_with_mass[index_eddy] = tmp[index_eddy]
            j_cen, i_cen = ndimage.center_of_mass(eddy_object_with_mass)
            j_cen, i_cen = j_cen + ijmin, i_cen + iimin
            lon_eddies = np.interp(i_cen, range(0, len(OW['lon'])),
                                   OW['lon'].values)
            lat_eddies = np.interp(j_cen, range(0, len(OW['lat'])),
                                   OW['lat'].values)
            if lon_eddies > 180:
                eddi[e]['lon'] = np.array([lon_eddies]) - 360.
            elif lon_eddies < -180:
                eddi[e]['lon'] = np.array([lon_eddies]) + 360.
            else:
                eddi[e]['lon'] = np.array([lon_eddies])
            eddi[e]['lat'] = np.array([lat_eddies])

            # store all eddy indices
            j_min = (data.lat.where(data.lat == OW.lat.min(), other=0)
                     ** 2).argmax().values
            i_min = (data.lon.where(data.lon == OW.lon.min(), other=0)
                     ** 2).argmax().values
            eddi[e]['eddy_j'] = index[0] + j_min
            eddi[e]['eddy_i'] = index[1] + i_min
            # assign (and calculated) area, and scale of eddies
            eddi[e]['area'] = np.array([((e1f.values[index] / 1000.)
                                         * (e2f.values[index] / 1000.)).sum()])
            eddi[e]['scale'] = np.array([np.sqrt(eddi[e]['area']
                                         / np.pi)])  # [km]
            e = e+1
    return eddi


def detect(data, det_param, ow_var, vort_var):
    ''' Detect eddies based on specified Okubo-Weiss parameter.

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset with the variables and coordinates needed for the
        detection of eddies. Need variables `ow_var` (Okubo-Weiss parameter)
        and `vort_var` (vorticity), as well as coordinates `time`, `lat` and
        `lon`.
    det_param : dict
        Dictionary of the parameters needed for the detection.
        The parameters are:
        det_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'min_dep': 1000, # minimum ocean depth where to look for eddies
            'res': 1./10., # resolution of the fields
            'OW_thr': -0.0001, # Okubo-Weiss threshold for eddy detection,
                               # either scalar or 2D field of the same size
                               # as data
            'OW_thr_name': 'OW_std', # name of Okubo-Weiss parameter threshold
            'OW_thr_factor': -0.3, # Okubo-Weiss parameter factor
            'Npix_min': 15, # min. num. grid cells to be considered as eddy
            'Npix_max': 1000 # max. num. grid cells to be considered as eddy
            }
    ow_var : str
        Name of the variable in `data` containing the Okubo-Weiss parameter.
    vort_var : str
        Name of the variable in `data` containing the vorticity field.

    Returns
    -------
    eddies : dict
        Dictionary containing information on all detected eddies.
        The dict has the form:
        eddies = {t: {e: {'time': array, # time stamp
                          'type': str, # 'cyclonic' or 'anticyclonic'
                          'lon': array, # central longitude
                          'lat': array, # central latitude
                          'scale': array, # diameter of the eddy
                          'area': array, # area of the eddy
                          'vort_extr': array, # vorticity max/min
                          'amp': array, # vorticity amplitude
                          'eddy_i': array, # i-indeces of the eddy
                          'eddy_j': array # j-indeces of the eddy
                          }}}
        where `t` is the time step and `e` is the eddy number.
    '''
    # Verify that the specified region lies within the dataset provided
    if (det_param['lon1'] < np.around(data['lon'].min())
       or det_param['lon2'] > np.around(data['lon'].max())):
        raise ValueError('`det_param`: min. and/or max. of longitude range'
                         + ' are outside the region contained in the dataset')
    if (det_param['lat1'] < np.around(data['lat'].min())
       or det_param['lat2'] > np.around(data['lat'].max())):
        raise ValueError('`det_param`: min. and/or max. of latitude range'
                         + ' are outside the region contained in the dataset')
    #
    print('preparing data for eddy detection'
          + ' (masking and region extracting etc.)')
    # Make sure longitude vector is monotonically increasing if we have a
    # latlon grid
    if det_param['grid'] == 'latlon':
        data, det_param = monotonic_lon(data, det_param)
    # Masking shallow regions and cut out the region specified in `det_param`
    OW = maskandcut(data, ow_var, det_param).compute()
    vort = maskandcut(data, vort_var, det_param).compute()
    OW_thr_name = det_param['OW_thr_name']
    # Define the names of the grid cell sizes depending on the model
    if det_param['model'] == 'MITgcm':
        e1f_name = 'dxV'
        e2f_name = 'dyU'
    if det_param['model'] == 'ORCA':
        e1f_name = 'e1f'
        e2f_name = 'e2f'
    e1f = maskandcut(data, e1f_name, det_param).compute()
    e2f = maskandcut(data, e2f_name, det_param).compute()
    if len(np.shape(data[OW_thr_name])) > 1:
        # If the Okubo-Weiss threshold is 2D, use `maskandcutOW` masking etc.
        OW_thr = maskandcut(data, OW_thr_name, det_param).compute()
        OW_thr = OW_thr * (det_param['OW_thr_factor'])
    else:
        # Else just use scalar from `det_param`
        OW_thr = det_param['OW_thr'] * (det_param['OW_thr_factor'])
    # Construct dict to write information on eddies to
    eddies = {}
    for tt in np.arange(0, len(OW['time'])):
        steps = np.around(np.linspace(0, len(OW['time']), 10))
        if tt in steps:
            print('detection at time step ', str(tt + 1), ' of ',
                  len(OW['time']))
        eddies[tt] = detect_core(data, det_param.copy(),
                                 OW, vort, tt, OW_thr, e1f, e2f)
    return eddies
