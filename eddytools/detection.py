'''detection

Collection of functions needed for the detection of mesoscale eddies
based on the Okubo-Weiss parameter. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.

'''

import numpy as np
import pandas as pd
from scipy import ndimage
from dask import bag as dask_bag


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


def distance_matrix(lons, lats):
    '''Calculates the distances (in km) between any pairs of points based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html

    Usage:
        d = distance_matrix(lons,lats)
    Input:
        lons: vector of longitudes
        lats: vector of latitudes
        (lons, lats need to have same number of elements)
    Output:
        d: Matrix with the distance from Point (lons[x],lats[x]) to every other
           Point (lons[y],lats[y]).
    '''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros((X,X))

    #Populate the matrix.
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[c<1,i2] = EARTH_RADIUS * np.arccos(c[c<1])
    return d


def detect_OW_core(data, det_param, OW, vort, t, OW_thr, e1f, e2f):
    ''' Core function for the detection of eddies, used by detect_OW().

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
        min_width = int(np.floor(np.sqrt(region_Npix) / 2))
        X_cen = int(np.floor(np.mean(index[1])))
        Y_cen = int(np.floor(np.mean(index[0])))
        Ypix_cen = np.sum(index[1] == X_cen)
        Xpix_cen = np.sum(index[0] == Y_cen)
        eddy_not_too_thin = ((Xpix_cen > min_width) & (Ypix_cen > min_width))
        if (eddy_area_within_limits & eddy_not_too_thin):
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
            e += 1
    return eddi


def detect_SSH_core(data, det_param, SSH, t, ssh_crits, e1f, e2f):
    '''
    Detect eddies present in field which satisfy the 5 criteria
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2.:
    1) SSH value above (below for cyclonic) given threshold
    2) The number of Pixels corresponding to an eddy is within a given intervall
    3) There is at least one local SSH maximum (minimum for cyclonic)
    4) The amplitude exceeds a given threshold
    5) The maximum distance between two points of an eddy is below a given value

    Npix_min, Npix_max, amp_thresh, d_thresh specify the constants
    used by the eddy detection algorithm (see Chelton paper for
    more details)

    Input:
        data: 2D filtered SSH field


    Output:
        eddies: dictionary containing the following variables
            lon_cen: longitude coordinate of detected eddy centers
            lat_cen: latitude coordinate of detected eddy centers
            amp: amplitude of detected eddies (see Chelton et al. 2011)
            area: area of the detected eddies in km*2
            scale: scale of the detected eddies (see Chelton et al. 2011)
    '''
    #set up grid
    field = SSH.isel(time=t).values
    len_deg_lat = 111.325 # length of 1 degree of latitude [km]
    llon, llat = np.meshgrid(SSH.lon, SSH.lat)
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    for cyc in ['cyclonic','anticyclonic']:
        # ssh_crits increasing for 'anticyclonic', decreasing for 'cyclonic'
        # flip to start with largest positive value for 'cylonic'
        crit_len = int(len(ssh_crits) / 2)
        if cyc == 'cyclonic':
            ssh_crits = np.flipud(ssh_crits)[crit_len:]
        if cyc == 'anticyclonic':
            ssh_crits = ssh_crits[crit_len:]
        # loop over ssh_crits and remove interior pixels of detected eddies
        # from subsequent loop steps
        for ssh_crit in ssh_crits:
        # 1. Find all regions with eta greater (less than) than ssh_crit for
        # anticyclonic (cyclonic) eddies (Chelton et al. 2011, App. B.2,
        # criterion 1)
            if cyc == 'anticyclonic':
                regions, nregions = ndimage.label(
                    (field > ssh_crit).astype(int))
            elif cyc == 'cyclonic':
                regions, nregions = ndimage.label(
                    (field < ssh_crit).astype(int))

            for iregion in list(range(nregions)):
                eddi[e] = {}
        # 2. Calculate number of pixels comprising detected region, reject if
        # not within [Npix_min, Npix_max]
                region = (regions==iregion+1).astype(int)
                region_Npix = region.sum()
                eddy_area_within_limits = (
                    (region_Npix < det_param['Npix_max'])
                    * (region_Npix > det_param['Npix_min']))
        # 3. Detect presence of local maximum (minimum) for anticylonic
        # (cyclonic) eddies, reject if non-existent
                interior = ndimage.binary_erosion(region)
                exterior = region.astype(bool) ^ interior
                if interior.sum() == 0:
                    continue
                if cyc == 'anticyclonic':
                    has_internal_ext = field[interior].max() > field[exterior].max()
                elif cyc == 'cyclonic':
                    has_internal_ext = field[interior].min() < field[exterior].min()
        # 4. Find amplitude of region, reject if < amp_thresh
                if cyc == 'anticyclonic':
                    amp = field[interior].max() - field[exterior].mean()
                elif cyc == 'cyclonic':
                    amp = field[exterior].mean() - field[interior].min()
                is_tall_eddy = amp >= amp_thresh
        # 5. Find maximum linear dimension of region, reject if < d_thresh
                if np.logical_not( eddy_area_within_limits
                                   * has_internal_ext * is_tall_eddy):
                    continue
                lon_ext = llon[exterior]
                lat_ext = llat[exterior]
                d = distance_matrix(lon_ext, lat_ext)
                is_small_eddy = d.max() < d_thresh
        # Detected eddies:
                if (eddy_area_within_limits * has_internal_ext
                    * is_tall_eddy * is_small_eddy):
                    # find centre of mass of eddy
                    eddy_object_with_mass = field * region
                    eddy_object_with_mass[np.isnan(eddy_object_with_mass)] = 0
                    j_cen, i_cen = ndimage.center_of_mass(eddy_object_with_mass)
                    lon_cen = np.interp(i_cen, range(0,len(lon)), lon)
                    lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
                    # store all eddy grid-points
                    index = get_indeces(region)
                    eddi[e]['eddy_j'] = index[0]
                    eddi[e]['eddy_i'] = index[1]
                    # assign (and calculated) amplitude, area, and scale of
                    # eddies
                    len_deg_lon = ((np.pi/180.) * 6371
                                   * np.cos( lat_cen * np.pi/180. )) #[km]
                    area = region_Npix * res**2 * len_deg_lat * len_deg_lon
                    # [km**2]
                    scale = np.sqrt(area / np.pi) # [km]
                    # remove its interior pixels from further eddy detection
                    eddy_mask = np.ones(field.shape)
                    eddy_mask[interior.astype(int)==1] = np.nan
                    field = field * eddy_mask
                    eddi[e]['time'] = data.time
                    eddi[e]['lon_cen'] = lon_cen
                    eddi[e]['lat_cen'] = lat_cen
                    eddi[e]['amp'] = amp
                    eddi[e]['area'] = area
                    eddi[e]['scale'] = scale
                    eddi[e]['type'] = cyc
                    e += 1
    return eddi


def detect_OW(data, det_param, ow_var, vort_var):
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
    ## set range of parallel executions
    pexps = range(0, len(OW['time']))
    ## generate dask bag instance
    seeds_bag = dask_bag.from_sequence(pexps)
    detection = dask_bag.map(
        lambda tt: detect_OW_core(data, det_param.copy(),
                                  OW, vort, tt, OW_thr, e1f, e2f)
        ,seeds_bag)
    eddies = detection.compute()
    return eddies


def detect_SSH(data, det_param, ssh_var):
    ''' Detect eddies based on SSH following Chelton 2011. Prepares the
    necessary input for detect_SSH_core that performs the actual detection.
    Parallel computation of timesteps using dask bag.

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset with the variables and coordinates needed for the
        detection of eddies. Need variables `ssh_var` (Sea Surface Height), as well as coordinates `time`, `lat` and `lon`.
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
            'ssh_thr': 1, # threshold for min SSH anomaly, any greater anomaly
                          # is considered for eddy detection
            'dssh': 0.02, # increment for SSH threshold
            'amp_thr': 0.05, # threshold for min amplitude of eddy
            'd_thr': 300, # threshold for max distance between any two points
                        # inside an eddy
            'Npix_min': 15, # min. num. grid cells to be considered as eddy
            'Npix_max': 1000 # max. num. grid cells to be considered as eddy
            }
    ssh_var : str
        Name of the variable in `data` containing the SSH.

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
    SSH = maskandcut(data, ssh_var, det_param).compute()
    # Define the names of the grid cell sizes depending on the model
    if det_param['model'] == 'MITgcm':
        e1f_name = 'dxV'
        e2f_name = 'dyU'
    if det_param['model'] == 'ORCA':
        e1f_name = 'e1f'
        e2f_name = 'e2f'
    e1f = maskandcut(data, e1f_name, det_param).compute()
    e2f = maskandcut(data, e2f_name, det_param).compute()
    ## create list of incremental threshold
    ssh_crits = np.arange(-det_param['ssh_thr'],
                          det_param['ssh_thr'] + det_param['dssh'],
                          det_param['dssh'])
    ssh_crits = np.sort(ssh_crits) # make sure its increasing order

    ## set range of parallel executions
    pexps = range(0, len(SSH['time']))

    ## generate dask bag instance
    seeds_bag = dask_bag.from_sequence(pexps)

    detection = dask_bag.map(
        lambda tt: detect_SSH_core(data, det_param.copy(), SSH, tt, ssh_crits,
                                   e1f, e2f)
        ,seeds_bag)

    eddies = detection.compute()
    return eddies
