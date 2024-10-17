'''detection
Collection of functions needed for the detection of mesoscale eddies
based on the Okubo-Weiss parameter. The data is assumed to have been
interpolated with the `interp.py` module of this package or at least
needs to have the same structure.
'''

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
import cftime as cft
import itertools
from scipy import interpolate
from contourpy import contour_generator
from scipy.spatial import ConvexHull
try:
    import multiprocessing as mp
except:
    print("multiprocessing not possible")
try:
    from dask import bag as dask_bag
except:
    print("Working without dask bags.")

def maskandcut(data, var, det_param, regrid_avoided=False):
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
            'Npix_max': 1000, # max. num. grid cells to be considered as eddy
            'no_long': False, # If True, elongated shapes will not be considered
            'no_two': False # If True, eddies with two minima in the OW
                            # parameter and a OW > OW_thr in between  will not
                            # be considered
            }
    regrid_avoided : bool
        If True indicates that regridding has been avoided during the
        interpolation and the data has 2D coordinates. Default is False.

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
        if regrid_avoided:
            data_masked = data[var].where(
                data[bathy] >= det_param['min_dep']).sel(
                    time=slice(det_param['start_time'], det_param['end_time']))
        else:
            data_masked = data[var].where(
                data[bathy] >= det_param['min_dep']).sel(
                    lat=slice(det_param['lat1'], det_param['lat2']),
                    lon=slice(det_param['lon1'], det_param['lon2']),
                    time=slice(det_param['start_time'], det_param['end_time']))
    else:
        if regrid_avoided:
            data_masked = data[var].where(
                data[bathy] >= det_param['min_dep'])
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
            'Npix_max': 1000, # max. num. grid cells to be considered as eddy
            'no_long': False, # If True, elongated shapes will not be considered
            'no_two': False # If True, eddies with two minima in the OW
                            # parameter and a OW > OW_thr in between  will not
                            # be considered
            }
    Returns
    -------
    var : xarray.DataArray
        Same as input `var` but with montonic longitude.
    det_param: dict
        Same as input `det_param` but with modified values for lon1 and lon2
    '''
    # Make sure the longitude is monotonically increasing for the interpolation
    if (var['lon'][0] > var['lon'][-1]).any():
        lon_mod = var['lon']\
            .where(var['lon']
                   >= var['lon'][0].max().values,
                   other=var['lon'] + 360)
        var = var.assign_coords({'lon': lon_mod})
        if (det_param['lon1'] < lon_mod[0]).any():
            det_param['lon1'] = det_param['lon1'] + 360
        if (det_param['lon2'] < lon_mod['lon'][0]).any():
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

def get_width(data, thr):
    '''Calculate the width of an eddy in grid cell space
    Parameters
    ----------
    data : array
        1D array with values of the OW parameter across the detected feature.
    thr : float
        The OW-threshold below which an area is considered and eddy.
    Returns
    -------
    width : int
        Width of the eddy as measured in number of grid cells.
    '''
    start = np.argmax(data < thr)
    end = np.argmax(data[start::] >= thr) + 1
    return np.sum(data[start:start+end] <= thr)


def compute_psi(u_psi, v_psi, dx, dy):
    # compute PSI to get eddy area
    # Indices for domain size
    lx = np.shape(u_psi)[0]
    ly = np.shape(u_psi)[1]
    # itegrate first row of v along longitude (first term of eq.A2)
    cx = np.cumsum(v_psi[0, :]) * dx[0, :]
    # integrate first column of u along latitude (second term of eq.A3)
    cy = np.cumsum(u_psi[:, 0]) * dy[:, 0]
    # compute streamfunction
    # PSI from integrating v firts and then u
    psi_xy = (-cx[None, :] + np.cumsum(u_psi, axis=0) * dy)
    # PSI from integrating u first and then v
    psi_yx = (-np.cumsum(v_psi, axis=1) * dx + cy[:, None])
    # final PSI as average between the two
    return (psi_xy + psi_yx) / 2


def inpolygon(point_i, point_j, x, y):
    p = np.array([x, y]).T
    hullp = ConvexHull(p)
    poly = plt.Polygon(p[hullp.vertices, :])
    return poly.contains_point(np.array([point_i, point_j]))


def inpolygon2D(points_i, points_j, x, y):
    p = np.array([x, y]).T
    hullp = ConvexHull(p)
    poly = plt.Polygon(p[hullp.vertices, :])
    return poly.contains_points(np.array([points_i, points_j]).T)


def interpolate_limits(pt_i, pt_j, lon, lat, dkm, vel, direct):
    # include only 'n' points around that point to interpolate vel
    # (griddata is a very slow process) --------------------------------
    # find the closest grid point to the curve extreme
    dist = np.sqrt((lon - pt_i)**2 + (lat - pt_j)**2)
    d_j, d_i = np.where(dist == np.min(dist))
    n = 4 # at least 4 points away to avoid a qhull precision warning!!!
    # resize coordinate and velocity matrices
    svel = vel[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
               int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    skm_i = lon[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
                int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    skm_j = lat[int(np.max([d_j-n, 1])):int(np.min([d_j+n, np.shape(vel)[0]])),
                int(np.max([d_i-n, 1])):int(np.min([d_i+n, np.shape(vel)[1]]))]
    # interpolate vel across the curve along different directions depending on
    # interpolate vel across the curve along different directions depending on
    # different extremes
    if direct == "N":
        pts = np.array([[pt_i, float(pt_j-dkm)], [pt_i, float(pt_j+dkm)]])
    elif direct == "S":
        pts = np.array([[pt_i, float(pt_j+dkm)], [pt_i, float(pt_j-dkm)]])
    elif direct == "W":
        pts = np.array([[float(pt_i+dkm), pt_j], [float(pt_i-dkm), pt_j]])
    elif direct == "E":
        pts = np.array([[float(pt_i-dkm), pt_j], [float(pt_i+dkm), pt_j]])
    # interpolate vel
    in_out_vel = interpolate.griddata(np.array([skm_i.flatten(), skm_j.flatten()]).T,
                                      svel.flatten(), pts)
    return in_out_vel


def get_eddy_indeces(lon, lat, ec_lon, ec_lat, psi, vel):
    # define the distance from the contour extremes at which velocity magnitude
    # is interpolated
    (c_j, c_i) = np.where((lat==ec_lat) & (lon==ec_lon))
    dkm = 0.05 * (lon[c_j, 1] - lon[c_j, 0])
    # compute contourlines of the streamfunction field (100 contours)
    cg = contour_generator(lon, lat, psi)
    C = [cg.lines(np.linspace(np.min(psi), np.max(psi), 100)[i]) for i in np.arange(100)]
    # intialize the two variables
    eddy_lim = []
    largest_curve=[]
    # rearrange all the contours in C to the structure array 'isolines'
    # each element of isolines contains all the vertices of a given contour
    # level of PSI
    isolines = {}
    isolines_max = []
    i = 0
    ii = 0
    there_are_contours = True
    while there_are_contours:
        for j in np.arange(len(C[ii])):
            isolines[i] = {}
            isolines[i]["x"] = []
            isolines[i]["y"] = []
            isolines[i]["x"] = list(C[ii][j][:, 0])
            isolines[i]["y"] = list(C[ii][j][:, 1])
            isolines_max.append(np.max(isolines[i]["y"]))
            i += 1
        ii += 1
        try:
            test = C[ii]
            there_are_contours = True
        except:
            there_are_contours = False
    # sort the contours accroding to their maximum latitude; this way the first
    # closed contour across which velocity increases will also be the largest
    # one (it's the one which extend further north).
    sorted_iso = np.argsort(isolines_max)[::-1]
    # restart the counter and initialize the two flags
    i = 0
    closed_indx = 0 # set to 1 when eddy shape is found
    largest_indx = 0 # set to 1 when largest closed contour is found
    # inspect all isolines until the eddy shape is determined
    # (closed_indx=1 stops the loop)
    while ((closed_indx == 0) & (i < len(isolines))):
        ii = sorted_iso[i]
        xdata = isolines[ii]["x"] # vertex lon's
        ydata = isolines[ii]["y"] # vertex lat's
        # conditions to have the largest closed contour around the center
        # (isolines already sorted by maximum latitude)
        # 1) closed contours
        # 2) detected eddy center inside the polygon
        if ((len(xdata) < 3) | (len(ydata) < 3)):
            i += 1
            continue
        try:
            inpo = inpolygon(ec_lon, ec_lat, xdata, ydata)
        except:
            inpo = False
        if (((xdata[0] == xdata[-1]) & (ydata[0] == ydata[-1])) & inpo):
            # find the contour extremes
            Nj = np.max(ydata)
            Ni = np.max(xdata[int(np.where(ydata==Nj)[0][0])])
            Sj = np.min(ydata)
            Si = np.min(xdata[int(np.where(ydata==Sj)[0][0])])
            Ei = np.max(xdata)
            Ej = np.min(ydata[int(np.where(xdata==Ei)[0][0])])
            Wi = np.min(xdata)
            Wj = np.max(ydata[int(np.where(xdata==Wi)[0][0])])
            # check if velocity across the contour increases
            direct = ['N', 'S', 'E', 'W']
            pts_I = [Ni, Si, Ei, Wi]
            pts_J = [Nj, Sj, Ej, Wj]
            # inspect one extreme at the time (faster)
            iii = 0 # counter
            smaller_vel = 0  # flag to stop the loop (1 if velocity decreases
                             # across the fourth extremes)
            smaller_vel1 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the third extremes)
            smaller_vel2 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the second extremes)
            smaller_vel3 = 0 # flag to stop the loop (1 if velocity decreases
                             # across the first extremes)
            while ((iii < len(direct)) & (smaller_vel == 0)):
                # interpolate velocity across the extreme
                in_out_vel = interpolate_limits(pts_I[iii], pts_J[iii],
                                                lon, lat, dkm, vel, direct[iii])
                # change the flag value if velocity decreases
                if (in_out_vel[0] > in_out_vel[1]):
                    if (smaller_vel3 == 0):
                        smaller_vel3 = 1
                    elif (smaller_vel2 == 0):
                        smaller_vel2 = 1
                    elif (smaller_vel1 == 0):
                        smaller_vel1 = 1
                    elif (smaller_vel == 0):
                        smaller_vel = 1
                iii += 1 # increase the counter
            # only if velocity increases across all four extremes the closed
            # contour is saved as eddy shape
            if (smaller_vel == 0):
                eddy_lim = [xdata, ydata]
                closed_indx = 1
            # largest closed conotur is saved as well
            if (largest_indx == 0):
                largest_curve = [xdata, ydata]
                largest_indx = 1
        i += 1 # increase the counter
    # in case velocity doesn't increase across the closed contour, eddy shape
    # is defined simply as the largest closed contour
    if ((eddy_lim==[]) & (largest_curve!=[])):
        eddy_lim = largest_curve
    if eddy_lim==[]:
        return None, None, None
    else:
        mask = inpolygon2D(lon.flatten(), lat.flatten(), eddy_lim[0], eddy_lim[1])
        eddy_mask = mask.reshape(np.shape(lon))
        eddy_j = np.where(eddy_mask)[0]
        eddy_i = np.where(eddy_mask)[1]
        return eddy_i, eddy_j, eddy_mask


def detect_OW_core(data, det_param, OW, vort, t, OW_thr, e1f, e2f,
                   regrid_avoided=False):
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
            'Npix_max': 1000, # max. num. grid cells to be considered as eddy
            'no_long': False, # If True, elongated shapes will not be considered
            'no_two': False # If True, eddies with two minima in the OW
                            # parameter and a OW > OW_thr in between  will not
                            # be considered
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
    regrid_avoided : bool
        If True indicates that regridding has been avoided during the
        interpolation and the data has 2D coordinates. Default is False.

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
                      'scale': array, # ~radius of the eddy
                      'area': array, # area of the eddy
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
    if regrid_avoided:
        len_OW_lat = len(OW['y'])
        len_OW_lon = len(OW['x'])
    else:
        len_OW_lat = len(OW['lat'])
        len_OW_lon = len(OW['lon'])
    # length of 1 degree of latitude [km]
    e = 0
    for iregion in list(range(nregions - 1)):
        index = region_index[iregion + 1]
        iimin = index[1].min()
        iimax = index[1].max() + 1
        ijmin = index[0].min()
        ijmax = index[0].max() + 1
        if ((iimax == 1) | (ijmax == 1)):
            continue
        iimin2 = iimin
        ijmin2 = ijmin
        if iimin == 0:
            iimin2 = 0
        else:
            iimin2 = iimin - 1
        if ijmin == 0:
            ijmin2 = 0
        else:
            ijmin2 = ijmin - 1
        region = (regions == iregion + 1).astype(int)
        # Loop through all regions detected as eddy at each time step
        eddi[e] = {}
        # Calculate number of pixels comprising detected region, reject if
        # not within [Npix_min, Npix_max]
        region_Npix = region.sum()
        eddy_area_within_limits = ((region_Npix < det_param['Npix_max'])
                                   * (region_Npix > det_param['Npix_min']))
        # define interior and exterior
        interior = ndimage.binary_erosion(region)
        exterior = region.astype(bool) ^ interior
        if interior.sum() == 0:
            del eddi[e]
            continue
        if (det_param['no_long'] or det_param['no_two']):
            min_width = int(np.floor(np.sqrt(region_Npix / np.pi)))
            X_cen = int(np.around(np.mean(index[1])))
            Y_cen = int(np.around(np.mean(index[0])))
        if det_param['no_two']:
            if len(np.shape(data[det_param['OW_thr_name']])) > 1:
                peak_thr = OW_thr.values[interior].mean()
            else:
                peak_thr = OW_thr
            X_peak_info = find_peaks(-OW.isel(time=t).values[Y_cen,
                              iimin:iimax], height=-peak_thr)
            Y_peak_info = find_peaks(-OW.isel(time=t).values[ijmin:ijmax,
                              X_cen], height=-peak_thr)
            X_peaks = len(X_peak_info[0])
            Y_peaks = len(Y_peak_info[0])
            if ((X_peaks > 1) | (Y_peaks > 1)):
                Ypix_cen1 = get_width(OW.isel(time=t).values[ijmin:ijmax,
                                X_cen], peak_thr)
                Ypix_cen2 = get_width(OW.isel(time=t).values[ijmax-1:ijmin2:-1,
                                X_cen], peak_thr)
                Xpix_cen1 = get_width(OW.isel(time=t).values[Y_cen,
                                iimin:iimax], peak_thr)
                Xpix_cen2 = get_width(OW.isel(time=t).values[Y_cen,
                                iimax-1:iimin2:-1], peak_thr)
                tmp_no_horseshoe = (((Xpix_cen1 > min_width)
                                   & (Ypix_cen1 > min_width)) |
                                    ((Xpix_cen2 > min_width)
                                   & (Ypix_cen2 > min_width)) |
                                    ((Xpix_cen2 > min_width)
                                   & (Ypix_cen1 > min_width)) |
                                    ((Xpix_cen1 > min_width)
                                   & (Ypix_cen2 > min_width)))
                if (X_peaks > 1):
                    X_peak1 = iimin + X_peak_info[0][0]
                    X_peak2 = iimin + X_peak_info[0][1]
                    Xmin = -np.max(X_peak_info[1]['peak_heights'])
                    Xmax = np.max(OW.isel(time=t).values[Y_cen,
                                                         X_peak1:X_peak2])
                    Xdiff = Xmin - Xmax
                    Xpix_ratio = Xpix_cen1 / Xpix_cen2
                    if ((Xpix_ratio > 2.5) | (Xpix_ratio < 0.4)):
                        Xdiff = 0
                else:
                    Xdiff = 0
                    Xmin = 1
                if (Y_peaks > 1):
                    Y_peak1 = ijmin + Y_peak_info[0][0]
                    Y_peak2 = ijmin + Y_peak_info[0][1]
                    Ymin = -np.max(Y_peak_info[1]['peak_heights'])
                    Ymax = np.max(OW.isel(time=t).values[Y_peak1:Y_peak2,
                                                         X_cen])
                    Ydiff = Ymin - Ymax
                    Ypix_ratio = Ypix_cen1 / Ypix_cen2
                    if ((Ypix_ratio > 2.5) | (Ypix_ratio < 0.4)):
                        Ydiff = 0
                else:
                    Ydiff = 0
                    Ymin = 1
                Xdiff_small = np.abs(Xdiff) < 0.75 * np.abs(Xmin)
                Ydiff_small = np.abs(Ydiff) < 0.75 * np.abs(Ymin)
                eddy_no_horseshoe = tmp_no_horseshoe & Xdiff_small & Ydiff_small
            else:
                eddy_no_horseshoe = True
        else:
            eddy_no_horseshoe = True
        if det_param['no_long']:
            Ypix_cen = np.sum(index[1] == X_cen)
            Xpix_cen = np.sum(index[0] == Y_cen)
            eddy_not_too_thin = ((Xpix_cen > min_width)
                               & (Ypix_cen > min_width))
        else:
            eddy_not_too_thin = True
        # check for local extrema
        has_internal_ext = (OW.isel(time=t).values[interior].min()
                            < OW.isel(time=t).values[exterior].min())
        if (eddy_area_within_limits & eddy_not_too_thin
            & eddy_no_horseshoe & has_internal_ext):
            # If the region is not too small and not too big, extract
            # eddy information
            eddi[e]['time'] = OW.isel(time=t)['time'].values
            # find centre of mass of eddy
            index_eddy = (index[0] - ijmin, index[1] - iimin)
            if regrid_avoided:
                tmp = OW.isel(time=t, y=slice(ijmin, ijmax),
                              x=slice(iimin, iimax)).values.copy()
            else:
                tmp = OW.isel(time=t, lat=slice(ijmin, ijmax),
                              lon=slice(iimin, iimax)).values.copy()
            eddy_object_with_mass = np.zeros_like(tmp)
            eddy_object_with_mass[index_eddy] = tmp[index_eddy]
            j_cen, i_cen = ndimage.center_of_mass(eddy_object_with_mass)
            j_cen, i_cen = j_cen + ijmin, i_cen + iimin
            if regrid_avoided:
                lon_eddies = np.interp(i_cen, range(0, len_OW_lon),
                                OW['lon'][int(np.around(j_cen)), :].values)
                lat_eddies = np.interp(j_cen, range(0, len_OW_lat),
                                OW['lat'][:, int(np.around(i_cen))].values)
            else:
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
            # calc vorticity amplitude
            if vort.isel(time=t).values[interior].mean() > 0:
                amp = (vort.isel(time=t).values[interior].max()
                       - vort.isel(time=t).values[exterior].mean())
            elif vort.isel(time=t).values[interior].mean() < 0:
                amp = (vort.isel(time=t).values[exterior].mean()
                       - vort.isel(time=t).values[interior].min())
            eddi[e]['amp'] = np.array([amp])
            # store all eddy indices
            if regrid_avoided:
                j_min = (data.y.where(data.y == OW.y.min(), other=0)
                         ** 2).argmax().values
                i_min = (data.x.where(data.x == OW.x.min(), other=0)
                         ** 2).argmax().values
            else:
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
            # get eddy type from Vorticity and store extrema
            if det_param['grid'] == 'latlon':
                if eddi[e]['lat'] < 0:
                    if vort.isel(time=t).values[index].mean() < 0:
                        eddi[e]['type'] = 'cyclonic'
                    elif vort.isel(time=t).values[index].mean() > 0:
                        eddi[e]['type'] = 'anticyclonic'
                elif eddi[e]['lat'] >= 0:
                    if vort.isel(time=t).values[index].mean() > 0:
                        eddi[e]['type'] = 'cyclonic'
                    elif vort.isel(time=t).values[index].mean() < 0:
                        eddi[e]['type'] = 'anticyclonic'
            elif det_param['grid'] == 'cartesian':
                if det_param['hemi'] == 'south':
                    if vort.isel(time=t).values[index].mean() < 0:
                        eddi[e]['type'] = 'cyclonic'
                    elif vort.isel(time=t).values[index].mean() > 0:
                        eddi[e]['type'] = 'anticyclonic'
                elif det_param['hemi'] == 'north':
                    if vort.isel(time=t).values[index].mean() > 0:
                        eddi[e]['type'] = 'cyclonic'
                    elif vort.isel(time=t).values[index].mean() < 0:
                        eddi[e]['type'] = 'anticyclonic'
            e += 1
        else:
            del eddi[e]
    return eddi


def detect_SSH_core(data, det_param, SSH, t, ssh_crits, e1f, e2f,
                   regrid_avoided=False):
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
    if regrid_avoided == True:
        raise ValueError("regrid_avoided cannot be used in combination"
                         + "with detection based on SSH (yet).")
    #set up grid
    llon, llat = np.meshgrid(SSH.lon, SSH.lat)
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    for cyc in ['cyclonic', 'anticyclonic']:
        field = SSH.isel(time=t).values
        # ssh_crits increasing for 'anticyclonic', decreasing for 'cyclonic'
        # flip to start with largest positive value for 'cylonic'
        if cyc == 'anticyclonic':
            ssh_crits = -ssh_crits
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
            region_index = get_indeces(regions)
            for iregion in list(range(nregions)):
                eddi[e] = {}
        # 2. Calculate number of pixels comprising detected region, reject if
        # not within [Npix_min, Npix_max]
                region = (regions==iregion+1).astype(int)
                region_Npix = region.sum()
                index = region_index[iregion + 1]
                eddy_area_within_limits = (
                    (region_Npix < det_param['Npix_max'])
                    * (region_Npix > det_param['Npix_min']))
        # 3. Detect presence of local maximum (minimum) for anticylonic
        # (cyclonic) eddies, reject if non-existent
                interior = ndimage.binary_erosion(region)
                exterior = region.astype(bool) ^ interior
                if interior.sum() == 0:
                    del eddi[e]
                    continue
                if cyc == 'anticyclonic':
                    has_internal_ext = (field[interior].max() >
                                        field[exterior].max())
                elif cyc == 'cyclonic':
                    has_internal_ext = (field[interior].min() <
                                        field[exterior].min())
        # 4. Find amplitude of region, reject if < amp_thresh
                if cyc == 'anticyclonic':
                    amp = field[interior].max() - field[exterior].mean()
                elif cyc == 'cyclonic':
                    amp = field[exterior].mean() - field[interior].min()
                is_tall_eddy = (amp >= det_param['amp_thr'])
        # 5. Find maximum linear dimension of region, reject if < d_thresh
                lon_ext = llon[exterior]
                lat_ext = llat[exterior]
                d = distance_matrix(lon_ext, lat_ext)
                is_small_eddy = d.max() < det_param['d_thr']
        # 6. Ratio of x- to y-extension of eddy has to be > 0.5
                min_width = int(np.around(np.sqrt(region_Npix) / 2))
                X_cen = int(np.around(np.mean(index[1])))
                Y_cen = int(np.around(np.mean(index[0])))
                Ypix_cen = np.sum(index[1] == X_cen)
                Xpix_cen = np.sum(index[0] == Y_cen)
                eddy_not_too_thin = ((Xpix_cen > min_width)
                                     & (Ypix_cen > min_width))
        # Detected eddies:
                if (eddy_area_within_limits * has_internal_ext
                    * is_tall_eddy * is_small_eddy * eddy_not_too_thin):
                    # find centre of mass of eddy
                    # find centre of mass of eddy
                    iimin = index[1].min()
                    iimax = index[1].max() + 1
                    ijmin = index[0].min()
                    ijmax = index[0].max() + 1
                    index_eddy = (index[0] - ijmin, index[1] - iimin)
                    tmp = SSH.isel(time=t, lat=slice(ijmin, ijmax),
                                  lon=slice(iimin, iimax)).values.copy()
                    eddy_object_with_mass = np.zeros_like(tmp)
                    eddy_object_with_mass[index_eddy] = tmp[index_eddy]
                    j_cen, i_cen = ndimage.center_of_mass(eddy_object_with_mass)
                    j_cen, i_cen = j_cen + ijmin, i_cen + iimin
                    lon_eddies = np.interp(i_cen, range(0, len(SSH['lon'])),
                                           SSH['lon'].values)
                    lat_eddies = np.interp(j_cen, range(0, len(SSH['lat'])),
                                           SSH['lat'].values)
                    if det_param['grid'] == 'latlon':
                        if lon_eddies > 180:
                            eddi[e]['lon'] = np.array([lon_eddies]) - 360.
                        elif lon_eddies < -180:
                            eddi[e]['lon'] = np.array([lon_eddies]) + 360.
                        else:
                            eddi[e]['lon'] = np.array([lon_eddies])
                    else:
                        eddi[e]['lon'] = np.array([lon_eddies])
                    eddi[e]['lat'] = np.array([lat_eddies])
                    # store all eddy indices
                    j_min = (data.lat.where(data.lat == SSH.lat.min(), other=0)
                             ** 2).argmax().values
                    i_min = (data.lon.where(data.lon == SSH.lon.min(), other=0)
                             ** 2).argmax().values
                    eddi[e]['eddy_j'] = index[0] + j_min
                    eddi[e]['eddy_i'] = index[1] + i_min
                    # assign (and calculated) amplitude, area, and scale of
                    # eddies
                    area = (region_Npix * det_param['res'] ** 2
                            * e2f * e1f)
                    # [km**2]
                    scale = np.sqrt(area / np.pi) # [km]
                    # remove its interior pixels from further eddy detection
                    eddy_mask = np.ones(field.shape)
                    eddy_mask[interior.astype(int)==1] = np.nan
                    field = field * eddy_mask
                    eddi[e]['time'] = SSH.isel(time=t).time.values
                    eddi[e]['amp'] = np.array([amp])
                    eddi[e]['area'] = np.array([area])
                    eddi[e]['scale'] = np.array([scale])
                    eddi[e]['type'] = cyc
                    e += 1
                else:
                    del eddi[e]
    return eddi


def detect_UV_core(data, det_param, U, V, SPEED, t, e1f, e2f,
                   regrid_avoided=False):
    if regrid_avoided == True:
        raise ValueError("regrid_avoided cannot be used (yet).")
    u = U.isel(time=t).values
    v = V.isel(time=t).values
    speed = SPEED.isel(time=t).values
    lon, lat = np.meshgrid(data.lon.sel(lon=slice(det_param["lon1"],
                                                  det_param["lon2"])).values,
                           data.lat.sel(lat=slice(det_param["lat1"],
                                                  det_param["lat2"])).values)
    a = det_param["a"]
    b = det_param["b"]
    rad = det_param["rad"]
    borders = np.max([a, b]) + 1
    bounds = np.shape(speed)
    # initialise eddy counter & output dict
    e = 0
    eddi = {}
    for i in np.arange(borders, len(v[:, 0])-borders+1):
        wrk = v[i, :]
        # reversal of direction in V
        s = np.sign(wrk)
        indx = np.where((np.diff(s) != 0) & (~np.isnan(np.diff(s))))[0]
        indx = indx[((indx > borders) & (indx < (len(wrk) - borders)))]
        for ii in np.arange(0, len(indx)):
            # v increase "a" points away from reversal
            # anticyclonic
            if wrk[indx[ii]] >= 0:
                if ((wrk[indx[ii]-a] > wrk[indx[ii]])
                    & (wrk[indx[ii]+1+a] < wrk[indx[ii]+1])):
                    var = -1
                else:
                    var = 0
            # cyclonic
            elif wrk[indx[ii]] < 0:
                if ((wrk[indx[ii]-a] < wrk[indx[ii]])
                    & (wrk[indx[ii]+1+a] > wrk[indx[ii]+1])):
                    var = 1
                else:
                    var = 0
            # reversal of direction in U and increase away from reversal
            # anticyclonic
            if var == -1:
                if (((u[i-a, indx[ii]] <= 0) & (u[i-a, indx[ii]] <= u[i-1, indx[ii]])
                     & (u[i+a, indx[ii]] >= 0) & (u[i+a, indx[ii]] >= u[i+1, indx[ii]]))
                    | ((u[i-a, indx[ii]+1] <= 0) & (u[i-a, indx[ii]+1] <= u[i-1, indx[ii]+1])
                       & (u[i+a, indx[ii]+1] >= 0) & (u[i+a, indx[ii]+1] >= u[i+1, indx[ii]+1]))):
                    var = -1
                    #eddy_uv.append([(lat[i, indx[ii]], lon[i, indx[ii]]),
                    #                (lat[i, indx[ii]+1], lon[i, indx[ii]+1])])
                else:
                    var=0
            # cyclonic
            elif var == 1:
                if (((u[i-a, indx[ii]] >= 0) & (u[i-a, indx[ii]] >= u[i-1, indx[ii]])
                     & (u[i+a, indx[ii]] <= 0) & (u[i+a, indx[ii]] <= u[i+1, indx[ii]]))
                    | ((u[i-a, indx[ii]+1] >= 0) & (u[i-a, indx[ii]+1] >= u[i-1, indx[ii]+1])
                       & (u[i+a, indx[ii]+1] <= 0) & (u[i+a, indx[ii]+1] <= u[i+1, indx[ii]+1]))):
                    var = 1
                    #eddy_uv.append([(lat[i, indx[ii]], lon[i, indx[ii]]),
                    #                (lat[i, indx[ii]+1], lon[i, indx[ii]+1])])
                else:
                    var=0
            # find the velocity minimum within the searching area defined by
            # "b" around the points that satisfy the first two constraints
            if var != 0:
                # velocity magnitude, latitude and longitude within the
                # searching area
                srch = speed[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                slat = lat[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                slon = lon[i-b:i+b, indx[ii]-b:indx[ii]+1+b]
                # position of the velocity minimum within the searching area
                X, Y = np.where(srch == np.min(srch))
                # second searching area centered around the velocity minimum
                # (bound prevents this area from extending outside the domain)
                srch2 = speed[int(max((i-b)+(X-1)-b, 1)):int(min((i-b)+(X-1)+b, bounds[0])),
                              int(max((indx[ii]-b)+(Y-1)-b, 1)):int(min((indx[ii]-b)+(Y-1)+b, bounds[1]))]
                # if the two minima coincide then it is a local minima
                if (np.min(srch2) != np.min(srch)):
                    var = 0
                #else:
                #    eddy_c.append([(slat[X, Y][0], slon[X, Y][0])])
            # check the rotation of the vectors along the boundary of the area
            # "a-1" around the points which satisfy the first three constraints
            d = a - 1
            if var != 0:
                # indices of the estimated center in the large domain
                i1, i2 = np.where((lat == slat[X, Y]) & (lon == slon[X, Y]))
                # velocities within "a-1" points from the estimated center
                u_small = u[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                v_small = v[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                lon_small = lon[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                lat_small = lat[int(max(i1-d, 1)):int(min(i1+d+1, bounds[0])),
                            int(max(i2-d, 1)):int(min(i2+d+1, bounds[1]))]
                # constraint is applied only if sea-points are within the area
                if ~np.isnan(u_small).all():
                    # boundary velocities
                    u_bound = [u_small[0, :], u_small[1::, -1],
                               u_small[-1, -2:0:-1], u_small[-1:0:-1, 0]]
                    u_bound = np.array([item for sublist in u_bound for item in sublist])
                    v_bound = [v_small[0, :], v_small[1::, -1],
                               v_small[-1, -2:0:-1], v_small[-1:0:-1, 0]]
                    v_bound = np.array([item for sublist in v_bound for item in sublist])
                    # vector defining which quadrant each boundary vector
                    # belongs to
                    quadrants = np.zeros_like(u_bound)
                    quadrants[((u_bound >= 0) & (v_bound >= 0))] = 1
                    quadrants[((u_bound < 0) & (v_bound >= 0))] = 2
                    quadrants[((u_bound < 0) & (v_bound < 0))] = 3
                    quadrants[((u_bound >= 0) & (v_bound < 0))] = 4
                    # used identify which is the firts fourth quadrant vector
                    spin = np.where(quadrants==4)[0]
                    # apply the constraint only if complete rotation and not
                    # all vectors in the fourth quadrant
                    if ((spin.size != 0) & (spin.size != quadrants.size)):
                        # if vectors start in 4 quadrant, then I add 4 to all
                        # quandrant positions from the first 1 occurrence
                        if spin[0] == 0:
                            spin = np.where(quadrants!=4)[0]
                            spin = spin[0] - 1
                        else:
                            spin = spin[-1]
                        quadrants[spin+1::] = quadrants[spin+1::] + 4
                        # inspect vector rotation:
                        # - no consecutive vectors more than one quadrant away
                        # - no backward rotation
                        if ((np.where(np.diff(quadrants) > 1)[0].size == 0)
                            & (np.where(np.diff(quadrants) < 0)[0].size == 0)):
                            u_large = u[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                        int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            v_large = v[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                        int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            speed_large = speed[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                                int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            lon_large = lon[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            lat_large = lat[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            e1f_large = e1f[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            e2f_large = e2f[int(max(i1-(rad*a), 1)):int(min(i1+(rad*a), bounds[0])),
                                            int(max(i2-(rad*a), 1)):int(min(i2+(rad*a), bounds[1]))]
                            psi = compute_psi(u_large, v_large, e1f_large, e2f_large)
                            eddy_i, eddy_j, eddy_mask = get_eddy_indeces(lon_large, lat_large,
                                                                         slon[X, Y][0], slat[X, Y][0], psi, speed_large)
                            if ((np.shape(eddy_i)!=()) & (np.shape(eddy_j)!=()) & (np.shape(eddy_mask)!=())):
                                eddi[e] = {}
                                eddi[e]["lon"] = slon[X, Y]
                                eddi[e]["lat"] = slat[X, Y]
                                eddi[e]['eddy_j'] = eddy_j + int(max(i2-(rad*a), 1))
                                eddi[e]['eddy_i'] = eddy_i + int(max(i1-(rad*a), 1))
                                eddi[e]['time'] = U.isel(time=t).time.values
                                eddi[e]['amp'] = np.array([np.nanmax(psi * eddy_mask) - np.nanmin(psi * eddy_mask)])
                                area = (e1f_large / 1000.) * (e2f_large / 1000.) * eddy_mask
                                eddi[e]['area'] = np.array([np.nansum(area)])
                                eddi[e]['scale'] = np.array([np.sqrt(eddi[e]['area'] / np.pi)])
                                if det_param["hemi"] == "north":
                                    if var == -1:
                                        eddi[e]['type'] = "anticyclonic"
                                    elif var ==1:
                                        eddi[e]['type'] = "cyclonic"
                                elif det_param["hemi"] == "south":
                                    if var == -1:
                                        eddi[e]['type'] = "cyclonic"
                                    elif var ==1:
                                        eddi[e]['type'] = "anticyclonic"
                                e += 1
    return eddi


def detect_OW(data, det_param, ow_var, vort_var,
              use_bags=False, use_mp=False, mp_cpu=2,
              regrid_avoided=False):
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
            'Npix_max': 1000, # max. num. grid cells to be considered as eddy
            'no_long': False, # If True, elongated shapes will not be considered
            'no_two': False # If True, eddies with two minima in the OW
                            # parameter and a OW > OW_thr in between  will not
                            # be considered
            }
    ow_var : str
        Name of the variable in `data` containing the Okubo-Weiss parameter.
    vort_var : str
        Name of the variable in `data` containing the vorticity field.
    use_bags : bool
        If True, dask_bags is used to parallelize the detection. Default is
        False.
    use_mp : bool
        If True, multiprocessing is used to parallelize the detection. Default
        is False.
    mp_cpu : int
        Number of cpus to use when using multiprocessing.
    regrid_avoided : bool
        If True indicates that regridding has been avoided during the
        interpolation and the data has 2D coordinates. Default is False.

    Returns
    -------
    eddies : dict
        Dictionary containing information on all detected eddies.
        The dict has the form:
        eddies = {t: {e: {'time': array, # time stamp
                          'type': str, # 'cyclonic' or 'anticyclonic'
                          'lon': array, # central longitude
                          'lat': array, # central latitude
                          'scale': array, # ~radius of the eddy
                          'area': array, # area of the eddy
                          'amp': array, # vorticity amplitude
                          'eddy_i': array, # i-indeces of the eddy
                          'eddy_j': array # j-indeces of the eddy
                          }}}
        where `t` is the time step and `e` is the eddy number.
    '''
    # make sure arguments are compatible
    if use_bags and use_mp:
        raise ValueError('Cannot use dask_bags and multiprocessing at the'
                         + 'same time. Set either `use_bags` or `use_mp`'
                         + 'to `False`.')
    # Verify that the specified region lies within the dataset provided
    if (det_param['lon1'] < np.around(data['lon'].min())
       or det_param['lon2'] > np.around(data['lon'].max())):
        raise ValueError('`det_param`: min. and/or max. of longitude range'
                         + ' are outside the region contained in the dataset')
    if (det_param['lat1'] < np.around(data['lat'].min())
       or det_param['lat2'] > np.around(data['lat'].max())):
        raise ValueError('`det_param`: min. and/or max. of latitude range'
                         + ' are outside the region contained in the dataset')
    if det_param['calendar'] == 'standard':
        start_time = np.datetime64(det_param['start_time'])
        end_time = np.datetime64(det_param['end_time'])
    elif det_param['calendar'] == '360_day':
        start_time = cft.Datetime360Day(int(det_param['start_time'][0:4]),
                                        int(det_param['start_time'][5:7]),
                                        int(det_param['start_time'][8:10]))
        end_time = cft.Datetime360Day(int(det_param['end_time'][0:4]),
                                      int(det_param['end_time'][5:7]),
                                      int(det_param['end_time'][8:10]))
    if (start_time > data['time'][-1]
        or end_time < data['time'][0]):
        raise ValueError('`det_param`: there is no overlap of the original time'
                         + ' axis and the desired time range for the'
                         + ' detection')
    #
    print('preparing data for eddy detection'
          + ' (masking and region extracting etc.)')
    # Make sure longitude vector is monotonically increasing if we have a
    # latlon grid
    if det_param['grid'] == 'latlon':
        data, det_param = monotonic_lon(data, det_param)
    # Masking shallow regions and cut out the region specified in `det_param`
    OW = maskandcut(data, ow_var, det_param, regrid_avoided=regrid_avoided)
    vort = maskandcut(data, vort_var, det_param, regrid_avoided=regrid_avoided)
    OW_thr_name = det_param['OW_thr_name']
    # Define the names of the grid cell sizes depending on the model
    if det_param['model'] == 'MITgcm':
        e1f_name = 'dxV'
        e2f_name = 'dyU'
    if det_param['model'] == 'ORCA':
        e1f_name = 'e1f'
        e2f_name = 'e2f'
    e1f = maskandcut(data, e1f_name, det_param, regrid_avoided=regrid_avoided)
    e2f = maskandcut(data, e2f_name, det_param, regrid_avoided=regrid_avoided)
    if len(np.shape(data[OW_thr_name])) > 1:
        # If the Okubo-Weiss threshold is 2D, use `maskandcutOW` masking etc.
        OW_thr = maskandcut(data, OW_thr_name, det_param,
                            regrid_avoided=regrid_avoided)
        OW_thr = OW_thr * (det_param['OW_thr_factor'])
    else:
        # Else just use scalar from `det_param`
        OW_thr = det_param['OW_thr'] * (det_param['OW_thr_factor'])
    if use_mp:
        OW = OW.compute()
        vort = vort.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        if len(np.shape(data[OW_thr_name])) > 1:
            OW_thr = OW_thr.compute()
        ## set range of parallel executions
        pexps = range(0, len(OW['time']))
        ## prepare arguments
        arguments = zip(
                        itertools.repeat(data),
                        itertools.repeat(det_param.copy()),
                        itertools.repeat(OW),
                        itertools.repeat(vort),
                        pexps,
                        itertools.repeat(OW_thr),
                        itertools.repeat(e1f),
                        itertools.repeat(e2f),
                        itertools.repeat(regrid_avoided)
                        )
        print("Detecting eddies in Okubo-Weiss parameter fields")
        if mp_cpu > mp.cpu_count():
            mp_cpu = mp.cpu_count()
        with mp.Pool(mp_cpu) as p:
            eddies = p.starmap(detect_OW_core, arguments)
        p.close()
        p.join()
    elif use_bags:
        OW = OW.compute()
        vort = vort.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        if len(np.shape(data[OW_thr_name])) > 1:
            OW_thr = OW_thr.compute()
        ## set range of parallel executions
        pexps = range(0, len(OW['time']))
        ## generate dask bag instance
        seeds_bag = dask_bag.from_sequence(pexps)
        print("Detecting eddies in Okubo-Weiss parameter fields")
        detection = dask_bag.map(
            lambda tt: detect_OW_core(data, det_param.copy(),
                                      OW, vort, tt, OW_thr, e1f, e2f,
                                      regrid_avoided=regrid_avoided)
                                 ,seeds_bag)
        eddies = detection.compute()
    else:
        eddies = {}
        OW = OW.compute()
        vort = vort.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        if len(np.shape(data[OW_thr_name])) > 1:
            OW_thr = OW_thr.compute()
        for tt in np.arange(0, len(OW['time'])):
            steps = np.around(np.linspace(0, len(OW['time']), 10))
            if tt in steps:
                print('detection at time step ', str(tt + 1), ' of ',
                      len(OW['time']))
            eddies[tt] = detect_OW_core(data, det_param.copy(),
                                        OW, vort, tt, OW_thr, e1f, e2f,
                                        regrid_avoided=regrid_avoided)
    return eddies


def detect_SSH(data, det_param, ssh_var,
              use_bags=False, use_mp=False, mp_cpu=2,
              regrid_avoided=False):
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
    use_bags : bool
        If True, dask_bags is used to parallelize the detection. Default is
        False.
    use_mp : bool
        If True, multiprocessing is used to parallelize the detection. Default
        is False.
    mp_cpu : int
        Number of cpus to use when using multiprocessing.
    regrid_avoided : bool
        If True indicates that regridding has been avoided during the
        interpolation and the data has 2D coordinates. Default is False.
    Returns
    -------
    eddies : dict
        Dictionary containing information on all detected eddies.
        The dict has the form:
        eddies = {t: {e: {'time': array, # time stamp
                          'type': str, # 'cyclonic' or 'anticyclonic'
                          'lon': array, # central longitude
                          'lat': array, # central latitude
                          'scale': array, # ~radius of the eddy
                          'area': array, # area of the eddy
                          'amp': array, # vorticity amplitude
                          'eddy_i': array, # i-indeces of the eddy
                          'eddy_j': array # j-indeces of the eddy
                          }}}
        where `t` is the time step and `e` is the eddy number.
    '''
    # make sure arguments are compatible
    if use_bags and use_mp:
        raise ValueError('Cannot use dask_bags and multiprocessing at the'
                         + 'same time. Set either `use_bags` or `use_mp`'
                         + 'to `False`.')
    if regrid_avoided == True:
        raise ValueError("regrid_avoided cannot be used in combination"
                         + "with detection based on SSH (yet).")
    # Verify that the specified region lies within the dataset provided
    if (det_param['lon1'] < np.around(data['lon'].min())
        or det_param['lon2'] > np.around(data['lon'].max())):
        raise ValueError('`det_param`: min. and/or max. of longitude range'
                         + ' are outside the region contained in the dataset')
    if (det_param['lat1'] < np.around(data['lat'].min())
        or det_param['lat2'] > np.around(data['lat'].max())):
        raise ValueError('`det_param`: min. and/or max. of latitude range'
                         + ' are outside the region contained in the dataset')
    if det_param['calendar'] == 'standard':
        start_time = np.datetime64(det_param['start_time'])
        end_time = np.datetime64(det_param['end_time'])
    elif det_param['calendar'] == '360_day':
        start_time = cft.Datetime360Day(int(det_param['start_time'][0:4]),
                                        int(det_param['start_time'][5:7]),
                                        int(det_param['start_time'][8:10]))
        end_time = cft.Datetime360Day(int(det_param['end_time'][0:4]),
                                      int(det_param['end_time'][5:7]),
                                      int(det_param['end_time'][8:10]))
    if (start_time > data['time'][-1]
        or end_time < data['time'][0]):
        raise ValueError('`det_param`: there is no overlap of the original time'
                         + ' axis and the desired time range for the'
                         + ' detection')
    #
    print('preparing data for eddy detection'
          + ' (masking and region extracting etc.)')
    # Make sure longitude vector is monotonically increasing if we have a
    # latlon grid
    if det_param['grid'] == 'latlon':
        data, det_param = monotonic_lon(data, det_param)
    # Masking shallow regions and cut out the region specified in `det_param`
    SSH = maskandcut(data, ssh_var, det_param)
    # Define the names of the grid cell sizes depending on the model
    if det_param['model'] == 'MITgcm':
        e1f_name = 'dxV'
        e2f_name = 'dyU'
    if det_param['model'] == 'ORCA':
        e1f_name = 'e1f'
        e2f_name = 'e2f'
    e1f = maskandcut(data, e1f_name, det_param)
    e2f = maskandcut(data, e2f_name, det_param)
    ## create list of incremental threshold
    ssh_crits = np.arange(-det_param['ssh_thr'],
                          det_param['ssh_thr'] + det_param['dssh'],
                          det_param['dssh'])
    ssh_crits = np.sort(ssh_crits) # make sure its increasing order
    if use_mp:
        SSH = SSH.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        ## set range of parallel executions
        pexps = range(0, len(SSH['time']))
        ## prepare arguments
        arguments = zip(
                        itertools.repeat(data),
                        itertools.repeat(det_param.copy()),
                        itertools.repeat(SSH),
                        pexps,
                        itertools.repeat(ssh_crits),
                        itertools.repeat(e1f),
                        itertools.repeat(e2f)
                        )
        print("Detecting eddies in SSH parameter fields")
        if mp_cpu > mp.cpu_count():
            mp_cpu = mp.cpu_count()
        with mp.Pool(mp_cpu) as p:
            eddies = p.starmap(detect_SSH_core, arguments)
        p.close()
        p.join()
    elif use_bags:
        SSH = SSH.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        ## set range of parallel executions
        pexps = range(0, len(SSH['time']))
        ## generate dask bag instance
        seeds_bag = dask_bag.from_sequence(pexps)
        print("Detecting eddies in SSH fields")
        detection = dask_bag.map(
            lambda tt: detect_SSH_core(data, det_param.copy(), SSH, tt,
                                       ssh_crits, e1f, e2f)
                                 ,seeds_bag)
        eddies = detection.compute()
    else:
        eddies = {}
        SSH = SSH.compute()
        e1f = e1f.compute()
        e2f = e2f.compute()
        for tt in np.arange(0, len(SSH['time'])):
            steps = np.around(np.linspace(0, len(SSH['time']), 10))
            if tt in steps:
                print('detection at time step ', str(tt + 1), ' of ',
                      len(SSH['time']))
            eddies[tt] = detect_SSH_core(data, det_param.copy(),
                                         SSH, tt, ssh_crits, e1f, e2f)
    return eddies


def detect_UV(data, det_param, u_var, v_var, speed_var,
              use_bags=False, use_mp=False, mp_cpu=2,
              regrid_avoided=False):
    # make sure arguments are compatible
    if use_bags and use_mp:
        raise ValueError('Cannot use dask_bags and multiprocessing at the'
                         + 'same time. Set either `use_bags` or `use_mp`'
                         + 'to `False`.')
    if regrid_avoided == True:
        raise ValueError("regrid_avoided cannot be used in combination"
                         + "with detection based on SSH (yet).")
    # Verify that the specified region lies within the dataset provided
    if (det_param['lon1'] < np.around(data['lon'].min())
        or det_param['lon2'] > np.around(data['lon'].max())):
        raise ValueError('`det_param`: min. and/or max. of longitude range'
                         + ' are outside the region contained in the dataset')
    if (det_param['lat1'] < np.around(data['lat'].min())
        or det_param['lat2'] > np.around(data['lat'].max())):
        raise ValueError('`det_param`: min. and/or max. of latitude range'
                         + ' are outside the region contained in the dataset')
    if det_param['calendar'] == 'standard':
        start_time = np.datetime64(det_param['start_time'])
        end_time = np.datetime64(det_param['end_time'])
    elif det_param['calendar'] == '360_day':
        start_time = cft.Datetime360Day(int(det_param['start_time'][0:4]),
                                        int(det_param['start_time'][5:7]),
                                        int(det_param['start_time'][8:10]))
        end_time = cft.Datetime360Day(int(det_param['end_time'][0:4]),
                                      int(det_param['end_time'][5:7]),
                                      int(det_param['end_time'][8:10]))
    if (start_time > data['time'][-1]
        or end_time < data['time'][0]):
        raise ValueError('`det_param`: there is no overlap of the original time'
                         + ' axis and the desired time range for the'
                         + ' detection')
    #
    print('preparing data for eddy detection'
          + ' (masking and region extracting etc.)')
    # Make sure longitude vector is monotonically increasing if we have a
    # latlon grid
    if det_param['grid'] == 'latlon':
        data, det_param = monotonic_lon(data, det_param)
    # Masking shallow regions and cut out the region specified in `det_param`
    U = maskandcut(data, u_var, det_param)
    V = maskandcut(data, v_var, det_param)
    SPEED = maskandcut(data, speed_var, det_param)
    # Define the names of the grid cell sizes depending on the model
    if det_param['model'] == 'MITgcm':
        e1f_name = 'dxV'
        e2f_name = 'dyU'
    if det_param['model'] == 'ORCA':
        e1f_name = 'e1f'
        e2f_name = 'e2f'
    e1f = maskandcut(data, e1f_name, det_param)
    e2f = maskandcut(data, e2f_name, det_param)
    if use_mp:
        U = U.compute()
        V = V.compute()
        SPEED = SPEED.compute()
        e1f = e1f.values
        e2f = e2f.values
        ## set range of parallel executions
        pexps = range(0, len(U['time']))
        ## prepare arguments
        arguments = zip(
                        itertools.repeat(data),
                        itertools.repeat(det_param.copy()),
                        itertools.repeat(U),
                        itertools.repeat(V),
                        itertools.repeat(SPEED),
                        pexps,
                        itertools.repeat(e1f),
                        itertools.repeat(e2f)
                        )
        print("Detecting eddies in velocity fields")
        if mp_cpu > mp.cpu_count():
            mp_cpu = mp.cpu_count()
        with mp.Pool(mp_cpu) as p:
            eddies = p.starmap(detect_UV_core, arguments)
        p.close()
        p.join()
    elif use_bags:
        U = U.compute()
        V = V.compute()
        SPEED = SPEED.compute()
        e1f = e1f.values
        e2f = e2f.values
        ## set range of parallel executions
        pexps = range(0, len(U['time']))
        ## generate dask bag instance
        seeds_bag = dask_bag.from_sequence(pexps)
        print("Detecting eddies in velocity fields")
        detection = dask_bag.map(
            lambda tt: detect_UV_core(data, det_param.copy(), U, V, SPEED, tt,
                                      e1f, e2f)
                                 ,seeds_bag)
        eddies = detection.compute()
    else:
        eddies = {}
        U = U.compute()
        V = V.compute()
        SPEED = SPEED.compute()
        e1f = e1f.values
        e2f = e2f.values
        for tt in np.arange(0, len(U['time'])):
            steps = np.around(np.linspace(0, len(U['time']), 10))
            if tt in steps:
                print('detection at time step ', str(tt + 1), ' of ',
                      len(U['time']))
            eddies[tt] = detect_UV_core(data, det_param.copy(),
                                        U, V, SPEED, tt, e1f, e2f)
    return eddies
