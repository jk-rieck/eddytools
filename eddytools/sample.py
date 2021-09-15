"""sample

Collection of functions to sample eddies tracked with `tracking.py` based
on their properties and add some eddy properties, like temperature etc...

"""
import numpy as np
import xarray as xr
import cftime as cft


def add_fields(sampled, interpolated, sample_param, var, lon1, lat1):
    """Add variable fields to the sampled eddies in `sampled`.

    Parameters
    ----------
    sampled : dict
        Dictionary containing eddy information.
    interpolated : xarray.DataSet
        Dataset containing the variables that should be added to `sampled`.
        Must be on the same grid as the one eddies have been detected on
        with `detection.py`
    sample_param : dict
        Dictionary of parameters needed for the eddy sampling.
        The parameters are:
        sample_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'max_time': 73, # maximum length of tracks to consider
                            # (model time steps)
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'type': 'anticyclonic', # type of eddy
            'lifetime': 20, # length of the eddy's track in days
            'size': 20, # eddy size (diameter in km)
            'range': False, # sample eddy within a range of `var_range`
            'ds_range': data_int.isel(z=9), # dataset of `var_range`
            'var_range': ['votemper'], # variable to base the range on
            'value_range': [[4, 7],], # range of `var_range`
            'split': False, # split eddies at a threshold in below and above
            'ds_split': data_int.isel(z=0), # dataset of `var_split`
            'var_split': ['votemper'], # variable to base split on
            'value_split': [5.0,], # split eddies at this value
            'sample_vars': ['votemper'], # variables to sample
            'save_location': datapath, # where to store the netcdf files
            'save_name': 'test_170_-175'
            }
    var : str
        Name of the variable to add to `sampled[i]`
    lon1 : int
        Integer describing how far the first index in i-direction is from the
        i=0 in the original fields
    lat1 : int
        Integer describing how far the first index in j-direction is from the
        j=0 in the original fields

    Returns
    -------
    sampled : dict
        As input `sampled` but with additional fields of `var` added to
        sampled[i].
    """
    # Initialize additional dictionary entries in which to write the fields
    sampled[var] = {}
    sampled[var + '_lon'] = {}
    sampled[var + '_lat'] = {}
    sampled[var + '_sec'] = {}
    sampled[var + '_sec_lon'] = {}
    sampled[var + '_sec_lat'] = {}
    sampled[var + '_around'] = {}
    sampled[var + '_sec_norm_lon'] = {}
    try:
        length = len(sampled['time'])
    except:
        length = 1
    for t in np.arange(0, length):
        # loop over all time steps of the eddy track
        t = int(t)
        if length == 1:
            time = sampled['time']
        else:
            time = sampled['time'][t]
        # get the indeces to use for extraction from `interpolated`
        indeces = np.vstack((sampled['eddy_i'][t] - (lon1),
                             sampled['eddy_j'][t] - (lat1)))
        t_index = np.min(np.where(interpolated['time'].values >= time))
        lon_index = xr.DataArray(indeces[0, :], dims=['lon'])
        lat_index = xr.DataArray(indeces[1, :], dims=['lat'])
        time_index = xr.DataArray([t_index], dims=['time'])
        # add the variable `var` and its coordinates inside the eddy to
        # `sampled[i]`
        if len(interpolated[var].shape) == 3:
            sampled[var][t] = interpolated[var][time_index,
                                                lat_index, lon_index]
        else:
            sampled[var][t] = interpolated[var][time_index, :,
                                                lat_index, lon_index]
        sampled[var + '_lon'][t] = sampled[var][t]['lon'].values
        sampled[var + '_lat'][t] = sampled[var][t]['lat'].values
        # add the values of `var` along a zonal section through the middle of
        # the eddy, together with the coordinates
        if len(interpolated[var].shape) == 3:
            sampled[var + '_sec'][t] = interpolated[var][
                time_index, int(lat_index.mean().values),
                int(lon_index.min().values):int(lon_index.max().values)]
        else:
            sampled[var + '_sec'][t] = interpolated[var][
                time_index, :, int(lat_index.mean().values),
                int(lon_index.min().values):int(lon_index.max().values)]
        sampled[var + '_sec_lon'][t] =\
            sampled[var + '_sec'][t]['lon'].values
        sampled[var + '_sec_lat'][t] =\
            sampled[var + '_sec'][t]['lat'].values
        # normalize longitude to the range (-0.5, 0.5) for easier comparison
        # of different eddies
        lon_for_diff = sampled[var + '_sec'][t]['lon'].values
        if sample_param['grid'] == 'latlon':
            lon_for_diff[lon_for_diff < 0] = (lon_for_diff[lon_for_diff < 0]
                                              + 360)
        diff_lon = lon_for_diff - np.nanmean(lon_for_diff)
        norm_lon = diff_lon / (diff_lon[-1] - diff_lon[0])
        sampled[var + '_sec_norm_lon'][t] = norm_lon
        sampled[var + '_sec'][t] = sampled[var + '_sec'][t].values
        # add a depth profile of the values of `var` in the surroundings of
        # the eddy to calculate anomalies
        sampled[var + '_around'][t] =\
            average_surroundings(indeces, interpolated[var], t_index)
        sampled[var][t] = sampled[var][t].values
    return sampled


def average_surroundings(indeces, interpolated, t_index):
    """Average surroundings of an eddy to calculate anomalies

    Parameters
    ----------
    indeces : list
        List of intergers, the indeces of the eddy in `interpolated`.
    interpolated : xarray.DataArray
        xarray DataArray of the variable to extract.
    t_index : int
        Index for the time dimension of `interpolated`.

    Returns
    -------
    around : array
        Average depth profile of the variable in `interpolated`. Averaged
        over the surroundings of the eddy (+1 radius in each direction).
    """
    # Calculate the radius of the eddy in "index space"
    radius = int(((np.max(indeces[0, :]) - np.min(indeces[0, :])) / 2))
    # add one radiues in each direction to define what are the surroundings
    imin = np.min(indeces[0, :]) - radius
    imax = np.max(indeces[0, :]) + radius + 1
    jmin = np.min(indeces[1, :]) - radius
    jmax = np.max(indeces[1, :]) + radius + 1
    if len(interpolated.shape) == 3:
        sum1 = interpolated[
                   t_index, jmin:jmax, imin:imax].sum(axis=(0, 1)).values
        count1 = np.count_nonzero(interpolated[
                     t_index, jmin:jmax, imin:imax], axis=(0, 1))
        list2 = [interpolated[t_index, indeces[1, j], indeces[0, j]].values
                 for j in np.arange(len(indeces[0, :]))]
        sum2 = np.sum(list2, axis=0)
        count2 = np.count_nonzero(list2, axis=0)
    else:
        sum1 = interpolated[
                   t_index, :, jmin:jmax, imin:imax].sum(axis=(1, 2)).values
        count1 = np.count_nonzero(interpolated[
                     t_index, :, jmin:jmax, imin:imax], axis=(1, 2))
        list2 = [interpolated[t_index, :, indeces[1, j], indeces[0, j]].values
                 for j in np.arange(len(indeces[0, :]))]
        sum2 = np.sum(list2, axis=0)
        count2 = np.count_nonzero(list2, axis=0)
    around = (sum1 - sum2) / (count1 - count2)
    return around


def monotonic_lon(var, lon):
    '''Make sure longitude is monotonically increasing in `var`.

    Parameters
    ----------
    var : xarray.DataArray
        Data array with the variable that is later to be interpolated.

    Returns
    -------
    var : xarray.DataArray
        Same as input `var` but with montonic longitude.
    '''
    # Make sure the longitude is monotonically increasing for the interpolation
    if var['lon'][0] > var['lon'][-1]:
        lon_mod = var['lon']\
            .where(var['lon']
                   >= np.around(var['lon'][0].values),
                   other=var['lon'] + 360)
        var = var.assign_coords({'lon': lon_mod})
        if lon < lon_mod[0]:
            lon = lon + 360
    return var, lon


def write_to_netcdf(file_name, sample, sample_param, data):
    """ Write a tracked eddy to a netcdf file on disk.

    Parameters
    ----------
    file_name : str
        File name of the `.nc` file that the eddy in `sample` should be written
        into.
    sample : dict
        Dictionary of the tracked eddy as returned from `sample_core()`.
    sample_param : dict
        Dictionary of parameters needed for the eddy sampling.
        The parameters are:
        sample_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'max_time': 73, # maximum length of tracks to consider
                            # (model time steps)
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'type': 'anticyclonic', # type of eddy
            'lifetime': 20, # length of the eddy's track in days
            'size': 20, # eddy size (diameter in km)
            'range': False, # sample eddy within a range of `var_range`
            'ds_range': data_int.isel(z=9), # dataset of `var_range`
            'var_range': ['votemper'], # variable to base the range on
            'value_range': [[4, 7],], # range of `var_range`
            'split': False, # split eddies at a threshold in below and above
            'ds_split': data_int.isel(z=0), # dataset of `var_split`
            'var_split': ['votemper'], # variable to base split on
            'value_split': [5.0,], # split eddies at this value
            'sample_vars': ['votemper'], # variables to sample
            'save_location': datapath, # where to store the netcdf files
            'save_name': 'test_170_-175'
            }
    data : xarray.DataSet
        Dataset from which to extract fields at eddy locations.

    Returns
    -------
    Writes a netcdf file to disk.
    """
    # Define maximum length of `points` and `sec_index` dimensions
    dummy_var = sample_param['sample_vars'][0]
    max_points = np.max([len(sample['eddy_i'][t])
                         for t in np.arange(0, len(sample['eddy_i']))])
    max_sec = np.max([len(sample[dummy_var + '_sec_lon'][t])
                      for t in np.arange(0, len(sample[dummy_var
                                                       + '_sec_lon']))])
    out_nc = xr.Dataset({'time': ('time', sample['time']),
                         'points': ('points', np.arange(0, max_points)),
                         'depth': ('depth', data['z'].data),
                         'sec_index': ('sec_index', np.arange(0, max_sec))})
    out_nc = out_nc.set_coords(['time', 'points', 'depth', 'sec_index'])

    out_nc = out_nc.update({'type': (sample['type'])})
    out_nc = out_nc.update({'exist_at_start': (sample['exist_at_start'])})
    out_nc = out_nc.update({'terminated': (sample['terminated'])})

    out_nc = out_nc.update({'time': (['time'], sample['time'])})
    out_nc = out_nc.update({'amp':
                            (['time'], sample['amp'].astype(np.float32))})
    out_nc = out_nc.update({'lon':
                            (['time'], sample['lon'].astype(np.float32))})
    out_nc = out_nc.update({'lat':
                            (['time'], sample['lat'].astype(np.float32))})
    out_nc = out_nc.update({'area':
                            (['time'], sample['area'].astype(np.float32))})
    out_nc = out_nc.update({'scale':
                            (['time'], sample['scale'].astype(np.float32))})

    len_t = len(sample['time'])
    len_z = len(data['z'])
    for var in sample_param['sample_vars']:
        dummy_around = np.zeros((len_t, len_z)) + np.nan
        dummy_lon = np.zeros((len_t, max_points)) + np.nan
        dummy_lat = np.zeros((len_t, max_points)) + np.nan
        dummy_var = np.zeros((len_t, len_z, max_points, max_points)) + np.nan
        dummy_sec_lon = np.zeros((len_t, max_sec)) + np.nan
        dummy_sec_lat = np.zeros((len_t)) + np.nan
        dummy_sec = np.zeros((len_t, len_z, max_sec)) + np.nan
        dummy_sec_norm_lon = np.zeros((len_t, max_sec)) + np.nan
        for t in np.arange(0, len(sample['time'])):
            len_p = len(sample['eddy_i'][t])
            len_s = len(sample[var + '_sec_lon'][t])
            dummy_around[t, :] = sample[var + '_around'][t]
            dummy_lon[t, 0:len_p] = sample[var + '_lon'][t]
            dummy_lat[t, 0:len_p] = sample[var + '_lat'][t]
            dummy_var[t, :, 0:len_p, 0:len_p] = sample[var][t]
            dummy_sec_lon[t, 0:len_s] = sample[var + '_sec_lon'][t]
            dummy_sec_lat[t] = sample[var + '_sec_lat'][t]
            dummy_sec[t, :, 0:len_s] = sample[var + '_sec'][t]
            dummy_sec_norm_lon[t, 0:len_s] = sample[var + '_sec_norm_lon'][t]
        out_nc = out_nc.update({var + '_around':
                                (['time', 'depth'],
                                 dummy_around.astype(np.float32))})
        out_nc = out_nc.update({var + '_lon':
                                (['time', 'points'],
                                 dummy_lon.astype(np.float32))})
        out_nc = out_nc.update({var + '_lat':
                                (['time', 'points'],
                                 dummy_lat.astype(np.float32))})
        out_nc = out_nc.update({var:
                                (['time', 'depth', 'points', 'points'],
                                 dummy_var.astype(np.float32))})
        out_nc = out_nc.update({var + '_sec_lon':
                                (['time', 'sec_index'],
                                 dummy_sec_lon.astype(np.float32))})
        out_nc = out_nc.update({var + '_sec_lat':
                                (['time'],
                                 dummy_sec_lat.astype(np.float32))})
        out_nc = out_nc.update({var + '_sec':
                                (['time', 'depth', 'sec_index'],
                                 dummy_sec.astype(np.float32))})
        out_nc = out_nc.update({var + '_sec_norm_lon':
                                (['time', 'sec_index'],
                                 dummy_sec_norm_lon.astype(np.float32))})

    comp = dict(zlib=True, complevel=1)
    encoding = {var: comp for var in out_nc.data_vars}
    out_nc.to_netcdf(file_name, format='NETCDF4',
                     encoding=encoding, mode='w')


def sample_core(track, data, data_whole, sample_param, i, j,
                lifetime, start_time, end_time):
    """ Core function for the sampling of eddies from a dictionary that
    fulfills certain criteria.

    Parameters
    ----------
    tracks : dict
        Dictionary containing eddy information.
    data : xarray.DataSet
        Dataset from which to extract fields at eddy locations.
    data_whole : xarray.DataSet
        Dataset containing all original data to add more and more years
        to `data` when necessary.
    sample_param : dict
        Dictionary of parameters needed for the eddy sampling.
        The parameters are:
        sample_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'max_time': 73, # maximum length of tracks to consider
                            # (model time steps)
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'type': 'anticyclonic', # type of eddy
            'lifetime': 20, # length of the eddy's track in days
            'size': 20, # eddy size (diameter in km)
            'range': False, # sample eddy within a range of `var_range`
            'ds_range': data_int.isel(z=9), # dataset of `var_range`
            'var_range': ['votemper'], # variable to base the range on
            'value_range': [[4, 7],], # range of `var_range`
            'split': False, # split eddies at a threshold in below and above
            'ds_split': data_int.isel(z=0), # dataset of `var_split`
            'var_split': ['votemper'], # variable to base split on
            'value_split': [5.0,], # split eddies at this value
            'sample_vars': ['votemper'], # variables to sample
            'save_location': datapath, # where to store the netcdf files
            'save_name': 'test_170_-175'
            }
    i : int
        Number of sampled eddy.
    j : int
        Number of sampled eddy if `split=True`, j=0 otherwise.
    lifetime : int
        Liftime in time steps needed for an eddy to be considered for
        sampling.
    start_time : np.datetime64
        Earliest time from which to consider sampling eddies.
    end_time : np.datetime64
        Latest time from which to consider sampling eddies.

    Returns
    -------
    sampled : dict
        Dictionary containing the sampled eddies. Is only created when
        `split`: False in `sample_param`.
    above : dict
        Dictionary containing the sampled eddies above the splitting threshold
        `value_split`. Is only created when `split`: True in `sample_param`.
    below : dict
        Dictionary containing the sampled eddies below the splitting threshold
        `value_split`. Is only created when `split`: True in `sample_param`.
    data : xarray.DataSet
        As input `data` but updated when new time steps need to be included.
    i : int
        Number of sampled eddy.
    j : int
        Number of sampled eddy if `split=True`, j=0 otherwise.
    """

    sampled = {}
    ab = 'NotDefined'
    # loop over all eddies in `tracks`
    # determine first and last time step of each eddy
    try:
        length = len(track['time'])
        time0 = np.array(track['time'])[0]
        time1 = np.array(track['time'])[-1]
    except:
        length = 1
        time0 = np.array(track['time'])
        time1 = np.array(track['time'])
    this_year = int(str(data['time'][-1].values)[0:4])
    diff_year = int(str(time1)[0:4]) - this_year
    if sample_param['grid'] == 'latlon':
        addlon = 2
        addlat = 1
    elif sample_param['grid'] == 'cartesian':
        addlon = 2e5
        addlat = 1e5
    lon1 = int(np.argmin(((data_whole['lon']
               - (sample_param['lon1'] - addlon)) ** 2).values))
    lon2 = int(np.argmin(((data_whole['lon']
               - (sample_param['lon2'] + addlon)) ** 2).values))
    lat1 = int(np.argmin(((data_whole['lat']
               - (sample_param['lat1'] - addlat)) ** 2).values))
    lat2 = int(np.argmin(((data_whole['lat']
               - (sample_param['lat2'] + addlat)) ** 2).values))
    # load next years data, if necessary
    if (int(str(time1)[0:4]) > this_year) & (time1 <= end_time):
        this_year = this_year + diff_year
        range_start =\
            (str(f'{int(str(data["time"][-1].values)[0:4]) + 1:04d}')
            + '-01-01')
        if sample_param['calendar'] == 'standard':
            last_day = '-12-31'
            time_chunk = 73
        elif sample_param['calendar'] == '360_day':
            last_day = '-12-30'
            time_chunk = 72
        range_end = str(f'{this_year:04d}') + last_day
        vars_to_compute = sample_param['sample_vars']
        if sample_param['range']:
            vars_to_compute.append(sample_param['var_range'][0])
        elif sample_param['split']:
            vars_to_compute.append(sample_param['var_split'][0])
        var = vars_to_compute[0]
        update_data = data_whole[var].sel(
                          time=slice(range_start, range_end)
                          ).isel(
                          lon=slice(lon1, lon2), lat=slice(lat1, lat2)
                          ).chunk(
                          {'time': time_chunk, 'lon': 100, 'lat': 100}
                          ).to_dataset()
        for v in np.arange(1, len(vars_to_compute)):
            var = vars_to_compute[v]
            update_var = data_whole[var].sel(
                             time=slice(range_start, range_end)
                             ).isel(
                             lon=slice(lon1, lon2), lat=slice(lat1, lat2)
                             ).chunk(
                             {'time': time_chunk, 'lon': 100, 'lat': 100}
                             )
            update_data = update_data.update({var: update_var})
        data = xr.concat([data, update_data], dim='time',
                         data_vars='minimal', coords='minimal')
        data = data.chunk({'time': time_chunk,
                           'lon': 100, 'lat': 100}).compute()
    # drop already used years if they are not needed anymore
    if int(str(data['time'][0].values)[0:4]) < int(str(time0)[0:4]):
        range_start =\
            (str(f'{int(str(data["time"][0].values)[0:4]) + 1:04d}')
            + '-01-01')
        range_end = str(f'{int(str(end_time)[0:4]):04d}') + last_day
        data = data.sel(time=slice(range_start, range_end)).compute()
    # determine lon and lat of eddy
    lon_for_sel = track['lon'][0]
    if sample_param['grid'] == 'latlon':
        if lon_for_sel > 180.:
            lon_for_sel = lon_for_sel - 360.
    lat_for_sel = track['lat'][0]
    # if necessary extract values of of `var_range` and `var_split` at the
    # eddy's location
    if sample_param['range']:
        data_range = sample_param['ds_range']
        if sample_param['grid'] == 'latlon':
            data_range, lon_for_sel = monotonic_lon(data_range, lon_for_sel)
        vars_range_ed =\
            data_range[sample_param['var_range'][0]]\
            .sel(time=time0, lat=lat_for_sel,
                 lon=lon_for_sel, method='nearest').values
    if sample_param['split']:
        data_split = sample_param['ds_split']
        if sample_param['grid'] == 'latlon':
            data_split, lon_for_sel = monotonic_lon(data_split, lon_for_sel)
        vars_split_ed =\
            data_split[sample_param['var_split'][0]]\
            .sel(time=time0, lat=lat_for_sel,
                 lon=lon_for_sel, method='nearest').values
    # construct bool of all conditions to be met by the eddy to be
    # considered in `sampled`
    lon1_cond = sample_param['lon1']
    lon2_cond = sample_param['lon2']
    lon_track_cond = track['lon'][0]
    if sample_param['grid'] == 'latlon':
        if ((lon2_cond < 0) & (lon1_cond > 0)):
            lon2_cond = lon2_cond + 360
            if lon_track_cond < 0:
                lon_track_cond = lon_track_cond + 360
    conditions_are_met = (
        (time0 >= start_time)
        & (time1 <= end_time)
        & (track['type'] == sample_param['type'])
        & (length > lifetime)
        & (track['scale'].mean() > sample_param['size'])
        & (lon_track_cond > lon1_cond)
        & (lon_track_cond < lon2_cond)
        & (track['lat'][0] > sample_param['lat1'])
        & (track['lat'][0] < sample_param['lat2']))
    # add conditions for `range` and `split` if necessary and then add
    # the required fields to the eddy dictionary
    if sample_param['range']:
        conditions_are_met = (
            conditions_are_met
            & (vars_range_ed > sample_param['value_range'][0][0])
            & (vars_range_ed < sample_param['value_range'][0][1]))
    if sample_param['split']:
        if conditions_are_met & (vars_split_ed
                                 < sample_param['value_split'][0]):
            ab = 'below'
            sampled = track.copy()
            for variable in sample_param['sample_vars']:
                sampled = add_fields(sampled, data, sample_param, variable,
                                     lon1, lat1)
            i = i + 1
        elif conditions_are_met & (vars_split_ed
                                   >= sample_param['value_split'][0]):
            ab = 'above'
            sampled = track.copy()
            for variable in sample_param['sample_vars']:
                sampled = add_fields(sampled, data, sample_param, variable,
                                     lon1, lat1)
            j = j + 1
    else:
        if conditions_are_met:
            sampled = track.copy()
            for variable in sample_param['sample_vars']:
                sampled = add_fields(sampled, data, sample_param, variable,
                                     lon1, lat1)
            i = i + 1
    if sample_param['split']:
        return sampled, i, j, data, ab
    else:
        return sampled, i, j, data


def prepare(data_in, sample_param, tracks):
    ''' Preparing data for sampling

    Parameters
    ----------
    data_in : xarray.DataSet
         Dataset from which to extract fields at eddy locations.
    sample_param : dict
        Dictionary of parameters needed for the eddy sampling.
        The parameters are:
        sample_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'max_time': 73, # maximum length of tracks to consider
                            # (model time steps)
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'type': 'anticyclonic', # type of eddy
            'lifetime': 20, # length of the eddy's track in days
            'size': 20, # eddy size (diameter in km)
            'range': False, # sample eddy within a range of `var_range`
            'ds_range': data_int.isel(z=9), # dataset of `var_range`
            'var_range': ['votemper'], # variable to base the range on
            'value_range': [[4, 7],], # range of `var_range`
            'split': False, # split eddies at a threshold in below and above
            'ds_split': data_int.isel(z=0), # dataset of `var_split`
            'var_split': ['votemper'], # variable to base split on
            'value_split': [5.0,], # split eddies at this value
            'sample_vars': ['votemper'], # variables to sample
            'save_location': datapath, # where to store the netcdf files
            'save_name': 'test_170_-175'
            }
    tracks : dict
        Dictionary containing eddy information.

    Returns
    -------
    computed_data : xarray.DataSet
        Xarray Dataset containing all the data needed for the sampling of
        eddies. This is an extraction from `data_in` to save memory.
    lifetime : int
        Lifetime in time steps needed for an eddy to be considered for
        sampling.
    start_time : np.datetime64
        Earliest time from which to consider sampling eddies.
    end_time : np.datetime64
        Latest time from which to consider sampling eddies.
    '''
    # Try to detect time step of eddy tracks in days
    if sample_param['calendar'] == '360_day':
        try:
            timestep = int((tracks[0]['time'][1] - tracks[0]['time'][0]).days)
        except:
            k = 0
            while len(tracks[k]['time']) < 2:
                k = k + 1
            timestep = int((tracks[k]['time'][1] - tracks[k]['time'][0]).days)
    elif sample_param['calendar'] == 'standard':
        try:
            timestep = ((tracks[0]['time'][1] - tracks[0]['time'][0])
                        / np.timedelta64(1, 'D'))
        except:
            k = 0
            while len(tracks[k]['time']) < 2:
                k = k + 1
            timestep = ((tracks[k]['time'][1] - tracks[k]['time'][0])
                        / np.timedelta64(1, 'D'))

    # Convert lifetime from days to indeces
    lifetime = sample_param['lifetime'] / timestep
    start_year = str(sample_param['start_time'][0:4])
    end_year = str(sample_param['end_time'][0:4])
    if sample_param['calendar'] == 'standard':
        start_time = np.datetime64(sample_param['start_time'])
        end_time = np.datetime64(sample_param['end_time'])
        last_day = '-12-31'
    elif sample_param['calendar'] == '360_day':
        start_time = cft.Datetime360Day(int(sample_param['start_time'][0:4]),
                                        int(sample_param['start_time'][5:7]),
                                        int(sample_param['start_time'][8:10]))
        end_time = cft.Datetime360Day(int(sample_param['end_time'][0:4]),
                                      int(sample_param['end_time'][5:7]),
                                      int(sample_param['end_time'][8:10]))
        last_day = '-12-30'
    if sample_param['grid'] == 'latlon':
        addlon = 2
        addlat = 1
    elif sample_param['grid'] == 'cartesian':
        addlon = 2e5
        addlat = 1e5
    lon1 = int(np.argmin(((data_in['lon']
               - (sample_param['lon1'] - addlon)) ** 2).values))
    lon2 = int(np.argmin(((data_in['lon']
               - (sample_param['lon2'] + addlon)) ** 2).values))
    lat1 = int(np.argmin(((data_in['lat']
               - (sample_param['lat1'] - addlat)) ** 2).values))
    lat2 = int(np.argmin(((data_in['lat']
               - (sample_param['lat2'] + addlat)) ** 2).values))
    vars_to_compute = sample_param['sample_vars']
    if sample_param['range']:
        vars_to_compute.append(sample_param['var_range'][0])
    elif sample_param['split']:
        vars_to_compute.append(sample_param['var_split'][0])
    computed_data = data_in[vars_to_compute[0]].sel(
                        time=slice(start_year + '-01-01',
                                   start_year + last_day)
                        ).isel(
                        lon=slice(lon1, lon2), lat=slice(lat1, lat2)
                        ).to_dataset()
    if len(vars_to_compute) > 1:
        for i in np.arange(1, len(vars_to_compute)):
            update_data = data_in[vars_to_compute[i]].sel(
                              time=slice(start_year + '-01-01',
                                         start_year + last_day)
                              ).isel(
                              lon=slice(lon1, lon2), lat=slice(lat1, lat2))
            computed_data = computed_data.update(
                                {vars_to_compute[i]: update_data})
    return computed_data.compute(), lifetime, start_time, end_time


def sample(tracks, data, sample_param):
    """ Sample eddies from dictionary that fulfill certain criteria and save
    them in pickle objects.

    Parameters
    ----------
    tracks : dict
        Dictionary containing eddy information.
    data : xarray.DataSet
        Dataset from which to extract fields at eddy locations.
    sample_param : dict
        Dictionary of parameters needed for the eddy sampling.
        The parameters are:
        sample_param = {
            'model': 'model_name', # either ORCA or MITgcm
            'grid': 'latlon', # either latlon or cartesian
            'start_time': 'YYYY-MM-DD', # time range start
            'end_time': 'YYYY-MM-DD', # time range end
            'calendar': 'standard', # calendar, must be either 360_day or
                                    # standard
            'max_time': 73, # maximum length of tracks to consider
                            # (model time steps)
            'lon1': -180, # minimum longitude of detection region, either in
                          # the range (-180, 180) degrees or in m for a
                          # cartesian grid
            'lon2': -130, # maximum longitude of detection region, either
                          # (-180, 180) degrees or m
            'lat1': -55, # minimum latitude of detection region, either
                          # (-90, 90) degrees or m
            'lat2': -30, # maximum latitude of detection region, either
                          # (-90, 90) degrees or m
            'type': 'anticyclonic', # type of eddy
            'lifetime': 20, # length of the eddy's track in days
            'size': 20, # eddy size (diameter in km)
            'range': False, # sample eddy within a range of `var_range`
            'ds_range': data_int.isel(z=9), # dataset of `var_range`
            'var_range': ['votemper'], # variable to base the range on
            'value_range': [[4, 7],], # range of `var_range`
            'split': False, # split eddies at a threshold in below and above
            'ds_split': data_int.isel(z=0), # dataset of `var_split`
            'var_split': ['votemper'], # variable to base split on
            'value_split': [5.0,], # split eddies at this value
            'sample_vars': ['votemper'], # variables to sample
            'save_location': datapath, # where to store the netcdf files
            'save_name': 'test_170_-175'
            }

    Returns
    -------
    Saves netcdf files to disk.
    """
    # Initialize
    i = 0
    j = 0
    # Prepare data
    computed_data, lifetime, start_time, end_time = prepare(data,
                                                            sample_param,
                                                            tracks)
    print('data prepared, now sampling')
    if sample_param['split']:
        ed_list = np.around(np.linspace(0, len(tracks), 10))
        for ed in np.arange(0, len(tracks)):
            if ed in ed_list:
                print('sampling eddy number ' + str(ed) + ' of '
                      + str(len(tracks)))
            i_before = i
            j_before = j
            # only consider tracks that are shorter than max_time
            if len(tracks[ed]['time']) < sample_param['max_time']:
                try:
                    sample, i, j, computed_data, ab = sample_core(
                        tracks[ed], computed_data, data, sample_param,
                        i, j, lifetime, start_time, end_time)
                except:
                    pass
            if i_before == i - 1:
                e_num = "%07d" % (i,)
                file_name = (sample_param['save_location']
                             + sample_param['save_name']
                             + '.' + sample_param['type'] + '.larger_'
                             + str(sample_param['size']) + '.longer_'
                             + str(sample_param['lifetime']) + '.' + str(e_num)
                             + '.' + ab + '_thr.nc')
                write_to_netcdf(file_name, sample, sample_param, data)
            elif j_before == j - 1:
                e_num = "%07d" % (j,)
                file_name = (sample_param['save_location']
                             + sample_param['save_name']
                             + '.' + sample_param['type'] + '.larger_'
                             + str(sample_param['size']) + '.longer_'
                             + str(sample_param['lifetime']) + '.' + str(e_num)
                             + '.' + ab + '_thr.nc')
                write_to_netcdf(file_name, sample, sample_param, data)
    else:
        ed_list = np.around(np.linspace(0, len(tracks), 10))
        for ed in np.arange(0, len(tracks)):
            if ed in ed_list:
                print('sampling eddy number ' + str(ed) + ' of '
                      + str(len(tracks)))
            i_before = i
            # only consider tracks that are shorter than max_time
            if len(tracks[ed]['time']) < sample_param['max_time']:
                sample, i, j, computed_data = sample_core(tracks[ed],
                                                          computed_data,
                                                          data,
                                                          sample_param,
                                                          i, j,
                                                          lifetime,
                                                          start_time,
                                                          end_time)
            if i_before == i - 1:
                e_num = "%07d" % (i,)
                file_name = (sample_param['save_location']
                             + sample_param['save_name']
                             + '.' + sample_param['type'] + '.larger_'
                             + str(sample_param['size']) + '.longer_'
                             + str(sample_param['lifetime']) + '.' + str(e_num)
                             + '.nc')
                write_to_netcdf(file_name, sample, sample_param, data)
