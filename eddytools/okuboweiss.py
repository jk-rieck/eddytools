''' okuboweiss

Collection of functions needed to calculate the Okubo-Weiss parameter
from horizontal velocities in a xgcm-compatible xarray dataset.

'''


def calc(data, grid, u_var, v_var):
    ''' Calculate Okubo-Weiss parameter from horizontal velocities.

    Parameters
    ----------
    data : xarray.DataSet
        xarray dataset containing the zonal and meridional velocities. The
        DataSet must be compatible with xgcm.
    grid : xgcm.Grid
        xgcm Grid describing the grid of `data`.
    u_var : str
        Name of the zonal velocity variable in `data`.
    v_var : str
        Name of the meridional velocity variable in `data`.

    Returns
    -------
    data : xarray.DataSet
        updated xarray dataset containing everything that the input dataset
        `data` contained plus the vorticity `vort` of the velocity field and
        the Okubo-Weiss paramter `OW`.
    '''
    # Arguments for the interpolation and differentiation
    diff_args = {'boundary': 'extend'}
    int_args = {'metric_weighted': False, 'boundary': 'extend'}
    # Compute the gradients of horizontal velocities and interpolate them onto
    # the T-grid
    # g11 : gradient of zonal velocity in zonal direction
    # g12 : gradient of zonal velocity in meridional direction
    # g21 : gradient of meridional velocity in zonal direction
    # g22 : gradient of meridional velocity in meridional direction
    g11 = grid.interp(grid.diff(grid.interp(data[u_var], 'X', **int_args),
                                'X', **diff_args), 'Y', **int_args)
    g12 = grid.diff(data[u_var], 'Y', **diff_args)
    g21 = grid.diff(data[v_var], 'X', **diff_args)
    g22 = grid.interp(grid.diff(grid.interp(data[v_var], 'X', **int_args),
                                'Y', **diff_args), 'Y', **int_args)
    # calculate the vorticity
    vort = (g21 - g12)
    # calculate the total strain
    s = (((g21 + g12) ** 2.) + ((g11 - g22) ** 2.)) ** 0.5
    # calculate the Okubo-Weiss parameter (total strain - vorticity)
    W = (s ** 2.) - (vort ** 2.)
    # write `vort` and `OW` to `data` and return `data`
    data = data.update({'vort': vort, 'OW': W})
    return data
